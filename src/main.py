import os
from dotenv import load_dotenv
from notion_client import Client
import httpx
import numpy as np
from sentence_transformers import SentenceTransformer
from notion_client.errors import APIResponseError, RequestTimeoutError
from tqdm import tqdm  
import json  
import faiss  
import time
import hnswlib

load_dotenv()

httpx_client = httpx.Client(
    timeout=httpx.Timeout(60.0, connect=10.0)  # 60s read/write, 10s connect
)

notion = Client(
    auth=os.getenv("NOTION_TOKEN"),
    client=httpx_client
)

"""
------------NOTION DATA--------------
"""
"""
    Fetching Pages
    - Connects to Notion 
    - Fetches all of the pages
    - Returns the Page ID 
"""
def all_pages():
    pages = {} # Stores the found Page ID and avoids duplicates
    cursor = None # 
    while True:
        resp = notion.search(
            **({"start_cursor": cursor} if cursor else {}), # Create a dictionary when true if not empty dict
            filter={"property": "object", "value": "page"} # Ensures we only get pages
        )
        for p in resp.get("results", []): # The API key returns a JSON results
            pages[p["id"]] = p # Doesn't allow Duplicates
        if not resp.get("has_more"):  
            break # If there are no more pages to fetch then break
        cursor = resp["next_cursor"] # If there are more pages to fetch then move to the next page using next cursor
    return list(pages.values()) # Returns a list of Pages 

"""
    Converting each block to text
"""

def plain(rt):
    """Extract plain text from a Notion rich_text array or title string."""
    if rt is None:
        return ""
    if isinstance(rt, str):
        # already plain text
        return rt
    if isinstance(rt, list):
        # list of dicts or strings
        return "".join(
            t.get("plain_text", str(t)) if isinstance(t, dict) else str(t)
            for t in rt
        )
    # Fallback for unexpected types
    return str(rt)
    # Notion returns an object that has, type, link, plain_text, and annotations (bold, italics) from this we need to extract only the plain text

def flatten_blocks(block_id):
    # Recursively collect all of the texts form the Notion page or block
    out = []
    skipped = []  # Tracks blocks we can't access (like ai_block)

    def walk(bid, parent_id=None):
        cursor = None
        while True:
            try:
                # retry loop for timeouts / transient errors
                max_retries = 5
                attempt = 0
                while True:
                    try:
                        resp = notion.blocks.children.list(block_id=bid, start_cursor=cursor) # Returns Children for example (First line: Heading, Second line: A bullet point. The Bullet point is considered as the children for the heading)
                        break
                    except RequestTimeoutError:
                        if attempt >= max_retries:
                            raise
                        wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s, 8s, 16s
                        print(f"[retry] timeout on children.list({bid}); sleeping {wait}s and retrying...")
                        time.sleep(wait)
                        attempt += 1
                    except APIResponseError as e:
                        # retry on transient 5xx or rate limits
                        if getattr(e, "code", None) in {"rate_limited"} or (500 <= getattr(e, "status", 0) < 600):
                            if attempt >= max_retries:
                                raise
                            wait = 2 ** attempt
                            print(f"[retry] {e} on {bid}; sleeping {wait}s and retrying...")
                            time.sleep(wait)
                            attempt += 1
                        else:
                            raise
            except APIResponseError as e:
                # couldn't list children for this block/page; record and stop recursing here
                skipped.append({
                    "block_id": bid,
                    "block_type": "unknown",
                    "reason": f"children.list error: {e.__class__.__name__}: {str(e)}",
                    "parent_id": parent_id
                })
                return
            except RequestTimeoutError as e:
                skipped.append({
                    "block_id": bid,
                    "block_type": "unknown",
                    "reason": f"timeout after retries: {e.__class__.__name__}",
                    "parent_id": parent_id
                })
                return

            # optional gentle pacing to avoid rate limits (≈5 req/s)
            time.sleep(0.2)

            for b in resp.get("results", []):
                t = b.get("type") # Gets if the string paragraph, heading_1, bulleted_list_items etc

                if t == "ai_block":
                    skipped.append({
                        "block_id": b["id"],
                        "block_type": "ai_block",
                        "reason": "unsupported by API",
                        "parent_id": parent_id
                    })
                    continue  # skip text extraction but keep track

                if t and isinstance(b.get(t), dict):
                    obj = b[t]
                    if "rich_text" in obj:
                        out.append(plain(obj["rich_text"]))
                    if "title" in obj:
                        out.append(plain(obj["title"]))

                if b.get("has_children"): # If more sub blocks exists then it recursively retrives them
                    walk(b["id"], parent_id=bid)

            if not resp.get("has_more"):
                break
            cursor = resp.get("next_cursor")

    walk(block_id)
    return "\n".join([x for x in out if x.strip()]), skipped # Joins all of the texts from out and puts them in new lines

"""
------------CHUNKING--------------
"""
def chunk_words(text, size = 450, overlap = 60):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


"""
Why chunking ?
- Size Control: e5-base-v2 → 512 tokens max
- Overlap: Helps to avoid the loss of context
- Efficenty 
"""
"""
------------EMBEDDING--------------
"""
EMB_MODEL = SentenceTransformer("intfloat/e5-large-v2")

# Embedding each chunk
def embed_passages(chunks):
    texts = [f"passage: {c}" for c in chunks]
    vecs = EMB_MODEL.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True # magnitude affects the dot, cos similarity, so we normalise so only the angle is considered
    )

    return np.asarray(vecs, dtype = "float32") # Returns a matrix for HNSW

def embed_query(query):
    # Search Query
    vec = EMB_MODEL.encode(
        [f"query: {query}"],
        normalize_embeddings=True
    )
    return np.asarray(vec, dtype = "float32")


# ---------------- MAIN: crawl -> chunk -> embed -> build HNSW ----------------
if __name__ == "__main__":
    print("Step 1: listing pages")
    ps = all_pages()
    print(f"-> pages found: {len(ps)}")
    if not ps:
        raise SystemExit("No pages visible to the integration. Share pages with your Notion integration.")

    # ------------ CONFIG ------------
    CHUNK_SIZE = 550
    CHUNK_OVERLAP = 80
    HNSW_M = 32 # Max number of *links* each node will keep to its nearest neighbors
    EF_CONSTRUCTION = 200 # Number of candidate neighbors considered when building each node's links

    OUTPUT_DIR = "artifacts"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    META_PATH  = os.path.join(OUTPUT_DIR, "meta.jsonl")
    EMB_PATH   = os.path.join(OUTPUT_DIR, "chunk_vecs.npy")
    SKIPPED_PATH = os.path.join(OUTPUT_DIR, "skipped_blocks.jsonl")

    all_chunks = []
    meta = []
    all_skipped = []

    # ------------ Crawl all pages ------------
    for p in tqdm(ps, desc="Downloading + Chunking"):
        page_id = p["id"]
        page_url = p.get("url", "")

        # get text + skipped items (ai_block etc.)
        text, skipped = flatten_blocks(page_id)

        # attach page context to skipped entries and accumulate
        for s in skipped:
            s["page_id"] = page_id
            s["page_url"] = page_url
        all_skipped.extend(skipped)

        if not text.strip():
            continue

        # chunk
        chunks = chunk_words(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        # accumulate chunks + metadata
        start_idx = len(all_chunks)
        all_chunks.extend(chunks)
        for i, ch in enumerate(chunks):
            meta.append({
                "page_id": page_id,
                "url": page_url,
                "chunk_idx": i,
                "text": ch
            })

    print(f"Total chunks collected: {len(all_chunks)}")
    if all_skipped:
        with open(SKIPPED_PATH, "w", encoding="utf-8") as f:
            for s in all_skipped:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Skipped items recorded: {len(all_skipped)} -> {SKIPPED_PATH}")

    if not all_chunks:
        raise SystemExit("No chunks produced. Check permissions/content.")

    # ------------ Embed all chunks ------------
    print("Embedding chunks...")
    chunk_vecs = embed_passages(all_chunks)
    print("Embeddings shape:", chunk_vecs.shape)  # (N, 768) for e5-base-v2

    # save embeddings + meta
    np.save(EMB_PATH, chunk_vecs)
    with open(META_PATH, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Saved: {EMB_PATH} and {META_PATH}")

    # ------------ Build HNSW index ------------
    dim = chunk_vecs.shape[1]
    num = chunk_vecs.shape[0]

    p = hnswlib.Index(space='cosine', dim = dim)

    # Graph with these values
    p.init_index(
        max_elements=num, 
        ef_construction = EF_CONSTRUCTION,
        M = HNSW_M    
    )

    # Add items
    ids = np.arange(num) 
    p.add_items(chunk_vecs, ids)

    p.set_ef(64)

    HNSWLIB_INDEX_PATH = os.path.join(OUTPUT_DIR, "notion_hnsw_hnswlib.index")
    p.save_index(HNSWLIB_INDEX_PATH)
    print(f"Saved HNSW (hnswlib) index -> {HNSWLIB_INDEX_PATH}")
    print("✅ Index build complete.")