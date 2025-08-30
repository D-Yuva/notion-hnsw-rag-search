import json
import numpy as np
import hnswlib
import os
import httpx
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

OUTPUT_DIR = "artifacts"
MODEL_NAME = "intfloat/e5-large-v2"
INDEX_PATH = os.path.join(OUTPUT_DIR, "notion_hnsw_hnswlib.index")
META_PATH  = os.path.join(OUTPUT_DIR, "meta.jsonl")

EMB_MODEL = SentenceTransformer(MODEL_NAME)

# --- Ollama (local) ---
OLLAMA_BASE  = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
if not OLLAMA_MODEL:
    raise RuntimeError("Missing OLLAMA_MODEL. Set it in your .env, e.g. OLLAMA_MODEL=llama3.1:8b-instruct")
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false")  # disable chain-of-thought by default
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "600"))  # allow slow first-token models

class LLMError(RuntimeError):
    pass

class OllamaClient:
    """Local Ollama chat client (no API key, default localhost:11434)."""
    def chat(self,
             messages: list[dict[str, str]],
             temperature: float = 0.2,
             max_tokens: int = 800) -> str:
        url = f"{OLLAMA_BASE.rstrip('/')}/api/chat"
        # Ollama uses num_predict instead of max_tokens; None/0 -> small cap
        options: dict = {
            "temperature": float(temperature),
            "num_predict": int(max_tokens) if max_tokens and max_tokens > 0 else 256,
        }
        # disable chain-of-thought so Ollama returns content instead of 'thinking'
        if OLLAMA_THINK.lower() in ("0", "false", "no"):
            options["think"] = False
        payload: dict = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        # simple retry for transient local errors (e.g., model loading)
        for attempt in range(3):
            try:
                with httpx.Client(timeout=OLLAMA_TIMEOUT_SEC) as client:
                    r = client.post(url, json=payload)
                if r.status_code >= 400:
                    if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                        import time; time.sleep(1.5 * (attempt + 1))
                        continue
                    raise LLMError(f"Ollama error {r.status_code}: {r.text}")
                data = r.json()
                msg = data.get("message", {})
                content = msg.get("content")
                if not content:
                    raise LLMError(f"Ollama unexpected response: {data}")
                return content
            except httpx.HTTPError as e:
                if attempt < 2:
                    import time; time.sleep(1.0 * (attempt + 1))
                    continue
                raise LLMError(f"HTTP error: {e}")
        raise LLMError("Ollama chat failed after retries")

_ollama = OllamaClient()

"""
    Converting my multi query into a vector array
"""

def embed_queries(qs: list[str]) -> np.ndarray:
    v = EMB_MODEL.encode([f"query: {q}" for q in qs], normalize_embeddings=True)
    return np.asarray(v, dtype="float32") # Returns an array of queries embed_queries(["what is AI?", "define machine learning"])

# Backward-compatible wrapper

def embed_query(q: str) -> np.ndarray:
    return embed_queries([q])

# Connects the MetaData (page_id, url, chunk_idx, text) with the chunks
def load_meta(path=META_PATH):
    metas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

def generate_subqueries(q: str, max_alts: int = 4) -> list[str]:
    """
    Generate sub-queries using a local LLM (Ollama) ONLY.
    - Always include the original query first.
    - If the LLM call fails or returns nothing parsable, we return [q] (no deterministic rewrites).
    """
    out = [q]
    want = max(0, max_alts - 1)

    prompt = f"""
    You are a retrieval assistant. Rewrite the user query into {want} DIVERSE search queries:
    - Keep each <= 15 words
    - Include one keyword-only version
    - Include one broader (step-back) version
    - Include one narrower/specific version
    - Do not invent entities not present in the original question
    Return ONLY a JSON array of strings, no explanations.

    User query: {q}
    """.strip()

    try:
        messages = [
            {"role": "system", "content": "You rewrite queries for retrieval. Return ONLY a JSON array of strings."},
            {"role": "user", "content": prompt}
        ]
        text = _ollama.chat(messages, temperature=0.2, max_tokens=600)
        rewrites = json.loads(text)
        for r in rewrites:
            s = (r or "").strip()
            if s and s not in out:
                out.append(s)
            if len(out) >= max_alts:
                break
    except Exception:
        # STRICT: no deterministic variants; just fall back to single-query mode
        return [q]

    # If LLM returned empty/invalid content, also fall back to single-query
    return out if len(out) > 1 else [q]

def search_knn_multi(p: hnswlib.Index, qvecs: np.ndarray, per_query_k: int = 50):
    # RUNS KNN for each query
    """
    labels_list: shape (Q, k) → the indices of the top-k chunks for all queries.
	dists_list: shape (Q, k) → the distances to those chunks.
    """
    labels_list, dists_list = [], []

    try:
        """
        labels_batch: shape (Q, k) → the indices of the top-k chunks for each query.
	    dists_batch: shape (Q, k) → the distances to those chunks.
        """
        labels_batch, dists_batch = p.knn_query(qvecs, k = per_query_k)
        for i in range(qvecs.shape[0]):
            labels_list.append(labels_batch[i].tolist())
            dists_list.append(dists_batch[i].tolist())
        return labels_list, dists_list
    except Exception:
        # fallback: run knn_query one by one (use 2D slice (1, dim))
        for i in range(qvecs.shape[0]):
            labels, dists = p.knn_query(qvecs[i:i+1], k=per_query_k)
            labels_list.append(labels[0].tolist())
            dists_list.append(dists[0].tolist())
        return labels_list, dists_list

def rrf_fuse(rank_lists: list[list[int]], k: int = 50, k_rrf: int = 60) -> list[int]:
    """
    Reciprocal Rank Fusion: combine multiple ranked lists into one.
    rank_lists: list of lists of doc ids (one ranked list per sub-query).
    Returns a single fused list of top-k doc ids.
    """
    from collections import defaultdict
    scores = defaultdict(float)  # Dictionary with key as document ID and the value as score
    for ranks in rank_lists: # Will parse through the list of document IDS -> labels_list
        for r, doc_id in enumerate(ranks, start=1):
            scores[int(doc_id)] += 1.0 / (k_rrf + r)
    # Sorts the array with the score, dict cannot be sorted 
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True) # Converts into a list of tuple -> From this scores = {10: 0.0489, 20: 0.0325, 30: 0.0320} to [(10, 0.0489), (20, 0.0325), (30, 0.0320)]
    return [doc_id for doc_id, _ in fused][:k] # Returns only the doc ID

# --- Answer generation using OLLAMA LLM ---
def generate_answer_with_ollama(query: str, context: str) -> str:
    """
    Ask the local LLM (Ollama) to answer using ONLY the provided context.
    Uses /api/chat only (messages format) with your exact instructions text.
    """
    instructions_text = (
        "You are a helpful assistant. Answer the user's question **only** using the context.\n\n"
        "Instructions:\n"
        "- Only use the relavent parts from the context, if you think few parts from the context does not match with the query then drop it.\n"
        "- If the answer is not in the context, say \"I don't know.\"\n"
        "- Be concise and factual.\n"
        "- Do not halucinate. "
    )
    user_block = f"Question:\n{query}\n\nContext:\n{context}"
    messages = [
        {"role": "system", "content": instructions_text},
        {"role": "user", "content": user_block}
    ]
    return _ollama.chat(messages, temperature=0.2, max_tokens=800).strip()

if __name__ == "__main__":
    dim = EMB_MODEL.get_sentence_embedding_dimension()  # Gets the dimension of e5-large-v2 (The embedded vector size)

    p = hnswlib.Index(space='cosine', dim=dim)
    p.load_index(INDEX_PATH)

    p.set_ef(246) # Decides the number of neighbours to consider before figuring out the top few

    metas = load_meta()

    try:
        while True:
            q = input("\n Query: ").strip()
            if not q:
                break

            # 1) Use LLM to generate diverse sub-queries (Ollama)
            subqs = generate_subqueries(q, max_alts=4)

            # 2) Embed all sub-queries at once
            qvecs = embed_queries(subqs) 

            # 3) Run HNSW KNN for each sub-query (batched when possible)
            per_labels, per_dists = search_knn_multi(p, qvecs, per_query_k=50)

            # 4) Fuse rankings with Reciprocal Rank Fusion (RRF)
            fused_indices = rrf_fuse(per_labels, k=10, k_rrf=60)

            # 5) Build a minimal context from top chunks (plain + short headers), then ask the LLM
            TOP_FOR_GEN = 5  # keep small so it fits in the model context
            parts = []
            sources = []
            for si, idx in enumerate(fused_indices[:TOP_FOR_GEN], start=1):
                m = metas[int(idx)]
                header = f"[S{si}] chunk #{m.get('chunk_idx')} — {m.get('url','')}"
                body = (m.get("text") or "").replace("\n", " ").strip()
                if not body:
                    continue
                parts.append(header + "\n" + body)
                sources.append((f"S{si}", m.get("chunk_idx"), m.get("url","")))
            context = "\n\n".join(parts)

            answer = generate_answer_with_ollama(q, context)

            print("\n=== Answer ===")
            print(answer)

            print("\n=== Sources ===")
            for tag, cidx, url in sources:
                print(f"{tag}: chunk #{cidx}  {url}")
    except KeyboardInterrupt:
        pass
