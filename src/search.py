# search.py
import json
import numpy as np
import hnswlib
import os
import httpx
import ollama
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

OUTPUT_DIR = "artifacts"
INDEX_PATH = os.path.join(OUTPUT_DIR, "notion_hnsw_hnswlib.index")
META_PATH  = os.path.join(OUTPUT_DIR, "meta.jsonl")

OLLAMA_BASE  = os.getenv("OLLAMA_BASE", "http://localhost:11435")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
if not OLLAMA_MODEL:
    raise RuntimeError("Missing OLLAMA_MODEL. Set it in your .env, e.g. OLLAMA_MODEL=llama3.1:8b-instruct")
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false")
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "600"))

class LLMError(RuntimeError):
    pass

class OllamaClient:
    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2, max_tokens: int = 800) -> str:
        url = f"{OLLAMA_BASE.rstrip('/')}/api/chat"
        options = {
            "temperature": float(temperature),
            "num_predict": int(max_tokens) if max_tokens and max_tokens > 0 else 256,
        }
        if OLLAMA_THINK.lower() in ("0", "false", "no"):
            options["think"] = False
        payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False, "options": options}
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

def ollama_embed(text: str) -> np.ndarray:
    try:
        response = ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
        return response["embedding"]
    except Exception as e:
        raise LLMError(f"Ollama embedding failed: {e}")

def embed_queries(qs: list[str]) -> np.ndarray:
    prefixed_qs = []
    for q in qs:
        prep_text = f"search_query: {q}"
        emb = ollama_embed(prep_text)
        prefixed_qs.append(emb)
    return np.asarray(prefixed_qs, dtype="float32")

def load_meta(path=META_PATH):
    metas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

def generate_subqueries(q: str, max_alts: int = 4) -> list[str]:
    out = [q]
    want = max(0, max_alts - 1)
    prompt = f"""
    You are a retrieval assistant. Rewrite the user query into {want} DIVERSE search queries:
    - Keep each <= 15 words
    - Include one keyword-only version
    - Include one broader version
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
        return [q]
    return out if len(out) > 1 else [q]

def search_knn_multi(p: hnswlib.Index, qvecs: np.ndarray, per_query_k: int = 50):
    labels_list, dists_list = [], []
    try:
        labels_batch, dists_batch = p.knn_query(qvecs, k = per_query_k)
        for i in range(qvecs.shape[0]):
            labels_list.append(labels_batch[i].tolist())
            dists_list.append(dists_batch[i].tolist())
        return labels_list, dists_list
    except Exception:
        for i in range(qvecs.shape[0]):
            labels, dists = p.knn_query(qvecs[i:i+1], k=per_query_k)
            labels_list.append(labels[0].tolist())
            dists_list.append(dists[0].tolist())
        return labels_list, dists_list

def rrf_fuse(rank_lists: list[list[int]], k: int = 50, k_rrf: int = 60) -> list[int]:
    from collections import defaultdict
    scores = defaultdict(float)
    for ranks in rank_lists:
        for r, doc_id in enumerate(ranks, start=1):
            scores[int(doc_id)] += 1.0 / (k_rrf + r)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in fused][:k]

def generate_answer_with_ollama(query: str, context: str) -> str:
    instructions_text = (
        "You are a helpful assistant. Answer the user's question **only** using the context.\n\n"
        "Instructions:\n"
        "- Only use the relevant parts from the context. If it is not present, say \"I don't know.\".\n"
        "- Be concise and factual.\n"
        "- Do not hallucinate."
    )
    user_block = f"Question:\n{query}\n\nContext:\n{context}"
    messages = [
        {"role": "system", "content": instructions_text},
        {"role": "user", "content": user_block}
    ]
    return _ollama.chat(messages, temperature=0.2, max_tokens=800).strip()

# ---------------------------
# Module-level initialization (load once)
# ---------------------------
_dim = 768
_index = None
_metas = None

def _init_index_and_meta():
    global _index, _metas
    if _index is None:
        p = hnswlib.Index(space='cosine', dim=_dim)
        p.load_index(INDEX_PATH)
        p.set_ef(246)
        _index = p
    if _metas is None:
        _metas = load_meta()
    return _index, _metas

def fetch_notion_content(query: str, top_for_gen: int = 5) -> dict:
    """
    Tool function to be used by agents.
    Returns: {"answer": str, "sources": [{"tag": "S1", "chunk_idx": 45, "url": "..."}...]}
    """
    p, metas = _init_index_and_meta()

    try:
        subqs = generate_subqueries(query, max_alts=4)
        qvecs = embed_queries(subqs)
        per_labels, per_dists = search_knn_multi(p, qvecs, per_query_k=50)
        fused_indices = rrf_fuse(per_labels, k=10, k_rrf=60)

        parts = []
        sources = []
        for si, idx in enumerate(fused_indices[:top_for_gen], start=1):
            m = metas[int(idx)]
            header = f"[S{si}] chunk #{m.get('chunk_idx')} — {m.get('url','')}"
            body = (m.get("text") or "").replace("\n", " ").strip()
            if not body:
                continue
            parts.append(header + "\n" + body)
            sources.append({"tag": f"S{si}", "chunk_idx": m.get("chunk_idx"), "url": m.get("url","")})
        context = "\n\n".join(parts)

        if not context:
            return {"answer": "I don't know.", "sources": []}

        answer = generate_answer_with_ollama(query, context)
        if not answer:
            answer = "I don't know."

        return {"answer": answer, "sources": sources}
    except Exception as e:
        # For production you might want to log stacktrace
        return {"answer": f"Search failed: {e}", "sources": []}

def run_search(query: str, top_for_gen: int = 5) -> dict:
    """
    Main callable entrypoint for integration with main.py or agents.
    Handles query routing, retrieval, and generation.
    """
    result = fetch_notion_content(query, top_for_gen=top_for_gen)

    print("\n=== Answer ===")
    print(result["answer"])
    print("\n=== Sources ===")
    for s in result["sources"]:
        print(f"{s['tag']}: chunk #{s['chunk_idx']}  {s['url']}")
    return result


# ---------------------------
# CLI mode (optional)
# ---------------------------
if __name__ == "__main__":
    try:
        while True:
            q = input("\n Query: ").strip()
            if not q:
                break
            run_search(q)
    except KeyboardInterrupt:
        pass
