import json
import numpy as np
import hnswlib
import os
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = "artifacts"
MODEL_NAME = "intfloat/e5-large-v2"
INDEX_PATH = os.path.join(OUTPUT_DIR, "notion_hnsw_hnswlib.index")
META_PATH  = os.path.join(OUTPUT_DIR, "meta.jsonl")

EMB_MODEL = SentenceTransformer(MODEL_NAME)

"""
    Converting my query into a vector array
"""
def embed_query(q: str) -> np.ndarray:
    v = EMB_MODEL.encode([f"query: {q}"], normalize_embeddings=True)
    return np.asarray(v, dtype = "float32")

# Connects the MetaData (page_id, url, chunk_idx, text) with the chunks
def load_meta(path=META_PATH):
    metas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

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
            qvec = embed_query(q)

            #Search using KNN
            labels, dists = p.knn_query(qvec, k = 150)
            print("\nTop hits:")
            for rank, (idx, dist) in enumerate(zip(labels[0], dists[0]), 1): #Matches the vector ID with the distance
                m = metas[int(idx)]
                sim = 1.0 - float(dist) # Similarity, higher similarity -> More closer
                snippet = m["text"][:300].replace("\n", " ") # Grabs the first 300 characters of the chunk text, and continues the newline into one single para
                print(f"[{rank}] sim={sim:.4f}  #{m['chunk_idx']}  {m.get('url','')}")
                print(snippet + "...\n")
    except KeyboardInterrupt:
        pass
