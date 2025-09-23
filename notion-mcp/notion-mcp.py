import os 
import json 
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import unicorn 

import main
import search

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    subqueries: list[str]
    context_chunks: list[dict]
    answer: str
    sources: list[dict]

@app.post("/updates")
async def update_index():
    try: 
        main.run_pipleline()
        return {"Success": True}
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))

@app.post("/query", response_model = QueryResponse)
async def query_rag(q: QueryRequest):
    try:
        # Generate subqueries 
        subqs = search.generate_subqueries(q.question, max_alts=4)
        # Embedding
        embed_vecs = search.embed_queries(subqs)
        # Load index and meta
        p = search.load_index()
        metas = search.load_meta()
        # KNN search
        per_labels, per_dists = search.search_knn_multi(p, embed_vecs, per_query_k=50)
        fused_indices = search.rrf_fuse(per_labels, k=10, k_rrf=60)
        protocol_context = "\n\n".join(parts)
        # Generate answer
        answer = search.generate_answer_with_ollama(q.question, protocol_context)
        response = QueryResponse(
            question=q.question,
            subqueries=subqs,
            context_chunks=context_chunks,
            answer=answer,
            sources=sources
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
