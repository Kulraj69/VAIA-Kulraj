from typing import List, Dict, Any, Tuple, Optional
import os
import numpy as np

from .db import get_db


VECTORS_COLLECTION_NAME = "vectors"
VECTOR_INDEX_ENV = "MONGO_VECTOR_INDEX_NAME"  # Atlas Search (vectorSearch) index name
VECTOR_NUM_CANDIDATES_ENV = "MONGO_VECTOR_NUM_CANDIDATES"  # for $vectorSearch
VECTOR_SIMILARITY_ENV = "MONGO_VECTOR_SIMILARITY"  # dotProduct | cosine | euclidean


async def upsert_vectors(items: List[Dict[str, Any]]) -> None:
    """
    items: [{
        "chunk_id": str,
        "text": str,
        "embedding": List[float],
        "metadata": Dict[str, Any],
    }]
    """
    db = await get_db()
    col = db[VECTORS_COLLECTION_NAME]
    for it in items:
        await col.update_one(
            {"chunk_id": it["chunk_id"]},
            {"$set": it},
            upsert=True,
        )


async def similarity_search(query_embedding: np.ndarray, top_k: int, source_file: Optional[str] = None) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Try Atlas $vectorSearch if configured. Otherwise, fallback to local cosine.
    If source_file is provided, only search chunks from that document.
    """
    db = await get_db()
    col = db[VECTORS_COLLECTION_NAME]

    index_name = os.environ.get(VECTOR_INDEX_ENV)  # e.g., "vectors_index"
    num_candidates = int(os.environ.get(VECTOR_NUM_CANDIDATES_ENV, "200"))

    # Build filter for source_file if provided
    filter_query = {}
    if source_file:
        filter_query["metadata.source_file"] = source_file

    if index_name:
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": index_name,
                        "path": "embedding",
                        "queryVector": query_embedding.astype(float).tolist(),
                        "numCandidates": num_candidates,
                        "limit": top_k,
                        **({"filter": filter_query} if filter_query else {}),
                    }
                },
                {"$project": {"_id": 0, "score": {"$meta": "vectorSearchScore"}, "chunk_id": 1, "text": 1, "metadata": 1, "embedding": 0}},
            ]
            docs = [doc async for doc in col.aggregate(pipeline)]
            return [(float(doc.get("score", 0.0)), {k: v for k, v in doc.items() if k != "score"}) for doc in docs]
        except Exception:
            # Fall back to local cosine if index not ready or unsupported environment
            pass

    # Local cosine fallback
    cursor = col.find(filter_query, {"_id": 0})
    docs = [doc async for doc in cursor]
    if not docs:
        return []
    matrix = np.array([doc.get("embedding", []) for doc in docs], dtype=np.float32)
    if matrix.size == 0:
        return []
    q = query_embedding.astype(np.float32)
    q_norm = np.linalg.norm(q) + 1e-8
    m_norm = np.linalg.norm(matrix, axis=1) + 1e-8
    sims = (matrix @ q) / (m_norm * q_norm)
    top_idx = np.argsort(-sims)[:top_k]
    results: List[Tuple[float, Dict[str, Any]]] = []
    for idx in top_idx:
        score = float(sims[idx])
        results.append((score, docs[int(idx)]))
    return results


