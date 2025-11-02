from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import asyncio

from .chroma_db import get_chroma_collection


VECTORS_COLLECTION_NAME = "vaia_collection"


async def upsert_vectors(items: List[Dict[str, Any]]) -> None:
    """
    Upsert vectors to ChromaDB collection.
    
    items: [{
        "chunk_id": str,
        "text": str,
        "embedding": List[float],
        "metadata": Dict[str, Any],
    }]
    """
    collection = get_chroma_collection(VECTORS_COLLECTION_NAME)
    
    # Prepare data for ChromaDB
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    
    for item in items:
        ids.append(item["chunk_id"])
        documents.append(item["text"])
        embeddings.append(item["embedding"])
        # Flatten metadata for ChromaDB
        metadata = item.get("metadata", {}).copy()
        metadatas.append(metadata)
    
    # Run ChromaDB upsert in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    )


async def similarity_search(query_embedding: np.ndarray, top_k: int, source_file: Optional[str] = None) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Search for similar vectors in ChromaDB.
    
    Returns: List of (score, document) tuples sorted by similarity.
    """
    collection = get_chroma_collection(VECTORS_COLLECTION_NAME)
    
    # Prepare query embedding
    query_embedding_list = query_embedding.tolist()
    
    # Build where filter for source_file if provided
    where_filter = None
    if source_file:
        where_filter = {"source_file": source_file}
    
    # Run ChromaDB query in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: collection.query(
            query_embeddings=[query_embedding_list],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
    )
    
    # Transform results to match expected format
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]
    
    # Convert distances to similarity scores (ChromaDB returns distances, higher is less similar)
    # We'll convert to similarity by negating distance for simplicity
    results_list: List[Tuple[float, Dict[str, Any]]] = []
    for idx in range(len(documents)):
        distance = distances[idx]
        # Convert distance to similarity score (assume cosine distance, smaller is better)
        # Similarity = 1 - distance (for cosine distance)
        similarity_score = 1.0 - distance if distance <= 1.0 else 0.0
        
        doc = {
            "chunk_id": ids[idx],
            "text": documents[idx],
            "metadata": metadatas[idx]
        }
        results_list.append((similarity_score, doc))
    
    # Sort by similarity score (highest first)
    results_list.sort(key=lambda x: x[0], reverse=True)
    
    return results_list
