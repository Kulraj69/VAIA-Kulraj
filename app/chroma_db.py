from typing import Optional
import os
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection


# Environment variable names
CHROMA_API_KEY_ENV = "CHROMA_API_KEY"
CHROMA_TENANT_ENV = "CHROMA_TENANT"
CHROMA_DATABASE_ENV = "CHROMA_DATABASE"

# Global connection instances
_client: Optional[ClientAPI] = None
_collection: Optional[Collection] = None


def get_chroma_client() -> ClientAPI:
    """Get or create ChromaDB Cloud client."""
    global _client
    if _client is None:
        api_key = os.environ.get(CHROMA_API_KEY_ENV)
        tenant = os.environ.get(CHROMA_TENANT_ENV)
        database = os.environ.get(CHROMA_DATABASE_ENV)
        
        if not api_key or not tenant:
            raise RuntimeError("CHROMA_API_KEY and CHROMA_TENANT must be set")
        
        _client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database if database else None
        )
    return _client


def get_chroma_collection(collection_name: str = "vaia_collection") -> Collection:
    """Get or create a ChromaDB collection."""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(name=collection_name)
    return _collection


def reset_connection():
    """Reset global connections (useful for testing)."""
    global _client, _collection
    _client = None
    _collection = None

