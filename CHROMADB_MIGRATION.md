# ChromaDB Migration Guide

## Overview

The system has been migrated from MongoDB to ChromaDB for vector storage. ChromaDB provides:
- Better vector search performance
- Simplified vector database operations
- Cloud-hosted solution with easy scaling
- Native vector similarity search

## Architecture Changes

### Before (MongoDB)
- MongoDB with vector search indexes
- Manual cosine similarity calculation for fallback
- More complex aggregation pipelines

### After (ChromaDB)
- ChromaDB Cloud client connection
- Native vector similarity search
- Simplified upsert and query operations
- Async-safe with thread pool execution

## Configuration

### Environment Variables

Add these to your `.streamlit/secrets.toml` or Streamlit Cloud secrets:

```toml
[chromadb]
api_key = "your-chroma-api-key"
tenant = "your-chroma-tenant-id"
database = "your-chroma-database-name"
```

### Streamlit Cloud Setup

In Streamlit Cloud settings, add secrets with the same structure:

```
[chromadb]
api_key = your-actual-api-key
tenant = your-actual-tenant-id
database = your-actual-database-name
```

## Testing

### Run Integration Test

```bash
python test_chromadb.py
```

This will:
1. Connect to ChromaDB
2. Generate embeddings for sample documents
3. Store them in ChromaDB
4. Test similarity search

### Manual Test via FastAPI

```bash
# Start server
uvicorn app.main:app --reload

# Ingest documents
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@data/innovate_inc_q3_2025.txt" \
  -F "section=market_research"

# Query with specialized agent
curl -X POST "http://localhost:8000/specialized-agent" \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the key market trends?","top_k":4}'
```

## Migration Notes

### Key Changes

1. **vector_store.py**: Completely rewritten for ChromaDB
   - `upsert_vectors()` now uses ChromaDB collection.upsert()
   - `similarity_search()` uses ChromaDB collection.query()
   - Async-safe with `run_in_executor()` for sync operations

2. **chroma_db.py**: New module
   - `get_chroma_client()`: Cloud client connection
   - `get_chroma_collection()`: Collection management
   - Global connection caching

3. **requirements.txt**: Added `chromadb==0.5.15`

### Backward Compatibility

- All existing agent functions work the same way
- No changes to API endpoints
- Same data structures for upserts and queries
- Seamless transition for users

## Data Migration

If you have existing data in MongoDB:

1. Export vectors from MongoDB
2. Convert to ChromaDB format
3. Run batch upsert with `upsert_vectors()`

Example migration script:

```python
# Export from MongoDB
from app.db import get_db

async def export_mongodb_vectors():
    db = await get_db()
    col = db["vectors"]
    vectors = []
    async for doc in col.find({}, {"_id": 0}):
        vectors.append(doc)
    return vectors

# Import to ChromaDB
from app.vector_store import upsert_vectors

async def import_to_chromadb(vectors):
    await upsert_vectors(vectors)
```

## Performance

ChromaDB advantages:
- **Faster queries**: Native vector search vs manual calculation
- **Better scaling**: Cloud-hosted automatic scaling
- **Simpler ops**: No index management needed
- **Cost**: Efficient pricing model for vector databases

## Troubleshooting

### Connection Issues

```python
# Check credentials
import os
print(os.environ.get("CHROMA_API_KEY"))
print(os.environ.get("CHROMA_TENANT"))
```

### Empty Results

Ensure documents were properly upserted:
```python
from app.chroma_db import get_chroma_collection
collection = get_chroma_collection()
print(collection.count())
```

### Async Errors

ChromaDB operations are sync, wrapped in async executor. If issues occur, check:
1. Event loop is running
2. Proper async/await usage
3. No blocking in main thread

## Next Steps

- ✅ ChromaDB integration complete
- ✅ Test script created
- ✅ Secrets configuration updated
- ⏳ Production deployment testing
- ⏳ Performance benchmarking
- ⏳ Data migration from MongoDB (if needed)

