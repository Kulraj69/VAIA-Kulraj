"""
Test script to create embeddings and store them in ChromaDB.
This demonstrates the ChromaDB integration working.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.llm import embed_texts
from app.vector_store import upsert_vectors
from app.chroma_db import get_chroma_collection

# Sample market research content for testing
SAMPLE_DOCUMENTS = [
    {
        "text": "The AI market is experiencing rapid growth with a projected CAGR of 22% through 2030. Key drivers include cloud adoption, enterprise automation, and machine learning advancements.",
        "chunk_id": "market_trends_chunk_1",
        "metadata": {
            "section": "market_trends",
            "source_file": "market_research_2025.txt",
            "index": 0
        }
    },
    {
        "text": "Strategic recommendations focus on three core areas: competitive positioning through innovation, market expansion into healthcare and finance sectors, and pricing optimization to maintain market share.",
        "chunk_id": "strategy_chunk_1",
        "metadata": {
            "section": "strategic_recommendations",
            "source_file": "market_research_2025.txt",
            "index": 1
        }
    },
    {
        "text": "SWOT Analysis: Strengths include robust technology stack and strong customer loyalty. Weaknesses: slower feature rollout and higher pricing. Opportunities: healthcare and finance expansion. Threats: aggressive pricing from competitors.",
        "chunk_id": "analysis_chunk_1",
        "metadata": {
            "section": "swot_analysis",
            "source_file": "market_research_2025.txt",
            "index": 2
        }
    },
    {
        "text": "Innovate Inc. holds 12% market share in the AI software segment. Top competitors include Synergy Systems (18%), FutureFlow (15%), and emerging QuantumLeap (3%).",
        "chunk_id": "competitive_chunk_1",
        "metadata": {
            "section": "competitive_landscape",
            "source_file": "market_research_2025.txt",
            "index": 3
        }
    }
]


async def test_chromadb_integration():
    """Test ChromaDB by creating embeddings and storing them."""
    print("🚀 Testing ChromaDB Integration\n")
    
    # Check if secrets are loaded
    import os
    if not os.environ.get("CHROMA_API_KEY"):
        print("❌ CHROMA_API_KEY not found. Loading secrets from .env...")
        from dotenv import load_dotenv
        load_dotenv()
    
    if not os.environ.get("CHROMA_API_KEY"):
        print("❌ Please set CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE environment variables")
        return
    
    print("✅ ChromaDB credentials loaded")
    
    try:
        # Test 1: Connect to ChromaDB
        print("\n1️⃣ Testing ChromaDB connection...")
        collection = get_chroma_collection()
        print("✅ Connected to ChromaDB collection:", collection.name)
        
        # Test 2: Generate embeddings
        print("\n2️⃣ Generating embeddings for sample documents...")
        texts = [doc["text"] for doc in SAMPLE_DOCUMENTS]
        embeddings = await embed_texts(texts)
        print(f"✅ Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
        
        # Test 3: Prepare data for upsert
        print("\n3️⃣ Preparing data for ChromaDB...")
        items = []
        for i, doc in enumerate(SAMPLE_DOCUMENTS):
            items.append({
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "embedding": embeddings[i].tolist(),
                "metadata": doc["metadata"]
            })
        print(f"✅ Prepared {len(items)} items")
        
        # Test 4: Upsert to ChromaDB
        print("\n4️⃣ Upserting to ChromaDB...")
        await upsert_vectors(items)
        print("✅ Successfully upserted vectors to ChromaDB")
        
        # Test 5: Verify data in ChromaDB
        print("\n5️⃣ Verifying stored data...")
        all_data = collection.get(ids=[doc["chunk_id"] for doc in SAMPLE_DOCUMENTS])
        print(f"✅ Found {len(all_data['ids'])} documents in collection")
        
        # Test 6: Test similarity search
        print("\n6️⃣ Testing similarity search...")
        from app.vector_store import similarity_search
        
        # Search for market trends
        query_text = "What are the key market trends and growth patterns?"
        print(f"   Query: {query_text}")
        query_emb = await embed_texts([query_text])
        results = await similarity_search(query_emb[0], top_k=2)
        
        print(f"   Found {len(results)} results:")
        for i, (score, doc) in enumerate(results):
            print(f"   {i+1}. Score: {score:.4f}")
            print(f"      Chunk: {doc['chunk_id']}")
            print(f"      Text: {doc['text'][:80]}...")
        
        print("\n✅ All tests passed! ChromaDB integration is working.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_chromadb_integration())

