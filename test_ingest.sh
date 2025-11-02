#!/bin/bash
# Test script to ingest PDF into ChromaDB via FastAPI

echo "🚀 Starting FastAPI ingestion test..."

# Check if server is running
if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
    echo "✅ FastAPI server is running"
else
    echo "❌ FastAPI server not running. Please start it first:"
    echo "   uvicorn app.main:app --reload --port 8000"
    exit 1
fi

echo ""
echo "📄 Ingesting market.pdf..."

# Ingest the PDF
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@market.pdf" \
  -F "section=market_research" \
  -F "max_chars=2000" \
  -F "overlap_chars=200"

echo ""
echo ""
echo "✅ Done! Check ChromaDB to verify data was stored."

