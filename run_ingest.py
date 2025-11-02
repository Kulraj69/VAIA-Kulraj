#!/usr/bin/env python3
"""
Quick script to ingest PDF into ChromaDB via FastAPI.
This simulates what the UI does.
"""
import requests
import sys
from pathlib import Path

def ingest_pdf(file_path: str):
    """Ingest a PDF file into ChromaDB via FastAPI."""
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/docs", timeout=2)
        if response.status_code != 200:
            print("❌ FastAPI server responded with error")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to FastAPI server at http://localhost:8000")
        print("   Please start it first:")
        print("   uvicorn app.main:app --reload --port 8000")
        return False
    
    print("✅ FastAPI server is running")
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"📄 Ingesting {file_path}...")
    
    # Prepare file upload
    files = {
        'file': ('market.pdf', open(file_path, 'rb'), 'application/pdf')
    }
    data = {
        'section': 'market_research',
        'max_chars': '2000',
        'overlap_chars': '200'
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/ingest",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Ingested {result.get('chunks', 0)} chunks")
            print(f"   Section: {result.get('section', 'N/A')}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False
    finally:
        files['file'][1].close()


if __name__ == "__main__":
    # Default to market.pdf if no argument provided
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "market.pdf"
    
    print("🚀 ChromaDB Ingestion Test\n")
    success = ingest_pdf(pdf_file)
    
    if success:
        print("\n✅ Data successfully added to ChromaDB!")
        print("   You can now query it via /specialized-agent endpoint")
    else:
        sys.exit(1)

