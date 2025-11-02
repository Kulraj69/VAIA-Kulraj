# AI Market Analyst — VAIA Take-Home

An end-to-end retrieval-augmented generation (RAG) pipeline that ingests market research documents and provides three key capabilities: Q&A, Market Findings (summary), and Structured Data Extraction (JSON).

## Table of Contents

- [Quick Start](#quick-start)
- [Setup & Installation](#setup--installation)
- [API Endpoints](#api-endpoints)
- [Design Decisions](#design-decisions)
- [Advanced Features](#advanced-features)
- [Testing & Validation](#testing--validation)

## Quick Start

### Start the Server

```bash
# Create virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# MongoDB Setup (use MongoDB Atlas or local Docker)
docker run -d --name mongo -p 27017:27017 mongo:latest

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Example API Calls

**1. Q&A Endpoint**

```bash
curl -X POST "http://localhost:8000/qa" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is Innovate Inc. market share and top competitors?"}'
```

**2. Market Findings (Summary)**

```bash
curl -X POST "http://localhost:8000/summary" \
  -H "Content-Type: application/json" \
  -d '{"query":"Summarize the competitive landscape and strategic priorities."}'
```

**3. Structured Data Extraction**

```bash
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{"query":"Extract the SWOT analysis and financial projections as JSON."}'
```

**4. Chat with Memory**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"user-session-1", "query":"How has Innovate Inc.\'s growth changed quarter over quarter?"}'
```

**5. Autonomous Routing (Optional)**

```bash
curl -X POST "http://localhost:8000/route" \
  -H "Content-Type: application/json" \
  -d '{"query":"Should Innovate lower price or speed up feature releases?"}'
```

## Setup & Installation

### 1. Clone Repository

```bash
git clone https://github.com/Kulraj69/VAIA-Kulraj.git
cd VAIA-Kulraj
```

### 2. Create Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY="sk-..."
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_API_VERSION="2024-02-15-preview"
AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4"
AZURE_OPENAI_EMBED_DEPLOYMENT="text-embedding-3-large"

# MongoDB Configuration
MONGODB_URI="mongodb://localhost:27017"
MONGO_DB_NAME="vaia_market_analyst"

# Optional: Pinecone or other vector DB
PINECONE_API_KEY="..."
```

### 4. Ingest Document

```bash
python -m app.ingest --file data/innovate_inc_q3_2025.txt
```

This will:
- Split the document into chunks
- Compute embeddings
- Store vectors + metadata in MongoDB (collection: 'vectors')

### 5. Run Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Streamlit UI (Optional)

A minimal Streamlit UI is available for interactive testing:

```bash
streamlit run app/streamlit_app.py
```

## API Endpoints

### 1. Q&A — `/qa`

**Method:** POST

**Request Body:**
```json
{
  "query": "What is Innovate Inc.'s market share?",
  "top_k": 4
}
```

**Response:**
```json
{
  "answer": "Innovate Inc. holds a 12% market share.",
  "sources": [
    {
      "chunk_id": "sec3_chunk1",
      "section": "Competitive Landscape"
    }
  ]
}
```

**Flow:**
1. Retrieve top_k chunks by cosine similarity
2. Compose prompt: system instructions + relevant chunks + user query
3. Call LLM with temperature 0.0 (deterministic)
4. Return answer + used chunk IDs

---

### 2. Market Findings (Summarization) — `/summary`

**Method:** POST

**Request Body:**
```json
{
  "query": "Summarize opportunities and threats for Innovate Inc.",
  "max_length": 300
}
```

**Response:**
A concise findings summary with bullets and action items.

**Flow:**
1. Retrieve relevant chunks
2. Use a summarize prompt tuned for market research findings
3. Temperature 0.0–0.2

---

### 3. Extract — `/extract`

**Method:** POST

**Request Body:**
```json
{
  "query": "Extract full SWOT, market size, competitors as JSON."
}
```

**Response:**
Strict JSON object matching the schema provided.

**Flow:**
1. Retrieve chunks
2. Use schema-enforced prompt
3. Validate JSON; attempt correction if malformed

---

### 4. Chat with Memory — `/query`

**Method:** POST

**Request Body:**
```json
{
  "session_id": "user-session-1",
  "query": "Follow up on last answer: what sectors fuel this growth?"
}
```

**Response:**
```json
{
  "answer": "The healthcare and finance sectors were key drivers of Q3 growth...",
  "sources": [...],
  "memory": [...]
}
```

**Flow:**
1. Retrieve prior session memory using session_id
2. Retrieve top_k chunks by vector similarity
3. Compose prompt: system instructions + relevant chunks + session history + user query
4. Call LLM with memory context
5. Store new Q&A in session memory
6. Return answer, used chunks, updated memory

---

### 5. Autonomous Routing — `/route`

**Method:** POST

**Request Body:**
```json
{
  "query": "What are the key market trends?"
}
```

**Response:**
Automatically routes to Q&A, Summary, or Extract based on query intent.

## Design Decisions

### Chunking Strategy

**Choice:** 350 tokens chunk size, 50 tokens overlap (token-based)

**Rationale:**
- Reports are semantically rich; 300–400 tokens preserve paragraph/section context
- 50-token overlap ensures boundary entities are captured in at least one chunk
- Token-based approach ensures consistency and avoids mid-token breaks

**Implementation:**
- Use tokenizer (tiktoken or HuggingFace) to count tokens
- Include section and chunk_id in metadata for provenance

### Embedding Model

**Primary:** `text-embedding-3-small` (OpenAI)

**Fallback:** `all-MiniLM-L6-v2` (sentence-transformers)

**Rationale:**
- `text-embedding-3-small` provides high-quality embeddings with fast latency
- `all-MiniLM-L6-v2` offers efficient open-source alternative for offline/credit-constrained scenarios

**Benchmarking:**
- Measure retrieval accuracy on ground-truth questions
- Measure latency over N runs
- Report tradeoffs and recommend primary model

### Vector Database

**Primary:** MongoDB

**Rationale:**
- Flexible, scalable, widely used
- Supports vector search with indexes (MongoDB Atlas Vector Search)
- Stores both embeddings and chat memory for sessions
- Consolidates vector and memory storage in a single backend

**Implementation:**
- Database: `MONGO_DB_NAME`
- Collections: `vectors` (embeddings), `memory` (chat history)
- Metadata: section, chunk_id, start_offset, source_file

### Retrieval Strategy

- Dense retrieval with top-K (K=4) candidates
- Reranking/LLM-based synthesis
- Include provenance: list chunk IDs/sections used

### Data Extraction Prompt Design

**Principles:**
1. Provide strict JSON schema up-front
2. Explicit instructions: "Return only JSON, do not add any explanatory text"
3. Include examples of expected JSON for ambiguous fields
4. Add validation pass with re-prompt for malformed JSON
5. Post-parse checks for field existence and types

**Example JSON Schema (SWOT):**

```json
{
  "company": "Innovate Inc.",
  "report_date": "Q3 2025",
  "market_size": {
    "current_usd": 15000000000,
    "cagr_percent": 22,
    "projected_2030_usd": 40000000000
  },
  "market_share_percent": 12,
  "competitors": [
    {
      "name": "Synergy Systems",
      "market_share_percent": 18
    },
    {
      "name": "FutureFlow",
      "market_share_percent": 15
    },
    {
      "name": "QuantumLeap",
      "market_share_percent": 3,
      "notes": "emerging, significant VC funding"
    }
  ],
  "swot": {
    "strengths": [
      "Robust and scalable architecture of Automata Pro",
      "Strong customer loyalty"
    ],
    "weaknesses": [
      "Slower feature rollout compared to competitors",
      "Higher price point"
    ],
    "opportunities": [
      "Expansion into the healthcare sector",
      "Expansion into the finance sector"
    ],
    "threats": [
      "Aggressive pricing from Synergy Systems",
      "Rapid innovation from QuantumLeap"
    ]
  },
  "conclusion": "Innovate Inc. is well-positioned for growth but must address feature velocity and pricing."
}
```

**Prompt Skeleton:**

```
You are a data extraction assistant. Given the following retrieved text chunks, extract the following JSON structure exactly as specified. Only return valid JSON. Do not add any explanation. If information is missing, use null for numeric fields or empty arrays for lists. Ensure numbers are integers or floats appropriately.

[CHUNKS]

Return the JSON now.
```

## Advanced Features

### Autonomy & Routing

**Approach:** Implement `router.py` with `decide_task(query)`.

**Options:**
1. **LLM-based:** Send short prompt to model: "Should this query be answered via Q&A, Summary, or Extract? Output one of ['QA','SUMMARY','EXTRACT'] only."
2. **Rule-based fallback:** Regex checks for keywords (e.g., "extract", "JSON", "SWOT" → EXTRACT)

**Rationale:** LLM provides flexible intent understanding; rules ensure determinism.

### Embedding Comparative Evaluation

**Comparison:** `text-embedding-3-small` vs `all-MiniLM-L6-v2`

**Metrics:**
- Retrieval quality: 10 queries, top-1/top-3 recall
- Latency: average embed time for 300-token chunk

Summarize results in README with sample retrieval accuracy table.

### Docker Setup

**Dockerfile:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Streamlit UI

Features:
- Upload document or select sample
- Ingestion status
- Query box with "auto" or tool selection
- Answer + source chunks + JSON (for extract) with copy button
- Simple and demonstrative interface

## Testing & Validation

### Unit Tests

- Chunker: preserve sentence boundaries
- Embeddings wrapper: correct shape
- Retriever: returns top_k
- JSON validator

### Integration Tests

Ingest file → run `/extract` on "SWOT" query → validate JSON schema.

### Demo Video

3–5 minute screencast:
1. Repo and README quick scan (30s)
2. Ingest pipeline running (45s)
3. Hitting `/qa` with sample query (30s)
4. Hitting `/summary` and `/extract` (45s)
5. Routing auto-detect and/or Streamlit UI (30–60s)

Host video as `demo_video.mp4` or link in README.

## Security, Costs & Notes

- **API Keys:** Keep out of repo (use `.env` or CI secrets)
- **Costs:** Small datasets with Chroma and OpenAI embeddings are low-cost
- **Caching:** Use low temperature for deterministic outputs; cache embeddings
- **Documentation:** Full FastAPI app with four endpoints, `ingest.py`, prompts, requirements.txt, Dockerfile, tests, and demo video

## Code Examples

### Token-Aware Chunking

```python
from tiktoken import encoding_for_model

enc = encoding_for_model("gpt-4o-mini")

def chunk_text(text, max_tokens=350, overlap=50):
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start = end - overlap
        if start < 0:
            start = 0
    return chunks
```

### RAG Prompt Skeleton

```
SYSTEM: You are a helpful market research analyst. Use the provided source chunks only to answer the user's question. Cite the chunk IDs you used.

CHUNKS:
[...]

USER: {user_query}
```

### Extraction Prompt Skeleton

```
You are an extractor. Given the following chunks, return EXACTLY ONE JSON object matching this schema: {...}. Return only JSON.

CHUNKS: [...]
```

## Repository Contents

- ✅ Full FastAPI app with four endpoints (added `/query` for memory/chat)
- ✅ `ingest.py` for token-aware chunking + embeddings
- ✅ `prompts.py` with templates
- ✅ `requirements.txt` and Dockerfile
- ✅ README.md (this file)
- ✅ Small test suite and demo_video.mp4
