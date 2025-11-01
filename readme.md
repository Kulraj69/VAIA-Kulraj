AI Market Analyst — VAIA Take-Home

An end-to-end retrieval + LLM pipeline that ingests a market research doc for Innovate Inc. and exposes three tools: Q&A, Market Findings (summary), and Structured Data Extraction (JSON).

Quick demo (examples)

Start server

# create venv & install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# MongoDB setup for Vector Store
# Ensure MongoDB is running and accessible via the URI below. You can use MongoDB Atlas or local Docker:
# docker run -d --name mongo -p 27017:27017 mongo:latest

# run
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


Example: QA

curl -X POST "http://localhost:8000/qa" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is Innovate Inc. market share and top competitors?"}'


Example: Market findings (summary)

curl -X POST "http://localhost:8000/summary" \
  -H "Content-Type: application/json" \
  -d '{"query":"Summarize the competitive landscape and strategic priorities."}'


Example: Extract (structured JSON)

curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{"query":"Extract the SWOT analysis and financial projections as JSON."}'


Example: Chat with Memory (NEW)

curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"user-session-1", "query":"How has Innovate Inc.'s growth changed quarter over quarter?"}'


Optional: Autonomous routing

curl -X POST "http://localhost:8000/route" \
  -H "Content-Type: application/json" \
  -d '{"query":"Should Innovate lower price or speed up feature releases?"}'

Setup & Run Instructions

Clone repo:

git clone https://github.com/<your-username>/ai-market-analyst.git
cd ai-market-analyst


Create environment:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


Add API keys (environment variables):

export OPENAI_API_KEY="sk-..."
export MONGODB_URI="mongodb://localhost:27017"  # Update as needed
export MONGO_DB_NAME="vaia_market_analyst"
# if using Pinecone or other DB:
export PINECONE_API_KEY="..."


Ingest the document (one-time)

python -m app.ingest --file data/innovate_inc_q3_2025.txt


This will:

split the document into chunks

compute embeddings

store vectors + metadata in MongoDB (collection: 'vectors')

Run server:

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


Use the endpoints (see examples above). A minimal Streamlit UI is available in ui/ (run streamlit run ui/app.py).

Design Decisions (Crucial)
Chunking Strategy

Choice: 350 tokens chunk size, 50 tokens overlap (token-based)

Why:

The report is short but semantically rich; chunking by ~300–400 tokens preserves paragraph/section context while keeping retrieval efficient.

50-token overlap ensures that entities or sentences that fall near boundaries are preserved in at least one chunk, improving retrieval recall.

Token-based (vs naive character split) ensures consistency across languages and avoids breaking mid-token or mid-sentence.

Implementation notes:

Use a tokenizer (e.g., tiktoken or HuggingFace tokenizer) to count tokens.

When saving chunk metadata, include section and chunk_id for provenance.

Embedding Model

Primary choice: text-embedding-3-small (OpenAI) or all-MiniLM-L6-v2 (sentence-transformers) as fallback.

Why:

text-embedding-3-small provides high-quality embeddings with good retrieval performance and faster latency for small/medium docs.

all-MiniLM-L6-v2 is an efficient, open-source alternative (smaller, fast local inference) — useful for an offline/credit-constrained comparison (bonus).

How to benchmark (Bonus 2):

Measure retrieval accuracy: query the ground-truth questions and check whether top-k retrieved chunks contain the expected answers (binary match).

Measure latency over N runs. Report tradeoffs and recommend primary model.

Vector Database

Primary choice: MongoDB

Why:

Flexible, scalable, widely used; supports vector search with indexes (e.g., using MongoDB Atlas Vector Search), can store both embeddings and chat memory for sessions.

Good for scaling and persistence, works well with both local and cloud environments. Consolidates vector and memory storage in a single DB backend.

Implementation details:

Use MongoDB database (env: MONGO_DB_NAME), a 'vectors' collection for chunk embeddings, and 'memory' collection for chats.

Store chunk text, embedding vector (array), metadata fields (section, chunk_id, start_offset, source_file).

Memory: For each session, store ordered chat turns (question, answer, timestamp, referenced chunk_ids).

Retrieval strategy

Use dense retrieval with top-K (K=4) candidates, followed by a reranking/LLM-based synthesis step where the LLM is given the retrieved chunks and asked to answer or summarize.

Always include provenance: list the chunk IDs/sections used.

Data Extraction Prompt (structured JSON)

Goal: Reliable JSON output for things like SWOT, market size numbers, competitor entries.

Prompt design principles:

Provide a strict JSON schema up-front (example below).

Give explicit instructions: "Return only JSON, do not add any explanatory text."

Include examples of expected JSON for ambiguous fields.

Add validation pass: If the LLM returns malformed JSON, re-prompt (or apply a small normalization script) to correct basic issues (unclosed brackets, trailing commas).

Post-parse checks: Ensure fields exist and types are correct; if not, attempt a one-shot fix by asking the model to "Return valid JSON that fits this schema" including the last model text as context.

Example JSON schema (SWOT):

{
  "company":"Innovate Inc.",
  "report_date":"Q3 2025",
  "market_size": {
    "current_usd": 15000000000,
    "cagr_percent": 22,
    "projected_2030_usd": 40000000000
  },
  "market_share_percent": 12,
  "competitors": [
    {"name":"Synergy Systems","market_share_percent":18},
    {"name":"FutureFlow","market_share_percent":15},
    {"name":"QuantumLeap","market_share_percent":3,"notes":"emerging, significant VC funding"}
  ],
  "swot": {
    "strengths":[ "Robust and scalable architecture of Automata Pro", "Strong customer loyalty" ],
    "weaknesses":[ "Slower feature rollout compared to competitors", "Higher price point" ],
    "opportunities":[ "Expansion into the healthcare sector", "Expansion into the finance sector" ],
    "threats":[ "Aggressive pricing from Synergy Systems", "Rapid innovation from QuantumLeap" ]
  },
  "conclusion":"Innovate Inc. is well-positioned for growth but must address feature velocity and pricing."
}


Prompt skeleton to LLM for extraction:

You are a data extraction assistant. Given the following retrieved text chunks (provide chunks), extract the following JSON structure exactly as specified (show schema). Only return valid JSON. Do not add any explanation. If information is missing, use null for numeric fields or empty arrays for lists. Ensure numbers are integers or floats appropriately.
[CHUNKS]
Return the JSON now.

API Usage (detailed)
1) Q&A — /qa

Method: POST
Body:

{"query":"What is Innovate Inc.'s market share?","top_k":4}


Server flow:

Retrieve top_k chunks by cosine similarity

Compose a prompt: system instructions + relevant chunks + user query

Call LLM with temperature 0.0 (deterministic)

Return answer + used chunk ids

Response:

{
  "answer":"Innovate Inc. holds a 12% market share.",
  "sources":[{"chunk_id":"sec3_chunk1","section":"Competitive Landscape"}]
}

2) Market findings (summarization) — /summary

Method: POST
Body:

{"query":"Summarize opportunities and threats for Innovate Inc.","max_length":300}


Server flow:

Retrieve relevant chunks

Use a summarize prompt tuned for market research findings (bullet list)

Temperature 0.0–0.2

Response: A concise findings summary with bullets and action items.

3) Extract — /extract

Method: POST
Body:

{"query":"Extract full SWOT, market size, competitors as JSON."}


Server flow:

Retrieve chunks

Use schema-enforced prompt above

Validate JSON; attempt correction if malformed.

Response: Strict JSON object matching schema.

4) Chat with Memory — /query

Method: POST
Body:

{"session_id":"user-session-1", "query":"Follow up on last answer: what sectors fuel this growth?"}

Server flow:
- Retrieve prior session memory (chat turns) using session_id
- Retrieve top_k chunks by vector similarity
- Compose prompt: system instructions + relevant chunks + session history + user query
- Call LLM (with memory context), deterministic output
- Store new user question and LLM answer in session memory
- Return answer, used chunks, updated memory

Response:
{
  "answer": "The healthcare and finance sectors were key drivers of Q3 growth...",
  "sources": [...],
  "memory": [...]  # All chat turns in order
}

Autonomy & Routing (Bonus 1)

Approach: Build router.py containing decide_task(query). Two options:

LLM-based: send short prompt to model: "Should this query be answered via Q&A, Summary, or Extract? Output one of ['QA','SUMMARY','EXTRACT'] only." Use temperature 0.

Rule-based fallback: simple regex checks: presence of "extract", "JSON", "SWOT", numeric terms → EXTRACT; "summarize", "findings", "key takeaways" → SUMMARY; else QA.

Why: LLM gives flexible intent understanding; rules ensure determinism for edge cases.

Embedding Comparative Evaluation (Bonus 2)

Plan:

Compare text-embedding-3-small vs all-MiniLM-L6-v2 on:

Retrieval quality: prepare 10 queries (e.g., "What is the CAGR?", "List competitors and market shares") — evaluate top-1/top-3 recall.

Latency: average embed time for a 300-token chunk.

Summarize results in README: include sample retrieval accuracy table and recommend model for this task.

Dockerfile (Bonus 3) — example
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

UI (Bonus 4)

Streamlit app that:

Uploads document (or selects sample)

Shows ingestion status

Query box with option "auto" or tool selection

Shows answer + source chunks + JSON (for extract) with copy button

Keep UI simple and demonstrative.

Testing & Validation

Unit tests for chunker (preserve sentence boundaries), embeddings wrapper (correct shape), retriever (returns top_k), JSON validator.

Integration test: ingest file → run /extract on "SWOT" query → validate JSON schema.

Demo Video

Make a 3–5 minute screencast showing:

Repo and README quick scan (30s)

Ingest pipeline running and producing vectors (45s)

Hitting /qa with sample query (30s)

Hitting /summary and /extract (45s)

If implemented: show routing auto-detect and/or the Streamlit UI (30–60s)

Host the video in the repo as demo_video.mp4 (or link in README to hosted video).

Security, Costs & Notes

Keep API keys out of repo (use .env or CI secrets).

For small dataset (this exercise) Chroma and small OpenAI embedding calls are low-cost. If using paid LLM endpoints, use low temperature for deterministic outputs and cache embeddings.

Example code snippets

Chunking (Python, token-aware)

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


RAG prompt skeleton

SYSTEM: You are a helpful market research analyst. Use the provided source chunks only to answer the user's question. Cite the chunk IDs you used.
CHUNKS:
[...]
USER: {user_query}


Extraction prompt skeleton

You are an extractor. Given the following chunks, return EXACTLY ONE JSON object matching this schema: {...}. Return only JSON.
CHUNKS: [...]

What I'll include in the repo

Full FastAPI app with four endpoints (added /query for memory/chat)

ingest.py for token-aware chunking + embeddings.

prompts.py with templates.

requirements.txt and Dockerfile.

README.md (this file).

small test suite and demo_video.mp4.