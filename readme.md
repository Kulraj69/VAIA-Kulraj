# AI Market Analyst — VAIA

An end-to-end retrieval-augmented generation (RAG) pipeline with **intelligent specialized agents** for market analysis. Features **ChromaDB** vector storage, **Azure OpenAI** embeddings, and **LangGraph** agent orchestration.

## 🌟 Key Features

- **3 Specialized Agents**: TRENDS, STRATEGY, and ANALYSIS with intelligent routing
- **ChromaDB Integration**: Cloud-hosted vector database for fast similarity search
- **Streamlit UI**: Interactive document chat with PDF ingestion
- **FastAPI Backend**: Production-ready API with async operations
- **Session Memory**: Contextual conversations with chat history
- **Multiple Modes**: Q&A, Summarization, Extraction, and Smart Routing

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Specialized Agents](#specialized-agents)
- [API Endpoints](#api-endpoints)
- [Streamlit UI](#streamlit-ui)
- [Testing](#testing)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/Kulraj69/VAIA-Kulraj.git
cd VAIA-Kulraj
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Credentials

**For Local Development** - Create `.streamlit/secrets.toml`:

```toml
[azure_openai]
endpoint = "https://your-endpoint.openai.azure.com/"
api_key = "your-api-key"
api_version = "2025-01-01-preview"
deployment = "gpt-4o-mini"
embed_deployment = "text-embedding-3-small"

[chromadb]
api_key = "your-chromadb-api-key"
tenant = "your-tenant-id"
database = "your-database-name"
```

**For FastAPI** - Create `.env`:

```bash
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small

CHROMA_API_KEY=your-chromadb-api-key
CHROMA_TENANT=your-tenant-id
CHROMA_DATABASE=your-database-name
```

### 3. Run Server

**FastAPI:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Streamlit:**
```bash
streamlit run streamlit_app.py
# or
streamlit run app/streamlit_app.py
```

### 4. Test It

```bash
# Ingest a document
python run_ingest.py market.pdf

# Query the specialized agent
curl -X POST "http://localhost:8000/specialized-agent" \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the key market trends?","top_k":4}'
```

## 🏗️ Architecture

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Router Agent (LLM Reasoning)   │
│  Selects: TRENDS|STRATEGY|ANALYSIS
└────────┬────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Query Enhancement           │
│  Based on agent specialization
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  ChromaDB Vector Search      │
│  Retrieve top-K chunks       │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Specialized Agent           │
│  Agent-specific prompts      │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Answer + Sources + Agent Type│
└──────────────────────────────┘
```

## 🤖 Specialized Agents

### TRENDS Agent
**Focus:** Market trends, growth patterns, forecasts, emerging developments

**Best for:**
- "What are the emerging trends in AI market?"
- "Show market growth projections"
- "What are the forecasted dynamics?"

### STRATEGY Agent
**Focus:** Strategic recommendations, competitive positioning, tactics

**Best for:**
- "What strategies should we adopt?"
- "How can we improve competitive positioning?"
- "Recommend go-to-market approaches"

### ANALYSIS Agent
**Focus:** Comprehensive analysis, SWOT, competitive landscape, detailed insights

**Best for:**
- "Perform a SWOT analysis"
- "Analyze the competitive landscape"
- "Break down financial metrics"

**Routing Logic:**
1. LLM analyzes query intent
2. Keyword-based fallback
3. Default to ANALYSIS for deep insights

## 📡 API Endpoints

### 1. Specialized Agent — `/specialized-agent` ⭐ **NEW**

Intelligent routing to the best agent based on query.

**Request:**
```json
{
  "query": "What are emerging market trends?",
  "session_id": "optional-session",
  "top_k": 4
}
```

**Response:**
```json
{
  "answer": "Based on analysis...",
  "sources": [
    {"chunk_id": "...", "section": "..."}
  ],
  "agent_type": "TRENDS"
}
```

### 2. Q&A — `/qa`

Basic question answering with source citations.

**Request:**
```json
{
  "query": "What is Innovate Inc.'s market share?",
  "top_k": 4
}
```

### 3. Summary — `/summary`

Concise summaries with key takeaways.

**Request:**
```json
{
  "query": "Summarize competitive landscape",
  "top_k": 4
}
```

### 4. Extract — `/extract`

Structured data extraction as JSON.

**Request:**
```json
{
  "query": "Extract SWOT analysis",
  "top_k": 4,
  "schema_hint": "JSON schema..."
}
```

### 5. Chat with Memory — `/query`

Contextual conversations with session history.

**Request:**
```json
{
  "session_id": "user-123",
  "query": "Follow up: what are the next steps?",
  "top_k": 4
}
```

### 6. Ingest PDF — `/ingest`

Upload and process PDFs for analysis.

**Request:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@market.pdf" \
  -F "section=market_research" \
  -F "max_chars=2000" \
  -F "overlap_chars=200"
```

## 🎨 Streamlit UI

### Features
- **PDF Upload**: Ingest documents via side panel
- **Chat Interface**: Real-time conversations with document context
- **Source Citations**: View chunk sources for transparency
- **Active Document**: Track current source for chat
- **Session Persistence**: Maintains conversation history

### Usage

```bash
streamlit run streamlit_app.py
```

1. Upload a PDF document
2. Wait for "Ingested X chunks" success message
3. Start chatting about the document
4. View sources for each response

## 🧪 Testing

### Test ChromaDB Connection

```bash
python test_chromadb.py
```

Tests:
- ✅ ChromaDB connectivity
- ✅ Embedding generation
- ✅ Vector storage
- ✅ Similarity search

### Test PDF Ingestion

```bash
python run_ingest.py market.pdf
```

### Integration Test

```bash
# Start server
uvicorn app.main:app --reload &

# Ingest data
python run_ingest.py market.pdf

# Query specialized agent
curl -X POST "http://localhost:8000/specialized-agent" \
  -H "Content-Type: application/json" \
  -d '{"query":"What are key market insights?","top_k":4}'
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Azure OpenAI (gpt-4o-mini) |
| **Embeddings** | Azure OpenAI (text-embedding-3-small) |
| **Vector DB** | ChromaDB Cloud |
| **Agent Framework** | LangGraph |
| **Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Chat Memory** | MongoDB |
| **Async Runtime** | asyncio + httpx |

## 📦 Key Components

### Core Modules

- `app/agent.py` - Specialized agents and routing logic
- `app/llm.py` - Azure OpenAI integration
- `app/chroma_db.py` - ChromaDB Cloud client
- `app/vector_store.py` - Vector storage and search
- `app/memory.py` - Session history management
- `app/main.py` - FastAPI endpoints
- `app/streamlit_app.py` - Streamlit UI

### Key Features

- **Async Operations**: Non-blocking I/O throughout
- **Thread Pool Execution**: Wraps sync ChromaDB calls
- **Secrets Management**: Streamlit secrets + .env fallback
- **Type Safety**: Type hints throughout
- **Error Handling**: Graceful fallbacks and retries

## 🚢 Deployment

### Streamlit Cloud

1. Connect your GitHub repo
2. Add secrets in Streamlit Cloud settings:
   - Azure OpenAI credentials
   - ChromaDB credentials
3. Deploy automatically

### FastAPI Production

```bash
# Docker
docker build -t vaia-analyst .
docker run -p 8000:8000 --env-file .env vaia-analyst

# Or with gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 📚 Design Decisions

### Vector Database: ChromaDB

**Why ChromaDB?**
- Native vector search optimized for embeddings
- Cloud-hosted with automatic scaling
- Simple API without index management
- Efficient pricing for vector workloads

**Migration:** Previously used MongoDB → ChromaDB for better performance

### Agent Architecture

**Reasoning → Acting Pattern:**
1. Router analyzes query intent
2. Selects best specialized agent
3. Enhances query for better retrieval
4. Agent processes with domain expertise
5. Returns contextualized answer

### Embedding Model

**Azure OpenAI text-embedding-3-small**
- 1536 dimensions
- Fast generation
- High-quality retrieval
- Production-ready

### Chunking Strategy

- Size: 2000 characters
- Overlap: 200 characters
- Metadata: section, chunk_id, source_file

## 🔒 Security

- Secrets stored in `.env` or Streamlit secrets
- `.gitignore` excludes sensitive files
- No hardcoded credentials
- API keys loaded at runtime

## 📊 Performance

- **Embedding Latency**: ~100-200ms per batch
- **Query Response**: ~2-5 seconds end-to-end
- **Vector Search**: <50ms with ChromaDB
- **Concurrent Users**: Scales with async architecture

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

MIT License

## 🙏 Acknowledgments

- Azure OpenAI for LLM capabilities
- ChromaDB for vector database
- Streamlit for UI framework
- LangGraph for agent orchestration

## 📖 Additional Documentation

- `SPECIALIZED_AGENTS.md` - Detailed agent documentation
- `CHROMADB_MIGRATION.md` - Vector database migration guide
- `.streamlit/secrets-setup.md` - Secrets configuration guide

---

**Built with ❤️ for intelligent market analysis**
