"""
Standalone Streamlit RAG Chat Application
No dependencies on other app modules - all logic is self-contained.
"""
import asyncio
import json
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import threading
from concurrent.futures import Future

import streamlit as st
import numpy as np
import httpx
from pypdf import PdfReader
from pymongo import MongoClient, UpdateOne
def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors using NumPy.
    Returns 0.0 if either vector has zero norm.
    """
    a = vec_a.astype(np.float32, copy=False)
    b = vec_b.astype(np.float32, copy=False)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ============================================================================
# Hardcoded Configuration
# ============================================================================

# Azure OpenAI Configuration (loaded from Streamlit secrets in production)
AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = st.secrets.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_EMBED_DEPLOYMENT = st.secrets.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")

# MongoDB Configuration (loaded from Streamlit secrets in production)
MONGODB_URI = st.secrets.get("MONGODB_URI", "")
MONGO_DB_NAME = st.secrets.get("MONGO_DB_NAME", "vaia_market_analyst")
VECTORS_COLLECTION_PREFIX = st.secrets.get("VECTORS_COLLECTION_PREFIX", "vaia_vectors")

# UI Constants
APP_TITLE = "VAIA — Document Chat"
SIDEBAR_TITLE = "Ingest PDF"
DEFAULT_SECTION = "document"
DEFAULT_MAX_CHARS = 2000
DEFAULT_OVERLAP_CHARS = 200
TOP_K_DEFAULT = 4

# Agent Prompts
Q_A_PROMPT = "You are a helpful market research analyst. Answer questions directly and clearly using only provided source chunks when citing."

MARKET_RESEARCH_PROMPT = (
    "You are an expert market research analyst. Focus on providing comprehensive market research findings, "
    "including market size, growth trends, competitive analysis, SWOT insights, and strategic recommendations. "
    "Structure your response with clear sections and key takeaways. Use only provided source chunks when citing."
)

EXTRACTION_PROMPT = (
    "You are a data extraction specialist. Extract structured information from the provided chunks and return "
    "ONLY a valid JSON object. If information is missing, use null or empty arrays. Do not include any explanatory text, "
    "only the JSON object. Use only provided source chunks when citing."
)


# ============================================================================
# Global State Management
# ============================================================================

_ASYNC_LOOP: Optional[asyncio.AbstractEventLoop] = None
_ASYNC_THREAD: Optional[threading.Thread] = None
_mongo_client: Optional[MongoClient] = None
_mongo_db = None


def get_mongo_client() -> MongoClient:
    """Get or create MongoDB client (cached)."""
    global _mongo_client
    if _mongo_client is None:
        try:
            _mongo_client = MongoClient(MONGODB_URI)
            # Test connection
            _mongo_client.admin.command('ping')
        except Exception as e:
            _mongo_client = None
            # Do not log or expose any portion of secrets/URIs
            raise RuntimeError("Failed to connect to MongoDB. Please verify credentials in Streamlit secrets.") from e
    return _mongo_client


def get_mongo_db():
    """Get MongoDB database instance."""
    global _mongo_db
    if _mongo_db is None:
        client = get_mongo_client()
        _mongo_db = client[MONGO_DB_NAME]
    return _mongo_db


def _ensure_async_loop() -> asyncio.AbstractEventLoop:
    """Ensure async event loop is running in background thread."""
    global _ASYNC_LOOP, _ASYNC_THREAD
    if _ASYNC_LOOP and _ASYNC_LOOP.is_running():
        return _ASYNC_LOOP
    loop = asyncio.new_event_loop()
    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()
    thread = threading.Thread(target=_run, name="streamlit-async-loop", daemon=True)
    thread.start()
    _ASYNC_LOOP = loop
    _ASYNC_THREAD = thread
    return loop


def run_async(coro: Any) -> Any:
    """Run async coroutine in background thread."""
    loop = _ensure_async_loop()
    fut: Future = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()




# ============================================================================
# Azure OpenAI Functions
# ============================================================================

async def embed_texts(texts: List[str]) -> np.ndarray:
    """Generate embeddings for texts using Azure OpenAI."""
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_EMBED_DEPLOYMENT}/embeddings"
    params = {"api-version": AZURE_OPENAI_API_VERSION}
    headers = {"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"}
    payload = {"input": texts}

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, params=params, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        vectors = [item["embedding"] for item in data.get("data", [])]
        return np.array(vectors, dtype=np.float32)


async def chat_complete(system_prompt: str, messages: List[dict]) -> str:
    """Complete chat using Azure OpenAI."""
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
    params = {"api-version": AZURE_OPENAI_API_VERSION}
    headers = {"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"}
    payload = {
        "temperature": 0.0,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, params=params, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        return content


# ============================================================================
# MongoDB Vector Operations
# ============================================================================

def upsert_vectors(items: List[Dict[str, Any]], collection_name: str) -> None:
    """Upsert vectors to a specific MongoDB collection."""
    db = get_mongo_db()
    collection = db[collection_name]
    
    # Prepare documents for MongoDB
    operations = []
    for item in items:
        doc = {
            "_id": item["chunk_id"],
            "text": item["text"],
            "embedding": item["embedding"],  # Store as list
            "metadata": item.get("metadata", {"source": "uploaded"}),
        }
        operations.append(doc)
    
    try:
        # Use bulk_write for upsert
        bulk_ops = [
            UpdateOne(
                {"_id": doc["_id"]},
                {"$set": doc},
                upsert=True
            )
            for doc in operations
        ]
        if bulk_ops:
            collection.bulk_write(bulk_ops)
    except Exception as e:
        raise RuntimeError(f"Failed to upsert to MongoDB: {str(e)}") from e


def similarity_search(query_embedding: np.ndarray, top_k: int, collection_name: str, source_file: Optional[str] = None) -> List[Tuple[float, Dict[str, Any]]]:
    """Search for similar vectors in a specific MongoDB collection using cosine similarity."""
    db = get_mongo_db()
    collection = db[collection_name]
    
    try:
        # Build query filter
        query_filter = {}
        if source_file:
            query_filter["metadata.source_file"] = source_file
        
        # Fetch all documents from collection (or filter by source_file if specified)
        cursor = collection.find(query_filter)
        all_docs = list(cursor)
        
        if not all_docs:
            return []
        
        # Compute cosine similarity for each document
        query_vec = np.array(query_embedding, dtype=np.float32)
        similarities: List[Tuple[float, Dict[str, Any]]] = []
        
        for doc in all_docs:
            doc_embedding = np.array(doc.get("embedding", []), dtype=np.float32)
            
            if len(doc_embedding) == 0 or len(doc_embedding) != len(query_vec):
                continue
            
            # Compute cosine similarity
            try:
                similarity_score = _cosine_similarity(query_vec, doc_embedding)
            except Exception:
                continue
            
            result_doc = {
                "chunk_id": doc.get("_id", ""),
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {})
            }
            similarities.append((similarity_score, result_doc))
        
        # Sort by similarity (highest first) and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]
        
    except Exception as e:
        raise RuntimeError(f"Failed to search MongoDB: {str(e)}") from e


# ============================================================================
# PDF Processing
# ============================================================================

@dataclass
class IngestConfig:
    section: str
    max_chars: int
    overlap_chars: int


def _chunk_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """Chunk text into overlapping segments."""
    if max_chars <= 0:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def _extract_pdf_text(data: bytes) -> str:
    """Extract text from PDF bytes."""
    reader = PdfReader(BytesIO(data))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def ingest_pdf(file_name: str, data: bytes, cfg: IngestConfig) -> Tuple[int, str]:
    """Ingest PDF: extract, chunk, embed, and store in MongoDB. Returns (chunk_count, collection_name)."""
    # Extract text from PDF
    content = _extract_pdf_text(data)
    if not content or len(content.strip()) == 0:
        raise ValueError("PDF contains no extractable text")
    
    # Chunk the text
    parts = _chunk_text(content, cfg.max_chars, cfg.overlap_chars)
    if len(parts) == 0:
        raise ValueError("No text chunks created from PDF")
    
    # Generate embeddings for all chunks
    vectors = run_async(embed_texts(parts))
    
    # Verify embeddings were created
    if len(vectors) != len(parts):
        raise ValueError(f"Embedding count mismatch: {len(vectors)} embeddings for {len(parts)} chunks")
    
    # Prepare data for ChromaDB
    to_upsert: List[Dict[str, Any]] = []
    for idx, (text, vec) in enumerate(zip(parts, vectors)):
        # Verify embedding vector is valid
        if vec is None or len(vec) == 0:
            raise ValueError(f"Invalid embedding for chunk {idx}")
        
        to_upsert.append({
            "chunk_id": f"{cfg.section}_chunk_{idx}_{uuid.uuid4().hex[:8]}",
            "text": text,
            "embedding": vec.tolist(),
            "metadata": {"section": cfg.section, "source_file": file_name, "index": idx},
        })

    # Create unique collection name for this document
    # Use hash of filename + timestamp to ensure uniqueness
    doc_hash = hashlib.md5(f"{file_name}_{uuid.uuid4()}".encode()).hexdigest()[:12]
    collection_name = f"{VECTORS_COLLECTION_PREFIX}_{doc_hash}"
    
    # Upsert embeddings to MongoDB in the document-specific collection
    if len(to_upsert) == 0:
        raise ValueError("No data to upsert")
    
    upsert_vectors(to_upsert, collection_name)
    
    return len(parts), collection_name


# ============================================================================
# RAG Chat Functions - Three Different Agents
# ============================================================================

def agent_qa(query: str, top_k: int, collection_name: str, source_file: Optional[str] = None, history: List[Dict[str, str]] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """General Q&A Agent: Direct question answering."""
    if history is None:
        history = []
    
    q_vec = run_async(embed_texts([query]))
    retrieved = similarity_search(q_vec[0], top_k=top_k, collection_name=collection_name, source_file=source_file)
    docs = [doc for _, doc in retrieved]
    
    chunk_texts = []
    sources = []
    for doc in docs:
        text = doc.get("text", "")
        chunk_id = doc.get("chunk_id", "")
        section = doc.get("metadata", {}).get("section", "")
        chunk_texts.append(f"[chunk_id={chunk_id}, section={section}]\n{text}")
        sources.append({"chunk_id": chunk_id, "section": section})
    
    messages = history + [
        {"role": "user", "content": f"CHUNKS:\n" + "\n\n".join(chunk_texts)},
        {"role": "user", "content": query},
    ]
    
    answer = run_async(chat_complete(Q_A_PROMPT, messages))
    return answer, sources


def agent_market_research(query: str, top_k: int, collection_name: str, source_file: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """Market Research Findings Agent: Comprehensive market analysis and findings."""
    q_vec = run_async(embed_texts([query]))
    retrieved = similarity_search(q_vec[0], top_k=top_k, collection_name=collection_name, source_file=source_file)
    docs = [doc for _, doc in retrieved]
    
    chunk_texts = []
    sources = []
    for doc in docs:
        text = doc.get("text", "")
        chunk_id = doc.get("chunk_id", "")
        section = doc.get("metadata", {}).get("section", "")
        chunk_texts.append(f"[chunk_id={chunk_id}, section={section}]\n{text}")
        sources.append({"chunk_id": chunk_id, "section": section})
    
    messages = [
        {"role": "user", "content": f"CHUNKS:\n" + "\n\n".join(chunk_texts)},
        {"role": "user", "content": f"Provide comprehensive market research findings for: {query}"},
    ]
    
    answer = run_async(chat_complete(MARKET_RESEARCH_PROMPT, messages))
    return answer, sources


def agent_extract(query: str, top_k: int, collection_name: str, source_file: Optional[str] = None, schema_hint: str = "") -> Tuple[str, List[Dict[str, Any]]]:
    """Structured Data Extraction Agent: Extract data in JSON format."""
    q_vec = run_async(embed_texts([query]))
    retrieved = similarity_search(q_vec[0], top_k=top_k, collection_name=collection_name, source_file=source_file)
    docs = [doc for _, doc in retrieved]
    
    chunk_texts = []
    sources = []
    for doc in docs:
        text = doc.get("text", "")
        chunk_id = doc.get("chunk_id", "")
        section = doc.get("metadata", {}).get("section", "")
        chunk_texts.append(f"[chunk_id={chunk_id}, section={section}]\n{text}")
        sources.append({"chunk_id": chunk_id, "section": section})
    
    extraction_query = f"Extract the following information as JSON: {query}"
    if schema_hint:
        extraction_query = f"{schema_hint}\n\n{extraction_query}"
    
    messages = [
        {"role": "user", "content": f"CHUNKS:\n" + "\n\n".join(chunk_texts)},
        {"role": "user", "content": extraction_query},
    ]
    
    answer = run_async(chat_complete(EXTRACTION_PROMPT, messages))
    return answer, sources


# ============================================================================
# Streamlit UI
# ============================================================================

def _init_session_state() -> None:
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "active_document" not in st.session_state:
        st.session_state.active_document = None


def sidebar_ingest() -> None:
    """Sidebar for PDF ingestion."""
    st.sidebar.title(SIDEBAR_TITLE)
    
    uploaded = st.sidebar.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
    
    cfg = IngestConfig(
        section=DEFAULT_SECTION,
        max_chars=DEFAULT_MAX_CHARS,
        overlap_chars=DEFAULT_OVERLAP_CHARS
    )

    if st.sidebar.button("Ingest PDF", type="primary", use_container_width=True, disabled=uploaded is None):
        if uploaded is None:
            st.sidebar.warning("Please upload a PDF first.")
            return
        try:
            # Show progress in sidebar
            status_placeholder = st.sidebar.empty()
            status_placeholder.info("🔄 Connecting to MongoDB...")
            
            # Test connection first
            try:
                get_mongo_client()
                status_placeholder.info("🔄 Extracting text from PDF...")
            except Exception as conn_error:
                status_placeholder.empty()
                st.sidebar.error(f"❌ MongoDB connection failed: {conn_error}")
                return
            
            pdf_data = uploaded.read()
            status_placeholder.info("🔄 Processing document...")
            
            # Ingest PDF (extract, chunk, embed, and store)
            chunks, collection_name = ingest_pdf(uploaded.name, pdf_data, cfg)
            status_placeholder.empty()
            st.session_state.active_document = uploaded.name
            st.session_state.active_collection = collection_name
            st.session_state.chat_history = []
            st.sidebar.success(f"✅ Successfully ingested and embedded {chunks} chunks! Ready to chat!")
        except Exception as exc:
            status_placeholder.empty()
            st.sidebar.error(f"❌ Failed to ingest: {exc}")


def main() -> None:
    """Main Streamlit application."""
    _init_session_state()
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide", initial_sidebar_state="expanded")
    
    # Force dark mode
    st.markdown("""
    <style>
        .stApp { background-color: #0E1117; color: #E5E7EB; }
        .main .block-container { background-color: #0E1117; }
        [data-testid="stHeader"] { background: transparent; }
        h1, h2, h3, h4, h5, h6 { color: #FAFAFA; }
        .stRadio label, .stCheckbox label, .stTextArea label { color: #F3F4F6; }
        /* Disabled controls readability */
        .stRadio [aria-disabled="true"], .stCheckbox [aria-disabled="true"] { color: #9CA3AF !important; opacity: 0.8; }

        /* Improve alert contrast */
        div[role="alert"] { background-color: #111827 !important; color: #E5E7EB !important; border: 1px solid #374151; }
        div[role="alert"] p, div[role="alert"] span { color: #E5E7EB !important; }
        /* Left accent bar */
        div[role="alert"] { border-left: 4px solid #22D3EE; }

        /* Sidebar colors */
        [data-testid="stSidebar"] { background-color: #0E1117; }
        [data-testid="stSidebar"] * { color: #E5E7EB; }
        /* Uploader dropzone */
        [data-testid="stFileUploaderDropzone"] { background: #0B1220; border: 1px dashed #475569; color: #E5E7EB; }
        [data-testid="stFileUploaderDropzone"] * { color: #E5E7EB; }

        /* Inputs on dark background */
        input, textarea, select { background-color: #111827 !important; color: #F3F4F6 !important; border-color: #374151 !important; }
        ::placeholder { color: #9CA3AF !important; }
        .stButton>button { background: #1F2937; color: #F9FAFB; border: 1px solid #374151; }
        .stButton>button:hover { background: #374151; border-color: #4B5563; }

        /* Markdown text contrast */
        [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: #E5E7EB; }
    </style>
    """, unsafe_allow_html=True)

    # CSS for chat input visibility
    st.markdown("""
    <style>
        input[data-testid="stChatInputTextInput"],
        textarea[data-testid="stChatInputTextInput"],
        .stChatInput input,
        .stChatInput textarea {
            color: rgba(250, 250, 250, 1) !important;
            background-color: #0B1220 !important;
            border-color: #334155 !important;
        }
        [data-testid="stAppViewContainer"] {
            --text-color: rgba(49, 51, 63, 1);
        }
        .stApp[data-theme="dark"] input[data-testid="stChatInputTextInput"],
        .stApp[data-theme="dark"] textarea[data-testid="stChatInputTextInput"] {
            color: rgba(250, 250, 250, 1) !important;
            background-color: #0B1220 !important;
        }
        .stApp[data-theme="light"] input[data-testid="stChatInputTextInput"],
        .stApp[data-theme="light"] textarea[data-testid="stChatInputTextInput"] {
            color: rgba(49, 51, 63, 1) !important;
        }
        .stChatInput { background-color: #0F172A !important; border: 1px solid #334155; border-radius: 8px; }
        .stChatInput button { background: #2563EB !important; color: #F9FAFB !important; border: 0 !important; }
        .stChatInput button:hover { background: #1D4ED8 !important; }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    sidebar_ingest()

    # Main content
    st.title(APP_TITLE)
    
    active_doc = st.session_state.get("active_document")
    if active_doc:
        st.info(f"📄 Active document: **{active_doc}**")
    else:
        st.warning("⚠️ Please ingest a PDF to start chatting.")

    # Agent selection
    agent_mode = st.radio(
        "Select Agent Mode:",
        ["General Q&A", "Market Research Findings", "Structured Data Extraction"],
        horizontal=True,
        disabled=not active_doc
    )
    
    show_sources = st.checkbox("Show sources", value=True)
    
    # Schema hint for extraction mode
    schema_hint = ""
    if agent_mode == "Structured Data Extraction":
        schema_hint = st.text_area(
            "Schema Hint (optional JSON structure):",
            placeholder='e.g., {"company": "", "market_size": 0, "competitors": []}',
            disabled=not active_doc,
            height=100
        )

    # Chat history - display all messages
    if st.session_state.chat_history:
        for turn in st.session_state.chat_history:
            if turn.get("role") == "user":
                st.chat_message("user").write(turn.get("content", ""))
            else:
                content = turn.get("content", "")
                if content:
                    # Display answer - format JSON if it's marked as extraction
                    if turn.get("is_extraction", False):
                        try:
                            answer_json = json.loads(content)
                            st.chat_message("assistant").json(answer_json)
                        except (json.JSONDecodeError, ValueError):
                            st.chat_message("assistant").write(content)
                    else:
                        st.chat_message("assistant").write(content)
                    
                    if show_sources and turn.get("sources"):
                        with st.expander("📎 Sources"):
                            st.json(turn.get("sources", []))

    # Chat input
    placeholder = "Ask about your documents…" if active_doc else "Please ingest a PDF first."
    user_msg = st.chat_input(placeholder, disabled=not active_doc)
    
    if user_msg and active_doc:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        st.chat_message("user").write(user_msg)
        
        # Get chat history for context
        history = []
        for turn in st.session_state.chat_history[:-1]:  # Exclude the just-added user message
            if turn.get("role") == "user":
                history.append({"role": "user", "content": turn.get("content", "")})
            elif turn.get("role") == "assistant":
                history.append({"role": "assistant", "content": turn.get("content", "")})
        
        # Get collection name for active document
        collection_name = st.session_state.get("active_collection")
        if not collection_name:
            st.warning("No collection found for active document. Please re-ingest the document.")
            return
        
        # Generate response based on selected agent
        with st.spinner("Thinking…"):
            try:
                # Debug: Check if collection exists and has data
                db = get_mongo_db()
                collection = db[collection_name]
                doc_count = collection.count_documents({})
                
                if doc_count == 0:
                    st.warning(f"⚠️ Collection '{collection_name}' is empty. Please re-ingest the document.")
                    return
                
                # Generate answer
                if agent_mode == "General Q&A":
                    answer, sources = agent_qa(user_msg, top_k=TOP_K_DEFAULT, collection_name=collection_name, source_file=active_doc, history=history)
                elif agent_mode == "Market Research Findings":
                    answer, sources = agent_market_research(user_msg, top_k=TOP_K_DEFAULT, collection_name=collection_name, source_file=active_doc)
                elif agent_mode == "Structured Data Extraction":
                    answer, sources = agent_extract(user_msg, top_k=TOP_K_DEFAULT, collection_name=collection_name, source_file=active_doc, schema_hint=schema_hint)
                else:
                    answer, sources = agent_qa(user_msg, top_k=TOP_K_DEFAULT, collection_name=collection_name, source_file=active_doc, history=history)
                
                # Ensure answer is not empty
                if not answer or len(answer.strip()) == 0:
                    answer = "Sorry, I couldn't generate a response. Please try again or rephrase your question."
                    st.warning("⚠️ Empty response generated. This might indicate an issue with the LLM or retrieved chunks.")
                
                # Add assistant response to history
                is_extraction = agent_mode == "Structured Data Extraction"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "is_extraction": is_extraction
                })
                
                # Force rerun to display the answer
                st.rerun()
                        
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "error": True
                })
                import traceback
                st.exception(e)
                st.rerun()
    
    elif user_msg and not active_doc:
        st.warning("Please ingest a PDF document first before asking questions.")


if __name__ == "__main__":
    main()
