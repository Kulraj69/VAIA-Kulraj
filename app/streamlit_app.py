import asyncio
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading
from concurrent.futures import Future

import streamlit as st

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import app modules - will be initialized properly in main()
from app.agent import run_react_agent, run_qa, run_summary, run_extract
from app.db import get_db
from app.llm import embed_texts
from app.memory import fetch_session_history, append_session_turn
from app.vector_store import upsert_vectors


# ---- UI constants (avoid hard-coded literals in logic) ----
APP_TITLE: str = "VAIA — Document Chat"
SIDEBAR_TITLE: str = "Ingest PDF"
DEFAULT_SECTION: str = "document"
DEFAULT_MAX_CHARS: int = 2000
DEFAULT_OVERLAP_CHARS: int = 200
TOP_K_MIN: int = 1
TOP_K_MAX: int = 20
TOP_K_DEFAULT: int = 4
CHAT_MODE: str = "Chat with Docs"
QA_MODE: str = "Basic Q&A"
SUMMARY_MODE: str = "Summarize"
EXTRACT_MODE: str = "Extract"
SHOW_SOURCES_LABEL: str = "Show sources"


_ASYNC_LOOP: Optional[asyncio.AbstractEventLoop] = None
_ASYNC_THREAD: Optional[threading.Thread] = None


def load_secrets_to_env():
    """Load secrets from Streamlit secrets.toml into os.environ."""
    try:
        # Azure OpenAI secrets
        if "azure_openai" in st.secrets:
            if "endpoint" in st.secrets["azure_openai"]:
                os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["azure_openai"]["endpoint"]
            if "deployment" in st.secrets["azure_openai"]:
                os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"] = st.secrets["azure_openai"]["deployment"]
                # Use separate embed_deployment if provided, otherwise use same as chat deployment
                if "embed_deployment" in st.secrets["azure_openai"]:
                    os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"] = st.secrets["azure_openai"]["embed_deployment"]
                else:
                    os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"] = st.secrets["azure_openai"]["deployment"]
            if "api_key" in st.secrets["azure_openai"]:
                os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["azure_openai"]["api_key"]
            if "api_version" in st.secrets["azure_openai"]:
                os.environ["AZURE_OPENAI_API_VERSION"] = st.secrets["azure_openai"]["api_version"]
        
        # MongoDB secrets (if present)
        if "mongodb" in st.secrets:
            if "uri" in st.secrets["mongodb"]:
                os.environ["MONGODB_URI"] = st.secrets["mongodb"]["uri"]
            if "database" in st.secrets["mongodb"]:
                os.environ["MONGO_DB_NAME"] = st.secrets["mongodb"]["database"]
        
        # ChromaDB secrets
        if "chromadb" in st.secrets:
            if "api_key" in st.secrets["chromadb"]:
                os.environ["CHROMA_API_KEY"] = st.secrets["chromadb"]["api_key"]
            if "tenant" in st.secrets["chromadb"]:
                os.environ["CHROMA_TENANT"] = st.secrets["chromadb"]["tenant"]
            if "database" in st.secrets["chromadb"]:
                os.environ["CHROMA_DATABASE"] = st.secrets["chromadb"]["database"]
    except AttributeError:
        # Secrets not available, try loading from .env as fallback
        from dotenv import load_dotenv
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=str(env_path), override=True)


def _ensure_async_loop() -> asyncio.AbstractEventLoop:
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
    loop = _ensure_async_loop()
    # Reset cached database connection before each call to ensure it uses the correct event loop
    # Motor client connections must be created in the same event loop they're used in
    import app.db as db_module
    # Store current loop ID to detect changes
    current_loop_id = id(loop)
    if "last_loop_id" not in st.session_state or st.session_state.get("last_loop_id") != current_loop_id:
        # Loop changed or first run, reset connection
        db_module._mongo_client = None
        db_module._mongo_db = None
        st.session_state.last_loop_id = current_loop_id
    fut: Future = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()


@dataclass
class IngestConfig:
    section: str
    max_chars: int
    overlap_chars: int


def _init_session_state() -> None:
    if "session_id" not in st.session_state:
        # Streamlit reruns; use a stable session id string
        st.session_state.session_id = "st_session"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # List[Dict[str, str]]
    if "active_document" not in st.session_state:
        st.session_state.active_document = None  # Track the currently active document for chat
    if "db_initialized" not in st.session_state:
        # Initialize database connection on first run using our async loop
        run_async(get_db())
        st.session_state.db_initialized = True


def _chunk_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
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
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def ingest_pdf(file_name: str, data: bytes, cfg: IngestConfig) -> int:
    content: str = _extract_pdf_text(data)
    parts: List[str] = _chunk_text(content, cfg.max_chars, cfg.overlap_chars)
    vectors = run_async(embed_texts(parts))

    to_upsert: List[Dict[str, Any]] = []
    import uuid

    for idx, (text, vec) in enumerate(zip(parts, vectors)):
        to_upsert.append(
            {
                "chunk_id": f"{cfg.section}_chunk_{idx}_{uuid.uuid4().hex[:8]}",
                "text": text,
                "embedding": vec.tolist(),
                "metadata": {"section": cfg.section, "source_file": file_name, "index": idx},
            }
        )

    run_async(upsert_vectors(to_upsert))
    return len(parts)


def run_mode_basic(mode: str, query: str, top_k: int, schema_hint: str = "") -> Tuple[str, List[Dict[str, Any]]]:
    _verify_env_vars()
    source_file: str = st.session_state.get("active_document")
    if mode == QA_MODE:
        result = run_async(run_qa(query, top_k, source_file=source_file))
    elif mode == SUMMARY_MODE:
        result = run_async(run_summary(query, top_k, source_file=source_file))
    elif mode == EXTRACT_MODE:
        result = run_async(run_extract(query, top_k, schema_hint=schema_hint, source_file=source_file))
    else:
        raise ValueError("Unsupported mode")
    return result.get("answer", ""), result.get("sources", [])


def run_chat(query: str, top_k: int) -> Tuple[str, List[Dict[str, Any]]]:
    _verify_env_vars()
    session_id: str = st.session_state.session_id
    source_file: str = st.session_state.get("active_document")
    history: List[Dict[str, str]] = run_async(fetch_session_history(session_id=session_id))
    result = run_async(
        run_react_agent(
            user_query=query,
            session_id=session_id,
            history=history,
            top_k=top_k,
            source_file=source_file,
        )
    )
    answer: str = result.get("answer", "")
    sources: List[Dict[str, Any]] = result.get("sources", [])
    run_async(
        append_session_turn(
            session_id=session_id,
            user_content=query,
            assistant_content=answer,
            sources=sources,
        )
    )
    # Add both messages to UI history after processing
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    return answer, sources


def _verify_env_vars() -> None:
    """Verify required environment variables are loaded."""
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "AZURE_OPENAI_EMBED_DEPLOYMENT",
    ]
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        # Try reloading secrets
        load_secrets_to_env()
        # Check again after reload
        missing = [var for var in required_vars if not os.environ.get(var)]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}. Please check your Streamlit secrets configuration.")


def sidebar_ingest() -> None:
    st.sidebar.title(SIDEBAR_TITLE)
    
    # Secure runtime credentials override (avoids hardcoding in code)
    with st.sidebar.expander("🔑 Credentials Override (dev)", expanded=False):
        az_endpoint = st.text_input("AZURE_OPENAI_ENDPOINT", value="")
        az_api_key = st.text_input("AZURE_OPENAI_API_KEY", value="", type="password")
        az_api_version = st.text_input("AZURE_OPENAI_API_VERSION", value="")
        az_chat = st.text_input("AZURE_OPENAI_CHAT_DEPLOYMENT", value="")
        az_embed = st.text_input("AZURE_OPENAI_EMBED_DEPLOYMENT", value="")
        mongo_uri = st.text_input("MONGODB_URI", value="")
        mongo_db = st.text_input("MONGO_DB_NAME", value="")

        if st.button("Apply overrides", use_container_width=True):
            if az_endpoint:
                os.environ["AZURE_OPENAI_ENDPOINT"] = az_endpoint
            if az_api_key:
                os.environ["AZURE_OPENAI_API_KEY"] = az_api_key
            if az_api_version:
                os.environ["AZURE_OPENAI_API_VERSION"] = az_api_version
            if az_chat:
                os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"] = az_chat
            if az_embed:
                os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"] = az_embed
            if mongo_uri:
                os.environ["MONGODB_URI"] = mongo_uri
            if mongo_db:
                os.environ["MONGO_DB_NAME"] = mongo_db
            st.success("Overrides applied for this session.")

    uploaded = st.sidebar.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)

    # Use default values for section and chunking parameters
    section: str = DEFAULT_SECTION
    max_chars: int = DEFAULT_MAX_CHARS
    overlap_chars: int = DEFAULT_OVERLAP_CHARS

    cfg = IngestConfig(section=section, max_chars=max_chars, overlap_chars=overlap_chars)

    if st.sidebar.button("Ingest PDF", type="primary", use_container_width=True, disabled=uploaded is None):
        if uploaded is None:
            st.sidebar.warning("Please upload a PDF first.")
            return
        try:
            _verify_env_vars()
            chunks = ingest_pdf(uploaded.name, uploaded.read(), cfg)
            # Set the newly ingested document as the active document
            st.session_state.active_document = uploaded.name
            # Reset chat history when a new document is ingested
            st.session_state.chat_history = []
            st.sidebar.success(f"Ingested {chunks} chunks into section '{cfg.section}'. Chat now uses '{uploaded.name}' as source.")
        except Exception as exc:
            st.sidebar.error(f"Failed to ingest: {exc}")


def main() -> None:
    # Load secrets first before initializing anything that needs env vars
    load_secrets_to_env()
    
    _init_session_state()
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")

    # Add custom CSS to ensure chat input text is visible
    st.markdown("""
    <style>
        /* Ensure chat input text is visible in all themes */
        input[data-testid="stChatInputTextInput"],
        textarea[data-testid="stChatInputTextInput"],
        .stChatInput input,
        .stChatInput textarea {
            color: rgba(250, 250, 250, 1) !important;
        }
        /* Light mode override */
        [data-testid="stAppViewContainer"] {
            --text-color: rgba(49, 51, 63, 1);
        }
        /* Additional selectors for Streamlit chat input */
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {
            color: inherit !important;
        }
        /* Dark theme specific */
        .stApp[data-theme="dark"] input[data-testid="stChatInputTextInput"],
        .stApp[data-theme="dark"] textarea[data-testid="stChatInputTextInput"],
        .stApp[data-theme="dark"] .stChatInput input,
        .stApp[data-theme="dark"] .stChatInput textarea {
            color: rgba(250, 250, 250, 1) !important;
        }
        /* Light theme specific */
        .stApp[data-theme="light"] input[data-testid="stChatInputTextInput"],
        .stApp[data-theme="light"] textarea[data-testid="stChatInputTextInput"],
        .stApp[data-theme="light"] .stChatInput input,
        .stApp[data-theme="light"] .stChatInput textarea {
            color: rgba(49, 51, 63, 1) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    sidebar_ingest()

    st.title(APP_TITLE)
    
    # Display active document if set
    active_doc = st.session_state.get("active_document")
    if active_doc:
        st.info(f"📄 Active document: **{active_doc}** (chatting from this source)")
    else:
        st.warning("⚠️ No document active. Please ingest a PDF to start chatting.")

    # Fixed mode: Chat with Docs; fixed Top K
    mode = CHAT_MODE
    top_k = TOP_K_DEFAULT
    show_sources = st.checkbox(SHOW_SOURCES_LABEL, value=True)

    with st.container(border=True):
        for turn in st.session_state.chat_history:
            if turn.get("role") == "user":
                st.chat_message("user").write(turn.get("content", ""))
            else:
                st.chat_message("assistant").write(turn.get("content", ""))

    # Disable chat input if no active document
    placeholder_text = "Ask about your documents…" if active_doc else "Please ingest a PDF first to start chatting."
    user_msg = st.chat_input(placeholder_text, disabled=not active_doc)
    if user_msg and active_doc:
        with st.spinner("Thinking…"):
            answer, sources = run_chat(user_msg, top_k)
        
        # Display both messages after processing
        st.chat_message("user").write(user_msg)
        st.chat_message("assistant").write(answer)
        
        if show_sources and sources:
            with st.expander("Sources"):
                st.json(sources)
    elif user_msg and not active_doc:
        st.warning("Please ingest a PDF document first before asking questions.")


if __name__ == "__main__":
    main()


