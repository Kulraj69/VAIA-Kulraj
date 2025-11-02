from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

from .db import get_db
from .memory import fetch_session_history, append_session_turn
from .agent import run_react_agent, run_qa, run_summary, run_extract, route_task, run_specialized_agent
from .vector_store import upsert_vectors
from .llm import embed_texts


class QueryRequest(BaseModel):
    session_id: Optional[str] = None
    query: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    memory: List[Dict[str, str]]


app = FastAPI(title="AI Market Analyst — VAIA", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    # Initialize MongoDB connection
    await get_db()


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest) -> QueryResponse:
    if not body.query:
        raise HTTPException(status_code=400, detail="query is required")

    # Load prior memory if session_id provided; else run stateless
    history: List[Dict[str, str]] = []
    if body.session_id:
        history = await fetch_session_history(session_id=body.session_id)

    # Run ReAct agent (LangGraph) to decide and answer
    try:
        agent_result = await run_react_agent(
            user_query=body.query,
            session_id=body.session_id or "",
            history=history,
            top_k=body.top_k,
        )
    except Exception as exc:  # Guard for unexpected failures
        raise HTTPException(status_code=500, detail=f"Agent failure: {exc}")

    answer: str = agent_result.get("answer", "")
    sources: List[Dict[str, Any]] = agent_result.get("sources", [])

    if not answer:
        raise HTTPException(status_code=500, detail="Agent returned empty answer")

    # Persist new turn only if session_id provided
    if body.session_id:
        await append_session_turn(
            session_id=body.session_id,
            user_content=body.query,
            assistant_content=answer,
            sources=sources,
        )
        updated_history = await fetch_session_history(session_id=body.session_id)
    else:
        updated_history = []

    return QueryResponse(answer=answer, sources=sources, memory=updated_history)


class IngestResponse(BaseModel):
    section: str
    chunks: int


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


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    file: UploadFile = File(...),
    section: str = Form(...),
    max_chars: int = Form(2000),
    overlap_chars: int = Form(200),
) -> IngestResponse:
    from io import BytesIO
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    data = await file.read()
    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(data))
        pages: List[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        content = "\n".join(pages)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {exc}")

    parts = _chunk_text(content, max_chars, overlap_chars)
    vectors = await embed_texts(parts)

    to_upsert: List[Dict[str, Any]] = []
    import uuid
    for idx, (text, vec) in enumerate(zip(parts, vectors)):
        to_upsert.append(
            {
                "chunk_id": f"{section}_chunk_{idx}_{uuid.uuid4().hex[:8]}",
                "text": text,
                "embedding": vec.tolist(),
                "metadata": {"section": section, "source_file": file.filename, "index": idx},
            }
        )

    await upsert_vectors(to_upsert)
    return IngestResponse(section=section, chunks=len(parts))


class BasicRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


class BasicResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


class SpecializedResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    agent_type: str


@app.post("/qa", response_model=BasicResponse)
async def qa_endpoint(body: BasicRequest) -> BasicResponse:
    result = await run_qa(body.query, body.top_k)
    if not result.get("answer"):
        raise HTTPException(status_code=500, detail="Empty answer")
    return BasicResponse(answer=result["answer"], sources=result.get("sources", []))


@app.post("/summary", response_model=BasicResponse)
async def summary_endpoint(body: BasicRequest) -> BasicResponse:
    result = await run_summary(body.query, body.top_k)
    if not result.get("answer"):
        raise HTTPException(status_code=500, detail="Empty answer")
    return BasicResponse(answer=result["answer"], sources=result.get("sources", []))


class ExtractRequest(BasicRequest):
    schema_hint: Optional[str] = None


@app.post("/extract", response_model=BasicResponse)
async def extract_endpoint(body: ExtractRequest) -> BasicResponse:
    result = await run_extract(body.query, body.top_k, schema_hint=body.schema_hint or "")
    if not result.get("answer"):
        raise HTTPException(status_code=500, detail="Empty answer")
    return BasicResponse(answer=result["answer"], sources=result.get("sources", []))


@app.post("/route", response_model=BasicResponse)
async def route_endpoint(body: BasicRequest) -> BasicResponse:
    decision = await route_task(body.query)
    if decision == "EXTRACT":
        result = await run_extract(body.query, body.top_k)
    elif decision == "SUMMARY":
        result = await run_summary(body.query, body.top_k)
    else:
        result = await run_qa(body.query, body.top_k)
    if not result.get("answer"):
        raise HTTPException(status_code=500, detail="Empty answer")
    return BasicResponse(answer=result["answer"], sources=result.get("sources", []))


class SpecializedQueryRequest(BaseModel):
    session_id: Optional[str] = None
    query: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


@app.post("/specialized-agent", response_model=SpecializedResponse)
async def specialized_agent_endpoint(body: SpecializedQueryRequest) -> SpecializedResponse:
    """
    New specialized agent endpoint with intelligent routing.
    
    Automatically selects the best agent (TRENDS, STRATEGY, or ANALYSIS) based on query intent.
    """
    if not body.query:
        raise HTTPException(status_code=400, detail="query is required")

    # Load prior memory if session_id provided
    history: List[Dict[str, str]] = []
    if body.session_id:
        history = await fetch_session_history(session_id=body.session_id)

    try:
        result = await run_specialized_agent(
            user_query=body.query,
            session_id=body.session_id or "",
            history=history,
            top_k=body.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent failure: {exc}")

    answer: str = result.get("answer", "")
    sources: List[Dict[str, Any]] = result.get("sources", [])
    agent_type: str = result.get("agent_type", "ANALYSIS")

    if not answer:
        raise HTTPException(status_code=500, detail="Agent returned empty answer")

    # Persist turn if session_id provided
    if body.session_id:
        await append_session_turn(
            session_id=body.session_id,
            user_content=body.query,
            assistant_content=answer,
            sources=sources,
        )

    return SpecializedResponse(
        answer=answer,
        sources=sources,
        agent_type=agent_type
    )


