from typing import Dict, Any, List, Optional

import numpy as np
from langgraph.graph import StateGraph, END

from .llm import chat_complete, embed_texts
from .vector_store import similarity_search


SYSTEM_PROMPT = (
    "You are a helpful market research analyst. Use only provided source chunks when citing."
)


class AgentState(dict):
    """Lightweight state container for LangGraph."""


async def _retrieve_node(state: AgentState) -> AgentState:
    query: str = state["query"]
    top_k: int = state["top_k"]
    source_file: str = state.get("source_file")

    q_vec = await embed_texts([query])
    retrieved = await similarity_search(query_embedding=q_vec[0], top_k=top_k, source_file=source_file)

    state["retrieved_docs"] = [doc for _, doc in retrieved]
    return state


async def _reason_node(state: AgentState) -> AgentState:
    history: List[Dict[str, str]] = state.get("history", [])
    query: str = state["query"]

    chunks = state.get("retrieved_docs", [])
    chunk_texts = []
    sources = []
    for doc in chunks:
        text = doc.get("text", "")
        chunk_id = doc.get("chunk_id", "")
        section = doc.get("metadata", {}).get("section", "")
        chunk_texts.append(f"[chunk_id={chunk_id}, section={section}]\n{text}")
        sources.append({"chunk_id": chunk_id, "section": section})

    messages = history + [
        {"role": "user", "content": f"CHUNKS:\n" + "\n\n".join(chunk_texts)},
        {"role": "user", "content": query},
    ]

    answer = await chat_complete(system_prompt=SYSTEM_PROMPT, messages=messages)

    state["answer"] = answer
    state["sources"] = sources
    return state


async def run_react_agent(
    user_query: str,
    session_id: str,
    history: List[Dict[str, str]],
    top_k: int,
    source_file: Optional[str] = None,
) -> Dict[str, Any]:
    # Execute retrieve -> reason sequentially to avoid graph runtime issues
    state: AgentState = AgentState({
        "query": user_query,
        "history": history,
        "top_k": top_k,
        "source_file": source_file,
    })
    state = await _retrieve_node(state)
    state = await _reason_node(state)
    return {"answer": state.get("answer", ""), "sources": state.get("sources", [])}


# --- Simpler mode-specific helpers ---

async def run_qa(query: str, top_k: int, source_file: Optional[str] = None) -> Dict[str, Any]:
    q_vec = await embed_texts([query])
    retrieved = await similarity_search(q_vec[0], top_k, source_file=source_file)
    docs = [doc for _, doc in retrieved]
    chunks = []
    sources: List[Dict[str, Any]] = []
    for d in docs:
        text = d.get("text", "")
        chunk_id = d.get("chunk_id", "")
        section = d.get("metadata", {}).get("section", "")
        chunks.append(f"[chunk_id={chunk_id}, section={section}]\n{text}")
        sources.append({"chunk_id": chunk_id, "section": section})
    messages = [
        {"role": "user", "content": "CHUNKS:\n" + "\n\n".join(chunks)},
        {"role": "user", "content": query},
    ]
    answer = await chat_complete(SYSTEM_PROMPT, messages)
    return {"answer": answer, "sources": sources}


async def run_summary(query: str, top_k: int, max_length: int = 300, source_file: Optional[str] = None) -> Dict[str, Any]:
    instruction = (
        f"Summarize succinctly in under {max_length} tokens. Use bullets and cite chunk ids where relevant."
    )
    return await run_qa(f"{instruction}\n{query}", top_k, source_file=source_file)


async def run_extract(query: str, top_k: int, schema_hint: str = "", source_file: Optional[str] = None) -> Dict[str, Any]:
    prefix = (
        "Return ONLY a single JSON object. If info missing, use nulls or empty arrays."
    )
    effective_query = f"{prefix}\n{schema_hint}\n{query}"
    return await run_qa(effective_query, top_k, source_file=source_file)


async def route_task(query: str) -> str:
    lower_q = query.lower()
    if any(k in lower_q for k in ["extract", "json", "schema", "swot", "market size", "competitors"]):
        return "EXTRACT"
    if any(k in lower_q for k in ["summarize", "summary", "findings", "key takeaways"]):
        return "SUMMARY"
    return "QA"


