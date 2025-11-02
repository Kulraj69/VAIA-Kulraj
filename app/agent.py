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
    """Simple keyword-based routing."""
    lower_q = query.lower()
    if any(k in lower_q for k in ["extract", "json", "schema", "swot", "market size", "competitors"]):
        return "EXTRACT"
    if any(k in lower_q for k in ["summarize", "summary", "findings", "key takeaways"]):
        return "SUMMARY"
    return "QA"


# ===== SPECIALIZED AGENTS FOR MARKET ANALYSIS =====

MARKET_TRENDS_PROMPT = (
    "You are an expert market trends analyst. Focus on identifying patterns, "
    "emerging trends, growth trajectories, market movements, and future forecasts. "
    "Provide insights on market dynamics, consumer behavior shifts, and industry evolution. "
    "Use only provided source chunks when citing. Structure responses with clear trend analysis."
)

STRATEGY_PROMPT = (
    "You are a strategic business advisor specializing in market strategies. "
    "Focus on actionable recommendations, competitive positioning, go-to-market strategies, "
    "and tactical approaches. Provide strategic insights and actionable advice based on market data. "
    "Use only provided source chunks when citing."
)

ANALYSIS_PROMPT = (
    "You are a deep market analyst who provides comprehensive analysis. "
    "Focus on detailed examination of market data, competitive landscape, "
    "SWOT analysis, financial metrics, and multi-dimensional insights. "
    "Break down complex data into clear, structured analysis. "
    "Use only provided source chunks when citing."
)

ROUTER_SYSTEM_PROMPT = (
    "You are an intelligent routing agent. Analyze the user's query and determine which "
    "specialized agent should handle it:\n"
    "- TRENDS: Questions about market trends, growth patterns, emerging developments, forecasts, dynamics\n"
    "- STRATEGY: Questions about strategic recommendations, competitive positioning, tactics, go-to-market plans\n"
    "- ANALYSIS: Requests for deep analysis, SWOT, competitive landscape, detailed insights, financial metrics\n"
    "\nRespond with ONLY one word: TRENDS, STRATEGY, or ANALYSIS"
)


async def _router_node(state: AgentState) -> AgentState:
    """Reasoning node to decide which agent to use."""
    query: str = state["query"]
    
    # Use LLM to intelligently route the query
    routing_message = [
        {"role": "user", "content": f"Classify this query: {query}"}
    ]
    
    try:
        decision = await chat_complete(system_prompt=ROUTER_SYSTEM_PROMPT, messages=routing_message)
        decision = decision.strip().upper()
        
        # Validate decision
        if decision not in ["TRENDS", "STRATEGY", "ANALYSIS"]:
            # Fallback to keyword-based routing
            decision = await route_task(query)
            if decision == "EXTRACT":
                decision = "ANALYSIS"
            elif decision == "SUMMARY":
                decision = "TRENDS"
            else:
                decision = "ANALYSIS"  # Default to analysis for deep dive
        
        state["agent_type"] = decision
    except Exception:
        # Fallback to keyword routing on error
        state["agent_type"] = "ANALYSIS"
    
    return state


async def _retrieve_context_node(state: AgentState) -> AgentState:
    """Retrieve relevant context based on agent type."""
    query: str = state["query"]
    top_k: int = state["top_k"]
    source_file: str = state.get("source_file")
    agent_type: str = state.get("agent_type", "ANALYSIS")

    # Enhance query based on agent specialization
    enhanced_query = query
    if agent_type == "TRENDS":
        enhanced_query = f"market trends, growth patterns, forecasts: {query}"
    elif agent_type == "STRATEGY":
        enhanced_query = f"strategic recommendations, competitive positioning, tactics: {query}"
    elif agent_type == "ANALYSIS":
        enhanced_query = f"detailed analysis, competitive landscape, SWOT: {query}"

    q_vec = await embed_texts([enhanced_query])
    retrieved = await similarity_search(query_embedding=q_vec[0], top_k=top_k, source_file=source_file)

    state["retrieved_docs"] = [doc for _, doc in retrieved]
    return state


async def _specialized_agent_node(state: AgentState) -> AgentState:
    """Specialized agent reasoning node."""
    agent_type: str = state.get("agent_type", "ANALYSIS")
    history: List[Dict[str, str]] = state.get("history", [])
    query: str = state["query"]

    # Select appropriate system prompt
    if agent_type == "TRENDS":
        system_prompt = MARKET_TRENDS_PROMPT
    elif agent_type == "STRATEGY":
        system_prompt = STRATEGY_PROMPT
    else:  # ANALYSIS
        system_prompt = ANALYSIS_PROMPT

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
        {"role": "user", "content": f"CONTEXT:\n" + "\n\n".join(chunk_texts)},
        {"role": "user", "content": query},
    ]

    answer = await chat_complete(system_prompt=system_prompt, messages=messages)

    state["answer"] = answer
    state["sources"] = sources
    return state


async def run_specialized_agent(
    user_query: str,
    session_id: str,
    history: List[Dict[str, str]],
    top_k: int,
    source_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    New specialized ReAct agent with intelligent routing.
    
    Flow: Router -> Retrieve -> Specialized Agent -> Answer
    """
    state: AgentState = AgentState({
        "query": user_query,
        "history": history,
        "top_k": top_k,
        "source_file": source_file,
    })
    
    # Execute: Router -> Retrieve -> Specialized Reasoning
    state = await _router_node(state)
    state = await _retrieve_context_node(state)
    state = await _specialized_agent_node(state)
    
    return {
        "answer": state.get("answer", ""),
        "sources": state.get("sources", []),
        "agent_type": state.get("agent_type", "ANALYSIS")
    }


