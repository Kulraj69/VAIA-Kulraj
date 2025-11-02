# Specialized ReAct Agents for Market Analysis

## Overview

The system now includes **3 specialized agents** for market analysis with **intelligent routing**:

1. **TRENDS Agent** - Market trends, growth patterns, forecasts
2. **STRATEGY Agent** - Strategic recommendations, competitive positioning
3. **ANALYSIS Agent** - Deep analysis, SWOT, competitive landscape

## Architecture

```
User Query
    ↓
Router Agent (LLM-based reasoning)
    ↓
[TRENDS | STRATEGY | ANALYSIS]
    ↓
Enhanced Retrieval
    ↓
Specialized Agent Reasoning
    ↓
Answer + Sources + Agent Type
```

## FastAPI Endpoint

### Endpoint: `POST /specialized-agent`

**Request:**
```json
{
  "query": "What are the emerging trends in the market?",
  "session_id": "optional-session-id",
  "top_k": 4
}
```

**Response:**
```json
{
  "answer": "Based on market analysis...",
  "sources": [
    {"chunk_id": "...", "section": "..."}
  ],
  "agent_type": "TRENDS"
}
```

## Testing

### Example 1: Trends Query
```bash
curl -X POST "http://localhost:8000/specialized-agent" \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the key growth trends and market forecasts?","top_k":4}'
```

**Expected agent_type:** `TRENDS`

### Example 2: Strategy Query
```bash
curl -X POST "http://localhost:8000/specialized-agent" \
  -H "Content-Type: application/json" \
  -d '{"query":"What strategic recommendations should we implement?","top_k":4}'
```

**Expected agent_type:** `STRATEGY`

### Example 3: Analysis Query
```bash
curl -X POST "http://localhost:8000/specialized-agent" \
  -H "Content-Type: application/json" \
  -d '{"query":"Provide a SWOT analysis of the competitive landscape","top_k":4}'
```

**Expected agent_type:** `ANALYSIS`

### Example 4: With Session Memory
```bash
curl -X POST "http://localhost:8000/specialized-agent" \
  -H "Content-Type: application/json" \
  -d '{
    "query":"Follow up: what are the next steps?",
    "session_id":"test-session-123",
    "top_k":4
  }'
```

## Agent Specializations

### TRENDS Agent
**Focus:** Market dynamics, growth patterns, emerging developments
**Best for:** 
- "What are the emerging trends?"
- "Show market growth projections"
- "What are the forecasted market dynamics?"

### STRATEGY Agent
**Focus:** Actionable recommendations, competitive tactics
**Best for:**
- "What strategies should we adopt?"
- "How can we improve competitive positioning?"
- "Recommend go-to-market approaches"

### ANALYSIS Agent
**Focus:** Comprehensive analysis, SWOT, detailed insights
**Best for:**
- "Perform a SWOT analysis"
- "Analyze the competitive landscape"
- "Break down financial metrics"

## Routing Logic

The Router Agent uses **LLM-based reasoning** with fallback:

1. **Primary:** LLM analyzes query intent
2. **Fallback:** Keyword-based routing
3. **Default:** ANALYSIS agent

### Query Enhancement

Each agent enhances retrieval with context:
- **TRENDS:** Adds "market trends, growth patterns, forecasts"
- **STRATEGY:** Adds "strategic recommendations, competitive positioning, tactics"
- **ANALYSIS:** Adds "detailed analysis, competitive landscape, SWOT"

## Code Structure

```python
# Agent implementations in app/agent.py
- _router_node()           # Intelligent routing
- _retrieve_context_node() # Enhanced retrieval
- _specialized_agent_node() # Specialized reasoning
- run_specialized_agent()  # Main orchestration

# Prompts
- MARKET_TRENDS_PROMPT    # Trends agent
- STRATEGY_PROMPT         # Strategy agent  
- ANALYSIS_PROMPT         # Analysis agent
- ROUTER_SYSTEM_PROMPT    # Router agent
```

## Migration

**No changes to Streamlit app yet** - this is tested via FastAPI first.

To integrate with Streamlit:
1. Replace `run_chat()` with `run_specialized_agent()`
2. Display `agent_type` in UI
3. Add agent type indicator to responses

## Next Steps

- ✅ Create specialized agents
- ✅ Add intelligent routing
- ✅ Implement FastAPI endpoint
- ✅ Test with various queries
- ⏳ Integrate with Streamlit
- ⏳ Add agent visualization
- ⏳ Fine-tune prompts

