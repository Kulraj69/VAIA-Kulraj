from typing import List
import os

import numpy as np
import httpx


# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT_ENV = "AZURE_OPENAI_ENDPOINT"  # e.g., https://tech-8568-resource.cognitiveservices.azure.com
AZURE_OPENAI_API_KEY_ENV = "AZURE_OPENAI_API_KEY"    # the provided key
AZURE_OPENAI_API_VERSION_ENV = "AZURE_OPENAI_API_VERSION"  # e.g., 2025-01-01-preview

# Deployment names
AZURE_OPENAI_CHAT_DEPLOYMENT_ENV = "AZURE_OPENAI_CHAT_DEPLOYMENT"       # e.g., gpt-4o-mini
AZURE_OPENAI_EMBED_DEPLOYMENT_ENV = "AZURE_OPENAI_EMBED_DEPLOYMENT"     # e.g., text-embedding-3-small


# Sensible defaults matching your provided URLs
DEFAULT_AZURE_API_VERSION = "2025-01-01-preview"


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


async def embed_texts(texts: List[str]) -> np.ndarray:
    endpoint = _require_env(AZURE_OPENAI_ENDPOINT_ENV)
    api_key = _require_env(AZURE_OPENAI_API_KEY_ENV)
    api_version = os.environ.get(AZURE_OPENAI_API_VERSION_ENV, DEFAULT_AZURE_API_VERSION)
    deployment = _require_env(AZURE_OPENAI_EMBED_DEPLOYMENT_ENV)

    url = f"{endpoint}/openai/deployments/{deployment}/embeddings"
    params = {"api-version": api_version}
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    payload = {"input": texts}

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, params=params, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        vectors = [item["embedding"] for item in data.get("data", [])]
        return np.array(vectors, dtype=np.float32)


async def chat_complete(system_prompt: str, messages: List[dict]) -> str:
    endpoint = _require_env(AZURE_OPENAI_ENDPOINT_ENV)
    api_key = _require_env(AZURE_OPENAI_API_KEY_ENV)
    api_version = os.environ.get(AZURE_OPENAI_API_VERSION_ENV, DEFAULT_AZURE_API_VERSION)
    deployment = _require_env(AZURE_OPENAI_CHAT_DEPLOYMENT_ENV)

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions"
    params = {"api-version": api_version}
    headers = {"api-key": api_key, "Content-Type": "application/json"}
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


