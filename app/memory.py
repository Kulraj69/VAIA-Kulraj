from typing import List, Dict, Any
from datetime import datetime, timezone

from .db import get_db


MEMORY_COLLECTION_NAME = "memory"


async def fetch_session_history(session_id: str) -> List[Dict[str, str]]:
    db = await get_db()
    doc = await db[MEMORY_COLLECTION_NAME].find_one({"session_id": session_id})
    if not doc:
        return []
    history = doc.get("turns", [])
    # Coerce to simple role/content pairs for prompt context
    result: List[Dict[str, str]] = []
    for turn in history:
        user_content = turn.get("user", {}).get("content", "")
        assistant_content = turn.get("assistant", {}).get("content", "")
        if user_content:
            result.append({"role": "user", "content": user_content})
        if assistant_content:
            result.append({"role": "assistant", "content": assistant_content})
    return result


async def append_session_turn(
    session_id: str,
    user_content: str,
    assistant_content: str,
    sources: List[Dict[str, Any]],
) -> None:
    db = await get_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    turn_doc = {
        "user": {"content": user_content, "ts": now_iso},
        "assistant": {"content": assistant_content, "ts": now_iso},
        "sources": sources,
    }
    await db[MEMORY_COLLECTION_NAME].update_one(
        {"session_id": session_id},
        {"$push": {"turns": turn_doc}, "$setOnInsert": {"created_at": now_iso}},
        upsert=True,
    )


