from typing import Optional
import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase


_mongo_client: Optional[AsyncIOMotorClient] = None
_mongo_db: Optional[AsyncIOMotorDatabase] = None


MONGODB_URI_ENV_VAR = "MONGODB_URI"
MONGO_DB_NAME_ENV_VAR = "MONGO_DB_NAME"


async def get_db() -> AsyncIOMotorDatabase:
    global _mongo_client, _mongo_db
    if _mongo_db is not None:
        return _mongo_db

    mongo_uri = os.environ.get(MONGODB_URI_ENV_VAR, "mongodb://localhost:27017")
    db_name = os.environ.get(MONGO_DB_NAME_ENV_VAR, "vaia_market_analyst")

    _mongo_client = AsyncIOMotorClient(mongo_uri)
    _mongo_db = _mongo_client[db_name]
    return _mongo_db


