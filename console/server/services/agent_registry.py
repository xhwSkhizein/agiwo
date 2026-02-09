"""
Agent registry — manages agent configurations and instantiation.

Persists AgentConfig to SQLite or MongoDB alongside the main storage.
"""

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

import aiosqlite
from pydantic import BaseModel, Field

from server.config import ConsoleConfig


class AgentConfigRecord(BaseModel):
    """Persisted agent configuration."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    model_provider: str
    model_name: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    model_params: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class AgentRegistry:
    """
    CRUD operations for agent configurations.

    Supports SQLite and MongoDB backends matching the console config.
    """

    def __init__(self, config: ConsoleConfig) -> None:
        self._config = config
        self._sqlite_conn: aiosqlite.Connection | None = None
        self._mongo_collection: Any = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        if self._config.storage_type == "sqlite":
            await self._init_sqlite()
        else:
            await self._init_mongodb()

        self._initialized = True

    async def close(self) -> None:
        if self._sqlite_conn is not None:
            await self._sqlite_conn.close()
            self._sqlite_conn = None
        self._initialized = False

    # ── CRUD ────────────────────────────────────────────────────────────

    async def list_agents(self, limit: int = 50, offset: int = 0) -> list[AgentConfigRecord]:
        if self._config.storage_type == "sqlite":
            return await self._sqlite_list(limit, offset)
        return await self._mongo_list(limit, offset)

    async def get_agent(self, agent_id: str) -> AgentConfigRecord | None:
        if self._config.storage_type == "sqlite":
            return await self._sqlite_get(agent_id)
        return await self._mongo_get(agent_id)

    async def create_agent(self, record: AgentConfigRecord) -> AgentConfigRecord:
        if self._config.storage_type == "sqlite":
            await self._sqlite_upsert(record)
        else:
            await self._mongo_upsert(record)
        return record

    async def update_agent(self, agent_id: str, updates: dict[str, Any]) -> AgentConfigRecord | None:
        existing = await self.get_agent(agent_id)
        if existing is None:
            return None

        for key, value in updates.items():
            if value is not None and hasattr(existing, key):
                setattr(existing, key, value)
        existing.updated_at = datetime.now()

        if self._config.storage_type == "sqlite":
            await self._sqlite_upsert(existing)
        else:
            await self._mongo_upsert(existing)
        return existing

    async def delete_agent(self, agent_id: str) -> bool:
        if self._config.storage_type == "sqlite":
            return await self._sqlite_delete(agent_id)
        return await self._mongo_delete(agent_id)

    # ── SQLite ──────────────────────────────────────────────────────────

    async def _init_sqlite(self) -> None:
        from pathlib import Path
        import os

        db_path = os.path.expanduser(self._config.sqlite_db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._sqlite_conn = await aiosqlite.connect(db_path)
        self._sqlite_conn.row_factory = aiosqlite.Row

        await self._sqlite_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_configs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                model_provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                system_prompt TEXT DEFAULT '',
                tools TEXT DEFAULT '[]',
                options TEXT DEFAULT '{}',
                model_params TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        await self._sqlite_conn.commit()

    async def _sqlite_list(self, limit: int, offset: int) -> list[AgentConfigRecord]:
        if self._sqlite_conn is None:
            return []

        rows = []
        async with self._sqlite_conn.execute(
            "SELECT * FROM agent_configs ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cursor:
            async for row in cursor:
                rows.append(self._sqlite_deserialize(row))
        return rows

    async def _sqlite_get(self, agent_id: str) -> AgentConfigRecord | None:
        if self._sqlite_conn is None:
            return None

        async with self._sqlite_conn.execute(
            "SELECT * FROM agent_configs WHERE id = ?", (agent_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is not None:
                return self._sqlite_deserialize(row)
        return None

    async def _sqlite_upsert(self, record: AgentConfigRecord) -> None:
        if self._sqlite_conn is None:
            return

        data = self._sqlite_serialize(record)
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        values = list(data.values())

        await self._sqlite_conn.execute(
            f"INSERT OR REPLACE INTO agent_configs ({columns}) VALUES ({placeholders})",
            values,
        )
        await self._sqlite_conn.commit()

    async def _sqlite_delete(self, agent_id: str) -> bool:
        if self._sqlite_conn is None:
            return False

        cursor = await self._sqlite_conn.execute(
            "DELETE FROM agent_configs WHERE id = ?", (agent_id,)
        )
        await self._sqlite_conn.commit()
        return cursor.rowcount > 0

    def _sqlite_serialize(self, record: AgentConfigRecord) -> dict[str, Any]:
        data = record.model_dump(mode="json")
        data["tools"] = json.dumps(data["tools"])
        data["options"] = json.dumps(data["options"])
        data["model_params"] = json.dumps(data["model_params"])
        if isinstance(data["created_at"], datetime):
            data["created_at"] = data["created_at"].isoformat()
        if isinstance(data["updated_at"], datetime):
            data["updated_at"] = data["updated_at"].isoformat()
        return data

    def _sqlite_deserialize(self, row: aiosqlite.Row) -> AgentConfigRecord:
        data = dict(row)
        data["tools"] = json.loads(data.get("tools") or "[]")
        data["options"] = json.loads(data.get("options") or "{}")
        data["model_params"] = json.loads(data.get("model_params") or "{}")
        return AgentConfigRecord.model_validate(data)

    # ── MongoDB ─────────────────────────────────────────────────────────

    async def _init_mongodb(self) -> None:
        from motor.motor_asyncio import AsyncIOMotorClient

        client = AsyncIOMotorClient(self._config.mongodb_uri)
        db = client[self._config.mongodb_db_name]
        self._mongo_collection = db["agent_configs"]
        await self._mongo_collection.create_index("id", unique=True)

    async def _mongo_list(self, limit: int, offset: int) -> list[AgentConfigRecord]:
        if self._mongo_collection is None:
            return []

        results = []
        cursor = (
            self._mongo_collection.find()
            .sort("updated_at", -1)
            .skip(offset)
            .limit(limit)
        )
        async for doc in cursor:
            doc.pop("_id", None)
            results.append(AgentConfigRecord.model_validate(doc))
        return results

    async def _mongo_get(self, agent_id: str) -> AgentConfigRecord | None:
        if self._mongo_collection is None:
            return None

        doc = await self._mongo_collection.find_one({"id": agent_id})
        if doc is not None:
            doc.pop("_id", None)
            return AgentConfigRecord.model_validate(doc)
        return None

    async def _mongo_upsert(self, record: AgentConfigRecord) -> None:
        if self._mongo_collection is None:
            return

        data = record.model_dump(mode="json")
        await self._mongo_collection.replace_one(
            {"id": record.id}, data, upsert=True
        )

    async def _mongo_delete(self, agent_id: str) -> bool:
        if self._mongo_collection is None:
            return False

        result = await self._mongo_collection.delete_one({"id": agent_id})
        return result.deleted_count > 0
