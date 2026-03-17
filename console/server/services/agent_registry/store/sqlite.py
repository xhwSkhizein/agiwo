"""SQLite backend for agent registry."""

import json
import os
from typing import Any

import aiosqlite

from agiwo.utils.sqlite_pool import get_shared_connection, release_shared_connection
from server.services.agent_registry.models import AgentConfigRecord


class SqliteAgentRegistryStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = os.path.expanduser(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self._conn is not None:
            return
        self._conn = await get_shared_connection(self._db_path)
        await self._create_tables()

    async def close(self) -> None:
        if self._conn is None:
            return
        await release_shared_connection(self._db_path)
        self._conn = None

    async def list_agents(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AgentConfigRecord]:
        conn = await self._require_conn()
        rows: list[AgentConfigRecord] = []
        async with conn.execute(
            """
            SELECT *
            FROM agent_configs
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ) as cursor:
            async for row in cursor:
                rows.append(self._deserialize_row(row))
        return rows

    async def get_agent(self, agent_id: str) -> AgentConfigRecord | None:
        conn = await self._require_conn()
        async with conn.execute(
            "SELECT * FROM agent_configs WHERE id = ?",
            (agent_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._deserialize_row(row)

    async def get_agent_by_name(self, agent_name: str) -> AgentConfigRecord | None:
        conn = await self._require_conn()
        async with conn.execute(
            """
            SELECT *
            FROM agent_configs
            WHERE name = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (agent_name,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._deserialize_row(row)

    async def upsert_agent(self, record: AgentConfigRecord) -> None:
        conn = await self._require_conn()
        data = self._serialize_record(record)
        await conn.execute(
            """
            INSERT OR REPLACE INTO agent_configs (
                id,
                name,
                description,
                model_provider,
                model_name,
                system_prompt,
                tools,
                options,
                model_params,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["name"],
                data["description"],
                data["model_provider"],
                data["model_name"],
                data["system_prompt"],
                data["tools"],
                data["options"],
                data["model_params"],
                data["created_at"],
                data["updated_at"],
            ),
        )
        await conn.commit()

    async def delete_agent(self, agent_id: str) -> bool:
        conn = await self._require_conn()
        cursor = await conn.execute(
            "DELETE FROM agent_configs WHERE id = ?",
            (agent_id,),
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def _create_tables(self) -> None:
        conn = await self._require_conn()
        await conn.execute(
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
        await conn.commit()

    async def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            await self.connect()
        assert self._conn is not None
        return self._conn

    def _serialize_record(self, record: AgentConfigRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "name": record.name,
            "description": record.description,
            "model_provider": record.model_provider,
            "model_name": record.model_name,
            "system_prompt": record.system_prompt,
            "tools": json.dumps(record.tools),
            "options": json.dumps(record.options),
            "model_params": json.dumps(record.model_params),
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }

    def _deserialize_row(self, row: aiosqlite.Row) -> AgentConfigRecord:
        return AgentConfigRecord.model_validate(
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "model_provider": row["model_provider"],
                "model_name": row["model_name"],
                "system_prompt": row["system_prompt"],
                "tools": json.loads(row["tools"] or "[]"),
                "options": json.loads(row["options"] or "{}"),
                "model_params": json.loads(row["model_params"] or "{}"),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )


__all__ = ["SqliteAgentRegistryStore"]
