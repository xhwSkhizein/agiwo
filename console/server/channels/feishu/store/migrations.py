"""One-time SQLite schema migrations for Feishu channel tables."""

import aiosqlite


async def migrate_chat_context_table(conn: aiosqlite.Connection) -> None:
    """Create or migrate feishu_channel_chat_context — scope_id is now the PK."""
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='feishu_channel_chat_context'"
    ) as cur:
        exists = await cur.fetchone()

    if not exists:
        await conn.execute(
            """
            CREATE TABLE feishu_channel_chat_context (
                scope_id TEXT PRIMARY KEY,
                channel_instance_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                chat_type TEXT NOT NULL,
                user_open_id TEXT NOT NULL,
                base_agent_id TEXT NOT NULL,
                current_session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        return

    async with conn.execute("PRAGMA table_info(feishu_channel_chat_context)") as cur:
        columns = {row[1] for row in await cur.fetchall()}

    if "id" not in columns:
        return

    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feishu_channel_chat_context_new (
            scope_id TEXT PRIMARY KEY,
            channel_instance_id TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            chat_type TEXT NOT NULL,
            user_open_id TEXT NOT NULL,
            base_agent_id TEXT NOT NULL,
            current_session_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    await conn.execute(
        """
        INSERT OR REPLACE INTO feishu_channel_chat_context_new
            (scope_id, channel_instance_id, chat_id, chat_type,
             user_open_id, base_agent_id, current_session_id, created_at, updated_at)
        SELECT scope_id, channel_instance_id, chat_id, chat_type,
               user_open_id, base_agent_id, current_session_id, created_at, updated_at
        FROM feishu_channel_chat_context
        """
    )
    await conn.execute("DROP TABLE feishu_channel_chat_context")
    await conn.execute(
        "ALTER TABLE feishu_channel_chat_context_new RENAME TO feishu_channel_chat_context"
    )


async def migrate_session_table(conn: aiosqlite.Connection) -> None:
    """Create or migrate feishu_session — rename chat_context_id to chat_context_scope_id."""
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='feishu_session'"
    ) as cur:
        exists = await cur.fetchone()

    if not exists:
        await conn.execute(
            """
            CREATE TABLE feishu_session (
                id TEXT PRIMARY KEY,
                chat_context_scope_id TEXT NOT NULL,
                base_agent_id TEXT NOT NULL,
                runtime_agent_id TEXT NOT NULL,
                scheduler_state_id TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                current_task_id TEXT,
                task_message_count INTEGER NOT NULL DEFAULT 0,
                source_session_id TEXT,
                source_task_id TEXT,
                fork_context_summary TEXT
            )
            """
        )
        return

    async with conn.execute("PRAGMA table_info(feishu_session)") as cur:
        columns = {row[1] for row in await cur.fetchall()}

    if "chat_context_scope_id" in columns:
        return

    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feishu_session_new (
            id TEXT PRIMARY KEY,
            chat_context_scope_id TEXT NOT NULL,
            base_agent_id TEXT NOT NULL,
            runtime_agent_id TEXT NOT NULL,
            scheduler_state_id TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            current_task_id TEXT,
            task_message_count INTEGER NOT NULL DEFAULT 0,
            source_session_id TEXT,
            source_task_id TEXT,
            fork_context_summary TEXT
        )
        """
    )
    await conn.execute(
        """
        INSERT OR REPLACE INTO feishu_session_new
            (id, chat_context_scope_id, base_agent_id, runtime_agent_id,
             scheduler_state_id, created_by, created_at, updated_at,
             current_task_id, task_message_count, source_session_id,
             source_task_id, fork_context_summary)
        SELECT
            s.id,
            COALESCE(c.scope_id, s.chat_context_id),
            s.base_agent_id, s.runtime_agent_id,
            s.scheduler_state_id, s.created_by, s.created_at, s.updated_at,
            s.current_task_id, s.task_message_count, s.source_session_id,
            s.source_task_id, s.fork_context_summary
        FROM feishu_session s
        LEFT JOIN feishu_channel_chat_context c ON s.chat_context_id = c.scope_id
            OR s.chat_context_id = c.id
        """
    )
    await conn.execute("DROP TABLE feishu_session")
    await conn.execute("ALTER TABLE feishu_session_new RENAME TO feishu_session")
