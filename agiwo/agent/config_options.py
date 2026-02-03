from dataclasses import dataclass
import os
import datetime

from agiwo.config.settings import settings
from agiwo.agent.session.base import SessionStore, InMemorySessionStore
from agiwo.agent.session.sqlite import SQLiteSessionStore
from agiwo.agent.session.mongo import MongoSessionStore
from agiwo.observability.collector import TraceCollector
from agiwo.observability.store import TraceStore
from agiwo.observability.sqlite_store import SQLiteTraceStore
from agiwo.skills.manager import SkillManager


@dataclass
class AgentConfigOptions:
    max_steps: int = 10
    timeout_per_step: int = 120  # seconds
    run_timeout: int = 600  # seconds
    parallel_tool_calls: bool = True
    max_output_tokens: int = 8196
    max_history_messages: int = 10

    # Agent Loop Configuration
    enable_termination_summary: bool = False
    termination_summary_prompt: str = ""

    enable_auto_compact: bool = False
    auto_compact_prompt: str = ""
    auto_compact_threshold: float = 0.75

    # runtime vars
    work_dir: str | None = None
    date_yyyyMMdd: str | None = None

    # skills
    skill_manager: SkillManager | None = None

    # side-effect saving
    session_store: SessionStore | None = None
    trace_collector: TraceCollector | None = None

    def __post_init__(self):
        self.work_dir = os.getcwd() if self.work_dir is None else self.work_dir
        self.date_yyyyMMdd = (
            datetime.datetime.now().strftime("%Y-%m-%d")
            if self.date_yyyyMMdd is None
            else self.date_yyyyMMdd
        )

        if self.skill_manager is None:
            if settings.is_skills_enabled:
                from pathlib import Path
                skill_dirs = [Path(d) for d in settings.skills_dirs]
                self.skill_manager = SkillManager(skill_dirs=skill_dirs)
            else:
                self.skill_manager = None

        if self.session_store is None:
            if settings.default_session_store == "sqlite":
                self.session_store = SQLiteSessionStore()
            elif settings.default_session_store == "mongo":
                self.session_store = MongoSessionStore()
            else:
                self.session_store = InMemorySessionStore()

        if self.trace_collector is None:
            if settings.default_trace_store == "mongo":
                trace_store = TraceStore(
                    mongo_uri=settings.mongo_uri,
                    db_name=settings.mongo_db_name,
                    collection_name=settings.trace_collection_name,
                    buffer_size=settings.trace_buffer_size,
                )
            elif settings.default_trace_store == "sqlite":
                trace_store = SQLiteTraceStore(
                    db_path=settings.sqlite_db_path,
                    collection_name=settings.trace_collection_name,
                    buffer_size=settings.trace_buffer_size,
                )
            else:
                trace_store = None
            self.trace_collector = (
                TraceCollector(store=trace_store) if trace_store else None
            )
