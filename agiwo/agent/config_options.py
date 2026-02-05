from dataclasses import dataclass
import os
import datetime

from agiwo.config.settings import settings
from agiwo.agent.session.base import SessionStore, InMemorySessionStore
from agiwo.agent.session.sqlite import SQLiteSessionStore
from agiwo.agent.session.mongo import MongoSessionStore
from agiwo.skill.manager import SkillManager


@dataclass
class AgentConfigOptions:
    max_steps: int = 10
    run_timeout: int = 600  # seconds
    max_output_tokens: int = 8196

    # Agent Loop Configuration
    enable_termination_summary: bool = False
    termination_summary_prompt: str = ""

    # runtime vars
    work_dir: str | None = None
    date_yyyyMMdd: str | None = None

    # skills
    skill_manager: SkillManager | None = None

    from typing import Any

    # tracing
    is_trace_enabled: bool = True
    stream_cleanup_timeout: float = 5.0
    trace_store: Any | None = None

    # side-effect saving
    session_store: SessionStore | None = None

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

                skill_dirs = [Path(d) for d in settings.skill_dirs]
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

        if settings.default_trace_store not in ("mongo", "sqlite"):
            self.is_trace_enabled = False
