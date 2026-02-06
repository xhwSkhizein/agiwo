from dataclasses import dataclass
import os
import datetime
from pathlib import Path

from agiwo.config.settings import settings
from agiwo.skill.manager import SkillManager


@dataclass
class AgentOptions:
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

    # stream
    stream_cleanup_timeout: float = 5.0

    def __post_init__(self):
        self.work_dir = os.getcwd() if self.work_dir is None else self.work_dir
        self.date_yyyyMMdd = (
            datetime.datetime.now().strftime("%Y-%m-%d")
            if self.date_yyyyMMdd is None
            else self.date_yyyyMMdd
        )

        if self.skill_manager is None:
            if settings.is_skills_enabled:
                skills_dirs = [Path(d) for d in settings.skills_dirs]
                self.skill_manager = SkillManager(skills_dirs=skills_dirs)
            else:
                self.skill_manager = None
