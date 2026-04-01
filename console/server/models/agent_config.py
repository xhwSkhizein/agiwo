"""Shared agent configuration models used by services and API view models."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agiwo.config.settings import settings
from agiwo.llm.config_policy import sanitize_model_params_data
from agiwo.skill.config import normalize_skill_dirs


def sanitize_agent_options_data(
    data: object,
    *,
    preserve_non_dict: bool = False,
) -> object:
    if not isinstance(data, dict):
        return data if preserve_non_dict else {}
    sanitized = dict(data)
    skills_dirs = normalize_skill_dirs(sanitized.get("skills_dirs"))
    if skills_dirs is None:
        sanitized.pop("skills_dirs", None)
    else:
        sanitized["skills_dirs"] = skills_dirs
    return sanitized


class AgentOptionsInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    config_root: str = ""
    max_steps: int = Field(default=50, ge=1)
    run_timeout: int = Field(default=600, ge=1)
    max_input_tokens_per_call: int | None = Field(default=None, ge=1)
    max_run_cost: float | None = Field(default=None, ge=0)
    enable_termination_summary: bool = True
    termination_summary_prompt: str = ""
    enable_skill: bool = Field(default_factory=lambda: settings.is_skills_enabled)
    skills_dirs: list[str] | None = None
    relevant_memory_max_token: int = Field(default=2048, ge=1)
    stream_cleanup_timeout: float = Field(default=300.0, gt=0)
    compact_prompt: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        return sanitize_agent_options_data(data, preserve_non_dict=True)


class ModelParamsInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    base_url: str | None = None
    api_key_env_name: str | None = None
    max_output_tokens: int = Field(default=4096, ge=1)
    max_context_window: int = Field(default=200000, ge=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2)
    cache_hit_price: float = Field(default=0.0, ge=0)
    input_price: float = Field(default=0.0, ge=0)
    output_price: float = Field(default=0.0, ge=0)
    aws_region: str | None = None
    aws_profile: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_plain_api_key(cls, data: object) -> object:
        return sanitize_model_params_data(
            data,
            preserve_non_dict=True,
            drop_null_keys=False,
        )
