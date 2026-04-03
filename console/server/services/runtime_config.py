"""Runtime-readable and runtime-editable Console configuration service."""

import asyncio
from typing import TypeAlias

from agiwo.config.settings import get_settings
from agiwo.llm.config_policy import validate_provider_model_params
from agiwo.skill.manager import get_global_skill_manager
from server.config import ConsoleConfig, DefaultAgentConfig
from server.models.runtime_config import (
    DefaultAgentConfigPayload,
    RuntimeConfigEditablePayload,
    RuntimeConfigResponse,
)

_RESTART_REQUIRED = [
    "console.server",
    "console.storage",
    "console.channels",
    "sdk.model/provider credentials",
]
_SECRET_KEY_PARTS = ("secret", "token", "password", "api_key")
_SECRET_KEY_EXACT_EXCEPTIONS = {"api_key_env_name"}
JSONValue: TypeAlias = (
    dict[str, "JSONValue"] | list["JSONValue"] | str | int | float | bool | None
)


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in _SECRET_KEY_EXACT_EXCEPTIONS:
        return False
    return any(part in lowered for part in _SECRET_KEY_PARTS)


def _mask_secrets(value: JSONValue, *, key: str | None = None) -> JSONValue:
    if key is not None and _is_secret_key(key):
        if value in (None, ""):
            return value
        return "***"
    if isinstance(value, dict):
        return {
            child_key: _mask_secrets(child_value, key=child_key)
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_mask_secrets(item) for item in value]
    return value


class RuntimeConfigService:
    """Owns the runtime-only config override surface for Console."""

    def __init__(self, console_config: ConsoleConfig) -> None:
        self._console_config = console_config
        self._lock = asyncio.Lock()

    async def get_snapshot(self) -> RuntimeConfigResponse:
        skill_manager = get_global_skill_manager()
        await skill_manager.initialize()
        editable = self._build_editable_payload(skill_manager=skill_manager)
        runtime_settings = get_settings()
        readonly = {
            "console": _mask_secrets(
                self._console_config.model_dump(mode="json", exclude={"default_agent"})
            ),
            "sdk": _mask_secrets(
                runtime_settings.model_dump(mode="json", exclude={"skills_dirs"})
            ),
        }
        effective = {
            "skills_dirs": list(editable.skills_dirs),
            "resolved_skills_dirs": [
                str(path) for path in skill_manager.get_resolved_skills_dirs()
            ],
            "default_agent": editable.default_agent.model_dump(mode="json"),
        }
        return RuntimeConfigResponse(
            editable=editable,
            effective=effective,
            readonly=readonly,
            runtime_only=True,
            restart_required=list(_RESTART_REQUIRED),
        )

    async def update(
        self, payload: RuntimeConfigEditablePayload
    ) -> RuntimeConfigResponse:
        async with self._lock:
            runtime_settings = get_settings()
            original_skills_dirs = list(runtime_settings.skills_dirs)
            original_default_agent = self._console_config.default_agent.model_copy(
                deep=True
            )
            try:
                model_params = payload.default_agent.model_params.model_dump(
                    exclude_none=True
                )
                validate_provider_model_params(
                    payload.default_agent.model_provider,
                    model_params,
                )
                runtime_settings.skills_dirs = list(payload.skills_dirs)

                skill_manager = get_global_skill_manager()
                await skill_manager.initialize()
                expanded_allowed_skills = (
                    skill_manager.expand_allowed_skills(
                        payload.default_agent.allowed_skills
                    )
                    or []
                )

                self._console_config.default_agent = DefaultAgentConfig(
                    id=payload.default_agent.id,
                    name=payload.default_agent.name,
                    description=payload.default_agent.description,
                    model_provider=payload.default_agent.model_provider,
                    model_name=payload.default_agent.model_name,
                    model_params=model_params,
                    system_prompt=payload.default_agent.system_prompt,
                    tools=list(payload.default_agent.tools),
                    allowed_skills=expanded_allowed_skills,
                )
                return await self.get_snapshot()
            except Exception:
                runtime_settings.skills_dirs = original_skills_dirs
                self._console_config.default_agent = original_default_agent
                raise

    def _build_editable_payload(
        self,
        *,
        skill_manager,
    ) -> RuntimeConfigEditablePayload:
        runtime_settings = get_settings()
        default_agent = self._console_config.default_agent
        expanded_allowed_skills = (
            skill_manager.expand_allowed_skills(default_agent.allowed_skills) or []
        )
        return RuntimeConfigEditablePayload(
            skills_dirs=list(runtime_settings.skills_dirs),
            default_agent=DefaultAgentConfigPayload(
                id=default_agent.id,
                name=default_agent.name,
                description=default_agent.description,
                model_provider=default_agent.model_provider,
                model_name=default_agent.model_name,
                system_prompt=default_agent.system_prompt,
                tools=list(default_agent.tools),
                allowed_skills=expanded_allowed_skills,
                model_params=default_agent.model_params,
            ),
        )


__all__ = ["RuntimeConfigService"]
