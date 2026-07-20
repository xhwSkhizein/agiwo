"""Phase-based hook registry for agent runtime extensibility."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import traceback as traceback_lib
from typing import Any, Protocol

from agiwo.agent.models.input import UserInput, UserMessage
from agiwo.agent.models.run import MemoryRecord
from agiwo.config.settings import get_settings
from agiwo.memory import WorkspaceMemoryService
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class HookPhase(str, Enum):
    """Agent runtime hook phases."""

    PREPARE = "prepare"
    ASSEMBLE_CONTEXT = "assemble_context"
    BEFORE_LLM = "before_llm"
    AFTER_LLM = "after_llm"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    BEFORE_COMPACTION = "before_compaction"
    AFTER_COMPACTION = "after_compaction"
    BEFORE_REVIEW = "before_review"
    BEFORE_TERMINATION = "before_termination"
    AFTER_TERMINATION = "after_termination"
    AFTER_STEP_COMMIT = "after_step_commit"
    RUN_FINALIZED = "run_finalized"
    MEMORY_PERSIST = "memory_persist"
    COMPACTION_FAILED = "compaction_failed"


class HookCapability(str, Enum):
    OBSERVE_ONLY = "observe_only"
    TRANSFORM = "transform"
    DECISION_SUPPORT = "decision_support"


class HookGroup(str, Enum):
    SYSTEM = "system"
    RUNTIME_ADAPTER = "runtime_adapter"
    USER = "user"


class MemoryHookContext(Protocol):
    agent_name: str
    agent_id: str


class PhaseHook(Protocol):
    async def __call__(self, payload: dict[str, Any]) -> object:
        """Execute a hook for the given payload."""


@dataclass(frozen=True)
class HookRegistration:
    phase: HookPhase
    capability: HookCapability
    handler_name: str
    handler: PhaseHook
    group: HookGroup = HookGroup.USER
    order: int = 100
    critical: bool = False


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    """Full contract for one hook phase: capabilities, field allowlists, critical."""

    allow_transform: bool = False
    allow_decision_support: bool = False
    allow_critical: bool = False
    transform_fields: frozenset[str] = frozenset()
    decision_support_fields: frozenset[str] = frozenset()


_DEFAULT_PHASE_SPEC = PhaseSpec()

PHASE_SPECS: dict[HookPhase, PhaseSpec] = {
    HookPhase.PREPARE: PhaseSpec(
        allow_transform=True,
        allow_critical=True,
        transform_fields=frozenset({"prelude_text"}),
    ),
    HookPhase.ASSEMBLE_CONTEXT: PhaseSpec(
        allow_transform=True,
        allow_critical=True,
        transform_fields=frozenset({"memories", "context_additions"}),
    ),
    HookPhase.BEFORE_LLM: PhaseSpec(
        allow_transform=True,
        allow_decision_support=True,
        allow_critical=True,
        transform_fields=frozenset({"messages", "model_settings_override"}),
        decision_support_fields=frozenset({"llm_advice"}),
    ),
    HookPhase.BEFORE_TOOL_CALL: PhaseSpec(
        allow_transform=True,
        allow_decision_support=True,
        allow_critical=True,
        transform_fields=frozenset({"parameters"}),
        decision_support_fields=frozenset({"tool_advice"}),
    ),
    HookPhase.BEFORE_COMPACTION: PhaseSpec(
        allow_decision_support=True,
        decision_support_fields=frozenset({"compaction_advice"}),
    ),
    HookPhase.BEFORE_REVIEW: PhaseSpec(
        allow_decision_support=True,
        decision_support_fields=frozenset({"review_advice"}),
    ),
    HookPhase.BEFORE_TERMINATION: PhaseSpec(
        allow_decision_support=True,
        decision_support_fields=frozenset({"termination_advice"}),
    ),
}


def phase_spec(phase: HookPhase) -> PhaseSpec:
    """Return the contract for ``phase`` (observe-only default when unset)."""
    return PHASE_SPECS.get(phase, _DEFAULT_PHASE_SPEC)


@dataclass
class HookRegistry:
    registrations: list[HookRegistration] = field(default_factory=list)

    _GROUP_ORDER = {
        HookGroup.SYSTEM: 0,
        HookGroup.RUNTIME_ADAPTER: 1,
        HookGroup.USER: 2,
    }

    def __post_init__(self) -> None:
        for registration in self.registrations:
            self._validate_registration(registration)

    def for_phase(self, phase: HookPhase) -> list[HookRegistration]:
        indexed = list(enumerate(self.registrations))
        matching = [(index, item) for index, item in indexed if item.phase == phase]
        matching.sort(
            key=lambda pair: (
                self._GROUP_ORDER[pair[1].group],
                pair[1].order,
                pair[0],
            )
        )
        return [item for _, item in matching]

    def has_phase(self, phase: HookPhase) -> bool:
        return bool(self.for_phase(phase))

    def has_handler(self, handler_name: str) -> bool:
        return any(item.handler_name == handler_name for item in self.registrations)

    def add(self, registration: HookRegistration) -> None:
        self._validate_registration(registration)
        self.registrations.append(registration)

    def _validate_registration(self, registration: HookRegistration) -> None:
        spec = phase_spec(registration.phase)
        allowed_capabilities = {HookCapability.OBSERVE_ONLY}
        if spec.allow_transform:
            allowed_capabilities.add(HookCapability.TRANSFORM)
        if spec.allow_decision_support:
            allowed_capabilities.add(HookCapability.DECISION_SUPPORT)

        if registration.capability not in allowed_capabilities:
            raise ValueError(
                "Unsupported hook capability for "
                f"{registration.phase.value}: {registration.handler_name}"
            )
        if registration.critical and not spec.allow_critical:
            raise ValueError(
                "Critical hooks are allowed only in early phases: "
                f"{registration.phase.value}: {registration.handler_name}"
            )

    def _merge_hook_result(
        self,
        phase: HookPhase,
        capability: HookCapability,
        current: dict[str, Any],
        result: object,
    ) -> dict[str, Any]:
        if not isinstance(result, dict):
            return current

        spec = phase_spec(phase)
        if capability is HookCapability.TRANSFORM:
            allowed = spec.transform_fields
        elif capability is HookCapability.DECISION_SUPPORT:
            allowed = spec.decision_support_fields
        else:
            return current

        changed_keys = {
            key
            for key, value in result.items()
            if key not in current or current.get(key) != value
        }
        illegal_keys = sorted(changed_keys - allowed)
        if illegal_keys:
            raise ValueError(
                "Hook result contains fields outside the phase allowlist for "
                f"{phase.value}: {', '.join(illegal_keys)}"
            )

        merged = dict(current)
        for key in allowed:
            if key in result:
                merged[key] = result[key]
        return merged

    async def _record_hook_failed(
        self,
        payload: dict[str, Any],
        *,
        phase: HookPhase,
        registration: HookRegistration,
        error: Exception,
    ) -> None:
        context = payload.get("context")
        if context is None or not hasattr(context, "session_runtime"):
            return
        session_runtime = getattr(context, "session_runtime", None)
        if session_runtime is None:
            return

        from agiwo.agent.runtime.state_writer import (  # noqa: PLC0415
            RunStateWriter,
        )

        traceback_text = "".join(
            traceback_lib.format_exception(type(error), error, error.__traceback__)
        ).strip()
        writer = RunStateWriter(context)
        await writer.record_hook_failed(
            phase=phase.value,
            handler_name=registration.handler_name,
            critical=registration.critical,
            error=str(error),
            traceback=traceback_text or None,
        )

    async def _dispatch(
        self,
        phase: HookPhase,
        payload: dict[str, Any],
        *,
        allow_transform: bool,
    ) -> dict[str, Any]:
        current = dict(payload)
        for registration in self.for_phase(phase):
            try:
                result = await registration.handler(dict(current))
            except Exception as error:  # noqa: BLE001 - hook isolation boundary
                logger.warning(
                    "hook_handler_failed",
                    phase=phase.value,
                    handler_name=registration.handler_name,
                    critical=registration.critical,
                    error=str(error),
                )
                await self._record_hook_failed(
                    payload,
                    phase=phase,
                    registration=registration,
                    error=error,
                )
                if registration.critical:
                    raise
                continue

            if not allow_transform:
                continue

            if registration.capability in {
                HookCapability.TRANSFORM,
                HookCapability.DECISION_SUPPORT,
            }:
                current = self._merge_hook_result(
                    phase,
                    registration.capability,
                    current,
                    result,
                )
        return current

    async def before_run(
        self,
        user_input: UserInput | None,
        context: object,
    ) -> str | None:
        payload = await self._dispatch(
            HookPhase.PREPARE,
            {"user_input": user_input, "context": context, "prelude_text": None},
            allow_transform=True,
        )
        result = payload.get("prelude_text")
        return result if isinstance(result, str) else None

    async def memory_retrieve(
        self,
        user_input: UserInput | None,
        context: object,
    ) -> list[MemoryRecord]:
        payload = await self._dispatch(
            HookPhase.ASSEMBLE_CONTEXT,
            {
                "user_input": user_input,
                "context": context,
                "memories": [],
                "context_additions": [],
            },
            allow_transform=True,
        )
        memories = payload.get("memories", [])
        return memories if isinstance(memories, list) else []

    async def before_llm_call(
        self,
        messages: list[dict[str, Any]],
        context: object | None = None,
    ) -> list[dict[str, Any]] | None:
        payload = await self._dispatch(
            HookPhase.BEFORE_LLM,
            {
                "messages": messages,
                "context": context,
                "model_settings_override": None,
                "llm_advice": None,
            },
            allow_transform=True,
        )
        modified = payload.get("messages")
        return modified if isinstance(modified, list) else None

    async def after_llm_call(self, step: object, context: object | None = None) -> None:
        await self._dispatch(
            HookPhase.AFTER_LLM,
            {"step": step, "context": context},
            allow_transform=False,
        )

    async def before_tool_call(
        self,
        tool_call_id: str,
        tool_name: str,
        parameters: dict[str, Any],
        context: object | None = None,
    ) -> dict[str, Any] | None:
        payload = await self._dispatch(
            HookPhase.BEFORE_TOOL_CALL,
            {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "parameters": dict(parameters),
                "context": context,
                "tool_advice": None,
            },
            allow_transform=True,
        )
        modified = payload.get("parameters")
        return modified if isinstance(modified, dict) else None

    async def after_tool_call(
        self,
        tool_call_id: str,
        tool_name: str,
        parameters: dict[str, Any],
        result: object,
        context: object | None = None,
    ) -> None:
        await self._dispatch(
            HookPhase.AFTER_TOOL_CALL,
            {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "parameters": dict(parameters),
                "result": result,
                "context": context,
            },
            allow_transform=False,
        )

    async def before_review(
        self,
        *,
        trigger_reason: str,
        milestone: object | None,
        step_count: int,
        context: object | None = None,
    ) -> str | None:
        payload = await self._dispatch(
            HookPhase.BEFORE_REVIEW,
            {
                "trigger_reason": trigger_reason,
                "milestone": milestone,
                "step_count": step_count,
                "context": context,
                "review_advice": None,
            },
            allow_transform=True,
        )
        advice = payload.get("review_advice")
        return advice if isinstance(advice, str) else None

    async def after_run(self, result: object, context: object) -> None:
        await self._dispatch(
            HookPhase.RUN_FINALIZED,
            {"result": result, "context": context},
            allow_transform=False,
        )

    async def memory_write(
        self,
        user_input: UserInput,
        result: object,
        context: object,
    ) -> None:
        await self._dispatch(
            HookPhase.MEMORY_PERSIST,
            {"user_input": user_input, "result": result, "context": context},
            allow_transform=False,
        )

    async def on_step(self, step: object, context: object | None = None) -> None:
        await self._dispatch(
            HookPhase.AFTER_STEP_COMMIT,
            {"step": step, "context": context},
            allow_transform=False,
        )

    async def compaction_failed(
        self,
        run_id: str,
        error: str,
        failure_count: int,
        context: object | None = None,
    ) -> None:
        await self._dispatch(
            HookPhase.COMPACTION_FAILED,
            {
                "run_id": run_id,
                "error": error,
                "failure_count": failure_count,
                "context": context,
            },
            allow_transform=False,
        )


def observe(
    phase: HookPhase,
    handler_name: str,
    handler: PhaseHook,
    *,
    group: HookGroup = HookGroup.USER,
    order: int = 100,
    critical: bool = False,
) -> HookRegistration:
    return HookRegistration(
        phase=phase,
        capability=HookCapability.OBSERVE_ONLY,
        handler_name=handler_name,
        handler=handler,
        group=group,
        order=order,
        critical=critical,
    )


def transform(
    phase: HookPhase,
    handler_name: str,
    handler: PhaseHook,
    *,
    group: HookGroup = HookGroup.USER,
    order: int = 100,
    critical: bool = False,
) -> HookRegistration:
    return HookRegistration(
        phase=phase,
        capability=HookCapability.TRANSFORM,
        handler_name=handler_name,
        handler=handler,
        group=group,
        order=order,
        critical=critical,
    )


def decision_support(
    phase: HookPhase,
    handler_name: str,
    handler: PhaseHook,
    *,
    group: HookGroup = HookGroup.USER,
    order: int = 100,
    critical: bool = False,
) -> HookRegistration:
    return HookRegistration(
        phase=phase,
        capability=HookCapability.DECISION_SUPPORT,
        handler_name=handler_name,
        handler=handler,
        group=group,
        order=order,
        critical=critical,
    )


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 0.9

    a_words = set(a.split())
    b_words = set(b.split())
    if not a_words or not b_words:
        return 0.0

    intersection = a_words & b_words
    union = a_words | b_words
    return len(intersection) / len(union)


def _compact_memory_query(query: str, *, max_chars: int = 400) -> str:
    """Shrink assignment-template blobs into an embedding-safe retrieval query."""
    text = (query or "").strip()
    if not text:
        return ""
    if "Current goal:" in text:
        text = text.split("Current goal:", 1)[1].strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def filter_relevant_memories(
    messages: list[Mapping[str, object]],
    memories: list[MemoryRecord],
) -> list[MemoryRecord]:
    if not memories:
        return []

    min_relevance_score = 0.5
    similarity_threshold = 0.8

    existing_texts: list[str] = [
        content
        for message in messages[:-1]
        if isinstance(content := message.get("content"), str)
    ]

    def _is_similar_to_history(content: str) -> bool:
        content_lower = content.lower()
        for text in existing_texts:
            if _text_similarity(content_lower, text.lower()) > similarity_threshold:
                return True
        return False

    filtered: list[MemoryRecord] = []
    seen_contents: set[str] = set()

    for memory in sorted(
        [m for m in memories if m.relevance_score is not None],
        key=lambda m: m.relevance_score or 0,
        reverse=True,
    ):
        if memory.relevance_score < min_relevance_score:
            continue

        content_normalized = memory.content.strip()
        if content_normalized in seen_contents:
            continue
        seen_contents.add(content_normalized)

        if _is_similar_to_history(content_normalized):
            continue

        filtered.append(memory)

    return filtered


class DefaultMemoryHook:
    """Default memory hook implementation using WorkspaceMemoryService."""

    def __init__(
        self,
        *,
        embedding_provider: str | None = None,
        top_k: int | None = None,
        root_path: str | None = None,
    ) -> None:
        self._top_k = top_k if top_k is not None else get_settings().memory_top_k
        self._memory_service = WorkspaceMemoryService(
            root_path=root_path,
            embedding_provider=embedding_provider,
        )

    def _resolve_workspace(self, context: MemoryHookContext):
        workspace = self._memory_service.resolve_workspace(
            agent_name=getattr(context, "agent_name", None),
            agent_id=getattr(context, "agent_id", None),
        )
        if workspace is None:
            return None
        return workspace.workspace

    async def retrieve_memories(
        self, user_input: UserInput, context: MemoryHookContext
    ) -> list[MemoryRecord]:
        query = _compact_memory_query(UserMessage.from_value(user_input).extract_text())
        if not query or len(query.strip()) < 3:
            return []

        try:
            workspace, results = await self._memory_service.search(
                agent_name=context.agent_name,
                agent_id=context.agent_id,
                query=query,
                top_k=self._top_k,
            )
        except Exception as error:  # noqa: BLE001 - memory retrieval boundary
            logger.warning("memory_retrieve_error", error=str(error), query=query[:50])
            return []

        if workspace is None:
            logger.debug("memory_retrieve_no_workspace", agent_id=context.agent_id)
            return []

        if not results:
            return []

        records: list[MemoryRecord] = []
        for r in results:
            content = f"[{r.path}:{r.start_line}-{r.end_line}] {r.text}"
            records.append(
                MemoryRecord(
                    content=content,
                    relevance_score=r.score,
                    source=r.path,
                    metadata={
                        "chunk_id": r.chunk_id,
                        "start_line": r.start_line,
                        "end_line": r.end_line,
                        "vector_score": r.vector_score,
                        "bm25_score": r.bm25_score,
                    },
                )
            )

        logger.debug(
            "memory_retrieved",
            query=query[:50],
            count=len(records),
            agent_id=context.agent_id,
        )
        return records


__all__ = [
    "DefaultMemoryHook",
    "HookCapability",
    "HookGroup",
    "HookPhase",
    "HookRegistration",
    "HookRegistry",
    "MemoryHookContext",
    "PHASE_SPECS",
    "PhaseHook",
    "PhaseSpec",
    "decision_support",
    "filter_relevant_memories",
    "observe",
    "phase_spec",
    "transform",
]
