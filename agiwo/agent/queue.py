"""Unified MainAgent loop queue vocabulary (ADR 0049 / CONTEXT)."""

from dataclasses import dataclass
from enum import Enum

from agiwo.agent.models.input import UserMessage


class QueueItemKind(str, Enum):
    """Kinds of items on the MainAgent unified loop queue."""

    USER_INPUT = "user_input"
    GATE_FEEDBACK = "gate_feedback"
    WORKER_REPORT = "worker_report"


@dataclass
class QueueItem:
    """One pending item on the MainAgent unified loop queue."""

    kind: QueueItemKind
    message: UserMessage | None = None
    text: str | None = None
    created_at: float = 0.0
    run_id: str | None = None
