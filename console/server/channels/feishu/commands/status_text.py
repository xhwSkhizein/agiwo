"""Shared scheduler status labels for Feishu commands."""

from agiwo.scheduler.models import AgentStateStatus

_STATUS_LABELS = {
    AgentStateStatus.PENDING: "等待中",
    AgentStateStatus.RUNNING: "运行中",
    AgentStateStatus.WAITING: "等待唤醒",
    AgentStateStatus.IDLE: "空闲",
    AgentStateStatus.QUEUED: "已排队",
    AgentStateStatus.COMPLETED: "已完成",
    AgentStateStatus.FAILED: "已失败",
}

_STATUS_EMOJI_MAP = {
    "运行中": "🟢",
    "等待中": "⏳",
    "队列中": "📋",
    "闲置": "⚪",
    "已完成": "✅",
    "失败": "❌",
    "取消": "🚫",
    "未启动": "⚪",
}


def format_scheduler_status(status: AgentStateStatus) -> str:
    """Return the user-facing Feishu label for a scheduler status."""
    return _STATUS_LABELS.get(status, status.value)


def status_to_emoji(status: str) -> str:
    """Convert scheduler status text to an emoji indicator."""
    for key, emoji in _STATUS_EMOJI_MAP.items():
        if key in status:
            return emoji
    return "⚪"


__all__ = ["format_scheduler_status", "status_to_emoji"]
