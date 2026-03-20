"""Dynamic consent notifier injection for Console channels."""

from agiwo.agent.agent import Agent
from agiwo.agent.tool_auth.notifier import ToolConsentNotifier
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


def inject_consent_notifier(agent: Agent, notifier: ToolConsentNotifier) -> None:
    """Dynamically inject consent notifier into agent's authorization runtime.
    
    This is necessary because notifier needs runtime context (chat_id, session_id)
    which is only available when agent execution starts, not at agent build time.
    
    Args:
        agent: Agent instance to inject notifier into
        notifier: Channel-specific consent notifier implementation
    """
    try:
        # Access internal structure to inject notifier
        # This is a workaround for architectural limitation
        if hasattr(agent, "_definition_runtime"):
            definition_runtime = agent._definition_runtime
            if hasattr(definition_runtime, "_executor"):
                executor = definition_runtime._executor
                if hasattr(executor, "tool_runtime"):
                    tool_runtime = executor.tool_runtime
                    if hasattr(tool_runtime, "auth_runtime"):
                        auth_runtime = tool_runtime.auth_runtime
                        if hasattr(auth_runtime, "_notifier"):
                            auth_runtime._notifier = notifier
                            logger.debug(
                                "consent_notifier_injected",
                                agent_id=agent.id,
                                notifier_type=type(notifier).__name__,
                            )
                            return
        
        logger.warning(
            "consent_notifier_injection_failed",
            agent_id=agent.id,
            reason="unable_to_access_auth_runtime",
        )
    except Exception as error:
        logger.error(
            "consent_notifier_injection_error",
            agent_id=agent.id,
            error=str(error),
            exc_info=True,
        )


__all__ = ["inject_consent_notifier"]
