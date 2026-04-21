# Config Hot Reload Semantics

## Overview

Agiwo supports hot reloading agent configurations from the registry. However, configuration changes do not immediately affect running agents due to runtime state consistency requirements.

## When Config Changes Take Effect

Configuration changes take effect **only** when the agent is in one of the following states:

- `IDLE`: Agent is not currently running
- `COMPLETED`: Agent has finished successfully
- `FAILED`: Agent has failed

If the agent is in `RUNNING`, `QUEUED`, or `WAITING` state, configuration changes are **deferred** until the agent reaches one of the applicable states.

## Implementation Details

### SDK Layer

`scheduler/engine.py:rebind_agent()`:
```python
async def rebind_agent(self, state_id: str, agent: Agent) -> bool:
    state = await self._store.get_state(state_id)
    if state is not None and state.status not in (
        AgentStateStatus.IDLE,
        AgentStateStatus.COMPLETED,
        AgentStateStatus.FAILED,
    ):
        return False
    await self._ensure_root_runtime_agent(agent, state_id)
    return True
```

Returns `False` when configuration cannot be applied immediately.

### Console Layer

`console/server/services/runtime/agent_runtime_cache.py`:
```python
rebound = await self._scheduler.rebind_agent(session.id, agent)
if not rebound:
    logger.info(
        "runtime_agent_refresh_deferred",
        runtime_agent_id=session.id,
        base_agent_id=session.base_agent_id,
        reason="state_active",
    )
```

Logs an info when config refresh is deferred.

## User Experience Impact

When a user updates an agent configuration in the registry:

1. **If agent is IDLE/COMPLETED/FAILED**: Configuration takes effect immediately on next interaction
2. **If agent is RUNNING/QUEUED/WAITING**: Configuration is deferred until agent reaches terminal state

This behavior ensures runtime state consistency and prevents unexpected behavior during active execution.

## Future Improvements

Potential enhancements:
- API layer signal: Return `config_refreshed: boolean` in config update responses
- Frontend notification: Show toast when config changes are deferred
- Force reload: Add explicit "force reload" option for advanced users
