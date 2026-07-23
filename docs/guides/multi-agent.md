# Multi-Agent & Composition

Agiwo supports composing agents in two ways: **Agent-as-Tool** for simple delegation, and the **Scheduler** for complex orchestration.

## Agent-as-Tool

Wrap an agent as a tool that another agent can call:

```python
from agiwo.agent import Agent, AgentConfig
from agiwo.llm import OpenAIModel

# Specialist agent
researcher = Agent(
    AgentConfig(
        name="researcher",
        description="Researches topics thoroughly and returns summaries",
        system_prompt="You are a research specialist. Be thorough and cite sources.",
    ),
    model=OpenAIModel(name="gpt-5.4"),
)

# Orchestrator agent that can delegate to the researcher
orchestrator = Agent(
    AgentConfig(
        name="orchestrator",
        description="Coordinates research tasks",
        system_prompt="Delegate independent research tasks to the researcher tool.",
    ),
    model=OpenAIModel(name="gpt-5.4"),
    tools=[researcher.as_tool()],
)

result = await orchestrator.run("Compare Python and Rust for systems programming")
```

### How it works

1. The orchestrator's LLM sees `researcher` as a regular tool
2. When called, the researcher agent runs its own execution loop
3. The researcher's final response is returned as the tool result
4. The orchestrator continues reasoning with the researcher's output

### Child configuration

The supported public composition API is `Agent.as_tool()`. If you need child agents with different prompts or tool sets, create a separate `Agent` instance with the configuration you want and wrap that instance as a tool.

## Scheduler Orchestration

For long-running child delegation and waitset control, use the slim Scheduler. Session user chat still belongs to `MainAgent.accept`.

```python
from agiwo.scheduler import Scheduler

async with Scheduler() as scheduler:
    await scheduler.register_worker_parent(
        state_id=orchestrator.id,
        session_id="sess-1",
        agent=orchestrator,
    )

    await scheduler.enqueue_input(
        orchestrator.id,
        "Coordinate the research pipeline",
        agent=orchestrator,
    )
    await scheduler.wait_for(orchestrator.id)

    await scheduler.enqueue_input(
        orchestrator.id,
        "Now analyze the cost implications",
        agent=orchestrator,
    )
    await scheduler.wait_for(orchestrator.id)

    await scheduler.enqueue_input(
        orchestrator.id,
        "Focus on enterprise use cases",
        agent=orchestrator,
    )
    await scheduler.cancel(orchestrator.id)
```

### Scheduler Tools

Agents running under the scheduler automatically get orchestration tools:

```python
# Inside an agent's system prompt:
"""
You can use these tools to coordinate work:
- spawn_child_agent: Create a fresh child agent for a sub-task
- fork_child_agent: Fork the current agent into a child that inherits context
- query_spawned_agent: Check on a child's progress
- cancel_agent: Stop a child that's no longer needed
- list_agents: See all active children
- sleep_and_wait: Wait for a condition before continuing
- declare_milestones: Declare concrete sub-goals for the current task
- review_trajectory: Respond to a system review and provide an experience summary when off-track
"""
```

## Patterns

### Pipeline

Chain agents sequentially with `Agent.run()` or Session turns:

```python
research = await researcher.run("Gather facts about X")
analysis = await analyzer.run(f"Analyze these findings: {research.response}")
report = await writer.run(f"Write a report based on: {analysis.response}")
```

### Fan-out / Fan-in

Use parallel `Agent.run()` calls, or spawn children from a scheduler-managed supervisor:

```python
results = []
for topic in ["topic_a", "topic_b", "topic_c"]:
    result = await researcher.run(f"Research {topic}")
    results.append(result.response)

final = await synthesizer.run(f"Synthesize these findings: {results}")
```

### Supervisor Pattern

A persistent supervisor registered on the Scheduler coordinates transient workers via spawn tools; external operators enqueue the next assignment:

```python
async with Scheduler() as scheduler:
    await scheduler.register_worker_parent(
        state_id=supervisor_agent.id,
        session_id="sess-supervisor",
        agent=supervisor_agent,
    )
    await scheduler.enqueue_input(
        supervisor_agent.id,
        "Manage the data processing pipeline",
        agent=supervisor_agent,
    )
    await scheduler.enqueue_input(
        supervisor_agent.id,
        "Priority: process batch 42 first",
        agent=supervisor_agent,
    )
```

## Cleanup

Remember to close agents when done:

```python
await researcher.close()
await orchestrator.close()
```

The Scheduler handles child agent lifecycle — you only need to close agents you created directly.
