# Multi-Agent & Composition

Agiwo supports composing agents in two ways: **Agent-as-Tool** for simple delegation, and the **Scheduler** for complex orchestration.

## Agent-as-Tool

Wrap an agent as a tool that another agent can call:

```python
from agiwo import Agent, AgentConfig
from agiwo.agent.runtime_tools import as_tool
from agiwo.llm import OpenAIModel

# Specialist agent
researcher = Agent(
    AgentConfig(
        name="researcher",
        description="Researches topics thoroughly and returns summaries",
        system_prompt="You are a research specialist. Be thorough and cite sources.",
    ),
    model=OpenAIModel(id="gpt-4o", name="gpt-4o"),
)

# Orchestrator agent that can delegate to the researcher
orchestrator = Agent(
    AgentConfig(
        name="orchestrator",
        description="Coordinates research tasks",
        system_prompt="Delegate independent research tasks to the researcher tool.",
    ),
    model=OpenAIModel(id="gpt-4o", name="gpt-4o"),
    tools=[as_tool(researcher)],
)

result = await orchestrator.run("Compare Python and Rust for systems programming")
```

### How it works

1. The orchestrator's LLM sees `researcher` as a regular tool
2. When called, the researcher agent runs its own execution loop
3. The researcher's final response is returned as the tool result
4. The orchestrator continues reasoning with the researcher's output

### Deriving Child Specs

For fine-grained control over child agents:

```python
child_spec = agent.derive_child_spec(
    child_id="focused-researcher",
    instruction="Focus only on technical details",
    system_prompt_override="You only discuss technical implementation details.",
    exclude_tool_names={"bash"},  # Don't give bash to this child
)
```

## Scheduler Orchestration

For long-running, persistent multi-agent setups, use the Scheduler:

```python
from agiwo import Scheduler, SchedulerConfig

async with Scheduler() as scheduler:
    # Submit a persistent orchestrator
    orchestrator_id = await scheduler.submit(
        orchestrator,
        "Coordinate the research pipeline",
        persistent=True,  # Stays alive after completion
    )

    # Feed more input later
    await scheduler.enqueue_input(orchestrator_id, "Now analyze the cost implications")

    # Stream events
    async for event in scheduler.stream(
        "Update the analysis",
        state_id=orchestrator_id,
    ):
        if event.type == "step_delta" and event.delta.content:
            print(event.delta.content, end="", flush=True)

    # Steer the agent
    await scheduler.steer(orchestrator_id, "Focus on enterprise use cases")

    # Cancel when done
    await scheduler.cancel(orchestrator_id)
```

### Scheduler Tools

Agents running under the scheduler automatically get orchestration tools:

```python
# Inside an agent's system prompt:
"""
You can use these tools to coordinate work:
- spawn_agent: Create a child agent for a sub-task
- query_spawned_agent: Check on a child's progress
- cancel_agent: Stop a child that's no longer needed
- list_agents: See all active children
- sleep_and_wait: Wait for a condition before continuing
"""
```

## Patterns

### Pipeline

Chain agents sequentially:

```python
async with Scheduler() as scheduler:
    # Step 1: Research
    research_result = await scheduler.run(researcher, "Gather facts about X")

    # Step 2: Analyze using research output
    analysis_result = await scheduler.run(
        analyzer,
        f"Analyze these findings: {research_result.response}",
    )

    # Step 3: Write report
    report = await scheduler.run(
        writer,
        f"Write a report based on: {analysis_result.response}",
    )
```

### Fan-out / Fan-in

Spawn multiple agents in parallel, collect results:

```python
async with Scheduler() as scheduler:
    # Fan out
    ids = []
    for topic in ["topic_a", "topic_b", "topic_c"]:
        state_id = await scheduler.submit(researcher, f"Research {topic}")
        ids.append(state_id)

    # Fan in
    results = []
    for state_id in ids:
        result = await scheduler.wait_for(state_id)
        results.append(result.response)

    # Synthesize
    final = await scheduler.run(
        synthesizer,
        f"Synthesize these findings: {results}",
    )
```

### Supervisor Pattern

A persistent supervisor coordinates transient workers:

```python
async with Scheduler() as scheduler:
    supervisor_id = await scheduler.submit(
        supervisor_agent,
        "Manage the data processing pipeline",
        persistent=True,
    )

    # The supervisor uses spawn_agent internally to create workers
    # External input can steer the supervisor
    await scheduler.enqueue_input(supervisor_id, "Priority: process batch 42 first")
```

## Cleanup

Remember to close agents when done:

```python
await researcher.close()
await orchestrator.close()
```

The Scheduler handles child agent lifecycle — you only need to close agents you created directly.
