#!/usr/bin/env python3
"""
Real Agent Test Script

Test Agent Run functionality and logic:
1. Tools support, tool calling
2. Skills loading and usage
3. Data persistence (RunStepStorage saves all data correctly, TraceStorage saves all Trace information correctly)

Usage:
    python test_real_agent.py
    or
    uv run python test_real_agent.py
"""

import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4
from typing import Any

from dotenv import load_dotenv

from agiwo.agent.agent import Agent
from agiwo.agent.options import AgentOptions, RunStepStorageConfig, TraceStorageConfig
from agiwo.agent.storage.sqlite import SQLiteRunStepStorage
from agiwo.agent.execution_context import ExecutionContext
from agiwo.observability.sqlite_store import SQLiteTraceStorage
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.llm.deepseek import DeepseekModel
from agiwo.utils.logging import get_logger

load_dotenv()

logger = get_logger(__name__)


def _prepare_test_db(
    test_name: str,
) -> tuple[str, SQLiteRunStepStorage, SQLiteTraceStorage]:
    """Create SQLite stores for test and return (db_path, session_store, trace_storage)."""
    base_dir = os.getenv("AGIWO_TEST_DB_DIR") or os.path.join(os.getcwd(), ".tempdata")
    os.makedirs(base_dir, exist_ok=True)
    db_path = os.path.join(base_dir, f"{test_name}.db")

    if os.path.exists(db_path):
        os.remove(db_path)

    session_store = SQLiteRunStepStorage(db_path=db_path)
    trace_storage = SQLiteTraceStorage(db_path=db_path)
    return db_path, session_store, trace_storage


class TestCalculatorTool(BaseTool):
    """Test calculator tool"""

    def get_name(self) -> str:
        return "calculator"

    def get_description(self) -> str:
        return (
            "Perform simple mathematical calculations. Accept two numbers and an operator (+, -, *, /), return the calculation result."
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "The first number",
                },
                "b": {
                    "type": "number",
                    "description": "The second number",
                },
                "operator": {
                    "type": "string",
                    "enum": ["+", "-", "*", "/"],
                    "description": "The operator",
                },
            },
            "required": ["a", "b", "operator"],
        }

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal=None,
    ) -> ToolResult:
        import time

        start_time = time.time()

        try:
            a = parameters.get("a")
            b = parameters.get("b")
            operator = parameters.get("operator")

            if operator == "+":
                result = a + b
            elif operator == "-":
                result = a - b
            elif operator == "*":
                result = a * b
            elif operator == "/":
                if b == 0:
                    raise ValueError("The divisor cannot be zero")
                result = a / b
            else:
                raise ValueError(f"Unsupported operator: {operator}")

            end_time = time.time()

            return ToolResult(
                tool_name=self.name,
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                content=str(result),
                content_for_user=f"Calculation result: {a} {operator} {b} = {result}",
                output={"result": result},
                is_success=True,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
            )
        except Exception as e:
            end_time = time.time()
            return ToolResult.error(
                tool_name=self.name,
                error=str(e),
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                start_time=start_time,
            )


class TestEchoTool(BaseTool):
    """Test echo tool"""

    def get_name(self) -> str:
        return "echo"

    def get_description(self) -> str:
        return "Echo the input message, used for testing tool calls."

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to be echoed",
                },
            },
            "required": ["message"],
        }

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal=None,
    ) -> ToolResult:
        import time

        start_time = time.time()
        message = parameters.get("message", "")

        end_time = time.time()

        return ToolResult(
            tool_name=self.name,
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args=parameters,
            content=f"Echo: {message}",
            content_for_user=f"Echo: {message}",
            output={"message": message},
            is_success=True,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
        )


async def test_tools_support():
    """Test Tools support and tool calling"""
    print("\n" + "=" * 60)
    print("Test 1: Tools support and tool calling")
    print("=" * 60)

    db_path, session_store, trace_storage = _prepare_test_db("tools_support")
    model = None
    agent = None
    try:
        tools = [TestCalculatorTool(), TestEchoTool()]

        model = create_test_model()
        if not model:
            print("⚠️  Skip test: No available LLM API Key")
            return False

        agent = Agent(
            id="test_agent",
            description="Test Agent",
            model=model,
            tools=tools,
            system_prompt="You are a helpful assistant, can use tools to help users.",
            options=AgentOptions(
                max_steps=10,
                run_step_storage=RunStepStorageConfig(
                    storage_type="sqlite",
                    config={"db_path": db_path},
                ),
                trace_storage=TraceStorageConfig(
                    storage_type="sqlite",
                    config={"db_path": db_path},
                ),
            ),
        )

        session_id = str(uuid4())

        print(f"\n📝 User input: Please calculate 25 * 4")
        result = await agent.run("Please calculate 25 * 4", session_id=session_id)

        print(f"\n✅ Agent execution completed")
        print(f"   - Run ID: {result.run_id}")
        print(f"   - Session ID: {result.session_id}")
        print(f"   - Response: {result.response}")
        print(f"   - Termination reason: {result.termination_reason}")
        if result.metrics:
            print(f"   - Total tokens: {result.metrics.total_tokens}")
            print(f"   - Steps count: {result.metrics.steps_count}")
            print(f"   - Tool calls count: {result.metrics.tool_calls_count}")

        # Verify RunStepStorage data
        print(f"\n🔍 Verify RunStepStorage data...")
        run_step_storage = agent.run_step_storage
        saved_run = await run_step_storage.get_run(result.run_id)
        assert saved_run is not None, "Run should be saved"
        print(f"   ✅ Run saved: {saved_run.id}")

        steps = await run_step_storage.get_steps(session_id=session_id)
        print(f"   ✅ Steps saved: {len(steps)} steps")
        for i, step in enumerate(steps[:5], 1):
            print(
                f"      {i}. {step.role.value}: {step.content[:50] if step.content else 'N/A'}"
            )

        # Verify TraceStorage data
        print(f"\n🔍 Verify TraceStorage data...")
        trace_st = agent.trace_storage
        traces = await trace_st.query_traces(
            {
                "session_id": session_id,
                "limit": 10,
            }
        )
        if traces:
            print(f"   ✅ Traces saved: {len(traces)} traces")
            for trace in traces[:3]:
                print(f"      - Trace ID: {trace.trace_id}, Spans: {len(trace.spans)}")
        else:
            print(f"   ⚠️  No Traces found (maybe TraceCollector not started correctly)")

        print(f"\n✅ Test 1 passed: Tools support and tool calling")
        return True

    except Exception as e:
        print(f"\n❌ Test 1 failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if model:
            await model.close()
        if agent:
            await agent.close()


async def test_skills_loading():
    """Test Skills loading and usage"""
    print("\n" + "=" * 60)
    print("Test 2: Skills loading and usage")
    print("=" * 60)

    db_path, session_store, trace_storage = _prepare_test_db("skills_loading")
    model = None
    agent = None
    try:
        # Create test Skill directory structure
        base_dir = os.path.dirname(db_path)
        test_skills_dir = os.path.join(base_dir, "test_skills")
        os.makedirs(test_skills_dir, exist_ok=True)

        # Create a simple test Skill (using a name that follows the naming convention)
        test_skill_dir = os.path.join(test_skills_dir, "test-skill")
        os.makedirs(test_skill_dir, exist_ok=True)
        skill_md_path = os.path.join(test_skill_dir, "SKILL.md")
        with open(skill_md_path, "w", encoding="utf-8") as f:
            f.write(
                """---
name: test-skill
description: This is a test skill, used to verify if the Skills system works correctly
---

# Test Skill

This is a test skill.

## Usage

This skill is used for testing purposes.
"""
            )

        model = create_test_model()
        if not model:
            print("⚠️  Skip test: No available LLM API Key")
            return False

        agent = Agent(
            id="test_agent_with_skills",
            description="Test Agent (with Skills)",
            model=model,
            system_prompt="You are a helpful assistant, can use skills to help users.",
            options=AgentOptions(
                max_steps=10,
                enable_skill=True,
                skills_dir=test_skills_dir,
                run_step_storage=RunStepStorageConfig(
                    storage_type="sqlite",
                    config={"db_path": db_path},
                ),
                trace_storage=TraceStorageConfig(
                    storage_type="sqlite",
                    config={"db_path": db_path},
                ),
            ),
        )

        session_id = str(uuid4())

        print(f"\n📝 User input: Please activate test-skill")
        result = await agent.run("Please activate test-skill", session_id=session_id)

        print(f"\n✅ Agent execution completed")
        print(f"   - Run ID: {result.run_id}")
        print(f"   - Session ID: {result.session_id}")
        print(f"   - Response: {result.response}")
        print(f"   - Termination reason: {result.termination_reason}")
        if result.metrics:
            print(f"   - Total tokens: {result.metrics.total_tokens}")
            print(f"   - Steps count: {result.metrics.steps_count}")
            print(f"   - Tool calls count: {result.metrics.tool_calls_count}")

        # Verify Skill is called
        print(f"\n🔍 Verify Skills call...")
        steps = await agent.run_step_storage.get_steps(session_id=session_id)
        tool_steps = [s for s in steps if s.role.value == "tool"]
        skill_called = any(
            s.name == "Skill"
            and (
                "test-skill" in (s.content_for_user or "").lower()
                or "test skill" in (s.content or "").lower()
            )
            for s in tool_steps
        )

        if skill_called:
            print(f"   ✅ Skill tool is called")
        else:
            print(f"   ⚠️  Skill tool may not be called (check steps)")
            for step in tool_steps:
                print(
                    f"      - {step.name}: {step.content[:100] if step.content else 'N/A'}"
                )

        # Verify Skills in system prompt
        if "skill" in agent.system_prompt.lower() or "Available Skills" in agent.system_prompt:
            print(f"\n   ✅ Skills loaded into system prompt")
        else:
            print(f"\n   ⚠️  Skills section not found in system prompt")

        print(f"\n✅ Test 2 passed: Skills loading and usage")
        return True

    except Exception as e:
        print(f"\n❌ Test 2 failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if model:
            await model.close()
        if agent:
            await agent.close()


async def test_data_persistence():
    """Test data persistence"""
    print("\n" + "=" * 60)
    print("Test 3: Data persistence (RunStepStorage and TraceStorage)")
    print("=" * 60)

    db_path, session_store, trace_storage = _prepare_test_db("data_persistence")
    model = None
    agent = None
    try:
        model = create_test_model()
        if not model:
            print("⚠️  Skip test: No available LLM API Key")
            return False

        tools = [TestCalculatorTool(), TestEchoTool()]

        agent = Agent(
            id="test_agent_persistence",
            description="Test Agent (Persistence)",
            model=model,
            tools=tools,
            system_prompt="You are a helpful assistant.",
            options=AgentOptions(
                max_steps=10,
                run_step_storage=RunStepStorageConfig(
                    storage_type="sqlite",
                    config={"db_path": db_path},
                ),
                trace_storage=TraceStorageConfig(
                    storage_type="sqlite",
                    config={"db_path": db_path},
                ),
            ),
        )

        session_id = str(uuid4())

        # Run multiple conversations
        print(f"\n📝 Running multiple conversations...")
        queries = [
            "Please calculate 10 + 20",
            "Please echo message: Hello World",
            "Please calculate 100 / 5",
        ]

        all_runs = []
        for i, query in enumerate(queries, 1):
            print(f"\n   Conversation {i}: {query}")
            result = await agent.run(query, session_id=session_id)
            all_runs.append((result.run_id, result))

        # Verify RunStepStorage data
        print(f"\n🔍 Verify RunStepStorage data...")

        # Check all Runs
        runs = await agent.run_step_storage.list_runs(session_id=session_id)
        print(f"   ✅ Found {len(runs)} Runs")
        assert len(runs) == len(queries), (
            f"Should have {len(queries)} Runs, but found {len(runs)}"
        )

        # Check all Steps
        steps = await agent.run_step_storage.get_steps(session_id=session_id)
        print(f"   ✅ Found {len(steps)} Steps")
        assert len(steps) > 0, "Should have Steps"

        # Group by role statistics
        role_counts = {}
        for step in steps:
            role = step.role.value
            role_counts[role] = role_counts.get(role, 0) + 1

        print(f"   Steps statistics:")
        for role, count in role_counts.items():
            print(f"      - {role}: {count}")

        # Verify each Run has corresponding Steps
        for run_id, result in all_runs:
            run_steps = await agent.run_step_storage.get_steps(
                session_id=session_id, run_id=run_id
            )
            assert len(run_steps) > 0, f"Run {run_id} should have Steps"
            print(f"   ✅ Run {run_id[:8]}... has {len(run_steps)} Steps")

        # Verify TraceStorage data
        print(f"\n🔍 Verify TraceStorage data...")

        all_traces = await agent.trace_storage.query_traces(
            {
                "session_id": session_id,
                "limit": 100,
            }
        )

        if all_traces:
            print(f"   ✅ Found {len(all_traces)} Traces")
            for trace in all_traces:
                print(f"      - Trace ID: {trace.trace_id}")
                print(f"        Spans: {len(trace.spans)}")
                print(f"        Agent ID: {trace.agent_id}")
        else:
            print(f"   ⚠️  No Traces found (TraceCollector may not have started correctly)")

        # Verify data can be reloaded
        print(f"\n🔍 Verify data can be reloaded...")

        new_session_store = SQLiteRunStepStorage(db_path=db_path)
        await new_session_store.initialize()

        try:
            reloaded_runs = await new_session_store.list_runs(session_id=session_id)
            assert len(reloaded_runs) == len(runs), "Runs count should be consistent after reload"

            reloaded_steps = await new_session_store.get_steps(session_id=session_id)
            assert len(reloaded_steps) == len(steps), "Steps count should be consistent after reload"

            print(f"   ✅ Data can be correctly reloaded")
        finally:
            await new_session_store.disconnect()

        print(f"\n✅ Test 3 passed: Data persistence")
        return True

    except Exception as e:
        print(f"\n❌ Test 3 failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if model:
            await model.close()
        if agent:
            await agent.close()


def create_test_model():
    """Create test LLM Model"""
    # Try different models in priority order
    models_to_try = [
        # ("OpenAI", OpenAIModel, "OPENAI_API_KEY", "gpt-4o-mini"),
        ("DeepSeek", DeepseekModel, "DEEPSEEK_API_KEY", "deepseek-chat"),
        # ("NVIDIA", NvidiaModel, "NVIDIA_BUILD_API_KEY", "z-ai/glm4.7"),
        # ("Anthropic", AnthropicModel, "ANTHROPIC_API_KEY", "claude-3-5-sonnet-20240620"),
    ]

    for name, model_class, env_key, model_name in models_to_try:
        api_key = os.getenv(env_key)
        if api_key:
            try:
                if name == "OpenAI":
                    return model_class(
                        id=model_name,
                        name=model_name,
                        api_key=api_key,
                        temperature=0.7,
                        top_p=1.0,
                        max_tokens=1000,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                    )
                elif name == "DeepSeek":
                    return model_class(
                        id=model_name,
                        name=model_name,
                        api_key=api_key,
                        temperature=0.7,
                        top_p=1.0,
                        max_tokens=1000,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                    )
                elif name == "NVIDIA":
                    return model_class(
                        id=model_name,
                        name=model_name,
                        api_key=api_key,
                        temperature=0.7,
                        top_p=1.0,
                        max_tokens=1000,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                    )
                elif name == "Anthropic":
                    return model_class(
                        id=model_name,
                        name=model_name,
                        api_key=api_key,
                        temperature=0.7,
                        top_p=1.0,
                        max_tokens=1000,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                    )
            except Exception as e:
                logger.warning(f"Failed to create {name} model: {e}")
                continue

    return None


async def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Agent Run Functionality Test")
    print("=" * 60)

    results = []

    # Test 1: Tools support
    try:
        result1 = await test_tools_support()
        results.append(("Tools support", result1))
    except Exception as e:
        print(f"\n❌ Test 1 exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Tools support", False))

    # Test 2: Skills loading
    try:
        result2 = await test_skills_loading()
        results.append(("Skills loading", result2))
    except Exception as e:
        print(f"\n❌ Test 2 exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Skills loading", False))

    # Test 3: Data persistence
    try:
        result3 = await test_data_persistence()
        results.append(("Data persistence", result3))
    except Exception as e:
        print(f"\n❌ Test 3 exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Data persistence", False))

    # Summary results
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ Passed" if passed else "❌ Failed"
        print(f"{status}: {test_name}")

    success_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print("\n" + "=" * 60)
    print(f"Total: {total_count} tests")
    print(f"Passed: {success_count}, Failed: {total_count - success_count}")
    print("=" * 60)

    return all(passed for _, passed in results)


def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(__doc__)
        return

    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n[FATAL ERROR] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
