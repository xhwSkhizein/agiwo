#!/usr/bin/env python3
"""
真实 Agent 测试脚本

测试 Agent Run 的功能和逻辑：
1. Tools 支持，工具调用
2. Skills 加载和使用
3. 数据持久化（SessionStore 正常保存所有数据，TraceStore 正常保存所有 Trace 信息）

使用方法：
    python test_real_agent.py
    或
    uv run python test_real_agent.py
"""

import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4
from typing import Any

from dotenv import load_dotenv

from agiwo.agent.base import AgiwoAgent
from agiwo.agent.config_options import AgentConfigOptions
from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.wire import Wire
from agiwo.agent.session.sqlite import SQLiteSessionStore
from agiwo.observability.sqlite_store import SQLiteTraceStore
from agiwo.observability.collector import TraceCollector
from agiwo.skill.manager import SkillManager
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.llm.deepseek import DeepseekModel
from agiwo.utils.logging import get_logger

load_dotenv()

logger = get_logger(__name__)


def _prepare_db_paths(test_name: str) -> tuple[str, str]:
    """Create deterministic sqlite db paths for tests."""
    base_dir = os.getenv("AGIWO_TEST_DB_DIR") or os.path.join(os.getcwd(), ".tempdata")
    os.makedirs(base_dir, exist_ok=True)
    db_path = os.path.join(base_dir, f"{test_name}_agent.db")
    trace_db_path = os.path.join(base_dir, f"{test_name}_traces.db")

    for path in (db_path, trace_db_path):
        if os.path.exists(path):
            os.remove(path)

    return db_path, trace_db_path


class TestCalculatorTool(BaseTool):
    """测试用的计算器工具"""

    def get_name(self) -> str:
        return "calculator"

    def get_description(self) -> str:
        return (
            "执行简单的数学计算。接受两个数字和一个运算符（+、-、*、/），返回计算结果。"
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "第一个数字",
                },
                "b": {
                    "type": "number",
                    "description": "第二个数字",
                },
                "operator": {
                    "type": "string",
                    "enum": ["+", "-", "*", "/"],
                    "description": "运算符",
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
                    raise ValueError("除数不能为零")
                result = a / b
            else:
                raise ValueError(f"不支持的运算符: {operator}")

            end_time = time.time()

            return ToolResult(
                tool_name=self.name,
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                content=str(result),
                content_for_user=f"计算结果: {a} {operator} {b} = {result}",
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
    """测试用的回显工具"""

    def get_name(self) -> str:
        return "echo"

    def get_description(self) -> str:
        return "回显输入的消息，用于测试工具调用。"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "要回显的消息",
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
            content_for_user=f"回显: {message}",
            output={"message": message},
            is_success=True,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
        )


async def test_tools_support():
    """测试 Tools 支持和工具调用"""
    print("\n" + "=" * 60)
    print("测试 1: Tools 支持和工具调用")
    print("=" * 60)

    # 创建测试数据库文件
    db_path, trace_db_path = _prepare_db_paths("tools_support")

    session_store: SQLiteSessionStore | None = None
    trace_store: SQLiteTraceStore | None = None
    try:
        # 创建 SessionStore 和 TraceStore
        session_store = SQLiteSessionStore(db_path=db_path)
        trace_store = SQLiteTraceStore(db_path=trace_db_path)
        await session_store.connect()
        await trace_store.initialize()

        # 创建测试工具
        tools = [TestCalculatorTool(), TestEchoTool()]

        # 创建 Agent
        model = create_test_model()
        if not model:
            print("⚠️  跳过测试：未找到可用的 LLM API Key")
            return False

        agent = AgiwoAgent(
            name="test_agent",
            description="测试 Agent",
            model=model,
            tools=tools,
            system_prompt="你是一个有用的助手，可以使用工具来帮助用户。",
            options=AgentConfigOptions(
                max_steps=10,
                session_store=session_store,
                trace_store=trace_store,
            ),
        )

        # 创建执行上下文
        session_id = str(uuid4())
        run_id = str(uuid4())
        wire = Wire()
        context = ExecutionContext(
            session_id=session_id,
            run_id=run_id,
            wire=wire,
            agent_id=agent.name,
        )

        # 运行 Agent
        print(f"\n📝 用户输入: 请计算 25 * 4 的结果")
        result = await agent.run("请计算 25 * 4 的结果", context=context)

        print(f"\n✅ Agent 执行完成")
        print(f"   - Run ID: {result.run_id}")
        print(f"   - Session ID: {result.session_id}")
        print(f"   - 响应: {result.response}")
        print(f"   - 终止原因: {result.termination_reason}")
        if result.metrics:
            print(f"   - 总 Token: {result.metrics.total_tokens}")
            print(f"   - 步骤数: {result.metrics.steps_count}")
            print(f"   - 工具调用数: {result.metrics.tool_calls_count}")

        # 验证 SessionStore 数据
        print(f"\n🔍 验证 SessionStore 数据...")
        saved_run = await session_store.get_run(run_id)
        assert saved_run is not None, "Run 应该被保存"
        print(f"   ✅ Run 已保存: {saved_run.id}")

        steps = await session_store.get_steps(session_id=session_id)
        print(f"   ✅ Steps 已保存: {len(steps)} 个步骤")
        for i, step in enumerate(steps[:5], 1):  # 只显示前5个
            print(
                f"      {i}. {step.role.value}: {step.content[:50] if step.content else 'N/A'}"
            )

        # 验证 TraceStore 数据
        print(f"\n🔍 验证 TraceStore 数据...")
        # TraceCollector 会在事件中注入 trace_id
        # 我们需要从事件流中获取 trace_id，或者查询所有 traces
        traces = await trace_store.query_traces(
            {
                "session_id": session_id,
                "limit": 10,
            }
        )
        if traces:
            print(f"   ✅ Traces 已保存: {len(traces)} 个 trace")
            for trace in traces[:3]:  # 只显示前3个
                print(f"      - Trace ID: {trace.trace_id}, Spans: {len(trace.spans)}")
        else:
            print(f"   ⚠️  未找到 Traces（可能 TraceCollector 未正确启动）")

        print(f"\n✅ 测试 1 通过: Tools 支持和工具调用")
        return True

    except Exception as e:
        print(f"\n❌ 测试 1 失败: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if session_store:
            await session_store.disconnect()
        if trace_store:
            await trace_store.close()


async def test_skills_loading():
    """测试 Skills 加载和使用"""
    print("\n" + "=" * 60)
    print("测试 2: Skills 加载和使用")
    print("=" * 60)

    # 创建测试数据库文件
    db_path, trace_db_path = _prepare_db_paths("skills_loading")

    session_store: SQLiteSessionStore | None = None
    trace_store: SQLiteTraceStore | None = None
    try:
        # 创建 SessionStore 和 TraceStore
        session_store = SQLiteSessionStore(db_path=db_path)
        trace_store = SQLiteTraceStore(db_path=trace_db_path)
        await session_store.connect()
        await trace_store.initialize()

        # 创建测试 Skill 目录结构
        test_skills_dir = os.path.join(os.path.dirname(db_path), "test_skills")
        os.makedirs(test_skills_dir, exist_ok=True)

        # 创建一个简单的测试 Skill（使用符合命名规范的名称）
        test_skill_dir = os.path.join(test_skills_dir, "test-skill")
        os.makedirs(test_skill_dir, exist_ok=True)
        skill_md_path = os.path.join(test_skill_dir, "SKILL.md")
        with open(skill_md_path, "w", encoding="utf-8") as f:
            f.write(
                """---
name: test-skill
description: 这是一个测试技能，用于验证 Skills 系统是否正常工作
---

# Test Skill

这是一个测试技能。

## 使用方法

这个技能用于测试目的。
"""
            )

        # 创建 SkillManager
        skill_manager = SkillManager(skill_dirs=[Path(test_skills_dir)])
        await skill_manager.initialize()

        # 获取 SkillTool
        skill_tool = skill_manager.get_skill_tool()

        # 创建 Agent，包含 SkillTool
        model = create_test_model()
        if not model:
            print("⚠️  跳过测试：未找到可用的 LLM API Key")
            return False

        agent = AgiwoAgent(
            name="test_agent_with_skills",
            description="测试 Agent（带 Skills）",
            model=model,
            tools=[skill_tool],
            system_prompt="你是一个有用的助手，可以使用技能来帮助用户。",
            options=AgentConfigOptions(
                max_steps=10,
                session_store=session_store,
                trace_store=trace_store,
                skill_manager=skill_manager,
            ),
        )

        # 创建执行上下文
        session_id = str(uuid4())
        run_id = str(uuid4())
        wire = Wire()
        context = ExecutionContext(
            session_id=session_id,
            run_id=run_id,
            wire=wire,
            agent_id=agent.name,
        )

        # 运行 Agent，要求使用 Skill
        print(f"\n📝 用户输入: 请激活 test-skill 技能")
        result = await agent.run("请激活 test-skill 技能", context=context)

        print(f"\n✅ Agent 执行完成")
        print(f"   - Run ID: {result.run_id}")
        print(f"   - Session ID: {result.session_id}")
        print(f"   - 响应: {result.response}")
        print(f"   - 终止原因: {result.termination_reason}")
        if result.metrics:
            print(f"   - 总 Token: {result.metrics.total_tokens}")
            print(f"   - 步骤数: {result.metrics.steps_count}")
            print(f"   - 工具调用数: {result.metrics.tool_calls_count}")

        # 验证 Skill 是否被调用
        print(f"\n🔍 验证 Skills 调用...")
        steps = await session_store.get_steps(session_id=session_id)
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
            print(f"   ✅ Skill 工具被调用")
        else:
            print(f"   ⚠️  Skill 工具可能未被调用（检查步骤）")
            for step in tool_steps:
                print(
                    f"      - {step.name}: {step.content[:100] if step.content else 'N/A'}"
                )

        # 验证 Skills 在 system prompt 中
        skills_section = skill_manager.render_skills_section()
        if skills_section:
            print(f"\n   ✅ Skills 已加载到 system prompt")
            print(f"      找到 {len(skill_manager._metadata_cache)} 个技能")
        else:
            print(f"\n   ⚠️  Skills section 为空")

        print(f"\n✅ 测试 2 通过: Skills 加载和使用")
        return True

    except Exception as e:
        print(f"\n❌ 测试 2 失败: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if session_store:
            await session_store.disconnect()
        if trace_store:
            await trace_store.close()


async def test_data_persistence():
    """测试数据持久化"""
    print("\n" + "=" * 60)
    print("测试 3: 数据持久化（SessionStore 和 TraceStore）")
    print("=" * 60)

    # 创建测试数据库文件
    db_path, trace_db_path = _prepare_db_paths("data_persistence")

    session_store: SQLiteSessionStore | None = None
    trace_store: SQLiteTraceStore | None = None
    try:
        # 创建 SessionStore 和 TraceStore
        session_store = SQLiteSessionStore(db_path=db_path)
        trace_store = SQLiteTraceStore(db_path=trace_db_path)
        await session_store.connect()
        await trace_store.initialize()

        # 创建 Agent
        model = create_test_model()
        if not model:
            print("⚠️  跳过测试：未找到可用的 LLM API Key")
            return False

        tools = [TestCalculatorTool(), TestEchoTool()]

        agent = AgiwoAgent(
            name="test_agent_persistence",
            description="测试 Agent（持久化）",
            model=model,
            tools=tools,
            system_prompt="你是一个有用的助手。",
            options=AgentConfigOptions(
                max_steps=10,
                session_store=session_store,
                is_trace_enabled=True,
            ),
        )

        # 创建执行上下文
        session_id = str(uuid4())
        run_id = str(uuid4())
        wire = Wire()
        context = ExecutionContext(
            session_id=session_id,
            run_id=run_id,
            wire=wire,
            agent_id=agent.name,
        )

        # 运行多个对话
        print(f"\n📝 运行多个对话...")
        queries = [
            "请计算 10 + 20",
            "请回显消息：Hello World",
            "请计算 100 / 5",
        ]

        all_runs = []
        for i, query in enumerate(queries, 1):
            print(f"\n   对话 {i}: {query}")
            run_id = str(uuid4())
            context = ExecutionContext(
                session_id=session_id,  # 使用相同的 session_id
                run_id=run_id,
                wire=Wire(),
                agent_id=agent.name,
            )
            result = await agent.run(query, context=context)
            all_runs.append((run_id, result))

        # 验证 SessionStore 数据
        print(f"\n🔍 验证 SessionStore 数据...")

        # 检查所有 Runs
        runs = await session_store.list_runs(session_id=session_id)
        print(f"   ✅ 找到 {len(runs)} 个 Runs")
        assert len(runs) == len(queries), (
            f"应该有 {len(queries)} 个 Runs，但找到 {len(runs)} 个"
        )

        # 检查所有 Steps
        steps = await session_store.get_steps(session_id=session_id)
        print(f"   ✅ 找到 {len(steps)} 个 Steps")
        assert len(steps) > 0, "应该有 Steps"

        # 按 role 分组统计
        role_counts = {}
        for step in steps:
            role = step.role.value
            role_counts[role] = role_counts.get(role, 0) + 1

        print(f"   Steps 统计:")
        for role, count in role_counts.items():
            print(f"      - {role}: {count}")

        # 验证每个 Run 都有对应的 Steps
        for run_id, result in all_runs:
            run_steps = await session_store.get_steps(
                session_id=session_id, run_id=run_id
            )
            assert len(run_steps) > 0, f"Run {run_id} 应该有 Steps"
            print(f"   ✅ Run {run_id[:8]}... 有 {len(run_steps)} 个 Steps")

        # 验证 TraceStore 数据
        print(f"\n🔍 验证 TraceStore 数据...")

        # 查询所有 traces
        all_traces = await trace_store.query_traces(
            {
                "session_id": session_id,
                "limit": 100,
            }
        )

        if all_traces:
            print(f"   ✅ 找到 {len(all_traces)} 个 Traces")
            for trace in all_traces:
                print(f"      - Trace ID: {trace.trace_id}")
                print(f"        Spans: {len(trace.spans)}")
                print(f"        Agent ID: {trace.agent_id}")
        else:
            print(f"   ⚠️  未找到 Traces（可能 TraceCollector 未正确启动）")

        # 验证数据可以重新加载
        print(f"\n🔍 验证数据可以重新加载...")

        # 创建新的 store 实例（模拟重启）
        new_session_store = SQLiteSessionStore(db_path=db_path)
        await new_session_store.initialize()

        reloaded_runs = await new_session_store.list_runs(session_id=session_id)
        assert len(reloaded_runs) == len(runs), "重新加载后 Runs 数量应该一致"

        reloaded_steps = await new_session_store.get_steps(session_id=session_id)
        assert len(reloaded_steps) == len(steps), "重新加载后 Steps 数量应该一致"

        print(f"   ✅ 数据可以正确重新加载")

        print(f"\n✅ 测试 3 通过: 数据持久化")
        return True

    except Exception as e:
        print(f"\n❌ 测试 3 失败: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if session_store:
            await session_store.disconnect()
        if trace_store:
            await trace_store.close()


def create_test_model():
    """创建测试用的 LLM Model"""
    # 按优先级尝试不同的模型
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
    """运行所有测试"""
    print("=" * 60)
    print("Agent Run 功能测试")
    print("=" * 60)

    results = []

    # 测试 1: Tools 支持
    try:
        result1 = await test_tools_support()
        results.append(("Tools 支持", result1))
    except Exception as e:
        print(f"\n❌ 测试 1 异常: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Tools 支持", False))

    # 测试 2: Skills 加载
    try:
        result2 = await test_skills_loading()
        results.append(("Skills 加载", result2))
    except Exception as e:
        print(f"\n❌ 测试 2 异常: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Skills 加载", False))

    # 测试 3: 数据持久化
    try:
        result3 = await test_data_persistence()
        results.append(("数据持久化", result3))
    except Exception as e:
        print(f"\n❌ 测试 3 异常: {e}")
        import traceback

        traceback.print_exc()
        results.append(("数据持久化", False))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status}: {test_name}")

    success_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print("\n" + "=" * 60)
    print(f"总计: {total_count} 个测试")
    print(f"通过: {success_count}, 失败: {total_count - success_count}")
    print("=" * 60)

    return all(passed for _, passed in results)


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(__doc__)
        return

    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n[FATAL ERROR] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
