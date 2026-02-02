#!/usr/bin/env python3
"""
真实 API 测试脚本

使用真实的 API key 测试各个 LLM 提供商的 API 调用。
需要在 .env 文件中配置相应的 API key。

环境变量配置：
- OPENAI_API_KEY
- DEEPSEEK_API_KEY
- NVIDIA_BUILD_API_KEY
- ANTHROPIC_API_KEY

使用方法：
    python test_real_api.py
    或
    uv run python test_real_api.py
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from agiwo.llm.anthropic import AnthropicModel
from agiwo.llm.deepseek import DeepseekModel
from agiwo.llm.nvidia import NvidiaModel
from agiwo.llm.openai import OpenAIModel


load_dotenv()


@dataclass
class ProviderConfig:
    """LLM 提供商配置"""

    name: str
    model_class: type
    model_id: str
    model_name: str
    api_key_env: str
    base_url: str | None = None


# nvidia_model = "moonshotai/kimi-k2.5"
nvidia_model = "z-ai/glm4.7"

PROVIDERS = [
    # ProviderConfig(
    #     name="OpenAI",
    #     model_class=OpenAIModel,
    #     model_id="gpt-4o-mini",
    #     model_name="gpt-4o-mini",
    #     api_key_env="OPENAI_API_KEY",
    #     base_url="https://api.openai.com/v1",
    # ),
    # ProviderConfig(
    #     name="DeepSeek",
    #     model_class=DeepseekModel,
    #     model_id="deepseek-chat",
    #     model_name="deepseek-chat",
    #     api_key_env="DEEPSEEK_API_KEY",
    #     base_url="https://api.deepseek.com",
    # ),
    ProviderConfig(
        name="NVIDIA",
        model_class=NvidiaModel,
        model_id=nvidia_model,
        model_name=nvidia_model,
        api_key_env="NVIDIA_BUILD_API_KEY",
        base_url="https://integrate.api.nvidia.com/v1",
    ),
    ProviderConfig(
        name="Anthropic",
        model_class=AnthropicModel,
        model_id="MiniMax-M2.1",
        model_name="MiniMax-M2.1",
        api_key_env="MINIMAX_API_KEY",
        base_url="https://api.minimaxi.com/anthropic",
    ),
]


async def test_provider(
    provider: ProviderConfig, test_messages: list[dict]
) -> dict[str, Any]:
    """
    测试单个 LLM 提供商

    Args:
        provider: 提供商配置
        test_messages: 测试消息列表

    Returns:
        测试结果字典
    """
    api_key = os.getenv(provider.api_key_env)

    if not api_key:
        return {
            "provider": provider.name,
            "status": "skipped",
            "reason": f"API key not found in environment variable: {provider.api_key_env}",
        }

    try:
        model_kwargs = {
            "id": provider.model_id,
            "name": provider.model_name,
            "api_key": api_key,
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 100,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

        if provider.base_url:
            model_kwargs["base_url"] = provider.base_url

        model = provider.model_class(**model_kwargs)

        print(f"\n{'=' * 60}")
        print(f"Testing {provider.name} ({provider.model_id})")
        print(f"{'=' * 60}")

        chunks = []
        full_content = ""
        full_thinking = ""
        total_usage = None

        async for chunk in model.arun_stream(test_messages):
            chunks.append(chunk)

            if chunk.content:
                full_content += chunk.content
                print(chunk.content, end="", flush=True)

            if chunk.reasoning_content:
                full_thinking += chunk.reasoning_content
                print(chunk.reasoning_content, end="", flush=True)

            if chunk.usage:
                total_usage = chunk.usage

            if chunk.finish_reason:
                print(f"\n[Finish reason: {chunk.finish_reason}]")

        print(f"\n\n[Total chunks: {len(chunks)}]")
        print(f"[Full thinking: {full_thinking}]")
        if total_usage:
            print(f"[Usage: {total_usage}]")

        return {
            "provider": provider.name,
            "status": "success",
            "model_id": provider.model_id,
            "chunks_count": len(chunks),
            "content_length": len(full_content),
            "content_preview": full_content[:100] + "..."
            if len(full_content) > 100
            else full_content,
            "thinking_length": len(full_thinking),
            "thinking_preview": full_thinking[:100] + "..."
            if len(full_thinking) > 100
            else full_thinking,
            "usage": total_usage,
            "finish_reason": chunks[-1].finish_reason if chunks else None,
        }

    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        print(f"\n[ERROR] {error_type}: {error_msg}")

        return {
            "provider": provider.name,
            "status": "failed",
            "error_type": error_type,
            "error_message": error_msg,
        }


async def run_all_tests():
    """运行所有提供商的测试"""
    test_messages = [{"role": "user", "content": "请用一句话介绍你自己。"}]

    print("=" * 60)
    print("LLM API 真实测试")
    print("=" * 60)
    print(f"\n测试消息: {test_messages[0]['content']}\n")

    results = []

    for provider in PROVIDERS:
        result = await test_provider(provider, test_messages)
        results.append(result)

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for result in results:
        status_icon = (
            "✓"
            if result["status"] == "success"
            else "✗"
            if result["status"] == "failed"
            else "○"
        )
        print(f"\n{status_icon} {result['provider']}: {result['status']}")

        if result["status"] == "success":
            print(f"  模型: {result['model_id']}")
            print(f"  内容长度: {result['content_length']} 字符")
            print(f"  块数量: {result['chunks_count']}")
            if result.get("usage"):
                usage = result["usage"]
                print(f"  Token 使用: {usage}")
        elif result["status"] == "failed":
            print(f"  错误: {result['error_type']}")
            print(f"  详情: {result['error_message']}")
        elif result["status"] == "skipped":
            print(f"  原因: {result['reason']}")

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")

    print("\n" + "=" * 60)
    print(f"总计: {len(results)} 个提供商")
    print(f"成功: {success_count}, 失败: {failed_count}, 跳过: {skipped_count}")
    print("=" * 60)

    return results


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(__doc__)
        return

    try:
        results = asyncio.run(run_all_tests())

        failed_providers = [r["provider"] for r in results if r["status"] == "failed"]
        if failed_providers:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n[FATAL ERROR] {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
