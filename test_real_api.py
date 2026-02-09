#!/usr/bin/env python3
"""
Real API Test Script

Test API calls for various LLM providers using real API keys.
Requires corresponding API key configuration in .env file.

Environment Variable Configuration:
- OPENAI_API_KEY
- DEEPSEEK_API_KEY
- NVIDIA_BUILD_API_KEY
- ANTHROPIC_API_KEY

Usage:
    python test_real_api.py
    or
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
    """LLM provider configuration"""

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
    Test a single LLM provider

    Args:
        provider: Provider configuration
        test_messages: Test messages list

    Returns:
        Test result dictionary
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
    """Run tests for all providers"""
    test_messages = [{"role": "user", "content": "Please introduce yourself in one sentence."}]

    print("=" * 60)
    print("LLM API Real Test")
    print("=" * 60)
    print(f"\nTest message: {test_messages[0]['content']}\n")

    results = []

    for provider in PROVIDERS:
        result = await test_provider(provider, test_messages)
        results.append(result)

    print("\n" + "=" * 60)
    print("Test Results Summary")
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
            print(f"  Model: {result['model_id']}")
            print(f"  Content length: {result['content_length']} chars")
            print(f"  Chunks count: {result['chunks_count']}")
            if result.get("usage"):
                usage = result["usage"]
                print(f"  Token usage: {usage}")
        elif result["status"] == "failed":
            print(f"  Error: {result['error_type']}")
            print(f"  Details: {result['error_message']}")
        elif result["status"] == "skipped":
            print(f"  Reason: {result['reason']}")

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")

    print("\n" + "=" * 60)
    print(f"Total: {len(results)} providers")
    print(f"Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}")
    print("=" * 60)

    return results


def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(__doc__)
        return

    try:
        results = asyncio.run(run_all_tests())

        failed_providers = [r["provider"] for r in results if r["status"] == "failed"]
        if failed_providers:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n[FATAL ERROR] {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
