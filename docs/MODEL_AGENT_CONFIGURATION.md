# Model/Agent Configuration Guide

This document explains the current model configuration rules in Agiwo and Console.
It is written for both maintainers and first-time users.

## 1. What This Guide Covers

This guide answers:

- Which fields are required to configure an Agent model.
- How Agent and Tool model config are loaded at runtime.
- Why plain-text `api_key` is not allowed in Agent records.
- How official providers differ from compatible providers.
- Why legacy provider names are no longer accepted.

## 2. Core Principles

The current model configuration design follows three rules:

1. Use explicit connection config for compatible providers.
2. Use one shared model factory for Agent and Tool model creation.
3. Never persist plain-text API keys in Agent records.

The standard model config shape is:

- `model_provider`
- `model_name`
- `model_params.base_url`
- `model_params.api_key_env_name`
- `model_params` sampling and pricing params

## 3. Provider Names and Semantics

Supported `model_provider` values:

- `openai`
- `openai-compatible`
- `deepseek`
- `anthropic`
- `anthropic-compatible`
- `nvidia`
- `bedrock-anthropic`

Legacy names are not accepted. Use only the values above.

## 4. Official vs Compatible Providers

### Official providers

Official providers are `openai`, `deepseek`, `anthropic`, `nvidia`, `bedrock-anthropic`.

They can use provider-level environment defaults when `base_url` or API key is not explicitly provided.

Examples:

- `openai` may use `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
- `anthropic` may use `ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL`.

### Compatible providers

Compatible providers are `openai-compatible` and `anthropic-compatible`.

They must have:

- explicit `base_url`
- explicit `api_key_env_name` (the environment variable name that holds the key)

They do not fall back to `OPENAI_*` or `ANTHROPIC_*` automatically.

## 5. Agent Config Schema (Persisted)

In Console, Agent model parameters are stored under `agent_configs.model_params`.

Allowed model params include:

- `base_url`
- `api_key_env_name`
- `max_output_tokens_per_call`
- `temperature`
- `top_p`
- `frequency_penalty`
- `presence_penalty`
- `cache_hit_price`
- `input_price`
- `output_price`

`api_key` is rejected by API schema and model factory.

## 6. Security Policy: No Plain-text API Key in Agent Record

Plain-text `api_key` is blocked at multiple layers:

- Request validation rejects `model_params.api_key`.
- Model factory rejects `api_key` in `params`.
- Agent registry sanitizes `model_params` and removes `api_key`.
- Console default agent env (`AGIWO_CONSOLE_DEFAULT_AGENT_MODEL_PARAMS`) also rejects `api_key`.

The expected pattern is:

- Store only `api_key_env_name` in records.
- Put the real secret value in process environment.

## 7. Runtime Resolution Flow

### Agent path

1. Console receives create/update request.
2. Pydantic schema validates provider and `model_params`.
3. AgentRegistry sanitizes `model_params`.
4. `build_model()` calls shared `create_model_from_dict()`.
5. Model factory resolves `api_key_env_name` by reading environment at runtime.

### Tool path (`web_reader`)

1. Tool config reads defaults from `AGIWO_TOOL_DEFAULT_MODEL_*`.
2. Tool builds `ModelConfig` with `model_provider`, `model_name`, `base_url`, `api_key_env_name`, and sampling params.
3. Tool creates model through the same shared model factory as Agent.

## 8. Environment Variables

### 8.1 Agent/Console side

Agent records store:

- `model_provider`
- `model_name`
- `model_params.base_url`
- `model_params.api_key_env_name`

Real secret:

- environment variable named by `api_key_env_name`

Example:

```env
AGIWO_CONSOLE_DEFAULT_AGENT_MODEL_PROVIDER=openai-compatible
AGIWO_CONSOLE_DEFAULT_AGENT_MODEL_NAME=MiniMax-M2.5
AGIWO_CONSOLE_DEFAULT_AGENT_MODEL_PARAMS={"base_url":"https://api.edgefn.net/v1","api_key_env_name":"MINIMAX_API_KEY"}

MINIMAX_API_KEY=your_real_key
```

### 8.2 Tool side (`web_reader`)

Tool model uses only global defaults:

- `AGIWO_TOOL_DEFAULT_MODEL_PROVIDER`
- `AGIWO_TOOL_DEFAULT_MODEL_NAME`
- `AGIWO_TOOL_DEFAULT_MODEL_BASE_URL`
- `AGIWO_TOOL_DEFAULT_MODEL_API_KEY_ENV_NAME`
- `AGIWO_TOOL_DEFAULT_MODEL_TEMPERATURE`
- `AGIWO_TOOL_DEFAULT_MODEL_TOP_P`
- `AGIWO_TOOL_DEFAULT_MODEL_MAX_TOKENS`

Example:

```env
AGIWO_TOOL_DEFAULT_MODEL_PROVIDER=openai-compatible
AGIWO_TOOL_DEFAULT_MODEL_NAME=MiniMax-M2.5
AGIWO_TOOL_DEFAULT_MODEL_BASE_URL=https://api.edgefn.net/v1
AGIWO_TOOL_DEFAULT_MODEL_API_KEY_ENV_NAME=MINIMAX_API_KEY
AGIWO_TOOL_DEFAULT_MODEL_TEMPERATURE=0.2
AGIWO_TOOL_DEFAULT_MODEL_TOP_P=1.0
AGIWO_TOOL_DEFAULT_MODEL_MAX_TOKENS=2048

MINIMAX_API_KEY=your_real_key
```

## 9. Record Sanitization Notes

Record sanitization occurs when records are validated/read/written by Console:

- remove `model_params.api_key`
- normalize `base_url` and `api_key_env_name` whitespace

## 10. Troubleshooting

### Error: `Input should be ...` for provider value `generic` / `anthropic-generic`

Cause:

- A legacy provider name is still present in `.env` or persisted agent data.

Fix:

- Replace `generic` with `openai-compatible`.
- Replace `anthropic-generic` with `anthropic-compatible`.

### Error: `openai-compatible models require an explicit base_url`

Cause:

- `model_params.base_url` is missing or empty.

Fix:

- Add `base_url` in Agent model params.
- Ensure it starts with `http://` or `https://`.

### Error: `openai-compatible models require api_key_env_name`

Cause:

- `model_params.api_key_env_name` is missing.

Fix:

- Add `api_key_env_name` in Agent model params.
- Set the corresponding environment variable with the real key.

### Error: `... api_key_env_name 'XXX' is not set`

Cause:

- `api_key_env_name` exists in config but that env var is not available in process runtime.

Fix:

- Export that variable in the environment used to start Console/SDK.
- Restart service after updating env.

### Error: `api_key is not supported in model params`

Cause:

- request payload or default config still includes plain-text `api_key`.

Fix:

- replace `api_key` with `api_key_env_name`.

## 11. Practical Checklist

When configuring a compatible model:

1. Set provider to `openai-compatible` or `anthropic-compatible`.
2. Fill `model_name`.
3. Fill `base_url`.
4. Fill `api_key_env_name`.
5. Define the real key in environment.
6. Optionally tune sampling params.

When configuring an official provider:

1. Set provider to official name.
2. Fill `model_name`.
3. Optionally set `base_url` and `api_key_env_name`.
4. Otherwise rely on provider-level env defaults.

## 12. Source of Truth

Key implementation files:

- `agiwo/llm/factory.py`
- `agiwo/config/settings.py`
- `console/server/schemas.py`
- `console/server/config.py`
- `console/server/services/agent_registry.py`
- `agiwo/tool/builtin/config.py`
- `agiwo/tool/builtin/web_reader/web_reader_tool.py`
