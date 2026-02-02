# AGIWO

AGIWO - AI Agent Framework

## 项目简介

AGIWO 是一个 AI Agent 框架，提供了统一的接口来调用多个 LLM 提供商（OpenAI、Anthropic、DeepSeek、NVIDIA）的 API。

## 环境要求

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) - 用于依赖管理

## 安装

使用 uv 安装项目依赖：

```bash
uv sync
```

这将创建虚拟环境并安装所有依赖项。

## 运行测试用例

### 单元测试（Mock 测试）

以下测试使用 mock 对象，不会调用真实的 API：

### 运行所有测试

```bash
uv run pytest tests/ -v
```

### 运行 LLM 模块的测试

```bash
uv run pytest tests/llm/ -v
```

### 运行特定测试文件

```bash
# 测试 OpenAI 模型
uv run pytest tests/llm/test_openai.py -v

# 测试 Anthropic 模型
uv run pytest tests/llm/test_anthropic.py -v

# 测试 DeepSeek 模型
uv run pytest tests/llm/test_deepseek.py -v

# 测试 NVIDIA 模型
uv run pytest tests/llm/test_nvidia.py -v

# 测试基础类和工具函数
uv run pytest tests/llm/test_base.py tests/llm/test_helper.py -v
```

### 运行特定测试用例

```bash
# 运行单个测试函数
uv run pytest tests/llm/test_openai.py::test_openai_model_arun_stream_basic -v

# 运行多个特定测试
uv run pytest tests/llm/test_openai.py::test_openai_model_arun_stream_basic tests/llm/test_openai.py::test_openai_model_arun_stream_with_usage -v
```

### 测试选项

- `-v` 或 `--verbose`: 显示详细输出
- `-s`: 显示 print 输出
- `--tb=short`: 显示简短的错误追踪
- `--tb=line`: 只显示错误行
- `-k <expression>`: 运行匹配表达式的测试

示例：

```bash
# 只运行包含 "basic" 的测试
uv run pytest tests/llm/ -k "basic" -v

# 显示简短错误追踪
uv run pytest tests/llm/ -v --tb=short
```

### 真实 API 测试

项目包含一个独立的真实 API 测试脚本 `test_real_api.py`，用于测试各个 LLM 提供商的真实 API 调用。

#### 配置 API Keys

1. 复制 `.env.example` 为 `.env`：
   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env` 文件，填入真实的 API keys：
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   NVIDIA_BUILD_API_KEY=your_nvidia_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

#### 运行真实 API 测试

```bash
# 使用 uv 运行
uv run python test_real_api.py

# 或直接运行（需要先激活虚拟环境）
python test_real_api.py
```

测试脚本会：
- 自动检测已配置的 API keys
- 依次测试所有配置的提供商
- 显示流式响应内容
- 输出详细的测试结果汇总
- 跳过未配置 API key 的提供商

**注意**：真实 API 测试会产生 API 调用费用，请谨慎使用。

## 项目结构

```
agiwo/
├── agiwo/
│   ├── llm/          # LLM 模型实现
│   │   ├── base.py    # 基础模型类
│   │   ├── openai.py  # OpenAI 模型
│   │   ├── anthropic.py  # Anthropic 模型
│   │   ├── deepseek.py   # DeepSeek 模型
│   │   ├── nvidia.py    # NVIDIA 模型
│   │   └── helper.py    # 工具函数
│   ├── config/        # 配置管理
│   └── utils/         # 工具函数
├── tests/             # 测试用例
│   └── llm/          # LLM 模块测试
└── pyproject.toml    # 项目配置
```

## 开发

### 添加新依赖

```bash
uv add <package-name>
```

### 添加开发依赖

```bash
uv add --dev <package-name>
```

### 激活虚拟环境

```bash
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows
```

## 许可证

MIT-License
