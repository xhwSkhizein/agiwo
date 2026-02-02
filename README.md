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
