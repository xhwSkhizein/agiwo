# Python 环境问题记录

本文档记录 Agiwo 项目在本地开发时遇到的 Python / uv / conda 环境问题、根因分析与已采取的修复措施。

## 现象

在本地启动 Console 或执行 `uv run` 时，出现以下症状：

1. **Python 进程被系统直接杀掉**
   - 退出码 `137`（`SIGKILL`）
   - macOS 崩溃报告：`SIGKILL (Code Signature Invalid)`

2. **uv 报错**
   ```text
   error: Querying Python at `.../.venv/bin/python3` failed with exit status signal: 9 (SIGKILL)
   ```

3. **Console 后端无法启动**
   - 若强行用临时虚拟环境绕过 Python 问题，还可能遇到 `.env` 配置键名过旧导致 `ConsoleConfig` 校验失败。

4. **临时 workaround 很奇怪**
   - 曾使用 `/tmp/agiwo-console-venv` 和 `/tmp/agiwo-console-start.env` 才能启动，这不是项目正常设计。

## 根因

### 1. 虚拟环境误用了 conda 的 Python

检查后发现，项目内两个虚拟环境实际都指向 conda：

```text
agiwo/.venv/bin/python3      -> /Users/hongv/miniconda3/bin/python3.13
console/.venv/bin/python3      -> /Users/hongv/miniconda3/bin/python3.13
```

原因是：

| 因素 | 实际情况 |
| --- | --- |
| PATH 优先级 | `which python` / `which python3` 指向 `/Users/hongv/miniconda3/bin/...` |
| uv 来源 | `uv` 本身也安装在 conda 环境中 |
| 项目未固定 Python | 仓库里原先没有 `.python-version`，也未限制 uv 只能使用 managed Python |
| uv 建 venv 行为 | `uv sync` 时自动选用 PATH 上可用的 Python，于是选中了 conda 3.13 |

### 2. conda Python 3.13 在本机不可用

在该 macOS 环境下，conda 自带的 `python3.13` 启动即被系统终止，原因为代码签名校验失败（`Code Signature Invalid`）。这与 Agiwo 代码无关，是**解释器本身无法运行**。

对比验证：

```text
/Users/hongv/miniconda3/bin/python3.13   -> SIGKILL (137)
/Users/hongv/.local/share/uv/python/.../python3.12 -> 正常
```

### 3. Console `.env` 使用了已废弃的扁平配置键

`console/.env` 中仍使用旧键名，例如：

```env
AGIWO_CONSOLE_HOST=0.0.0.0
AGIWO_CONSOLE_PORT=8422
AGIWO_CONSOLE_FEISHU_ENABLED=true
```

而当前 `ConsoleConfig` 已改为嵌套结构，并要求显式拒绝旧键，例如：

```env
AGIWO_CONSOLE_SERVER__HOST=0.0.0.0
AGIWO_CONSOLE_SERVER__PORT=8422
AGIWO_CONSOLE_CHANNELS__FEISHU__ENABLED=true
```

因此即使 Python 环境修好，直接 `serve --env-file .env` 仍可能启动失败。

## 已做的事情

### 1. 固定项目 Python 为 uv-managed 3.12

- 新增根目录 `.python-version`，内容为 `3.12`
- 在根目录 `pyproject.toml` 与 `console/pyproject.toml` 增加：

```toml
[tool.uv]
python-preference = "only-managed"
```

作用：禁止 uv 使用 PATH 上的 conda / 系统 Python 创建虚拟环境，只使用 uv 自己下载管理的 CPython。

### 2. 新增开发环境修复脚本

新增 `scripts/setup_dev_env.py`，用于：

- 安装并 pin uv-managed Python（当前为 3.12）
- 删除旧的 `.venv` 并重建 SDK / Console 两套虚拟环境
- 检测 venv 是否仍指向 conda
- 自动迁移 `console/.env` 与 `console/.env.example.full` 中的旧配置键名
- 可选安装 git hooks（`--install-hooks`）

推荐用法：

```bash
cd agiwo
uv run python scripts/setup_dev_env.py --install-hooks
```

若旧 `.venv` 已坏到 `uv run` 都无法执行，可先删除再运行：

```bash
rm -rf .venv console/.venv
python3 scripts/setup_dev_env.py --install-hooks
```

### 3. 重建本地虚拟环境

已删除基于 conda 的旧 venv，并重新 `uv sync`。修复后解释器来源为：

```text
~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python3.12
```

### 4. 迁移 Console 环境变量

已将 `console/.env` 和 `console/.env.example.full` 中的旧键名迁移为新嵌套格式，例如：

| 旧键 | 新键 |
| --- | --- |
| `AGIWO_CONSOLE_HOST` | `AGIWO_CONSOLE_SERVER__HOST` |
| `AGIWO_CONSOLE_PORT` | `AGIWO_CONSOLE_SERVER__PORT` |
| `AGIWO_CONSOLE_RUN_STEP_STORAGE_TYPE` | `AGIWO_CONSOLE_STORAGE__RUN_LOG_TYPE` |
| `AGIWO_CONSOLE_TRACE_STORAGE_TYPE` | `AGIWO_CONSOLE_STORAGE__TRACE_TYPE` |
| `AGIWO_CONSOLE_METADATA_STORAGE_TYPE` | `AGIWO_CONSOLE_STORAGE__METADATA_TYPE` |
| `AGIWO_CONSOLE_FEISHU_*` | `AGIWO_CONSOLE_CHANNELS__FEISHU__*` |

### 5. 更新文档

- `README.md`：开发环境安装步骤改为推荐 `setup_dev_env.py`
- `AGENTS.md`：同步更新安装说明

### 6. 补充测试

新增 `tests/scripts/test_setup_dev_env.py`，覆盖 Console `.env` 键名迁移逻辑。

## 正常启动方式（修复后）

**SDK / 测试：**

```bash
cd agiwo
uv run python -V
uv run pytest tests/ -q
```

**Console 后端：**

```bash
cd console
uv run agiwo-console serve --env-file .env
```

**Console 前端：**

```bash
cd console/web
npm run dev
```

默认地址：

- API：`http://localhost:8422`
- Web UI：`http://localhost:3000`

## 可清理的临时文件

若之前为绕过问题创建过临时环境，可手动删除：

```bash
rm -rf /tmp/agiwo-console-venv /tmp/agiwo-console-start.env
```

这些路径不应再作为常规启动方式。

## 关于 conda 的结论

- **不需要卸载 conda**
- conda 仍可管理其他项目
- 对本仓库而言，关键是：**不要让 uv 用 conda Python 建 `.venv`**
- 通过 `.python-version` + `python-preference = "only-managed"` 已规避该问题

## 若问题复发，优先检查

1. `readlink -f .venv/bin/python3` 是否又指回 `miniconda3` / `anaconda3`
2. `uv run python -V` 是否能正常输出版本号
3. `console/.env` 是否又出现 `AGIWO_CONSOLE_HOST` 等旧键名
4. 必要时重新执行：

```bash
uv run python scripts/setup_dev_env.py --install-hooks
```
