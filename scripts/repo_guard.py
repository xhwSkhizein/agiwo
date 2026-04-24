"""
Repository-specific lint checks that target architecture drift.
"""

import argparse
import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lint_common import (
    ROOT,
    collect_changed_paths,
    collect_python_paths,
    iter_all_python_paths,
    normalize_paths,
)

ALLOWED_OS_GETENV = {
    Path("agiwo/config/settings.py"),
    Path("agiwo/llm/factory.py"),
}
ALLOWED_MANUAL_USER_INPUT_DECODING = {
    Path("agiwo/agent/models/input.py"),
    Path("agiwo/agent/input_codec.py"),
    Path("tests"),
    Path("console/tests"),
}
ALLOWED_AGENT_INTERNAL_IMPORT_PREFIXES = (
    Path("agiwo/agent"),
    Path("tests"),
    Path("console/tests"),
)
ALLOWED_DEAD_PERMISSION_API_PREFIXES = (
    Path("tests"),
    Path("console/tests"),
)
ALLOWED_BARE_ANY_ANNOTATION_PREFIXES = (
    Path("tests"),
    Path("console/tests"),
)

# FIXME: review later & fix
ALLOWED_BARE_ANY_ANNOTATION_PATHS = {
    Path("agiwo/agent/models/config.py"),
    Path("agiwo/agent/storage/serialization.py"),
    Path("agiwo/llm/config_policy.py"),
    Path("agiwo/llm/event_normalizer.py"),
    Path("agiwo/llm/factory.py"),
    Path("agiwo/llm/usage_resolver.py"),
    Path("agiwo/tool/base.py"),
    Path("agiwo/observability/store.py"),
    Path("agiwo/tool/builtin/bash_tool/security.py"),
    Path("agiwo/tool/builtin/bash_tool/tool.py"),
    Path("agiwo/tool/builtin/bash_tool/types.py"),
    Path("agiwo/tool/builtin/web_reader/web_reader_tool.py"),
    Path("agiwo/tool/storage/citation/mongo_store.py"),
    Path("agiwo/utils/serialization.py"),
}
ALLOWED_DIRECT_TOOLRESULT_CONSTRUCTOR_PREFIXES = (
    Path("tests"),
    Path("console/tests"),
)
ALLOWED_CONSOLE_APP_STATE_ACCESS_PATHS = {
    Path("console/server/dependencies.py"),
}
ALLOWED_CONSOLE_API_IMPORT_PREFIXES = (
    Path("console/server/routers"),
    Path("console/tests"),
)
ALLOWED_CONSOLE_API_IMPORT_PATHS = {
    Path("console/server/models/view.py"),
    Path("console/server/response_serialization.py"),
}
ALLOWED_SESSION_IDENTITY_ASSIGN_PATHS = {
    Path("console/server/models/session.py"),
}
ALLOWED_SESSION_IDENTITY_ASSIGN_PREFIXES = (Path("console/tests"),)
ALLOWED_FEISHU_SDK_IMPORT_PATHS = {
    Path("console/server/channels/feishu/connection.py"),
    Path("console/server/channels/feishu/sdk_adapter.py"),
}
ALLOWED_RAW_SETTINGS_SKILLS_DIRS_PATHS = {
    Path("agiwo/config/settings.py"),
}
ALLOWED_RAW_SETTINGS_SKILLS_DIRS_PREFIXES = (
    Path("tests"),
    Path("console/tests"),
)
ALLOWED_STEPRECORD_CONSTRUCTOR_PREFIXES = (
    Path("tests"),
    Path("console/tests"),
)
ALLOWED_STEPRECORD_CONSTRUCTOR_PATHS = {
    Path("agiwo/agent/storage/serialization.py"),
}
ALLOWED_STORAGE_BACKEND_CONSTRUCTOR_PREFIXES = (
    Path("tests"),
    Path("console/tests"),
)
ALLOWED_STORAGE_BACKEND_CONSTRUCTOR_PATHS = {
    Path("agiwo/agent/storage/factory.py"),
    Path("agiwo/observability/factory.py"),
}
ALLOWED_MODEL_CONSTRUCTOR_PREFIXES = (
    Path("agiwo/llm"),
    Path("tests"),
    Path("console/tests"),
)
WARNING_ONLY_CODES = {
    "AGW004",
}
FILE_GROWTH_BUDGETS = (
    (re.compile(r"^console/server/routers/[^/]+\.py$"), 180),
    (re.compile(r"^console/server/channels/agent_runtime\.py$"), 450),
    (re.compile(r"^console/server/channels/feishu/store\.py$"), 420),
    (re.compile(r"^console/server/services/agent_registry\.py$"), 320),
    (re.compile(r"^console/server/models/view\.py$"), 420),
    (re.compile(r"^console/server/response_serialization\.py$"), 280),
    (re.compile(r"^agiwo/tool/builtin/bash_tool/tool\.py$"), 840),
    (re.compile(r"^agiwo/config/settings\.py$"), 390),
    (re.compile(r"^agiwo/observability/collector\.py$"), 600),
    (re.compile(r"^agiwo/agent/agent\.py$"), 560),
    (re.compile(r"^agiwo/agent/run_loop\.py$"), 560),
    (re.compile(r"^agiwo/agent/nested/agent_tool\.py$"), 220),
    (re.compile(r"^agiwo/agent/models/config\.py$"), 360),
    (re.compile(r"^agiwo/scheduler/coordinator\.py$"), 180),
    (re.compile(r"^agiwo/scheduler/runner\.py$"), 720),
    (re.compile(r"^agiwo/scheduler/engine\.py$"), 900),
    (re.compile(r"^agiwo/scheduler/tools\.py$"), 690),
    (re.compile(r"^agiwo/scheduler/scheduler\.py$"), 320),
    (re.compile(r"^agiwo/tool/permission/manager\.py$"), 530),
    (re.compile(r"^agiwo/tool/permission/store\.py$"), 500),
)
STEPRECORD_CONSTRUCTOR_NAMES = {
    "StepRecord",
}
STORAGE_BACKEND_CONSTRUCTOR_NAMES = {
    "SQLiteRunStepStorage",
    "MongoRunStepStorage",
    "SQLiteTraceStorage",
    "MongoTraceStorage",
}
MODEL_CONSTRUCTOR_NAMES = {
    "OpenAIModel",
    "AnthropicModel",
    "DeepseekModel",
    "NvidiaModel",
    "BedrockAnthropicModel",
}
SCHEDULER_FACADE_DELEGATE_ATTRS = {
    "_dispatch_service",
    "_engine",
    "_executor",
    "_output_service",
    "_tick_engine",
}
SESSION_IDENTITY_FIELD_NAMES = {
    "current_session_id",
    "base_agent_id",
    "runtime_agent_id",
    "scheduler_state_id",
}
SESSION_CONTEXT_ERROR_CODES = {
    "chat_context_not_found",
    "session_not_found",
    "session_not_in_current_chat_context",
}
FEISHU_ENVELOPE_ENTRYPOINTS = {
    Path("console/server/channels/feishu/message_parser.py"): {
        "parse_inbound_message",
    },
    Path("console/server/channels/feishu/inbound_handler.py"): {
        "process_envelope",
        "process_payload",
    },
    Path("console/server/channels/feishu/service.py"): {
        "_process_incoming_envelope",
        "_process_incoming_payload",
    },
}


@dataclass(frozen=True)
class GuardError:
    path: Path
    line: int
    code: str
    message: str


def _make_error(path: Path, line: int, code: str, message: str) -> GuardError:
    return GuardError(path=path, line=line, code=code, message=message)


def _is_allowed_prefix(path: Path, prefixes: tuple[Path, ...] | set[Path]) -> bool:
    path_str = path.as_posix()
    for prefix in prefixes:
        prefix_str = prefix.as_posix()
        if path_str == prefix_str or path_str.startswith(f"{prefix_str}/"):
            return True
    return False


def _imports_agent_internal(module_name: str | None) -> bool:
    if module_name is None:
        return False
    return module_name in {
        "agiwo.agent.run_loop",
        "agiwo.agent.tool_executor",
        "agiwo.agent.compaction",
        "agiwo.agent.llm_caller",
        "agiwo.agent.prompt",
        "agiwo.agent.runtime",
    } or module_name.startswith(
        (
            "agiwo.agent.run_loop.",
            "agiwo.agent.tool_executor.",
            "agiwo.agent.compaction.",
            "agiwo.agent.llm_caller.",
            "agiwo.agent.prompt.",
            "agiwo.agent.runtime.",
        )
    )


def _imports_agent_v2(module_name: str | None) -> bool:
    if module_name is None:
        return False
    return module_name == "agiwo.agent_v2" or module_name.startswith("agiwo.agent_v2.")


def _imports_bash_tool_impl(module_name: str | None) -> bool:
    if module_name is None:
        return False
    return module_name == "agiwo.tool.builtin.bash_tool" or module_name.startswith(
        "agiwo.tool.builtin.bash_tool."
    )


def _imports_scheduler_store_impl(module_name: str | None) -> bool:
    if module_name is None:
        return False
    return module_name == "agiwo.scheduler.store" or module_name.startswith(
        "agiwo.scheduler.store."
    )


def _imports_feishu_sdk_impl(module_name: str | None) -> bool:
    if module_name is None:
        return False
    return module_name == "lark_oapi" or module_name.startswith("lark_oapi.")


def _is_console_api_boundary_import(module_name: str | None) -> bool:
    return module_name in {"server.models.view", "server.response_serialization"}


def _allows_console_api_boundary_import(path: Path) -> bool:
    return path in ALLOWED_CONSOLE_API_IMPORT_PATHS or _is_allowed_prefix(
        path,
        ALLOWED_CONSOLE_API_IMPORT_PREFIXES,
    )


def _allows_session_identity_assignment(path: Path) -> bool:
    return path in ALLOWED_SESSION_IDENTITY_ASSIGN_PATHS or _is_allowed_prefix(
        path,
        ALLOWED_SESSION_IDENTITY_ASSIGN_PREFIXES,
    )


def _is_session_identity_assignment_target(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr in SESSION_IDENTITY_FIELD_NAMES


def _is_os_getenv_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "getenv"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
    )


def _is_manual_user_input_decode_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_dict"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"ContentPart", "UserMessage"}
    )


def _get_call_target_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_scheduler_private_access(node: ast.Attribute) -> bool:
    return (
        isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
        and node.value.attr == "_scheduler"
        and node.attr.startswith("_")
    )


def _is_scheduler_facade_delegate(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in SCHEDULER_FACADE_DELEGATE_ATTRS
    )


def _is_scheduler_delegate_private_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith("_")
        and _is_scheduler_facade_delegate(node.func.value)
    )


def _strip_docstring_stmt(
    body: list[ast.stmt],
) -> list[ast.stmt]:
    if not body:
        return body
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return body[1:]
    return body


def _unwrap_statement_call(stmt: ast.stmt) -> ast.Call | None:
    value: ast.AST | None = None
    if isinstance(stmt, ast.Return):
        value = stmt.value
    elif isinstance(stmt, ast.Expr):
        value = stmt.value
    if isinstance(value, ast.Await):
        value = value.value
    return value if isinstance(value, ast.Call) else None


def _is_thin_scheduler_wrapper(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    if not node.name.startswith("_") or node.name.startswith("__"):
        return False
    body = _strip_docstring_stmt(node.body)
    if len(body) != 1:
        return False
    call = _unwrap_statement_call(body[0])
    return (
        call is not None
        and isinstance(call.func, ast.Attribute)
        and _is_scheduler_facade_delegate(call.func.value)
    )


def _is_raw_settings_skills_dirs_access(node: ast.Attribute) -> bool:
    if node.attr != "skills_dirs":
        return False
    if (
        isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "get_settings"
    ):
        return True
    return isinstance(node.value, ast.Name) and node.value.id == "settings"


def _is_console_app_state_access(node: ast.Attribute) -> bool:
    return node.attr == "state" and (
        isinstance(node.value, ast.Name)
        and node.value.id == "app"
        or isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "request"
        and node.value.attr == "app"
    )


def _is_agent_state_storage_access(node: ast.Attribute) -> bool:
    return node.attr == "agent_state_storage"


def _is_any_reference(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id == "Any"
        or isinstance(node, ast.Attribute)
        and node.attr == "Any"
    )


def _is_annotation_name(node: ast.AST, names: set[str]) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id in names
        or isinstance(node, ast.Attribute)
        and node.attr in names
    )


def _iter_annotation_items(node: ast.Subscript) -> list[ast.AST]:
    if isinstance(node.slice, ast.Tuple):
        return list(node.slice.elts)
    return [node.slice]


def _is_banned_bare_any_annotation(node: ast.AST | None) -> bool:
    if node is None:
        return False
    is_banned = _is_any_reference(node)
    if not is_banned and isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        is_banned = _is_banned_bare_any_annotation(
            node.left
        ) or _is_banned_bare_any_annotation(node.right)
    elif not is_banned and isinstance(node, ast.Subscript):
        items = _iter_annotation_items(node)
        if _is_annotation_name(node.value, {"Optional", "Required", "NotRequired"}):
            is_banned = _is_banned_bare_any_annotation(node.slice)
        elif _is_annotation_name(node.value, {"Union"}):
            is_banned = any(_is_banned_bare_any_annotation(item) for item in items)
        elif _is_annotation_name(node.value, {"Annotated"}) and items:
            is_banned = _is_banned_bare_any_annotation(items[0])
    return is_banned


def _is_dict_annotation(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Subscript) and _is_annotation_name(
        node.value,
        {"dict", "Dict"},
    )


def _allows_bare_any_annotation(path: Path) -> bool:
    return path in ALLOWED_BARE_ANY_ANNOTATION_PATHS or _is_allowed_prefix(
        path, ALLOWED_BARE_ANY_ANNOTATION_PREFIXES
    )


def _is_direct_toolresult_constructor(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "ToolResult"


def _allows_direct_toolresult_constructor(path: Path) -> bool:
    return _is_allowed_prefix(path, ALLOWED_DIRECT_TOOLRESULT_CONSTRUCTOR_PREFIXES)


def _literal_string_contains(node: ast.AST, needle: str) -> bool:
    normalized = needle.casefold()
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and normalized in child.value.casefold()
        ):
            return True
    return False


def _contains_session_context_error_code(node: ast.Call) -> bool:
    return any(
        _literal_string_contains(arg, code)
        for arg in node.args
        for code in SESSION_CONTEXT_ERROR_CODES
    )


def _requires_feishu_inbound_envelope(path: Path, function_name: str) -> bool:
    required_entrypoints = FEISHU_ENVELOPE_ENTRYPOINTS.get(path)
    return required_entrypoints is not None and function_name in required_entrypoints


def _detect_feishu_inbound_envelope_error(
    path: Path,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> GuardError | None:
    if not _requires_feishu_inbound_envelope(path, node.name):
        return None
    non_self_args = [arg for arg in node.args.args if arg.arg != "self"]
    if (
        non_self_args
        and _is_annotation_name(
            non_self_args[0].annotation,
            {"FeishuInboundEnvelope"},
        )
        and not _is_dict_annotation(non_self_args[0].annotation)
    ):
        return None
    return _make_error(
        path,
        node.lineno,
        "AGW024",
        (
            "Feishu inbound entrypoints must accept FeishuInboundEnvelope; "
            "do not reintroduce raw dict payload contracts in parser/handler/service."
        ),
    )


def _detect_annotation_error(path: Path, line: int) -> GuardError | None:
    if _allows_bare_any_annotation(path):
        return None
    return _make_error(
        path,
        line,
        "AGW008",
        (
            "Do not use bare Any in type annotations; use a concrete type, "
            "or an explicit open payload shape like `dict[str, Any]`."
        ),
    )


def _detect_special_function_errors(
    path: Path,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[GuardError]:
    errors: list[GuardError] = []
    if node.name == "needs_permissions" and not _is_allowed_prefix(
        path, ALLOWED_DEAD_PERMISSION_API_PREFIXES
    ):
        errors.append(
            _make_error(
                path,
                node.lineno,
                "AGW005",
                (
                    "Do not extend the dead `needs_permissions` API surface until "
                    "permission checks are wired through ToolExecutor."
                ),
            )
        )
    if path == Path("agiwo/scheduler/scheduler.py") and _is_thin_scheduler_wrapper(
        node
    ):
        errors.append(
            _make_error(
                path,
                node.lineno,
                "AGW016",
                (
                    "Do not add thin private wrapper methods to Scheduler; move the "
                    "runtime behavior into a real runtime/core object and keep "
                    "Scheduler as a public facade."
                ),
            )
        )
    return errors


def _iter_function_args(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.arg]:
    args = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ]
    if node.args.vararg is not None:
        args.append(node.args.vararg)
    if node.args.kwarg is not None:
        args.append(node.args.kwarg)
    return args


def _detect_function_errors(
    path: Path,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[GuardError]:
    errors = _detect_special_function_errors(path, node)
    for arg in _iter_function_args(node):
        if (
            arg.annotation is not None
            and _is_any_reference(arg.annotation)
            and (arg is node.args.vararg or arg is node.args.kwarg)
        ):
            continue
        error = _detect_annotation_error(path, arg.lineno)
        if error is not None and _is_banned_bare_any_annotation(arg.annotation):
            errors.append(error)

    error = _detect_annotation_error(path, node.lineno)
    if error is not None and _is_banned_bare_any_annotation(node.returns):
        errors.append(error)
    feishu_error = _detect_feishu_inbound_envelope_error(path, node)
    if feishu_error is not None:
        errors.append(feishu_error)
    return errors


def _is_model_dump_exclude_unset_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "model_dump"
        and any(
            keyword.arg == "exclude_unset"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
    )


def _detect_call_errors(path: Path, node: ast.Call) -> list[GuardError]:
    errors: list[GuardError] = []
    call_target = _get_call_target_name(node)
    if (
        path.as_posix().startswith("console/server/channels/")
        and call_target == "ValueError"
        and _contains_session_context_error_code(node)
    ):
        errors.append(
            _make_error(
                path,
                node.lineno,
                "AGW020",
                (
                    "Do not use string-coded ValueError for session-context "
                    "transitions; raise the explicit SessionContextError types "
                    "from session_binding.py instead."
                ),
            )
        )
    if path == Path(
        "agiwo/scheduler/scheduler.py"
    ) and _is_scheduler_delegate_private_call(node):
        errors.append(
            _make_error(
                path,
                node.lineno,
                "AGW015",
                (
                    "Scheduler must not call private methods on executor/tick/"
                    "dispatch/output helpers; expose a public runtime API instead."
                ),
            )
        )
    if _is_os_getenv_call(node) and path not in ALLOWED_OS_GETENV:
        errors.append(
            _make_error(
                path,
                node.lineno,
                "AGW001",
                (
                    "Do not add os.getenv outside the configuration layer; route "
                    "new settings through agiwo/config/settings.py."
                ),
            )
        )
    if _is_manual_user_input_decode_call(node) and not _is_allowed_prefix(
        path, ALLOWED_MANUAL_USER_INPUT_DECODING
    ):
        errors.append(
            _make_error(
                path,
                node.lineno,
                "AGW002",
                (
                    "Do not manually rebuild UserInput payloads; reuse "
                    "serialize_user_input()/deserialize_user_input()."
                ),
            )
        )
    if (
        call_target in STEPRECORD_CONSTRUCTOR_NAMES
        and path not in ALLOWED_STEPRECORD_CONSTRUCTOR_PATHS
        and not _is_allowed_prefix(path, ALLOWED_STEPRECORD_CONSTRUCTOR_PREFIXES)
    ):
        errors.append(
            _make_error(
                path,
                node.lineno,
                "AGW012",
                (
                    "Prefer StepRecord.user()/assistant()/tool(); keep direct "
                    "StepRecord(...) construction in serialization/tests only."
                ),
            )
        )
    if (
        call_target in STORAGE_BACKEND_CONSTRUCTOR_NAMES
        and path not in ALLOWED_STORAGE_BACKEND_CONSTRUCTOR_PATHS
        and not _is_allowed_prefix(path, ALLOWED_STORAGE_BACKEND_CONSTRUCTOR_PREFIXES)
    ):
        errors.append(
            _make_error(
                path,
                node.lineno,
                "AGW013",
                (
                    "Prefer storage constructor/config injection; do not directly "
                    "instantiate storage backend classes outside the factory/tests."
                ),
            )
        )
    if call_target in MODEL_CONSTRUCTOR_NAMES and not _is_allowed_prefix(
        path, ALLOWED_MODEL_CONSTRUCTOR_PREFIXES
    ):
        errors.append(
            _make_error(
                path,
                node.lineno,
                "AGW014",
                (
                    "Prefer agiwo.llm.factory.create_model(); do not directly "
                    "instantiate concrete provider models outside agiwo.llm/tests."
                ),
            )
        )
    if _is_direct_toolresult_constructor(
        node
    ) and not _allows_direct_toolresult_constructor(path):
        errors.append(
            _make_error(
                path,
                node.lineno,
                "AGW009",
                (
                    "Do not construct ToolResult directly in production code; use "
                    "ToolResult.success()/failed()/aborted()/denied() so result "
                    "shape stays centralized."
                ),
            )
        )
    if path == Path(
        "console/server/routers/agents.py"
    ) and _is_model_dump_exclude_unset_call(node):
        errors.append(
            _make_error(
                path,
                node.lineno,
                "AGW026",
                (
                    "Agent config PUT must serialize a full replacement payload; "
                    "do not reintroduce model_dump(exclude_unset=True) patch semantics."
                ),
            )
        )
    return errors


def _detect_import_name_errors(
    path: Path, module_name: str, line: int
) -> list[GuardError]:
    errors: list[GuardError] = []
    if (
        _is_console_api_boundary_import(module_name)
        and path.as_posix().startswith("console/server/")
        and not _allows_console_api_boundary_import(path)
    ):
        code = "AGW017" if module_name == "server.models.view" else "AGW018"
        message = (
            "Keep server.models.view API-only; move shared models/helpers to "
            "server.models before importing them from services/channels."
            if module_name == "server.models.view"
            else "Keep server.response_serialization at the API boundary; "
            "services/channels must use server.models or core serializers instead."
        )
        errors.append(_make_error(path, line, code, message))
    if _imports_agent_v2(module_name):
        errors.append(
            _make_error(
                path,
                line,
                "AGW035",
                (
                    "Do not import agiwo.agent_v2; the canonical agent SDK surface "
                    "lives under agiwo.agent."
                ),
            )
        )
    if _imports_agent_internal(module_name) and not _is_allowed_prefix(
        path, ALLOWED_AGENT_INTERNAL_IMPORT_PREFIXES
    ):
        errors.append(
            _make_error(
                path,
                line,
                "AGW003",
                (
                    "Do not import agiwo.agent internal execution modules "
                    "(run_loop/tool_executor/compaction/"
                    "llm_caller/prompt/runtime) outside "
                    "agiwo.agent/tests; depend on agiwo.agent public surfaces "
                    "or stable boundary modules instead."
                ),
            )
        )
    if _imports_bash_tool_impl(module_name) and path.as_posix().startswith(
        "agiwo/scheduler/"
    ):
        errors.append(
            _make_error(
                path,
                line,
                "AGW006",
                (
                    "Scheduler code must not depend on the builtin BashTool "
                    "implementation; introduce a narrower port."
                ),
            )
        )
    if path == Path(
        "console/server/services/storage_manager.py"
    ) and _imports_scheduler_store_impl(module_name):
        errors.append(
            _make_error(
                path,
                line,
                "AGW022",
                (
                    "StorageManager must not depend on agiwo.scheduler.store; "
                    "Scheduler owns agent_state_storage and only exposes its "
                    "config via build_agent_state_storage_config()."
                ),
            )
        )
    if (
        _imports_feishu_sdk_impl(module_name)
        and path.as_posix().startswith("console/server/channels/feishu/")
        and path not in ALLOWED_FEISHU_SDK_IMPORT_PATHS
    ):
        errors.append(
            _make_error(
                path,
                line,
                "AGW023",
                (
                    "Feishu SDK imports must stay inside connection.py or "
                    "sdk_adapter.py; downstream code should depend on "
                    "FeishuConnection/FeishuInboundEnvelope instead of lark_oapi."
                ),
            )
        )
    return errors


def _detect_attribute_errors(path: Path, node: ast.Attribute) -> list[GuardError]:
    errors: list[GuardError] = []
    if path == Path("agiwo/scheduler/tools.py") and _is_scheduler_private_access(node):
        errors.append(
            _make_error(
                path,
                node.lineno,
                "AGW007",
                (
                    "Scheduler tools must not reach into Scheduler private state; "
                    "expose an explicit scheduler port instead."
                ),
            )
        )
    if (
        path.as_posix().startswith("console/server/")
        and path not in ALLOWED_CONSOLE_APP_STATE_ACCESS_PATHS
        and _is_console_app_state_access(node)
    ):
        errors.append(
            _make_error(
                path,
                node.lineno,
                "AGW011",
                (
                    "Do not access app.state directly outside console/server/"
                    "dependencies.py; route runtime access through ConsoleRuntime "
                    "dependencies instead."
                ),
            )
        )
    if (
        _is_raw_settings_skills_dirs_access(node)
        and path not in ALLOWED_RAW_SETTINGS_SKILLS_DIRS_PATHS
        and not _is_allowed_prefix(path, ALLOWED_RAW_SETTINGS_SKILLS_DIRS_PREFIXES)
    ):
        errors.append(
            _make_error(
                path,
                node.lineno,
                "AGW044",
                (
                    "Do not access settings.skills_dirs directly; use "
                    "get_settings().get_env_skills_dirs() to respect "
                    "explicit-env-only semantics."
                ),
            )
        )
    if path.as_posix().startswith("console/server/") and _is_agent_state_storage_access(
        node
    ):
        errors.append(
            _make_error(
                path,
                node.lineno,
                "AGW021",
                (
                    "Do not access storage_manager.agent_state_storage; "
                    "Scheduler owns agent_state_storage, so read it via "
                    "runtime.scheduler.store."
                ),
            )
        )
    return errors


def _detect_annassign_errors(path: Path, node: ast.AnnAssign) -> list[GuardError]:
    error = _detect_annotation_error(path, node.lineno)
    if error is None or not _is_banned_bare_any_annotation(node.annotation):
        return []
    return [error]


def _detect_assign_errors(
    path: Path, node: ast.Assign | ast.AugAssign
) -> list[GuardError]:
    targets: list[ast.AST]
    if isinstance(node, ast.Assign):
        targets = node.targets
    else:
        targets = [node.target]

    errors: list[GuardError] = []
    if path.as_posix().startswith(
        "console/server/channels/"
    ) and not _allows_session_identity_assignment(path):
        for target in targets:
            if _is_session_identity_assignment_target(target):
                errors.append(
                    _make_error(
                        path,
                        node.lineno,
                        "AGW019",
                        (
                            "Do not mutate session/chat identity fields outside "
                            "session_binding.py; route updates through explicit "
                            "SessionBinding/SessionIdentity transitions."
                        ),
                    )
                )
                break

    return errors


def _resolve_import_from_module(path: Path, node: ast.ImportFrom) -> str | None:
    """Resolve an ``ImportFrom`` node to its fully-qualified module name.

    For absolute imports (``node.level == 0``), ``node.module`` is already
    fully-qualified.  For relative imports we derive the package prefix from
    *path* (which is relative to the repo root) and combine it with
    ``node.module``.
    """
    if node.level == 0:
        return node.module

    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    # Walk up ``node.level`` directories (level=1 → current package)
    for _ in range(node.level):
        if parts:
            parts.pop()
    package_prefix = ".".join(parts)
    if node.module:
        return f"{package_prefix}.{node.module}" if package_prefix else node.module
    return package_prefix or None


def _detect_import_from_errors(path: Path, node: ast.ImportFrom) -> list[GuardError]:
    errors: list[GuardError] = []
    if node.module == "__future__":
        for alias in node.names:
            if alias.name == "annotations":
                errors.append(
                    _make_error(
                        path,
                        node.lineno,
                        "AGW010",
                        (
                            "Do not use `from __future__ import annotations`; "
                            "the project standard is Python 3.10+ native annotations."
                        ),
                    )
                )
    resolved = _resolve_import_from_module(path, node)
    if resolved is not None:
        errors.extend(_detect_import_name_errors(path, resolved, node.lineno))
    return errors


def _detect_ast_errors(path: Path, tree: ast.AST) -> list[GuardError]:
    errors: list[GuardError] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            errors.extend(_detect_function_errors(path, node))
            continue

        if isinstance(node, ast.Call):
            errors.extend(_detect_call_errors(path, node))
            continue

        if isinstance(node, (ast.Assign, ast.AugAssign)):
            errors.extend(_detect_assign_errors(path, node))
            continue

        if isinstance(node, ast.AnnAssign):
            errors.extend(_detect_annassign_errors(path, node))
            continue

        if isinstance(node, ast.Import):
            for alias in node.names:
                errors.extend(_detect_import_name_errors(path, alias.name, node.lineno))

        if isinstance(node, ast.ImportFrom):
            errors.extend(_detect_import_from_errors(path, node))

        if isinstance(node, ast.Attribute):
            errors.extend(_detect_attribute_errors(path, node))
    return errors


def _get_line_budget(path: Path) -> int | None:
    path_str = path.as_posix()
    for pattern, limit in FILE_GROWTH_BUDGETS:
        if pattern.match(path_str):
            return limit
    return None


def _read_head_line_count(path: Path) -> int | None:
    completed = subprocess.run(
        ["git", "show", f"HEAD:{path.as_posix()}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return len(completed.stdout.splitlines())


def _detect_growth_budget_error(path: Path, content: str) -> GuardError | None:
    limit = _get_line_budget(path)
    if limit is None:
        return None

    current_count = len(content.splitlines())
    head_count = _read_head_line_count(path)
    if head_count is None:
        if current_count <= limit:
            return None
    elif current_count <= head_count or current_count <= limit:
        return None

    return GuardError(
        path=path,
        line=1,
        code="AGW004",
        message=(
            f"File grew to {current_count} lines, exceeding the guardrail budget of "
            f"{limit}. Split responsibilities before expanding this hotspot."
        ),
    )


def _find_first_match_line(content: str, pattern: str) -> int | None:
    match = re.search(pattern, content)
    if match is None:
        return None
    return content.count("\n", 0, match.start()) + 1


def _detect_agent_v2_text_errors(path: Path, content: str) -> list[GuardError]:
    if path == Path("scripts/repo_guard.py"):
        return []
    line = _find_first_match_line(content, r"\bagiwo\.agent_v2\b")
    if line is None:
        return []
    return [
        _make_error(
            path,
            line,
            "AGW036",
            (
                "Do not reference agiwo.agent_v2 in docs or code; the canonical "
                "agent package is agiwo.agent."
            ),
        )
    ]


def _detect_agent_config_text_errors(path: Path, content: str) -> list[GuardError]:
    errors: list[GuardError] = []
    if path.as_posix().startswith("console/server/"):
        line = _find_first_match_line(content, r"\bAgentConfigUpdate\b")
        if line is not None:
            errors.append(
                _make_error(
                    path,
                    line,
                    "AGW027",
                    (
                        "Do not reintroduce AgentConfigUpdate; agent config writes "
                        "must use AgentConfigReplace/full PUT semantics."
                    ),
                )
            )
    if path == Path("console/server/services/agent_registry.py"):
        for pattern in (r"\bdef\s+update_agent\b", r"\b_merge_nested_dict\b"):
            line = _find_first_match_line(content, pattern)
            if line is None:
                continue
            errors.append(
                _make_error(
                    path,
                    line,
                    "AGW028",
                    (
                        "AgentRegistry must not reintroduce partial update merge "
                        "helpers; keep agent writes as full replacement only."
                    ),
                )
            )
    return errors


def _detect_feishu_parser_text_errors(path: Path, content: str) -> list[GuardError]:
    errors: list[GuardError] = []
    if path == Path("console/server/channels/feishu/message_builder.py"):
        line = _find_first_match_line(
            content,
            r"server\.channels\.feishu\.message_parser|FeishuMessageParser",
        )
        if line is not None:
            errors.append(
                _make_error(
                    path,
                    line,
                    "AGW029",
                    (
                        "FeishuUserMessageBuilder must depend on "
                        "FeishuContentExtractor/FeishuGroupHistoryStore, not "
                        "the parser facade."
                    ),
                )
            )
    elif path == Path("console/server/channels/feishu/message_parser.py"):
        for pattern in (
            r"\bdef\s+resolve_sender_name\b",
            r"\bdef\s+record_group_message\b",
            r"\bdef\s+get_group_history_lines\b",
            r"\bdef\s+normalize_message_text\b",
            r"_sender_name_cache",
            r"_group_recent_messages",
        ):
            line = _find_first_match_line(content, pattern)
            if line is None:
                continue
            errors.append(
                _make_error(
                    path,
                    line,
                    "AGW030",
                    (
                        "Keep FeishuMessageParser as a thin facade; sender "
                        "resolution, text normalization, and group history "
                        "belong in dedicated collaborators."
                    ),
                )
            )
    return errors


def _detect_session_domain_text_errors(path: Path, content: str) -> list[GuardError]:
    errors: list[GuardError] = []
    if path == Path("console/server/channels/session_binding.py"):
        for pattern in (
            r"\bdef\s+create_initial_session_binding\b",
            r"\bdef\s+create_session_binding\b",
            r"\bdef\s+switch_current_session\b",
            r"\bdef\s+build_session_binding\b",
            r"\bdef\s+set_chat_context_",
            r"\bclass\s+SessionBindingMutation\b",
            r"\bclass\s+SessionSwitchMutation\b",
        ):
            line = _find_first_match_line(content, pattern)
            if line is None:
                continue
            errors.append(
                _make_error(
                    path,
                    line,
                    "AGW033",
                    (
                        "Keep session_binding.py focused on business-level "
                        "session domain operations; do not reintroduce the old "
                        "field-level transition helper set."
                    ),
                )
            )
    elif path == Path("console/server/channels/session_context_service.py"):
        for pattern in (
            r"await self\._store\.upsert_chat_context\(mutation\.",
            r"await self\._store\.upsert_session\(mutation\.",
            r"await self\._store\.upsert_chat_context\(target\)",
            r"await self\._store\.upsert_session\(target\)",
        ):
            line = _find_first_match_line(content, pattern)
            if line is None:
                continue
            errors.append(
                _make_error(
                    path,
                    line,
                    "AGW034",
                    (
                        "SessionContextService must persist session-domain "
                        "actions via the atomic apply_session_mutation() "
                        "boundary, not paired upsert_* calls."
                    ),
                )
            )
    return errors


def _detect_feishu_sdk_text_errors(path: Path, content: str) -> list[GuardError]:
    if not path.as_posix().startswith("console/server/channels/feishu/"):
        return []
    if path == Path("console/server/channels/feishu/sdk_adapter.py"):
        return []

    errors: list[GuardError] = []
    for pattern in (
        r"\._auto_reconnect\b",
        r"lark_ws_client_module\.loop",
    ):
        line = _find_first_match_line(content, pattern)
        if line is None:
            continue
        errors.append(
            _make_error(
                path,
                line,
                "AGW035",
                (
                    "Only sdk_adapter.py may touch Feishu SDK private fields or "
                    "module-level globals; keep other modules on the stable "
                    "FeishuConnection boundary."
                ),
            )
        )
    return errors


def _detect_tool_reference_domain_text_errors(
    path: Path,
    content: str,
) -> list[GuardError]:
    errors: list[GuardError] = []
    if path == Path("console/server/models/agent_config.py"):
        line = _find_first_match_line(content, r"\btools:\s*list\[str\]")
        if line is not None:
            errors.append(
                _make_error(
                    path,
                    line,
                    "AGW036",
                    (
                        "Shared agent config models must not fall back to bare "
                        "list[str] tool semantics; normalize tool references at "
                        "the API boundary before they reach services."
                    ),
                )
            )
    return errors


def _detect_console_tool_catalog_contract_text_errors(
    path: Path,
    content: str,
) -> list[GuardError]:
    errors: list[GuardError] = []
    if path == Path("console/server/tools.py"):
        for pattern in (
            r"def\s+parse_reference\([^\n]+\)\s*->\s*ToolReference\s*\|\s*None",
            r"return\s+None",
        ):
            line = _find_first_match_line(content, pattern)
            if line is None:
                continue
            errors.append(
                _make_error(
                    path,
                    line,
                    "AGW037",
                    (
                        "ConsoleToolCatalog must reject invalid tool refs "
                        "instead of silently returning None/filtering them out."
                    ),
                )
            )
        for pattern in (
            r"\bbuild_tools\(\s*self,\s*\n?\s*tool_names:\s*list\[str\]\s*[,)]",
            r"\bbuild_tools\(\s*self,\s*\n?\s*tool_refs:\s*list\[str\]\s*[,)]",
        ):
            line = _find_first_match_line(content, pattern)
            if line is None:
                continue
            errors.append(
                _make_error(
                    path,
                    line,
                    "AGW038",
                    (
                        "ConsoleToolCatalog.build_tools() must consume the "
                        "formal ToolReference contract, not raw unvalidated "
                        "tool-name lists."
                    ),
                )
            )
    return errors


def _detect_agent_lifecycle_stable_id_errors(
    path: Path,
    content: str,
) -> list[GuardError]:
    if path != Path("console/server/services/agent_lifecycle.py"):
        return []
    if not re.search(r"\bAgent\s*\(", content):
        return []
    if re.search(r"id\s*=\s*id\s+or\s+config\.id", content):
        return []
    line = _find_first_match_line(content, r"\bAgent\s*\(") or 1
    return [
        _make_error(
            path,
            line,
            "AGW043",
            (
                "build_agent must construct Agent with `id=id or config.id` to "
                "guarantee stable agent identity across HTTP requests; a random "
                "default id breaks conversation history continuity."
            ),
        )
    ]


def _detect_agent_runtime_text_errors(path: Path, content: str) -> list[GuardError]:
    if not (
        path.as_posix().startswith("agiwo/")
        or path.as_posix().startswith("tests/")
        or path.as_posix().startswith("console/")
    ):
        return []
    errors: list[GuardError] = []
    line = _find_first_match_line(content, r"\battach_state\s*\(")
    if line is not None:
        errors.append(
            _make_error(
                path,
                line,
                "AGW041",
                (
                    "Do not reintroduce recorder attach-state flows; keep run state, "
                    "recorder setup, and execution ownership in one agent run phase."
                ),
            )
        )

    line = _find_first_match_line(content, r"\bexecution_bootstrap\b")
    if line is not None:
        errors.append(
            _make_error(
                path,
                line,
                "AGW042",
                (
                    "Do not reintroduce execution_bootstrap; keep run preparation "
                    "inside agiwo.agent.agent/execute_run instead of reviving a "
                    "separate bootstrap layer."
                ),
            )
        )
    return errors


def _detect_agents_router_tool_text_errors(
    path: Path,
    content: str,
) -> list[GuardError]:
    errors: list[GuardError] = []
    if path == Path("console/server/routers/agents.py"):
        for pattern in (
            r"\bget_available_builtin_tools\b",
            r"f\"agent:",
        ):
            line = _find_first_match_line(content, pattern)
            if line is None:
                continue
            errors.append(
                _make_error(
                    path,
                    line,
                    "AGW032",
                    (
                        "Agents router must list tools via ConsoleToolCatalog; "
                        "do not rebuild builtin/agent tool payloads inline."
                    ),
                )
            )
    return errors


def _detect_console_tool_catalog_text_errors(
    path: Path, content: str
) -> list[GuardError]:
    errors = _detect_tool_reference_domain_text_errors(path, content)
    errors.extend(_detect_console_tool_catalog_contract_text_errors(path, content))
    errors.extend(_detect_agents_router_tool_text_errors(path, content))
    return errors


def _detect_text_guard_errors(path: Path, content: str) -> list[GuardError]:
    errors = _detect_agent_v2_text_errors(path, content)
    errors.extend(_detect_agent_config_text_errors(path, content))
    errors.extend(_detect_feishu_parser_text_errors(path, content))
    errors.extend(_detect_session_domain_text_errors(path, content))
    errors.extend(_detect_feishu_sdk_text_errors(path, content))
    errors.extend(_detect_agent_runtime_text_errors(path, content))
    errors.extend(_detect_console_tool_catalog_text_errors(path, content))
    errors.extend(_detect_agent_lifecycle_stable_id_errors(path, content))
    return errors


def _check_file(path: Path) -> list[GuardError]:
    content = (ROOT / path).read_text(encoding="utf-8")
    try:
        tree = ast.parse(content, filename=path.as_posix())
    except SyntaxError as exc:
        return [
            GuardError(
                path=path,
                line=exc.lineno or 1,
                code="AGW000",
                message=str(exc),
            )
        ]

    errors = _detect_ast_errors(path, tree)
    errors.extend(_detect_text_guard_errors(path, content))
    growth_error = _detect_growth_budget_error(path, content)
    if growth_error is not None:
        errors.append(growth_error)
    return errors


def _resolve_target_paths(files: list[str], *, check_all: bool) -> list[Path]:
    if check_all:
        return iter_all_python_paths()
    if files:
        return collect_python_paths(normalize_paths(files))
    return collect_python_paths(collect_changed_paths())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repository-specific lint checks.")
    parser.add_argument("files", nargs="*", help="Explicit files to check.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all Python files instead of only changed files.",
    )
    args = parser.parse_args()

    paths = _resolve_target_paths(args.files, check_all=args.all)
    if not paths:
        print("repo_guard: no Python files to check")
        return

    findings: list[GuardError] = []
    for path in paths:
        findings.extend(_check_file(path))

    if not findings:
        print(f"repo_guard: ok ({len(paths)} files)")
        return

    blocking = [
        finding for finding in findings if finding.code not in WARNING_ONLY_CODES
    ]
    warnings = [finding for finding in findings if finding.code in WARNING_ONLY_CODES]

    for warning in sorted(
        warnings,
        key=lambda item: (item.path.as_posix(), item.line, item.code),
    ):
        print(
            f"{warning.path}:{warning.line}: warning {warning.code} {warning.message}"
        )

    if not blocking:
        print(f"repo_guard: ok ({len(paths)} files, {len(warnings)} warnings)")
        return

    for error in sorted(
        blocking,
        key=lambda item: (item.path.as_posix(), item.line, item.code),
    ):
        print(f"{error.path}:{error.line}: {error.code} {error.message}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
