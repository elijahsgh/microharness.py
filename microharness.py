#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "pydantic-ai[web-fetch]>=2.61.0",
#     "pydantic-ai-harness[coder,acp,code-mode]>=0.61.0",
#     "opentelemetry-sdk>=1.30.0",
#     "opentelemetry-exporter-otlp>=1.30.0",
#     "typer>=0.15.0",
#     "pyyaml>=6.0.2",
# ]
# ///
"""Microharness: A single-file Pydantic AI Harness application with ACP and single-query modes.

Run directly with uv:
    uv run microharness.py                        # Starts ACP server over stdio (default agent)
    uv run microharness.py acp -a default         # Starts ACP server for a named agent directory
    uv run microharness.py query "Your prompt"    # Runs a single query and prints the output
    uv run microharness.py init-spec -a my-agent  # Scaffolds agents/<agent>/ (agent.yaml, rules/, knowledge/, tools/, hooks/, skills/)
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import fnmatch
import importlib
import importlib.util
import inspect
import json
import os
import re
import sqlite3
import subprocess
import sys
import warnings
import zipfile
from collections.abc import Callable, Generator, Hashable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

import acp
import httpx2
import typer
import yaml
from acp import schema as acp_schema
from bs4 import BeautifulSoup
from markdownify import markdownify as _html_to_md
from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    AgentSpec,
    ApprovalRequired,
    FunctionToolset,
    ModelRequestContext,
    ModelResolutionContext,
    ModelRetry,
    RunContext,
    Tool,
)
from pydantic_ai._http import create_async_httpx2_client
from pydantic_ai._ssrf import (
    _domain_key,
    _keeps_credentials,
    _read_capped_body,
    build_url_with_ip,
    extract_host_and_port,
    resolve_redirect_url,
    validate_and_resolve_url,
)
from pydantic_ai.capabilities import (
    AbstractCapability,
    Capability,
    CapabilityOrdering,
    CombinedCapability,
    Hooks,
    Instrumentation,
    LocalWorkspace,
    Toolset,
    WrapperCapability,
    WrapRunHandler,
    WrapToolExecuteHandler,
)
from pydantic_ai.capabilities import (
    WebFetch as _PydanticAIWebFetch,
)
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import OutputContext
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.code_mode import CodeMode as _CodeMode
from pydantic_ai_harness.coder import Coder as _Coder
from pydantic_ai_harness.experimental import HarnessExperimentalWarning
from pydantic_ai_harness.planning import (
    InMemoryPlanStore,
    PlanItem,
    Planning,
    PlanStore,
    TaskStatus,
)
from pydantic_ai_harness.researcher import Researcher as _Researcher
from pydantic_ai_harness.skills._loader import SkillDefinition, parse_skill
from pydantic_ai_harness.subagents import SubAgents as _SubAgents

sys.dont_write_bytecode = True
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
warnings.filterwarnings("ignore", category=HarnessExperimentalWarning)

from pydantic_ai_harness.experimental.acp import (
    AcpSession,
    AcpSessionConfig,
    InMemorySessionStore,
    ToolCallPermission,
    ToolCallPresentation,
    acp_filesystem,
    acp_terminal,
    chain_presenters,
    default_coding_presenter,
    default_permission_scope,
    run_acp_stdio_sync,
)

SCRIPT_DIR = Path(__file__).resolve().parent
INITIAL_CWD = Path.cwd().resolve()
DEFAULT_AGENTS_DIR = Path("agents")
DEFAULT_AGENT_NAME = "default"
DEFAULT_SPEC_FILENAME = "agent.yaml"
DEFAULT_WORKDIR = Path("workdir")
DEFAULT_WORKDIR_SPEC = "./workdir"
DEFAULT_TOOLS_DIR = Path("tools")
DEFAULT_HOOKS_DIR = Path("hooks")
DEFAULT_SKILLS_DIR = Path("skills")
DEFAULT_RULES_DIR = Path("rules")
DEFAULT_KNOWLEDGE_DIR = Path("knowledge")
DEFAULT_WEB_DIR = Path("web")
DEFAULT_LEMONADE_BASE_URL = os.environ.get(
    "LEMONADE_BASE_URL", "http://localhost:13305/v1"
)
DEFAULT_MODEL = "lemonade:unsloth/Qwen3.8-27B-GGUF"

ApprovalPreset = Literal["write", "shell", "all", "custom", "none"]
ApprovalRememberScope = Literal["exact", "tool", "command_prefix"]
ApprovalSubagentMode = Literal["approve_delegation", "allow", "deny"]
ApprovalCliMode = Literal["allow", "prompt", "deny"]

_AGENT_DIR_CTX: ContextVar[Path] = ContextVar(
    "_AGENT_DIR_CTX", default=SCRIPT_DIR / DEFAULT_AGENTS_DIR / DEFAULT_AGENT_NAME
)
_OTEL_CLI_CTX: ContextVar[dict[str, Any] | None] = ContextVar(
    "_OTEL_CLI_CTX", default=None
)
_ACP_MODE_CTX: ContextVar[bool] = ContextVar("_ACP_MODE_CTX", default=False)
_APPROVALS_CLI_CTX: ContextVar[ApprovalPreset | None] = ContextVar(
    "_APPROVALS_CLI_CTX", default=None
)
_RUN_DEPTH_CTX: ContextVar[int] = ContextVar("_RUN_DEPTH_CTX", default=0)
_IN_RUN_CODE_CTX: ContextVar[bool] = ContextVar("_IN_RUN_CODE_CTX", default=False)


def _ensure_workspace_dir(path: str | Path = DEFAULT_WORKDIR) -> Path:
    """Resolve a workspace directory path against `INITIAL_CWD` and ensure it exists on disk."""
    expanded = Path(path).expanduser()
    resolved = (
        expanded if expanded.is_absolute() else (INITIAL_CWD / expanded)
    ).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def make_default_spec_dict(agent_name: str = DEFAULT_AGENT_NAME) -> dict[str, Any]:
    """Build the default AgentSpec dictionary for a named agent."""
    return {
        "name": agent_name,
        "model": DEFAULT_MODEL,
        "instructions": (
            "You are an autonomous coding and research assistant running inside Microharness."
        ),
        "capabilities": [
            {"Lemonade": {"base_url": DEFAULT_LEMONADE_BASE_URL}},
            {"LocalWorkspace": DEFAULT_WORKDIR_SPEC},
            {"Rules": "./rules"},
            {"Knowledge": "./knowledge"},
            {"WebFetch": "./web"},
            {"LocalTools": "./tools"},
            {"Skills": "./skills"},
            {"Hooks": {"modules": ["./hooks"]}},
            {
                "Approvals": {
                    "enabled": True,
                    "preset": "write",
                    "remember_scope": "exact",
                    "approve_web_fetch_save": True,
                    "subagent_mode": "approve_delegation",
                    "cli_mode": "allow",
                }
            },
            {
                "OpenTelemetry": {
                    "enabled": True,
                    "protocol": "http/protobuf",
                    "service_name": agent_name,
                    "include_content": True,
                    "include_binary_content": True,
                    "include_model_request_parameters": True,
                    "version": 5,
                    "instrument_all": True,
                }
            },
            {"Coder": {"sub_agents": True, "caching": True}},
            "Planning",
            "CodeMode",
        ],
    }


def _env_bool(name: str, default: bool | None = None) -> bool | None:
    """Parse a boolean environment variable (`1/true/yes/on` vs `0/false/no/off`)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    norm = raw.strip().lower()
    if norm in ("1", "true", "yes", "on"):
        return True
    if norm in ("0", "false", "no", "off"):
        return False
    return default


def _parse_kv_pairs(raw: str | list[str] | dict[str, str] | None) -> dict[str, str]:
    """Parse comma-separated `key=value` pairs or a dict into a `dict[str, str]`."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    items = raw.split(",") if isinstance(raw, str) else raw
    return {
        k.strip(): v.strip()
        for item in items
        if "=" in item
        for k, v in [item.split("=", 1)]
        if k.strip()
    }


def _as_str_list(val: str | Sequence[str]) -> list[str]:
    """Normalize a string or sequence of strings into a list of strings."""
    return [val] if isinstance(val, str) else list(val)


def _normalize_name_set(names: Sequence[str] | None) -> set[str] | None:
    """Normalize an optional list of names into a lowercase set."""
    return {n.strip().lower() for n in names} if names is not None else None


def _make_name_filter(
    include: Sequence[str] | None, exclude: Sequence[str] | None
) -> Callable[[str], bool]:
    """Build a case-insensitive predicate checking `include` and `exclude` name sets."""
    inc_set = _normalize_name_set(include)
    exc_set = _normalize_name_set(exclude) or set()

    def _allowed(name: str) -> bool:
        norm = name.strip().lower()
        return (inc_set is None or norm in inc_set) and norm not in exc_set

    return _allowed


@contextlib.contextmanager
def _agent_dir_scope(agent_dir: Path | None) -> Generator[None]:
    """Temporarily bind `_AGENT_DIR_CTX` to `agent_dir` when provided."""
    token = _AGENT_DIR_CTX.set(agent_dir) if agent_dir is not None else None
    try:
        yield
    finally:
        if token is not None:
            _AGENT_DIR_CTX.reset(token)


def _record_mtime(snapshot: list[tuple[str, int]], *paths: Path) -> None:
    """Append `(path_str, mtime_ns)` entries to `snapshot`, ignoring OSError."""
    for p in paths:
        with contextlib.suppress(OSError):
            snapshot.append((str(p), p.stat().st_mtime_ns))


def _is_default_subdir_ref(raw: str | Path, subdir: Path) -> bool:
    """Return True if `raw` refers to the standard default subdirectory (`./<name>` or `<name>`)."""
    return str(raw) in {str(subdir), f"./{subdir.name}", subdir.name}


def _resolve_agent_relative_path(raw: str | Path) -> Path:
    """Resolve a relative path against the active agent's directory first, then INITIAL_CWD."""
    p = Path(raw)
    if p.is_absolute():
        return p
    agent_dir = _AGENT_DIR_CTX.get()
    agent_candidate = agent_dir / p
    init_candidate = INITIAL_CWD / p
    if agent_candidate.exists() or not init_candidate.exists():
        return agent_candidate
    return init_candidate


def _iter_public_files(
    directory: Path, pattern: str = "*.py", recursive: bool = True
) -> list[Path]:
    """Return sorted matching files under `directory`, skipping hidden or `_`-prefixed paths."""
    if not directory.is_dir():
        return []
    iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)
    return [
        f.resolve()
        for f in sorted(iterator)
        if f.is_file()
        and not any(
            part.startswith(("_", ".")) for part in f.relative_to(directory).parts
        )
    ]


_MODULE_CACHE: dict[Path, tuple[int, Any]] = {}


def _load_python_module_from_file(file_path: Path) -> Any:
    """Load a Python module from a file path without writing `.pyc` / `__pycache__` files, cached by mtime."""
    resolved = file_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Python file not found: {file_path}")
    mtime_ns = resolved.stat().st_mtime_ns
    cached = _MODULE_CACHE.get(resolved)
    if cached is not None and cached[0] == mtime_ns:
        return cached[1]
    module_name = f"_microharness_dynamic_{resolved.stem}_{abs(hash(str(resolved)))}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[resolved] = (mtime_ns, module)
    return module


def _load_python_object(target: str) -> Any:
    """Load a Python callable or object from 'path/to/file.py:attr' or 'module.name:attr'."""
    mod_part, attr_name = target.rsplit(":", 1) if ":" in target else (target, None)
    file_candidate = _resolve_agent_relative_path(mod_part)
    if file_candidate.suffix == ".py" or file_candidate.exists():
        module = _load_python_module_from_file(file_candidate)
    else:
        module = importlib.import_module(mod_part)
    return getattr(module, attr_name) if attr_name is not None else module


def _resolve_lemonade_model(
    model_id: str,
    base_url: str = DEFAULT_LEMONADE_BASE_URL,
    api_key: str = "lemonade",
) -> Model | None:
    """Resolve a `lemonade:<model>` or bare model ID into an OpenAIChatModel, or None."""
    if model_id == "test":
        return None
    if model_id.startswith("lemonade:"):
        name = model_id.removeprefix("lemonade:")
    elif ":" not in model_id:
        name = model_id
    else:
        return None
    return OpenAIChatModel(
        name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
    )


@dataclass
class _SpecCombinedCapability(CombinedCapability[Any]):
    """Base dataclass for CombinedCapability wrappers usable in YAML Agent Specs."""

    capabilities: Sequence[AbstractCapability[Any]] = field(
        default=(), init=False, repr=False
    )


@dataclass
class Lemonade(AbstractCapability[Any]):
    """Connects `lemonade:<model>` (or bare model IDs) to a local Lemonade AI Server."""

    base_url: str = DEFAULT_LEMONADE_BASE_URL
    api_key: str = "lemonade"

    async def resolve_model_id(
        self,
        ctx: ModelResolutionContext[Any],
        *,
        model_id: str,
    ) -> Model | None:
        return _resolve_lemonade_model(
            model_id, base_url=self.base_url, api_key=self.api_key
        )


@dataclass
class LocalWorkspaceSpec(LocalWorkspace[Any]):
    """Spec-compatible `LocalWorkspace` that defaults to `./workdir` and ensures the directory exists."""

    working_dir: str | Path = DEFAULT_WORKDIR_SPEC

    def __post_init__(self) -> None:
        self.working_dir = str(_ensure_workspace_dir(self.working_dir))
        super().__post_init__()

    @classmethod
    def get_serialization_name(cls) -> str:
        return "LocalWorkspace"


def _is_protected_harness_path(target: Path, agent_dir: Path) -> bool:
    """Return True if `target` is a protected Microharness source, schema, or agent config path."""
    resolved = target.resolve()
    resolved_agent = agent_dir.resolve()
    if resolved in (
        SCRIPT_DIR / "microharness.py",
        SCRIPT_DIR / "agent_schema.json",
        resolved_agent / DEFAULT_SPEC_FILENAME,
    ) or resolved.is_relative_to((SCRIPT_DIR / DEFAULT_AGENTS_DIR).resolve()):
        return True
    return any(
        resolved.is_relative_to(resolved_agent / subdir)
        for subdir in (
            DEFAULT_RULES_DIR,
            DEFAULT_HOOKS_DIR,
            DEFAULT_TOOLS_DIR,
            DEFAULT_SKILLS_DIR,
            DEFAULT_KNOWLEDGE_DIR,
            DEFAULT_WEB_DIR,
        )
    )


async def _resolve_ctx_workspace(ctx: RunContext[Any]) -> Path:
    """Return the resolved active workspace directory for `ctx`."""
    if ctx.workspace is not None:
        return Path(await ctx.workspace.working_dir()).resolve()
    return _ensure_workspace_dir()


@dataclass
class _HarnessSelfProtectionCapability(AbstractCapability[Any]):
    """Syncs process cwd / MICROHARNESS_WORKDIR to the run workspace and blocks writes to Microharness's own files."""

    agent_dir: Path
    protect_harness_files: bool = True
    sync_process_cwd: bool = True

    async def before_run(self, ctx: RunContext[Any]) -> None:
        ws_dir = await _resolve_ctx_workspace(ctx)
        os.environ["MICROHARNESS_WORKDIR"] = str(ws_dir)
        if self.sync_process_cwd and ws_dir.is_dir():
            with contextlib.suppress(OSError):
                os.chdir(ws_dir)

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        if self.protect_harness_files and call.tool_name in (
            "write_file",
            "edit_file",
            "create_directory",
        ):
            ws_dir = await _resolve_ctx_workspace(ctx)
            for raw_p in _extract_paths_from_tool_args(args):
                cand = (
                    Path(raw_p) if os.path.isabs(raw_p) else (ws_dir / raw_p)
                ).resolve()
                if _is_protected_harness_path(cand, self.agent_dir):
                    raise ModelRetry(
                        f"Access denied: modifying protected Microharness file '{raw_p}' ({cand}) is not permitted."
                    )
        return args


@dataclass
class CoderSpec(_SpecCombinedCapability):
    """Spec-compatible wrapper for `pydantic_ai_harness.coder.Coder`."""

    instructions: str | None = None
    unrestricted_filesystem: bool = False
    repo_context: bool = True
    sub_agents: bool = True
    agent_folders: str | list[str] | None = None
    caching: bool = True
    protect_harness_files: bool = True
    sync_process_cwd: bool = True

    def __post_init__(self) -> None:
        agent_dir = _AGENT_DIR_CTX.get()
        self.capabilities = [
            _HarnessSelfProtectionCapability(
                agent_dir=agent_dir,
                protect_harness_files=self.protect_harness_files,
                sync_process_cwd=self.sync_process_cwd,
            ),
            _Coder(
                instructions=self.instructions,
                unrestricted_filesystem=self.unrestricted_filesystem,
                repo_context=self.repo_context,
                sub_agents=self.sub_agents,
                agent_folders=self.agent_folders,
                caching=self.caching,
            ),
        ]
        super().__post_init__()

    @classmethod
    def get_serialization_name(cls) -> str:
        return "Coder"


@dataclass
class ResearcherSpec(_SpecCombinedCapability):
    """Spec-compatible wrapper for `pydantic_ai_harness.researcher.Researcher`."""

    instructions: str | None = None

    def __post_init__(self) -> None:
        cap = (
            _Researcher(instructions=self.instructions)
            if self.instructions is not None
            else _Researcher()
        )
        self.capabilities = [cap]
        super().__post_init__()

    @classmethod
    def get_serialization_name(cls) -> str:
        return "Researcher"


@dataclass
class SubAgentsSpec(_SpecCombinedCapability):
    """Spec-compatible wrapper for `pydantic_ai_harness.subagents.SubAgents`."""

    include_self: bool = True
    agent_folders: str | list[str] | None = "auto"
    max_depth: int = 3

    def __post_init__(self) -> None:
        self.capabilities = [
            _SubAgents(
                include_self=self.include_self,
                agent_folders=self.agent_folders,
                max_depth=self.max_depth,
            )
        ]
        super().__post_init__()

    @classmethod
    def get_serialization_name(cls) -> str:
        return "SubAgents"


_NATIVE_HARNESS_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "write_file",
        "edit_file",
        "list_files",
        "list_directory",
        "grep",
        "search_files",
        "find_files",
        "create_directory",
        "file_info",
        "shell",
        "run_command",
        "start_command",
        "check_command",
        "stop_command",
        "delegate_task",
        "load_capability",
        "search_tools",
        "write_plan",
        "read_plan",
        "add_task",
        "update_task_status",
        "update_task_statuses",
        "remove_task",
        "add_subtask",
        "set_dependency",
        "get_available_tasks",
        "list_skills",
        "load_skill",
        "read_skill_resource",
        "web_fetch",
    }
)


@dataclass
class CodeModeSpec(_SpecCombinedCapability):
    """Spec-compatible wrapper for `pydantic_ai_harness.code_mode.CodeMode`."""

    tools: Literal["all"] | list[str] = "all"
    max_retries: int = 3
    max_tool_calls: int = 100
    eager: bool = False
    speculate: list[str] | Literal["declared"] | None = None
    monty_sandbox_url: str | None = None
    dynamic_catalog: bool = False
    keep_local_tools_native: bool = True
    keep_coder_tools_native: bool = True

    def __post_init__(self) -> None:
        if self.tools == "all" and (
            self.keep_local_tools_native or self.keep_coder_tools_native
        ):

            def _selector(ctx: RunContext[Any], tool_def: ToolDefinition) -> bool:
                meta = tool_def.metadata or {}
                if "code_mode" in meta:
                    return bool(meta["code_mode"])
                if self.keep_local_tools_native and meta.get(
                    "microharness_local_tool", False
                ):
                    return False
                return not (
                    self.keep_coder_tools_native
                    and tool_def.name in _NATIVE_HARNESS_TOOLS
                )

            effective_tools: Any = _selector
        else:
            effective_tools = self.tools

        self.capabilities = [
            _CodeMode(
                tools=effective_tools,
                max_retries=self.max_retries,
                max_tool_calls=self.max_tool_calls,
                eager=self.eager,
                speculate=self.speculate,
                monty_sandbox_url=self.monty_sandbox_url,
                dynamic_catalog=self.dynamic_catalog,
            )
        ]
        super().__post_init__()

    @classmethod
    def get_serialization_name(cls) -> str:
        return "CodeMode"


def _collect_python_tools(
    targets: Sequence[str | Path],
    *,
    pattern: str = "*.py",
    recursive: bool = True,
    prefix_with_module: bool = False,
    fixed_prefix: str | None = None,
    requires_approval: bool = False,
    timeout: float | None = None,
    max_retries: int | None = None,
    include_module_docstrings: bool = True,
    toolset_id: str | None = None,
) -> tuple[
    list[AbstractToolset[Any]], list[AbstractCapability[Any]], list[str], list[str]
]:
    """Discover Python tools from files/directories/module targets.

    Returns `(toolsets, extra_capabilities, module_instructions, tool_summaries)`.
    """
    toolset = FunctionToolset(
        max_retries=max_retries if max_retries is not None else 1,
        timeout=timeout,
        requires_approval=requires_approval,
        id=toolset_id,
    )
    extra_toolsets: list[AbstractToolset[Any]] = []
    extra_capabilities: list[AbstractCapability[Any]] = []
    module_instructions: list[str] = []
    tool_summaries: list[str] = []
    registered_tool_count = 0

    def _prefix_for(module_stem: str | None) -> str | None:
        if fixed_prefix:
            return fixed_prefix
        if prefix_with_module and module_stem:
            return module_stem
        return None

    def _record_summary(
        fn: Any, effective_name: str, is_code_mode: bool = False
    ) -> None:
        try:
            sig = inspect.signature(fn)
            params = [
                str(p)
                for i, p in enumerate(sig.parameters.values())
                if not (
                    i == 0 and (p.name == "ctx" or "RunContext" in str(p.annotation))
                )
            ]
            sig_str = f"{effective_name}({', '.join(params)})"
            if sig.return_annotation is not inspect.Signature.empty:
                ret = (
                    sig.return_annotation.__name__
                    if hasattr(sig.return_annotation, "__name__")
                    else str(sig.return_annotation)
                )
                sig_str += f" -> {ret}"
        except (ValueError, TypeError):
            sig_str = f"{effective_name}(...)"
        doc_first_line = (inspect.getdoc(fn) or "").strip().splitlines()
        if is_code_mode:
            summary = f"- `await {sig_str}` *(CodeMode — invoke inside `run_code`)*"
        else:
            summary = f"- `{sig_str}` *(Native tool — invoke directly)*"
        if doc_first_line:
            summary += f": {doc_first_line[0].strip()}"
        tool_summaries.append(summary)

    def _stamp_tool_meta(t_obj: Tool[Any], default_cm: bool) -> bool:
        existing_meta = t_obj.metadata or {}
        is_cm = bool(existing_meta.get("code_mode", default_cm))
        t_obj.metadata = {
            **existing_meta,
            "microharness_local_tool": True,
            "code_mode": is_cm,
        }
        return is_cm

    def _add_item(
        item: Any, module_stem: str | None = None, default_code_mode: bool = False
    ) -> None:
        nonlocal registered_tool_count
        prefix = _prefix_for(module_stem)
        if isinstance(item, AbstractCapability):
            extra_capabilities.append(item)
        elif isinstance(item, AbstractToolset):
            extra_toolsets.append(item)
        elif isinstance(item, Tool):
            if prefix and not item.name.startswith(f"{prefix}__"):
                item.name = f"{prefix}__{item.name}"
            is_cm = _stamp_tool_meta(item, default_code_mode)
            toolset.add_tool(item)
            registered_tool_count += 1
            _record_summary(item.function, item.name, is_code_mode=is_cm)
        elif callable(item):
            fn_name = getattr(item, "__name__", "tool")
            custom_name = (
                f"{prefix}__{fn_name}"
                if (prefix and not fn_name.startswith(f"{prefix}__"))
                else None
            )
            is_cm = bool(getattr(item, "__code_mode__", default_code_mode))
            toolset.add_function(
                item,
                name=custom_name,
                retries=max_retries,
                timeout=timeout,
                requires_approval=requires_approval,
                metadata={"microharness_local_tool": True, "code_mode": is_cm},
            )
            registered_tool_count += 1
            _record_summary(item, custom_name or fn_name, is_code_mode=is_cm)

    def _load_file(py_file: Path) -> None:
        nonlocal registered_tool_count
        module = _load_python_module_from_file(py_file)
        stem = py_file.stem
        mod_code_mode = bool(
            getattr(module, "CODE_MODE", getattr(module, "__code_mode__", False))
        )

        if include_module_docstrings:
            doc = (
                getattr(module, "INSTRUCTIONS", None)
                or getattr(module, "__instructions__", None)
                or inspect.getdoc(module)
            )
            if isinstance(doc, str) and doc.strip():
                module_instructions.append(f"### Tools ({py_file.name})\n{doc.strip()}")

        if hasattr(module, "toolset") and isinstance(module.toolset, AbstractToolset):
            extra_toolsets.append(module.toolset)
            return

        if hasattr(module, "register_tools") and callable(module.register_tools):
            before_names = set(toolset.tools.keys())
            res = module.register_tools(toolset)
            if isinstance(res, AbstractToolset) and res is not toolset:
                extra_toolsets.append(res)
            elif isinstance(res, AbstractCapability):
                extra_capabilities.append(res)
            new_names = set(toolset.tools.keys()) - before_names
            registered_tool_count += len(new_names)
            for t_name in sorted(new_names):
                t_obj = toolset.tools[t_name]
                is_cm = _stamp_tool_meta(t_obj, mod_code_mode)
                _record_summary(t_obj.function, t_name, is_code_mode=is_cm)
            return

        explicit_list = getattr(module, "__tools__", getattr(module, "TOOLS", None))
        if isinstance(explicit_list, Sequence) and not isinstance(
            explicit_list, (str, bytes)
        ):
            for entry in explicit_list:
                _add_item(entry, module_stem=stem, default_code_mode=mod_code_mode)
            return

        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if (
                not name.startswith("_")
                and getattr(obj, "__module__", None) == module.__name__
            ):
                _add_item(obj, module_stem=stem, default_code_mode=mod_code_mode)

    for target in targets:
        target_str = str(target)
        if ":" in target_str and not _resolve_agent_relative_path(target).exists():
            mod_part = target_str.rsplit(":", 1)[0]
            _add_item(_load_python_object(target_str), module_stem=Path(mod_part).stem)
            continue

        candidate = (
            target
            if isinstance(target, Path) and target.is_absolute()
            else _resolve_agent_relative_path(target)
        )
        if candidate.is_file():
            _load_file(candidate)
        elif candidate.is_dir():
            for py_file in _iter_public_files(candidate, pattern, recursive):
                _load_file(py_file)
        elif not _is_default_subdir_ref(target_str, DEFAULT_TOOLS_DIR):
            raise FileNotFoundError(f"Tools path not found: {target}")

    all_toolsets: list[AbstractToolset[Any]] = (
        [toolset] if registered_tool_count > 0 else []
    ) + extra_toolsets
    return all_toolsets, extra_capabilities, module_instructions, tool_summaries


@dataclass
class SkillsSpec(_SpecCombinedCapability):
    """Discovers Agent Skills (`SKILL.md` + optional `tools.py` / `tools/*.py`) relative to the agent directory."""

    directories: str | list[str] = "./skills"
    include: list[str] | None = None
    exclude: list[str] | None = None
    missing_directories: Literal["error", "skip"] = "skip"
    duplicate_names: Literal["error", "keep_first"] = "keep_first"
    unknown_names: Literal["error", "skip"] = "skip"
    prefix_tools_with_skill: bool = False

    def __post_init__(self) -> None:
        raw_dirs = _as_str_list(self.directories)
        discovered_by_name: dict[str, tuple[Path, SkillDefinition]] = {}

        for raw_dir in raw_dirs:
            lib_dir = _resolve_agent_relative_path(raw_dir).resolve()
            if not lib_dir.exists():
                if self.missing_directories == "error" and not _is_default_subdir_ref(
                    raw_dir, DEFAULT_SKILLS_DIR
                ):
                    raise ValueError(
                        f"Skill library directory does not exist: {lib_dir}"
                    )
                continue
            if not lib_dir.is_dir() or (lib_dir / "SKILL.md").is_file():
                msg = f"Skill library path is not a directory of skills: {lib_dir}"
                if self.missing_directories == "error":
                    raise ValueError(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
                continue

            for child in sorted(lib_dir.iterdir()):
                if not child.is_dir() or child.name.startswith((".", "_")):
                    continue
                skill_md = child / "SKILL.md"
                if not skill_md.is_file():
                    continue
                try:
                    text = skill_md.read_text(encoding="utf-8")
                    skill_def = parse_skill(text, str(skill_md.resolve()))
                except (UnicodeDecodeError, ValueError) as err:
                    warnings.warn(
                        f"Skipping {skill_md}: {err}", UserWarning, stacklevel=2
                    )
                    continue

                if skill_def.name in discovered_by_name:
                    prev_dir, prev_def = discovered_by_name[skill_def.name]
                    if (
                        prev_dir.resolve() != child.resolve()
                        and prev_def.body != skill_def.body
                    ):
                        dup_msg = f"Duplicate skill name {skill_def.name!r} in {skill_md} (already defined in {prev_def.path})"
                        if self.duplicate_names == "error":
                            raise ValueError(dup_msg)
                        warnings.warn(dup_msg, UserWarning, stacklevel=2)
                    continue
                discovered_by_name[skill_def.name] = (child, skill_def)

        available_names = set(discovered_by_name.keys())
        inc_set = _normalize_name_set(self.include)
        exc_set = _normalize_name_set(self.exclude) or set()

        if self.unknown_names == "error":
            for label, sel in (("include", inc_set), ("exclude", exc_set)):
                if sel:
                    unknown = sorted(sel - available_names)
                    if unknown:
                        raise ValueError(
                            f"Unknown skills in {label}: {', '.join(unknown)}"
                        )

        selected_names = (
            (inc_set & available_names)
            if inc_set is not None
            else (available_names - exc_set)
        )
        skill_capabilities: list[AbstractCapability[Any]] = []

        for name in sorted(selected_names):
            skill_dir, skill_def = discovered_by_name[name]
            if not skill_def.model_invocable:
                continue

            tool_targets: list[Path] = []
            if (skill_dir / "tools.py").is_file():
                tool_targets.append(skill_dir / "tools.py")
            if (skill_dir / "tools").is_dir():
                tool_targets.append(skill_dir / "tools")

            prefix = name.replace("-", "_") if self.prefix_tools_with_skill else None
            toolsets, extra_caps, mod_instructions, tool_summaries = (
                _collect_python_tools(
                    tool_targets,
                    fixed_prefix=prefix,
                    toolset_id=name,
                )
                if tool_targets
                else ([], [], [], [])
            )

            rendered_parts = [skill_def.render()]
            if mod_instructions:
                rendered_parts.extend(mod_instructions)
            if tool_summaries:
                rendered_parts.append(
                    "## Bundled Skill Tools\n"
                    "Loading this skill unlocked the following Python tools:\n"
                    + "\n".join(tool_summaries)
                )

            skill_capabilities.append(
                Capability(
                    id=skill_def.name,
                    description=skill_def.description.replace("\n", "\n  "),
                    instructions="\n\n".join(rendered_parts),
                    toolsets=toolsets or None,
                    defer_loading=True,
                )
            )
            skill_capabilities.extend(extra_caps)

        self.capabilities = skill_capabilities
        super().__post_init__()

    @classmethod
    def get_serialization_name(cls) -> str:
        return "Skills"


@dataclass
class LocalToolsSpec(_SpecCombinedCapability):
    """Discovers and registers local Python functions/toolsets from a directory of `.py` files or module specs."""

    paths: str | list[str] = "./tools"
    pattern: str = "*.py"
    recursive: bool = True
    prefix_with_module: bool = False
    requires_approval: bool = False
    timeout: float | None = None
    max_retries: int | None = None
    include_module_docstrings: bool = True
    _combined_instructions: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        raw_paths = _as_str_list(self.paths)
        toolsets, extra_capabilities, module_instructions, tool_summaries = (
            _collect_python_tools(
                raw_paths,
                pattern=self.pattern,
                recursive=self.recursive,
                prefix_with_module=self.prefix_with_module,
                requires_approval=self.requires_approval,
                timeout=self.timeout,
                max_retries=self.max_retries,
                include_module_docstrings=self.include_module_docstrings,
            )
        )
        parts = list(module_instructions)
        if tool_summaries:
            parts.append("## Local Tools\n" + "\n".join(tool_summaries))
        self._combined_instructions = "\n\n".join(parts) if parts else None
        self.capabilities = [*(Toolset(ts) for ts in toolsets), *extra_capabilities]
        super().__post_init__()

    def get_instructions(self) -> str | None:
        return self._combined_instructions

    @classmethod
    def get_serialization_name(cls) -> str:
        return "LocalTools"


RuleTrigger = Literal["always_on", "glob", "model_decision", "manual"]

_RULE_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL
)
_TRIGGER_ALIASES: dict[str, RuleTrigger] = {
    "always_on": "always_on",
    "always": "always_on",
    "glob": "glob",
    "file": "glob",
    "files": "glob",
    "paths": "glob",
    "model_decision": "model_decision",
    "model": "model_decision",
    "agent": "model_decision",
    "manual": "manual",
    "keyword": "manual",
    "keywords": "manual",
    "prompt": "manual",
}


@dataclass(frozen=True)
class RuleDefinition:
    """Parsed rule metadata and Markdown body from a `<agent>/rules/*.md` file."""

    name: str
    trigger: RuleTrigger
    body: str
    path: Path
    description: str | None = None
    globs: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    _mention_re: re.Pattern[str] = field(init=False, repr=False, compare=False)
    _keywords_lower: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_mention_re",
            re.compile(rf"(?<!\w)@{re.escape(self.name)}(?!\w)"),
        )
        object.__setattr__(
            self,
            "_keywords_lower",
            tuple(kw.lower() for kw in self.keywords),
        )

    def matches_prompt(self, text_lower: str) -> bool:
        """Return True if `text_lower` mentions `@<name>` or contains any keyword."""
        return bool(self._mention_re.search(text_lower)) or any(
            kw in text_lower for kw in self._keywords_lower
        )

    def render(self) -> str:
        heading = f"# Rule: {self.name}"
        if self.description:
            heading += f"\n*{self.description}*"
        return f"{heading}\n\n{self.body}" if self.body else heading


def _as_str_tuple(raw: Any) -> tuple[str, ...]:
    if not raw:
        return ()
    if isinstance(raw, str):
        return tuple(p.strip() for p in raw.split(",") if p.strip())
    if isinstance(raw, Sequence):
        return tuple(str(p).strip() for p in raw if str(p).strip())
    return (str(raw).strip(),)


def _split_frontmatter(
    text: str, *, strict: bool = False, source: Path | str = ""
) -> tuple[dict[str, Any], str]:
    """Split optional YAML frontmatter (`--- ... ---`) and return `(frontmatter_dict, stripped_body)`."""
    match = _RULE_FRONTMATTER_RE.match(text)
    if not match:
        return {}, text.strip()
    body = text[match.end() :].strip()
    if strict:
        fm_raw = yaml.safe_load(match.group(1))
        if fm_raw is None:
            return {}, body
        if not isinstance(fm_raw, dict):
            raise ValueError(f"YAML frontmatter in {source} must be a mapping")
        return fm_raw, body
    with contextlib.suppress(yaml.YAMLError):
        loaded = yaml.safe_load(match.group(1))
        if isinstance(loaded, dict):
            return loaded, body
    return {}, body


def _parse_rule_file(md_file: Path) -> RuleDefinition:
    """Parse a Markdown rule file with optional YAML frontmatter."""
    text = md_file.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(text, strict=True, source=md_file)

    if not body:
        raise ValueError(f"Rule file {md_file} has an empty body")

    raw_name = str(fm.get("name") or md_file.stem).strip().lower()
    name = re.sub(r"[^a-z0-9-]+", "-", raw_name).strip("-") or md_file.stem.lower()

    raw_globs = (
        fm.get("globs") or fm.get("glob") or fm.get("paths") or fm.get("applyTo")
    )
    globs = _as_str_tuple(raw_globs)
    keywords = _as_str_tuple(fm.get("keywords"))
    raw_desc = fm.get("description")
    description = (
        str(raw_desc).strip()
        if raw_desc is not None and str(raw_desc).strip()
        else None
    )

    raw_trigger = fm.get("trigger")
    if raw_trigger is not None:
        norm_trig = str(raw_trigger).strip().lower().replace("-", "_")
        if norm_trig not in _TRIGGER_ALIASES:
            raise ValueError(
                f"Invalid rule trigger {raw_trigger!r} in {md_file}; "
                "expected 'always_on', 'glob', 'model_decision', or 'manual'"
            )
        trigger: RuleTrigger = _TRIGGER_ALIASES[norm_trig]
    elif fm.get("alwaysApply") is True:
        trigger = "always_on"
    elif globs:
        trigger = "glob"
    elif keywords:
        trigger = "manual"
    elif description and fm.get("alwaysApply") is False:
        trigger = "model_decision"
    else:
        trigger = "always_on"

    if trigger == "model_decision" and not description:
        raise ValueError(
            f"Rule {md_file} with trigger='model_decision' requires a non-empty 'description'"
        )
    if trigger == "glob" and not globs:
        raise ValueError(
            f"Rule {md_file} with trigger='glob' requires 'globs' in frontmatter"
        )

    return RuleDefinition(
        name=name,
        trigger=trigger,
        body=body,
        path=md_file.resolve(),
        description=description,
        globs=globs,
        keywords=keywords,
    )


def _path_matches_globs(
    raw_path: str, globs: Sequence[str], root_dir: str | Path | None = None
) -> bool:
    """Return True if `raw_path` matches any glob pattern in `globs`."""
    cleaned = raw_path.strip().replace("\\", "/")
    if not cleaned:
        return False
    candidates: list[PurePosixPath] = [PurePosixPath(cleaned)]
    if root_dir is not None:
        with contextlib.suppress(ValueError, OSError):
            rel = Path(cleaned).resolve().relative_to(Path(root_dir).resolve())
            candidates.insert(0, PurePosixPath(rel.as_posix()))
    name_only = PurePosixPath(candidates[0].name)
    if name_only not in candidates:
        candidates.append(name_only)

    for pattern in globs:
        pat = pattern.strip().replace("\\", "/")
        if not pat:
            continue
        pats = [pat]
        if pat.startswith("**/"):
            pats.append(pat.removeprefix("**/"))
        for cand in candidates:
            if any(cand.full_match(p) for p in pats):
                return True
    return False


def _extract_paths_from_tool_args(args: Any) -> list[str]:
    """Extract candidate file/directory paths from a tool argument dict."""
    if not isinstance(args, dict):
        return []
    paths: list[str] = []
    for key in ("path", "file_path", "filepath", "directory", "target_file"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            paths.append(val.strip())
    for key in ("paths", "files"):
        val = args.get(key)
        if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
            paths.extend(
                str(x).strip() for x in val if isinstance(x, str) and str(x).strip()
            )
    return paths


def _extract_user_prompt_text(part: UserPromptPart) -> str:
    """Extract text from a UserPromptPart (`str` in CLI mode, `Sequence[UserContent]` in ACP mode)."""
    if isinstance(part.content, str):
        return part.content
    if isinstance(part.content, Sequence):
        return "\n".join(item for item in part.content if isinstance(item, str))
    return str(part.content)


@dataclass
class _DynamicRulesCapability(AbstractCapability[Any]):
    """Runtime capability that activates `glob` and `manual` rules once per run with deduplication."""

    glob_rules: tuple[RuleDefinition, ...] = ()
    manual_rules: tuple[RuleDefinition, ...] = ()
    loader: (
        Callable[[], tuple[tuple[RuleDefinition, ...], tuple[RuleDefinition, ...]]]
        | None
    ) = None
    _activated: set[str] = field(default_factory=set, init=False, repr=False)

    async def for_run(self, ctx: RunContext[Any]) -> _DynamicRulesCapability:
        if self.loader is not None:
            glob_rules, manual_rules = self.loader()
        else:
            glob_rules, manual_rules = self.glob_rules, self.manual_rules
        return _DynamicRulesCapability(
            glob_rules=glob_rules,
            manual_rules=manual_rules,
            loader=self.loader,
        )

    def _match_glob_rules(
        self, path_str: str, root_dir: str | Path | None = None
    ) -> list[RuleDefinition]:
        matched: list[RuleDefinition] = []
        for rule in self.glob_rules:
            if rule.name not in self._activated and _path_matches_globs(
                path_str, rule.globs, root_dir
            ):
                self._activated.add(rule.name)
                matched.append(rule)
        return matched

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        for path_str in _extract_paths_from_tool_args(args):
            for rule in self._match_glob_rules(path_str):
                ctx.enqueue(SystemPromptPart(rule.render()))
        return result

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        newly_triggered: list[RuleDefinition] = []

        last_final_response_idx = -1
        for idx, msg in enumerate(request_context.messages):
            if isinstance(msg, ModelResponse) and not any(
                isinstance(p, ToolCallPart) for p in msg.parts
            ):
                last_final_response_idx = idx

        for idx, msg in enumerate(request_context.messages):
            is_current_turn = idx > last_final_response_idx
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    if not is_current_turn:
                        continue
                    text = _extract_user_prompt_text(part)
                    if not text or text.lstrip().startswith("<system>"):
                        continue
                    text_lower = text.lower()
                    for rule in self.manual_rules:
                        if rule.name not in self._activated and rule.matches_prompt(
                            text_lower
                        ):
                            self._activated.add(rule.name)
                            newly_triggered.append(rule)
                elif isinstance(part, ToolCallPart) and isinstance(part.args, dict):
                    for path_str in _extract_paths_from_tool_args(part.args):
                        newly_triggered.extend(self._match_glob_rules(path_str))
                elif isinstance(part, ToolReturnPart) and isinstance(
                    part.metadata, dict
                ):
                    nested_calls = part.metadata.get("tool_calls")
                    if isinstance(nested_calls, dict):
                        for tc in nested_calls.values():
                            tc_args = getattr(tc, "args", None)
                            if isinstance(tc_args, dict):
                                for path_str in _extract_paths_from_tool_args(tc_args):
                                    newly_triggered.extend(
                                        self._match_glob_rules(path_str)
                                    )

        if (
            newly_triggered
            and request_context.messages
            and isinstance(request_context.messages[-1], ModelRequest)
        ):
            last_msg = request_context.messages[-1]
            last_msg.parts = [
                *last_msg.parts,
                *(SystemPromptPart(rule.render()) for rule in newly_triggered),
            ]

        return request_context


@dataclass
class RulesSpec(_SpecCombinedCapability):
    """Discovers and applies Markdown rules (`<agent>/rules/*.md`) across 4 trigger modes:
    `always_on`, `glob`, `model_decision`, and `manual` (`@<rule-name>`).
    """

    paths: str | list[str] = "./rules"
    pattern: str = "*.md"
    recursive: bool = True
    include: list[str] | None = None
    exclude: list[str] | None = None
    _agent_dir: Path | None = field(default=None, init=False, repr=False)
    _always_on_instructions: str | None = field(default=None, init=False, repr=False)
    _cached_snapshot: tuple[tuple[str, int], ...] = field(
        default=(), init=False, repr=False
    )
    _cached_dynamic: tuple[tuple[RuleDefinition, ...], tuple[RuleDefinition, ...]] = (
        field(default=((), ()), init=False, repr=False)
    )

    def _discover_md_files(self) -> tuple[list[Path], tuple[tuple[str, int], ...]]:
        with _agent_dir_scope(self._agent_dir):
            raw_paths = _as_str_list(self.paths)
            md_files: list[Path] = []
            for raw_p in raw_paths:
                cand = _resolve_agent_relative_path(raw_p)
                if cand.is_file():
                    md_files.append(cand.resolve())
                elif cand.is_dir():
                    md_files.extend(
                        _iter_public_files(cand, self.pattern, self.recursive)
                    )
                elif not _is_default_subdir_ref(raw_p, DEFAULT_RULES_DIR):
                    raise FileNotFoundError(f"Rules path not found: {raw_p}")
            snapshot: list[tuple[str, int]] = []
            _record_mtime(snapshot, *md_files)
            return md_files, tuple(snapshot)

    def _parse_rules(
        self, md_files: Sequence[Path]
    ) -> tuple[
        list[RuleDefinition],
        list[RuleDefinition],
        list[RuleDefinition],
        list[RuleDefinition],
    ]:
        rule_allowed = _make_name_filter(self.include, self.exclude)

        seen_paths: set[Path] = set()
        seen_names: set[str] = set()
        always_on: list[RuleDefinition] = []
        model_decision: list[RuleDefinition] = []
        glob_rules: list[RuleDefinition] = []
        manual_rules: list[RuleDefinition] = []

        for md_file in md_files:
            if md_file in seen_paths:
                continue
            seen_paths.add(md_file)
            try:
                rule = _parse_rule_file(md_file)
            except (UnicodeDecodeError, ValueError, yaml.YAMLError) as err:
                warnings.warn(
                    f"Skipping rule {md_file}: {err}", UserWarning, stacklevel=2
                )
                continue

            if not rule_allowed(rule.name) or rule.name in seen_names:
                continue
            seen_names.add(rule.name)

            if rule.trigger == "always_on":
                always_on.append(rule)
            elif rule.trigger == "model_decision":
                model_decision.append(rule)
            elif rule.trigger == "glob":
                glob_rules.append(rule)
            elif rule.trigger == "manual":
                manual_rules.append(rule)

        return always_on, model_decision, glob_rules, manual_rules

    def _load_dynamic_rules(
        self,
    ) -> tuple[tuple[RuleDefinition, ...], tuple[RuleDefinition, ...]]:
        md_files, snapshot = self._discover_md_files()
        if snapshot != self._cached_snapshot:
            _, _, glob_rules, manual_rules = self._parse_rules(md_files)
            self._cached_snapshot = snapshot
            self._cached_dynamic = (tuple(glob_rules), tuple(manual_rules))
        return self._cached_dynamic

    def __post_init__(self) -> None:
        self._agent_dir = _AGENT_DIR_CTX.get()
        md_files, snapshot = self._discover_md_files()
        always_on, model_decision, glob_rules, manual_rules = self._parse_rules(
            md_files
        )
        self._cached_snapshot = snapshot
        self._cached_dynamic = (tuple(glob_rules), tuple(manual_rules))
        sub_caps: list[AbstractCapability[Any]] = [
            Capability(
                id=f"rule-{rule.name}",
                description=f"[Rule] {rule.description}".replace("\n", "\n  "),
                instructions=rule.render(),
                defer_loading=True,
            )
            for rule in model_decision
        ]
        sub_caps.append(
            _DynamicRulesCapability(
                glob_rules=self._cached_dynamic[0],
                manual_rules=self._cached_dynamic[1],
                loader=self._load_dynamic_rules,
            )
        )

        self._always_on_instructions = (
            "\n\n".join(r.render() for r in always_on) if always_on else None
        )
        self.capabilities = sub_caps
        super().__post_init__()

    def get_instructions(self) -> str | None:
        return self._always_on_instructions

    @classmethod
    def get_serialization_name(cls) -> str:
        return "Rules"


_MD_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)#?\s]+)(?:#[^)\s]*)?\)")
_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
KnowledgeSearchMode = Literal["auto", "fts", "regex", "literal"]


def _normalize_posix_relpath(base_dir: PurePosixPath, target: str) -> str | None:
    """Resolve a relative Markdown link target against `base_dir` into a normalized POSIX path."""
    if (
        not target
        or "://" in target
        or target.startswith(("mailto:", "tel:", "data:", "#"))
    ):
        return None
    joined = (
        PurePosixPath(target.lstrip("/"))
        if target.startswith("/")
        else (base_dir / target)
    )
    parts: list[str] = []
    for part in joined.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
        else:
            parts.append(part)
    if not parts:
        return None
    norm = "/".join(parts)
    return norm if norm.endswith(".md") else f"{norm}.md"


@dataclass(frozen=True)
class KnowledgeEntry:
    """Parsed Open Knowledge Format (OKF) Markdown document."""

    bundle: str
    path: str
    doc_type: str
    title: str
    description: str
    resource: str
    tags: tuple[str, ...]
    timestamp: str
    body: str
    raw_text: str
    outgoing_links: tuple[str, ...]


def _parse_okf_markdown(bundle: str, rel_path: str, raw_text: str) -> KnowledgeEntry:
    """Parse an Open Knowledge Format (OKF) Markdown file with optional YAML frontmatter."""
    fm, body = _split_frontmatter(raw_text)

    posix_path = PurePosixPath(rel_path)
    lower_name = posix_path.name.lower()
    default_type = (
        "index"
        if lower_name == "index.md"
        else ("log" if lower_name == "log.md" else "concept")
    )
    raw_type = fm.get("type")
    doc_type = (
        str(raw_type).strip()
        if raw_type is not None and str(raw_type).strip()
        else default_type
    )

    raw_title = fm.get("title")
    if raw_title is not None and str(raw_title).strip():
        title = str(raw_title).strip()
    elif h_match := _HEADING_RE.search(body):
        title = h_match.group(1).strip()
    else:
        title = posix_path.stem.replace("-", " ").replace("_", " ")

    raw_desc = fm.get("description")
    if raw_desc is not None and str(raw_desc).strip():
        description = str(raw_desc).strip()
    else:
        first_para = next(
            (
                line.strip()
                for line in body.splitlines()
                if line.strip()
                and not line.lstrip().startswith(("#", "-", "*", ">", "`", "|"))
            ),
            "",
        )
        description = first_para[:200]

    raw_res = fm.get("resource")
    resource = (
        str(raw_res).strip() if raw_res is not None and str(raw_res).strip() else ""
    )
    tags = _as_str_tuple(fm.get("tags"))
    raw_ts = fm.get("timestamp")
    timestamp = (
        str(raw_ts).strip() if raw_ts is not None and str(raw_ts).strip() else ""
    )

    parent_dir = posix_path.parent
    links: list[str] = []
    seen_links: set[str] = set()
    for m in _MD_LINK_RE.finditer(body):
        resolved_link = _normalize_posix_relpath(parent_dir, m.group(1))
        if (
            resolved_link
            and resolved_link != rel_path
            and resolved_link not in seen_links
        ):
            seen_links.add(resolved_link)
            links.append(resolved_link)

    return KnowledgeEntry(
        bundle=bundle,
        path=rel_path,
        doc_type=doc_type,
        title=title,
        description=description,
        resource=resource,
        tags=tags,
        timestamp=timestamp,
        body=body,
        raw_text=raw_text,
        outgoing_links=tuple(links),
    )


def _load_zip_okf_entries(
    zip_path: Path, bundle_name: str | None = None
) -> list[KnowledgeEntry]:
    """Read all `.md` files from an Open Knowledge Format `.zip` archive in memory."""
    bundle = bundle_name or zip_path.stem
    raw_members: list[tuple[PurePosixPath, str]] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            p = PurePosixPath(info.filename)
            if p.suffix.lower() != ".md":
                continue
            if any(
                part.startswith((".", "_")) or part == "__MACOSX" for part in p.parts
            ):
                continue
            try:
                text = zf.read(info.filename).decode("utf-8")
            except UnicodeDecodeError:
                continue
            raw_members.append((p, text))

    if not raw_members:
        return []

    first_parts = {p.parts[0] for p, _ in raw_members if len(p.parts) > 1}
    only_root = next(iter(first_parts)) if len(first_parts) == 1 else None
    strip_root = (
        only_root is not None
        and all(len(p.parts) > 1 for p, _ in raw_members)
        and (
            only_root.lower() == bundle.lower()
            or any(
                p.as_posix().lower() == f"{only_root.lower()}/index.md"
                for p, _ in raw_members
            )
        )
    )

    entries: list[KnowledgeEntry] = []
    for p, text in sorted(raw_members, key=lambda item: item[0].as_posix()):
        rel = PurePosixPath(*p.parts[1:]).as_posix() if strip_root else p.as_posix()
        entries.append(_parse_okf_markdown(bundle, rel, text))
    return entries


def _load_dir_okf_entries(
    dir_path: Path, bundle_name: str | None = None
) -> list[KnowledgeEntry]:
    """Read all `.md` files from an unpacked Open Knowledge Format bundle directory."""
    bundle = bundle_name or dir_path.name
    entries: list[KnowledgeEntry] = []
    for md_file in _iter_public_files(dir_path, "*.md", recursive=True):
        rel = md_file.relative_to(dir_path.resolve()).as_posix()
        try:
            text = md_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        entries.append(_parse_okf_markdown(bundle, rel, text))
    return entries


class KnowledgeIndex:
    """In-memory SQLite FTS5 BM25 + regex/literal search index and link graph for OKF bundles."""

    def __init__(self, entries: Sequence[KnowledgeEntry]) -> None:
        self.entries: tuple[KnowledgeEntry, ...] = tuple(entries)
        self.by_key: dict[tuple[str, str], KnowledgeEntry] = {}
        self.by_bundle: dict[str, list[KnowledgeEntry]] = {}
        backlinks_map: dict[tuple[str, str], list[str]] = {}

        for entry in self.entries:
            key = (entry.bundle, entry.path)
            self.by_key[key] = entry
            self.by_bundle.setdefault(entry.bundle, []).append(entry)

        for entry in self.entries:
            for target_path in entry.outgoing_links:
                target_key = (entry.bundle, target_path)
                if target_key in self.by_key:
                    backlinks_map.setdefault(target_key, []).append(entry.path)

        self.backlinks: dict[tuple[str, str], tuple[str, ...]] = {
            k: tuple(sorted(set(v))) for k, v in backlinks_map.items()
        }

        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.execute(
            "CREATE VIRTUAL TABLE okf_fts USING fts5("
            "bundle, path, doc_type, title, description, tags, resource, body, "
            "tokenize='porter unicode61'"
            ")"
        )
        if self.entries:
            self._conn.executemany(
                "INSERT INTO okf_fts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        e.bundle,
                        e.path,
                        e.doc_type,
                        e.title,
                        e.description,
                        " ".join(e.tags),
                        e.resource,
                        e.body,
                    )
                    for e in self.entries
                ],
            )

    def _entry_matches_filters(
        self,
        entry: KnowledgeEntry,
        *,
        bundle: str | None = None,
        doc_type: str | None = None,
        tag: str | None = None,
        path_prefix: str | None = None,
    ) -> bool:
        if bundle and entry.bundle.lower() != bundle.strip().lower():
            return False
        if doc_type and entry.doc_type.lower() != doc_type.strip().lower():
            return False
        if tag:
            tag_lower = tag.strip().lower()
            if not any(t.lower() == tag_lower for t in entry.tags):
                return False
        if path_prefix:
            pref = path_prefix.strip().lstrip("/").lower()
            if not entry.path.lower().startswith(pref):
                return False
        return True

    def _format_hit(
        self,
        entry: KnowledgeEntry,
        *,
        snippet: str,
        score: float | None = None,
        match_lines: list[int] | None = None,
    ) -> dict[str, Any]:
        hit: dict[str, Any] = {
            "bundle": entry.bundle,
            "path": entry.path,
            "type": entry.doc_type,
            "title": entry.title,
            "description": entry.description,
            "tags": list(entry.tags),
            "snippet": snippet,
            "outgoing_links": list(entry.outgoing_links),
            "backlinks": list(self.backlinks.get((entry.bundle, entry.path), ())),
        }
        if entry.resource:
            hit["resource"] = entry.resource
        if score is not None:
            hit["score"] = round(score, 4)
        if match_lines is not None:
            hit["match_lines"] = match_lines
        return hit

    def _run_fts_query(
        self,
        fts_query: str,
        *,
        bundle: str | None = None,
        doc_type: str | None = None,
        tag: str | None = None,
        path_prefix: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT bundle, path, "
            "snippet(okf_fts, 7, '«', '»', '…', 24) AS snip, "
            "-bm25(okf_fts, 1.0, 4.0, 4.0, 10.0, 6.0, 8.0, 4.0, 1.0) AS score "
            "FROM okf_fts WHERE okf_fts MATCH ? ORDER BY score DESC LIMIT ?",
            (fts_query, max(limit * 5, 50)),
        )
        q_tokens = [
            t.lower()
            for t in _FTS_TOKEN_RE.findall(fts_query)
            if t.upper() not in ("AND", "OR", "NOT", "NEAR")
        ]
        scored_hits: list[tuple[float, int, dict[str, Any]]] = []
        for idx, (b_name, p_str, snip, raw_score) in enumerate(cur.fetchall()):
            entry = self.by_key.get((b_name, p_str))
            if entry is None or not self._entry_matches_filters(
                entry,
                bundle=bundle,
                doc_type=doc_type,
                tag=tag,
                path_prefix=path_prefix,
            ):
                continue
            title_l = entry.title.lower()
            tags_l = " ".join(entry.tags).lower()
            desc_l = entry.description.lower()
            path_l = entry.path.lower()
            body_l = entry.body.lower()
            boost = 0.0
            for tok in q_tokens:
                if tok in title_l:
                    boost += 10.0
                if tok in tags_l:
                    boost += 8.0
                if tok in desc_l:
                    boost += 5.0
                if tok in path_l:
                    boost += 3.0
                if tok in body_l:
                    boost += 1.0
            final_score = max(0.0, float(raw_score)) + boost
            hit = self._format_hit(
                entry, snippet=snip or entry.description, score=final_score
            )
            scored_hits.append((final_score, -idx, hit))
        scored_hits.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored_hits[:limit]]

    def _run_line_search(
        self,
        query: str,
        *,
        is_regex: bool,
        bundle: str | None = None,
        doc_type: str | None = None,
        tag: str | None = None,
        path_prefix: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        pattern_str = query if is_regex else re.escape(query)
        compiled = re.compile(pattern_str, re.IGNORECASE)
        hits: list[dict[str, Any]] = []

        for entry in self.entries:
            if not self._entry_matches_filters(
                entry,
                bundle=bundle,
                doc_type=doc_type,
                tag=tag,
                path_prefix=path_prefix,
            ):
                continue
            lines = entry.raw_text.splitlines()
            matched_line_nums: list[int] = []
            snippets: list[str] = []
            for idx, line in enumerate(lines):
                if compiled.search(line):
                    matched_line_nums.append(idx + 1)
                    if len(snippets) < 3:
                        start = max(0, idx - 1)
                        end = min(len(lines), idx + 2)
                        ctx_block = "\n".join(
                            f"L{ln + 1}: {lines[ln]}" for ln in range(start, end)
                        )
                        snippets.append(ctx_block)
            if matched_line_nums:
                hits.append(
                    self._format_hit(
                        entry,
                        snippet="\n---\n".join(snippets),
                        score=float(len(matched_line_nums)),
                        match_lines=matched_line_nums[:20],
                    )
                )
                if len(hits) >= limit:
                    break
        return hits

    def search(
        self,
        query: str,
        *,
        bundle: str | None = None,
        doc_type: str | None = None,
        tag: str | None = None,
        path_prefix: str | None = None,
        mode: KnowledgeSearchMode = "auto",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search indexed OKF entries using SQLite FTS5 BM25 ranking and/or regex/literal line matching."""
        cleaned = query.strip()
        eff_limit = max(1, min(limit, 50))
        if not cleaned:
            return [
                self._format_hit(e, snippet=e.description)
                for e in self.entries
                if self._entry_matches_filters(
                    e,
                    bundle=bundle,
                    doc_type=doc_type,
                    tag=tag,
                    path_prefix=path_prefix,
                )
            ][:eff_limit]

        if mode == "regex":
            return self._run_line_search(
                cleaned,
                is_regex=True,
                bundle=bundle,
                doc_type=doc_type,
                tag=tag,
                path_prefix=path_prefix,
                limit=eff_limit,
            )
        if mode == "literal":
            return self._run_line_search(
                cleaned,
                is_regex=False,
                bundle=bundle,
                doc_type=doc_type,
                tag=tag,
                path_prefix=path_prefix,
                limit=eff_limit,
            )

        if mode == "fts":
            try:
                return self._run_fts_query(
                    cleaned,
                    bundle=bundle,
                    doc_type=doc_type,
                    tag=tag,
                    path_prefix=path_prefix,
                    limit=eff_limit,
                )
            except sqlite3.OperationalError:
                pass

        tokens = _FTS_TOKEN_RE.findall(cleaned)
        if tokens:
            for joiner in (" AND ", " OR "):
                fts_q = joiner.join(f'"{t}"*' for t in tokens)
                with contextlib.suppress(sqlite3.OperationalError):
                    hits = self._run_fts_query(
                        fts_q,
                        bundle=bundle,
                        doc_type=doc_type,
                        tag=tag,
                        path_prefix=path_prefix,
                        limit=eff_limit,
                    )
                    if hits:
                        return hits

        return self._run_line_search(
            cleaned,
            is_regex=False,
            bundle=bundle,
            doc_type=doc_type,
            tag=tag,
            path_prefix=path_prefix,
            limit=eff_limit,
        )

    def read(
        self,
        path: str,
        *,
        bundle: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Read a specific OKF Markdown entry along with its frontmatter, outgoing links, and backlinks."""
        raw = path.strip().lstrip("/")
        eff_bundle = bundle.strip() if bundle else None
        if ":" in raw and eff_bundle is None:
            b_part, p_part = raw.split(":", 1)
            if b_part in self.by_bundle:
                eff_bundle = b_part
                raw = p_part.lstrip("/")
        elif "/" in raw and eff_bundle is None:
            first_seg, rest = raw.split("/", 1)
            if first_seg in self.by_bundle:
                eff_bundle = first_seg
                raw = rest

        candidates = (
            [raw] if raw.endswith(".md") else [f"{raw}.md", raw, f"{raw}/index.md"]
        )
        matched: KnowledgeEntry | None = None

        for cand in candidates:
            if eff_bundle is not None:
                if (entry := self.by_key.get((eff_bundle, cand))) is not None:
                    matched = entry
                    break
            else:
                for entry in self.entries:
                    if entry.path == cand or entry.path.lower() == cand.lower():
                        matched = entry
                        break
                if matched is not None:
                    break

        if matched is None:
            available = [f"{e.bundle}:{e.path}" for e in self.entries[:25]]
            raise ValueError(
                f"Knowledge entry {path!r} not found"
                + (f" in bundle {eff_bundle!r}" if eff_bundle else "")
                + f". Available entries: {', '.join(available)}"
            )

        lines = matched.body.splitlines()
        total_lines = len(lines)
        if start_line is not None or end_line is not None:
            s_idx = max(0, (start_line or 1) - 1)
            e_idx = min(total_lines, end_line or total_lines)
            content = "\n".join(lines[s_idx:e_idx])
        else:
            content = matched.body

        return {
            "bundle": matched.bundle,
            "path": matched.path,
            "type": matched.doc_type,
            "title": matched.title,
            "description": matched.description,
            "resource": matched.resource,
            "tags": list(matched.tags),
            "timestamp": matched.timestamp,
            "total_lines": total_lines,
            "outgoing_links": list(matched.outgoing_links),
            "backlinks": list(self.backlinks.get((matched.bundle, matched.path), ())),
            "content": content,
        }

    def list_entries(
        self,
        bundle: str | None = None,
        *,
        doc_type: str | None = None,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """List available OKF knowledge bundles and matching concept entries."""
        bundles_summary: list[dict[str, Any]] = []
        for b_name, b_entries in sorted(self.by_bundle.items()):
            if bundle and b_name.lower() != bundle.strip().lower():
                continue
            types = sorted({e.doc_type for e in b_entries})
            tags = sorted({t for e in b_entries for t in e.tags})
            index_entry = self.by_key.get((b_name, "index.md"))
            bundles_summary.append(
                {
                    "bundle": b_name,
                    "entry_count": len(b_entries),
                    "types": types,
                    "tags": tags,
                    "has_index": index_entry is not None,
                    "index_description": index_entry.description
                    if index_entry
                    else None,
                }
            )

        filtered_entries = [
            {
                "bundle": e.bundle,
                "path": e.path,
                "type": e.doc_type,
                "title": e.title,
                "description": e.description,
                "tags": list(e.tags),
                **({"resource": e.resource} if e.resource else {}),
            }
            for e in self.entries
            if self._entry_matches_filters(e, bundle=bundle, doc_type=doc_type, tag=tag)
        ]
        return {
            "total_bundles": len(bundles_summary),
            "total_entries": len(filtered_entries),
            "bundles": bundles_summary,
            "entries": filtered_entries,
        }

    def render_catalog(self, include_index_overview: bool = True) -> str | None:
        """Render a progressive-disclosure catalog of discovered OKF bundles for the system prompt."""
        if not self.entries:
            return None
        lines = [
            "## Knowledge Bundles (Open Knowledge Format)",
            (
                "Use `search_knowledge(query=...)` (BM25 + regex search), `read_knowledge(path=...)`, and "
                "`list_knowledge()` to query these knowledge bundles without shell commands:"
            ),
        ]
        for b_name, b_entries in sorted(self.by_bundle.items()):
            types_str = ", ".join(sorted({e.doc_type for e in b_entries}))
            tags_set = sorted({t for e in b_entries for t in e.tags})
            tags_str = f" | tags: {', '.join(tags_set[:10])}" if tags_set else ""
            index_entry = self.by_key.get((b_name, "index.md"))
            desc_str = (
                f" — {index_entry.description}"
                if (index_entry and index_entry.description)
                else ""
            )
            lines.append(
                f"- **`{b_name}`** ({len(b_entries)} entries | types: `{types_str}`{tags_str}){desc_str}"
            )
            if include_index_overview and index_entry is not None:
                preview_lines = [
                    ln.rstrip() for ln in index_entry.body.splitlines() if ln.strip()
                ][:15]
                if preview_lines:
                    indented = "\n".join(f"  > {ln}" for ln in preview_lines)
                    lines.append(f"  Entry point (`{b_name}:index.md`):\n{indented}")
        return "\n".join(lines)


@dataclass
class KnowledgeSpec(_SpecCombinedCapability):
    """Discovers and indexes Open Knowledge Format (`.zip` archives and directories) in `<agent>/knowledge/`."""

    paths: str | list[str] = "./knowledge"
    include_bundles: list[str] | None = None
    exclude_bundles: list[str] | None = None
    inject_catalog: bool = True
    include_index_overview: bool = True
    code_mode: bool = False
    _agent_dir: Path | None = field(default=None, init=False, repr=False)
    _cached_snapshot: tuple[tuple[str, int], ...] = field(
        default=(), init=False, repr=False
    )
    _cached_index: KnowledgeIndex | None = field(default=None, init=False, repr=False)

    def _discover_sources(
        self,
    ) -> tuple[list[tuple[str, Path, bool]], tuple[tuple[str, int], ...]]:
        """Return `(sources, snapshot)` where each source is `(bundle_name, path, is_zip)`."""
        with _agent_dir_scope(self._agent_dir):
            raw_paths = _as_str_list(self.paths)
            sources: list[tuple[str, Path, bool]] = []
            snapshot: list[tuple[str, int]] = []
            bundle_allowed = _make_name_filter(
                self.include_bundles, self.exclude_bundles
            )

            for raw_p in raw_paths:
                cand = _resolve_agent_relative_path(raw_p)
                if cand.is_file() and cand.suffix.lower() == ".zip":
                    resolved = cand.resolve()
                    if bundle_allowed(resolved.stem):
                        sources.append((resolved.stem, resolved, True))
                        _record_mtime(snapshot, resolved)
                elif cand.is_dir():
                    resolved_dir = cand.resolve()
                    for zip_file in _iter_public_files(
                        resolved_dir, "*.zip", recursive=False
                    ):
                        if bundle_allowed(zip_file.stem):
                            sources.append((zip_file.stem, zip_file, True))
                            _record_mtime(snapshot, zip_file)
                    for child in sorted(resolved_dir.iterdir()):
                        if (
                            child.is_dir()
                            and not child.name.startswith((".", "_"))
                            and bundle_allowed(child.name)
                        ):
                            sources.append((child.name, child.resolve(), False))
                            _record_mtime(
                                snapshot,
                                *_iter_public_files(child, "*.md", recursive=True),
                            )
                    top_mds = _iter_public_files(resolved_dir, "*.md", recursive=False)
                    if top_mds and bundle_allowed(resolved_dir.name):
                        sources.append((resolved_dir.name, resolved_dir, False))
                        _record_mtime(snapshot, *top_mds)
                elif not _is_default_subdir_ref(raw_p, DEFAULT_KNOWLEDGE_DIR):
                    raise FileNotFoundError(f"Knowledge path not found: {raw_p}")
            return sources, tuple(snapshot)

    def _get_index(self) -> KnowledgeIndex:
        sources, snapshot = self._discover_sources()
        if self._cached_index is None or snapshot != self._cached_snapshot:
            all_entries: list[KnowledgeEntry] = []
            seen_keys: set[tuple[str, str]] = set()
            for bundle_name, src_path, is_zip in sources:
                try:
                    loaded = (
                        _load_zip_okf_entries(src_path, bundle_name)
                        if is_zip
                        else _load_dir_okf_entries(src_path, bundle_name)
                    )
                except (OSError, zipfile.BadZipFile) as err:
                    warnings.warn(
                        f"Skipping knowledge bundle {src_path}: {err}",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                for entry in loaded:
                    key = (entry.bundle, entry.path)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_entries.append(entry)
            self._cached_snapshot = snapshot
            self._cached_index = KnowledgeIndex(all_entries)
        return self._cached_index

    def __post_init__(self) -> None:
        self._agent_dir = _AGENT_DIR_CTX.get()
        self._get_index()

        toolset = FunctionToolset(id="knowledge")
        meta = {"microharness_local_tool": True, "code_mode": self.code_mode}

        def search_knowledge(
            query: str,
            bundle: str | None = None,
            doc_type: str | None = None,
            tag: str | None = None,
            path_prefix: str | None = None,
            mode: KnowledgeSearchMode = "auto",
            limit: int = 10,
        ) -> list[dict[str, Any]]:
            """Search Open Knowledge Format (OKF) bundles using fast in-memory BM25 FTS5 and regex matching.

            Args:
                query: Search keywords, FTS5 query (`term1 AND term2`, `"exact phrase"`, `prefix*`), or regex pattern.
                bundle: Optional bundle name filter.
                doc_type: Optional OKF frontmatter `type` filter (e.g. 'concept', 'howto', 'index', 'table', 'metric').
                tag: Optional OKF frontmatter `tags` filter.
                path_prefix: Optional relative path prefix filter (e.g. 'concepts/').
                mode: Search mode: 'auto' (BM25 FTS5 with literal fallback), 'fts', 'regex', or 'literal'.
                limit: Maximum number of results to return (1-50, default 10).
            """
            return self._get_index().search(
                query,
                bundle=bundle,
                doc_type=doc_type,
                tag=tag,
                path_prefix=path_prefix,
                mode=mode,
                limit=limit,
            )

        def read_knowledge(
            path: str,
            bundle: str | None = None,
            start_line: int | None = None,
            end_line: int | None = None,
        ) -> dict[str, Any]:
            """Read an Open Knowledge Format (OKF) Markdown file, its frontmatter, outgoing links, and backlinks.

            Args:
                path: Entry path within the bundle (e.g. 'index.md', 'concepts/overview.md', or 'bundle:index.md').
                bundle: Optional bundle name when multiple bundles are loaded.
                start_line: Optional 1-based start line number.
                end_line: Optional 1-based end line number (inclusive).
            """
            return self._get_index().read(
                path, bundle=bundle, start_line=start_line, end_line=end_line
            )

        def list_knowledge(
            bundle: str | None = None,
            doc_type: str | None = None,
            tag: str | None = None,
        ) -> dict[str, Any]:
            """List available Open Knowledge Format (OKF) bundles and their entries, types, and tags.

            Args:
                bundle: Optional bundle name filter.
                doc_type: Optional OKF frontmatter `type` filter.
                tag: Optional OKF frontmatter `tags` filter.
            """
            return self._get_index().list_entries(bundle, doc_type=doc_type, tag=tag)

        toolset.add_function(search_knowledge, metadata=meta)
        toolset.add_function(read_knowledge, metadata=meta)
        toolset.add_function(list_knowledge, metadata=meta)

        self.capabilities = [Toolset(toolset)]
        super().__post_init__()

    def get_instructions(self) -> Any:
        if not self.inject_catalog:
            return None

        def _render_catalog_instructions(ctx: RunContext[Any]) -> str | None:
            return self._get_index().render_catalog(
                include_index_overview=self.include_index_overview
            )

        return _render_catalog_instructions

    @classmethod
    def get_serialization_name(cls) -> str:
        return "Knowledge"


WebFetchFormat = Literal["markdown", "raw", "text", "json"]
_EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")
_SHELL_FETCH_RE = re.compile(
    r"(?:^|[;&|()\s`])(?:curl|wget|aria2c|httpie|xh|lynx|w3m)\b"
    r"|(?:\burllib\.request\b|\brequests\.(?:get|post|put|delete|head|request|Session)\b"
    r"|\bhttpx2?\.(?:get|post|put|delete|head|request|AsyncClient|Client)\b"
    r"|\baiohttp\.ClientSession\b)"
)
_SENSITIVE_REDIRECT_HEADERS: frozenset[str] = frozenset(
    {"authorization", "cookie", "proxy-authorization"}
)


def _parse_pattern_file(file_path: Path) -> list[str]:
    """Read non-empty, non-comment domain/URL patterns from an allowlist or blocklist file."""
    if not file_path.is_file():
        return []
    patterns: list[str] = []
    with contextlib.suppress(OSError, UnicodeDecodeError):
        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            cleaned = raw_line.split("#", 1)[0].strip()
            if cleaned:
                patterns.append(cleaned)
    return patterns


def _domain_or_url_matches(
    url: str,
    host_key: str,
    pattern: str,
    *,
    match_subdomains: bool = True,
) -> bool:
    """Return True if `url` or normalized `host_key` matches a domain/glob `pattern`."""
    pat = pattern.strip().lower()
    if not pat:
        return False
    if "://" in pat or "/" in pat:
        return fnmatch.fnmatchcase(url.lower(), pat)
    if pat.startswith("*."):
        base_key = _domain_key(pat.removeprefix("*."))
        return bool(base_key) and (
            host_key == base_key or host_key.endswith(f".{base_key}")
        )
    if "*" in pat or "?" in pat:
        return fnmatch.fnmatchcase(host_key, pat)
    base_key = _domain_key(pat.removeprefix("."))
    if not base_key:
        return False
    return host_key == base_key or (
        match_subdomains and host_key.endswith(f".{base_key}")
    )


def _check_url_policy(
    url: str,
    *,
    allowed_domains: Sequence[str],
    blocked_domains: Sequence[str],
    match_subdomains: bool = True,
) -> str:
    """Validate `url` against `blocked_domains` and `allowed_domains`, raising `ModelRetry` on violation."""
    try:
        hostname, _, _, _ = extract_host_and_port(url)
    except ValueError as err:
        raise ModelRetry(f"Invalid URL '{url}': {err}") from err

    host_key = _domain_key(hostname)
    for blocked in blocked_domains:
        if _domain_or_url_matches(
            url, host_key, blocked, match_subdomains=match_subdomains
        ):
            raise ModelRetry(
                f"Access denied: URL '{url}' (host '{hostname}') matches blocked pattern '{blocked}'."
            )

    if allowed_domains and not any(
        _domain_or_url_matches(
            url, host_key, allowed, match_subdomains=match_subdomains
        )
        for allowed in allowed_domains
    ):
        raise ModelRetry(
            f"Access denied: URL '{url}' (host '{hostname}') is not in the WebFetch allowlist ({list(allowed_domains)})."
        )
    return hostname


def _html_to_markdown_and_title(html_text: str) -> tuple[str, str]:
    """Extract `<title>` and convert HTML to clean Markdown using BeautifulSoup and markdownify."""
    soup = BeautifulSoup(html_text, "html.parser")
    title = soup.title.string.strip() if (soup.title and soup.title.string) else ""
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    md = _html_to_md(str(soup), heading_style="ATX").strip()
    return _EXCESSIVE_NEWLINES_RE.sub("\n\n", md), title


def _html_to_text_and_title(html_text: str) -> tuple[str, str]:
    """Extract `<title>` and visible text from HTML."""
    soup = BeautifulSoup(html_text, "html.parser")
    title = soup.title.string.strip() if (soup.title and soup.title.string) else ""
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return _EXCESSIVE_NEWLINES_RE.sub("\n\n", text), title


@dataclass
class _ShellFetchGuardCapability(AbstractCapability[Any]):
    """Blocks `curl`, `wget`, and inline HTTP calls in the `shell` tool so the agent uses `web_fetch` (`httpx2`)."""

    enabled: bool = True

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        if self.enabled and call.tool_name in (
            "shell",
            "run_command",
            "start_command",
        ):
            cmd = str(args.get("command", ""))
            if _SHELL_FETCH_RE.search(cmd):
                raise ModelRetry(
                    "Direct network requests via shell (`curl`, `wget`, `httpie`, or inline Python HTTP clients) "
                    "are disabled by policy. Always use the native `web_fetch` tool (powered by httpx2, supporting "
                    "`format='markdown' | 'raw' | 'text' | 'json'` and `save_to='<path>'`) instead."
                )
        return args


@dataclass
class WebFetchSpec(_SpecCombinedCapability):
    """Configurable URL fetching capability supporting Microharness's custom `httpx2` client (`<agent>/web/`) or Pydantic AI's built-in `WebFetch`."""

    path: str = "./web"
    enabled: bool = True
    backend: Literal["custom", "pydantic_ai"] = "custom"
    native: bool = False
    local: bool = True
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    match_subdomains: bool = True
    allow_local_urls: bool = False
    block_shell_fetch: bool = True
    timeout: int = 30
    max_redirects: int = 10
    max_content_length: int | None = 50_000
    max_download_bytes: int = 50 * 1024 * 1024
    code_mode: bool = False
    _agent_dir: Path | None = field(default=None, init=False, repr=False)
    _cached_snapshot: tuple[tuple[str, int], ...] = field(
        default=(), init=False, repr=False
    )
    _cached_state: tuple[tuple[str, ...], tuple[str, ...], Any] | None = field(
        default=None, init=False, repr=False
    )

    def _load_web_state(self) -> tuple[tuple[str, ...], tuple[str, ...], Any]:
        """Return `(allowed_patterns, blocked_patterns, client_module_or_none)`, hot-reloaded on file edits."""
        with _agent_dir_scope(self._agent_dir):
            web_dir = _resolve_agent_relative_path(self.path)
            if web_dir.is_file():
                web_dir = web_dir.parent
            allow_file = web_dir / "allowlist.txt"
            block_file = web_dir / "blocklist.txt"
            client_file = web_dir / "client.py"

            snapshot: list[tuple[str, int]] = []
            _record_mtime(snapshot, allow_file, block_file, client_file)
            snap_tuple = tuple(snapshot)

            if self._cached_state is None or snap_tuple != self._cached_snapshot:
                file_allowed = _parse_pattern_file(allow_file)
                file_blocked = _parse_pattern_file(block_file)
                merged_allowed = tuple(
                    dict.fromkeys([*(self.allowed_domains or ()), *file_allowed])
                )
                merged_blocked = tuple(
                    dict.fromkeys([*(self.blocked_domains or ()), *file_blocked])
                )
                client_mod = (
                    _load_python_module_from_file(client_file)
                    if client_file.is_file()
                    else None
                )
                self._cached_snapshot = snap_tuple
                self._cached_state = (merged_allowed, merged_blocked, client_mod)
            return self._cached_state

    async def _fetch_with_httpx2(
        self,
        ctx: RunContext[Any],
        url: str,
        *,
        fmt: WebFetchFormat = "markdown",
        headers: dict[str, str] | None = None,
        save_to: str | None = None,
        max_content_length: int | None = None,
    ) -> dict[str, Any]:
        allowed_pats, blocked_pats, client_mod = self._load_web_state()
        _check_url_policy(
            url,
            allowed_domains=allowed_pats,
            blocked_domains=blocked_pats,
            match_subdomains=self.match_subdomains,
        )

        accept_default = (
            "application/json, text/plain;q=0.9, */*;q=0.8"
            if fmt == "json"
            else "text/markdown, text/html, application/xhtml+xml, application/json, text/plain;q=0.9, */*;q=0.8"
        )
        req_headers: dict[str, str] = {
            "Accept": accept_default,
            "Accept-Encoding": "identity, gzip",
        }
        if headers:
            req_headers.update({str(k): str(v) for k, v in headers.items()})
        if (
            client_mod is not None
            and hasattr(client_mod, "prepare_headers")
            and callable(client_mod.prepare_headers)
        ):
            prepared = client_mod.prepare_headers(url, req_headers)
            if isinstance(prepared, dict):
                req_headers = {str(k): str(v) for k, v in prepared.items()}
        req_headers["Accept-Encoding"] = "identity, gzip"

        client_factory = (
            client_mod.create_client
            if (
                client_mod is not None
                and hasattr(client_mod, "create_client")
                and callable(client_mod.create_client)
            )
            else create_async_httpx2_client
        )

        current_url = url
        active_headers = dict(req_headers)
        raw_bytes = b""
        status_code = 200
        content_type = "text/plain"
        charset = "utf-8"

        try:
            async with client_factory(timeout=float(self.timeout)) as client:
                for _hop in range(self.max_redirects + 1):
                    _check_url_policy(
                        current_url,
                        allowed_domains=allowed_pats,
                        blocked_domains=blocked_pats,
                        match_subdomains=self.match_subdomains,
                    )
                    try:
                        resolved = await validate_and_resolve_url(
                            current_url, allow_local=self.allow_local_urls
                        )
                    except ValueError as err:
                        raise ModelRetry(f"Blocked URL '{current_url}': {err}") from err

                    ip_url = build_url_with_ip(resolved)
                    default_port = 443 if resolved.is_https else 80
                    host_header = (
                        resolved.hostname
                        if resolved.port == default_port
                        else f"{resolved.hostname}:{resolved.port}"
                    )
                    hop_headers = {**active_headers, "Host": host_header}
                    extensions = (
                        {"sni_hostname": resolved.hostname} if resolved.is_https else {}
                    )
                    request = client.build_request(
                        "GET", ip_url, headers=hop_headers, extensions=extensions
                    )
                    response = await client.send(
                        request, follow_redirects=False, stream=True
                    )
                    try:
                        if response.is_redirect:
                            await response.aread()
                            location = response.headers.get("location")
                            if not location:
                                raise ModelRetry(
                                    f"Redirect response ({response.status_code}) from '{current_url}' missing Location header."
                                )
                            next_url = resolve_redirect_url(current_url, location)
                            if not _keeps_credentials(current_url, next_url):
                                active_headers = {
                                    k: v
                                    for k, v in active_headers.items()
                                    if k.lower() not in _SENSITIVE_REDIRECT_HEADERS
                                }
                            current_url = next_url
                            continue

                        response.raise_for_status()
                        raw_bytes = await _read_capped_body(
                            response, self.max_download_bytes
                        )
                        status_code = response.status_code
                        content_type = response.headers.get(
                            "content-type", "text/plain"
                        )
                        charset = response.charset_encoding or "utf-8"
                        break
                    finally:
                        await response.aclose()
                else:
                    raise ModelRetry(
                        f"Exceeded maximum redirects ({self.max_redirects}) while fetching '{url}'."
                    )
        except httpx2.HTTPStatusError as err:
            raise ModelRetry(
                f"HTTP {err.response.status_code} error fetching '{current_url}'."
            ) from err
        except httpx2.RequestError as err:
            raise ModelRetry(
                f"Network error fetching '{current_url}' with httpx2: {err}"
            ) from err
        except ValueError as err:
            raise ModelRetry(
                f"Error reading response from '{current_url}': {err}"
            ) from err

        decoded_text = raw_bytes.decode(charset, errors="replace")
        media_type = content_type.split(";", 1)[0].strip().lower()
        is_html = media_type in ("text/html", "application/xhtml+xml") or (
            media_type == "text/plain"
            and decoded_text.lstrip()[:32]
            .lower()
            .startswith(("<!doctype html", "<html"))
        )

        title = ""
        if fmt == "raw":
            rendered = decoded_text
        elif fmt == "json":
            try:
                parsed_json = json.loads(decoded_text)
                rendered = json.dumps(parsed_json, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                rendered = decoded_text
        elif fmt == "text":
            if is_html:
                rendered, title = _html_to_text_and_title(decoded_text)
            else:
                rendered = decoded_text
        else:
            if is_html:
                rendered, title = _html_to_markdown_and_title(decoded_text)
            else:
                rendered = decoded_text

        saved_path_str: str | None = None
        if save_to is not None and save_to.strip():
            ws_dir = await _resolve_ctx_workspace(ctx)
            raw_save = Path(save_to.strip())
            target_save = (
                raw_save if raw_save.is_absolute() else (ws_dir / raw_save)
            ).resolve()
            if not target_save.is_relative_to(ws_dir):
                raise ModelRetry(
                    f"Access denied: save_to path '{save_to}' must be inside the workspace directory ({ws_dir})."
                )
            agent_dir = self._agent_dir or _AGENT_DIR_CTX.get()
            if _is_protected_harness_path(target_save, agent_dir):
                raise ModelRetry(
                    f"Access denied: modifying protected Microharness file '{save_to}' is not permitted."
                )
            target_save.parent.mkdir(parents=True, exist_ok=True)
            target_save.write_text(rendered, encoding="utf-8")
            saved_path_str = str(target_save)

        cap_limit = (
            max_content_length
            if max_content_length is not None
            else self.max_content_length
        )
        truncated = False
        if cap_limit is not None and cap_limit > 0 and len(rendered) > cap_limit:
            rendered = (
                rendered[:cap_limit]
                + f"\n\n... [truncated {len(rendered) - cap_limit} chars; pass format='raw' or save_to='<file>' to inspect full content]"
            )
            truncated = True

        result_dict: dict[str, Any] = {
            "url": current_url,
            "status_code": status_code,
            "content_type": media_type,
            "format": fmt,
            "title": title,
            "truncated": truncated,
            "content": rendered,
        }
        if saved_path_str is not None:
            result_dict["saved_to"] = saved_path_str

        if (
            client_mod is not None
            and hasattr(client_mod, "transform_response")
            and callable(client_mod.transform_response)
        ):
            transformed = client_mod.transform_response(current_url, result_dict)
            if isinstance(transformed, dict):
                return transformed
        return result_dict

    def __post_init__(self) -> None:
        self._agent_dir = _AGENT_DIR_CTX.get()
        if not self.enabled:
            self.capabilities = []
            super().__post_init__()
            return

        allowed_pats, blocked_pats, _ = self._load_web_state()
        caps: list[AbstractCapability[Any]] = []

        if self.backend == "pydantic_ai":
            caps.append(
                _PydanticAIWebFetch(
                    native=self.native,
                    local=self.local,
                    allowed_domains=list(allowed_pats) if allowed_pats else None,
                    blocked_domains=list(blocked_pats) if blocked_pats else None,
                )
            )
        else:
            toolset = FunctionToolset(id="web_fetch")
            meta = {"microharness_local_tool": True, "code_mode": self.code_mode}

            async def web_fetch(
                ctx: RunContext[Any],
                url: str,
                format: WebFetchFormat = "markdown",
                headers: dict[str, str] | None = None,
                save_to: str | None = None,
                max_content_length: int | None = None,
            ) -> dict[str, Any]:
                """Fetch a URL natively using httpx2 with allowlist/blocklist & SSRF protection.

                Always use this tool instead of shell commands (`curl`, `wget`, etc.) when fetching web pages, APIs, or files.

                Args:
                    url: The HTTP or HTTPS URL to fetch.
                    format: Output format: 'markdown' (default, converts HTML to Markdown), 'raw' (preserves raw HTML/scripts/JSON), 'text' (visible text), or 'json' (formatted JSON).
                    headers: Optional HTTP request headers merged with `<agent>/web/client.py` defaults.
                    save_to: Optional relative file path inside the workspace (`./workdir`) to save the full un-truncated output (formatted according to `format`) on disk.
                    max_content_length: Optional character limit for the returned `content` field (default 50,000).
                """
                return await self._fetch_with_httpx2(
                    ctx,
                    url,
                    fmt=format,
                    headers=headers,
                    save_to=save_to,
                    max_content_length=max_content_length,
                )

            toolset.add_function(web_fetch, metadata=meta)
            caps.append(Toolset(toolset))

        if self.block_shell_fetch:
            caps.append(_ShellFetchGuardCapability(enabled=True))
        self.capabilities = caps
        super().__post_init__()

    def get_instructions(self) -> str | None:
        if not self.enabled:
            return None
        lines = [
            "## Native Web Fetching (`web_fetch` via `httpx2`)",
            "- Use the `web_fetch` tool for HTTP/HTTPS requests, web page downloads, and API calls.",
        ]
        if self.block_shell_fetch:
            lines.append(
                "- Never use `curl`, `wget`, `aria2c`, `httpie`, `lynx`, `w3m`, or inline Python HTTP clients in the `shell` tool."
            )
        if self.backend == "custom":
            lines.append(
                "- Use `format='markdown'` (default) for readable articles/docs, `format='raw'` when you need raw HTML or embedded `<script>`/JSON data, `format='json'` for JSON endpoints, and `save_to='<filename>'` to save large payloads directly into `./workdir`."
            )
        return "\n".join(lines)

    @classmethod
    def get_serialization_name(cls) -> str:
        return "WebFetch"


class HookRule(BaseModel):
    """Configuration for a single lifecycle hook (shell command or Python function)."""

    command: str | None = None
    python: str | None = None
    tools: list[str] | None = None
    timeout: float | None = None
    on_failure: Literal["retry", "abort", "ignore"] = "retry"


HookSpecInput = (
    str | HookRule | dict[str, Any] | list[str | HookRule | dict[str, Any]] | None
)


def _normalize_hook_rules(raw: HookSpecInput) -> list[HookRule]:
    """Normalize a YAML hook entry or list of entries into a list of HookRule objects."""
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    rules: list[HookRule] = []
    for item in items:
        if isinstance(item, HookRule):
            rules.append(item)
        elif isinstance(item, str):
            rules.append(
                HookRule(python=item.removeprefix("python:"))
                if item.startswith("python:")
                else HookRule(command=item)
            )
        elif isinstance(item, dict):
            rules.append(HookRule.model_validate(item))
    return rules


def load_hooks_from_module_spec(target: str) -> AbstractCapability[Any]:
    """Load a Hooks/Capability instance or `register_hooks(hooks)` from a file/module target."""
    loaded = _load_python_object(target)
    if isinstance(loaded, AbstractCapability):
        return loaded
    if callable(loaded):
        hooks_cap = Hooks()
        res = loaded(hooks_cap)
        return res if isinstance(res, AbstractCapability) else hooks_cap
    if hasattr(loaded, "hooks") and isinstance(loaded.hooks, AbstractCapability):
        return loaded.hooks
    if hasattr(loaded, "register_hooks") and callable(loaded.register_hooks):
        hooks_cap = Hooks()
        loaded.register_hooks(hooks_cap)
        return hooks_cap
    raise ValueError(
        f"Hook module '{target}' must expose a `hooks = Hooks()` instance, an AbstractCapability, "
        "or a `register_hooks(hooks: Hooks)` function."
    )


def _decode_stream(data: bytes | str | None) -> str:
    """Decode subprocess stdout/stderr into a stripped UTF-8 string."""
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace").strip()
    return str(data).strip()


async def _run_shell_hook(
    rule: HookRule,
    event_name: str,
    ctx: RunContext[Any],
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Execute a shell command hook with JSON stdin and MICROHARNESS_* environment variables."""
    assert rule.command is not None
    env = os.environ.copy()
    env["MICROHARNESS_HOOK_EVENT"] = event_name
    env["MICROHARNESS_RUN_STEP"] = str(getattr(ctx, "run_step", 0))
    if "tool_name" in payload:
        env["MICROHARNESS_TOOL_NAME"] = str(payload["tool_name"])
    if payload.get("tool_call_id") is not None:
        env["MICROHARNESS_TOOL_CALL_ID"] = str(payload["tool_call_id"])
    if "args" in payload:
        with contextlib.suppress(TypeError, ValueError):
            env["MICROHARNESS_TOOL_ARGS_JSON"] = json.dumps(
                payload["args"], default=str
            )

    stdin_bytes = json.dumps({"event": event_name, **payload}, default=str).encode(
        "utf-8"
    )
    proc = await asyncio.to_thread(
        subprocess.run,
        rule.command,
        shell=True,
        input=stdin_bytes,
        capture_output=True,
        env=env,
        timeout=rule.timeout,
    )
    stdout_text = _decode_stream(proc.stdout)
    stderr_text = _decode_stream(proc.stderr)

    if stderr_text and proc.returncode == 0:
        sys.stderr.write(f"[hook:{event_name}] {stderr_text}\n")
        sys.stderr.flush()

    if proc.returncode != 0:
        feedback = (
            stderr_text
            or stdout_text
            or f"Hook command `{rule.command}` exited with code {proc.returncode}"
        )
        if proc.returncode == 2 or rule.on_failure == "retry":
            raise ModelRetry(feedback)
        if rule.on_failure == "abort":
            raise RuntimeError(
                f"Hook `{rule.command}` aborted run (exit {proc.returncode}): {feedback}"
            )
        sys.stderr.write(
            f"[hook:{event_name}] Warning (exit {proc.returncode}): {feedback}\n"
        )
        sys.stderr.flush()
        return None

    if stdout_text.startswith("{") and stdout_text.endswith("}"):
        with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
            parsed = json.loads(stdout_text)
            if isinstance(parsed, dict):
                return parsed
    return None


def _build_shell_hook_callback(event_name: str, rule: HookRule) -> Callable[..., Any]:
    """Create a typed async hook callback matching Pydantic AI's Hooks protocol for `event_name`."""
    if event_name == "before_run":

        async def _before_run(ctx: RunContext[Any], /) -> None:
            await _run_shell_hook(rule, event_name, ctx, {})

        return _before_run

    if event_name == "after_run":

        async def _after_run(
            ctx: RunContext[Any], /, *, result: AgentRunResult[Any]
        ) -> AgentRunResult[Any]:
            await _run_shell_hook(rule, event_name, ctx, {"output": str(result.output)})
            return result

        return _after_run

    if event_name == "before_model_request":

        async def _before_model_request(
            ctx: RunContext[Any], request_context: ModelRequestContext, /
        ) -> ModelRequestContext:
            await _run_shell_hook(
                rule, event_name, ctx, {"message_count": len(request_context.messages)}
            )
            return request_context

        return _before_model_request

    if event_name == "after_model_request":

        async def _after_model_request(
            ctx: RunContext[Any],
            /,
            *,
            request_context: ModelRequestContext,
            response: ModelResponse,
        ) -> ModelResponse:
            await _run_shell_hook(
                rule,
                event_name,
                ctx,
                {"model_name": response.model_name, "part_count": len(response.parts)},
            )
            return response

        return _after_model_request

    if event_name in (
        "before_tool_validate",
        "after_tool_validate",
        "before_tool_execute",
    ):

        async def _tool_pre(
            ctx: RunContext[Any],
            /,
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: Any,
        ) -> Any:
            override = await _run_shell_hook(
                rule,
                event_name,
                ctx,
                {
                    "tool_name": call.tool_name,
                    "tool_call_id": call.tool_call_id,
                    "args": args,
                },
            )
            return (
                override["args"]
                if override and isinstance(override.get("args"), dict)
                else args
            )

        return _tool_pre

    if event_name == "after_tool_execute":

        async def _after_tool_execute(
            ctx: RunContext[Any],
            /,
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
            result: Any,
        ) -> Any:
            override = await _run_shell_hook(
                rule,
                event_name,
                ctx,
                {
                    "tool_name": call.tool_name,
                    "tool_call_id": call.tool_call_id,
                    "args": args,
                    "result": str(result),
                },
            )
            return override["result"] if override and "result" in override else result

        return _after_tool_execute

    if event_name in (
        "before_output_validate",
        "after_output_validate",
        "before_output_process",
        "after_output_process",
    ):

        async def _output_hook(
            ctx: RunContext[Any], /, *, output_context: OutputContext, output: Any
        ) -> Any:
            override = await _run_shell_hook(
                rule, event_name, ctx, {"output": str(output)}
            )
            return override["output"] if override and "output" in override else output

        return _output_hook

    raise ValueError(f"Unsupported shell hook event: {event_name}")


_TOOL_HOOK_EVENTS: frozenset[str] = frozenset(
    {
        "before_tool_validate",
        "after_tool_validate",
        "before_tool_execute",
        "after_tool_execute",
    }
)


def _resolve_hook_targets(
    modules: Sequence[str],
    agent_dir: Path | None = None,
) -> tuple[list[str], tuple[tuple[str, int], ...]]:
    """Resolve hook module targets into concrete targets and a file mtime snapshot for live reloading."""
    with _agent_dir_scope(agent_dir):
        resolved_targets: list[str] = []
        snapshot: list[tuple[str, int]] = []
        for mod_target in modules:
            candidate_dir = _resolve_agent_relative_path(mod_target)
            if ":" not in mod_target and candidate_dir.is_dir():
                py_files = _iter_public_files(candidate_dir)
                resolved_targets.extend(str(f) for f in py_files)
                _record_mtime(snapshot, *py_files)
            elif (
                ":" not in mod_target
                and _is_default_subdir_ref(mod_target, DEFAULT_HOOKS_DIR)
                and not candidate_dir.exists()
            ):
                continue
            else:
                resolved_targets.append(mod_target)
                cand_file = _resolve_agent_relative_path(mod_target.partition(":")[0])
                if cand_file.is_file():
                    _record_mtime(snapshot, cand_file.resolve())
        return resolved_targets, tuple(snapshot)


@dataclass
class _DynamicHooksCapability(AbstractCapability[Any]):
    """Per-run capability that hot-reloads `<agent>/hooks/*.py` when hook files are added or edited in ACP mode."""

    modules: tuple[str, ...]
    agent_dir: Path | None = None
    _cached_snapshot: tuple[tuple[str, int], ...] = field(default=(), repr=False)
    _cached_caps: tuple[AbstractCapability[Any], ...] = field(default=(), repr=False)

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        targets, snapshot = _resolve_hook_targets(self.modules, self.agent_dir)
        if snapshot != self._cached_snapshot or (targets and not self._cached_caps):
            with _agent_dir_scope(self.agent_dir):
                loaded = tuple(load_hooks_from_module_spec(t) for t in targets)
            self._cached_snapshot = snapshot
            self._cached_caps = loaded
        if not self._cached_caps:
            return self
        combined = CombinedCapability(list(self._cached_caps))
        return await combined.for_run(ctx)


@dataclass
class HooksSpec(_SpecCombinedCapability):
    """Declarative Hooks capability for YAML Agent Specs supporting shell commands & Python hooks."""

    modules: list[str] | None = None
    event_rules: dict[str, HookSpecInput] = field(default_factory=dict)

    def __post_init__(self) -> None:
        caps: list[AbstractCapability[Any]] = []
        agent_dir = _AGENT_DIR_CTX.get()
        if self.modules:
            targets, snapshot = _resolve_hook_targets(self.modules, agent_dir)
            initial_caps = tuple(load_hooks_from_module_spec(t) for t in targets)
            caps.append(
                _DynamicHooksCapability(
                    modules=tuple(self.modules),
                    agent_dir=agent_dir,
                    _cached_snapshot=snapshot,
                    _cached_caps=initial_caps,
                )
            )

        inline_hooks = Hooks()
        has_inline = False

        for event_name, raw_rules in self.event_rules.items():
            for rule in _normalize_hook_rules(raw_rules):
                if rule.python:
                    fn = _load_python_object(rule.python)
                elif rule.command:
                    fn = _build_shell_hook_callback(event_name, rule)
                else:
                    continue

                registrar = getattr(inline_hooks.on, event_name)
                reg_kwargs: dict[str, Any] = {}
                if rule.timeout is not None:
                    reg_kwargs["timeout"] = rule.timeout
                if rule.tools is not None and event_name in _TOOL_HOOK_EVENTS:
                    reg_kwargs["tools"] = rule.tools

                (registrar(**reg_kwargs) if reg_kwargs else registrar)(fn)
                has_inline = True

        if has_inline:
            caps.append(inline_hooks)

        self.capabilities = caps
        super().__post_init__()

    @classmethod
    def from_spec(
        cls,
        modules: list[str] | None = None,
        before_run: HookSpecInput = None,
        after_run: HookSpecInput = None,
        before_model_request: HookSpecInput = None,
        after_model_request: HookSpecInput = None,
        before_tool_validate: HookSpecInput = None,
        after_tool_validate: HookSpecInput = None,
        before_tool_execute: HookSpecInput = None,
        after_tool_execute: HookSpecInput = None,
        before_output_validate: HookSpecInput = None,
        after_output_validate: HookSpecInput = None,
        before_output_process: HookSpecInput = None,
        after_output_process: HookSpecInput = None,
    ) -> HooksSpec:
        return cls(
            modules=modules,
            event_rules={
                "before_run": before_run,
                "after_run": after_run,
                "before_model_request": before_model_request,
                "after_model_request": after_model_request,
                "before_tool_validate": before_tool_validate,
                "after_tool_validate": after_tool_validate,
                "before_tool_execute": before_tool_execute,
                "after_tool_execute": after_tool_execute,
                "before_output_validate": before_output_validate,
                "after_output_validate": after_output_validate,
                "before_output_process": before_output_process,
                "after_output_process": after_output_process,
            },
        )

    @classmethod
    def get_serialization_name(cls) -> str:
        return "Hooks"


_OTEL_PROVIDERS_FLUSH_REGISTERED: bool = False
_ACTIVE_PROVIDERS: list[Any] = []
_OTEL_VERSIONS: dict[str, Literal[2, 3, 4, 5, 6]] = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
}


def flush_opentelemetry() -> None:
    """Force-flush all active OpenTelemetry TracerProviders and MeterProviders."""
    for provider in _ACTIVE_PROVIDERS:
        with contextlib.suppress(OSError, RuntimeError, ValueError):
            provider.force_flush()


def _signal_url(explicit: str | None, base: str | None, suffix: str) -> str | None:
    """Resolve an OTLP HTTP signal URL from an explicit URL or base endpoint."""
    if explicit:
        return explicit
    if not base:
        return None
    stripped = base.rstrip("/")
    return stripped if stripped.endswith(suffix) else f"{stripped}{suffix}"


def setup_opentelemetry_instrumentation(
    *,
    enabled: bool | None = None,
    endpoint: str | None = None,
    traces_endpoint: str | None = None,
    metrics_endpoint: str | None = None,
    protocol: str | None = None,
    service_name: str | None = None,
    headers: dict[str, str] | str | None = None,
    resource_attributes: dict[str, str] | str | None = None,
    console: bool | None = None,
    include_content: bool | None = None,
    include_binary_content: bool | None = None,
    include_model_request_parameters: bool | None = None,
    version: Literal[2, 3, 4, 5, 6] | None = None,
    use_aggregated_usage_attribute_names: bool = True,
    instrument_all: bool = True,
) -> InstrumentationSettings | None:
    """Configure OpenTelemetry SDK providers/exporters from args + env vars and return InstrumentationSettings."""
    global _OTEL_PROVIDERS_FLUSH_REGISTERED

    if _env_bool("OTEL_SDK_DISABLED", False):
        return None

    cli = _OTEL_CLI_CTX.get() or {}
    enabled = cli["enabled"] if cli.get("enabled") is not None else enabled
    endpoint = cli["endpoint"] if cli.get("endpoint") is not None else endpoint
    service_name = (
        cli["service_name"] if cli.get("service_name") is not None else service_name
    )
    console = cli["console"] if cli.get("console") is not None else console

    env_enabled = _env_bool("MICROHARNESS_OTEL_ENABLED", None)
    if enabled is False or env_enabled is False:
        return None

    effective_console = (
        console
        if console is not None
        else bool(_env_bool("MICROHARNESS_OTEL_CONSOLE", False))
    )
    effective_endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    effective_traces_endpoint = traces_endpoint or os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    )
    effective_metrics_endpoint = metrics_endpoint or os.environ.get(
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"
    )

    if not (
        enabled is True
        or env_enabled is True
        or effective_console
        or effective_endpoint
        or effective_traces_endpoint
        or effective_metrics_endpoint
    ):
        return None

    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter as GrpcMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GrpcSpanExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter as HttpMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HttpSpanExporter,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    res_attrs = {
        **_parse_kv_pairs(os.environ.get("OTEL_RESOURCE_ATTRIBUTES")),
        **_parse_kv_pairs(resource_attributes),
        "service.name": service_name
        or os.environ.get("OTEL_SERVICE_NAME")
        or "microharness",
    }
    resource = Resource.create(res_attrs)
    merged_headers = {
        **_parse_kv_pairs(os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")),
        **_parse_kv_pairs(headers),
    } or None

    effective_protocol = (
        (
            protocol
            or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL")
            or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL")
            or "http/protobuf"
        )
        .strip()
        .lower()
    )

    tracer_provider = TracerProvider(resource=resource)
    metric_readers: list[Any] = []

    if effective_endpoint or effective_traces_endpoint or effective_metrics_endpoint:
        if effective_protocol == "grpc":
            if span_target := (effective_traces_endpoint or effective_endpoint):
                tracer_provider.add_span_processor(
                    BatchSpanProcessor(
                        GrpcSpanExporter(endpoint=span_target, headers=merged_headers)
                    )
                )
            if metric_target := (effective_metrics_endpoint or effective_endpoint):
                metric_readers.append(
                    PeriodicExportingMetricReader(
                        GrpcMetricExporter(
                            endpoint=metric_target, headers=merged_headers
                        )
                    )
                )
        else:
            if span_url := _signal_url(
                effective_traces_endpoint, effective_endpoint, "/v1/traces"
            ):
                tracer_provider.add_span_processor(
                    BatchSpanProcessor(
                        HttpSpanExporter(endpoint=span_url, headers=merged_headers)
                    )
                )
            if metric_url := _signal_url(
                effective_metrics_endpoint, effective_endpoint, "/v1/metrics"
            ):
                metric_readers.append(
                    PeriodicExportingMetricReader(
                        HttpMetricExporter(endpoint=metric_url, headers=merged_headers)
                    )
                )

    if effective_console:
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(ConsoleSpanExporter(out=sys.stderr))
        )

    meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
    if not _ACTIVE_PROVIDERS:
        trace.set_tracer_provider(tracer_provider)
        metrics.set_meter_provider(meter_provider)

    _ACTIVE_PROVIDERS.extend((tracer_provider, meter_provider))
    if not _OTEL_PROVIDERS_FLUSH_REGISTERED:
        atexit.register(flush_opentelemetry)
        _OTEL_PROVIDERS_FLUSH_REGISTERED = True

    def _resolve_flag(
        env_key: str, explicit: bool | None, default: bool = True
    ) -> bool:
        env_val = _env_bool(env_key, None)
        if env_val is not None:
            return env_val
        return explicit if explicit is not None else default

    env_version_raw = (os.environ.get("MICROHARNESS_OTEL_VERSION") or "").strip()
    eff_version: Literal[2, 3, 4, 5, 6] = version or _OTEL_VERSIONS.get(
        env_version_raw, 5
    )

    settings = InstrumentationSettings(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        include_content=_resolve_flag(
            "MICROHARNESS_OTEL_INCLUDE_CONTENT", include_content
        ),
        include_binary_content=_resolve_flag(
            "MICROHARNESS_OTEL_INCLUDE_BINARY_CONTENT", include_binary_content
        ),
        include_model_request_parameters=_resolve_flag(
            "MICROHARNESS_OTEL_INCLUDE_PARAMETERS", include_model_request_parameters
        ),
        version=eff_version,
        use_aggregated_usage_attribute_names=use_aggregated_usage_attribute_names,
    )
    if instrument_all:
        Agent.instrument_all(settings)
    return settings


@dataclass
class OpenTelemetrySpec(_SpecCombinedCapability):
    """Configures OpenTelemetry SDK (OTLP HTTP/gRPC or console) and Pydantic AI Instrumentation."""

    enabled: bool = True
    endpoint: str | None = None
    traces_endpoint: str | None = None
    metrics_endpoint: str | None = None
    protocol: Literal["http/protobuf", "http", "grpc"] | None = None
    service_name: str | None = None
    headers: dict[str, str] | str | None = None
    resource_attributes: dict[str, str] | str | None = None
    console: bool | None = None
    include_content: bool | None = None
    include_binary_content: bool | None = None
    include_model_request_parameters: bool | None = None
    version: Literal[2, 3, 4, 5, 6] = 5
    use_aggregated_usage_attribute_names: bool = True
    instrument_all: bool = True

    def __post_init__(self) -> None:
        settings = setup_opentelemetry_instrumentation(
            enabled=self.enabled,
            endpoint=self.endpoint,
            traces_endpoint=self.traces_endpoint,
            metrics_endpoint=self.metrics_endpoint,
            protocol=self.protocol,
            service_name=self.service_name,
            headers=self.headers,
            resource_attributes=self.resource_attributes,
            console=self.console,
            include_content=self.include_content,
            include_binary_content=self.include_binary_content,
            include_model_request_parameters=self.include_model_request_parameters,
            version=self.version,
            use_aggregated_usage_attribute_names=self.use_aggregated_usage_attribute_names,
            instrument_all=self.instrument_all,
        )
        self.capabilities = (
            [Instrumentation(settings=settings)] if settings is not None else []
        )
        super().__post_init__()

    @classmethod
    def get_serialization_name(cls) -> str:
        return "OpenTelemetry"


_SHELL_EXEC_TOOLS: frozenset[str] = frozenset(
    {"shell", "run_command", "start_command", "stop_command"}
)
_SHELL_COMMAND_TOOLS: frozenset[str] = frozenset(
    {"shell", "run_command", "start_command"}
)
_WRITE_MUTATING_TOOLS: frozenset[str] = frozenset(
    {"write_file", "edit_file", "create_directory", "delete_file"}
)
_INTERNAL_META_TOOLS: frozenset[str] = frozenset({"load_capability", "search_tools"})
_SHELL_COMPOUND_RE = re.compile(r"[;&|`<>\n\r]|\$\(")


def _is_simple_shell_command(command: str) -> bool:
    """Return True if `command` has no shell chaining, redirection, or subshell operators."""
    return bool(command) and not _SHELL_COMPOUND_RE.search(command)


def _matches_tool_pattern(tool_name: str, patterns: Sequence[str] | None) -> bool:
    """Return True if `tool_name` matches any exact name or glob in `patterns`."""
    if not patterns:
        return False
    return any(
        tool_name == pat or fnmatch.fnmatchcase(tool_name, pat)
        for pat in patterns
        if pat
    )


def _matches_shell_pattern(command: str, patterns: Sequence[str] | None) -> bool:
    """Return True if `command` matches any glob or command prefix in `patterns`."""
    if not patterns or not command:
        return False
    for raw_pat in patterns:
        pat = raw_pat.strip()
        if not pat:
            continue
        if any(ch in pat for ch in ("*", "?", "[")):
            if fnmatch.fnmatchcase(command, pat):
                return True
        elif command == pat or command.startswith(f"{pat} "):
            return True
    return False


def build_acp_permission_policy(
    remember_scope: ApprovalRememberScope = "exact",
) -> Callable[[ToolCallPermission], Hashable]:
    """Build an ACP `permission_policy` controlling how the ACP client's 'Always allow'/'Always reject' is scoped."""
    if remember_scope == "exact":
        return default_permission_scope

    def _policy(call: ToolCallPermission) -> Hashable:
        if remember_scope == "tool":
            return call.tool_name
        if call.tool_name in _SHELL_COMMAND_TOOLS:
            cmd = str(call.args.get("command") or "").strip()
            if cmd and _is_simple_shell_command(cmd):
                return (call.tool_name, cmd.split()[0])
            return default_permission_scope(call)
        return call.tool_name

    return _policy


def _prompt_cli_tool_approval(tool_name: str, args: Mapping[str, Any]) -> bool:
    """Prompt the user on `/dev/tty` (or interactive stdin) to approve a CLI tool call."""
    summary_arg = ""
    if tool_name in _SHELL_COMMAND_TOOLS:
        summary_arg = str(args.get("command") or "")
    elif "path" in args:
        summary_arg = str(args.get("path") or "")
    elif "url" in args:
        summary_arg = str(args.get("url") or "")
    label = f"{tool_name}({summary_arg})" if summary_arg else tool_name
    prompt_str = f"\n[Approve tool] {label} [y/N]: "

    with (
        contextlib.suppress(OSError),
        open("/dev/tty", "r+", encoding="utf-8") as tty,
    ):
        tty.write(prompt_str)
        tty.flush()
        ans = tty.readline().strip().lower()
        return ans in ("y", "yes")

    if sys.stdin.isatty():
        sys.stderr.write(prompt_str)
        sys.stderr.flush()
        ans = sys.stdin.readline().strip().lower()
        return ans in ("y", "yes")
    return False


@dataclass
class ApprovalsSpec(AbstractCapability[Any]):
    """Configurable human-in-the-loop tool approval policy for ACP clients and CLI."""

    preset: ApprovalPreset = "write"
    enabled: bool = True
    tools: list[str] | None = None
    exclude_tools: list[str] | None = None
    shell_commands: list[str] | None = None
    safe_shell_commands: list[str] | None = None
    approve_web_fetch_save: bool = True
    remember_scope: ApprovalRememberScope = "exact"
    subagent_mode: ApprovalSubagentMode = "approve_delegation"
    cli_mode: ApprovalCliMode = "allow"

    def get_ordering(self) -> CapabilityOrdering | None:
        return CapabilityOrdering(
            wrapped_by=(
                CoderSpec,
                WebFetchSpec,
                HooksSpec,
                _HarnessSelfProtectionCapability,
                _ShellFetchGuardCapability,
            )
        )

    def _requires_approval_for_call(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        *,
        preset: ApprovalPreset,
    ) -> bool:
        if _matches_tool_pattern(tool_name, self.exclude_tools):
            return False

        if tool_name == "delegate_task":
            if self.subagent_mode == "allow":
                return False
            return preset in ("write", "shell", "all") or _matches_tool_pattern(
                tool_name, self.tools
            )

        explicitly_included = _matches_tool_pattern(tool_name, self.tools)
        in_base_set = explicitly_included
        if not in_base_set:
            if preset == "all":
                in_base_set = tool_name not in _INTERNAL_META_TOOLS
            elif preset == "write":
                in_base_set = (
                    tool_name in _SHELL_EXEC_TOOLS
                    or tool_name in _WRITE_MUTATING_TOOLS
                    or tool_name == "run_code"
                    or (
                        self.approve_web_fetch_save
                        and tool_name == "web_fetch"
                        and bool(str(args.get("save_to") or "").strip())
                    )
                )
            elif preset == "shell":
                in_base_set = tool_name in _SHELL_EXEC_TOOLS or tool_name == "run_code"
            elif preset == "custom" and tool_name in _SHELL_COMMAND_TOOLS:
                in_base_set = bool(self.shell_commands)

        if not in_base_set:
            return False

        if tool_name in _SHELL_COMMAND_TOOLS:
            cmd = str(args.get("command") or "").strip()
            if (
                self.safe_shell_commands
                and _is_simple_shell_command(cmd)
                and _matches_shell_pattern(cmd, self.safe_shell_commands)
                and not _matches_shell_pattern(cmd, self.shell_commands)
            ):
                return False
            if self.shell_commands is not None:
                return _matches_shell_pattern(cmd, self.shell_commands)

        return True

    async def wrap_run(
        self,
        ctx: RunContext[Any],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        depth = _RUN_DEPTH_CTX.get()
        token = _RUN_DEPTH_CTX.set(depth + 1)
        try:
            return await handler()
        finally:
            _RUN_DEPTH_CTX.reset(token)

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        if call.tool_name == "run_code":
            token = _IN_RUN_CODE_CTX.set(True)
            try:
                return await handler(args)
            finally:
                _IN_RUN_CODE_CTX.reset(token)
        return await handler(args)

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        if ctx.tool_call_approved:
            return args
        if call.tool_name != "run_code" and _IN_RUN_CODE_CTX.get():
            return args

        cli_override = _APPROVALS_CLI_CTX.get()
        if cli_override is not None:
            effective_enabled = cli_override != "none"
            effective_preset: ApprovalPreset = cli_override
        else:
            effective_enabled = self.enabled and self.preset != "none"
            effective_preset = self.preset

        if not effective_enabled:
            return args

        if not self._requires_approval_for_call(
            call.tool_name, args, preset=effective_preset
        ):
            return args

        if _RUN_DEPTH_CTX.get() > 1:
            if self.subagent_mode == "deny":
                raise ModelRetry(
                    f"Access denied: sub-agent execution of '{call.tool_name}' is blocked by Approvals(subagent_mode='deny')."
                )
            return args

        if _ACP_MODE_CTX.get():
            raise ApprovalRequired()

        if self.cli_mode == "allow":
            return args
        if self.cli_mode == "prompt" and _prompt_cli_tool_approval(
            call.tool_name, args
        ):
            return args
        raise ModelRetry(
            f"Tool '{call.tool_name}' was not approved for execution in CLI mode."
        )

    @classmethod
    def get_serialization_name(cls) -> str:
        return "Approvals"


_HARNESS_SPEC_CAPABILITY_NAMES: tuple[str, ...] = (
    "Advisor",
    "ApprovalReviewer",
    "BackgroundTools",
    "ClampOversizedMessages",
    "ClearToolResults",
    "ConversationSearch",
    "DayAI",
    "DockerSandbox",
    "DeduplicateFileReads",
    "FallbackCompaction",
    "FileSystem",
    "Grain",
    "LocalStack",
    "Macroscope",
    "Memory",
    "Ordinal",
    "Planning",
    "PostHog",
    "PydanticAIDocs",
    "Pylon",
    "RelevanceCompaction",
    "ReportContextUsage",
    "RepoContext",
    "Shell",
    "SlidingWindowCompaction",
    "SmartSearch",
    "SpendLimits",
    "SSHWorkspace",
    "StackOne",
    "StepPersistence",
    "SummarizingCompaction",
    "TieredCompaction",
    "ToolCallJudge",
    "WarnNearLimits",
    "WarnOnCacheBusts",
)


def get_custom_capability_types() -> list[type[AbstractCapability[Any]]]:
    """Return all Pydantic AI Harness and Microharness capabilities usable in YAML Agent Specs."""
    import pydantic_ai_harness

    types: list[type[AbstractCapability[Any]]] = [
        Lemonade,
        LocalWorkspaceSpec,
        RulesSpec,
        KnowledgeSpec,
        WebFetchSpec,
        LocalToolsSpec,
        SkillsSpec,
        HooksSpec,
        ApprovalsSpec,
        OpenTelemetrySpec,
        CoderSpec,
        CodeModeSpec,
        ResearcherSpec,
        SubAgentsSpec,
    ]
    for name in _HARNESS_SPEC_CAPABILITY_NAMES:
        cap_cls = getattr(pydantic_ai_harness, name, None)
        if (
            isinstance(cap_cls, type)
            and issubclass(cap_cls, AbstractCapability)
            and "__dataclass_fields__" in cap_cls.__dict__
            and cap_cls.get_serialization_name() is not None
        ):
            types.append(cap_cls)
    return types


def resolve_agent_location(
    agent_name: str | None = None,
    spec_path: Path | None = None,
) -> tuple[Path | None, Path, str]:
    """Resolve `(spec_file_or_none, agent_dir, effective_agent_name)` from `--agent` and `--spec`."""
    if spec_path is not None:
        if spec_path.is_dir():
            spec_file = spec_path / DEFAULT_SPEC_FILENAME
            if not spec_file.exists():
                raise typer.BadParameter(
                    f"Agent spec file not found in directory: {spec_file}"
                )
            return spec_file.resolve(), spec_path.resolve(), spec_path.resolve().name
        if not spec_path.exists():
            raise typer.BadParameter(f"Agent spec file not found: {spec_path}")
        resolved_spec = spec_path.resolve()
        return resolved_spec, resolved_spec.parent, resolved_spec.parent.name

    effective_name = agent_name or DEFAULT_AGENT_NAME
    for cand in (
        INITIAL_CWD / DEFAULT_AGENTS_DIR / effective_name,
        SCRIPT_DIR / DEFAULT_AGENTS_DIR / effective_name,
        INITIAL_CWD / effective_name,
        SCRIPT_DIR / effective_name,
    ):
        if cand.is_file():
            resolved_spec = cand.resolve()
            return resolved_spec, resolved_spec.parent, resolved_spec.stem
        if cand.is_dir():
            for fname in (DEFAULT_SPEC_FILENAME, f"{cand.name}.yaml"):
                if (spec_file := cand / fname).exists():
                    return spec_file.resolve(), cand.resolve(), cand.resolve().name

    if agent_name is not None and agent_name != DEFAULT_AGENT_NAME:
        raise typer.BadParameter(
            f"Agent '{agent_name}' not found. Expected subdirectory '{DEFAULT_AGENTS_DIR / agent_name / DEFAULT_SPEC_FILENAME}'."
        )

    if (legacy_root_spec := INITIAL_CWD / DEFAULT_SPEC_FILENAME).exists():
        return legacy_root_spec.resolve(), INITIAL_CWD, DEFAULT_AGENT_NAME

    default_dir = SCRIPT_DIR / DEFAULT_AGENTS_DIR / DEFAULT_AGENT_NAME
    return None, (default_dir if default_dir.is_dir() else SCRIPT_DIR), effective_name


def load_agent_spec(
    agent_name: str | None = None,
    spec_path: Path | None = None,
) -> tuple[AgentSpec, Path]:
    """Load an AgentSpec from the resolved agent subdirectory or spec file."""
    resolved_spec, agent_dir, effective_name = resolve_agent_location(
        agent_name, spec_path
    )
    _AGENT_DIR_CTX.set(agent_dir)
    if resolved_spec is not None:
        return AgentSpec.from_file(resolved_spec), agent_dir
    return AgentSpec.from_dict(make_default_spec_dict(effective_name)), agent_dir


_WORKSPACE_CAP_NAMES: frozenset[str] = frozenset(
    {
        "LocalWorkspace",
        "DockerSandbox",
        "SSHWorkspace",
        "ModalSandbox",
        "E2BSandbox",
        "SpritesSandbox",
        "DaytonaSandbox",
        "SmolSandbox",
        "BoxLiteSandbox",
        "BubblewrapSandbox",
    }
)


@dataclass(frozen=True)
class RuntimeOptions:
    """CLI and runtime overrides shared across ACP and single-query execution modes."""

    agent_name: str = DEFAULT_AGENT_NAME
    spec_path: Path | None = None
    model: str | None = None
    workspace: Path | None = None
    workspace_mode: Literal["server", "client"] = "server"
    instructions: str | None = None
    tools_paths: Sequence[str] | None = None
    hooks_modules: Sequence[str] | None = None
    approvals: ApprovalPreset | None = None
    otel_enabled: bool | None = None
    otel_endpoint: str | None = None
    otel_service_name: str | None = None
    otel_console: bool | None = None


def build_agent(
    opts: RuntimeOptions,
    *,
    defer_model_check: bool = False,
) -> tuple[Agent[None, Any], AgentSpec]:
    """Build a Pydantic AI Agent from the agent subdirectory's YAML spec and runtime overrides."""
    spec, agent_dir = load_agent_spec(
        agent_name=opts.agent_name, spec_path=opts.spec_path
    )
    spec_cap_names = {cap.name for cap in spec.capabilities}

    extra_capabilities: list[AbstractCapability[Any]] = []
    if "Lemonade" not in spec_cap_names:
        extra_capabilities.append(Lemonade())

    if opts.workspace is not None:
        extra_capabilities.append(
            LocalWorkspaceSpec(str(_ensure_workspace_dir(opts.workspace)))
        )
    elif not (spec_cap_names & _WORKSPACE_CAP_NAMES):
        extra_capabilities.append(LocalWorkspaceSpec(DEFAULT_WORKDIR_SPEC))

    if opts.tools_paths:
        extra_capabilities.append(LocalToolsSpec(paths=list(opts.tools_paths)))
    elif (
        "LocalTools" not in spec_cap_names and (agent_dir / DEFAULT_TOOLS_DIR).is_dir()
    ):
        extra_capabilities.append(
            LocalToolsSpec(paths=str(agent_dir / DEFAULT_TOOLS_DIR))
        )

    if "Rules" not in spec_cap_names and (agent_dir / DEFAULT_RULES_DIR).is_dir():
        extra_capabilities.append(RulesSpec(paths=str(agent_dir / DEFAULT_RULES_DIR)))

    if (
        "Knowledge" not in spec_cap_names
        and (agent_dir / DEFAULT_KNOWLEDGE_DIR).is_dir()
    ):
        extra_capabilities.append(
            KnowledgeSpec(paths=str(agent_dir / DEFAULT_KNOWLEDGE_DIR))
        )

    if "Skills" not in spec_cap_names and (agent_dir / DEFAULT_SKILLS_DIR).is_dir():
        extra_capabilities.append(
            SkillsSpec(directories=str(agent_dir / DEFAULT_SKILLS_DIR))
        )

    if opts.hooks_modules:
        extra_capabilities.append(HooksSpec(modules=list(opts.hooks_modules)))
    elif "Hooks" not in spec_cap_names and (agent_dir / DEFAULT_HOOKS_DIR).is_dir():
        extra_capabilities.append(
            HooksSpec(modules=[str(agent_dir / DEFAULT_HOOKS_DIR)])
        )

    _APPROVALS_CLI_CTX.set(opts.approvals)
    if (
        opts.approvals is not None
        and opts.approvals != "none"
        and "Approvals" not in spec_cap_names
    ):
        extra_capabilities.append(ApprovalsSpec(preset=opts.approvals))

    _OTEL_CLI_CTX.set(
        {
            "enabled": opts.otel_enabled,
            "endpoint": opts.otel_endpoint,
            "service_name": opts.otel_service_name,
            "console": opts.otel_console,
        }
    )

    if "OpenTelemetry" not in spec_cap_names:
        otel_settings = setup_opentelemetry_instrumentation(
            enabled=opts.otel_enabled,
            endpoint=opts.otel_endpoint,
            service_name=opts.otel_service_name or spec.name or "microharness",
            console=opts.otel_console,
        )
        if otel_settings is not None:
            extra_capabilities.append(Instrumentation(settings=otel_settings))

    kwargs: dict[str, Any] = {
        "custom_capability_types": get_custom_capability_types(),
        "defer_model_check": defer_model_check,
    }
    effective_model = (
        opts.model or spec.model or os.environ.get("PYDANTIC_AI_MODEL", DEFAULT_MODEL)
    )
    if effective_model == "test":
        kwargs["model"] = TestModel(call_tools=[])
    elif opts.model is not None or spec.model is None:
        kwargs["model"] = effective_model

    if opts.instructions is not None:
        kwargs["instructions"] = opts.instructions
    if extra_capabilities:
        kwargs["capabilities"] = extra_capabilities

    return Agent.from_spec(spec, **kwargs), spec


_PLANNING_WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "write_plan",
        "add_task",
        "update_task_status",
        "update_task_statuses",
        "remove_task",
        "add_subtask",
        "set_dependency",
    }
)
_DIFF_PRESENTED_TOOLS: frozenset[str] = frozenset(
    {"edit_file", "write_file", "create_directory"}
)
_MAX_ACP_TOOL_CONTENT_CHARS = 16_384


def _format_acp_tool_result_text(tool_name: str, result: Any) -> str | None:
    """Format a completed tool result into a readable text block for ACP `ContentToolCallContent`."""
    if (
        result is None
        or tool_name in _DIFF_PRESENTED_TOOLS
        or tool_name in _PLANNING_WRITE_TOOLS
    ):
        return None

    if tool_name == "search_knowledge" and isinstance(result, list):
        if not result:
            return "No matching knowledge entries found."
        lines: list[str] = []
        for hit in result:
            if not isinstance(hit, dict):
                continue
            loc = f"{hit.get('bundle', '')}:{hit.get('path', '')}".strip(":")
            title = hit.get("title") or loc
            dtype = hit.get("type", "concept")
            snippet = hit.get("snippet") or ""
            lines.append(f"- **{title}** (`{loc}`, type: `{dtype}`)")
            if snippet:
                lines.append(f"  {snippet}")
        text = "\n".join(lines)
    elif tool_name == "read_knowledge" and isinstance(result, dict):
        if "error" in result:
            text = f"Error: {result['error']}"
        else:
            loc = f"{result.get('bundle', '')}:{result.get('path', '')}".strip(":")
            meta_parts = [f"path: `{loc}`", f"type: `{result.get('type', 'concept')}`"]
            if out_links := result.get("outgoing_links"):
                meta_parts.append(f"links: {len(out_links)}")
            if backlinks := result.get("backlinks"):
                meta_parts.append(f"backlinks: {len(backlinks)}")
            header = " | ".join(meta_parts)
            body = str(result.get("content", "")).strip()
            text = f"{header}\n\n{body}" if body else header
    elif tool_name == "web_fetch" and isinstance(result, dict):
        meta_parts = [
            f"url: `{result.get('url', '')}`",
            f"status: `{result.get('status_code', 200)}`",
            f"format: `{result.get('format', 'markdown')}`",
        ]
        if saved_to := result.get("saved_to"):
            meta_parts.append(f"saved_to: `{saved_to}`")
        header = " | ".join(meta_parts)
        body = str(result.get("content", "")).strip()
        text = f"{header}\n\n{body}" if body else header
    elif isinstance(result, str):
        text = result.strip()
    elif isinstance(result, (dict, list)):
        try:
            text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(result).strip()
    else:
        text = str(result).strip()

    if not text:
        return None
    if len(text) > _MAX_ACP_TOOL_CONTENT_CHARS:
        text = (
            text[:_MAX_ACP_TOOL_CONTENT_CHARS]
            + f"\n... [truncated {len(text) - _MAX_ACP_TOOL_CONTENT_CHARS} chars]"
        )
    return text


def _plan_items_to_acp_entries(
    items: Sequence[PlanItem],
) -> list[acp_schema.PlanEntry]:
    """Convert `pydantic_ai_harness.planning` items into ACP `PlanEntry` checklist items."""
    entries: list[acp_schema.PlanEntry] = []
    for item in items:
        prefix = "  " if item.parent_id is not None else ""
        if item.status is TaskStatus.completed:
            status: acp_schema.PlanEntryStatus = "completed"
            label = f"{prefix}{item.content}"
        elif item.status is TaskStatus.cancelled:
            status = "completed"
            label = f"{prefix}[cancelled] {item.content}"
        elif item.status is TaskStatus.in_progress:
            status = "in_progress"
            label = f"{prefix}{item.content}"
        elif item.status is TaskStatus.blocked:
            status = "pending"
            label = f"{prefix}[blocked] {item.content}"
        else:
            status = "pending"
            label = f"{prefix}{item.content}"
        entries.append(acp.plan_entry(label, priority="medium", status=status))
    return entries


@dataclass
class _AcpPlanBridgeCapability(AbstractCapability[None]):
    """Per-session capability that persists `Planning` across ACP turns and streams `acp.update_plan` to the ACP client UI."""

    session: AcpSession
    session_store: PlanStore = field(default_factory=InMemoryPlanStore)

    def _find_planning(self, ctx: RunContext[Any]) -> Planning[Any] | None:
        if ctx.root_capability is None:
            return None
        found: Planning[Any] | None = None

        def _visit(candidate: AbstractCapability[Any]) -> None:
            nonlocal found
            while isinstance(candidate, WrapperCapability):
                candidate = candidate.wrapped
            if isinstance(candidate, Planning):
                found = candidate

        ctx.root_capability.apply(_visit)
        return found

    async def before_run(self, ctx: RunContext[None]) -> None:
        planning_cap = self._find_planning(ctx)
        if (
            planning_cap is not None
            and planning_cap.store is None
            and planning_cap.store_resolver is None
        ):
            planning_cap._resolved_store = self.session_store

    async def after_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        if call.tool_name in _PLANNING_WRITE_TOOLS:
            planning_cap = self._find_planning(ctx)
            store = (
                planning_cap.resolve_store(ctx)
                if planning_cap is not None
                else self.session_store
            )
            items = await store.get_items()
            entries = _plan_items_to_acp_entries(items)
            with contextlib.suppress(Exception):
                await self.session.client.session_update(
                    session_id=self.session.session_id,
                    update=acp.update_plan(entries),
                )
        elif (
            formatted_text := _format_acp_tool_result_text(call.tool_name, result)
        ) is not None:
            with contextlib.suppress(Exception):
                await self.session.client.session_update(
                    session_id=self.session.session_id,
                    update=acp.update_tool_call(
                        tool_call_id=call.tool_call_id,
                        content=[acp.tool_content(acp.text_block(formatted_text))],
                    ),
                )
        return result


def _nonempty_str(args: Mapping[str, object], key: str) -> str | None:
    val = args.get(key)
    return val if isinstance(val, str) and val else None


def _any_str(args: Mapping[str, object], key: str) -> str | None:
    val = args.get(key)
    return val if isinstance(val, str) else None


def _resolve_workspace_path(raw_path: str, workspace_dir: Path) -> str | None:
    """Resolve a workspace-relative or absolute path inside `workspace_dir` to an absolute path string."""
    ws_root = workspace_dir.resolve()
    if os.path.isabs(raw_path):
        return str(Path(raw_path).resolve())
    resolved = (ws_root / raw_path).resolve()
    if not resolved.is_relative_to(ws_root):
        return None
    return str(resolved)


def build_microharness_tool_presenter(
    workspace_resolver: Callable[[], Path] | Path | None = None,
) -> Callable[[ToolCallPart], ToolCallPresentation | None]:
    """Build a rich ACP `ToolCallPresenter` for Microharness's native server-side tools."""

    def _current_workspace() -> Path:
        if isinstance(workspace_resolver, Path):
            return workspace_resolver
        if workspace_resolver is not None:
            return workspace_resolver()
        return Path.cwd()

    def _present(call: ToolCallPart) -> ToolCallPresentation | None:
        args = call.args_as_dict()
        name = call.tool_name
        ws_dir = _current_workspace()

        if name == "edit_file":
            path = _nonempty_str(args, "path")
            old_text = _nonempty_str(args, "old_text")
            new_text = _any_str(args, "new_text")
            if path is None or old_text is None or new_text is None:
                return None
            abs_path = _resolve_workspace_path(path, ws_dir)
            if abs_path is None:
                return None
            return ToolCallPresentation(
                kind="edit",
                title=f"Edit {path}",
                locations=(acp_schema.ToolCallLocation(path=abs_path),),
                content=(
                    acp.tool_diff_content(
                        path=abs_path, new_text=new_text, old_text=old_text
                    ),
                ),
            )

        if name == "write_file":
            path = _nonempty_str(args, "path")
            content = _any_str(args, "content")
            if path is None or content is None:
                return None
            abs_path = _resolve_workspace_path(path, ws_dir)
            if abs_path is None:
                return None
            old_text: str | None = None
            target_file = Path(abs_path)
            if target_file.is_file():
                with contextlib.suppress(OSError):
                    old_text = target_file.read_text(encoding="utf-8", errors="replace")
            return ToolCallPresentation(
                kind="edit",
                title=f"Write {path}",
                locations=(acp_schema.ToolCallLocation(path=abs_path),),
                content=(
                    acp.tool_diff_content(
                        path=abs_path, new_text=content, old_text=old_text
                    ),
                ),
            )

        if name in ("read_file", "file_info"):
            path = _nonempty_str(args, "path")
            if path is None:
                return None
            abs_path = _resolve_workspace_path(path, ws_dir)
            if abs_path is None:
                return None
            raw_line = args.get("line_start") or args.get("offset")
            line = (
                raw_line
                if isinstance(raw_line, int) and not isinstance(raw_line, bool)
                else None
            )
            return ToolCallPresentation(
                kind="read",
                title=f"Read {path}",
                locations=(acp_schema.ToolCallLocation(path=abs_path, line=line),),
            )

        if name in ("list_files", "list_directory"):
            raw_path = _nonempty_str(args, "path") or "."
            abs_path = _resolve_workspace_path(raw_path, ws_dir)
            pattern = _nonempty_str(args, "pattern")
            title = f"List {raw_path} ({pattern})" if pattern else f"List {raw_path}"
            return ToolCallPresentation(
                kind="search",
                title=title,
                locations=(
                    (acp_schema.ToolCallLocation(path=abs_path),)
                    if abs_path is not None
                    else ()
                ),
            )

        if name in ("grep", "search_files", "find_files"):
            pattern = _nonempty_str(args, "pattern") or _nonempty_str(args, "glob")
            if pattern is None:
                return None
            raw_path = _nonempty_str(args, "path") or "."
            abs_path = _resolve_workspace_path(raw_path, ws_dir)
            return ToolCallPresentation(
                kind="search",
                title=f"Search {pattern!r} in {raw_path}",
                locations=(
                    (acp_schema.ToolCallLocation(path=abs_path),)
                    if abs_path is not None
                    else ()
                ),
            )

        if name == "create_directory":
            path = _nonempty_str(args, "path")
            if path is None:
                return None
            abs_path = _resolve_workspace_path(path, ws_dir)
            if abs_path is None:
                return None
            return ToolCallPresentation(
                kind="edit",
                title=f"Create directory {path}",
                locations=(acp_schema.ToolCallLocation(path=abs_path),),
            )

        if name in ("shell", "run_command", "start_command"):
            command = _nonempty_str(args, "command")
            if command is None:
                return None
            return ToolCallPresentation(
                kind="execute",
                title=f"Shell: {command}",
            )

        if name in ("check_command", "stop_command"):
            command_id = _nonempty_str(args, "command_id")
            if command_id is None:
                return None
            return ToolCallPresentation(
                kind="execute",
                title=f"{name}: {command_id}",
            )

        if name == "search_knowledge":
            query = _nonempty_str(args, "query")
            if query is None:
                return None
            return ToolCallPresentation(
                kind="search",
                title=f"Search knowledge: {query}",
            )

        if name == "list_knowledge":
            bundle = _nonempty_str(args, "bundle")
            return ToolCallPresentation(
                kind="search",
                title=f"List knowledge ({bundle})" if bundle else "List knowledge",
            )

        if name == "read_knowledge":
            path = _nonempty_str(args, "path")
            if path is None:
                return None
            return ToolCallPresentation(
                kind="read",
                title=f"Read knowledge: {path}",
            )

        if name == "web_fetch":
            url = _nonempty_str(args, "url")
            if url is None:
                return None
            fmt = _nonempty_str(args, "format") or "markdown"
            save_to = _nonempty_str(args, "save_to")
            abs_save = (
                _resolve_workspace_path(save_to, ws_dir)
                if save_to is not None
                else None
            )
            return ToolCallPresentation(
                kind="fetch",
                title=f"Fetch {url} ({fmt})",
                locations=(
                    (acp_schema.ToolCallLocation(path=abs_save),)
                    if abs_save is not None
                    else ()
                ),
            )

        if name == "run_code":
            code = _nonempty_str(args, "code") or _nonempty_str(args, "script")
            return ToolCallPresentation(
                kind="execute",
                title="Run Python code",
                content=(
                    (acp.tool_content(acp.text_block(f"```python\n{code}\n```")),)
                    if code is not None
                    else ()
                ),
            )

        if name == "delegate_task":
            subagent = (
                _nonempty_str(args, "subagent_name")
                or _nonempty_str(args, "agent")
                or "subagent"
            )
            prompt = _nonempty_str(args, "prompt") or _nonempty_str(args, "task") or ""
            short_prompt = (prompt[:60] + "...") if len(prompt) > 63 else prompt
            return ToolCallPresentation(
                kind="think",
                title=(
                    f"Delegate ({subagent}): {short_prompt}"
                    if short_prompt
                    else f"Delegate ({subagent})"
                ),
            )

        return None

    return chain_presenters(_present, default_coding_presenter)


def _resolve_server_workspace(opts: RuntimeOptions, spec: AgentSpec) -> Path:
    """Resolve the server-side workspace directory from CLI `--workspace`, spec `LocalWorkspace`, or `./workdir`."""
    if opts.workspace is not None:
        return _ensure_workspace_dir(opts.workspace)
    for cap in spec.capabilities:
        if cap.name == "LocalWorkspace":
            if isinstance(cap.arguments, str) and cap.arguments:
                return _ensure_workspace_dir(cap.arguments)
            if isinstance(cap.arguments, dict):
                raw_dir = (
                    cap.arguments.get("working_dir")
                    or cap.arguments.get("root_dir")
                    or cap.arguments.get("path")
                )
                if isinstance(raw_dir, str) and raw_dir:
                    return _ensure_workspace_dir(raw_dir)
    return _ensure_workspace_dir(DEFAULT_WORKDIR)


def start_acp_server(
    opts: RuntimeOptions,
    *,
    editor_tools: bool = False,
    offer_all_models: bool = True,
) -> None:
    """Run the agent as an ACP (Agent Client Protocol) server over stdio."""
    _ACP_MODE_CTX.set(True)
    agent, spec = build_agent(opts, defer_model_check=True)
    server_workspace = _resolve_server_workspace(opts, spec)
    active_workspace_ref: list[Path] = [server_workspace]

    def session_config(session: AcpSession) -> AcpSessionConfig[None]:
        target_ws = (
            _ensure_workspace_dir(session.cwd)
            if (
                opts.workspace is None
                and opts.workspace_mode == "client"
                and session.cwd
            )
            else server_workspace
        )
        active_workspace_ref[0] = target_ws
        session_caps: list[AbstractCapability[None]] = [
            _AcpPlanBridgeCapability(session=session),
        ]
        if editor_tools:
            if (fs_cap := acp_filesystem(session)) is not None:
                session_caps.append(fs_cap)
            if (term_cap := acp_terminal(session)) is not None:
                session_caps.append(term_cap)
        return AcpSessionConfig(
            deps=None,
            capabilities=session_caps,
            workspace=LocalWorkspaceBackend(str(target_ws)),
        )

    lemonade_base_url = DEFAULT_LEMONADE_BASE_URL
    lemonade_api_key = "lemonade"
    remember_scope: ApprovalRememberScope = "exact"
    for cap in spec.capabilities:
        if cap.name == "Lemonade" and isinstance(cap.arguments, dict):
            lemonade_base_url = str(cap.arguments.get("base_url", lemonade_base_url))
            lemonade_api_key = str(cap.arguments.get("api_key", lemonade_api_key))
        elif cap.name == "Approvals" and isinstance(cap.arguments, dict):
            raw_scope = cap.arguments.get("remember_scope")
            if raw_scope in ("exact", "tool", "command_prefix"):
                remember_scope = raw_scope

    models_option: str | Sequence[str] | None = None
    if offer_all_models:
        from pydantic_ai.models import known_model_names

        configured = (
            opts.model
            or spec.model
            or os.environ.get("PYDANTIC_AI_MODEL", DEFAULT_MODEL)
        )
        all_models = [m for m in known_model_names() if m != configured]
        models_option = [configured, *all_models]

    def model_resolver(model_id: str) -> Any:
        if model_id == "test":
            return TestModel(call_tools=[])
        if resolved := _resolve_lemonade_model(
            model_id, base_url=lemonade_base_url, api_key=lemonade_api_key
        ):
            return resolved
        return model_id

    run_acp_stdio_sync(
        agent,
        name=spec.name or "microharness",
        session_config=session_config,
        permission_policy=build_acp_permission_policy(remember_scope),
        session_store=InMemorySessionStore(),
        models=models_option,
        model_resolver=model_resolver,
        prompt_capabilities=acp_schema.PromptCapabilities(
            image=False,
            audio=False,
            embedded_context=False,
        ),
        tool_presenter=build_microharness_tool_presenter(
            lambda: active_workspace_ref[0]
        ),
    )


async def _run_query_stream(
    agent: Agent[None, Any], prompt: str, workspace_dir: Path
) -> None:
    """Stream the agent's text output to stdout as it arrives."""
    ws_backend = LocalWorkspaceBackend(str(workspace_dir.resolve()))
    async with agent.run_stream(prompt, workspace=ws_backend) as result:
        async for chunk in result.stream_text(delta=True):
            sys.stdout.write(chunk)
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()


def run_single_query(
    prompt: str, opts: RuntimeOptions, *, stream: bool = False
) -> None:
    """Execute a single prompt against the agent and print the response."""
    _ACP_MODE_CTX.set(False)
    agent, spec = build_agent(opts)
    target_workspace = _resolve_server_workspace(opts, spec)

    try:
        if stream:
            asyncio.run(_run_query_stream(agent, prompt, target_workspace))
        else:
            ws_backend = LocalWorkspaceBackend(str(target_workspace.resolve()))
            result = agent.run_sync(prompt, workspace=ws_backend)
            typer.echo(result.output)
    finally:
        flush_opentelemetry()


app = typer.Typer(
    name="microharness",
    help=(
        "Single-file Pydantic AI Harness application supporting ACP (Agent Client Protocol) "
        "and single-query execution from packaged agent subdirectories."
    ),
    no_args_is_help=False,
    add_completion=False,
)

AgentOption = Annotated[
    str,
    typer.Option(
        "--agent",
        "-a",
        help="Agent name / subdirectory containing agent.yaml, tools/, hooks/, and skills/.",
    ),
]
SpecOption = Annotated[
    Path | None,
    typer.Option(
        "--spec",
        "-s",
        help="Explicit path to a YAML/JSON agent spec file (overrides --agent).",
    ),
]
ModelOption = Annotated[
    str | None,
    typer.Option(
        "--model",
        "-m",
        help="Override the model name in the spec (e.g. 'lemonade:unsloth/Qwen3.8-27B-GGUF', 'test').",
    ),
]
WorkspaceOption = Annotated[
    Path | None,
    typer.Option(
        "--workdir",
        "--workspace",
        "--wd",
        "-wd",
        help="Working directory for the agent's workspace tools.",
    ),
]
WorkspaceModeOption = Annotated[
    Literal["server", "client"],
    typer.Option(
        "--workspace-mode",
        help="ACP workspace resolution: 'server' (default, execute inside Microharness's own workspace) or 'client' (use the ACP client's session cwd).",
    ),
]
InstructionsOption = Annotated[
    str | None,
    typer.Option(
        "--instructions",
        "-i",
        help="Additional instructions appended to the spec's instructions.",
    ),
]
ToolsOption = Annotated[
    list[str] | None,
    typer.Option(
        "--tools",
        "-t",
        help="Directory path(s), `.py` file(s), or `file.py:func` target(s) exposing local Python tools.",
    ),
]
HooksOption = Annotated[
    list[str] | None,
    typer.Option(
        "--hooks",
        "-H",
        help="Python hook file/module/directory path(s) exposing `hooks = Hooks()` or `register_hooks(hooks)`.",
    ),
]
ApprovalsOption = Annotated[
    ApprovalPreset | None,
    typer.Option(
        "--approvals",
        help="Override the Approvals preset in agent.yaml ('write', 'shell', 'all', 'custom', or 'none' for autonomous/YOLO mode).",
    ),
]
OtelEnabledOption = Annotated[
    bool | None,
    typer.Option(
        "--otel/--no-otel",
        help="Enable or disable OpenTelemetry tracing/metrics instrumentation.",
    ),
]
OtelEndpointOption = Annotated[
    str | None,
    typer.Option(
        "--otel-endpoint",
        help="OTLP collector endpoint URL (overrides OTEL_EXPORTER_OTLP_ENDPOINT).",
    ),
]
OtelServiceNameOption = Annotated[
    str | None,
    typer.Option(
        "--otel-service-name",
        help="OpenTelemetry service.name attribute (overrides OTEL_SERVICE_NAME).",
    ),
]
OtelConsoleOption = Annotated[
    bool | None,
    typer.Option(
        "--otel-console/--no-otel-console",
        help="Export OpenTelemetry spans to stderr for local debugging.",
    ),
]


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    agent: AgentOption = DEFAULT_AGENT_NAME,
    spec: SpecOption = None,
    model: ModelOption = None,
    workspace: WorkspaceOption = None,
    workspace_mode: WorkspaceModeOption = "server",
    instructions: InstructionsOption = None,
    tools: ToolsOption = None,
    hooks: HooksOption = None,
    approvals: ApprovalsOption = None,
    otel: OtelEnabledOption = None,
    otel_endpoint: OtelEndpointOption = None,
    otel_service_name: OtelServiceNameOption = None,
    otel_console: OtelConsoleOption = None,
) -> None:
    """Start Microharness. Defaults to ACP stdio mode when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        start_acp_server(
            RuntimeOptions(
                agent_name=agent,
                spec_path=spec,
                model=model,
                workspace=workspace,
                workspace_mode=workspace_mode,
                instructions=instructions,
                tools_paths=tools,
                hooks_modules=hooks,
                approvals=approvals,
                otel_enabled=otel,
                otel_endpoint=otel_endpoint,
                otel_service_name=otel_service_name,
                otel_console=otel_console,
            )
        )


@app.command("acp")
def acp_cmd(
    agent: AgentOption = DEFAULT_AGENT_NAME,
    spec: SpecOption = None,
    model: ModelOption = None,
    workspace: WorkspaceOption = None,
    workspace_mode: WorkspaceModeOption = "server",
    instructions: InstructionsOption = None,
    tools: ToolsOption = None,
    hooks: HooksOption = None,
    approvals: ApprovalsOption = None,
    otel: OtelEnabledOption = None,
    otel_endpoint: OtelEndpointOption = None,
    otel_service_name: OtelServiceNameOption = None,
    otel_console: OtelConsoleOption = None,
    editor_tools: Annotated[
        bool,
        typer.Option(
            "--editor-tools/--no-editor-tools",
            help="Also attach editor-native ACP filesystem/terminal capabilities when offered by the client.",
        ),
    ] = False,
    offer_all_models: Annotated[
        bool,
        typer.Option(
            "--offer-all-models/--no-offer-all-models",
            help="Advertise all known Pydantic AI models in the ACP session model selector.",
        ),
    ] = True,
) -> None:
    """Start the agent in ACP (Agent Client Protocol) mode over stdio."""
    start_acp_server(
        RuntimeOptions(
            agent_name=agent,
            spec_path=spec,
            model=model,
            workspace=workspace,
            workspace_mode=workspace_mode,
            instructions=instructions,
            tools_paths=tools,
            hooks_modules=hooks,
            approvals=approvals,
            otel_enabled=otel,
            otel_endpoint=otel_endpoint,
            otel_service_name=otel_service_name,
            otel_console=otel_console,
        ),
        editor_tools=editor_tools,
        offer_all_models=offer_all_models,
    )


@app.command("query")
def query_cmd(
    prompt: Annotated[
        str, typer.Argument(help="The query/prompt to send to the agent.")
    ],
    agent: AgentOption = DEFAULT_AGENT_NAME,
    spec: SpecOption = None,
    model: ModelOption = None,
    workspace: WorkspaceOption = None,
    instructions: InstructionsOption = None,
    tools: ToolsOption = None,
    hooks: HooksOption = None,
    approvals: ApprovalsOption = None,
    otel: OtelEnabledOption = None,
    otel_endpoint: OtelEndpointOption = None,
    otel_service_name: OtelServiceNameOption = None,
    otel_console: OtelConsoleOption = None,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream/--no-stream",
            help="Stream text deltas to stdout as they are generated.",
        ),
    ] = False,
) -> None:
    """Run a single query against the agent and output the result."""
    run_single_query(
        prompt,
        RuntimeOptions(
            agent_name=agent,
            spec_path=spec,
            model=model,
            workspace=workspace,
            instructions=instructions,
            tools_paths=tools,
            hooks_modules=hooks,
            approvals=approvals,
            otel_enabled=otel,
            otel_endpoint=otel_endpoint,
            otel_service_name=otel_service_name,
            otel_console=otel_console,
        ),
        stream=stream,
    )


@app.command("init-spec")
def init_spec_cmd(
    agent: AgentOption = DEFAULT_AGENT_NAME,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Explicit path where the YAML agent spec will be written (defaults to ./agents/<agent>/agent.yaml).",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f", help="Overwrite the spec file if it already exists."
        ),
    ] = False,
) -> None:
    """Scaffold an agent subdirectory (agents/<agent>/agent.yaml, rules/, knowledge/, tools/, hooks/, skills/) and JSON Schema."""
    raw_agent_path = Path(agent)
    agent_dir = (
        output.parent
        if output is not None
        else (
            raw_agent_path
            if len(raw_agent_path.parts) > 1
            else (DEFAULT_AGENTS_DIR / raw_agent_path)
        )
    )
    spec_file = output if output is not None else (agent_dir / DEFAULT_SPEC_FILENAME)

    if spec_file.exists() and not force:
        typer.secho(
            f"Refusing to overwrite existing '{spec_file}'. Pass --force to overwrite.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    _ensure_workspace_dir(DEFAULT_WORKDIR)
    agent_dir.mkdir(parents=True, exist_ok=True)
    for subdir in (
        DEFAULT_RULES_DIR,
        DEFAULT_KNOWLEDGE_DIR,
        DEFAULT_WEB_DIR,
        DEFAULT_TOOLS_DIR,
        DEFAULT_HOOKS_DIR,
        DEFAULT_SKILLS_DIR,
    ):
        (agent_dir / subdir).mkdir(parents=True, exist_ok=True)

    web_dir = agent_dir / DEFAULT_WEB_DIR
    allowlist_file = web_dir / "allowlist.txt"
    if not allowlist_file.exists():
        allowlist_file.write_text(
            "# WebFetch Domain / URL Allowlist\n"
            "# One domain, wildcard (*.example.com), or URL glob per line.\n"
            "# When this file has no active entries, all non-blocked public domains are allowed.\n",
            encoding="utf-8",
        )
    blocklist_file = web_dir / "blocklist.txt"
    if not blocklist_file.exists():
        blocklist_file.write_text(
            "# WebFetch Domain / URL Blocklist\n"
            "# One domain, wildcard (*.example.com), or URL glob per line.\n"
            "*.internal\n*.local\nmetadata.google.internal\n",
            encoding="utf-8",
        )

    root_schema_file = SCRIPT_DIR / "agent_schema.json"
    try:
        rel_schema_path = os.path.relpath(
            root_schema_file, start=spec_file.resolve().parent
        )
    except ValueError:
        rel_schema_path = str(root_schema_file)

    spec = AgentSpec.from_dict(make_default_spec_dict(agent))
    spec.to_file(
        spec_file,
        schema_path=rel_schema_path,
        custom_capability_types=get_custom_capability_types(),
    )
    typer.echo(
        f"Initialized agent directory: {agent_dir}/ (rules/, knowledge/, web/, tools/, hooks/, skills/)"
    )
    typer.echo(f"Created agent spec: {spec_file}")
    typer.echo(f"Updated JSON schema: {root_schema_file}")


if __name__ == "__main__":
    app()
