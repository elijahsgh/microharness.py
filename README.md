# Microharness

**Microharness** is a single-file Python 3.14+ application powered by [`uv`](https://docs.astral.sh/uv/), [Pydantic AI](https://ai.pydantic.dev/), and **Pydantic AI Harness**.

It uses PEP 723 inline script metadata (`# /// script`) inside [`microharness.py`](./microharness.py) so that a single command—`uv run microharness.py` or `./microharness.py`—automatically provisions dependencies and runs your agent in either **ACP (Agent Client Protocol)** mode or **Single-Query** CLI mode, configured via self-contained agent subdirectories inside [`agents/`](./agents/) (such as [`agents/default/`](./agents/default/)).

---

## TL;DR Quickstart

1. **Install [`uv`](https://docs.astral.sh/uv/) and [`toad`](https://github.com/batrachianai/toad)** (the terminal ACP client UI):
   ```bash
   # Install uv (if not already installed)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install Toad via uv tool (or curl -fsSL batrachian.ai/install | sh)
   uv tool install batrachian-toad
   ```
2. **Launch Microharness in Toad**:
   ```bash
   toad acp "./microharness.py"
   # or:
   toad acp "uv run microharness.py"
   ```

---

## Features

- **Zero-Setup Single-File Application**: Embedded PEP 723 dependencies (`pydantic-ai`, `pydantic-ai-harness[coder,acp,code-mode]`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`, `typer`, `pyyaml`) run directly with `uv run microharness.py` on Python 3.14+.
- **Packaged Agent Subdirectories (`--agent` / `-a`)**: Each agent lives in its own subdirectory (`agents/<agent>/`) bundling its `agent.yaml`, `rules/`, `knowledge/`, local Python `tools/`, `hooks/`, and `skills/`.
- **Dedicated `./workdir` Isolation & Harness Self-Protection**:
  - Defaults `LocalWorkspace` to `./workdir` (automatically created on startup) so all `FileSystem`, `Shell`, `SmartSearch`, `RepoContext`, and `CodeMode` operations start inside `./workdir` instead of the Microharness root directory.
  - Synchronizes `os.getcwd()` and `MICROHARNESS_WORKDIR` to the active workspace before each run (`sync_process_cwd: true`), and blocks `write_file`, `edit_file`, and `create_directory` from modifying `microharness.py`, `agent_schema.json`, `agents/`, or `<agent>/{agent.yaml,rules,hooks,tools,skills,knowledge}` (`protect_harness_files: true`).
- **Dual Execution Modes**:
  - **ACP Mode (Default)**: Serves the agent over `stdio` using the Agent Client Protocol (`run_acp_stdio_sync`) for clients like [Toad](https://github.com/batrachianai/toad) and [Zed](https://zed.dev/docs/ai/external-agents), executing all tools (`Coder`, `Planning`, `Knowledge`, `LocalTools`, `Skills`) natively inside the Microharness server workspace (`--workspace-mode server` by default) while streaming rich `session/update` notifications (`tool_call`, `diff`, and `plan`) to the UI.
  - **Single-Query Mode**: Executes a one-shot prompt from the terminal (`uv run microharness.py query "..."`), with optional token streaming (`--stream`).
- **Declarative YAML Agent Specs**: Configure your model, instructions, settings, and 40+ Pydantic AI / Harness capabilities in `<agent>/agent.yaml` with full IDE autocomplete backed by `agent_schema.json`.
- **First-Class Lemonade AI Server Integration**: Built-in `Lemonade` capability and `lemonade:<model>` prefix configured by default for `unsloth/Qwen3.8-27B-GGUF` on `http://localhost:13305/v1`.
- **Agent Rules (`Rules`)**: Place Markdown files (`<agent>/rules/*.md`) with 4 trigger modes (`always_on`, `glob`, `model_decision`, and `manual` via `@<rule-name>` or `keywords`), hot-reloaded on every run.
- **Open Knowledge Format (`Knowledge`)**: Drop Google Open Knowledge Format (OKF) `.zip` archives (or unpacked directories) into `<agent>/knowledge/` for ultra-fast, shell-free in-memory **SQLite FTS5 BM25 + Porter stemming** search, regex line search, link/backlink graph traversal, and progressive disclosure via `index.md`.
- **Native `httpx2` Web Fetching (`WebFetch`)**: Dedicated `<agent>/web/` directory bundling `allowlist.txt`, `blocklist.txt`, and an extensible `client.py` module (for injecting Bearer tokens, API keys, custom headers, or custom `httpx2.AsyncClient` settings), supporting `markdown`, `raw`, `text`, and `json` formats plus `save_to` into `./workdir`, while blocking `curl`/`wget` bypasses in `shell`.
- **Local Python Tools (`LocalTools`)**: Drop `.py` files into `<agent>/tools/` (or pass `--tools`) to expose either top-level native tools or `CodeMode` (`CODE_MODE = True`) sandbox functions without generating `__pycache__` clutter.
- **Bundled Agent Skills (`Skills`)**: Drop `<skill-name>/SKILL.md` directories (plus optional deferred `tools.py` or `scripts/`) into `<agent>/skills/`.
- **Lifecycle Hooks (Shell & Python)**: Drop `.py` hook modules into `<agent>/hooks/` (hot-reloaded per run in ACP mode) or configure shell/Python hooks in `agent.yaml` or via `--hooks`.
- **OpenTelemetry & Instrumentation (`OpenTelemetry`)**: Full OpenTelemetry tracing and metrics support (`OTLP` HTTP/gRPC and console exporters) configurable via `OTEL_*` environment variables, `agent.yaml`, or `--otel*` CLI flags.

---

## Agent Subdirectory Layout

Each agent is packaged in its own subdirectory inside [`agents/`](./agents/) (defaulting to [`agents/default/`](./agents/default/)):

```text
microharness/
├── microharness.py                        # Single-file uv application
├── agent_schema.json                      # Shared JSON Schema for YAML autocompletion
├── workdir/                               # Default isolated agent working directory (./workdir)
└── agents/                                # Dedicated directory for packaged agents
    └── default/                           # "--agent default" (used when --agent is omitted)
        ├── agent.yaml                     # Declarative AgentSpec for this agent
        ├── rules/                         # Markdown rules (*.md) with 4 trigger modes
        │   └── block-eicar-by-rule.md     # Example manual/keyword rule (--DERPDERPEICARBYRULE--)
        ├── knowledge/                     # Open Knowledge Format (OKF) .zip bundles & directories
        │   └── okf-guide.zip              # Bundled OKF guide (index.md, log.md, concepts/, howto/)
        ├── web/                           # Native httpx2 WebFetch policy & extensible client
        │   ├── allowlist.txt              # Allowed domains/globs (empty = allow all non-blocked)
        │   ├── blocklist.txt              # Blocked domains/globs (enforced on URLs & redirects)
        │   └── client.py                  # User-extensible httpx2 request/auth/response hooks
        ├── tools/                         # Local Python tool modules (*.py)
        │   ├── flip_coin.py               # Native top-level tool example
        │   └── spin_the_wheel.py          # CodeMode (CODE_MODE = True) tool example
        ├── hooks/                         # Local Python hook modules (*.py)
        │   └── block_eicar.py             # Incoming (--DERPDERPEICAR--) & outgoing (--DERPDERPEICAROUTGOING--) hooks
        └── skills/                        # Agent Skills (<skill-name>/SKILL.md)
            └── roll-dice/                 # Example skill with tools.py and scripts/roll_dice.py
```

Relative paths inside `agents/<agent>/agent.yaml` (`./rules`, `./knowledge`, `./tools`, `./hooks`, `./skills`) always resolve relative to that agent's subdirectory, even when Microharness is launched from another working directory or when the runtime shifts `os.getcwd()` into `./workdir` (or an external `--workspace`).

---

## Workspace Isolation (`./workdir`) & Self-Protection

To prevent an autonomous agent from accidentally overwriting its own rules, hooks, skills, or `microharness.py` while using `Coder`, `Shell`, or `CodeMode`:

1. **Dedicated `./workdir` Default (`LocalWorkspace: ./workdir`)**:
   - Microharness defaults `LocalWorkspace` to `./workdir` (automatically created on startup).
   - `FileSystem` (`read_file`, `write_file`, `edit_file`, `list_files`, `create_directory`), `SmartSearch` (`grep`), `RepoContext`, and `Shell` (`shell`, `start_command`) all execute inside `./workdir` by default, rejecting `../` path traversal outside `./workdir` unless `unrestricted_filesystem: true` is set.
2. **Process Working Directory Sync (`sync_process_cwd: true`)**:
   - Before every run, `_HarnessSelfProtectionCapability` sets `MICROHARNESS_WORKDIR` to the resolved workspace path and `os.chdir()`s the Python process into `./workdir` (or `--workspace`), so in-process `CodeMode` (`run_code`) scripts, local tools, and subprocesses without an explicit `cwd` also default to `./workdir`.
   - Microharness captures `INITIAL_CWD` and `_AGENT_DIR_CTX` at startup so live hot-reloading of `<agent>/rules/`, `<agent>/knowledge/`, and `<agent>/hooks/` continues to resolve cleanly against the agent's directory.
3. **Harness Self-Protection (`protect_harness_files: true`)**:
   - Even if `unrestricted_filesystem: true` is enabled or `--workspace .` points to the repository root, `Coder` blocks `write_file`, `edit_file`, and `create_directory` from modifying `microharness.py`, `agent_schema.json`, `<agent>/agent.yaml`, or anything inside `<agent>/{rules,hooks,tools,skills,knowledge}/` with a `ModelRetry` error.

---

## Quick Start

### 1. Run in ACP Mode (Default)

Running `microharness.py` with no subcommand starts the ACP JSON-RPC server over `stdio` using the `default` agent (`./agents/default/agent.yaml`):

```bash
uv run microharness.py
```

Or select a specific agent subdirectory (`--agent` / `-a`):

```bash
uv run microharness.py acp --agent default --editor-tools
```

### 2. Run a Single Query

Execute a single prompt against the `default` agent (or pass `-a <agent_name>`):

```bash
uv run microharness.py query "Summarize the architecture of this repository."
```

Stream tokens as they are generated, select an agent subdirectory, or enable OpenTelemetry console spans:

```bash
uv run microharness.py query "Refactor the parser module" \
  --agent default \
  --stream \
  --otel-console \
  --model lemonade:unsloth/Qwen3.8-27B-GGUF
```

### 3. Scaffold a New Agent Subdirectory (`init-spec`)

Create a new agent subdirectory (`agents/<agent>/agent.yaml`, `rules/`, `knowledge/`, `tools/`, `hooks/`, `skills/`) and update the root `agent_schema.json`:

```bash
uv run microharness.py init-spec --agent researcher
# Or regenerate the default agent spec:
uv run microharness.py init-spec --agent default --force
```

---

## CLI Reference

```text
Usage: microharness.py [OPTIONS] COMMAND [ARGS]...

Options:
  -a, --agent <name>                      Agent name / subdirectory containing agent.yaml, tools/, hooks/, and skills/ [default: default].
  -s, --spec <path>                       Explicit path to a YAML/JSON agent spec file (overrides --agent).
  -m, --model <str>                       Override the model name in the spec (e.g. 'lemonade:unsloth/Qwen3.8-27B-GGUF', 'test').
  -w, --workspace <path>                  Working directory for the agent's workspace tools.
  --workspace-mode [server|client]        ACP workspace resolution: 'server' (default, execute inside Microharness's own workspace) or 'client' (use the ACP client's session cwd).
  -i, --instructions <str>                Additional instructions appended to the spec's instructions.
  -t, --tools <str>                       Directory path(s), `.py` file(s), or `file.py:func` target(s) exposing local Python tools.
  -H, --hooks <str>                       Python hook file/module/directory path(s) exposing `hooks = Hooks()` or `register_hooks(hooks)`.
  --otel / --no-otel                      Enable or disable OpenTelemetry tracing/metrics instrumentation.
  --otel-endpoint <url>                   OTLP collector endpoint URL (overrides OTEL_EXPORTER_OTLP_ENDPOINT).
  --otel-service-name <str>               OpenTelemetry service.name attribute (overrides OTEL_SERVICE_NAME).
  --otel-console / --no-otel-console      Export OpenTelemetry spans to stderr for local debugging.
  --help                                  Show help message and exit.

Commands:
  acp        Start the agent in ACP (Agent Client Protocol) mode over stdio.
  query      Run a single query against the agent and output the result.
  init-spec  Scaffold an agent subdirectory (<agent>/agent.yaml, tools/, hooks/, skills/) and JSON Schema.
```

---

## Connecting Toad & ACP Clients (Server-Side Execution + Full UI Observability)

Microharness is architected so that **all tool execution (`read_file`, `write_file`, `edit_file`, `list_files`, `grep`, `shell`, `Planning`, `Knowledge`, `LocalTools`, `Skills`) happens 100% natively inside the Microharness server environment**, while streaming rich `session/update` telemetry (`tool_call`, `diff`, and `plan` checklists) back to the ACP client UI:

1. **Zero Local Client Pollution**:
   - During the ACP `initialize` handshake, Microharness advertises `promptCapabilities.embeddedContext = false` and never invokes client-side `fs/read_text_file`, `fs/write_text_file`, or `terminal/create` RPCs (unless you explicitly opt in with `--editor-tools`).
   - By default (`--workspace-mode server`), Microharness roots its `LocalWorkspaceBackend` in its own server workspace (`--workspace`, `LocalWorkspace` in `agent.yaml`, or Microharness's working directory) rather than the client's `session.cwd`.
2. **Full Toad UI Observability (`tool_call`, `diff`, and `plan`)**:
   - `CodeModeSpec` keeps `Coder` and `Planning` tools (`edit_file`, `write_file`, `read_file`, `list_files`, `grep`, `shell`, `write_plan`, `update_task_status`, etc.) as top-level native tools (`keep_coder_tools_native: true`) so every call emits a top-level ACP `session/update` notification instead of being hidden inside `run_code`.
   - `build_microharness_tool_presenter` attaches `FileEditToolCallContent` inline unified diffs (capturing before/after text from the server workspace on both `edit_file` and `write_file`), `ToolCallLocation` links, and `ToolKind` badges (`read`, `edit`, `search`, `execute`, `think`).
   - `_AcpPlanBridgeCapability` bridges `Planning` tool calls (`write_plan`, `add_task`, `update_task_status`, `update_task_statuses`, `remove_task`, `add_subtask`) directly into ACP `AgentPlanUpdate` (`session_update="plan"`) notifications so Toad renders live task checklists as the agent works, and attaches formatted `ContentToolCallContent` (`acp.tool_content(acp.text_block(...))`) blocks to completed non-diff tool calls (`shell`, `run_code`, `grep`, `search_knowledge`, `read_knowledge`, `flip_coin`, `roll_dice`, etc.).

### Connecting Toad

```bash
toad acp "uv run /path/to/microharness/microharness.py --agent default"
```

To point Microharness at a specific isolated server workspace directory (e.g., a container checkout or git worktree):

```bash
toad acp "uv run /path/to/microharness/microharness.py --agent default --workspace /path/to/server/worktree"
```

### Connecting Zed

Add Microharness to your Zed `settings.json` under `agent_servers` (pass `--workspace-mode client` if you want Microharness to use Zed's opened project `session.cwd`, or omit it to stay in Microharness's server workspace):

```json
{
  "agent_servers": {
    "Microharness": {
      "type": "custom",
      "command": "uv",
      "args": [
        "run",
        "/path/to/microharness/microharness.py",
        "--agent",
        "default"
      ]
    }
  }
}
```

Pass `--editor-tools` to `acp` only if you want to route file and terminal operations through the client's editor buffers/terminal RPCs.

---

## Agent Specification (`agents/<agent>/agent.yaml`)

By default, Microharness loads [`./agents/default/agent.yaml`](./agents/default/agent.yaml):

```yaml
# yaml-language-server: $schema=../../agent_schema.json
model: lemonade:unsloth/Qwen3.8-27B-GGUF
name: default
instructions: You are an autonomous coding and research assistant running inside Microharness.
capabilities:
  - Lemonade:
      base_url: http://localhost:13305/v1
  - LocalWorkspace: ./workdir
  - Rules: ./rules
  - Knowledge: ./knowledge
  - WebFetch: ./web
  - LocalTools: ./tools
  - Skills: ./skills
  - Hooks:
      modules:
        - ./hooks
  - OpenTelemetry:
      enabled: true
      protocol: http/protobuf
      service_name: default
      include_content: true
      include_binary_content: true
      include_model_request_parameters: true
      version: 5
      instrument_all: true
  - Coder:
      sub_agents: true
      caching: true
  - WebFetch:
      backend: custom
      path: ./web
      block_shell_fetch: true
  - Planning
  - CodeMode
```

### Lemonade AI Server Configuration

The `Lemonade` capability routes any `lemonade:<model_id>` (or bare model ID without a provider prefix, such as `unsloth/Qwen3.8-27B-GGUF`) to your local Lemonade AI Server's OpenAI-compatible Chat Completions endpoint (`http://localhost:13305/v1` by default, or `LEMONADE_BASE_URL` in the environment).

You can also use any standard Pydantic AI model string (`anthropic:claude-sonnet-4-6`, `openai:gpt-5`, `google-gla:gemini-2.5-pro`, `ollama:qwen3`, etc.).

### Available Capabilities in `agent.yaml`

All built-in Pydantic AI capabilities and Pydantic AI Harness capabilities are registered and available in `agent.yaml`:

- **Combined Harnesses**: `Coder`, `Researcher`, `CodeMode`
- **Local Extensions & Hooks**: `LocalTools`, `Hooks`, `OpenTelemetry`, `Lemonade`
- **Workspaces & Execution**: `LocalWorkspace`, `FileSystem`, `Shell`, `SmartSearch`, `DockerSandbox`, `SSHWorkspace`
- **Reasoning, Planning & Delegation**: `Thinking`, `Planning`, `SubAgents`, `Advisor`, `BackgroundTools`
- **Context & Compaction**: `Caching`, `ClearToolResults`, `SlidingWindowCompaction`, `SummarizingCompaction`, `RelevanceCompaction`, `TieredCompaction`, `FallbackCompaction`, `ClampOversizedMessages`, `DeduplicateFileReads`, `WarnNearLimits`, `WarnOnCacheBusts`
- **Memory & Repo Context**: `Memory`, `ConversationSearch`, `RepoContext`, `Skills`, `PydanticAIDocs`
- **Web & Integrations**: `WebSearch`, `WebFetch`, `MCP`, `GitHub`, `Linear`, `Slack`, `Notion`, `PostHog`, `Macroscope`, `LocalStack`
- **Safety & Observability**: `SpendLimits`, `ToolCallJudge`, `ApprovalReviewer`, `StepPersistence`, `Instrumentation`

---

## Native `httpx2` Web Fetching (`WebFetch`)

Microharness supports four configurable policies for web fetching in `agents/<agent>/agent.yaml`:

| Policy | `agent.yaml` Configuration | Behavior |
| :--- | :--- | :--- |
| **1. Custom `httpx2` + Block Shell Fetch (Default)** | `WebFetch: {backend: custom, path: ./web, block_shell_fetch: true}` | Uses `httpx2` with `agents/<agent>/web/{allowlist.txt,blocklist.txt,client.py}`, supports `format="markdown" \| "raw" \| "text" \| "json"` and `save_to`, and blocks `curl`/`wget` in `shell`. |
| **2. Custom `httpx2` + Allow Shell Fetch** | `WebFetch: {backend: custom, path: ./web, block_shell_fetch: false}` | Provides the custom `web_fetch` tool with allowlist/blocklist and `client.py` hooks while still allowing `curl`/`wget` in `shell`. |
| **3. Pydantic AI Built-in `WebFetch`** | `WebFetch: {backend: pydantic_ai, path: ./web, block_shell_fetch: false}` | Delegates to `pydantic_ai.capabilities.WebFetch` (local `httpx2` Markdown converter or provider-native server-side fetch when `native: true`), automatically seeding `allowed_domains` / `blocked_domains` from `agents/<agent>/web/`. |
| **4. Shell-Only (`curl` / `wget`)** | Omit `WebFetch` or set `WebFetch: {enabled: false}` | No `web_fetch` tool is registered; the agent uses `shell` (`curl`, `wget`, etc.) directly. |

### Custom `agents/<agent>/web/` Bundle

When `backend: custom` (the default), `WebFetch` reads its configuration from `agents/<agent>/web/`:

- **`allowlist.txt`**: Optional newline-separated domain allowlist (`example.com`, `*.github.com`). Empty or comments-only allows all domains not in `blocklist.txt`.
- **`blocklist.txt`**: Newline-separated domain blocklist evaluated before the allowlist on both the initial URL and every redirect hop.
- **`client.py`**: User-extensible Python hooks for custom `httpx2` client creation, per-domain bearer tokens/headers, and response post-processing:
  - `create_client(*, timeout: float = 30.0) -> httpx2.AsyncClient`
  - `prepare_headers(url: str, headers: dict[str, str]) -> dict[str, str]`
  - `transform_response(url: str, result: dict[str, Any]) -> dict[str, Any]`

---

## Local Python Tools (`LocalTools`)

The `LocalTools` capability lets you extend an agent by dropping `.py` files into its `<agent>/tools/` directory (or any directory/file configured in `agent.yaml` or passed via `--tools` / `-t`).

- Missing default `./tools` directories are silently ignored, so `- LocalTools: ./tools` can stay in `<agent>/agent.yaml` even when `tools/` is empty.
- Dynamic loading suppresses bytecode generation (`sys.dont_write_bytecode`), so loading local tools or hooks never creates `__pycache__` folders in your workspace.
- Any top-level module docstring (or `INSTRUCTIONS = "..."` constant) inside a tool `.py` file is automatically appended to the agent's instructions (unless `include_module_docstrings: false`).

### 3 Ways to Author Tool `.py` Files

#### 1. Zero-Boilerplate Public Functions (Default)
Any top-level public function (`not name.startswith("_")`) defined directly in a `.py` file inside `<agent>/tools/` is automatically registered as an agent tool. Sync (`def`) and async (`async def`) functions—as well as functions accepting `ctx: RunContext` as their first parameter—are supported automatically:

```python
# agents/default/tools/system_info.py
"""System diagnostic tools for checking host resources."""

import platform
import shutil
from pydantic_ai import RunContext


def disk_usage(path: str = ".") -> dict[str, int]:
    """Return total, used, and free disk space in bytes for the given path."""
    usage = shutil.disk_usage(path)
    return {"total": usage.total, "used": usage.used, "free": usage.free}


async def os_summary(ctx: RunContext) -> str:
    """Return a summary of the current operating system and Python runtime."""
    return f"{platform.system()} {platform.release()} (Python {platform.python_version()} at step {ctx.run_step})"
```

#### 2. Explicit `__tools__` / `TOOLS` List
If your file defines helper functions that shouldn't be exposed or needs per-tool settings (`requires_approval=True`, `timeout`, `max_retries`), define `__tools__` (or `TOOLS`):

```python
# agents/default/tools/deploy.py
from pydantic_ai import Tool


def _internal_helper(env: str) -> str:
    return env.strip().lower()


def check_status(env: str = "staging") -> str:
    """Check deployment status for an environment."""
    return f"{_internal_helper(env)}: healthy"


def trigger_deploy(env: str) -> str:
    """Trigger a deployment to the target environment."""
    return f"Deployed to {_internal_helper(env)}"


__tools__ = [
    check_status,
    Tool(trigger_deploy, requires_approval=True, timeout=120.0),
]
```

#### 3. Native `FunctionToolset` or `register_tools(toolset)`
Expose `toolset = FunctionToolset(...)` or `def register_tools(toolset: FunctionToolset) -> None:` for full programmatic control:

```python
# agents/default/tools/custom_toolset.py
from pydantic_ai import FunctionToolset

toolset = FunctionToolset()


@toolset.tool
def reverse_text(text: str) -> str:
    """Reverse the input string."""
    return text[::-1]
```

### Configuring `LocalTools` in `agent.yaml`

Use either the directory shorthand or the full configuration object:

```yaml
capabilities:
  # Shorthand (resolves relative to <agent>/)
  - LocalTools: ./tools

  # Or full configuration
  - LocalTools:
      paths:
        - ./tools
        - ./scripts/db_helpers.py:run_readonly_query
      pattern: "*.py"
      recursive: true
      prefix_with_module: false        # Set true to prefix tool names as `<module>__<func>`
      requires_approval: false
      timeout: 30.0
      max_retries: 2
      include_module_docstrings: true
```

---

## Bundled Agent Skills (`Skills`)

Each agent subdirectory includes a `skills/` folder (`- Skills: ./skills` in `<agent>/agent.yaml`). Every skill is a self-contained subfolder under `<agent>/skills/<skill-name>/` containing a `SKILL.md` file and optional deferred Python tools (`tools.py` or `tools/*.py`), scripts, or reference docs:

```text
agents/default/
└── skills/
    └── data-inspector/
        ├── SKILL.md          # Required: YAML frontmatter (with description) + Markdown instructions
        ├── tools.py          # Optional: Deferred Python tools (or a tools/*.py directory)
        ├── scripts/          # Optional: Helper scripts the agent can execute
        └── references/       # Optional: Reference docs or templates
```

### 1. `SKILL.md` & Required `description`

Each `SKILL.md` must begin with YAML frontmatter containing a non-empty `description` (and an optional `name` matching the folder name, using lowercase letters, numbers, and single hyphens `[a-z0-9-]+`):

```markdown
<!-- agents/default/skills/data-inspector/SKILL.md -->
---
name: data-inspector
description: >
  Inspect, summarize, and validate CSV or Parquet files. Load this skill
  whenever the user asks to profile a dataset or check data quality.
---

# Data Inspector Workflow
1. Call `inspect_rows` to check column headers and sample values.
2. Consult `references/quality_rules.md` in this skill directory for validation rules.
```

### 2. Optional Deferred Skill Tools (`tools.py` or `tools/*.py`)

You can bundle Python tools directly inside a skill folder using the same authoring conventions as `LocalTools` (plain functions, `RunContext`, `@tool`, `TOOLS`, or `register_tools(toolset)`):

```python
# agents/default/skills/data-inspector/tools.py
def inspect_rows(path: str, limit: int = 5) -> list[str]:
    """Read the first N lines of a data file."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for _, line in zip(range(limit), f)]
```

### 3. Two-Stage Deferred Loading at Runtime

1. **At Run Start**: Neither the `SKILL.md` body nor the skill's Python tools consume context tokens or clutter the initial tool list. Instead, the model sees a compact `<context id="capability:deferred_capabilities">` catalog of **`<skill-name>: <description>`** entries and the `load_capability` tool.
2. **When Activated (`load_capability(id="data-inspector")`)**:
   - The full `SKILL.md` instructions are returned to the agent, including the resolved absolute directory header (`Skill directory: /abs/path/to/agents/default/skills/data-inspector. Relative paths in this skill resolve against it.`) so the agent can locate any bundled `scripts/` or `references/`.
   - Any Python tools in that skill's `tools.py` or `tools/*.py` are unlocked simultaneously (along with a `## Bundled Skill Tools` signature summary).

### `Skills` Configuration in `agent.yaml`

```yaml
capabilities:
  # Shorthand (resolves relative to <agent>/)
  - Skills: ./skills

  # Or full configuration
  - Skills:
      directories:
        - ./skills
      include: null                   # Or list of skill names to include
      exclude: null                   # Or list of skill names to exclude
      prefix_tools_with_skill: false  # Set true to prefix skill tool names as `<skill_name>__<func>`
```

---

## Agent Rules (`Rules`)

Each agent subdirectory includes a `rules/` folder (`- Rules: ./rules` in `<agent>/agent.yaml`) where you can place Markdown rule files (`<agent>/rules/*.md`) with optional YAML frontmatter.

### Supported Trigger Modes

| Trigger Mode | Frontmatter | Behavior |
| :--- | :--- | :--- |
| **`always_on`** *(Default)* | `trigger: always_on` (or no frontmatter) | Injected unconditionally into the agent's static system prompt prefix at run start. |
| **`glob`** | `trigger: glob` + `globs: ["**/*.py", ...]` | Automatically injected once per run when the agent reads, writes, edits, or lists a file matching any glob pattern. |
| **`model_decision`** | `trigger: model_decision` + `description: "..."` | Registered as a deferred capability (`id="rule-<name>"`). The agent sees the description in its catalog and loads the rule via `load_capability` when relevant. |
| **`manual`** | `trigger: manual` (+ optional `keywords: [...]`) | Activated when the user mentions `@<rule-name>` (or any configured keyword) in their prompt. |

### Example Rule (`agents/default/rules/python-style.md`)

```markdown
---
name: python-style
trigger: glob
description: Python 3.14+ typing and style conventions
globs:
  - "**/*.py"
---

# Python Style Guidelines
- Prefer modern union syntax (`X | None`) and built-in generic types (`list[str]`, `dict[str, Any]`).
- Keep functions small, cohesive, and explicitly typed.
```

---

## Open Knowledge Format (`Knowledge`)

Each agent subdirectory includes a `knowledge/` folder (`- Knowledge: ./knowledge` in `<agent>/agent.yaml`) where you can drop **Google Open Knowledge Format (OKF)** `.zip` archives (such as [`agents/default/knowledge/okf-guide.zip`](./agents/default/knowledge/okf-guide.zip)) or unpacked OKF bundle directories.

### How It Works (Shell-Free In-Memory Search)

Instead of requiring shell access (`grep` / `ripgrep`) or extracting `.zip` archives to disk:
1. **Direct In-Memory ZIP Indexing (`zipfile` + SQLite FTS5)**: Reads `.md` files directly from `.zip` archives into an in-memory SQLite FTS5 virtual table (`porter unicode61` tokenizer) with mtime-based hot-reloading.
2. **Progressive Disclosure via `index.md`**: At run start, `Knowledge` injects a compact **Knowledge Catalog** into the agent's system prompt listing each discovered bundle, entry count, available `type`s/tags, and its root `index.md` overview.
3. **Link & Backlink Knowledge Graph**: Automatically parses relative Markdown links (`[label](./target.md)`) and computes **incoming backlinks** for every concept document.
4. **Three Native Tools**:
   - `search_knowledge(query, bundle=..., doc_type=..., tag=..., path_prefix=..., mode="auto" | "fts" | "regex" | "literal", limit=10)`: Ranked BM25 + field-boosted full-text search with Porter stemming, contextual `snippet()` highlights, and regex/literal line-number fallback.
   - `read_knowledge(path, bundle=..., start_line=..., end_line=...)`: Reads a concept file along with its OKF frontmatter (`type`, `title`, `description`, `resource`, `tags`, `timestamp`), `outgoing_links`, and `backlinks`.
   - `list_knowledge(bundle=..., doc_type=..., tag=...)`: Lists discovered bundles and filters entries by `type` or `tag`.

### Configuring `Knowledge` in `agent.yaml`

```yaml
capabilities:
  # Shorthand (resolves relative to <agent>/)
  - Knowledge: ./knowledge

  # Or full configuration
  - Knowledge:
      paths:
        - ./knowledge
      include_bundles: null
      exclude_bundles: null
      inject_catalog: true
      include_index_overview: true
      code_mode: false
```

---

## Native `httpx2` Web Fetching (`WebFetch`)

Each agent subdirectory includes a `web/` folder (`- WebFetch: ./web` in `<agent>/agent.yaml`) that configures native `httpx2` URL fetching, domain/URL allowlists and blocklists, user-extensible request/auth hooks, and a shell guard that prevents the agent from bypassing policy via `curl` or `wget`:

```text
agents/default/web/
├── allowlist.txt      # Allowed domains, *.subdomains, or URL globs (empty = allow all non-blocked)
├── blocklist.txt      # Blocked domains, *.subdomains, or URL globs (always takes precedence)
└── client.py          # User-extensible httpx2 hooks (prepare_headers, create_client, transform_response)
```

### 1. Native `web_fetch` Tool (`markdown`, `raw`, `text`, `json` + `save_to`)
- **`format="markdown"` *(Default)***: Converts HTML pages into clean Markdown while stripping `<script>` and `<style>` tags.
- **`format="raw"`**: Returns the raw HTML/text body (preserving `<script>` tags, embedded faceplate JSON, etc. without needing `curl` or `wget`).
- **`format="text"`**: Extracts visible text from HTML.
- **`format="json"`**: Parses and pretty-prints JSON responses.
- **`save_to="<relative-path>"`**: Saves the full un-truncated response directly to a file inside `./workdir` so the agent can inspect large payloads with `read_file`, `grep`, or Python.

### 2. Domain & URL Allowlist / Blocklist (`allowlist.txt` & `blocklist.txt`)
- Both files are hot-reloaded on modification and enforced on **both the initial URL and every HTTP redirect hop** (alongside SSRF protection against private/loopback/cloud-metadata IPs).
- `blocklist.txt` always takes precedence over `allowlist.txt`.

### 3. Extending Web Interactions in `agents/<agent>/web/client.py`
Edit [`agents/default/web/client.py`](./agents/default/web/client.py) to inject Bearer tokens, API keys, custom headers, proxies, or response transformations (hot-reloaded automatically):

```python
# agents/default/web/client.py
import os
from urllib.parse import urlparse
import httpx2
from pydantic_ai._http import create_async_httpx2_client

DEFAULT_USER_AGENT = "Microharness-WebFetch/1.0 (+httpx2)"


def prepare_headers(url: str, headers: dict[str, str]) -> dict[str, str]:
    merged = {"User-Agent": DEFAULT_USER_AGENT, **headers}
    host = (urlparse(url).hostname or "").lower()
    if host in ("api.github.com", "github.com") and (
        gh_token := os.environ.get("GITHUB_TOKEN")
    ):
        merged.setdefault("Authorization", f"Bearer {gh_token}")
    return merged


def create_client(*, timeout: float = 30.0) -> httpx2.AsyncClient:
    return create_async_httpx2_client(timeout=timeout)
```

### 4. Shell Bypass Prevention (`block_shell_fetch: true`)
When `block_shell_fetch: true` (the default), any attempt by the agent to run `curl`, `wget`, `aria2c`, `httpie`, `lynx`, `w3m`, or inline Python HTTP clients inside the `shell` tool is intercepted before execution with a `ModelRetry` instructing the model to use `web_fetch` instead.

---

## Hooks

Microharness provides a declarative `Hooks` capability (`HooksSpec`) that wraps Pydantic AI's [`Hooks`](https://ai.pydantic.dev/core-concepts/hooks/) system. You can define **shell command hooks**, **Python function hooks**, or **Python hook modules** across any lifecycle stage.

### Supported Hook Events

| Lifecycle Stage | Events |
| :--- | :--- |
| **Agent Run** | `before_run`, `after_run` |
| **Model Request** | `before_model_request`, `after_model_request` |
| **Tool Validation** | `before_tool_validate`, `after_tool_validate` |
| **Tool Execution** | `before_tool_execute`, `after_tool_execute` |
| **Output Validation & Processing** | `before_output_validate`, `after_output_validate`, `before_output_process`, `after_output_process` |

### Example 1: Shell Command Hooks in `agent.yaml`

Shell hooks receive context via both **`MICROHARNESS_*` environment variables** and a **JSON payload on `stdin`**:

- `MICROHARNESS_HOOK_EVENT`: Name of the lifecycle event (e.g., `before_tool_execute`).
- `MICROHARNESS_RUN_STEP`: Current step number in the agent run.
- `MICROHARNESS_TOOL_NAME`: Name of the tool being called (for tool hooks).
- `MICROHARNESS_TOOL_CALL_ID`: Tool call identifier.
- `MICROHARNESS_TOOL_ARGS_JSON`: JSON-serialized tool arguments.

```yaml
  - Hooks:
      before_run:
        - 'echo "[audit] Starting run at $(date -Iseconds)" >&2'

      before_tool_execute:
        - command: |
            if echo "$MICROHARNESS_TOOL_ARGS_JSON" | grep -qE "rm -rf /|git push --force"; then
              echo "Blocked dangerous command in $MICROHARNESS_TOOL_NAME" >&2
              exit 2
            fi
          tools: ["shell"]
          timeout: 5.0
          on_failure: retry

      after_tool_execute:
        - command: 'echo "[audit] Modified file via $MICROHARNESS_TOOL_NAME" >&2'
          tools: ["write_file", "edit_file"]
          on_failure: ignore
```

#### Shell Hook Input/Output & Exit Code Semantics

1. **Exit Code `0` (Success)**:
   - Any output written to `stderr` is logged to the terminal/stderr stream (`[hook:<event>] ...`).
   - If `stdout` is a JSON object:
     - In `before_tool_validate`, `after_tool_validate`, or `before_tool_execute`: returning `{"args": {...}}` replaces the tool call arguments passed to the tool.
     - In `after_tool_execute`: returning `{"result": ...}` replaces the tool result returned to the model.
     - In output hooks: returning `{"output": ...}` replaces the output value.
2. **Exit Code `2` or `on_failure: retry`**:
   - Raises `pydantic_ai.ModelRetry` with the hook's `stderr` (or `stdout`) message so the model sees why its action was rejected and can self-correct on the next step.
3. **Non-Zero Exit Code with `on_failure: abort`**:
   - Immediately aborts the agent run with a `RuntimeError`.
4. **Non-Zero Exit Code with `on_failure: ignore`**:
   - Logs a warning to `stderr` and continues the run.

### Example 2: Rewriting Tool Arguments with a Shell Hook

```yaml
  - Hooks:
      before_tool_execute:
        - command: |
            python3 -c '
            import json, sys
            data = json.load(sys.stdin)
            args = data.get("args", {})
            if args.get("timeout", 270) > 60:
                args["timeout"] = 60
            print(json.dumps({"args": args}))
            '
          tools: ["shell"]
```

### Example 3: Python Hooks (`modules` and `python:` Callables)

Any `.py` file placed inside `<agent>/hooks/` (e.g. `agents/default/hooks/my_hooks.py`) is automatically loaded when `modules: [./hooks]` is configured in `<agent>/agent.yaml`:

```python
# agents/default/hooks/my_hooks.py
import sys
from typing import Any
from pydantic_ai import ModelRequestContext, ModelRetry, RunContext, ToolDefinition
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ToolCallPart

hooks = Hooks()


@hooks.on.before_model_request
async def log_model_request(
    ctx: RunContext[Any], request_context: ModelRequestContext
) -> ModelRequestContext:
    sys.stderr.write(
        f"[py-hook] Step {ctx.run_step}: sending {len(request_context.messages)} messages\n"
    )
    return request_context


@hooks.on.before_tool_execute(tools=["write_file", "edit_file"], timeout=5.0)
async def protect_lockfiles(
    ctx: RunContext[Any],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
) -> dict[str, Any]:
    path = str(args.get("path", ""))
    if path.endswith("uv.lock"):
        raise ModelRetry("Do not edit uv.lock directly; use `uv` commands instead.")
    return args


async def audit_after_tool(
    ctx: RunContext[Any],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
    result: Any,
) -> Any:
    sys.stderr.write(f"[py-hook] Finished tool {call.tool_name}\n")
    return result
```

Reference the `hooks/` directory or specific hook functions in `<agent>/agent.yaml`:

```yaml
  - Hooks:
      modules:
        - ./hooks
      after_tool_execute:
        - python: ./hooks/my_hooks.py:audit_after_tool
```

Or attach hook files/directories at runtime from the CLI:

```bash
uv run microharness.py query "Inspect the project files" --hooks ./agents/default/hooks
uv run microharness.py acp --hooks ./agents/default/hooks
```

### Bundled EICAR Interception Examples (`agents/default/`)

The `agents/default/` agent includes three test signatures demonstrating all three ways to intercept or terminate a query across both `query` and streaming `acp` modes:

| Signature | Mechanism | File | Behavior |
| :--- | :--- | :--- | :--- |
| **`--DERPDERPEICAR--`** | **Incoming Hook** (`@hooks.on.model_request`) | [`agents/default/hooks/block_eicar.py`](./agents/default/hooks/block_eicar.py) | Intercepts the prompt **before** the model is called and immediately returns a refusal response. |
| **`--DERPDERPEICAROUTGOING--`** | **Outgoing Hook** (`@hooks.on.run_event_stream`, `after_model_request`, `after_tool_execute`, `after_output_process`) | [`agents/default/hooks/block_eicar.py`](./agents/default/hooks/block_eicar.py) | Holds back partial prefix matches across streaming deltas and blocks any model response, tool output, or stream chunk containing the signature from reaching the user. |
| **`--DERPDERPEICARBYRULE--`** | **Rule** (`trigger: manual`, `keywords`) | [`agents/default/rules/block-eicar-by-rule.md`](./agents/default/rules/block-eicar-by-rule.md) | Dynamically injects a critical security rule into the model request instructing the LLM to refuse the query. |

```bash
# 1. Test incoming hook (model is never called)
uv run microharness.py query "Hello --DERPDERPEICAR-- world" --no-otel

# 2. Test outgoing hook (blocked on return/stream)
uv run microharness.py query "Repeat --DERPDERPEICAROUTGOING-- back to me" --no-otel

# 3. Test rule-based refusal (injected via Rules capability)
uv run microharness.py query "Please process --DERPDERPEICARBYRULE--" --no-otel
```

---

## OpenTelemetry & Instrumentation

Microharness integrates OpenTelemetry (`opentelemetry-sdk` + `opentelemetry-exporter-otlp`) with Pydantic AI's [`Instrumentation`](https://ai.pydantic.dev/logfire/) capability and `Agent.instrument_all()`, emitting OpenTelemetry GenAI semantic convention spans and token/cost metrics for agent runs, model requests, tool calls, and sub-agents.

> [!CAUTION]
> **Full Message & Tool Content Logging Is Enabled by Default (`include_content=True`)**
>
> Unlike standard OpenTelemetry GenAI contrib instrumentors (which omit prompt and completion bodies unless `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` is set), **Pydantic AI's `InstrumentationSettings` defaults to `include_content=True` and `include_binary_content=True` on the client.**
>
> **What this means in practice:**
> - **Everything in the AI context is transmitted to your OTel server:** Full user prompts, system instructions, model responses, internal `<thinking>` blocks, tool call arguments, and tool execution outputs (`read_file`, `shell`, `grep`, custom tools) are serialized into span attributes (`gen_ai.input.messages`, `gen_ai.output.messages`, `pydantic_ai.all_messages`, `gen_ai.system_instructions`, and tool result attributes).
> - **Credential & Secret Exposure Risk:** Because an autonomous coding agent routinely reads configuration files, environment variables, `.env` files, command outputs, and proprietary source code, **any API keys, tokens, passwords, database records, or PII that enter the agent's context window or tool results will be sent to and stored in your OpenTelemetry collector/backend.**
> - **Binary Payloads:** When `include_binary_content=True`, any images, PDFs, or audio files inspected by the agent are base64-encoded directly into the exported spans.

### Disabling Message & Tool Content Logging

Content redaction/omission happens **at the client (inside Microharness)** before spans leave the process. If you are exporting telemetry to a shared, cloud-hosted, or non-local OpenTelemetry backend—or working in repositories that handle secrets or sensitive data—you should disable content capture so only structural telemetry (span hierarchy, model/tool names, latency, token usage, and cache hit metrics) is exported:

#### Option A: Disable via Environment Variables
```bash
export MICROHARNESS_OTEL_INCLUDE_CONTENT="false"
export MICROHARNESS_OTEL_INCLUDE_BINARY_CONTENT="false"
# Optional: also omit tool JSON schemas and model request parameters
export MICROHARNESS_OTEL_INCLUDE_PARAMETERS="false"
```

#### Option B: Disable in `agent.yaml`
```yaml
capabilities:
  - OpenTelemetry:
      endpoint: http://localhost:4318
      include_content: false
      include_binary_content: false
      include_model_request_parameters: false
```

*( Or if using Pydantic AI's built-in `Instrumentation` capability directly in `agent.yaml`:)*
```yaml
capabilities:
  - Instrumentation:
      include_content: false
      include_binary_content: false
      include_model_request_parameters: false
```

### Environment Variables

OpenTelemetry activates automatically whenever `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`), `MICROHARNESS_OTEL_ENABLED=1`, `MICROHARNESS_OTEL_CONSOLE=1`, or `--otel` is set:

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `None` | Base OTLP collector URL (e.g. `http://localhost:4318` for HTTP or `http://localhost:4317` for gRPC). |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | `None` | Trace-specific endpoint override (e.g. `http://localhost:4318/v1/traces`). |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | `None` | Metrics-specific endpoint override (e.g. `http://localhost:4318/v1/metrics`). |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` | Transport protocol: `http/protobuf` (or `http`) vs `grpc`. |
| `OTEL_EXPORTER_OTLP_HEADERS` | `None` | Comma-separated `key=value` headers (e.g. `Authorization=Bearer <token>`). |
| `OTEL_SERVICE_NAME` | `microharness` | OpenTelemetry `service.name` resource attribute (defaults to `spec.name`). |
| `OTEL_RESOURCE_ATTRIBUTES` | `None` | Comma-separated `key=value` resource attributes (e.g. `deployment.environment=dev,service.version=1.0.0`). |
| `OTEL_SDK_DISABLED` | `false` | Standard OTel kill-switch (`true` disables OpenTelemetry). |
| `MICROHARNESS_OTEL_ENABLED` | Auto | Force-enable (`1`/`true`) or disable (`0`/`false`) OpenTelemetry instrumentation. |
| `MICROHARNESS_OTEL_CONSOLE` | `0` | Set to `1`/`true` to export spans to `stderr` (`ConsoleSpanExporter`). |
| `MICROHARNESS_OTEL_INCLUDE_CONTENT` | `true` | Include message prompts, completions, and tool arguments/results in span attributes. |
| `MICROHARNESS_OTEL_INCLUDE_BINARY_CONTENT` | `true` | Include base64 binary content in spans. |
| `MICROHARNESS_OTEL_INCLUDE_PARAMETERS` | `true` | Include model request parameters and tool schemas in spans. |
| `MICROHARNESS_OTEL_VERSION` | `5` | Pydantic AI GenAI semantic convention version (`2`, `3`, `4`, `5`, or `6`). |

### Usage via Environment Variables or CLI

Send traces and metrics to a local OTLP collector (Jaeger, Arize Phoenix, Grafana Tempo, Langfuse, SigNoz, or OpenTelemetry Collector):

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
export OTEL_SERVICE_NAME="microharness-dev"
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=local"

uv run microharness.py query "Analyze the repository structure."
```

Or inspect spans directly on `stderr` without running a collector:

```bash
uv run microharness.py query "Hello!" --otel-console
```

### Declarative Configuration in `agent.yaml`

You can also configure OpenTelemetry directly in `agent.yaml` using the `OpenTelemetry` capability:

```yaml
capabilities:
  - OpenTelemetry:
      enabled: true
      endpoint: http://localhost:4318
      protocol: http/protobuf
      service_name: microharness
      headers:
        Authorization: "Bearer my-secret-token"
      resource_attributes:
        deployment.environment: development
      console: false
      include_content: true
      include_binary_content: true
      include_model_request_parameters: true
      version: 5
      instrument_all: true
```
