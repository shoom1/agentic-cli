# Agentic CLI

A framework for building domain-specific agentic CLI applications powered by LLM agents.

## Overview

Agentic CLI provides the core infrastructure for building interactive CLI applications that leverage LLM agents for complex tasks. It offers:

- **Pluggable Orchestration**: Choose Google ADK (the default) or Deep Agents for agent workflows; both run every model provider. LangGraph is deprecated and will be removed in 0.8.0
- **Any Model, Per Agent**: Gemini, Claude, GPT, OpenRouter's models and your own OpenAI-compatible server (vLLM), mixed in one agent tree; models named by reference or role, with one set of thinking levels
- **Rich Terminal UI**: Dual thinking boxes, markdown rendering, and streaming responses via `thinking-prompt`
- **Declarative Agents**: Define agents with simple configuration objects
- **Native Tool Architecture**: Backend-specific tool factories for each backend, with automatic HITL confirmation for dangerous tools
- **Built-in Tools**: Python execution, stateful sandbox, file operations, web search, web fetch, arXiv search
- **Knowledge Base**: Semantic + BM25 hybrid search with RRF fusion, per-document markdown sidecars, and agent-authored concept pages
- **Semantic Memory**: Embedding-backed memory with lifecycle management, contradiction detection, and forgetting policy
- **Durable Sessions**: Conversations persist continuously by default (SQLite; PostgreSQL or ephemeral memory optional) and resume across CLI restarts with `--session <id>`
- **Context Window Management**: Native trim detection, summarization near the model's input budget on Deep Agents, and token-usage visibility
- **Model Catalog**: Live model listings from each provider, cached for a day, shown by `/models`
- **Type-safe Configuration**: Composable settings mixins (`pydantic-settings`)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BaseCLIApp                                  │
│  - Terminal UI (thinking-prompt): response + task-progress boxes    │
│  - Command registry (/help, /status, /settings, /sandbox, ...)      │
│  - Message history, background initialization                       │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BaseWorkflowManager                              │
│  - Agent orchestration, event streaming                             │
│  - Session save/resume, usage tracking, context-trim detection      │
│  - Service registry: shared state for tools (KB, memory, sandbox…)  │
├─────────────────────────────────────────────────────────────────────┤
│  GoogleADKWorkflowManager (default)                                 │
│  + PermissionPlugin, LLMLoggingPlugin, TaskProgressPlugin           │
├─────────────────────────────────────────────────────────────────────┤
│  DeepAgentsWorkflowManager (deepagents extra)                       │
│  + PermissionMiddleware, SubAgentMiddleware (task), summarization   │
├─────────────────────────────────────────────────────────────────────┤
│  LangGraphWorkflowManager (langgraph extra; deprecated)             │
│  + wrap_tool_for_permission, native ToolNode                        │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         AgentConfig[]                               │
│  - name, prompt, tools, sub_agents, description, model              │
└─────────────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.12+
- A key for at least one model provider (Google, Anthropic, OpenAI or OpenRouter), or an OpenAI-compatible server of your own (see [Models and providers](#models-and-providers))

## Installation

### Basic Installation (Google ADK)

```bash
pip install agentic-cli
```

This installs Google ADK 2 (2.12 or later) with MCP support, SQLAlchemy for
the durable session store, and the OpenAI SDK (for GPT, OpenRouter and
OpenAI-compatible servers). The ADK backend runs every provider.

### With Deep Agents

```bash
pip install 'agentic-cli[deepagents]'
```

The extra installs the Deep Agents backend: Deep Agents (0.7.x, held below
0.8 while it is in beta), the LangChain packages it builds its models with
(`langchain-google-genai`, `langchain-anthropic`, `langchain-openai`,
`langchain-openrouter`), LangGraph's sqlite and Postgres checkpointers for its
sessions, and `langchain-mcp-adapters` for MCP servers. The LangChain packages
bring two constraints a core install does not have: `pydantic` stays below
2.13 (every `openrouter` release caps it there), and `anthropic` moves to 1.x,
which runs on `httpx2`, so code that hands the Anthropic SDK an `httpx` client
passes an `httpx2` one instead. Set `orchestrator` to `deepagents` to run on
it (see [Workflow Managers](#workflow-managers)).

### With LangGraph Support (deprecated)

```bash
pip install agentic-cli[langgraph]
```

The LangGraph backend is deprecated and will be removed in 0.8.0; move to
[Deep Agents](#with-deep-agents). The extra installs the same LangChain model
packages, with the same two constraints. It installs beside `deepagents`.

### Development Installation

```bash
git clone https://github.com/shoom1/agentic-cli.git
cd agentic-cli
pip install -e ".[dev]"
```

### Using Conda

```bash
conda env create -f environment.yml
conda run -n agenticcli pip install -e .
```

## Quick Start

Create a minimal CLI application in just a few lines:

```python
import asyncio
from agentic_cli import BaseCLIApp, BaseSettings
from agentic_cli.cli import AppInfo
from agentic_cli.workflow import AgentConfig
from agentic_cli.tools import register_tool, ToolCategory
from agentic_cli.workflow.permissions import EXEMPT

# Define your tools. Tools are permission-gated, so register each one with
# the capabilities it needs (EXEMPT = a pure function that touches nothing).
@register_tool(category=ToolCategory.OTHER, capabilities=EXEMPT)
def greet(name: str) -> dict:
    """Greet a person by name."""
    return {"success": True, "greeting": f"Hello, {name}!"}

# Configure your agent
AGENTS = [
    AgentConfig(
        name="assistant",
        prompt="You are a helpful assistant. Use the greet tool when asked to greet someone.",
        tools=[greet],
    ),
]

# Create and run your app
if __name__ == "__main__":
    app = BaseCLIApp(
        app_info=AppInfo(name="My App", version="0.1.0"),
        agent_configs=AGENTS,
        settings=BaseSettings(),
    )
    asyncio.run(app.run())
```

Run with your API key:

```bash
export GOOGLE_API_KEY="your-api-key"
python my_app.py
```

### Customizing the CLI

`AppInfo` is the banner the CLI prints at startup: the app's name and
version, or a `welcome_message` of your own (a string, or a callable that
returns one or a Rich renderable). How the session behaves is `BaseCLIApp`'s.
A subclass adds commands (`register_commands()`), chooses what `/settings`
shows (`get_ui_setting_keys()`), and recolors a theme with
`get_theme(name)`:

```python
import dataclasses

from thinking_prompt import ThinkingPromptStyles

class MyApp(BaseCLIApp):
    def get_theme(self, name: str) -> str | ThinkingPromptStyles:
        if name == "dark":
            return dataclasses.replace(ThinkingPromptStyles.dark(), thinking_box="fg:#9b59b6")
        return super().get_theme(name)
```

The session draws in `get_theme(name)` for the user's `ui_theme` (see
[Theme and dialog placement](#theme-and-dialog-placement)): `dark` unless they
chose another, which is thinking-prompt's dark theme in agentic-cli's colors.
Any other theme name comes back as it is, for thinking-prompt's own theme,
since colors tuned for a dark background would be unreadable on a light one.
To change one color and keep agentic-cli's others, start from
`super().get_theme("dark")` instead of `ThinkingPromptStyles.dark()`.

An app written for 0.6 makes two edits: it drops `echo_thinking=` from
`AppInfo`, which thinking-prompt 0.4 refuses, and turns `get_styles()` into
`get_theme(name)`. An app still defining `get_styles()` fails at
construction with a `TypeError` that says so.

```python
# 0.6
AppInfo(name="App", version="1.0", welcome_message=banner, echo_thinking=False)

def get_styles(self) -> ThinkingPromptStyles:
    return ThinkingPromptStyles(thinking_box="fg:#9b59b6", ...)

# 0.7
AppInfo(name="App", version="1.0", welcome_message=banner)

def get_theme(self, name: str) -> str | ThinkingPromptStyles:
    if name == "dark":
        return dataclasses.replace(ThinkingPromptStyles.dark(), thinking_box="fg:#9b59b6", ...)
    return super().get_theme(name)
```

## Workflow Managers

### GoogleADKWorkflowManager (Default)

Uses Google's Agent Development Kit for orchestration. Best for:
- Simple agent hierarchies
- Prompt caching on Claude and Gemini (see [Prompt caching](#prompt-caching))
- Quick prototyping

```python
from agentic_cli import GoogleADKWorkflowManager

manager = GoogleADKWorkflowManager(
    agent_configs=AGENTS,
    settings=settings,
    model="google:gemini-2.5-flash",  # Optional: the default model otherwise
)
```

ADK integrations:
- **Model factory** — `workflow.adk.model_factory.build_adk_model` builds each
  agent's model from its resolved model: ADK's `Gemini`, `AnthropicLlm` and
  `OpenAIResponsesLlm`, and `OpenAICompatibleLlm` for OpenRouter and your
  endpoints, which shows their reasoning text
- **PermissionPlugin** — evaluates each tool's declared capabilities against the permission engine and prompts the user when no rule matches
- **LLMLoggingPlugin** — structured logging of LLM requests/responses
- **TaskProgressPlugin** — streams the task checklist into its own thinking box

### DeepAgentsWorkflowManager

Builds each agent as a LangChain `create_agent` graph with selected Deep
Agents middleware (its `task` delegation and its summarization) and, on
Claude, LangChain's Anthropic prompt caching. Deep Agents' file tools, virtual
filesystem and `write_todos` are not used: the agents run on the framework's
tools, permission engine and models, as on ADK. Best for:
- A coordinator that stays in charge and hands each sub-agent a brief
- Long conversations, summarized before they outgrow the model
- Apps moving off the LangGraph backend

```python
from agentic_cli import DeepAgentsWorkflowManager

manager = DeepAgentsWorkflowManager(
    agent_configs=AGENTS,
    settings=settings,
    model="anthropic:claude-sonnet-4-6",  # Optional: the default model otherwise
)
```

What it supports:
- **Delegation through `task`**: a coordinator (an agent with `sub_agents`)
  gets Deep Agents' `task` tool, which lists each sub-agent as
  `- name: description`, so give sub-agents real descriptions. A sub-agent
  sees only the brief the coordinator writes, and the coordinator's plan and
  tasks, never the conversation. Its last answer is the call's result, and the
  coordinator answers the user. Two sub-agents can run at once. `task` is a
  reserved tool name.
- **The permission gate on every call**: `PermissionMiddleware`, in every
  agent's stack, identifies each tool by the object the registry issued and
  asks through the input callback, for sub-agents' calls too. A tool that
  raises fails its call with `{"success": false, "error": "..."}`, as on ADK,
  and the turn goes on.
- **Sessions**: `session_store` picks a LangGraph checkpointer: sqlite (the
  default) in `{workspace}/sessions/checkpoints.db`, next to ADK's
  `sessions.db`, Postgres at `postgres_uri`, or memory. Each conversation is
  the thread of its whole `SessionRef`, so two users never share one.
- **Context**: Deep Agents' summarization runs in every agent, at 85% of the
  model's input budget (95% of its context window, less its `max_tokens`),
  keeping the newest 10% of that budget word for word. The checkpoint keeps
  every message. With `context_window_enabled`,
  `context_window_trigger_tokens` and `context_window_target_tokens` set the
  thresholds instead.
- **Prompt caching** on Claude, as on ADK (see [Prompt caching](#prompt-caching)).
- **MCP tools, Agent Skills, raw LLM logging and job resume**, from the same
  `AgentConfig.mcp_servers`, `AgentConfig.skills`, `raw_llm_logging` and
  `job_auto_resume` (or `/resume`) as on ADK. An MCP tool is gated under the
  `mcp` capability, with its name as the target.

How it differs from ADK:
- **Delegation**: on ADK, a coordinator transfers the conversation to a
  sub-agent (`transfer_to_agent`), which sees all of it and answers the user.
  On Deep Agents, the sub-agent gets a brief and the coordinator answers.
- **No Gemini explicit caches**: Gemini runs on its implicit caching, which is
  automatic and bills no storage.
- **No code-execution events**: `CODE_EXECUTION`, `EXECUTABLE_CODE` and
  `FILE_DATA` come from Gemini's code execution on ADK.
- **A finished job** reaches the root agent as a message of its own, which
  opens with `[Automatic job notification — not written by the user]`, rather
  than as the result of the call that started it.
- **Its own sessions**: neither backend reads the other's, so switching the
  orchestrator starts a new conversation.

Requires: `pip install 'agentic-cli[deepagents]'`

### LangGraphWorkflowManager (deprecated)

Deprecated in 0.7.0 and removed in 0.8.0: use
[Deep Agents](#deepagentsworkflowmanager) instead. Building it raises a
`DeprecationWarning`, the CLI warns at startup when `orchestrator` is
`langgraph`, and sessions saved by it do not carry over. Its known defects
stay unfixed.

Uses LangGraph for orchestration. Best for:
- Cyclical workflows (self-validation, iterative refinement)
- State checkpointing and time-travel debugging
- Complex multi-agent coordination

```python
from agentic_cli.workflow.langgraph import LangGraphWorkflowManager

manager = LangGraphWorkflowManager(
    agent_configs=AGENTS,
    settings=settings,
    checkpointer="memory",  # "memory", "postgres", "sqlite", or None
)
```

Features:
- **One chat model per agent**: `workflow.langchain_models.build_chat_model` builds
  it once, from the agent's resolved model, with its key, timeout and retries:
  `ChatGoogleGenerativeAI` (the Gemini API, not Vertex AI), `ChatAnthropic`,
  `ChatOpenAI` on the Responses API, `ChatOpenRouter`, and `OpenAICompatibleChat`
  for your endpoints
- **Per-agent settings and thinking**: each agent's `model_settings` apply, and
  thinking text reaches the thinking box for every provider
- **Retry policies**: Automatic retry with exponential backoff
- **Permission wrapper**: Wraps each tool to gate execution through the permission engine
- **Event streaming**: Real-time workflow events via `WorkflowEvent`

Requires: `pip install agentic-cli[langgraph]`

### Comparison

| Feature | Google ADK | Deep Agents | LangGraph (deprecated) |
|---------|------------|-------------|------------------------|
| Install | Core | `deepagents` extra | `langgraph` extra |
| Model providers | Gemini, Claude, GPT (Responses API), OpenRouter, OpenAI-compatible endpoints | The same | The same |
| Delegation | `transfer_to_agent`: the sub-agent takes the conversation | `task`: the sub-agent gets a brief, the coordinator answers | Routing between agent nodes |
| State persistence | SQLite (default), PostgreSQL, or memory | The same, as a LangGraph checkpointer | Memory, PostgreSQL, or SQLite |
| Thinking support | Every model with a thinking knob, one set of levels | The same | The same |
| Prompt caching | Claude (three breakpoints) and Gemini (server-side caches) | Claude (three breakpoints) | Claude on Anthropic's API (one breakpoint) |
| Context | None on normal turns (`context_window_enabled` applies only to Gemini's Live API) | Summarization near the input budget | Trimming, with `context_window_enabled` |
| MCP tools, Agent Skills | Yes | Yes | Ignored, with a warning |
| Raw LLM logging, job resume | Yes | Yes | No |
| Code-execution events | Gemini's | None | None |
| Retry handling | Built-in | Built-in | Built-in with backoff |
| Permission gate | PermissionPlugin | PermissionMiddleware | wrap_tool_for_permission |

### Auto-selection via Settings

```python
from agentic_cli import create_workflow_manager_from_settings

settings = BaseSettings(orchestrator="deepagents")  # or "adk" (the default)
manager = create_workflow_manager_from_settings(agent_configs=AGENTS, settings=settings)
```

`orchestrator="langgraph"` still works in 0.7.x, with a deprecation warning.

## Configuration

### BaseSettings

Settings are organized into composable mixins (`AppSettingsMixin`, `CLISettingsMixin`, `WorkflowSettingsMixin`). `BaseSettings` composes all three.

All settings can be configured via environment variables with the `AGENTIC_` prefix, via constructor arguments, or from layered JSON/dotenv files:

```python
from pathlib import Path
from pydantic_settings import SettingsConfigDict
from agentic_cli import BaseSettings

class MySettings(BaseSettings):
    model_config = SettingsConfigDict(
        # Absolute (user-level) paths stay trusted — see the trust note below.
        env_file=str(Path.home() / ".my_app" / ".env"),
        env_prefix="MYAPP_",  # Custom prefix
    )
    app_name: str = "my_app"
    workspace_dir: Path = Path.home() / ".my_app"
```

### Precedence and the trust boundary

Sources are consulted highest-precedence first (JSON sources only when the file exists):

1. **Constructor arguments** — `MySettings(google_api_key=...)`
2. **Environment variables** — `MYAPP_*` / `AGENTIC_*`
3. **Project config** — `./.{app_name}/settings.json` — **untrusted**
4. **User config** — `~/.{app_name}/settings.json`
5. **Dotenv** — whatever `env_file` points at
6. **Field defaults**

Levels 1, 2 and 4 are trusted; **level 3 is not**, and neither is a
*cwd-relative* `env_file`. A cloned repository can ship both, so both are
filtered down to an explicit allowlist of benign keys (model and behaviour
settings, timeouts, sandbox *resource* limits, `session_store`, log verbosity).
Security-relevant keys set from an untrusted source — executor backend, sandbox
image/mounts/user, OS-sandbox policy, `skills_dirs`, shell sandbox settings,
`workspace_dir`, `raw_llm_logging`, permission rules, `model_endpoints` (the
addresses a key is sent to), and any credential — are dropped with a logged
warning rather than applied.

**Consequence:** put API keys in real environment variables or an
absolute/user-level file. A key placed in a cwd `.env` is deliberately ignored.

### Key Settings

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| `google_api_key` | `GOOGLE_API_KEY` | None | Google API key for Gemini |
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | None | Anthropic API key for Claude |
| `openai_api_key` | `OPENAI_API_KEY` | None | OpenAI API key for GPT |
| `openrouter_api_key` | `OPENROUTER_API_KEY` | None | OpenRouter API key |
| `default_model` | `AGENTIC_DEFAULT_MODEL` | Auto | The default model: a `provider:model` reference or a model id |
| `model_roles` | `AGENTIC_MODEL_ROLES` | {} | A model per role: `fast`, `strong`, or an app's own |
| `pinned_models` | `AGENTIC_PINNED_MODELS` | [] | Models the `/settings` model pickers always offer |
| `model_endpoints` | `AGENTIC_MODEL_ENDPOINTS` | {} | Your OpenAI-compatible servers (never from a project file) |
| `thinking_effort` | `AGENTIC_THINKING_EFFORT` | "medium" | Thinking level: none, low, medium, high |
| `prompt_caching_enabled` | `AGENTIC_PROMPT_CACHING_ENABLED` | true | Prompt caching: Claude, and Gemini on ADK |
| `orchestrator` | `AGENTIC_ORCHESTRATOR` | "adk" | Orchestrator: `adk`, `deepagents`, or `langgraph` (deprecated) |
| `session_store` | `AGENTIC_SESSION_STORE` | "sqlite" | Where conversations are kept: `sqlite`, `postgres`, or `memory` |
| `workspace_dir` | `AGENTIC_WORKSPACE_DIR` | ~/.agentic | Storage directory |
| `log_level` | `AGENTIC_LOG_LEVEL` | "warning" | Logging level |
| `ui_theme` | `AGENTIC_UI_THEME` | "dark" | The CLI's theme: `dark`, `light`, `mono`, `terminal` or `auto` (user settings only) |
| `dialog_placement` | `AGENTIC_DIALOG_PLACEMENT` | "box" | Dialogs in a framed `box`, or `inline` under the prompt (user settings only) |
| `tavily_api_key` | `TAVILY_API_KEY` | None | Tavily API key for web search |
| `brave_api_key` | `BRAVE_API_KEY` | None | Brave Search API key |
| `search_backend` | `AGENTIC_SEARCH_BACKEND` | Auto | Web search provider (tavily/brave) |
| `webfetch_max_download_bytes` | `AGENTIC_WEBFETCH_MAX_DOWNLOAD_BYTES` | 5242880 | Largest page body read and saved |
| `webfetch_saved_max_age_days` | `AGENTIC_WEBFETCH_SAVED_MAX_AGE_DAYS` | 7 | Saved pages older than this are deleted |
| `webfetch_saved_max_mb` | `AGENTIC_WEBFETCH_SAVED_MAX_MB` | 200 | Oldest saved pages are deleted past this size |
| `retry_max_attempts` | `AGENTIC_RETRY_MAX_ATTEMPTS` | 3 | Max retries on transient errors |
| `retry_initial_delay` | `AGENTIC_RETRY_INITIAL_DELAY` | 2.0 | Initial retry backoff (seconds) |

Settings are persisted by `SettingsPersistence` (atomic writes) and editable at runtime via the `/settings` command.

### Theme and dialog placement

`ui_theme` and `dialog_placement` are the user's: how the CLI looks applies in
every project. They come from `~/.{app_name}/settings.json` or the environment
(`AGENTIC_UI_THEME` and `AGENTIC_DIALOG_PLACEMENT`, or your app's prefix); a
project file or a cwd `.env` that sets them is ignored with a warning, like
any other user-scoped key. `settings_persistence.USER_PREFERENCE_KEYS` names
them.

`/settings` does not show them unless the app lists them, and they are the
only user-scoped keys it may list (any other is left out, with a warning):

```python
class MyApp(BaseCLIApp):
    def get_ui_setting_keys(self) -> list[str]:
        return [*super().get_ui_setting_keys(), "ui_theme", "dialog_placement"]
```

A Save writes them to the user file, and one set back to its default leaves
it, so the next start follows the default again. `examples/research_demo`
lists both.

- **The theme** switches at once, through `get_theme(name)`. thinking-prompt
  repaints: it clears the screen and the terminal's scrollback, shell output
  from before the session included, and prints the conversation again in the
  new theme. Messages and markdown take the new colors; Rich panels and tables
  (`/help`'s, for one) keep the colors they were printed in. Under `auto`,
  thinking-prompt picks the theme: `mono` when `NO_COLOR` is set, `light` or
  `dark` from `COLORFGBG` when the terminal sets it, and `terminal` (the
  terminal's own palette) otherwise; agentic-cli's dark colors do not apply
  there. `NO_COLOR` draws in mono whatever the theme.
- **The placement** applies from the next dialog agentic-cli opens: `/settings`
  and the tools' questions, permission prompts included. An app's own dialogs
  keep the placement the session started with, unless they pass
  `placement=self.settings.dialog_placement`.

### Settings Context

For multi-tenant or isolated contexts:

```python
from agentic_cli import SettingsContext

with SettingsContext(custom_settings):
    # All code here uses custom_settings
    result = my_tool()
```

## Models and providers

Each agent runs on a model named by a reference or a role, decided once, when
the workflow starts. Every backend runs every provider: Gemini, Claude, GPT,
OpenRouter's models, and models on an OpenAI-compatible server of your own
(vLLM and the like), and one agent tree can mix them. What agentic-cli knows
about models lives in `agentic_cli.models` (provisional in 0.7.0: its names
may still change): a table of model families shipped with the package, merged
with each provider's live model list.

### References and roles

A model is named the same way in `default_model`, `model_roles`,
`pinned_models`, `AgentConfig.model`, `Manager(model=...)` and agent YAML:

| You write | It means |
|-----------|----------|
| `anthropic:claude-sonnet-4-6` | `provider:model`. The provider is `google`, `anthropic`, `openai`, `openrouter`, or one of your endpoints |
| `openrouter:qwen/qwen3-235b-a22b:free` | Split on the first colon only, so the model id keeps its own colons |
| `gemini-2.5-flash`, `claude-sonnet-4-6`, `gpt-5.5`, `o3` | A bare `gemini-`, `claude-`, `gpt-` or `o<digit>` id names its provider |
| `fast` | Any other bare name is a role. A role name has no digits, so a bare `qwen3-32b` is an error that asks for its provider (`local-vllm:qwen3-32b`) |

A role names a model by what it is for:

- `default` is `default_model`. An agent without a `model` runs on it.
- `fast` and `strong` are mapped in `model_roles`.
- An app adds a role by naming it in an agent (`AgentConfig(model="vision")`),
  and may ship a mapping by overriding the `model_roles` default in its
  settings class. An app role nobody mapped runs on the default model, with a
  warning.
- A role maps to a model, never to another role, and `model_roles` cannot map
  `default`.

A built-in role nobody mapped takes the newest model of its tier from the first
provider with a key, in the order Google, Anthropic, OpenAI. With only a Google
key and no `default_model`, that is Gemini Flash, as in 0.6. OpenRouter models
and your endpoints never fill a role on their own: map them.

| Provider | `fast` | `default` | `strong` |
|----------|--------|-----------|----------|
| Google | Flash-Lite | Flash | Pro |
| Anthropic | Haiku | Sonnet | Opus |
| OpenAI | mini | standard | pro |

A project chooses its models in `./.my_app/settings.json`:

```json
{
  "default_model": "anthropic:claude-sonnet-4-6",
  "model_roles": {
    "fast": "google:gemini-2.5-flash-lite",
    "strong": "openrouter:qwen/qwen3-235b-a22b"
  },
  "pinned_models": ["openrouter:meta-llama/llama-3.3-70b-instruct"]
}
```

and its agents name roles or models:

```python
from agentic_cli import AgentConfig

AGENTS = [
    AgentConfig(name="coordinator", prompt="Plan the work.", sub_agents=["drafter", "checker"]),
    AgentConfig(name="drafter", prompt="Draft quickly.", model="fast"),
    AgentConfig(name="checker", prompt="Check carefully.", model="strong"),
]
```

The coordinator runs on Claude Sonnet 4.6, the drafter on Gemini 2.5
Flash-Lite, and the checker on Qwen3 through OpenRouter. Every model is checked
when the workflow starts: a reference to an endpoint you did not define, a
provider without its key, or a model its provider's listing lacks fails with
one error that names each agent concerned.

### Provider keys

| Provider | Setting | Environment variable |
|----------|---------|----------------------|
| Google (the Gemini API) | `google_api_key` | `GOOGLE_API_KEY` |
| Anthropic | `anthropic_api_key` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai_api_key` | `OPENAI_API_KEY` |
| OpenRouter | `openrouter_api_key` | `OPENROUTER_API_KEY` |

Keys are never saved and never shown (not in `repr`, not by `/models`), and a
project settings file or a cwd `.env` cannot set them: put them in the
environment or a user-level dotenv file (see
[Precedence and the trust boundary](#precedence-and-the-trust-boundary)). Each
model's client is given its key. Google, Anthropic and OpenAI models go to
their SDK's default host; Vertex AI is not supported.

### Your own servers: vLLM and other OpenAI-compatible endpoints

Define an endpoint in your user settings, `~/.my_app/settings.json`:

```json
{
  "model_endpoints": {
    "local-vllm": {
      "base_url": "http://localhost:8000/v1",
      "api_key_env": "LOCAL_VLLM_KEY",
      "models": {
        "Qwen/Qwen3-32B": {
          "context_window": 32768,
          "max_output": 8192,
          "thinking": "chat_template"
        }
      }
    }
  }
}
```

`local-vllm:Qwen/Qwen3-32B` is then a reference like any other, for a role or
an agent.

- **Only your user settings (or the environment) can define an endpoint.**
  `model_endpoints` holds the addresses a key is sent to. A project settings
  file, which a cloned repository can ship, or a cwd `.env` that sets it is
  ignored with a warning, and `/settings` never writes it there. A project
  chooses among the endpoints you defined (`default_model`, `model_roles`,
  `pinned_models`); a reference to one you did not define fails at startup.
- The key comes only from the environment variable `api_key_env` names, and is
  never stored. If that variable is not set, startup fails for a model that
  runs on the endpoint, naming the variable (a `.env` file does not set it). Without `api_key_env` the client sends the
  placeholder key `EMPTY`.
- An endpoint's name is lowercase letters, digits, `_` and `-`, starting with a
  letter, and cannot be a built-in provider's. `base_url` is `http` or
  `https`, with no user name, password, query or fragment.
- What you can declare about a model: `context_window`, `max_output`, `tools`
  (default `true`), `image_input` (`false`), `reasoning_text` (`true`) and
  `thinking`: `none` (the default), `chat_template` (vLLM's
  `chat_template_kwargs.enable_thinking`), or `effort` (`reasoning_effort`,
  with the model's `effort_levels`, by default `low`, `medium` and `high`).
- A model the server serves but you did not declare is assumed to take tools,
  with the server's `max_model_len` as its context window.
- Reasoning text shows only when the server runs with a reasoning parser.

### Thinking

`thinking_effort` (`none`, `low`, `medium` or `high`; `medium` by default)
applies to every agent that sets no thinking of its own. An agent sets its own
in `model_settings`, with one of those, `budget`, or a native level of its
model:

```python
from agentic_cli import AgentConfig, ModelSettings, ThinkingSettings

reviewer = AgentConfig(
    name="reviewer",
    prompt="Review the draft.",
    model="anthropic:claude-opus-4-7",
    model_settings=ModelSettings(thinking=ThinkingSettings(mode="xhigh")),
)
```

What a level means depends on how the model thinks:

| The model thinks by | `none` | `low` / `medium` / `high` | `budget` (`budget_tokens=N`) |
|---------------------|--------|---------------------------|------------------------------|
| A budget: Gemini 2.5 | Off (budget 0); Pro, which cannot stop, its minimum | 4096 / 12288 / 24576 tokens | N, within the model's range |
| Levels: Gemini 3 | Its lowest level | Its `low` / `medium` / `high` (3 Pro: `medium` is `high`) | Error |
| Adaptive thinking: Claude 4.6 and later | Off; on the models that cannot stop (below), effort `low` | Effort `low` / `medium` / `high` | Error |
| A budget: Claude up to 4.5 | Off | 4096 / 10000 / 32000 tokens | N, at least 1024 |
| Reasoning effort: GPT and OpenRouter reasoning models, endpoints that declare `effort` | The lowest effort the model has | Effort `low` / `medium` / `high` | Error |
| A chat template: endpoints that declare `chat_template` | `enable_thinking: false` | `enable_thinking: true` | Error |

- Some Claude models cannot turn thinking off: Opus 5.5, Sonnet 5.5, Fable 5
  and Fable 5.1 answer thinking `disabled` with HTTP 400. On them `none` runs
  adaptive thinking at effort `low`. Anthropic's model listing says which
  models cannot (`capabilities.thinking.types.disabled`); without a listing,
  the shipped table names these four.
- `high` is the model's own `high`, not its top level (Claude's top is `max`).
- A native level, such as `minimal`, `xhigh` or `max`, works on a model that
  has it.
- A level the model lacks, or `budget` on a model without budget thinking,
  fails at startup with an error that names the agent, the model and the levels
  it takes.
- A model with no thinking knob ignores the setting.
- `/settings` shows what the effort means for the default model, such as
  `high → anthropic:claude-sonnet-4-6: adaptive, effort high`, or
  `none → anthropic:claude-sonnet-5-5: adaptive, effort low`.

### Other per-agent settings

`ModelSettings` also takes `temperature`, `top_p`, `top_k`, `max_tokens`,
`stop_sequences` and `extra`, and every backend applies them:

- `max_tokens` is the agent's value, else the provider's default (8192 for
  Claude), within the model's max output. On Claude with a thinking budget it
  is raised to at least the budget plus 8192, since thinking counts against it.
- Temperature, top-p and top-k are dropped, with a debug log, where the model
  rejects them: Claude while it thinks, the Claude models that take none at
  all (Opus 4.7 and later, Sonnet 5 and later, Fable), and GPT reasoning
  models.
- `extra` passes provider-specific fields. A field the model's client does not
  take is logged and dropped.

### Prompt caching

`prompt_caching_enabled` (on by default) turns caching on for the whole app.
What it does depends on the backend and the model:

- **On ADK**, the app gets ADK's context cache when any agent runs on Claude or
  Gemini (an agent without a `model` runs on the default model). A native ADK
  YAML app (`adk_config_path`) has no agent list to check, so its default model
  decides. The cache belongs to the app: the setting turns it on or off for
  every agent at once.
- **On Deep Agents**, each agent on Claude on Anthropic's API marks three cache
  breakpoints, as on ADK: its system prompt, its last tool and the
  conversation, for 5 minutes. A sub-agent's own prompt and tools are cached
  too. Gemini gets no explicit cache, only its implicit caching.
- **On LangGraph**, it marks the system prompt of Claude on Anthropic's API:
  one cache breakpoint. LangGraph creates no Gemini cache.
- GPT, OpenRouter and endpoint models get nothing from it on any backend.
  OpenAI caches long prompts on its own.

What it costs:

- **Claude**: on ADK and Deep Agents, every request from the first turn marks
  three cache breakpoints, on the last tool, the system instruction and the
  conversation so far. A cache write costs 1.25 times the input price and a
  read about a tenth of it (less on Opus 5.5 and Fable 5.1). The cache lives
  5 minutes and each read renews it, so a conversation's first turn costs a
  little more and each later turn within 5 minutes much less. ADK's
  1800-second TTL keeps Claude on that 5-minute cache: the hour-long one costs
  twice the input price to write.
- **Gemini** (ADK only): from an agent's second turn, once its system
  instruction, tools and conversation reach 2048 tokens (4096 on Gemini 3), ADK
  creates a cache on Google's side and later requests name it instead of
  resending the prefix. ADK makes a new one after 10 invocations. Google bills
  the cache's storage until it expires (30 minutes after it is made) or is
  replaced, and the tokens read from it at its lower cached rate, so a short
  session can cost more than it saves. Creating a cache, or deleting an old
  one, may take at most 60 seconds, and an agent tries at most once a turn: a
  create that fails or hangs costs that turn its caching, never the turn.

To turn caching off, set `"prompt_caching_enabled": false` in
`~/.my_app/settings.json`; a project file cannot set it. On ADK and Deep
Agents, `/status` shows the cached tokens read so far. A system prompt that
changes every turn (a timestamp, say) defeats both caches.

### The model catalog and `/models`

The catalog merges three sources, highest precedence first: your endpoint
declarations, each provider's live model list, and the shipped table (model
families, tiers, thinking rules, and the numbers used when a listing fails).

- Listings are cached in `~/.my_app/model_cache.json` (private, versioned,
  holding no key) for 24 hours. Startup waits only to list a provider with a
  key, or an endpoint, that has no cache entry yet: all of them at once, each
  for at most 10 seconds. An older entry is used while it is listed again in
  the background.
- A model a provider does not list is rejected only when that provider's
  listing is fresh. A failed, untried or older listing accepts any id.
- `/models` shows each provider's status (credentials, whether its listing is
  fresh, stale, failed or not tried, when it was fetched, how many models, and
  the last error), then the models of each provider with a key: reference,
  context window, max output, thinking, tools, cache, where the values came
  from, and the roles that use the model. It works before the workflow is
  initialized, from the cache.
- `/models <text>` shows only the models whose reference contains the text,
  ignoring case: `/models qwen` among OpenRouter's hundreds.
- `/models refresh` lists every provider again now, and says which providers'
  models changed. Run it after a key change to an account with other models.
- `/settings` has a model picker for the default model and one for each of
  the `fast` and `strong` roles. Each offers the models that can run: those of
  each provider with a key, your endpoints' models, and the OpenRouter models a
  role maps or `pinned_models` lists. Picking a role writes `model_roles` to
  the project settings file and restarts the workflow on it, keeping the
  conversation. An app adds a picker for a role of its own with the key
  `model_role:<role>` in `get_ui_setting_keys()`.
- In code, an initialized manager's `manager.catalog` gives
  `catalog.records("google")`, `catalog.status()` and `await catalog.refresh()`,
  and `manager.model` is the default model's reference.

### How each backend runs them

- Every backend runs every provider and builds each agent's model once, when
  the workflow starts: ADK with `workflow.adk.model_factory.build_adk_model`,
  Deep Agents and LangGraph with `workflow.langchain_models.build_chat_model`.
- GPT runs on OpenAI's Responses API on all of them, without storing responses
  (`store=False`). OpenRouter and endpoints run on Chat Completions.
- Each agent's `model_settings` apply on every backend, and thinking text
  reaches the thinking box for every provider.
- The `[deepagents]` and `[langgraph]` extras pin `pydantic` below 2.13 and
  bring `anthropic` 1.x (see [Installation](#with-deep-agents)).
- On ADK, GPT's reasoning is not passed back across tool calls, and neither is
  OpenRouter's `reasoning_details`; Deep Agents and LangGraph pass both back.
- After a switch to another provider's model, no backend sends the new model
  the old one's reasoning.

## Agent Configuration

Agents are defined declaratively using `AgentConfig`:

```python
from agentic_cli import AgentConfig

# Simple agent
assistant = AgentConfig(
    name="assistant",
    prompt="You are a helpful assistant.",
    tools=[my_tool],
)

# Agent with dynamic prompt
def get_prompt():
    return f"Today is {datetime.now().strftime('%Y-%m-%d')}. Help the user."

dynamic_agent = AgentConfig(
    name="dynamic",
    prompt=get_prompt,  # Callable for dynamic prompts
    tools=[tool_a, tool_b],
)

# A prompt factory may also take the manager's settings explicitly. Either way
# it is evaluated under the manager's settings, not the global singleton.
def get_scoped_prompt(settings):
    return f"You are {settings.app_name}."

# Coordinator with sub-agents
coordinator = AgentConfig(
    name="coordinator",
    prompt="Route requests to the appropriate specialist.",
    tools=[],
    sub_agents=["researcher", "analyst"],  # Names of other agents
    description="Routes work to specialists",
)

researcher = AgentConfig(
    name="researcher",
    prompt="Research topics thoroughly.",
    tools=[web_search],
)

analyst = AgentConfig(
    name="analyst",
    prompt="Analyze data and provide insights.",
    tools=[calculate],
)

# Pass all configs to workflow manager
configs = [coordinator, researcher, analyst]
```

### AgentConfig Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Unique identifier (must be unique across the config list) |
| `prompt` | str \| Callable | System instruction; a callable may take no arguments or a single `settings` argument |
| `tools` | list[Callable] | Available tool functions |
| `sub_agents` | list[str] | Names of agents this one can delegate to |
| `description` | str | Short description for routing (defaults to `""`). On Deep Agents the coordinator's `task` tool lists it; without one the agent is "The <name> agent." |
| `model` | str \| None | A model reference (`anthropic:claude-sonnet-4-6`), a bare model id, or a role (`fast`); without one, the `default` role (see [Models and providers](#models-and-providers)) |
| `model_settings` | ModelSettings \| None | The agent's thinking, `max_tokens`, sampling settings and `extra` (see [Thinking](#thinking)) |
| `mcp_servers` | list[MCPServerConfig] | MCP servers whose tools the agent gets (ADK and Deep Agents) |
| `skills` | list[str] | Agent Skills: a skill folder's absolute path, or a name under `skills_dirs` (ADK and Deep Agents; never looked up in the working directory) |

The agent graph is validated before anything is allocated — ahead of model
discovery and service creation. Duplicate names, unknown `sub_agents`
references, self-references, delegation cycles, a sub-agent shared by two
parents, and **more than one root** raise `AgentGraphError` naming the
offending agents. Exactly one agent may be unreferenced: the runner starts from
a single root, so agents under any other root would never run. Agents are built
in dependency order, so declaration order does not matter. Then every agent's
model is resolved, with the default model, before anything is built: an agent
whose model cannot run fails startup, named in the error.

A callable `prompt` may take **no arguments** (including all-defaulted ones) or
**exactly one argument**, which receives the manager's settings. Other
signatures, `async def` factories, and non-string results raise
`AgentGraphError` naming the agent.

### Turn and lifecycle contracts

A workflow manager runs **one turn at a time**: `process()` and
`resume_with_job_result()` serialize on a turn lock, and
`initialize_services()`/`reinitialize()`/`cleanup()` take it too, so the backend
is never torn down mid-stream. Admission also re-verifies the backend is live
after acquiring the lock, so a turn queued behind a cleanup reinitializes (or
fails cleanly) instead of running against released resources. Run separate
managers for genuine parallelism.

`MessageProcessor.process()` returns a `TurnResult` (`TurnStatus.COMPLETED` /
`FAILED` / `UNAVAILABLE`); a turn is never replayed by the harness, so a
surfaced rate limit fails it explicitly. An `EventType.ERROR` event is
rendered as it arrives — `recoverable=True` is a warning and the stream still
decides the outcome, anything else makes the turn `FAILED` (`delivered`
False). Ctrl+C stops the request the agent is processing: it cancels the CLI's
input handler, which runs every turn, so a stopped turn raises
`asyncio.CancelledError`. Cancelling the caller cancels and awaits the event
consumer before the turn is torn down; the boxes the request's processing
opened end with it, and `Operation cancelled...` says once that the request
was stopped.

The HITL input callback is **context-local**: `set_input_callback()` binds it
for the calling context (and any task started from it), so two consumers of one
manager cannot capture each other's prompts.

`WorkflowController` exposes a derived `WorkflowState`
(`uninitialized`/`initializing`/`ready`/`failed`/`closed`) that never reports
`ready` over an uninitialized manager, and `controller.workflow` raises unless
the state is `ready`. Lifecycle transitions (init, reinitialize, orchestrator
swap, close) are serialized, so nothing is published after `close()`. A failed
in-place reinitialization keeps the manager — it still owns the session service
it preserved — and the next initialization revives it rather than discarding
the conversation.

`SessionRef(app_name, user_id, session_id)` is the conversation identity used by
every session API. All of these are importable from `agentic_cli`.

## Tools

### Creating Tools

Tools are regular Python functions with type hints and docstrings. **All tools return `{"success": bool, ...}` dicts** — never raise exceptions.

```python
from agentic_cli.tools import register_tool, ToolCategory
from agentic_cli.workflow.permissions import Capability, EXEMPT

@register_tool(
    category=ToolCategory.NETWORK,
    capabilities=[Capability("http.read", target="https://db.example.com/search")],
    description="Search the database for matching records.",
)
def search_database(query: str, limit: int = 10) -> dict:
    """Search the database for matching records.

    Args:
        query: Search query string
        limit: Maximum number of results (default: 10)

    Returns:
        Dict with results and count
    """
    results = db.search(query, limit=limit)
    return {"success": True, "results": results, "count": len(results)}
```

**Registration is required for a tool to run.** The permission engine is on by default, and any tool without a capability declaration is denied at call time (fail-closed) on every backend. You can still pass raw callables into `AgentConfig.tools`, but unless they are registered with `@register_tool` they will be blocked — register every tool with `capabilities=` (or `EXEMPT` for pure, side-effect-free functions). Registration also gives the tool registry metadata and tool-summary formatting.

Identity is the *object the registry issued*, not the name: a callable the
registry never issued is denied however it is named, and is left untouched by
tool assembly.

**A tool may declare the services it needs** with `requires=` — e.g.
`@register_tool(..., requires="knowledge_base")`, or a tuple for several. The
manager builds only the services its agents' tools actually declare.

**One name means one tool.** `register_tool()` raises on a name that is already
registered. If a tool genuinely needs a per-backend implementation, declare the
shared contract once and register each backend's version against it:

```python
from agentic_cli.tools import declare_tool, register_tool, ToolCategory
from agentic_cli.workflow.permissions import EXEMPT

# Declare the shared contract once, under a name your application owns.
declare_tool(
    "myapp_scratchpad",
    description="Store a short note for the rest of this turn.",
    capabilities=EXEMPT,
    category=ToolCategory.OTHER,
)

# Then register one implementation per backend against that contract.
@register_tool(
    variant_of="myapp_scratchpad",
    capabilities=EXEMPT,
    category=ToolCategory.OTHER,
)
def myapp_scratchpad_adk(note: str) -> dict:
    """Store a short note for the rest of this turn."""
    return {"success": True, "note": note}
```

Variants share one identity and one permission contract while keeping their own
signature and docstring, and a bare name that maps to several variants raises
rather than silently picking one by import order.

The framework reserves `task`, Deep Agents' delegation tool, whichever backend
runs: registering another tool named `task` raises.

### Capabilities

Tool access is gated by the **permission engine** (see the HITL section below). Each registered tool declares what it touches via `capabilities=`:

- **`Capability("namespace.action", target_arg="...")`** — e.g. `Capability("filesystem.write", target_arg="path")`, `Capability("http.read", target_arg="url")`, `Capability("shell.exec", target_arg="command")`. The engine resolves `target_arg` against the actual tool-call arguments and matches against rules.
- **`Capability("namespace.action", target="...")`** — a fixed target, for a tool that always reaches the same resource, e.g. `Capability("http.read", target="https://export.arxiv.org/api/query")`.
- Filesystem, network (`http.*`) and shell capabilities must name their resource with `target_arg` or `target`. Without one the target would be the wildcard, so registering the tool raises `ValueError`: remembering an approval of it would approve the capability for every resource.
- **`EXEMPT`** — opts the tool out of the engine entirely. Use for tools that need no permission check: pure functions with no side effects, and backend-internal tools (e.g. ADK `transfer_to_agent`, Deep Agents' `task`, backend state tools).

### Built-in Tools

#### Python Execution

Two complementary tools:

- **`execute_python`** — stateless, sandboxed Python via `SafePythonExecutor`. Use for quick calculations.
- **`sandbox_execute`** — stateful, multi-turn Python in a Jupyter kernel. Variables, imports, and DataFrames persist across calls within a session. Shares the workspace filesystem. Network and `pip install` are blocked.

```python
from agentic_cli.tools import execute_python
# Stateless: result = execute_python("import math; math.sqrt(2)")

from agentic_cli.tools.sandbox import sandbox_execute
# Stateful (requires sandbox service):
# sandbox_execute("x = 42", session_id="analysis")
# sandbox_execute("print(x * 2)", session_id="analysis")  # sees x
```

The `/sandbox` CLI command lists and resets sandbox sessions.

#### Docker-isolated sandbox

`sandbox_execute` can run inside a network-isolated Docker container:

1. Ensure Docker (or podman) is installed and running, and pull the image once:
   `docker pull quay.io/jupyter/scipy-notebook:python-3.12`
2. In settings: `sandbox_execute_enabled = true`, `sandbox_backend = "jupyter_docker"`.

Each session runs in its own container with `--network none`, a read-only
rootfs, dropped Linux capabilities, and memory/CPU/PID caps. Stage data as
files via `sandbox_data_mounts` (`"/host/path:name"` → `/workspace/data/name`,
read-only). Live network access, package installs, and S3 are not supported in
this mode.

#### Web Search

Search the web using pluggable backends (Tavily or Brave):

```python
from agentic_cli.tools import web_search

# Or call directly
results = await web_search("Python async programming", max_results=5)
# Returns: {"success": True, "results": [{"title": "...", "url": "...", "snippet": "..."}], ...}
```

Backends auto-select based on available API keys. Set `TAVILY_API_KEY` or `BRAVE_API_KEY`.

#### Web Fetch

Fetch web content and summarize with LLM:

```python
from agentic_cli.tools import web_fetch

result = await web_fetch(
    url="https://example.com/article",
    prompt="Extract the main points from this article",
)
```

Features: URL validation, robots.txt compliance, SSRF protection, content caching, PDF extraction (including arXiv).

Every HTML, text, Markdown, JSON, XML or PDF page `web_fetch` reads is saved as received in `./.{app_name}/fetched/` (a folder git ignores), and the result's `saved_path` names it. Pass `saved_path` to `kb_ingest_file` to add the page to the knowledge base. HTML, text, Markdown, JSON and XML pages are read up to `webfetch_max_download_bytes` (5 MB; PDFs use `webfetch_max_pdf_bytes` instead); the summarizer sees the first `webfetch_max_content_bytes` (100 KB). Saved pages older than `webfetch_saved_max_age_days` (7) are deleted, then the oldest until the folder is under `webfetch_saved_max_mb` (200).

#### ArXiv Search

Search and fetch academic papers:

```python
from agentic_cli.tools import search_arxiv, fetch_arxiv_paper

results = search_arxiv("transformer attention", max_results=10, categories=["cs.CL"])
paper = fetch_arxiv_paper("1706.03762")  # "Attention Is All You Need"
```

The `ArxivSearchSource` (in `tools/arxiv_source.py`) is cached via the service registry and reused across tools. To pull a paper into the KB, use `ingest_arxiv_paper(arxiv_id)` — it composes the arxiv source with the knowledge base.

#### File Operations

Categorized file tools with permission levels:

**READ tools (safe)**

```python
from agentic_cli.tools import read_file, grep, glob, list_dir, diff_compare

read_file("src/main.py", offset=0, limit=100)
grep("def.*async", path="src/", file_pattern="*.py", recursive=True)
glob("**/*.py", path="src/", include_metadata=True)
list_dir("src/", include_hidden=False)
diff_compare(source_a="old.txt", source_b="new.txt")
```

**WRITE tools**

```python
from agentic_cli.tools import write_file, edit_file

write_file("output.txt", content="Hello, World!", create_dirs=True)
edit_file("config.py", old_text="DEBUG = True", new_text="DEBUG = False")
```

Writes are atomic (temp file + rename) via `agentic_cli.file_utils`.

#### Shell Executor

> **Note**: Shell execution is **disabled by default** (`_SHELL_TOOL_ENABLED = False` in `tools/shell/executor.py`) while security safeguards are being validated.

The shell tool provides layered security:
- Input preprocessing (encoding/obfuscation detection)
- Command tokenization and classification
- Path analysis and OS-native sandboxing (macOS seatbelt / Linux namespaces)
- Risk assessment with approval workflows
- Audit logging

```python
from agentic_cli.tools import shell_executor, is_shell_enabled

if is_shell_enabled():
    result = shell_executor("ls -la", working_dir="/project")
```

#### Knowledge Base Tools

Renamed tools and a new concept-pages subsystem:

```python
from agentic_cli.tools import (
    kb_search, kb_ingest_text, kb_ingest_file,
    kb_read, kb_list,
    kb_write_concept, kb_search_concepts,
    KB_READER_TOOLS, KB_WRITER_TOOLS,
)
```

| Tool | Purpose |
|------|---------|
| `kb_search` | Hybrid BM25 + vector search with RRF fusion, filters by source/date; keyword-only without the `kb` extra or with `knowledge_base_use_mock` |
| `kb_ingest_text` | Ingest in-memory text content (no FS or network access) |
| `kb_ingest_file` | Ingest a local PDF, HTML or UTF-8 text file, or a page `web_fetch` saved (`saved_path`, recorded with its URL); HTML goes in as its main text. Declares `filesystem.read(path)` |
| `kb_read` | Return the per-document markdown sidecar (lazy-generated on first read) |
| `kb_list` | List documents, optionally filtered |
| `kb_write_concept` | Agent-authored concept/summary page (slugged, merged on overwrite) |
| `kb_search_concepts` | Search concept pages (title-weighted, case-insensitive) |

The single `kb_ingest` tool was split in 0.5.2 so each entry point declares the right capability — text-only stays under `kb.write`, while file ingestion trips the `filesystem.read` check. For a web page, read it with `web_fetch`, then pass its `saved_path` to `kb_ingest_file`. For arXiv papers, prefer `ingest_arxiv_paper(arxiv_id)`.

Every way into the knowledge base (text, a local file, a saved page, a fetched URL, an arXiv paper) runs through one pipeline, `KnowledgeBase.ingest_text` or `ingest_bytes`: convert, fill in defaults (a value you pass always wins), enrich with one summarizer call, store. The converter (`agentic_cli.memory.kb.convert_document`) handles files: PDFs through pypdf, HTML through trafilatura, which keeps a page's main text and drops menus, sidebars and footers. trafilatura comes with the `kb` extra (`pip install agentic-cli[kb]`), or on its own (`pip install trafilatura`); without it, html2text converts the whole page.

Bundle convenience:

```python
researcher = AgentConfig(name="researcher", prompt=..., tools=KB_WRITER_TOOLS)
reader     = AgentConfig(name="reader",     prompt=..., tools=KB_READER_TOOLS)
```

Direct manager API for embedding in custom flows:

```python
from pathlib import Path
from agentic_cli.memory import EmbeddingConfig
from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType

kb = KnowledgeBaseManager(
    Path("~/.myapp/knowledge_base").expanduser(),
    embedding=EmbeddingConfig(model_name="all-MiniLM-L6-v2"),
    summarizer=my_summarizer,  # optional: anything with `async summarize(content, prompt) -> str`
)
content = "Machine learning is..."
payload = await kb.generate_sidecar_payload(content, title="ML Introduction")  # calls the summarizer
doc = kb.ingest_document(
    content=content,
    title="ML Introduction",
    source_type=SourceType.WEB,
    source_url="https://example.com/ml",
    summary=payload["summary"] or None,
    sidecar_payload=payload,
)
results = kb.search("neural networks", top_k=5)        # hybrid by default
concepts = kb.concepts.search("attention")             # concept pages
await kb.backfill_sidecars()                           # regenerate markdown summaries
```

`KnowledgeBase` (provisional: its name and methods may change in 0.7.0) serves several directories as one, as the workflow does with the project and user knowledge bases:

```python
from pathlib import Path

from agentic_cli.memory.kb import KnowledgeBase

kb = KnowledgeBase.open(
    {"project": Path(".myapp/knowledge_base"), "user": Path("~/.myapp/knowledge_base").expanduser()},
    summarizer=my_summarizer,
)
stored = await kb.ingest_bytes(Path("paper.pdf").read_bytes(), extension=".pdf", default_title="paper")  # ScopedDocument: stored.document, stored.scope
results = kb.search("neural networks", top_k=5)          # both scopes, merged by rank
found = kb.find_document("Attention")                     # ScopedDocument(document, scope) or None
recent = kb.list_documents(title_or_author="vaswani")     # the catalogue, newest first
if found is not None:
    text, truncated = kb.read_text(found, max_chars=10_000)
    sidecar = await kb.sidecar(found)                     # written first when missing
backfilled = await kb.backfill_sidecars()                 # one BackfillResult per scope
```

The workflow exposes it as `workflow.knowledge_base` (and the memory store as `workflow.memory_store`); tools find it under the `knowledge_base` service key.

`ingest_document` is synchronous and never calls the summarizer itself.

The KB also maintains `index.md` and an append-only `ingest_log.md` audit trail. Source-type constants (`arxiv`, `ssrn`, `web`, `internal`, `user`, `local`) live on `SourceType`.

Ingesting a document writes a summary and a sidecar with the workflow's LLM when `knowledge_base_summarize` is on (the default; one call per document). Turn it off in `~/.{app_name}/settings.json` to store the first ~500 characters instead. A knowledge base you build yourself calls the `summarizer=` you pass it in `generate_summary`, `generate_sidecar_payload` and `backfill_sidecars`; `ingest_document` stores the summary and sidecar payload you give it, or a preview. `KnowledgeBase.ingest_text(content, ...)` and `ingest_bytes(data, extension=..., ...)` do the conversion and the summarizer call for you; they raise `EmptyDocument`, `UnsupportedDocument` or `ConversionFailed`, `KeyError` for an unknown `scope=`, and `ValueError` for an unknown `source_type`; errors from storing the document propagate.

The workflow loads the embedding model once per process: both knowledge bases and the memory store share one `EmbeddingService`. A knowledge base you build yourself uses the `embedding_service=` you pass it (FAISS sizes its index from the service), so several knowledge bases can share one model too.

#### Memory Tools

Semantic, lifecycle-aware memory:

```python
from agentic_cli.tools import memory_tools
# save_memory, search_memory, update_memory, delete_memory
```

The store itself is `agentic_cli.memory.MemoryStore`, part of the memory
package that depends on nothing else in agentic-cli:

```python
from pathlib import Path
from agentic_cli.memory import MemoryStore

store = MemoryStore(Path("~/.myapp/memory").expanduser())
store.store("User prefers metric units", tags=["preference"])
```

`MemoryStore` features:
- Embedding-backed semantic search
- Contradiction detection via `store_with_similarity_check`
- `ForgettingPolicy` with `apply_forgetting()` for bounded retention
- Archive filtering and `load_all` with tag/source filters

Each index records the embedding model that built it: a knowledge base in `embeddings/model.json`, the memory store in `embedding_model.json`. After a change of `embedding_model`, the next open embeds the stored text again (a large knowledge base makes that one start slower; the log gives the chunk count and the time taken). An index from 0.6.3 with no record is kept when its dimension matches the model. Delete `embeddings/index.faiss` to force a rebuild.

#### HITL (Human-in-the-Loop)

Tool calls are gated by the **permission engine** (`workflow/permissions/`). Each tool declares a list of capabilities (e.g. `filesystem.write(path=...)`); the engine evaluates them against rules from four sources (builtin defaults, user `~/.{app_name}/settings.json`, project `./.{app_name}/settings.json`, in-memory session). The builtin defaults allow reads inside the project, the knowledge base and memory, and the arXiv tools' two fixed endpoints (the API and `https://arxiv.org/pdf`), and deny writes to system and credential directories and to the app's own config. When no rule matches, the user is prompted with `Allow once` / `Allow for this session` / `Allow always for this project` / `Deny`. A filesystem grant covers the directory it names (or the directory a file is in), never the filesystem root or a directory above your home. A URL grant (`web_fetch`) answered "for this session" or "always" covers that site, `https://host/**`: the exact host, not its other subdomains, and HTTPS only (a plain-HTTP URL, an IP address or a local name keeps an exact-URL grant). "Allow once" approves one URL, and the prompt shows both the URL and the site. To pre-approve a site, add a rule to `~/.{app_name}/settings.json`: `{"permissions": {"allow": [{"capability": "http.read", "target": "https://docs.python.org/**"}]}}`.

**"Allow always" grants are user-owned, not project-owned.** They persist to
`~/.{app_name}/project_grants.json`, keyed by the *resolved project path*, so
they apply to that checkout on this machine only. They are deliberately not
written into `./.{app_name}/settings.json`: a repository could otherwise ship
pre-approved allow-rules that a clone would silently honour. A clone at a
different path therefore starts with no grants, and a repo-committed grant file
is never loaded.

## CLI Commands

Built-in slash commands available in all apps:

| Command | Aliases | Description |
|---------|---------|-------------|
| `/help` | | Show available commands and usage information |
| `/status` | | Show current session, workflow, and context-window status |
| `/clear` | | Clear the screen |
| `/exit` | `/quit` | Exit the application |
| `/settings` | `/set`, `/config` | Interactive settings editor: the default model, the `fast` and `strong` roles' models, thinking effort (saved to the project file) |
| `/models` | | Each provider's listing status and models (`/models <text>` filters, `/models refresh` lists again) |
| `/sandbox` | `/sb` | List / reset stateful sandbox sessions |
| `/jobs` | | List and manage long-running background jobs (`/jobs all`, `/jobs <id>`, `/jobs cancel <id>`, `/jobs clean`) |
| `/resume` | | Resume the agent with results from finished background jobs |
| `/papers` | `/docs` | List knowledge-base documents (filter by source, query, --global) |
| `/sessions` | `/sess` | List saved sessions (and delete with `--delete=<id>`) |

Apps can add more of their own. `research_demo`, for instance, registers
`/memory`, `/files` and `/kb-backfill` on top of the built-ins above.

### Adding Custom Commands

```python
from agentic_cli.cli.commands import Command, CommandCategory

class MyCommand(Command):
    def __init__(self):
        super().__init__(
            name="mycommand",
            description="Do something custom",
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: Any) -> None:
        app.session.add_response(f"Executed with args: {args}")

class MyApp(BaseCLIApp):
    def _register_builtin_commands(self) -> None:
        super()._register_builtin_commands()
        self.command_registry.register(MyCommand())
```

## Events

`WorkflowEvent` types for UI integration:

| EventType | Description |
|-----------|-------------|
| `TEXT` | Final text response |
| `THINKING` | Model reasoning |
| `TOOL_CALL` | Tool invocation |
| `TOOL_RESULT` | Tool result |
| `CODE_EXECUTION` | Code execution result (Gemini on ADK) |
| `EXECUTABLE_CODE` | Code the model intends to run (Gemini on ADK) |
| `FILE_DATA` | Binary/file payload from a tool (ADK) |
| `ERROR` | Error message (`recoverable`, `error_code` metadata) |
| `TASK_PROGRESS` | Task checklist update (drives the task box) |
| `CONTEXT_TRIMMED` | Context trimming event (drop count, remaining tokens), or a summary on Deep Agents (`source="summarization"`) |
| `LLM_REQUEST` / `LLM_RESPONSE` / `LLM_USAGE` | LLM traffic + token accounting |

On Deep Agents every event carries `metadata["agent"]`, the agent that
produced it, and `LLM_USAGE` names that agent's own model.

### Processing Events

```python
async for event in manager.process(message, user_id="user1"):
    if event.type == EventType.TEXT:
        print(event.content)
    elif event.type == EventType.THINKING:
        print(f"[Thinking] {event.content}")
    elif event.type == EventType.TOOL_CALL:
        print(f"Calling: {event.metadata['tool_name']}")
    elif event.type == EventType.TOOL_RESULT:
        print(f"Result: {event.metadata['result']}")
```

### Dual Thinking Boxes + Task Progress

The CLI renders two independent thinking boxes: one for the model's response stream, one for the task checklist:

```
Calling: web_search
--- Tasks: 1/3 ---
Research:
  [✓] Gather data
  [▸] Analyze results
  [ ] Draft report
```

Status icons:
- `[✓]` Completed
- `[▸]` In progress
- `[ ]` Pending
- `[-]` Cancelled

Task state persists across turns within a session.

## Examples

See the `examples/` directory for complete working examples:

**Getting Started**
- `hello_agent.py` — Simple assistant using Google ADK
- `hello_deepagents.py` — Same assistant using Deep Agents orchestration
- `hello_langgraph.py` — Same assistant using LangGraph orchestration (deprecated; removed in 0.8.0)

**Feature Demos**
- `arxiv_demo.py` — ArXiv paper search and analysis
- `fileops_demo.py` — File operation tools (read, write, grep, glob)
- `jobs_demo.py` — Long-running background jobs with a live status-bar monitor; a finished job resumes the agent (ADK, or Deep Agents with `AGENTIC_ORCHESTRATOR=deepagents`)
- `memory_demo.py` — Memory persistence system (save/search/update/delete)
- `shell_demo.py` — Shell security pattern detection
- `webfetch_demo.py` — Web fetching and summarization
- `websearch_demo.py` — Web search with multiple backends

**Full Applications**
- `research_demo/` — Full-featured research assistant with KB ingest + concept pages, semantic memory, sandbox execution, and durable sessions. Installed as the `research-demo` console script.

Run examples:

```bash
export GOOGLE_API_KEY="your-key"
python examples/hello_agent.py

# Feature demos
python examples/fileops_demo.py
python examples/shell_demo.py

# Or with Deep Agents
pip install 'agentic-cli[deepagents]'
python examples/hello_deepagents.py

# Jobs demo on Deep Agents
AGENTIC_ORCHESTRATOR=deepagents python examples/jobs_demo.py

# Or with LangGraph (deprecated)
pip install agentic-cli[langgraph]
python examples/hello_langgraph.py

# Research demo (full features)
research-demo                      # console script, installed with the package
python -m research_demo            # equivalent module invocation
python -m research_demo --session my-research   # resume a durable session by id

# From a checkout without installing, run it from the repository root:
python -m examples.research_demo
```

Sessions are durable by default: every run gets a session id, `--session <id>`
resumes a stored one, and `/sessions` lists or deletes them. The demo adds
`/memory`, `/files` and `/kb-backfill` to the built-in commands.

## Development

### Running Tests

```bash
conda run -n agenticcli python -m pytest tests/ -v

# With coverage
conda run -n agenticcli python -m pytest tests/ -v --cov=agentic_cli
```

### Project Structure

```
agentic-cli/
├── src/agentic_cli/
│   ├── __init__.py               # Package exports, lazy imports
│   ├── config.py                 # BaseSettings, SettingsContext
│   ├── settings_mixins.py        # AppSettingsMixin, CLISettingsMixin
│   ├── settings_persistence.py   # SettingsPersistence
│   ├── constants.py              # Shared constants (truncation, limits)
│   ├── file_utils.py             # Atomic write helpers
│   ├── logging.py                # Structlog configuration
│   ├── cli/
│   │   ├── app.py                # BaseCLIApp
│   │   ├── commands.py           # Command, CommandRegistry
│   │   ├── builtin_commands.py   # help/status/clear/exit/sandbox/jobs/resume/papers/sessions/models
│   │   ├── settings_command.py   # /settings
│   │   ├── settings_introspection.py
│   │   ├── workflow_controller.py
│   │   ├── message_processor.py
│   │   └── usage_tracker.py      # Token / context-window accounting
│   ├── workflow/
│   │   ├── base_manager.py       # BaseWorkflowManager (abstract)
│   │   ├── factory.py            # create_workflow_manager_from_settings
│   │   ├── service_registry.py   # Unified service ContextVar for tools
│   │   ├── events.py             # WorkflowEvent, EventType
│   │   ├── config.py             # AgentConfig
│   │   ├── settings.py           # WorkflowSettingsMixin
│   │   ├── retry.py              # Exponential-backoff retry
│   │   ├── tool_summaries.py     # Tool result one-liner summaries
│   │   ├── memory_services.py    # Settings → agentic_cli.memory arguments
│   │   ├── model_services.py     # Settings → agentic_cli.models arguments; build_catalog
│   │   ├── model_settings.py     # ModelSettings, ThinkingSettings (per agent)
│   │   ├── model_wire.py         # What both adapters send to OpenAI-compatible servers
│   │   ├── langchain_models.py   # build_chat_model: a resolved model → LangChain chat model
│   │   ├── checkpointers.py      # open_checkpointer: memory / sqlite / Postgres (Deep Agents, LangGraph)
│   │   ├── llm_log.py            # LLMLogWriter: the raw_llm_logging JSONL file, redact_secrets
│   │   ├── tool_errors.py        # A raising tool's error result (ADK and Deep Agents)
│   │   ├── sessions.py           # SessionRef, the active turn, the active tool call
│   │   ├── mcp.py                # MCPServerConfig → ADK toolset / langchain-mcp-adapters connection
│   │   ├── adk/
│   │   │   ├── manager.py                # GoogleADKWorkflowManager
│   │   │   ├── model_factory.py          # build_adk_model: a resolved model → ADK model + config
│   │   │   ├── openai_compatible.py      # OpenAICompatibleLlm (OpenRouter, endpoints)
│   │   │   ├── anthropic_llm.py          # DirectAnthropicLlm (bare claude-* in ADK YAML)
│   │   │   ├── event_processor.py
│   │   │   ├── plugins.py                # LLMLoggingPlugin
│   │   │   ├── permission_plugin.py      # PermissionPlugin (capability gating)
│   │   │   └── task_progress_plugin.py
│   │   ├── deepagents/                   # The Deep Agents backend (deepagents extra)
│   │   │   ├── manager.py                # DeepAgentsWorkflowManager
│   │   │   ├── agent_builder.py          # One create_agent graph per AgentConfig; task delegation
│   │   │   ├── middleware.py             # Permission gate, tool errors, active tool call, foreign reasoning
│   │   │   ├── tools.py                  # Framework tools → LangChain tools
│   │   │   ├── event_processor.py        # The stream → WorkflowEvents; the run callback
│   │   │   ├── sessions.py               # Threads keyed by SessionRef; the session hooks
│   │   │   ├── summarization.py          # Thresholds from the input budget; no archive
│   │   │   ├── mcp_tools.py              # MCPSessions: one session per agent and server
│   │   │   └── state.py                  # AgentCliState: plan and tasks
│   │   ├── langgraph/                    # Deprecated; removed in 0.8.0
│   │   │   ├── manager.py                # LangGraphWorkflowManager
│   │   │   ├── graph_builder.py
│   │   │   ├── permission_wrap.py        # wrap_tool_for_permission
│   │   │   ├── state.py
│   │   │   └── persistence/              # Checkpointers and stores
│   │   └── permissions/                  # Framework-independent permission engine
│   │       ├── capabilities.py           # Capability, ResolvedCapability, EXEMPT
│   │       ├── rules.py                  # Rule, Effect, RuleSource, CheckResult, AskScope
│   │       ├── matchers.py               # Path/URL/Shell/StringGlob matchers
│   │       ├── store.py                  # PermissionContext, BUILTIN_RULES, JSON load/save
│   │       ├── prompt.py                 # build_request + parse_response
│   │       └── engine.py                 # PermissionEngine
│   ├── tools/
│   │   ├── registry.py           # ToolRegistry, ToolCategory, register_tool
│   │   ├── factories.py          # Backend-aware tool factories
│   │   ├── executor.py           # SafePythonExecutor
│   │   ├── arxiv_tools.py        # search_arxiv, fetch_arxiv_paper, ingest_arxiv_paper
│   │   ├── arxiv_source.py       # ArxivSearchSource (service-registered)
│   │   ├── search_sources.py     # SearchSource, SearchSourceResult
│   │   ├── execution_tools.py    # execute_python
│   │   ├── interaction_tools.py  # ask_clarification
│   │   ├── knowledge_tools.py    # kb_search / kb_ingest_text / kb_ingest_file /
│   │   │                         #   kb_read / kb_list /
│   │   │                         #   kb_write_concept / kb_search_concepts
│   │   ├── file_read.py          # read_file, diff_compare
│   │   ├── file_write.py         # write_file, edit_file (atomic)
│   │   ├── grep_tool.py          # grep
│   │   ├── glob_tool.py          # glob, list_dir
│   │   ├── search.py             # web_search (Tavily / Brave)
│   │   ├── webfetch_tool.py      # web_fetch (orchestrator)
│   │   ├── pdf_utils.py          # PDF text extraction helpers
│   │   ├── memory_tools.py       # save/search/update/delete (store: agentic_cli.memory)
│   │   ├── _core/                # Shared planning/task logic, backend-neutral declarations
│   │   │   ├── planning.py
│   │   │   ├── tasks.py
│   │   │   ├── state_tools.py    # save_plan / get_plan / save_tasks / get_tasks, declared
│   │   │   └── delegation.py     # task, declared (Deep Agents' delegation tool)
│   │   ├── adk/state_tools.py       # ADK-native planning/task tools
│   │   ├── langgraph/state_tools.py # LangGraph-library planning/task tools (Deep Agents, LangGraph)
│   │   ├── skills/               # Agent Skills: SkillStore, ADK toolset, LangChain tools (langchain_tools.py)
│   │   ├── jobs/                 # Long-running jobs: JobManager, backends, job_* tools, resume.py
│   │   ├── sandbox/
│   │   │   ├── __init__.py       # sandbox_execute tool
│   │   │   ├── manager.py        # SandboxManager + session lifecycle
│   │   │   ├── models.py
│   │   │   └── backends/         # Jupyter local backend
│   │   ├── shell/                # Layered shell executor (disabled by default)
│   │   │   ├── executor.py
│   │   │   ├── os_sandbox/       # macOS seatbelt / Linux namespace sandboxes
│   │   │   └── ...               # tokenizer, classifier, sandbox, audit
│   │   └── webfetch/             # Fetcher, converter, validator, robots, summarizer, saved pages
│   ├── memory/                   # Knowledge base + memory store; imports nothing else from agentic_cli
│   │   ├── __init__.py           # MemoryStore, MemoryItem, ForgettingPolicy, EmbeddingService, EmbeddingConfig, MockEmbeddingService
│   │   ├── _core/                # Shared, private: embeddings.py, mock_embeddings.py, io.py, log.py
│   │   ├── store.py              # MemoryStore
│   │   └── kb/                   # Knowledge base (agentic_cli.memory.kb)
│   │       ├── manager.py                # KnowledgeBaseManager(base_dir, embedding=, summarizer=), Summarizer
│   │       ├── knowledge_base.py         # KnowledgeBase (provisional): the scopes as one, and the ingestion pipeline; ScopedDocument, BackfillResult
│   │       ├── enrich.py                 # The summary/sidecar prompts and parser: one summarizer call per document
│   │       ├── models.py                 # Document, SearchResult, SourceType
│   │       ├── convert.py                # convert_document: trafilatura / html2text, pypdf, text
│   │       ├── vector_store.py, bm25_index.py (+ _bm25_backends.py), concepts.py, sidecar.py
│   │       └── _mock_vector_store.py, _mock_bm25.py
│   ├── models/                   # Model knowledge (provisional); no provider SDK
│   │   ├── refs.py                   # provider:model references, bare ids, role names
│   │   ├── endpoints.py              # ProviderKind, EndpointSpec, Endpoint, build_endpoints
│   │   ├── records.py                # ModelRecord, ThinkingRule
│   │   ├── table.py, data/table.json # The shipped table: families, tiers, numbers, fallbacks
│   │   ├── roles.py                  # RoleMap: roles → references
│   │   ├── resolve.py                # resolve_model → ResolvedModel
│   │   ├── catalog.py                # ModelCatalog: listings, validation, the listing cache
│   │   └── discovery/                # One lister per provider, over httpx
│   └── persistence/              # (sessions are persisted natively by each
│                                 #  orchestrator's store; the legacy JSON
│                                 #  SessionPersistence layer was removed)
├── examples/
│   ├── hello_agent.py            # Basic ADK example
│   ├── hello_deepagents.py       # Basic Deep Agents example
│   ├── hello_langgraph.py        # Basic LangGraph example (deprecated)
│   ├── *_demo.py                 # Feature demonstration scripts
│   └── research_demo/            # Full-featured example (pip-installable)
└── tests/
    ├── conftest.py               # MockContext and shared fixtures
    ├── tools/                    # Tool-specific tests (incl. sandbox)
    ├── workflow/                 # Workflow unit tests
    └── integration/              # ADK, Deep Agents and LangGraph pipeline tests
```

## License

MIT
