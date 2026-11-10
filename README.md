# Agentic CLI

A framework for building domain-specific agentic CLI applications powered by LLM agents.

## Overview

Agentic CLI provides the core infrastructure for building interactive CLI applications that leverage LLM agents for complex tasks. It offers:

- **Pluggable Orchestration**: Choose Google ADK (the default) or Deep Agents for agent workflows; both run every model provider. LangGraph is deprecated and will be removed in 0.8.0
- **Any Model, Per Agent**: Gemini, Claude and GPT built in, and any OpenAI-compatible host configured by name (OpenRouter, Together, Fireworks, your own vLLM), mixed in one agent tree; models named by reference or role, with one set of thinking levels
- **Rich Terminal UI**: Dual thinking boxes, markdown rendering, and streaming responses via `thinking-prompt`
- **Declarative Agents**: Define agents with simple configuration objects
- **Native Tool Architecture**: Backend-specific tool factories for each backend, with automatic HITL confirmation for dangerous tools
- **Built-in Tools**: Python execution, stateful sandbox, file operations, web search, web fetch, arXiv search
- **Knowledge Base**: Semantic + BM25 hybrid search with RRF fusion, per-document markdown sidecars, and agent-authored concept pages
- **Memory**: markdown files in a user and a project scope, keyword search, history, proposals for review
- **Durable Sessions**: Conversations persist continuously by default (SQLite; PostgreSQL or ephemeral memory optional) and resume across CLI restarts: `--continue` (the newest), `--resume <id>`, or `/resume` in the app for a picker
- **Context Window Management**: Summarization near the model's input budget on ADK and Deep Agents, requests trimmed to fit, a provider's overflow recovered or explained, and token-usage visibility
- **A Short Model List**: the app ships the models its pickers offer, and the user and the project add to them; each provider's live listing, cached for a day, checks them, and `/models` shows them and searches the full listings
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
│  - Session save/resume, usage per agent, context budgets            │
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
- A key for at least one model provider (Google, Anthropic or OpenAI, or a host such as OpenRouter configured in `model_providers`), or an OpenAI-compatible server of your own (see [Models and providers](#models-and-providers))

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

### Resuming a conversation

Conversations are saved as they run (`session_store`: sqlite by default).
Run the app through `BaseCLIApp.main()` and every app takes the same flags:

```bash
my-app                  # a new conversation; at exit, once a turn ran, its id is named
my-app --continue       # the newest saved conversation no other run has open (-c)
my-app --resume <id>    # that one (-r); an id that is not saved exits with status 1
```

Inside the app, `/resume` opens a picker (Enter opens a conversation, Esc
keeps the current one) and `/resume <id>` opens one by id; `/sessions` lists them with their titles (the first prompt), turns and
last update. A resumed conversation shows its last 10 messages again. The
model and the settings stay this run's; "Allow for this session" grants,
sandbox kernels and the task list are not restored. One conversation is
open in one process at a time: two would lose a turn, so a conversation
another run has open is refused ("Conversation '<id>' is open in another
process (1234)."; `--continue` passes over it to the newest one free among
the 50 newest, and says so). A run holds the conversation it was given
(`--resume`, `--continue`, `/resume`) from the start, also when the workflow
fails to start, and a new one from its first turn: no store lists it before
then, and a run in which no turn runs leaves no lock file (a line typed
while the workflow is down takes the hold before its retried start). Fixed with
`/settings` (another orchestrator, while nothing of the conversation is
stored), a failed start runs it, and holds it, where the next start runs. The session store is the run's: a change of
`session_store` takes effect at the next start.
A conversation belongs to the backend it began on (`orchestrator`, or
`/new <backend>`) for its whole life, and each backend keeps its own store.
`--continue`, `--resume <id>`, `/resume` and `/sessions` read the
conversations of every installed backend, and one another backend keeps
opens there: the app starts that backend ("Resumed on deepagents: …"; at
startup it builds that backend in the first place), carrying this run's
"Allow for this session" grants, background jobs, sandbox kernels and
knowledge bases over to it; a backend that fails to start changes nothing.
`/sessions` names each conversation's backend. An id two backends keep (an
earlier version switched the backend under an open conversation) is two
conversations: `/resume <id>` opens the running backend's and names the
other, the picker opens the row picked, and `/sessions --delete=<id>
--backend=<name>` deletes one.
A line typed before `--continue` or `--resume` has opened its
conversation waits for it, then runs there. An app switches with
`await app.resume_session(id)` or `await app.start_new_session()`; setting
`app.session_id` is deprecated.

`main()` parses the flags, builds the app with `from_args` (by default
`cls()`), runs it and exits with its `exit_status`. Point the console
script at it, and add flags of your own with two classmethods (`-c`, `-r`
and `-h` are taken):

```python
import argparse

class MyApp(BaseCLIApp):
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose  # before super().__init__(): the hooks run in it
        super().__init__(
            app_info=AppInfo(name="My App", version="0.1.0"),
            agent_configs=AGENTS,
            settings=BaseSettings(),
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--verbose", action="store_true")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "MyApp":
        return cls(verbose=args.verbose)

# pyproject.toml: [project.scripts] my-app = "my_app.cli:MyApp.main"
if __name__ == "__main__":
    MyApp.main()
```

An app built and run directly (`asyncio.run(app.run())`) starts a new
conversation, and is pointed at `/resume <id>` at exit; set
`app.session_start = SessionStart("continue")` (or `"resume"` with an id;
`agentic_cli.cli.session_args`) before `run()` to start otherwise,
and exit with `app.exit_status` (1 when `"resume"` names a conversation that
is not saved: `run()` then returns before the prompt).

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

The hooks run while the app is built: `get_theme()` as the session is built,
and `register_commands()` last, once the session, the settings and the
workflow controller exist. An attribute of your own that a hook reads is set
before `super().__init__()`, as `self.verbose` is above. A command
registered later (`app.command_registry.register(...)`) completes and is
listed by `/help` too.

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
    model="google:gemini-3.8-flash",  # Optional: the default model otherwise
)
```

ADK integrations:
- **Model factory** — `workflow.adk.model_factory.build_adk_model` builds each
  agent's model from its resolved model: ADK's `Gemini`, `AnthropicLlm` and
  `OpenAIResponsesLlm`, and `OpenAICompatibleLlm` for the providers in
  `model_providers` (OpenRouter and OpenAI-compatible hosts), which shows
  their reasoning text
- **Delegation** — a sub-agent runs in ADK's `single_turn` mode: its parent
  calls it as a tool named after it and gets its answer back in the same turn
  (see [Sub-agent context](#sub-agent-context))
- **Summarizing** — ADK's `EventsCompactionConfig` with the framework's
  summarizer: the main agent's conversation is summarized on its own model
  once a prompt reaches the trigger Deep Agents uses (the smaller of
  `context_summarize_at` of the input budget and `context_max_tokens`),
  keeping the current turn and the newest 12% of that word for word; a
  sub-agent never summarizes the shared session. The fit guard leaves the
  oldest turns out of a request still over its agent's budget, and a
  request the provider refuses as too long is summarized or trimmed and
  retried once, then ends in one clear error (see
  [Context management](#context-management)). A native ADK YAML app gets
  none of it
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
    model="anthropic:claude-sonnet-5-5",  # Optional: the default model otherwise
)
```

What it supports:
- **Delegation through `task`**: a coordinator (an agent with `sub_agents`)
  gets Deep Agents' `task` tool, which lists each sub-agent as
  `- name: description`, so give sub-agents real descriptions. A sub-agent
  sees the brief the coordinator writes and the coordinator's plan and tasks;
  with `context="inherit"` it is a fork that also continues the
  coordinator's conversation (see [Sub-agent context](#sub-agent-context)).
  Its last answer is the call's result, and the coordinator answers the user.
  Two sub-agents can run at once. `task` is a reserved tool name.
- **The permission gate on every call**: `PermissionMiddleware`, in every
  agent's stack, identifies each tool by the object the registry issued and
  asks through the input callback, for sub-agents' calls too. A tool that
  raises fails its call with `{"success": false, "error": "..."}`, as on ADK,
  and the turn goes on.
- **Sessions**: `session_store` picks a LangGraph checkpointer: sqlite (the
  default) in `{workspace}/sessions/checkpoints.db`, next to ADK's
  `sessions.db`, Postgres at `postgres_uri`, or memory. Each conversation is
  the thread of its whole `SessionRef`, so two users never share one.
- **Context**: Deep Agents' summarization runs in every agent, on its own
  model, once a prompt reaches the smaller of `context_summarize_at` (85%) of
  the model's input budget (95% of its context window, less its
  `max_tokens`) and `context_max_tokens` (200,000 tokens), keeping the newest
  12% of that word for word. A model whose window is unknown counts as
  128,000 tokens. A request the provider refuses as too long is summarized
  and retried once; a conversation that still does not fit ends the turn
  with one error naming the model. The checkpoint keeps every message.
- **Prompt caching** on Claude, as on ADK (see [Prompt caching](#prompt-caching)).
- **MCP tools, Agent Skills, raw LLM logging and job resume**, from the same
  `AgentConfig.mcp_servers`, `AgentConfig.skills`, `raw_llm_logging` and
  `job_auto_resume` (or `/jobs resume`) as on ADK. An MCP tool is gated under the
  `mcp` capability, with its name as the target. A finished job reaches the
  root agent as a message of its own, which opens with `[Automatic job
  notification — not written by the user]`, in the conversation that started
  it, as on ADK.

How it differs from ADK:
- **The delegation tool's name**: a sub-agent is delegated to and returns
  on both backends, but the coordinator calls `task` on Deep Agents and a
  tool named after the sub-agent on ADK (see
  [Sub-agent context](#sub-agent-context)).
- **No Gemini explicit caches**: Gemini runs on its implicit caching, which is
  automatic and bills no storage.
- **No code-execution events**: `CODE_EXECUTION`, `EXECUTABLE_CODE` and
  `FILE_DATA` come from Gemini's code execution on ADK.
- **A sub-agent's failure ends the turn** with one error; on ADK the
  coordinator carries on, and the user gets a warning.
- **Its own sessions**: neither backend reads the other's, so a
  conversation stays on the backend it began on; `/resume`, `--continue`
  and `--resume` open it there, whichever backend runs. `orchestrator` is the
  backend of new conversations: changed in `/settings`, it starts a new
  conversation on the new backend at once when the open one holds nothing
  yet (no turn ran in it), and otherwise applies to the next `/new` ("(applies
  to the next /new)"); `/new deepagents` (or `adk`, `langgraph`) starts one
  on the backend named. A model switch never changes the backend.

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
  `ChatOpenAI` on the Responses API, `ChatOpenRouter` for an `openrouter`
  provider, and `OpenAICompatibleChat` for an OpenAI-compatible one
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
| Model providers | Gemini, Claude, GPT (Responses API), and OpenRouter and OpenAI-compatible hosts (`model_providers`) | The same | The same |
| Delegation | `<sub-agent>(request=...)`: the sub-agent answers its parent, the coordinator answers the user | `task`: the same | Routing between agent nodes |
| Sub-agent context | `fresh` or `inherit` | `fresh` or `inherit` (a fork) | Ignored |
| State persistence | SQLite (default), PostgreSQL, or memory | The same, as a LangGraph checkpointer | Memory, PostgreSQL, or SQLite |
| Thinking support | Every model with a thinking knob, one set of levels | The same | The same |
| Prompt caching | Claude (three breakpoints) and Gemini (server-side caches) | Claude (three breakpoints) | Claude on Anthropic's API (one breakpoint) |
| Context | The main agent's conversation summarized at `context_summarize_at` of the input budget, or `context_max_tokens`; requests over the budget trimmed; one retry on an overflow | Every agent summarized at the same point; one retry on an overflow | None |
| Tool result cap | `tool_result_max_chars` | `tool_result_max_chars` | None |
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
*cwd-relative* `env_file`, the class's or one given to the constructor
(`_env_file`). A cloned repository can ship both, so both are
filtered down to an explicit allowlist of benign keys (model and behaviour
settings, timeouts, sandbox *resource* limits, `session_store`, log verbosity).
Security-relevant keys set from an untrusted source — executor backend, sandbox
image/mounts/user, OS-sandbox policy, `skills_dirs`,
`workspace_dir`, `raw_llm_logging`, permission rules, `model_providers` (the
addresses a key is sent to), and any credential — are dropped with a logged
warning rather than applied. A key that names no setting at all (one an
older version had, say) is dropped silently.

**Consequence:** put API keys in real environment variables or an
absolute/user-level file. A key placed in a cwd `.env` is deliberately ignored.

`app_name` names the files above, and the ones settings are saved to, so only
the constructor or the settings class sets it: an `app_name` in the environment
(`AGENTIC_APP_NAME`, or your prefix's), in either settings file or in an env
file is ignored, with a warning (`settings_app_name_ignored`). The one
exception is the app's own: one that enables pydantic-settings'
`cli_parse_args` gets a command-line source above all of these, which can set
it.

These sources and their filtering come with the settings mixins
(`WorkflowSettingsMixin`, `AppSettingsMixin`): a class composed from them over
pydantic-settings' own `BaseSettings` reads the same sources, in the same
order, filtered the same way. The mixins must come before pydantic-settings'
class, as in `BaseSettings`: a settings class that lists that class first, or
between the mixins, is refused with a `TypeError` when it is defined (its
sources would be pydantic-settings' own), unless it defines
`settings_customise_sources` itself. Its configured providers' keys are read
from the environment and the absolute entries of its class's `env_file`.
pydantic-settings' `secrets_dir` is not read by any of these classes; one
configured is reported (`settings_secrets_dir_ignored`).

### Key Settings

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| `google_api_key` | `GOOGLE_API_KEY` | None | Google API key for Gemini |
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | None | Anthropic API key for Claude |
| `openai_api_key` | `OPENAI_API_KEY` | None | OpenAI API key for GPT |
| `default_model` | `AGENTIC_DEFAULT_MODEL` | Auto | The default model: a `provider:model` reference or a model id |
| `model_roles` | `AGENTIC_MODEL_ROLES` | {} | A model per role: `fast`, `strong`, or an app's own |
| `models` | `AGENTIC_MODELS` | The app's | The models the pickers offer, by `provider:model`, with optional details; added to the app's (see [Models and providers](#models-and-providers)) |
| `model_providers` | `AGENTIC_MODEL_PROVIDERS` | The app's | OpenRouter and OpenAI-compatible hosts by name (never from a project file) |
| `thinking_effort` | `AGENTIC_THINKING_EFFORT` | "medium" | Thinking level: none, low, medium, high |
| `prompt_caching_enabled` | `AGENTIC_PROMPT_CACHING_ENABLED` | true | Prompt caching: Claude, and Gemini on ADK |
| `orchestrator` | `AGENTIC_ORCHESTRATOR` | "adk" | The backend new conversations run on: `adk`, `deepagents`, or `langgraph` (deprecated); a conversation keeps its own |
| `session_store` | `AGENTIC_SESSION_STORE` | "sqlite" | Where conversations are kept: `sqlite`, `postgres`, or `memory` |
| `context_summarize_at` | `AGENTIC_CONTEXT_SUMMARIZE_AT` | 0.85 | Summarize the earlier conversation at this fraction of the model's input budget (0.5 to 0.95) |
| `context_max_tokens` | `AGENTIC_CONTEXT_MAX_TOKENS` | 200000 | Or once a prompt reaches this many tokens, if that comes first (0: no ceiling; otherwise at least 16000) |
| `tool_result_max_chars` | `AGENTIC_TOOL_RESULT_MAX_CHARS` | 50000 | A longer tool result is cut, its full output saved in `./.{app_name}/tool-output/` (at least 2000; see [Context management](#context-management)) |
| `memory_index_enabled` | `AGENTIC_MEMORY_INDEX_ENABLED` | true | Send the main agent an index of saved memories at the start of each conversation (see [Memory Tools](#memory-tools)) |
| `memory_index_max_bytes` | `AGENTIC_MEMORY_INDEX_MAX_BYTES` | 25000 | The memory index's size cap in bytes (1000 to 100000; at most 200 lines, and a tenth of the main agent's input budget) |
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

Settings are **read** by constructing the settings class: `MySettings()`, or
`get_settings()` / `reload_settings()` for the framework's own instance. The
constructor, the environment, the project file's allowlisted keys, the user
file and the env files (see [Precedence](#precedence-and-the-trust-boundary))
are its sources. Two readers take only what they need from the same files: the
permission engine reads the project file's deny rules, and a configured
provider's key is looked up in the environment and the absolute env files.
Settings are **written** one way:
`/settings` calls `BaseCLIApp.save_settings(keys=...)`, which calls
`SettingsPersistence(app_name).save(settings, keys=...)` (atomic writes). A save
writes only the keys it names, as they stand, each into the file the loader
reads it from: allowlisted keys into `./.{app_name}/settings.json`, the others
into `~/.{app_name}/settings.json`; a credential is never written. Nothing
saves the whole settings object, which would freeze values that came from the
environment or the app's defaults.

`config.set_settings(settings)` sets the settings a script uses when it calls
tools outside a workflow manager (`get_settings()` returns them there); a
manager's own turns use the settings it was built with.

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

A Save writes the value picked to the user file, even one equal to the
default: what an app starts on can be set by the app itself (`model_post_init`,
a validator, an env file below the user file), so a save cannot tell it from
the class default and never removes a key. To follow the app again, delete the
key from the file. `examples/research_demo` lists both.

- **The theme** switches at once, through `get_theme(name)`. thinking-prompt
  repaints: it clears the screen and the terminal's scrollback, shell output
  from before the session included, and prints the conversation again in the
  new theme. Messages and markdown take the new colors; Rich panels and tables
  (`/help`'s, for one) keep the colors they were printed in. Under `auto`,
  thinking-prompt picks the theme: `mono` when `NO_COLOR` is set, `light` or
  `dark` from `COLORFGBG` when the terminal sets it, and `terminal` (the
  terminal's own palette) otherwise; agentic-cli's dark colors do not apply
  there. `NO_COLOR` draws in mono whatever the theme.
- **The placement** applies from the next dialog agentic-cli opens: `/settings`,
  the `/resume` picker, `/memory`'s dialogs and the tools' questions,
  permission prompts included. An app's own dialogs keep the placement the
  session started with, unless they pass
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
the workflow starts. Three providers are built in: Google (Gemini), Anthropic
(Claude) and OpenAI (GPT). Any other host is configured by name in
`model_providers`: OpenRouter, Together, Fireworks, or a vLLM server of your
own. Every backend runs every provider, and one agent tree can mix them.

The model pickers offer a short list, `models`: the app ships it, and the user
and the project add to it. What agentic-cli knows about models lives in
`agentic_cli.models` (provisional in 0.7.0: its names may still change): a
table of model families shipped with the package, merged with each provider's
live model list and with what the `models` entries say.

### References

A model is named the same way in `models`, `default_model`, `model_roles`,
`AgentConfig.model`, `Manager(model=...)` and agent YAML:

| You write | It means |
|-----------|----------|
| `anthropic:claude-sonnet-5-5` | `provider:model`. The provider is `google`, `anthropic`, `openai`, or the name of one in `model_providers` |
| `openrouter:qwen/qwen3.7-plus` | A configured provider's model, under the id its host gives it |
| `openrouter:vendor/model:free` | Split on the first colon only, so a model id keeps its own colons |
| `gemini-3.8-flash`, `claude-sonnet-5-5`, `gpt-6.1-sol`, `o3` | A bare `gemini-`, `claude-`, `gpt-` or `o<digit>` id names its provider (not in `models`, whose keys are always `provider:model`) |
| `fast` | Any other bare name is a role (see [Roles](#roles)). A role name has no digits, so a bare `qwen3-32b` is an error that asks for its provider (`local:qwen3-32b`) |

### Built-in providers and their keys

| Provider | Setting | Environment variable |
|----------|---------|----------------------|
| Google (the Gemini API) | `google_api_key` | `GOOGLE_API_KEY` |
| Anthropic | `anthropic_api_key` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai_api_key` | `OPENAI_API_KEY` |

Keys are never saved and never shown (not in `repr`, not by `/models`), and a
project settings file or a cwd `.env` cannot set them: put them in the
environment or a user-level dotenv file (see
[Precedence and the trust boundary](#precedence-and-the-trust-boundary)). Each
model's client is given its key. Google, Anthropic and OpenAI models go to
their SDK's default host; Vertex AI is not supported.

### Configured providers: `model_providers`

Every other provider is an entry of `model_providers`, by name. A user defines
them in `~/.my_app/settings.json`:

```json
{
  "model_providers": {
    "openrouter": {
      "kind": "openrouter",
      "base_url": "https://openrouter.ai/api/v1",
      "api_key_env": "OPENROUTER_API_KEY"
    },
    "together": {
      "base_url": "https://api.together.xyz/v1",
      "api_key_env": "TOGETHER_API_KEY"
    },
    "fireworks": {
      "base_url": "https://api.fireworks.ai/inference/v1",
      "api_key_env": "FIREWORKS_API_KEY"
    },
    "local": {
      "base_url": "http://localhost:8000/v1"
    }
  }
}
```

| Field | Default | Meaning |
|-------|---------|---------|
| `kind` | `"openai_compatible"` | The API it speaks: `openai_compatible` (Together, Fireworks, vLLM and the like) or `openrouter` |
| `base_url` | required | `http` or `https`, a host, an optional port and path; no user name, password, query or fragment |
| `api_key_env` | none | The environment variable holding the key. Without one, the client sends the placeholder key `EMPTY` (never `OPENAI_API_KEY`), as a server without authentication expects. Required for `kind: openrouter` |
| `list_models` | `true` | `false`: the provider is never listed, and every model id is accepted (see below) |

- **A name** is lowercase letters, digits, `_` and `-`, starting with a
  letter, and cannot be `google`, `anthropic` or `openai`. `openrouter` is an
  ordinary name: what makes a provider OpenRouter is `kind: openrouter`, under
  any name. That kind adds OpenRouter's model listing (limits, image input,
  tools, whether the model reasons), the `HTTP-Referer` and `X-Title` headers
  naming the app, and `reasoning.effort` (`none` to `xhigh`) for a model the
  listing says reasons.
- **Where it is defined**: the app's settings class, the user file or the
  environment (`AGENTIC_MODEL_PROVIDERS`, or your app's prefix, as JSON). A
  provider holds the address its key is sent to, so a project settings file,
  which a cloned repository can ship, or a cwd `.env` that sets
  `model_providers` is ignored with a warning. A user's entry adds a
  provider, or overrides fields of the app's provider of the same name one by
  one (only its `api_key_env`, say).
- **Its key** is read from the variable `api_key_env` names: in the process
  environment, then in the app's user-level env files (each `env_file` of the
  settings class that is absolute once `~` is expanded, such as the research
  demo's `~/.research_demo/.env`), the environment winning. A cwd `.env`, a
  project file or a `models` entry never supplies it, and it is never stored.
  When the variable is not set, the provider cannot be called: its `models`
  entries are not offered, and a model that runs on it fails at startup,
  naming the variable and where it is read.
- **Together and Fireworks** speak the OpenAI API, so they need no `kind`.
  Once the key is set, `/models together:` lists every Together model (a
  search matches the `provider:model` reference): copy the exact id into
  `models` as `together:<id>`.
- **A local vLLM server** names no key here. vLLM lists the model it serves,
  with its window (`max_model_len`). A server without a `/models` route, or
  one whose listing leaves out models it serves, takes `"list_models": false`:
  it is never listed, any id is accepted, and a model's facts come from its
  `models` entry and the table, so give its `context_window` (an unknown
  window counts as 128,000 tokens).
- `/settings` never shows `model_providers`, and no save writes it: it is
  edited by hand.

### The list: `models`

`models` maps a `provider:model` reference to what is known about that model.
Each key is a model the model pickers offer; every field is optional:

```json
{
  "models": {
    "anthropic:claude-sonnet-5-5": {},
    "google:gemini-3.8-flash": {},
    "openrouter:qwen/qwen3.7-plus": {},
    "openrouter:deepseek/deepseek-v4-pro": {},
    "local:Qwen/Qwen3-32B": {"context_window": 32768, "thinking": "chat_template"}
  }
}
```

| Field | For | Meaning |
|-------|-----|---------|
| `context_window`, `max_output` | Any model | Token counts, at least 1 |
| `tools`, `image_input` | Any model | Whether it takes tools, and images |
| `thinking` | A configured provider's model | `none`, `chat_template` (vLLM's `chat_template_kwargs.enable_thinking`) or `effort` (`reasoning_effort`, or OpenRouter's `reasoning.effort`); on an `openrouter` provider, `none` or `effort` |
| `effort_levels` | A configured provider's model | Its effort levels, lowest first, as lowercase names: by default `low`, `medium`, `high`, or the levels OpenRouter's listing gives a reasoning model |
| `reasoning_text` | A configured provider's model | Whether it sends its reasoning as text. `/models` notes a thinking model that does not (`· no reasoning text` in the Thinking column); it changes nothing else: reasoning text is kept whenever a server sends it |

- A field that is set wins over the provider's listing and the shipped
  table; one left out keeps theirs. An empty entry, `{}`, adds the model to
  the list and changes nothing about it.
- The thinking fields are for a configured provider's model only. Google's,
  Anthropic's and OpenAI's thinking comes from their listings and the table,
  so an entry on one of their models that sets `thinking`, `effort_levels` or
  `reasoning_text` fails when the settings load. So does a bare id as a key
  (`claude-sonnet-5-5`): the message gives its full form, so that one model
  has one key.
- **Who sets it**: the app's settings class, then the user file, the project
  file and the environment (`AGENTIC_MODELS`, or your app's prefix, as JSON),
  in that order. Each adds models, or overrides an earlier entry's fields
  one by one; nothing removes an entry. A project may add entries: they only
  choose among the providers the app and the user defined, and their numbers
  are hints. An entry on a provider this user has not defined is not
  offered, with a warning, rather than an error: a project file is shared
  by users with different providers.
- `/settings` never shows `models`, and no save writes it, so the app's
  entries are never copied into a user's or a project's file. Edit it by hand.

An app ships its list, and any providers of its own, by overriding the two
fields' defaults in its settings class (`examples/research_demo/settings.py`
ships a list of Google's and Anthropic's models, and leaves every other
provider to the user):

```python
from pydantic import Field

from agentic_cli import BaseSettings
from agentic_cli.models import ModelEntry, ProviderSpec

# A field declared again replaces its title and description: carry them over.
_PROVIDERS = BaseSettings.model_fields["model_providers"]
_MODELS = BaseSettings.model_fields["models"]


class MySettings(BaseSettings):
    model_providers: dict[str, ProviderSpec] = Field(
        default_factory=lambda: {
            "openrouter": ProviderSpec(
                kind="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key_env="OPENROUTER_API_KEY",
            ),
        },
        title=_PROVIDERS.title,
        description=_PROVIDERS.description,
    )
    models: dict[str, ModelEntry] = Field(
        default_factory=lambda: {
            "anthropic:claude-sonnet-5-5": ModelEntry(),
            "google:gemini-3.8-flash": ModelEntry(),
            "openrouter:qwen/qwen3.7-plus": ModelEntry(),
        },
        title=_MODELS.title,
        description=_MODELS.description,
    )
```

The user's and the project's entries are added to these, field by field, as
above. An app with no list still works: the pickers then offer the newest
fast, default and strong model of each built-in provider with a key.

### Roles

A role names a model by what it is for:

- `default` is `default_model`. An agent without a `model` runs on it.
- `fast` and `strong` are mapped in `model_roles`.
- An app adds a role by naming it in an agent (`AgentConfig(model="vision")`),
  and may ship a mapping by overriding the `model_roles` default in its
  settings class. An app role nobody mapped runs on the default model, with a
  warning.
- A role maps to a model, never to another role, and `model_roles` cannot map
  `default`.

A built-in role nobody mapped (`default`, `fast`, `strong`) is computed from
the model's tier:

1. The newest of its tier among the `models` entries the list offers, of
   the provider the list names first among those with one of that tier: with
   `google:gemini-3.5-flash` listed before `anthropic:claude-sonnet-5-5`,
   `default` is the Gemini model.
2. Else the newest model of its tier from the first provider with a key, in
   the order Google, Anthropic, OpenAI. With only a Google key, no list and no
   `default_model`, that is the newest Gemini Flash (`gemini-3.8-flash` before
   any listing; 0.6 ran `gemini-2.5-flash`).
3. Else the workflow does not start, and the error asks for `default_model`
   or `model_roles[<role>]`. Without a Google, Anthropic or OpenAI key that is
   known before any listing, so `validate_settings` says it, once.

| Provider | `fast` | `default` | `strong` |
|----------|--------|-----------|----------|
| Google | Flash-Lite | Flash | Pro |
| Anthropic | Haiku | Sonnet | Opus |
| OpenAI | Luna, mini, nano | Terra, a plain id (`gpt-5.5`) | Sol, Astra, a plain id's `-pro` (`gpt-5.5-pro`) |

A tier's `-pro` id (`gpt-6-sol-pro`), the same model in a costlier reasoning
mode, has no tier, so a role never picks it. Only Google's, Anthropic's and
OpenAI's models have a tier. A configured
provider's model never fills a role on its own: an app or a user that runs on
configured providers alone sets `default_model`, and maps the roles its
agents name.

A project chooses its models in `./.my_app/settings.json`:

```json
{
  "default_model": "anthropic:claude-sonnet-5-5",
  "model_roles": {
    "fast": "google:gemini-3.5-flash-lite",
    "strong": "openrouter:deepseek/deepseek-v4-pro"
  }
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

The coordinator runs on Claude Sonnet 5.5, the drafter on Gemini 3.5
Flash-Lite, and the checker on DeepSeek V4 Pro through OpenRouter. A model a
setting, a role or an agent names joins the list. Every model is resolved when
the workflow starts: a reference to a provider that is neither built in nor
defined in `model_providers`, or a provider that cannot be called (it names
its key variable and where it is read), fails with one error that names each
agent concerned. A model its provider does not serve does not stop the start:
the settings decide what runs, and one warning names the model, the setting
that names it and what to do (see [Checking ids](#listings-and-the-model-cache)).

### The model pickers and `/models`

`/settings` has a model picker for the default model and one for each of the
`fast` and `strong` roles. Each offers exactly the list:

1. The `models` entries that can run: the app's, then the user's, the
   project's and the environment's.
2. Each model `default_model`, `model_roles`, an agent or the manager's own
   `model` names, and the model each of `default`, `fast` and `strong`
   resolves to (see [Roles](#roles)), that can run and is not already listed.
3. Only when the list offers no entry: each built-in provider with a key,
   its newest fast, default and strong model (at most three each), from its
   listing or the shipped table.

A model can run when its provider can be called: a built-in provider with
its key, or a configured provider whose `api_key_env` is set (or that names
none). A model a setting names (an entry, `default_model`, a role's model or
an agent's) is not offered when its provider is not defined, cannot be
called, or does not serve it (a listing this process made lacks it, or a
listing recorded it as absent), and `/models` says why for each. A model the
settings name that its provider does not serve, and an entry whose provider is
not defined, is also warned of once, when a listing shows it. Listings never
add to the list. Each model appears once. Picking writes
`default_model`, or the role's `model_roles` entry, to the project settings
file and restarts the workflow on it, keeping the conversation. Before the
workflow is ready, the pickers offer the list from the entries and the
shipped table. An app adds a picker for a role of its own with the key
`model_role:<role>` in `get_ui_setting_keys()`.

- **`/models`** shows the list, then each model the settings name that it
  leaves out (an entry, or `default_model`, a role or an agent's model) with
  why ("no key: set TOGETHER_API_KEY", "provider 'together' is not defined in
  model_providers", "not served by together"). Per model: the context window,
  max output, thinking, tools, cache, where each value came from (`config`,
  `listing`, `table`, `default`) and the roles that use it. Below, each
  provider in use: whether it has a key (or needs none), whether its listing
  is fresh, stale, failed or not listed, when it was fetched, how many models
  it shows (its listing holds more, which `/models <text>` finds), and the
  last error. Before the workflow is initialized it reads the model cache,
  and lists what a start would wait on (see
  [Listings and the model cache](#listings-and-the-model-cache)).
- **`/models <text>`** searches the full listing of each provider that can be
  called (and is listed), ignoring case, by reference, and marks the models
  already in the list: how to find an exact id to copy into `models`
  (`/models llama`, or `/models together:` for one provider's models). A
  provider not in use is listed the first time it is searched, and any whose
  listing is not fresh is listed again.
- **`/models refresh`** lists again now every provider in use or searched,
  and says which providers' models changed. Run it after a key change to an
  account with other models.
- In code, an initialized manager's `manager.catalog` gives
  `catalog.offered()` (the entries and the models the settings and the
  agents name), `catalog.not_offered()` (those it leaves out, with why),
  `catalog.records("google")`,
  `catalog.status()`, `await catalog.search(text)` and
  `await catalog.refresh()`, and `manager.model` is the default model's
  reference. The pickers' list adds the roles' models:
  `catalog.offered(extra=model_services.role_models(settings, catalog, providers))`,
  with `providers = model_services.providers_from_settings(settings)`.

### Listings and the model cache

A model's facts come from, highest precedence first: its `models` entry, its
provider's listing, and the shipped table (model families, tiers, thinking
rules, and the numbers used without a listing); otherwise a conservative
record that takes tools and does not think, and whose unknown window counts
as 128,000 tokens.

- **Only the providers in use are listed**: those of the entries that can
  run, those of the models a setting, a role or an agent names, and the
  built-in providers with a key when the list offers no entry, or when a
  role the agents run on is computed and the list has no model of its tier.
  A provider with `list_models: false` is in use but never listed.
- **Startup uses the models the settings name at once**, and waits on a
  listing only to fill a gap they leave: no `models` entry can run (the
  list's fast, default and strong models come from the built-in providers'
  listings), a role the agents run on is computed (it has no model of its
  own: it takes the newest of its tier among the entries a listing serves, so
  the start lists the provider of the entry it would take unless a cached
  listing serves it, and with no entry of its tier the built-in providers),
  or nothing describes a model the settings name (no cached listing has it,
  no family rule of the shipped table matches it, and its `models` entry, if
  any, does not give its `context_window` and `thinking`: its listing is
  where its limits and thinking come from). Those listings run at once, each
  for at most 10 seconds, and a provider the cache holds a listing or a
  failure for is not waited on for the built-ins. Every other provider in use
  is listed in the background after the start (one the cache holds nothing
  for, an older or failed entry, or an entry that lacks a model the settings
  name): that is how the models the settings name are checked. An app whose
  list and settings name models the table knows starts without the network.
- **Listings are cached** in `~/.my_app/model_cache.json` (private,
  versioned, holding no key) for 24 hours, one entry per provider with its
  address and kind: a change to either lists it again. An older entry is used
  while it is listed again in the background, and a listing counts as older
  once it is 24 hours old in a long session too. A listing that fails at startup
  is listed once more in the background: until one succeeds the provider's
  ids are not checked and its models' limits and thinking come from the
  shipped table (an OpenRouter or configured provider's models, which the
  table does not know, run without thinking and with the 128,000-token
  window floor unless a `models` entry says otherwise). The agents are
  resolved once, at startup, and keep that for the session: the listing
  reaches the next start, `/models`, the pickers and side calls. Every
  process of the app shares
  the file, and a listing writes only its own provider's entry and keeps the
  others as the file holds them (an entry this release cannot read too; a file
  of a newer release is neither read nor written: this process keeps its
  listings in memory and logs once that every start lists); a failed listing
  never replaces one of the provider's address and kind.
- **Checking ids**: a model the settings name (a `models` entry,
  `default_model`, a role's model, an agent's or the manager's) is never
  refused. The workflow starts on it as named, and the listings check it: at
  startup when the start lists its provider anyway, else in the background.
  When its provider does not serve it, one warning says so, naming the setting
  (or the computed role that runs on it) and what to do, the list leaves it
  out (`/models` says why), and a turn on it fails at the provider; a model
  nothing describes runs on the conservative record (no thinking, the
  128,000-token window floor) unless a `models` entry says otherwise. A listing read from the cache that lacks such
  a model does not count until a new listing confirms it, since the cache can
  predate a model. The cache records the absence (of every project's settings
  that share the file), so a retired model costs one listing, not one at every
  start. Any other id (one `/models` looks up) is rejected only when its
  provider's listing is fresh (under 24 hours old, however long the session
  has run); a failed, untried or older listing accepts it, and a rejection
  read from the cache is confirmed by one new listing first.
- **An OpenAI-compatible host's listing** may be `{"data": [...]}` or a bare
  array, and a model's window is read from its `context_length`,
  `context_window` or `max_model_len`, the first present. Its thinking comes
  only from its `models` entry.

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
    model="anthropic:claude-opus-5-5",
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
| Reasoning effort: GPT, OpenRouter's reasoning models, and `models` entries that say `effort` | The lowest effort the model has | Effort `low` / `medium` / `high` | Error |
| A chat template: `models` entries that say `chat_template` | `enable_thinking: false` | `enable_thinking: true` | Error |

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
  `high → anthropic:claude-sonnet-5-5: adaptive, effort high`, or
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
- GPT and the configured providers' models (OpenRouter's among them) get
  nothing from it on any backend. OpenAI caches long prompts on its own.

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

### How each backend runs them

- Every backend runs every provider and builds each agent's model once, when
  the workflow starts: ADK with `workflow.adk.model_factory.build_adk_model`,
  Deep Agents and LangGraph with `workflow.langchain_models.build_chat_model`.
- GPT runs on OpenAI's Responses API on all of them, without storing responses
  (`store=False`). A configured provider's models run on Chat Completions.
- Each agent's `model_settings` apply on every backend, and thinking text
  reaches the thinking box for every provider.
- The `[deepagents]` and `[langgraph]` extras pin `pydantic` below 2.13 and
  bring `anthropic` 1.x (see [Installation](#with-deep-agents)).
- On ADK, GPT's reasoning is not passed back across tool calls, and neither is
  an `openrouter` provider's `reasoning_details`; Deep Agents and LangGraph
  pass both back.
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
| `model` | str \| None | A model reference (`anthropic:claude-sonnet-5-5`), a bare model id, or a role (`fast`); without one, the `default` role (see [Models and providers](#models-and-providers)) |
| `model_settings` | ModelSettings \| None | The agent's thinking, `max_tokens`, sampling settings and `extra` (see [Thinking](#thinking)) |
| `mcp_servers` | list[MCPServerConfig] | MCP servers whose tools the agent gets (ADK and Deep Agents) |
| `skills` | list[str] | Agent Skills: a skill folder's absolute path, or a name under `skills_dirs` (ADK and Deep Agents; never looked up in the working directory) |
| `context` | `"fresh"` \| `"inherit"` \| None | What a sub-agent sees when its parent delegates to it: the request alone (`"fresh"`, what None means) or its parent's conversation too (`"inherit"`); see [Sub-agent context](#sub-agent-context) (ADK and Deep Agents) |

The agent graph is validated before anything is allocated — ahead of model
discovery and service creation. Duplicate names, unknown `sub_agents`
references, self-references, delegation cycles, a sub-agent shared by two
parents, and **more than one root** raise `AgentGraphError` naming the
offending agents. Exactly one agent may be unreferenced: the runner starts from
a single root, so agents under any other root would never run. A `context` on
the root, `"inherit"` on an agent with sub-agents of its own and an unknown
`context` value raise it too, and on ADK, when the agents are built, so does a
sub-agent named like one of its parent's tools. Agents are built
in dependency order, so declaration order does not matter. Then every agent's
model is resolved, with the default model, before anything is built: an agent
whose model cannot run fails startup, named in the error.

A callable `prompt` may take **no arguments** (including all-defaulted ones) or
**exactly one argument**, which receives the manager's settings. Other
signatures, `async def` factories, and non-string results raise
`AgentGraphError` naming the agent.

### Sub-agent context

A sub-agent is a task its parent hands off and gets back in the same turn, on
both backends. The parent calls it with a request; the sub-agent works with its
own prompt, tools and model, and its last answer is the call's result. Only the
main agent replies to the user: a sub-agent's own text shows in the events box
under its name (and is left out of fact extraction), its answer passes the
[tool result cap](#tool-results), and the next turn starts at the main agent
again. Delegating needs no permission; each tool the sub-agent calls is gated
like the main agent's. Give each sub-agent a `description`: its parent chooses
by it, and one without is offered as "The <name> agent.", with a warning.

Each sub-agent's `context` sets what it sees:

| `context` | The sub-agent sees |
|-----------|--------------------|
| `"fresh"` (the default) | The request alone (and on Deep Agents the parent's plan and tasks) |
| `"inherit"` | Also its parent's conversation as it stands: after any summary, without the parent's reasoning |

```python
coordinator = AgentConfig(
    name="coordinator",
    prompt="Delegate research to researcher and reviews to reviewer, each with a "
           "complete request; they report back to you, and you answer the user.",
    sub_agents=["researcher", "reviewer"],
)
researcher = AgentConfig(name="researcher", prompt="Find facts.",
                         description="Finds facts.")                  # fresh
reviewer = AgentConfig(name="reviewer", prompt="Check what was said.",
                       description="Reviews the conversation.", context="inherit")
```

- **Prompts say "delegate to X".** The tool the parent calls is named after
  the sub-agent on ADK (`researcher(request=...)`) and is `task` on Deep
  Agents, so a coordinator's prompt should tell it to delegate to an agent
  with a complete request, never name a tool. A fresh sub-agent knows only
  what the request says. An inheriting one sees the conversation, not its
  parent's instructions: it keeps its own prompt, so put what it must know in
  the request too. (On Deep Agents the coordinator's `task` tool says a fork
  inherits its system prompt; it does not.) A specialist's prompt should
  have it report its result, not address the user: its answer goes to its
  parent.
- **On ADK**, a sub-agent runs in ADK's `single_turn` mode with transfers
  off, so no agent built from a config is offered `transfer_to_agent`; an
  inheriting one is shown its parent's turns as quoted context. **On Deep
  Agents**, a fresh sub-agent is isolated and an inheriting one is a fork
  (Deep Agents' `mode="fork"`, in beta): the conversation, then Deep Agents'
  preamble and the request. Its own summarization compacts an inherited
  conversation too large for its model.
- **The limits in 0.7.0.** `context` is for sub-agents only, and `"inherit"`
  only for an agent without sub-agents of its own: Deep Agents tells a forked
  sub-agent that delegating "will be refused", which a coordinator would
  believe. And `"inherit"` only for the main agent's own sub-agents: below
  another sub-agent, ADK would show it the main agent's whole conversation,
  what its fresh parent was kept from. A sub-agent on ADK cannot share its
  name with one of its parent's tools. Each raises `AgentGraphError` at
  startup.
- **A sub-agent's failure.** On Deep Agents, a sub-agent whose model call
  fails ends the turn with one error naming it. On ADK the turn goes on:
  ADK hands the parent "Error running sub-agent: …", and the user sees a
  warning, `<agent> failed: <the provider's reason>`, while the parent
  answers.
- **A sub-agent's job.** A long-running tool on a sub-agent (a job starter)
  returns the job's id at once, and the sub-agent answers its parent in the
  same turn. When the job finishes, the main agent is told, in a message of
  its own that opens with `[Automatic job notification — not written by the
  user]`, on both backends, which fact extraction leaves out. The main
  agent's own job is told the same way (a native ADK YAML app's still
  answers the call that started it).
- **A native ADK YAML app** (`adk_config_path`) is built by ADK and keeps
  `transfer_to_agent`: the agent it transfers to takes the conversation and
  answers the user. It is library-only for now: an app's own code runs it,
  `GoogleADKWorkflowManager(agent_configs=[], adk_config_path=...)`, and no
  `BaseCLIApp` starts one (the CLI builds its manager from agent configs).

### Defining agents in YAML

The same agents can be written in a YAML file. `load_agents_from_yaml(path)`
reads it into the `AgentConfig`s above, and `create_workflow_manager_from_yaml(path, settings)`
loads the file and builds the manager (it is
`create_workflow_manager_from_settings(load_agents_from_yaml(path), settings)`):

```yaml
# agents.yaml
agents:
  - name: coordinator
    instruction_file: prompts/coordinator.md   # relative to this file
    sub_agents: [researcher]
  - name: researcher
    description: Analyzes text it is given and reports the figures.
    context: fresh
    prompt: |
      You are a researcher. Count words with the word_count tool, never by
      eye, and report the figure to the agent that asked.
    tools: [word_count]                         # a registered name
```

```python
from agentic_cli.workflow import load_agents_from_yaml

import my_app.tools  # registers word_count with @register_tool: first, so the name resolves

AGENT_CONFIGS = load_agents_from_yaml("agents.yaml")
app = BaseCLIApp(app_info=info, agent_configs=AGENT_CONFIGS, settings=settings)
```

A bare list of agents, without the `agents:` key, is accepted too. The fields
are `AgentConfig`'s, and an unknown one is an error:

| Field | Description |
|-------|-------------|
| `name` | Required, unique |
| `prompt` (or `instruction`), `instruction_file` | The system prompt, inline or read from a file; one is required, and `instruction_file` wins when both are set. A relative `instruction_file` is relative to the YAML file, not the working directory |
| `description` | What the agent does: its parent chooses whom to delegate to by it |
| `model`, `model_settings` | A model reference or role, and `ModelSettings` as a mapping (`temperature`, `max_tokens`, `thinking: {mode: high}`, ...) |
| `tools` | A list of names: a registered tool's name (`web_search`, or your own), or a dotted path `my_pkg.tools.fn` to a function registered with `@register_tool` |
| `mcp_servers`, `skills` | As on `AgentConfig`: MCP servers (`name`, `transport`, `command`/`args` or `url`, ...) and skill names |
| `sub_agents`, `context` | The agents it may delegate to, and what a sub-agent sees: `fresh` or `inherit` (see [Sub-agent context](#sub-agent-context)) |
| `include_state_tools` | `true` by default: the plan and task tools are added |

- **A tool is a name, resolved when the manager is built.** A bare name is
  looked up in the tool registry, so the module that registers it must be
  imported first, as above; a name nothing registered raises `ValueError`
  naming it. A dotted path is imported, then must be the registered
  function: one that is not is denied at every call, and named at startup by a
  `tool_not_registered` warning (see [Creating Tools](#creating-tools)).
- **Load only YAML you trust.** A dotted path imports its module, which runs
  its code; an `mcp_servers` entry's `command` and `args` start that program
  without a permission prompt (Deep Agents starts it with the workflow, ADK
  when it first lists the agent's tools); and an
  `instruction_file`, absolute or relative (`../..`), reads whatever it points
  at. The framework never loads a YAML file on its own: your code names it.
  A native ADK `root_agent.yaml` is a different thing, built by ADK itself
  (`adk_config_path`; see [Sub-agent context](#sub-agent-context)).

`examples/yaml_agents/` is a runnable app on this: a coordinator that delegates
word counting to a researcher whose tool is named in its YAML.

### Turn and lifecycle contracts

A workflow manager runs **one turn at a time**: `process()`,
`resume_with_job_result()` and `compact()` serialize on a turn lock, and
`reinitialize()` and `cleanup()` take it too (after the lifecycle lock they
share with `initialize_services()`), so the backend is never torn down
mid-stream. `initialize_services()` takes only the lifecycle lock: a turn
initializes before it takes the turn lock, which keeps the two from
deadlocking. Admission also re-verifies the backend is live after acquiring
the turn lock, so a turn queued behind a cleanup reinitializes (or fails
cleanly) instead of running against released resources. Run separate managers
for genuine parallelism.

Every turn starts the same way, `async with self._turn(session_id, user_id) as
ref:` (`BaseWorkflowManager`'s): the conversation is resolved and named in the
log context, the turn admitted, and the settings, the services and the active
turn set for the block. The rest of the turn protocol is the base's too. A
finished job's resume (`resume_with_job_result(record, result=None)`) checks
that the record names its conversation, runs as a turn of it, checks the
conversation is there, reads the job's result on a worker thread when it is
not given (the job manager takes a lock other processes share), and hands the
notice to the backend; `can_resume(record)` makes the same checks without the
turn; `compact(session, focus=None)` runs the backend's summary as a turn.

The lifecycle is `BaseWorkflowManager`'s: `initialize_services()` (a failed
attempt is rolled back), `cleanup()`, `reinitialize()` and the one release
they all run (`_release_resources`). The release marks the manager
uninitialized, takes what the backend holds, closes the services, then closes
the rest in one task of its own, in a fixed order, awaited under a shield:
Ctrl+C during a model switch stops only the wait, and nothing is left open. A
model switch (`reinitialize(preserve_sessions=True)`) keeps the conversation's
session store and services, also when it fails. A backend of your own
implements `backend_type`, `_get_state_tools`, `_do_initialize` (its build)
and `process` (inside `_turn`), and two hooks for what it holds:
`_detach_backend()` (what each initialization builds: agents, a runner, MCP
sessions) and `_detach_store()` (the store its conversations live in, which a
model switch keeps). Each is synchronous (`def`), sets its handles to `None`
and returns `(name, close)` pairs; both default to nothing. For job resume it
implements `_conversation_exists(ref)` and `_deliver_job_notice(notice,
record, result, ref)` (tell the root agent `notice` as one user-role message,
yielding the turn's events), and for `/compact` `_compact_turn(session,
focus)`; `supports_job_resume` and `supports_compact` are derived from them
(or from its own `resume_with_job_result` or `compact`, which it keeps), and
the CLI asks them. `BaseWorkflowManager`'s docstring ends with such a backend.

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
was stopped. The whole request stops, its sub-agents too, and every tool call
the model made is given one result, so the next turn starts from a history
every provider accepts: a call that finished keeps its result, a running one
is answered "Interrupted by the user before it finished; it may have partly
run…", one waiting at its permission prompt or at a question "Not run: the
user cancelled it.", and a request stopped before its reply leaves
"(Interrupted by the user.)" as the agent's reply. A call a crash left open
is answered the same way on the next turn. Background jobs the request
started keep running, and a line after the cancel names each still queued
or running, with how to stop it: "Still running in the background: build
(j1). Stop one with /jobs cancel <id>."

The HITL input callback is **context-local**: `set_input_callback()` binds it
for the calling context (and any task started from it), so two consumers of one
manager cannot capture each other's prompts.

`WorkflowController` exposes a derived `WorkflowState`
(`uninitialized`/`initializing`/`ready`/`failed`/`closed`) that never reports
`ready` over an uninitialized manager, and `controller.workflow` raises unless
the state is `ready`. Lifecycle transitions (init, reinitialize, a switch of
backend, close) are serialized, so nothing is published after `close()`. The
controller runs the open conversation's backend (`controller.backend`): the
settings' `orchestrator` at its first start, then the one
`switch_backend(backend)` chose for a new conversation; `reinitialize` (a
model switch) is always in place. A failed
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

**Registration is required for a tool to run.** The permission engine is on by default, and any tool without a capability declaration is denied at call time (fail-closed) on every backend. You can still pass raw callables into `AgentConfig.tools`, but unless they are registered with `@register_tool` they will be blocked — register every tool with `capabilities=` (or `EXEMPT` for pure, side-effect-free functions). Building the agents logs a `tool_not_registered` warning for each such tool (a plain callable, or a `FunctionTool` or LangChain `@tool` wrapping an unregistered function), naming the agent and the tool. Registration also gives the tool registry metadata and tool-summary formatting.

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
- **`EXEMPT`** — opts the tool out of the engine entirely. Use for tools that need no permission check: pure functions with no side effects, and backend-internal tools (e.g. delegation: Deep Agents' `task` and ADK's tool for each sub-agent; ADK's `transfer_to_agent` in a native ADK app; backend state tools).

### Built-in Tools

#### Python Execution

Two complementary tools:

- **`execute_python`** — stateless Python via `SafePythonExecutor`, in a fresh subprocess inside the OS sandbox (sandbox-exec on macOS, bubblewrap's `bwrap` on Linux). Use for quick calculations.
- **`sandbox_execute`** — stateful, multi-turn Python in a Jupyter kernel. Variables, imports, and DataFrames persist across calls within a session. Off unless `stateful_executor_backend` is set: `"docker"` runs each session in an isolated container (below); `"local"` runs the kernel on your machine as you, with your files, network and privileges, in the app's own Python environment (a `%pip install` there changes the app's dependencies). The local backend isolates and restricts nothing, so use it only where you would run the code yourself.

```python
from agentic_cli.tools import execute_python
# Stateless: result = execute_python("import math; math.sqrt(2)")

from agentic_cli.tools.sandbox import sandbox_execute
# Stateful (requires sandbox service):
# sandbox_execute("x = 42", session_id="analysis")
# sandbox_execute("print(x * 2)", session_id="analysis")  # sees x
```

The `/sandbox` CLI command lists and resets sandbox sessions.

`execute_python` runs its code only inside the OS sandbox: it writes only to
the working directory (and `os_sandbox_writable_paths`), reads no credential
store, no `~/.{app_name}` and not the project's `.env`, and has no network
unless `os_sandbox_allow_network` is on. On a host with no sandbox backend
(Windows has none), when the sandbox cannot be set up, or when the sandbox tool
fails as it starts (on Ubuntu 24.04 and later, `bwrap` needs unprivileged user
namespaces, which AppArmor restricts by default), the call is refused with an
error that keeps the cause and says how to get a working sandbox. Setting
`os_sandbox_enabled` to false (a user setting: in `~/.{app_name}/settings.json`
or the app's environment variable, not the project file or `/settings`) runs
the code without a sandbox, with your privileges, and so does
`os_sandbox_strict: false` when the sandbox is unavailable: the executor then
allows only standard modules (math,
statistics, json, re, datetime, collections, ...) and checks the code, but
those checks are no security boundary (CPython has no in-process sandbox, and
those modules reach the host themselves), so every such call is approved on
its own. `sandbox_execute` on the `local` backend runs with your privileges
too.

#### Docker-isolated sandbox

The `docker` backend is the one that isolates the code: `sandbox_execute` runs
inside a network-isolated container. To use it:

1. Ensure Docker (or podman) is installed and running, and pull the image once:
   `docker pull quay.io/jupyter/scipy-notebook:python-3.12`
2. In settings: `stateful_executor_backend = "docker"`.

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

Every HTML, text, Markdown, JSON, XML or PDF page `web_fetch` reads is saved as received in `./.{app_name}/fetched/` (a folder git ignores), and the result's `saved_path` names it. Pass `saved_path` to `kb_ingest_file` to add the page to the knowledge base. HTML, text, Markdown, JSON and XML pages are read up to `webfetch_max_download_bytes` (5 MB; PDFs use `webfetch_max_pdf_bytes` instead). The summary is written from the page converted whole (an HTML page's main text with its links, html2text's whole-page text for a page over 1 MB; a PDF's text with `--- Page N ---` markers; any other text type as it is), then cut at `webfetch_max_content_bytes` (100 KB of text); `truncated` says the text, or the page itself, was cut. A page that is not text (an image, an archive), a PDF cut at `webfetch_max_pdf_bytes` and a page with no text (empty, built by scripts, a scanned PDF) are errors, not summaries. Saved pages older than `webfetch_saved_max_age_days` (7) are deleted, then the oldest until the folder is under `webfetch_saved_max_mb` (200).

#### ArXiv Search

Search and fetch academic papers:

```python
from agentic_cli.tools import search_arxiv, fetch_arxiv_paper

results = await search_arxiv("transformer attention", max_results=10, categories=["cs.CL"])
paper = await fetch_arxiv_paper("1706.03762")  # "Attention Is All You Need"
```

The `ArxivSearchSource` (in `tools/arxiv_source.py`) is cached via the service registry and reused across tools. To pull a paper into the KB, use `ingest_arxiv_paper(arxiv_id)` — it composes the arxiv source with the knowledge base. It reads the paper's PDF up to `arxiv_max_pdf_bytes` (50 MB; `web_fetch` keeps its own 5 MB): a larger PDF fails the call, and nothing is stored. When the PDF cannot be had otherwise (none, a 404, an answer that is no PDF), the abstract is stored and the result says why (`pdf_download_error`); while arXiv is refusing or not answering requests (a 429 or 503, a timeout, a connection error), the call fails instead.

arXiv asks for one request every three seconds, on one connection at a time, so the source sends its requests (searches, metadata, PDFs) one at a time, each at least 3 s after the last. A search or metadata request times out after 30 s without an answer (a PDF, after 60 s). A request arXiv refuses for going too fast (HTTP 429 or 503) is retried after the wait arXiv asks for (`Retry-After`), else after 3, 6 and 12 s, and the calls queued behind it wait too. When the retries run out, or arXiv asks for more than a minute, the call fails, and so does any other at once, for that long (at least a minute, at most an hour), saying when to try again; after a request arXiv did not answer, for 30 s. A search is a coroutine (`await source.asearch(...)`); `source.search(...)` is its synchronous form, for code outside an event loop (inside one it raises, rather than block the loop).

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

Writes are atomic (temp file + rename) via `agentic_cli.file_utils`. `edit_file` changes
one place: `old_text` must appear exactly once unless `replace_all=True`. The
file tools read and write UTF-8 (a file in another encoding is refused, never
converted), and an edit keeps the file's line endings. `glob` and `grep`'s
`file_pattern` take brace patterns such as `*.{js,ts}`.

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
| `kb_search` | Hybrid BM25 + vector search with RRF fusion, filters by source/date; keyword-only without the `kb` extra, with `knowledge_base_use_mock`, or when the embedding model did not load |
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

Each knowledge base folder keeps what it holds in one SQLite file, `kb.sqlite`: the documents, their chunks, each chunk's keyword terms and vector, and an FTS5 keyword index, each change one transaction (`kb.sqlite-wal` and `-shm` sit beside it while it is open). Beside it are the files you can read: `documents/<id>.md` (each document's sidecar, rendered from the database), `files/` (the stored originals, such as PDFs), `concepts/`, `index.md` and `ingest_log.md`. A sidecar or stored file the database has no row for, which a crash can leave, is removed when the knowledge base next opens. A folder 0.6 wrote (`metadata.json`, `documents/<id>.json`, `embeddings/`) is imported into `kb.sqlite` when it first opens: each document is chunked again, its vectors are embedded at the first open that has the model, the old files move into `pre-0.7.0/` and `embeddings/` is deleted. A `metadata.json` that is not JSON at all (a git-LFS pointer not pulled yet) is not imported: that scope is left out until it can be. What an older version still running writes afterwards is moved into `pre-0.7.0/after-import-<time>/`, its stored files and sidecars included, and is not in the knowledge base; while such a session holds the folder's `.lock`, an open waits up to 60 s for it. Delete `pre-0.7.0/` once the knowledge base looks right; 0.6 cannot read `kb.sqlite`, so going back to it means restoring those files by hand. Deleting a document leaves none of its text on disk: the database overwrites what it deletes, drops it from the keyword index at once, and empties its write-ahead log after the delete; while another session reads, the log is emptied at a later delete or when this session closes. An SQLite older than 3.44 cannot read a keyword index a newer one wrote, so a session on one searches by scanning instead, and the text of a document it deletes stays in that index until a session with a newer SQLite opens the knowledge base and rebuilds it. WAL needs shared memory, so a knowledge base on a network filesystem is unsupported (as is the session database).

A project's knowledge base can arrive with a cloned repository, so the knowledge base reads and appends its files only when they are regular files, never through a symlink (one that is not counts as missing, and the next write replaces it). A directory whose `documents`, `files`, `concepts` or `pre-0.7.0` folder, `kb.sqlite` (or its `-wal`, `-shm` or `-journal`) or `.lock` file is a symlink, that holds a rollback journal naming another file for SQLite to delete (a super-journal, which a knowledge base never writes), or whose database holds anything but a knowledge base's tables (a trigger, a view), does not open (`UnsafeKnowledgeBase`); `KnowledgeBase.open` leaves that scope out with a warning. The user knowledge base's own directory may be a symlink: to keep it on another disk, link that directory, not a folder inside it. The project knowledge base's may not, nor may `./.{app_name}`: the workflow opens it `within` the working directory (`KnowledgeBase.open(within=...)`), and a project scope reached through a link is left out the same way. So is a scope that fails to open for any other reason (a folder it cannot create, such as a read-only working directory's, or a `kb.sqlite` that is not a database, such as a git-LFS pointer; it is never replaced), with a `kb_scope_unavailable` warning: the other scope takes the writes, and the one left out is not written to. `KnowledgeBase.open` raises only when no scope opens.

Ingesting a document writes a summary and a sidecar with the workflow's LLM when `knowledge_base_summarize` is on (the default; one call per document). Turn it off in `~/.{app_name}/settings.json` to store the first ~500 characters instead. A knowledge base you build yourself calls the `summarizer=` you pass it in `generate_summary`, `generate_sidecar_payload` and `backfill_sidecars`; `ingest_document` stores the summary and sidecar payload you give it, or a preview. `KnowledgeBase.ingest_text(content, ...)` and `ingest_bytes(data, extension=..., ...)` do the conversion and the summarizer call for you; they raise `EmptyDocument`, `UnsupportedDocument` or `ConversionFailed`, `KeyError` for an unknown `scope=`, and `ValueError` for an unknown `source_type`; errors from storing the document propagate.

Both knowledge base scopes share one `EmbeddingService`, so the workflow loads the embedding model once (`/papers`, opening a knowledge base the workflow has not, builds its own). An `EmbeddingService` whose model fails to load remembers the error and does not try again. A knowledge base you build yourself uses the `embedding_service=` you pass it (any service but a `MockEmbeddingService` searches by meaning; the `kb` extra is needed only to build a sentence-transformers one), so several knowledge bases can share one model too. A model that does not load (the first run while offline, before the model is cached) leaves the knowledge base searching by keyword for that session, with one warning (`kb_keyword_search_only`); the app starts. Such a session stores its chunks without vectors, and the next open with a working model embeds them. Ingesting embeds on a worker thread, before the change takes the database's write lock, and every knowledge-base call an async method or kb tool makes runs there too: a change may wait for another session's (up to 60 s, then `KnowledgeBaseBusy`), and the event loop does not wait with it. A thread cannot be stopped, so an ingest cancelled with Ctrl+C still stores its document.

#### Memory Tools

Memories are short notes kept across conversations, as markdown files:

```python
from agentic_cli.tools import memory_tools
# save_memory, read_memory, search_memory, update_memory, delete_memory
```

Each memory is one file with a frontmatter (`id`, a one-line `description`, `type`: `user`, `feedback`, `project` or `reference`, `source`, `conversation`, `created`, `updated`) and its text. User memories (true across projects) live in `<workspace_dir>/memory/`; project memories in `<workspace_dir>/projects/<name>-<hash>/memory/`, keyed by the working directory (there is no project scope when the app starts at home, `/` or above home). You can read and edit the files: the store sees an edit at its next call, keeps keys you add, and never rewrites, moves or deletes a file it cannot parse (it lists it as a problem, a proposal file too). A change keeps the previous version in the scope's `.history/` for 30 days (at most 20 versions a memory). Search is by keyword (BM25, the knowledge base's word rules, Chinese, Japanese and Korean included), computed when asked; it writes nothing.

`save_memory(description, content, type, scope="project", user_requested=False)` refuses a description over 150 characters or on two lines, a text over 4,000 characters, a description another memory has, text that looks like a credential, and a scope over 1,000 memories, each with a result saying why. `read_memory(item_id)` returns one memory; `update_memory` keeps its id and its previous version; `delete_memory` can be undone for 30 days.

The main agent does not have to search to know what was saved: each conversation it gets an index of the memories, one line each (`- [id] date type: description`, newest first, project memories first, at most 200 lines and `memory_index_max_bytes` bytes, and no more than a tenth of the main agent's input budget, so on a model with a small window it is shorter; the rest reached with `search_memory`), framed as notes that may be outdated and are not instructions, with `read_memory(id)` for the full text of one. The index is taken once per conversation and stays the same for its life, so the prompt cache holds and a memory saved meanwhile shows in the next conversation; it is never stored in the conversation, so a summary cannot drop it. `/resume`, `/compact` and any summary take a new one. Sub-agents get none, and neither does an app whose main agent has no memory tool. On ADK it is the first user content of each of the main agent's requests (the fit guard never drops it, and the overflow retry keeps it); on Deep Agents it is the first text block of the first human message (the summarization counts it). `memory_index_enabled` turns it off.

Once a conversation has read outside content (a fetched page, a search result, a document, a program's output, an MCP answer), every `save_memory`, `update_memory` and `delete_memory` in it asks you, call by call, with the memory shown in full (see the permission engine below); a person's own settings can allow them. The memory folders are denied to `write_file` and `edit_file`, and code `execute_python` runs inside the OS sandbox can neither read nor write them, even where `workspace_dir` sits in a folder the sandbox may write; code that runs outside the sandbox (turned off, or `sandbox_execute`'s local kernel) runs with your privileges and is approved call by call.

With `auto_extract_session_facts` on, the end of a conversation (`/new`, `/resume`, exit) asks the default model for at most five memories worth keeping and files them as proposals in the scope's `.inbox/`, never as memories: a candidate already known, waiting, or discarded before is dropped. `/memory inbox` reviews them (Accept, Discard or Skip), and the app says how many wait at startup and after `/new` or `/resume`. The first start of 0.7.0 imports 0.6's `memories.json` into the user scope once (what the model saved as memories, what the extractor wrote as proposals) and moves the old files into `memory/pre-0.7.0/`; extracted items past the inbox's 100 stay only in that old file (the log counts them), and what a 0.6 session still running writes later is imported at the next start, new ids only. A memory folder that cannot open (a link, which is refused, or one that cannot be written) never stops the app: a project scope that fails is left out, and a user scope that fails leaves the memory tools answering that the store is not available.

The store itself is `agentic_cli.memory.MemoryStore` (provisional), part of the memory package that depends on nothing else in agentic-cli:

```python
from pathlib import Path
from agentic_cli.memory import MemoryStore

store = MemoryStore(Path("~/.myapp/memory").expanduser(), Path("~/.myapp/projects/app/memory").expanduser())
memory = store.save("Prefers metric units", "Report distances in km.", "user", scope="user")
store.search("metric")
store.index()  # the text the main agent gets
```

A knowledge base records the embedding model its vectors belong to, in its database. After a change of `embedding_model`, the next open embeds the stored text again (a large knowledge base makes that one start slower; the log gives the chunk count and the time taken, `kb_reembedding` / `kb_reembedded`); it embeds in batches, each saved as it is done, so an open interrupted part way carries on where it stopped. A session whose model is not the one recorded (another session switched it meanwhile) searches by keyword for the rest of its run (`kb_model_changed_elsewhere`). The default model, `all-MiniLM-L6-v2`, is trained on English; for documents in other languages set `embedding_model` to a multilingual model, such as `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, like the default).

#### HITL (Human-in-the-Loop)

Tool calls are gated by the **permission engine** (`workflow/permissions/`). Each tool declares a list of capabilities (e.g. `filesystem.write(path=...)`); the engine evaluates them against the builtin defaults; the `permissions` setting, the allow and deny rules of every trusted source together (the app's settings-class default, the user's `~/.{app_name}/settings.json`, the environment such as `AGENTIC_PERMISSIONS`, an env file the settings read that is not cwd-relative such as `~/.{app_name}/.env`, and the settings constructor); the deny rules of the project's `./.{app_name}/settings.json` (a cloned repository can ship one, so its allow rules are ignored, and a malformed entry is skipped with a warning); the grants saved for the project; and the session's grants. The builtin defaults allow reads inside the project, the knowledge base and memory, and the arXiv tools' two fixed endpoints (the API and `https://arxiv.org/pdf`). They deny what the OS sandbox protects (both are built from `agentic_cli/protected_paths.py`): reading or writing credential stores (`~/.ssh`, `~/.gnupg`, `~/.aws`, `~/.azure`, `~/.kube`, `~/.config/gcloud`, `~/.config/gh`, `~/.docker/config.json`, `~/.netrc`, `~/.git-credentials`, `~/.pypirc`, `~/.npmrc`); reading the app's keys and settings (`~/.{app_name}/.env`, `~/.{app_name}/settings.json` and any env file its settings read) and the project's env files at any depth (`.env`, `.env.*`, `.envrc`); writing the app's own `~/.{app_name}` (its settings and grants), shell start-up files (zsh's, bash's, fish's), `~/.gitconfig` and `~/.config/git`, `~/.claude`, system directories, the project's `.git` and its `./.{app_name}`. The OS sandbox also hides all of `~/.{app_name}` from the code it runs. A deny wins over any allow rule or grant. A memory write in a conversation that has read outside content (a web page, a document, a program's output or an MCP server's answer) is approved for each call: the prompt offers only `Allow once` and `Deny` and shows the memory; no builtin allow or grant covers it, an allow rule in your settings does. The memory folders (`<workspace_dir>/memory`, `<workspace_dir>/projects`) are denied to `write_file` and `edit_file`, and code `execute_python` runs inside the OS sandbox can neither read nor write them nor move `workspace_dir` aside (code that runs outside the sandbox runs with your privileges and is approved call by call). Reading the working directory needs no prompt only when it is a project directory: started from your home directory, above it or at `/`, reads ask (except the tool outputs and pages the framework saves in `./.{app_name}`). `grep` does not search hidden files or directories below its `path` (name one as `path` to search it). When no rule matches, the user is prompted with `Allow once` / `Allow for this session` / `Allow always for this project` / `Deny`. The prompt names the exact file and, on the next line, what a session or "always" grant would cover: `(for this session / always: all of /p/**)`. A filesystem grant covers the directory it names (or the directory a file is in), never the filesystem root, your home directory itself or a directory above it: a file directly in your home is granted exactly. A filesystem deny rule ignores case (on macOS `./.ENV` is `./.env`); an allow rule does not, so another spelling of an allowed path asks. An argument whose remembered rule would hold a wildcard character (`*` or `?` in the directory a grant covers, `*`, `?` or `[` in a command) is approved for that call only: the prompt offers only `Allow once` and `Deny`. A URL, in a rule or a call, is read as the HTTP client sends it: the host lowercased, IDNA-encoded (`faß.de` is `xn--fa-hia.de`, `host。com` is `host.com`) and without a trailing dot, an IPv4 address in any notation dotted-decimal (`0x7f.1` and `[::ffff:127.0.0.1]` are `127.0.0.1`); the path with `..` segments removed and escapes of plain characters decoded. So a deny on `https://host/admin/**` also covers `https://host/x/../admin/k`, `https://host/%61dmin/k` and `https://host./admin/k`, the prompt shows that form, and a URL the client cannot send is denied; `webfetch_blocked_domains` compares hosts the same way. A URL grant (`web_fetch`) answered "for this session" or "always" covers that site, `https://host/**`: the exact host, not its other subdomains, and HTTPS only (a plain-HTTP URL, an IP address or a local name keeps an exact-URL grant). "Allow once" approves one URL, and the prompt shows both the URL and the site. To pre-approve a site, add a rule to `~/.{app_name}/settings.json`: `{"permissions": {"allow": [{"capability": "http.read", "target": "https://docs.python.org/**"}]}}`. An app ships rules as its settings class's `permissions` default, and they apply beside yours: an allow rule in your settings file keeps the app's deny rules, as a deny list in the environment keeps your file's. No settings save writes `permissions`; each file keeps the section you wrote.

**"Allow always" grants are user-owned, not project-owned.** They persist to
`~/.{app_name}/project_grants.json`, keyed by the *resolved project path*, so
they apply to that checkout on this machine only. They are deliberately not
written into `./.{app_name}/settings.json`: a repository could otherwise ship
pre-approved allow-rules that a clone would silently honour. A clone at a
different path therefore starts with no grants, and a repo-committed grant file
is never loaded.

**Code that may run on your machine without a sandbox is approved call by
call.** That is `execute_python` with `os_sandbox_enabled` or
`os_sandbox_strict` off, and `sandbox_execute` on the `local` backend: the
prompt offers only `Allow once` and `Deny` and says why, and no grant approves
it, not even one given while the code ran in the sandbox (an allow rule you
write in `~/.{app_name}/settings.json` still does: that is your policy). The
settings are read at each call, so a change applies at once. Every prompt for
code shows all of it.

### Long-running jobs

A long-running tool starts work in a `JobManager` and returns a job id at once;
the framework ships the tools that read a job, not the ones that start it,
since a starter decides what runs and how (`examples/jobs_demo.py` has one).
`agentic_cli.tools.JOB_TOOLS` is `[job_status]`: a job's state, a stdout tail
and, once it has finished, its result, the one companion most agents need to a
starter. `JOB_MANAGEMENT_TOOLS` is `[job_status, job_result, job_logs,
job_cancel, job_list]`, for an app that also lets the model read logs, list and
cancel jobs (the user does that with `/jobs`).

`JobManager.register_backend(backend)` takes a `JobBackend` instance and keeps
it under its `name` (one of the same name is replaced), beside the two a
manager starts with, `subprocess` and `inprocess`; a starter then submits to
it with `submit(backend="<name>")`. A manager reads the jobs on disk when it
is built, before the app registers its backends, and leaves a job whose
backend it has not registered as it is (not polled, launched or cancelled)
until one that has it looks at it; `has_backend(name)` says whether one is
registered. So a job on a backend the app no longer registers stays as it
was, RUNNING or queued, until its folder under `~/.{app_name}/jobs/` is
removed (`/jobs clean` removes finished jobs only). The manager is
`workflow.job_manager`, or `get_service("job_manager")` in a tool that
declares `requires="job_manager"`.
It exists only when some agent's tool declares that need (`job_status` and the
other `job_*` tools do), and `workflow.job_manager` is `None` before then.
A backend implements `start`, `poll` and `cancel`, and may override `logs`,
`result`, `pin` and `close`. Its `survives_restart` says whether a manager
other than the one that submitted a job may launch it from its saved record
(another CLI, or the app after a restart or in a new conversation on another backend):
`subprocess` sets it, since its command and working directory are saved;
`inprocess` does not, since its callable cannot be, and nor does a backend
that keeps the default, False. A job on such a backend is claimed at submit
by the manager that submitted it, and one still queued when that manager
closes is recorded FAILED.

## Context management

### Summarizing

A conversation that outgrows the model is summarized and goes on, on ADK and
Deep Agents, once a prompt reaches the smaller of `context_summarize_at`
(85%) of the model's input budget (95% of its context window, less its
`max_tokens`) and `context_max_tokens` (200,000 tokens). The newest 12% of
that, and the current turn, are kept word for word; the summary, user-role
text starting "Here is a summary of the conversation to date:", stands for
the rest, and the stored session keeps every message. A model whose window
is unknown counts as 128,000 tokens. While a summary is written the CLI
shows "Summarizing the earlier conversation…".

On ADK only the main agent's conversation is summarized: a sub-agent leaves
the shared session alone (on Deep Agents each agent summarizes its own). A
request still over its agent's budget (the summary follows the previous
call's size, a sub-agent may run on a smaller model) has its oldest whole
turns left out of that request, never a tool call without its result, with
"[Earlier turns were left out to fit the context window.]" folded into the
first kept message; the summary rides in that message too, so no provider
sees two user messages in a row. The summary call is sized to the model that
writes it: after a switch to a smaller model, the middle of a range too long
for it is left out, the summary so far kept whole. A request the provider
refuses as too long is summarized (the main agent: the retry is the summary
and the turns it kept) or trimmed (a sub-agent), and retried once; if it
still does not fit, the turn ends with "This conversation no longer fits
<model>'s context window. Use /compact or /new." A summary that fails is one
warning a turn, and the next call tries again. On Deep Agents a summary
request the model refuses as too long is sent once more with the middle of
the conversation left out, whatever the provider calls the refusal
(OpenRouter and OpenAI-compatible servers included). With `raw_llm_logging` on,
the log records a request as it was sent, after any trim.

### Tool results

One tool result cannot fill the context. On ADK and Deep Agents, a result
longer than `tool_result_max_chars` (50,000 characters) reaches the model
cut, with its shape and as much of its beginning and end as the limit holds:

- In a dict, the longest strings are cut first (nested ones included), so
  `success`, `error` and the other keys stay. A cut string keeps its first
  70% and last 30% around `…[N characters cut]…`.
- A note says how much was cut and where the full output is. A dict gets it
  under `truncated`, which replaces a tool's own flag; a string gets it at
  its end: `Cut: 182,344 characters. Full output:
  ./.myapp/tool-output/read_file-3f9a1c2b7d4e.txt; read it with read_file
  using offset and limit.`
- The agent reads the full output in parts with `read_file` (offset and
  limit, in lines), with no permission prompt inside the project. A cut
  string is saved whole, on its own lines, after the result's other values,
  and lines over 2,000 characters are split, so a one-line output (minified
  JSON, base64) pages too. Reading a saved output back cuts it again
  without saving a second copy.

`./.{app_name}/tool-output/` is created private (0700, files 0600) with a
`.gitignore` of `*`. As with `web_fetch`'s saved pages, the location is
fixed, a symlinked folder is refused, and after each save outputs older than
7 days are deleted, then the oldest until the folder holds 100 MB (outputs
saved in the last minute are spared); only the folder's own files are
touched. If an output cannot be saved (a read-only project, a symlinked
folder), the call still succeeds: the result is cut, its note says the full
output could not be saved, and the log warns once per folder.

Every result passes the cap: registered tools, MCP and skill tools,
`job_result`, and a sub-agent's answer. Each tool's own limits
stay (`web_fetch` returns a summary, `grep` a number of matches); the cap is
the net beneath them. On ADK it is the `ToolResultCapPlugin`, for apps built
from agent configs (a native ADK YAML app keeps its own `after_tool_callback`s
and gets no cap). ADK's Claude and OpenAI Responses adapters send a model
only a result's `content`, or else its `result`, so the plugin puts the note
there as well. On Deep Agents it is the `ToolResultCapMiddleware`, between
the permission gate and the tool-error middleware, so an error result is
capped too. A `/settings` change of the limit rebuilds the agents and keeps
the conversation.

### Settings

Three settings, the same on ADK and Deep Agents (LangGraph, deprecated,
manages no context). A project may set them, and a `/settings` change of
any of them rebuilds the agents and keeps the conversation.

| Setting | Default | What it does |
|---------|---------|--------------|
| `context_summarize_at` | 0.85 | The share of the input budget (95% of the window, less `max_tokens`) at which the conversation is summarized, from 0.5 to 0.95 |
| `context_max_tokens` | 200,000 | A ceiling on that point, in tokens: on a 1M-token model the summary comes at 200,000. 0 for none; otherwise at least 16,000 |
| `tool_result_max_chars` | 50,000 | A tool result longer than this is cut and its full output saved (see Tool results); at least 2,000 |

A model whose input budget is under 16,000 tokens (a `max_tokens` close to
its window) fails to start with `ContextBudgetError`, naming the agent, its
model and its `max_tokens`.

### Seeing it

- **The status bar** shows the main agent's last prompt against the size
  at which its conversation is summarized: `ctx 45k/170k (26%)`. That is
  the point that matters, not the raw window: on a 1M-token model "/1M"
  would hide that the summary comes at 200,000. The bar names the main
  agent's own model, which may not be the default model. A sub-agent's
  calls and a summary's never move it, nor does a job resuming a
  conversation `/new` left; an inheriting sub-agent's prompt
  holds the conversation, but it is not the main agent's context. On a
  model whose window is unknown (the 128k floor in use) it shows the prompt
  alone, `ctx 45k`, as on LangGraph, which manages no context. A
  `/settings` change that rebuilds the agents (a model,
  `context_summarize_at`, `context_max_tokens`) shows the new trigger at
  once; a new conversation (on any backend) starts without it, and the
  segment waits for its first call, as a resumed session's does.
- **`/status`** adds the main agent's rows after Services: Main Agent (its
  name and model), Context Window (or "unknown, 128k floor in use"), Input
  Budget, Summarize At and Last Prompt. A manager's
  `context_budget(agent_name=None)` gives the same numbers for any agent,
  the root agent's by default.
- **Warnings** say what happened to the context: "Context summarized: 48
  messages, 152k → 9k tokens", or, when the fit guard trims a request (ADK,
  at most once a turn), "Context: 3 earlier turns left out of this request
  to fit the window". A sub-agent's own context is named first:
  "researcher: context summarized: …" (on Deep Agents each agent summarizes
  its own). A prompt smaller than the last is not taken for a trim.

### `/compact` and `/new`

- **`/compact [focus]`** summarizes the main agent's conversation now, as
  it is summarized when it grows: the newest 12% of the trigger and the
  newest turn stay word for word, and the rest becomes the summary the next
  requests start with; a second one replaces the first summary. The rest of
  the line tells the summary what to keep above all: `/compact keep the API
  decisions and open questions`. It runs as a turn does: "Summarizing the
  earlier conversation…" while it writes, then "Context summarized: …";
  Ctrl+C stops it, and nothing is written. `ctx` shows the smaller context
  from the next call on. With nothing before the kept tail it says "Nothing
  to summarize yet."; a summary that fails says "Could not summarize:
  <error>". LangGraph has none ("/compact is not supported on this
  backend"). An app runs the same with `BaseCLIApp.compact(focus=None)`, or
  calls the manager's `compact(session, focus=None)` itself
  (`supports_compact`; a native ADK YAML app, which no `BaseCLIApp` starts,
  has none either: its `compact` raises `NotImplementedError`).
- **`/new`** starts a new conversation without a restart, through
  `BaseCLIApp.start_new_session()`, which an app can call too: a new
  session id, the task box and the usage counters reset, and, with
  `auto_extract_session_facts` on, memories proposed for review from the
  conversation being left first when a turn ran in it, as at exit ("Looking for memories to
  propose…" while it runs). Its line names the conversation left when `/sessions` lists it
  (a turn ran in it). Ctrl+C stops it, and nothing changes. The model, the
  settings, "Allow for this session" grants, sandbox kernels and background
  jobs carry on, and a job reports to the conversation that started it,
  which its line names. `/clear` only clears the screen. The new
  conversation runs on the settings' `orchestrator`, the backend of new
  conversations; `/new deepagents` (or `adk`, `langgraph`;
  `start_new_session(backend=...)`) starts it on another, without saving
  it: the line names the backend ("New conversation <id> on deepagents."),
  the memories of the conversation left are proposed first, and an unknown
  backend, or one whose extra is not installed, is refused. A new backend
  starts a new manager on this run's services: in-process jobs run on,
  sandbox kernels keep their variables, "Allow for this session" grants
  hold, and the embedding model is not loaded again.

## CLI Commands

Built-in slash commands available in all apps:

| Command | Aliases | Description |
|---------|---------|-------------|
| `/help [command]` | | The commands by category; with a name or alias, that command's usage and examples |
| `/status` | | Show current session, workflow, and context-window status |
| `/clear` | | Clear the screen |
| `/exit` | `/quit` | Exit the application |
| `/settings` | `/set`, `/config` | Interactive settings editor: the default model, the `fast` and `strong` roles' models, thinking effort (saved to the project file) |
| `/models` | | The list the model pickers offer, each model's facts and why an entry is left out, and the providers in use (`/models <text>` searches the full listings, `/models refresh` lists again) |
| `/sandbox` | `/sb` | List / reset stateful sandbox sessions |
| `/jobs` | | List and manage long-running background jobs (`/jobs all`, `/jobs <id>`, `/jobs cancel <id>`, `/jobs clean`; `/jobs resume` tells the agent this conversation's finished jobs) |
| `/papers` | `/docs` | List knowledge-base documents (filter by source, query, --global) |
| `/sessions` | `/sess` | List the saved sessions of every installed backend: backend, title, turns, last update (delete one with `--delete=<id>`; `--backend=<name>` picks one of an id two backends keep) |
| `/compact` | | Summarize the conversation now, keeping the newest turns word for word (`/compact <focus>`: what the summary should keep above all) |
| `/new [backend]` | | Start a new conversation, on the backend named (`adk`, `deepagents`, `langgraph`) or the settings' `orchestrator`; the current one stays in `/sessions` |
| `/resume` | | Reopen a saved conversation, on the backend that keeps it: pick one (Enter opens it, Esc cancels), or `/resume <id>`; the last messages are shown again |
| `/memory` | | Review saved memories: the list of both scopes, one memory (Delete, History), `/memory <id>`, `/memory history` (restore a version), `/memory inbox` (Accept, Discard or Skip the proposals the extractor filed), `/memory stale` (not updated for 90 days) |

Apps can add more of their own. `research_demo`, for instance, registers
`/files` and `/kb-backfill` on top of the built-ins above.

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
    def register_commands(self) -> None:
        self.command_registry.register(MyCommand())
```

`register_commands()` runs last in the constructor, after the built-ins
(`_register_builtin_commands()`), so a command of the same name or alias
replaces a built-in one. An app that overrides `_register_builtin_commands()`
instead, calling `super()._register_builtin_commands()` first and then
registering its own, keeps working: its commands are registered just before
`register_commands()` runs.

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
| `CONTEXT_SUMMARIZING` | A summary of the earlier conversation is starting (reported while the stream waits on it) |
| `CONTEXT_TRIMMED` | The context was made smaller: a summary (`source="summarization"`, with `messages_summarized`, `tokens_before`, `tokens_after`), or older turns left out of one request on ADK (`source="fit_guard"`, with `turns_dropped`) |
| `LLM_REQUEST` / `LLM_RESPONSE` / `LLM_USAGE` | LLM traffic + token accounting |

On Deep Agents every event carries `metadata["agent"]`, the agent that
produced it. On ADK and Deep Agents `LLM_USAGE` names the agent whose call it
was (`agent`) and that agent's own model, with its `context_trigger` (the
prompt size its conversation is summarized at) and `window_known`; a
sub-agent's call is `purpose="subagent"` and a summary's
`purpose="summarization"`. A manager's `context_budget(agent_name=None)`
gives an agent's numbers, the root agent's (`root_agent_name`) by default.

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
- `memory_demo.py` — The memory store: markdown files in two scopes (save/search/update/restore, the index, proposals)
- `webfetch_demo.py` — Web fetching and summarization
- `websearch_demo.py` — Web search with multiple backends
- `yaml_agents/` — Agents defined in a YAML file (`agents.yaml`, prompts in files, a tool by its registered name), loaded with `load_agents_from_yaml`: a coordinator that delegates word counting to a researcher; run `python examples/yaml_agents/app.py` (see [Defining agents in YAML](#defining-agents-in-yaml))

**Full Applications**
- `research_demo/` — Full-featured research assistant with KB ingest + concept pages, memory files, sandbox execution, and durable sessions. Installed as the `research-demo` console script. It ships a model list of Gemini and Claude models: put the keys you have (`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`) in `~/.research_demo/.env`. It defines no other provider: OpenRouter, or an OpenAI-compatible service, is yours to define in `~/.research_demo/settings.json`, with its models, and its key goes in the same `.env` (the module docstring of `examples/research_demo/settings.py` shows OpenRouter's entry). A configured provider's model has no tier, so with only such a provider's key also choose the default model: pick one in `/settings`, which works while the workflow is down, or set `default_model` in `~/.research_demo/settings.json`.

Run examples:

```bash
export GOOGLE_API_KEY="your-key"
python examples/hello_agent.py

# Feature demos
python examples/fileops_demo.py

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
research-demo --continue           # the newest saved conversation
research-demo --resume <id>        # a saved conversation by id (/resume in the app: pick one)

# From a checkout without installing, run it from the repository root:
python -m examples.research_demo
```

Sessions are durable by default: every run gets a session id, named at exit
when it is stored; `--continue` and `--resume` reopen one, `/resume` switches
to another without a restart, and `/sessions` lists or deletes them (see
[Resuming a conversation](#resuming-a-conversation)). The demo adds
`/files` and `/kb-backfill` to the built-in commands.

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
│   ├── protected_paths.py        # Locations no tool reaches (permission engine + OS sandbox)
│   ├── logging.py                # Structlog configuration
│   ├── cli/
│   │   ├── app.py                # BaseCLIApp
│   │   ├── commands.py           # Command, CommandRegistry
│   │   ├── builtin_commands.py   # help/status/clear/exit/sandbox/jobs/resume/papers/sessions/models
│   │   ├── memory_command.py     # /memory
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
│   │   ├── retry.py              # is_rate_limit_error: a provider's 429 (the provider clients retry)
│   │   ├── tool_summaries.py     # Tool result one-liner summaries
│   │   ├── memory_services.py    # Settings → agentic_cli.memory arguments
│   │   ├── model_services.py     # Settings → agentic_cli.models arguments: keys, providers, entries; build_catalog
│   │   ├── model_settings.py     # ModelSettings, ThinkingSettings (per agent)
│   │   ├── model_wire.py         # What both adapters send to OpenRouter and OpenAI-compatible hosts
│   │   ├── langchain_models.py   # build_chat_model: a resolved model → LangChain chat model
│   │   ├── checkpointers.py      # open_checkpointer: memory / sqlite / Postgres (Deep Agents, LangGraph)
│   │   ├── llm_log.py            # LLMLogWriter: the raw_llm_logging JSONL file, redact_secrets
│   │   ├── tool_errors.py        # A raising tool's error result (ADK and Deep Agents)
│   │   ├── tool_output.py        # The tool result cap; full outputs in ./.{app_name}/tool-output/
│   │   ├── summary_prompt.py     # The summary prompt both backends use (LangChain's, MIT)
│   │   ├── sessions.py           # SessionRef, the active turn, the active tool call
│   │   ├── mcp.py                # MCPServerConfig → ADK toolset / langchain-mcp-adapters connection
│   │   ├── adk/
│   │   │   ├── manager.py                # GoogleADKWorkflowManager
│   │   │   ├── delegation.py             # Sub-agents as single_turn tools; their permission binding
│   │   │   ├── compaction.py             # AgentCliSummarizer: the main agent's conversation summarized
│   │   │   ├── fit_guard_plugin.py       # FitGuardPlugin: a request over its budget trimmed to fit
│   │   │   ├── overflow_plugin.py        # OverflowPlugin: a provider's overflow retried once, or one clear error
│   │   │   ├── model_factory.py          # build_adk_model: a resolved model → ADK model + config
│   │   │   ├── openai_compatible.py      # OpenAICompatibleLlm (the model_providers: OpenRouter, OpenAI-compatible hosts)
│   │   │   ├── anthropic_llm.py          # DirectAnthropicLlm (bare claude-* in ADK YAML)
│   │   │   ├── event_processor.py
│   │   │   ├── plugins.py                # LLMLoggingPlugin
│   │   │   ├── permission_plugin.py      # PermissionPlugin (capability gating)
│   │   │   ├── task_progress_plugin.py
│   │   │   └── tool_result_cap_plugin.py # ToolResultCapPlugin (the tool result cap)
│   │   ├── deepagents/                   # The Deep Agents backend (deepagents extra)
│   │   │   ├── manager.py                # DeepAgentsWorkflowManager
│   │   │   ├── agent_builder.py          # One create_agent graph per AgentConfig; task delegation
│   │   │   ├── middleware.py             # Permission gate, result cap, tool errors, active tool call, foreign reasoning
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
│   │   ├── exec_policy.py        # Which code may run on the host, from the settings
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
│   │   ├── memory_tools.py       # save/read/search/update/delete (store: agentic_cli.memory)
│   │   ├── _core/                # Shared planning/task logic, backend-neutral declarations
│   │   │   ├── planning.py
│   │   │   ├── tasks.py
│   │   │   ├── state_tools.py    # save_plan / get_plan / save_tasks / get_tasks, declared
│   │   │   └── delegation.py     # task, declared (Deep Agents' delegation tool)
│   │   ├── adk/state_tools.py       # ADK-native planning/task tools
│   │   ├── langgraph/state_tools.py # LangGraph-library planning/task tools (Deep Agents, LangGraph)
│   │   ├── skills/               # Agent Skills: SkillStore, ADK toolset, LangChain tools (langchain_tools.py)
│   │   ├── jobs/                 # Long-running jobs: JobManager, backends, job_* tools, resume.py
│   │   ├── os_sandbox/           # OS sandbox for execute_python: macOS seatbelt, Linux bubblewrap
│   │   ├── sandbox/
│   │   │   ├── __init__.py       # sandbox_execute tool
│   │   │   ├── manager.py        # SandboxManager + session lifecycle
│   │   │   ├── models.py
│   │   │   └── backends/         # Jupyter backends: docker (isolated), local (not isolated)
│   │   └── webfetch/             # Fetcher, converter, validator, robots, summarizer, saved pages
│   ├── memory/                   # Knowledge base + memory store; imports nothing else from agentic_cli
│   │   ├── __init__.py           # MemoryStore, Memory, Proposal, Candidate, the store's errors, extraction_prompt, looks_like_secret, EmbeddingService, EmbeddingConfig, MockEmbeddingService
│   │   ├── _core/                # Shared, private: embeddings.py, mock_embeddings.py, tokenize.py, secrets.py, io.py, log.py
│   │   ├── store/                # MemoryStore: memory files, scopes, search, index, history, proposals, 0.6 import
│   │   └── kb/                   # Knowledge base (agentic_cli.memory.kb)
│   │       ├── manager.py                # KnowledgeBaseManager(base_dir, embedding=, summarizer=), Summarizer: one folder on kb.sqlite
│   │       ├── _db.py                    # kb.sqlite: opened as untrusted, the schema, transactions, rows read one by one
│   │       ├── _search.py, _vectors.py   # Filters, FTS5 keyword ranking, rank fusion; a session's numpy vector matrix
│   │       ├── _import_06.py             # A 0.6 folder imported once; its old files into pre-0.7.0/
│   │       ├── knowledge_base.py         # KnowledgeBase (provisional): the scopes as one, and the ingestion pipeline; ScopedDocument, BackfillResult
│   │       ├── enrich.py                 # The summary/sidecar prompts and parser: one summarizer call per document
│   │       ├── models.py                 # Document, SearchResult, SourceType
│   │       ├── convert.py                # convert_document: trafilatura / html2text, pypdf, text
│   │       └── concepts.py, sidecar.py
│   ├── models/                   # Model knowledge (provisional); no provider SDK
│   │   ├── refs.py                   # provider:model references, bare ids, role names
│   │   ├── providers.py              # ProviderKind, ProviderSpec (model_providers), Provider, build_providers
│   │   ├── entries.py                # ModelEntry (the models setting), parse_entry_key
│   │   ├── records.py                # ModelRecord, ThinkingRule
│   │   ├── table.py, data/table.json # The shipped table: families, tiers, numbers, fallbacks
│   │   ├── roles.py                  # RoleMap: roles → references (unmapped: the list first)
│   │   ├── resolve.py                # resolve_model → ResolvedModel
│   │   ├── catalog.py                # ModelCatalog: entries, listings, validation; the list (offered), search; the listing cache
│   │   └── discovery/                # One lister per provider, over httpx
│   └── persistence/              # (sessions are persisted natively by each
│                                 #  orchestrator's store; the legacy JSON
│                                 #  SessionPersistence layer was removed)
├── examples/
│   ├── hello_agent.py            # Basic ADK example
│   ├── hello_deepagents.py       # Basic Deep Agents example
│   ├── hello_langgraph.py        # Basic LangGraph example (deprecated)
│   ├── *_demo.py                 # Feature demonstration scripts
│   ├── yaml_agents/              # Agents defined in YAML (agents.yaml, prompts/, tools.py, app.py)
│   └── research_demo/            # Full-featured example (pip-installable)
└── tests/
    ├── conftest.py               # MockContext and shared fixtures
    ├── tools/                    # Tool-specific tests (incl. sandbox)
    ├── workflow/                 # Workflow unit tests
    └── integration/              # ADK, Deep Agents and LangGraph pipeline tests
```

## License

MIT
