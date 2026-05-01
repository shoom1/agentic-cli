# Agentic CLI

A framework for building domain-specific agentic CLI applications powered by LLM agents.

## Overview

Agentic CLI provides the core infrastructure for building interactive CLI applications that leverage LLM agents for complex tasks. It offers:

- **Pluggable Orchestration**: Choose between Google ADK or LangGraph for agent workflows
- **Rich Terminal UI**: Dual thinking boxes, markdown rendering, and streaming responses via `thinking-prompt`
- **Declarative Agents**: Define agents with simple configuration objects
- **Native Tool Architecture**: Backend-specific tool factories for ADK and LangGraph, with automatic HITL confirmation for dangerous tools
- **Built-in Tools**: Python execution, stateful sandbox, file operations, web search, web fetch, arXiv search
- **Knowledge Base**: Semantic + BM25 hybrid search with RRF fusion, per-document markdown sidecars, and agent-authored concept pages
- **Semantic Memory**: Embedding-backed memory with lifecycle management, contradiction detection, and forgetting policy
- **Tool Reflection**: Bounded per-tool heuristic memory learned from failures
- **Session Save/Resume**: Persistent conversations across CLI restarts
- **Context Window Management**: Native trim detection and token-usage visibility
- **Dynamic Model Registry**: Live model discovery from provider APIs
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
├─────────────────────────────┬───────────────────────────────────────┤
│   GoogleADKWorkflowManager  │     LangGraphWorkflowManager          │
│   (default)                 │     (optional: langgraph extra)       │
│   + PermissionPlugin        │     + wrap_tool_for_permission        │
│   + LLMLoggingPlugin        │     + native ToolNode                 │
│   + TaskProgressPlugin      │                                       │
└─────────────────────────────┴───────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         AgentConfig[]                               │
│  - name, prompt, tools, sub_agents, description, model              │
└─────────────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.12+
- Google API key (for Gemini models) or Anthropic API key (for Claude models)

## Installation

### Basic Installation (Google ADK)

```bash
pip install agentic-cli
```

### With LangGraph Support

```bash
pip install agentic-cli[langgraph]
```

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

# Define your tools
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

## Workflow Managers

### GoogleADKWorkflowManager (Default)

Uses Google's Agent Development Kit for orchestration. Best for:
- Simple agent hierarchies
- Google Gemini models with native thinking support
- Quick prototyping

```python
from agentic_cli import GoogleADKWorkflowManager

manager = GoogleADKWorkflowManager(
    agent_configs=AGENTS,
    settings=settings,
    model="gemini-2.5-flash",  # Optional: auto-detected from API keys
)
```

ADK integrations:
- **PermissionPlugin** — evaluates each tool's declared capabilities against the permission engine and prompts the user when no rule matches
- **LLMLoggingPlugin** — structured logging of LLM requests/responses
- **TaskProgressPlugin** — streams the task checklist into its own thinking box

### LangGraphWorkflowManager

Uses LangGraph for orchestration. Best for:
- Cyclical workflows (self-validation, iterative refinement)
- Model-agnostic operation (OpenAI, Anthropic, Google via GenAI)
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
- **Explicit provider support**: Uses `langchain-google-genai` for Gemini (not VertexAI)
- **Thinking mode**: Native support for Claude and Gemini thinking/reasoning
- **Retry policies**: Automatic retry with exponential backoff
- **Permission wrapper**: Wraps each tool to gate execution through the permission engine
- **Event streaming**: Real-time workflow events via `WorkflowEvent`

Requires: `pip install agentic-cli[langgraph]`

### Comparison

| Feature | Google ADK | LangGraph |
|---------|------------|-----------|
| Setup complexity | Simple | Moderate |
| Cyclical workflows | Limited | Native |
| Multi-provider | Google only | OpenAI, Anthropic, Google (GenAI) |
| State persistence | In-memory | Memory, PostgreSQL, or SQLite |
| Thinking support | Native (Gemini) | Native (Claude & Gemini) |
| Retry handling | Built-in | Built-in with backoff |
| Permission gate | PermissionPlugin | wrap_tool_for_permission |
| Context trimming | Native | Native |

### Auto-selection via Settings

```python
from agentic_cli import create_workflow_manager_from_settings

settings = BaseSettings(orchestrator="langgraph")  # or "adk" (default)
manager = create_workflow_manager_from_settings(agent_configs=AGENTS, settings=settings)
```

## Configuration

### BaseSettings

Settings are organized into composable mixins (`AppSettingsMixin`, `CLISettingsMixin`, `WorkflowSettingsMixin`). `BaseSettings` composes all three.

All settings can be configured via environment variables with the `AGENTIC_` prefix or in a `.env` file:

```python
from pathlib import Path
from pydantic_settings import SettingsConfigDict
from agentic_cli import BaseSettings

class MySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MYAPP_",  # Custom prefix
    )
    app_name: str = "my_app"
    workspace_dir: Path = Path.home() / ".my_app"
```

### Key Settings

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| `google_api_key` | `GOOGLE_API_KEY` | None | Google API key for Gemini |
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | None | Anthropic API key for Claude |
| `default_model` | `AGENTIC_DEFAULT_MODEL` | Auto | Model to use |
| `thinking_effort` | `AGENTIC_THINKING_EFFORT` | "medium" | Thinking level: none, low, medium, high |
| `orchestrator` | `AGENTIC_ORCHESTRATOR` | "adk" | Orchestrator: `adk` or `langgraph` |
| `langgraph_checkpointer` | `AGENTIC_LANGGRAPH_CHECKPOINTER` | "memory" | `memory`, `postgres`, or `null` |
| `workspace_dir` | `AGENTIC_WORKSPACE_DIR` | ~/.agentic | Storage directory |
| `log_level` | `AGENTIC_LOG_LEVEL` | "warning" | Logging level |
| `tavily_api_key` | `TAVILY_API_KEY` | None | Tavily API key for web search |
| `brave_api_key` | `BRAVE_API_KEY` | None | Brave Search API key |
| `search_backend` | `AGENTIC_SEARCH_BACKEND` | Auto | Web search provider (tavily/brave) |
| `webfetch_model` | `AGENTIC_WEBFETCH_MODEL` | Auto | Model for web content summarization |
| `retry_max_attempts` | `AGENTIC_RETRY_MAX_ATTEMPTS` | 3 | Max retries on transient errors |
| `retry_initial_delay` | `AGENTIC_RETRY_INITIAL_DELAY` | 2.0 | Initial retry backoff (seconds) |

Settings are persisted by `SettingsPersistence` (atomic writes) and editable at runtime via the `/settings` command.

### Settings Context

For multi-tenant or isolated contexts:

```python
from agentic_cli import SettingsContext

with SettingsContext(custom_settings):
    # All code here uses custom_settings
    result = my_tool()
```

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
| `name` | str | Unique identifier |
| `prompt` | str \| Callable | System instruction |
| `tools` | list[Callable] | Available tool functions |
| `sub_agents` | list[str] | Names of agents this one can delegate to |
| `description` | str | Short description for routing |
| `model` | str \| None | Model override (defaults to manager's model) |

## Tools

### Creating Tools

Tools are regular Python functions with type hints and docstrings. **All tools return `{"success": bool, ...}` dicts** — never raise exceptions.

```python
from agentic_cli.tools import register_tool, ToolCategory
from agentic_cli.workflow.permissions import Capability, EXEMPT

@register_tool(
    category=ToolCategory.NETWORK,
    capabilities=[Capability("http.read")],
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

Registering via `@register_tool` is optional — you can pass raw callables into `AgentConfig.tools`. Registering gives the tool metadata for the registry, capability-aware permission gating, and tool-summary formatting.

### Capabilities

Tool access is gated by the **permission engine** (see the HITL section below). Each registered tool declares what it touches via `capabilities=`:

- **`Capability("namespace.action", target_arg="...")`** — e.g. `Capability("filesystem.write", target_arg="path")`, `Capability("http.read", target_arg="url")`, `Capability("shell.exec", target_arg="command")`. The engine resolves `target_arg` against the actual tool-call arguments and matches against rules.
- **`EXEMPT`** — opts the tool out of the engine entirely. Reserved for backend-internal tools (e.g. ADK `transfer_to_agent`, backend state tools).

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

#### ArXiv Search

Search and fetch academic papers:

```python
from agentic_cli.tools import search_arxiv, fetch_arxiv_paper

results = search_arxiv("transformer attention", max_results=10, categories=["cs.CL"])
paper = fetch_arxiv_paper("1706.03762")  # "Attention Is All You Need"
```

The `ArxivSearchSource` (in `knowledge_base/sources.py`) is cached via the service registry and reused across tools. To pull a paper into the KB, use `ingest_arxiv_paper(arxiv_id)` — it composes the arxiv source with `kb_manager`.

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
    kb_search, kb_ingest_text, kb_ingest_file, kb_ingest_url,
    kb_read, kb_list,
    kb_write_concept, kb_search_concepts,
    KB_READER_TOOLS, KB_WRITER_TOOLS,
)
```

| Tool | Purpose |
|------|---------|
| `kb_search` | Hybrid BM25 + vector search with RRF fusion, filters by source/date |
| `kb_ingest_text` | Ingest in-memory text content (no FS or network access) |
| `kb_ingest_file` | Ingest a local file; declares `filesystem.read(path)` for the permission engine |
| `kb_ingest_url` | Ingest content from an http(s) URL; routed through the hardened `ContentFetcher` and declares `http.read(url)` |
| `kb_read` | Return the per-document markdown sidecar (lazy-generated on first read) |
| `kb_list` | List documents, optionally filtered |
| `kb_write_concept` | Agent-authored concept/summary page (slugged, merged on overwrite) |
| `kb_search_concepts` | Search concept pages (title-weighted, case-insensitive) |

The single `kb_ingest` tool was split in 0.5.2 so each entry point declares the right capability — text-only stays under `kb.write`, while file and URL ingestion correctly trip `filesystem.read` / `http.read` checks. For arXiv papers, prefer `ingest_arxiv_paper(arxiv_id)`.

Bundle convenience:

```python
researcher = AgentConfig(name="researcher", prompt=..., tools=KB_WRITER_TOOLS)
reader     = AgentConfig(name="reader",     prompt=..., tools=KB_READER_TOOLS)
```

Direct manager API for embedding in custom flows:

```python
from agentic_cli.knowledge_base import KnowledgeBaseManager, SourceType

kb = KnowledgeBaseManager(settings=settings)
doc = await kb.ingest_document(
    content="Machine learning is...",
    title="ML Introduction",
    source_type=SourceType.WEB,
    source_url="https://example.com/ml",
)
results = kb.search("neural networks", top_k=5)        # hybrid by default
concepts = kb.concepts.search("attention")             # concept pages
await kb.backfill_sidecars()                           # regenerate markdown summaries
```

The KB also maintains `index.md` and an append-only `ingest_log.md` audit trail. Source-type constants (`arxiv`, `ssrn`, `web`, `internal`, `user`, `local`) live on `SourceType`.

#### Memory Tools

Semantic, lifecycle-aware memory:

```python
from agentic_cli.tools import memory_tools
# save_memory, search_memory, update_memory, delete_memory
```

`MemoryStore` features:
- Embedding-backed semantic search
- Contradiction detection via `store_with_similarity_check`
- `ForgettingPolicy` with `apply_forgetting()` for bounded retention
- Archive filtering and `load_all` with tag/source filters

#### Tool Reflection

Bounded heuristic memory learned from tool failures:

```python
from agentic_cli.tools import reflection_tools
# save_reflection(tool_name, error_summary, heuristic)
```

Each tool keeps at most N reflections (FIFO eviction). Reflections can be injected into tool descriptions to help agents avoid repeating mistakes. Wired via session-end hook.

#### HITL (Human-in-the-Loop)

Tool calls are gated by the **permission engine** (`workflow/permissions/`). Each tool declares a list of capabilities (e.g. `filesystem.write(path=...)`); the engine evaluates them against rules from four sources (builtin defaults, user `~/.{app_name}/settings.json`, project `./.{app_name}/settings.json`, in-memory session). When no rule matches, the user is prompted with `Allow once / Allow for session / Allow always (save to project) / Deny`. Always-grants persist into the project settings file so the next run picks them up automatically.

See `docs/superpowers/specs/2026-04-18-permissions-system-design.md` for the full design.

## CLI Commands

Built-in slash commands available in all apps:

| Command | Aliases | Description |
|---------|---------|-------------|
| `/help` | | Show available commands and usage information |
| `/status` | | Show current session, workflow, and context-window status |
| `/clear` | | Clear the screen |
| `/exit` | `/quit` | Exit the application |
| `/settings` | | Interactive settings editor (with persistence) |
| `/sandbox` | `/sb` | List / reset stateful sandbox sessions |
| `/papers` | `/docs` | List knowledge-base documents (filter by source, query, --global) |
| `/sessions` | `/sess` | List saved sessions (and delete with `--delete=<id>`) |

Apps can add more. Examples like `research_demo` ship with commands like `/save`, `/resume`, and `/kb-backfill`.

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
| `CODE_EXECUTION` | Code execution result |
| `EXECUTABLE_CODE` | Code the model intends to run |
| `FILE_DATA` | Binary/file payload from a tool |
| `ERROR` | Error message (`recoverable`, `error_code` metadata) |
| `TASK_PROGRESS` | Task checklist update (drives the task box) |
| `CONTEXT_TRIMMED` | Context trimming event (drop count, remaining tokens) |
| `LLM_REQUEST` / `LLM_RESPONSE` / `LLM_USAGE` | LLM traffic + token accounting |

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
- `hello_langgraph.py` — Same assistant using LangGraph orchestration

**Feature Demos**
- `arxiv_demo.py` — ArXiv paper search and analysis
- `fileops_demo.py` — File operation tools (read, write, grep, glob)
- `memory_demo.py` — Memory persistence system (save/search/update/delete)
- `shell_demo.py` — Shell security pattern detection
- `webfetch_demo.py` — Web fetching and summarization
- `websearch_demo.py` — Web search with multiple backends

**Full Applications**
- `research_demo/` — Full-featured research assistant with KB ingest + concept pages, semantic memory, sandbox execution, session save/resume. Installable as a console script.

Run examples:

```bash
export GOOGLE_API_KEY="your-key"
python examples/hello_agent.py

# Feature demos
python examples/fileops_demo.py
python examples/shell_demo.py

# Or with LangGraph
pip install agentic-cli[langgraph]
python examples/hello_langgraph.py

# Research demo (full features)
python -m examples.research_demo
```

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
│   │   ├── builtin_commands.py   # help/status/clear/exit/sandbox/papers/sessions
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
│   │   ├── models.py             # Shared data models
│   │   ├── config.py             # AgentConfig
│   │   ├── settings.py           # WorkflowSettingsMixin
│   │   ├── retry.py              # Exponential-backoff retry
│   │   ├── tool_summaries.py     # Tool result one-liner summaries
│   │   ├── adk/
│   │   │   ├── manager.py                # GoogleADKWorkflowManager
│   │   │   ├── event_processor.py
│   │   │   ├── plugins.py                # LLMLoggingPlugin
│   │   │   ├── permission_plugin.py      # PermissionPlugin (capability gating)
│   │   │   └── task_progress_plugin.py
│   │   ├── langgraph/
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
│   │   ├── execution_tools.py    # execute_python
│   │   ├── interaction_tools.py  # ask_clarification
│   │   ├── knowledge_tools.py    # kb_search / kb_ingest_text / kb_ingest_file /
│   │   │                         #   kb_ingest_url / kb_read / kb_list /
│   │   │                         #   kb_write_concept / kb_search_concepts
│   │   ├── file_read.py          # read_file, diff_compare
│   │   ├── file_write.py         # write_file, edit_file (atomic)
│   │   ├── grep_tool.py          # grep
│   │   ├── glob_tool.py          # glob, list_dir
│   │   ├── search.py             # web_search (Tavily / Brave)
│   │   ├── webfetch_tool.py      # web_fetch (orchestrator)
│   │   ├── pdf_utils.py          # PDF text extraction helpers
│   │   ├── memory_tools.py       # save/search/update/delete + MemoryStore
│   │   ├── reflection_tools.py   # save_reflection + ToolReflectionStore
│   │   ├── _core/                # Shared planning/task logic
│   │   │   ├── planning.py
│   │   │   └── tasks.py
│   │   ├── adk/state_tools.py       # ADK-native planning/task tools
│   │   ├── langgraph/state_tools.py # LangGraph-native planning/task tools
│   │   ├── sandbox/
│   │   │   ├── __init__.py       # sandbox_execute tool
│   │   │   ├── manager.py        # SandboxManager + session lifecycle
│   │   │   ├── models.py
│   │   │   └── backends/         # Jupyter local backend
│   │   ├── shell/                # Layered shell executor (disabled by default)
│   │   │   ├── executor.py
│   │   │   ├── os_sandbox/       # macOS seatbelt / Linux namespace sandboxes
│   │   │   └── ...               # tokenizer, classifier, sandbox, audit
│   │   └── webfetch/             # Fetcher, converter, validator, robots, summarizer
│   ├── knowledge_base/
│   │   ├── manager.py            # KnowledgeBaseManager (+ .concepts)
│   │   ├── models.py             # Document, SearchResult, SourceType, …
│   │   ├── embeddings.py         # EmbeddingService
│   │   ├── vector_store.py       # FAISS-backed vector store
│   │   ├── bm25_index.py         # BM25 index (hybrid search)
│   │   ├── concepts.py           # ConceptStore (agent-authored pages)
│   │   ├── sidecar.py            # Per-doc markdown sidecar render/parse
│   │   ├── sources.py            # SearchSource + ArxivSearchSource
│   │   └── _mocks.py             # MockEmbeddingService, MockVectorStore, mock BM25
│   └── persistence/
│       └── session.py            # SessionPersistence (save/resume)
├── examples/
│   ├── hello_agent.py            # Basic ADK example
│   ├── hello_langgraph.py        # Basic LangGraph example
│   ├── *_demo.py                 # Feature demonstration scripts
│   └── research_demo/            # Full-featured example (pip-installable)
└── tests/
    ├── conftest.py               # MockContext and shared fixtures
    ├── tools/                    # Tool-specific tests (incl. sandbox)
    ├── workflow/                 # Workflow unit tests
    └── integration/              # ADK + LangGraph pipeline tests
```

## License

MIT
