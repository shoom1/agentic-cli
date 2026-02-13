# Agentic CLI

A framework for building domain-specific agentic CLI applications powered by LLM agents.

## Overview

Agentic CLI provides the core infrastructure for building interactive CLI applications that leverage LLM agents for complex tasks. It offers:

- **Pluggable Orchestration**: Choose between Google ADK or LangGraph for agent workflows
- **Rich Terminal UI**: Thinking boxes, markdown rendering, and streaming responses via `thinking-prompt`
- **Declarative Agents**: Define agents with simple configuration objects
- **Built-in Tools**: Python execution, file operations, knowledge base, web search, web fetch, arXiv search
- **Session Persistence**: Save and restore conversation sessions
- **Type-safe Configuration**: Settings management with pydantic-settings

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BaseCLIApp                                  │
│  - Terminal UI (thinking-prompt)                                    │
│  - Command registry (/help, /status, /clear, etc.)                  │
│  - Message history                                                  │
│  - Background initialization (no first-message lag)                 │
│  - Task progress display in thinking box                            │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BaseWorkflowManager                              │
│  - Agent orchestration                                              │
│  - Event streaming                                                  │
│  - Session management                                               │
├─────────────────────────────┬───────────────────────────────────────┤
│   GoogleADKWorkflowManager  │     LangGraphWorkflowManager          │
│   (Default)                 │     (Optional: langgraph extra)       │
└─────────────────────────────┴───────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         AgentConfig[]                               │
│  - name, prompt, tools, sub_agents, description                     │
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
    return {"greeting": f"Hello, {name}!"}

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
    model="gemini-2.0-flash",  # Optional: auto-detected from API keys
)
```

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
| Retry handling | Manual | Built-in with backoff |

### Auto-selection via Settings

```python
from agentic_cli import create_workflow_manager_from_settings

settings = BaseSettings(orchestrator="langgraph")  # or "adk"
manager = create_workflow_manager_from_settings(agent_configs=AGENTS, settings=settings)
```

## Configuration

### BaseSettings

All settings can be configured via environment variables with the `AGENTIC_` prefix or in a `.env` file:

```python
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
| `orchestrator` | `AGENTIC_ORCHESTRATOR` | "adk" | Orchestrator: adk or langgraph |
| `workspace_dir` | `AGENTIC_WORKSPACE_DIR` | ~/.agentic | Storage directory |
| `log_level` | `AGENTIC_LOG_LEVEL` | "warning" | Logging level |
| `tavily_api_key` | `TAVILY_API_KEY` | None | Tavily API key for web search |
| `brave_api_key` | `BRAVE_API_KEY` | None | Brave Search API key |
| `search_backend` | `AGENTIC_SEARCH_BACKEND` | Auto | Web search provider (tavily/brave) |
| `webfetch_model` | `AGENTIC_WEBFETCH_MODEL` | Auto | Model for web content summarization |

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

Tools are regular Python functions with type hints and docstrings:

```python
def search_database(query: str, limit: int = 10) -> dict:
    """Search the database for matching records.

    Args:
        query: Search query string
        limit: Maximum number of results (default: 10)

    Returns:
        Dict with results and count
    """
    results = db.search(query, limit=limit)
    return {"results": results, "count": len(results)}
```

### Built-in Tools

#### SafePythonExecutor

Execute Python code in a sandboxed environment:

```python
from agentic_cli.tools import SafePythonExecutor

executor = SafePythonExecutor(default_timeout=30)
result = executor.execute("""
import numpy as np
data = np.array([1, 2, 3, 4, 5])
np.mean(data)
""")
# result = {"success": True, "result": "3.0", "output": "", "error": ""}
```

Allowed modules: numpy, pandas, scipy, math, json, datetime, collections, itertools, re, random

#### Web Search

Search the web using pluggable backends (Tavily or Brave):

```python
from agentic_cli.tools import web_search

# Use as agent tool
agent = AgentConfig(
    name="researcher",
    tools=[web_search],
)

# Or call directly
results = await web_search("Python async programming", max_results=5)
# Returns: {"results": [{"title": "...", "url": "...", "snippet": "..."}], ...}
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
# Returns: {"url": "...", "summary": "...", "content_length": ...}
```

Features: URL validation, robots.txt compliance, SSRF protection, content caching.

#### ArXiv Search

Search and analyze academic papers:

```python
from agentic_cli.tools import search_arxiv, fetch_arxiv_paper

# Search papers
results = search_arxiv("transformer attention", max_results=10, categories=["cs.CL"])

# Fetch paper details
paper = fetch_arxiv_paper("1706.03762")  # "Attention Is All You Need"
```

#### File Operations

Categorized file tools with permission levels:

**READ Tools (Safe)**

```python
from agentic_cli.tools import read_file, grep, glob, list_dir, diff_compare

# Read file contents
result = read_file("src/main.py", offset=0, limit=100)
# Returns: {"success": True, "content": "...", "size": 1234, "lines_read": 100}

# Search for patterns (ripgrep-like)
result = grep("def.*async", path="src/", file_pattern="*.py", recursive=True)
# Returns: {"success": True, "matches": [...], "file_count": 5}

# Find files by pattern
result = glob("**/*.py", path="src/", include_metadata=True)
# Returns: {"success": True, "files": [...], "count": 42}

# List directory contents
result = list_dir("src/", include_hidden=False)
# Returns: {"success": True, "entries": [...]}

# Compare files or text
result = diff_compare(source_a="old.txt", source_b="new.txt")
# Returns: {"success": True, "diff": "...", "similarity": 0.85}
```

**WRITE Tools (Caution)**

```python
from agentic_cli.tools import write_file, edit_file

# Write file (creates or overwrites)
result = write_file("output.txt", content="Hello, World!", create_dirs=True)
# Returns: {"success": True, "path": "...", "size": 13, "created": True}

# Edit file (sed-like replacement)
result = edit_file("config.py", old_text="DEBUG = True", new_text="DEBUG = False")
# Returns: {"success": True, "replacements": 1}
```

#### Shell Executor

> **Note**: Shell execution is currently **disabled by default** while security safeguards are being validated.

The shell tool provides layered security with 8 defense layers:
- Input preprocessing (encoding/obfuscation detection)
- Command tokenization and classification
- Path analysis and sandboxing
- Risk assessment with approval workflows
- Audit logging

```python
from agentic_cli.tools import shell_executor, is_shell_enabled

# Check if shell is enabled
if is_shell_enabled():
    result = shell_executor("ls -la", working_dir="/project")
else:
    print("Shell tool disabled pending security validation")
```

#### KnowledgeBaseManager

Semantic search over documents:

```python
from agentic_cli.knowledge_base import KnowledgeBaseManager, SourceType

kb = KnowledgeBaseManager(settings=settings)

# Ingest a document
doc = kb.ingest_document(
    content="Machine learning is...",
    title="ML Introduction",
    source_type=SourceType.WEB,
    source_url="https://example.com/ml",
)

# Search
results = kb.search("neural networks", top_k=5)
```

## CLI Commands

Built-in slash commands available in all apps:

| Command | Aliases | Description |
|---------|---------|-------------|
| `/help` | | Show available commands |
| `/status` | | Show session and workflow status |
| `/clear` | | Clear the screen |
| `/exit` | `/quit` | Exit the application |

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

# In your app
class MyApp(BaseCLIApp):
    def register_commands(self) -> None:
        super().register_commands()
        self.command_registry.register(MyCommand())
```

## Events

WorkflowEvent types for UI integration:

| EventType | Description | Metadata |
|-----------|-------------|----------|
| `TEXT` | Final text response | session_id |
| `THINKING` | Model reasoning | session_id |
| `TOOL_CALL` | Tool invocation | tool_name, tool_args |
| `TOOL_RESULT` | Tool result | tool_name, result, success |
| `CODE_EXECUTION` | Code execution result | outcome |
| `ERROR` | Error message | recoverable, error_code |
| `USER_INPUT_REQUIRED` | Tool needs user input | request_id, prompt |
| `TASK_PROGRESS` | Task graph update | current_task_description, progress |

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

### Task Progress Display

When using plan checkboxes or task tools, the CLI thinking box dynamically shows task progress:

```
Calling: web_search
--- Tasks: 1/3 ---
Research:
  [x] Gather data
  [ ] Analyze results
Writing:
  [ ] Draft report
```

Status icons (task tools):
- `[x]` Completed
- `[>]` In progress
- `[ ]` Pending
- `[-]` Cancelled

## Examples

See the `examples/` directory for complete working examples:

**Getting Started**
- **hello_agent.py** - Simple assistant using Google ADK
- **hello_langgraph.py** - Same assistant using LangGraph orchestration

**Feature Demos**
- **arxiv_demo.py** - ArXiv paper search and analysis
- **fileops_demo.py** - File operation tools (read, write, grep, glob)
- **memory_demo.py** - Memory persistence system
- **planning_demo.py** - Task graph and planning tools
- **shell_demo.py** - Shell security pattern detection
- **webfetch_demo.py** - Web fetching and summarization
- **websearch_demo.py** - Web search with multiple backends

**Full Applications**
- **research_demo/** - Full-featured research assistant with memory, planning, and file operations

Run examples:

```bash
export GOOGLE_API_KEY="your-key"
python examples/hello_agent.py

# Feature demos (no API key needed for some)
python examples/fileops_demo.py
python examples/shell_demo.py

# Or with LangGraph (requires langgraph extra)
pip install agentic-cli[langgraph]
python examples/hello_langgraph.py

# Research demo (full features)
python -m examples.research_demo
```

## Development

### Running Tests

```bash
# With conda
conda run -n agenticcli python -m pytest tests/ -v

# With pip
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=agentic_cli
```

### Project Structure

```
agentic-cli/
├── src/agentic_cli/
│   ├── __init__.py           # Package exports
│   ├── config.py             # BaseSettings, SettingsContext
│   ├── constants.py          # Shared constants (truncation, limits)
│   ├── resolvers.py          # Model/path constants (GOOGLE_MODELS, etc.)
│   ├── settings_persistence.py
│   ├── logging.py            # Structlog configuration
│   ├── cli/
│   │   ├── app.py            # BaseCLIApp
│   │   ├── commands.py       # Command, CommandRegistry
│   │   ├── builtin_commands.py
│   │   ├── workflow_controller.py  # Workflow orchestration
│   │   ├── message_processor.py    # Event stream processing
│   │   └── settings*.py      # Settings UI (introspection, dialog)
│   ├── workflow/
│   │   ├── base_manager.py   # BaseWorkflowManager (abstract)
│   │   ├── events.py         # WorkflowEvent, EventType
│   │   ├── config.py         # AgentConfig
│   │   ├── context.py        # Context variables for tools
│   │   ├── thinking.py       # ThinkingDetector
│   │   ├── task_progress.py  # Task progress events
│   │   ├── tool_summaries.py # Tool result summaries
│   │   ├── settings.py       # WorkflowSettingsMixin
│   │   ├── adk/              # ADK orchestrator
│   │   │   ├── manager.py    # GoogleADKWorkflowManager
│   │   │   ├── event_processor.py  # ADK event processing
│   │   │   └── llm_event_logger.py # LLM traffic logging
│   │   └── langgraph/        # LangGraph submodule
│   │       ├── manager.py    # LangGraphWorkflowManager
│   │       ├── graph_builder.py # Graph + LLM factory
│   │       ├── state.py      # AgentState
│   │       ├── persistence/  # Checkpointers and stores
│   │       └── tools/        # LangChain-compatible wrappers
│   ├── tools/
│   │   ├── registry.py       # ToolRegistry, ToolCategory, PermissionLevel
│   │   ├── executor.py       # SafePythonExecutor
│   │   ├── arxiv_tools.py    # search_arxiv, fetch_arxiv_paper
│   │   ├── arxiv_source.py   # ArxivSearchSource
│   │   ├── execution_tools.py # execute_python
│   │   ├── interaction_tools.py # ask_clarification
│   │   ├── knowledge_tools.py # search/ingest_to_knowledge_base
│   │   ├── file_read.py      # read_file, diff_compare
│   │   ├── file_write.py     # write_file, edit_file
│   │   ├── grep_tool.py      # grep (pattern search)
│   │   ├── glob_tool.py      # glob, list_dir (file discovery)
│   │   ├── search.py         # Web search (Tavily, Brave)
│   │   ├── webfetch_tool.py  # Web content fetching
│   │   ├── memory_tools.py   # MemoryStore, save/search_memory
│   │   ├── planning_tools.py # PlanStore, save/get_plan
│   │   ├── task_tools.py     # TaskStore, save/get_tasks
│   │   ├── hitl_tools.py     # request_approval, ApprovalManager
│   │   └── shell/            # Shell executor with security
│   │       ├── executor.py   # Main entry point (disabled by default)
│   │       ├── tokenizer.py  # Command parsing
│   │       ├── classifier.py # Risk classification
│   │       ├── sandbox.py    # Execution sandboxing
│   │       └── audit.py      # Security logging
│   ├── knowledge_base/
│   │   ├── manager.py        # KnowledgeBaseManager
│   │   ├── models.py         # Document, SearchResult
│   │   ├── embeddings.py     # EmbeddingService
│   │   ├── vector_store.py   # VectorStore
│   │   ├── _mocks.py         # MockEmbeddingService, MockVectorStore
│   │   └── sources.py        # ArxivSearchSource
│   └── persistence/
│       ├── session.py        # SessionPersistence
│       ├── artifacts.py      # ArtifactManager
│       └── _utils.py         # Atomic write utilities
├── examples/
│   ├── hello_agent.py        # Basic ADK example
│   ├── hello_langgraph.py    # Basic LangGraph example
│   ├── *_demo.py             # Feature demonstration scripts
│   └── research_demo/        # Full-featured example
└── tests/
```

## License

MIT
