# Agentic CLI

A framework for building domain-specific agentic CLI applications powered by LLM agents.

## Overview

Agentic CLI provides the core infrastructure for building interactive CLI applications that leverage LLM agents for complex tasks. It offers:

- **Pluggable Orchestration**: Choose between Google ADK or LangGraph for agent workflows
- **Rich Terminal UI**: Thinking boxes, markdown rendering, and streaming responses via `thinking-prompt`
- **Declarative Agents**: Define agents with simple configuration objects
- **Built-in Tools**: Python execution, knowledge base search, web search
- **Session Persistence**: Save and restore conversation sessions
- **Type-safe Configuration**: Settings management with pydantic-settings

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BaseCLIApp                                  │
│  - Terminal UI (thinking-prompt)                                    │
│  - Command registry (/help, /status, /clear, etc.)                  │
│  - Message history                                                  │
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
    def get_custom_commands(self) -> list[Command]:
        return [MyCommand()]
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

## Examples

See the `examples/` directory for complete working examples:

- **hello_agent.py** - Simple assistant using Google ADK
- **hello_langgraph.py** - Same assistant using LangGraph orchestration
- **research_demo/** - Full-featured research assistant with memory, planning, and file operations

Run examples:

```bash
export GOOGLE_API_KEY="your-key"
python examples/hello_agent.py

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
│   ├── cli/
│   │   ├── app.py            # BaseCLIApp
│   │   └── commands.py       # Command, CommandRegistry
│   ├── workflow/
│   │   ├── events.py         # WorkflowEvent, EventType
│   │   ├── config.py         # AgentConfig
│   │   ├── adk_manager.py    # GoogleADKWorkflowManager
│   │   └── langgraph/        # LangGraph submodule
│   │       ├── manager.py    # LangGraphWorkflowManager
│   │       ├── state.py      # AgentState, CheckpointData
│   │       ├── persistence/  # Checkpointers and stores
│   │       └── tools/        # Shell, file search tools
│   ├── tools/
│   │   └── executor.py       # SafePythonExecutor
│   └── knowledge_base/
│       └── manager.py        # KnowledgeBaseManager
├── examples/
│   ├── hello_agent.py
│   ├── hello_langgraph.py
│   └── research_demo/        # Full-featured example
└── tests/
```

## License

MIT
