# Agentic CLI - Shared Framework for Agentic Applications

## Project Overview

Agentic CLI is a shared library providing the core infrastructure for building domain-specific CLI applications powered by LLM agents.

## Tech Stack

- **Language**: Python 3.12+
- **CLI UI**: `thinking_prompt` - enhanced CLI with thinking boxes and markdown
- **Workflow**: Google ADK + LangGraph - dual orchestration backends (selectable via settings)
- **Config**: `pydantic-settings` - type-safe configuration
- **Logging**: `structlog` - structured logging

## Project Structure

```
agentic-cli/
├── src/agentic_cli/
│   ├── __init__.py           # Package exports, lazy imports
│   ├── config.py             # BaseSettings (pydantic-settings)
│   ├── constants.py          # Shared constants, truncate()
│   ├── resolvers.py          # Model/path constants (GOOGLE_MODELS, etc.)
│   ├── settings_persistence.py
│   ├── logging.py
│   ├── cli/
│   │   ├── app.py            # BaseCLIApp
│   │   ├── commands.py       # Command, CommandRegistry
│   │   ├── builtin_commands.py
│   │   ├── workflow_controller.py  # WorkflowController, factory
│   │   ├── message_processor.py
│   │   └── settings*.py      # Settings UI (introspection, dialog)
│   ├── workflow/
│   │   ├── base_manager.py   # BaseWorkflowManager (abstract)
│   │   ├── task_progress.py  # build_task_progress_event(), parse_plan_progress()
│   │   ├── events.py         # WorkflowEvent, EventType
│   │   ├── thinking.py       # ThinkingDetector
│   │   ├── config.py         # AgentConfig
│   │   ├── context.py        # ContextVars for tool access (get_context_*())
│   │   ├── adk/              # ADK orchestrator
│   │   │   ├── manager.py    # GoogleADKWorkflowManager
│   │   │   ├── event_processor.py  # ADKEventProcessor
│   │   │   └── llm_event_logger.py # LLM traffic logging
│   │   └── langgraph/        # LangGraph orchestrator
│   │       ├── manager.py    # LangGraphWorkflowManager
│   │       ├── graph_builder.py # LangGraphBuilder (graph + LLM factory)
│   │       ├── state.py
│   │       ├── persistence/  # Checkpointers, stores
│   │       └── tools/        # LangChain-compatible wrappers
│   ├── tools/
│   │   ├── registry.py       # ToolRegistry, @register_tool, ToolCategory, PermissionLevel
│   │   ├── executor.py       # SafePythonExecutor
│   │   ├── knowledge_tools.py # search_knowledge_base, ingest_to_knowledge_base
│   │   ├── arxiv_tools.py    # search_arxiv, fetch_arxiv_paper, analyze_arxiv_paper
│   │   ├── execution_tools.py # execute_python
│   │   ├── interaction_tools.py # ask_clarification
│   │   ├── file_read.py      # read_file, diff_compare
│   │   ├── file_write.py     # write_file, edit_file
│   │   ├── glob_tool.py      # glob
│   │   ├── grep_tool.py      # grep
│   │   ├── search.py         # web_search (Tavily/Brave backends)
│   │   ├── webfetch_tool.py  # web_fetch (orchestrator)
│   │   ├── memory_tools.py   # save_memory, search_memory + MemoryStore
│   │   ├── planning_tools.py # save_plan, get_plan + PlanStore
│   │   ├── task_tools.py     # save_tasks, get_tasks + TaskStore
│   │   ├── hitl_tools.py     # request_approval + ApprovalManager, HITLConfig
│   │   ├── shell/            # 8-layer shell security
│   │   └── webfetch/         # Fetcher, converter, validator, robots
│   ├── knowledge_base/
│   │   ├── models.py         # Document, SearchResult
│   │   ├── embeddings.py     # EmbeddingService
│   │   ├── vector_store.py   # VectorStore (FAISS)
│   │   ├── _mocks.py         # MockEmbeddingService, MockVectorStore
│   │   └── manager.py        # KnowledgeBaseManager
│   └── persistence/
│       ├── session.py        # SessionPersistence
│       ├── artifacts.py      # ArtifactManager
│       └── _utils.py         # Atomic write utilities
├── tests/
│   ├── conftest.py           # MockContext, shared fixtures
│   ├── test_*.py             # Unit tests
│   ├── tools/                # Tool-specific tests
│   └── integration/          # ADK & LangGraph pipeline tests
└── examples/                 # Demo scripts
```

## Running Commands

**IMPORTANT**: Always use `conda run -n agenticcli` prefix for running commands:

```bash
# Create the environment (first time only)
conda env create -f environment.yml

# Install package
conda run -n agenticcli pip install -e .

# Run tests
conda run -n agenticcli python -m pytest tests/ -v

# Run Python
conda run -n agenticcli python -c "from agentic_cli import BaseCLIApp; print(BaseCLIApp)"
```

## Branching Strategy

- **main**: Stable branch, matches latest release. Only updated via merges from `develop` when releasing.
- **develop**: Integration branch for ongoing work. Small fixes can be committed directly here.
- **feature/\***: Feature branches for larger changes. Branch from `develop`, merge back to `develop`.

Workflow:
1. For small fixes: commit directly to `develop`
2. For features: create `feature/<name>` from `develop`, work there, merge back to `develop`
3. When ready to release: merge `develop` → `main` and tag the release

## Development Principles

### Code Style
- Follow PEP 8 style guidelines
- Use type hints throughout
- Prefer descriptive variable names

### Key Design Decisions
- **Abstract base classes**: BaseCLIApp and BaseWorkflowManager for domain extension
- **Dual orchestrator**: ADK and LangGraph backends, selectable via settings
- **Lazy initialization**: Defer heavy imports until needed
- **Event-based streaming**: Real-time updates via AsyncGenerator
- **UI-agnostic workflow**: WorkflowEvent objects can be consumed by any UI

### Key Design Patterns
- **Tool error handling**: All tools return `{"success": bool, ...}` dicts. Never raise `ToolError`.
- **Tool registration**: Use `@register_tool(category=..., permission_level=..., description=...)` decorator. Tools are auto-discovered via the global `ToolRegistry`.
- **Store consolidation**: Stores and managers (MemoryStore, PlanStore, TaskStore, ApprovalManager) live inside their tool files (e.g. `memory_tools.py`, `hitl_tools.py`), not in separate packages.
- **Context access**: Tools use `get_context_*()` functions from `workflow.context` to access managers and stores via ContextVars.
- **Atomic writes**: Use `atomic_write_json`/`atomic_write_text` from `persistence/_utils.py` for file persistence.

### Console Output
All console output must go through `ThinkingPromptSession` methods. Never use `rich.Console` or `print()` directly.

Available session methods:
- `session.add_response(text, markdown=True)` - Display text/markdown response
- `session.add_rich(renderable)` - Display Rich renderables (Panel, Table, etc.)
- `session.add_message(role, content)` - Add message to history
- `session.add_error(content)` - Display error message
- `session.add_warning(content)` - Display warning message
- `session.add_success(content)` - Display success message
- `session.clear()` - Clear the terminal screen

## Testing

- **Framework**: pytest with `asyncio_mode = "auto"`
- **MockContext**: From `tests/conftest.py` — provides isolated settings and temp dirs for all tests
- **MockVectorStore** and **MockEmbeddingService**: In `knowledge_base/_mocks.py` for testing without ML dependencies
- **FAISS tests**: Guard with `pytest.importorskip("faiss")` since FAISS is not installed in dev env
- **Integration tests**: `tests/integration/` covers ADK and LangGraph pipeline tests
