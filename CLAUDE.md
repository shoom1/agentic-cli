# Agentic CLI - Shared Framework for Agentic Applications

## Project Overview

Agentic CLI is a shared library providing the core infrastructure for building domain-specific CLI applications powered by LLM agents.

## Tech Stack

- **Language**: Python 3.12+
- **CLI UI**: `thinking-prompt` - enhanced CLI with thinking boxes and markdown
- **Workflow**: Google ADK + LangGraph - dual orchestration backends (selectable via settings)
- **Config**: `pydantic-settings` - type-safe configuration
- **Logging**: `structlog` - structured logging

## Project Structure

```
agentic-cli/
├── src/agentic_cli/
│   ├── __init__.py           # Package exports, lazy imports
│   ├── config.py             # BaseSettings (pydantic-settings)
│   ├── settings_mixins.py    # Composable settings field groups
│   ├── settings_persistence.py # save_settings() (excludes SECRET_FIELDS)
│   ├── constants.py          # Shared constants, truncate()
│   ├── file_utils.py         # atomic_write_json / atomic_write_text
│   ├── logging.py
│   ├── cli/
│   │   ├── app.py            # BaseCLIApp
│   │   ├── commands.py       # Command, CommandRegistry
│   │   ├── builtin_commands.py
│   │   ├── workflow_controller.py  # WorkflowController (lazy/background init, orchestrator swap)
│   │   ├── message_processor.py    # WorkflowEvent → ThinkingPromptSession rendering
│   │   ├── settings_command.py     # /settings command
│   │   ├── settings_introspection.py # Pydantic field → UI item introspection
│   │   └── usage_tracker.py        # Token usage / status bar
│   ├── workflow/
│   │   ├── base_manager.py   # BaseWorkflowManager (abstract; service detection, tool assembly)
│   │   ├── factory.py        # create_workflow_manager_from_settings (ADK vs LangGraph routing)
│   │   ├── service_registry.py # get_service/require_service + ContextVar registry
│   │   ├── events.py         # WorkflowEvent, EventType
│   │   ├── config.py         # AgentConfig
│   │   ├── models.py
│   │   ├── settings.py       # Workflow/tool settings schema
│   │   ├── retry.py          # Rate-limit retry helpers
│   │   ├── tool_summaries.py
│   │   ├── permissions/      # Framework-independent capability engine
│   │   │   ├── engine.py     # PermissionEngine (deny-wins, default-ASK)
│   │   │   ├── capabilities.py # Capability, EXEMPT
│   │   │   ├── matchers.py   # PathMatcher, URLMatcher, ShellMatcher, StringGlobMatcher
│   │   │   └── rules.py, store.py, prompt.py
│   │   ├── adk/              # ADK orchestrator
│   │   │   ├── manager.py    # GoogleADKWorkflowManager
│   │   │   ├── event_processor.py  # ADKEventProcessor
│   │   │   ├── permission_plugin.py # PermissionPlugin (gates tool calls)
│   │   │   ├── task_progress_plugin.py # Emits TASK_PROGRESS events
│   │   │   └── plugins.py    # LLM traffic logging (raw_llm_logging)
│   │   └── langgraph/        # LangGraph orchestrator
│   │       ├── manager.py    # LangGraphWorkflowManager
│   │       ├── graph_builder.py # LangGraphBuilder (graph + LLM factory)
│   │       ├── state.py
│   │       ├── permission_wrap.py # wrap_tool_for_permission
│   │       └── persistence/  # Checkpointers, stores
│   ├── tools/
│   │   ├── registry.py       # ToolRegistry, @register_tool, ToolCategory
│   │   ├── factories.py      # Service-bound tool builders (per-manager flavors)
│   │   ├── executor.py       # SafePythonExecutor (CORE_MODULES; SANDBOXED_MODULES gated on OS sandbox)
│   │   ├── execution_tools.py # execute_python
│   │   ├── knowledge_tools.py # kb_search, kb_ingest_{text,file,url}, kb_list, kb_read, kb_write_concept, kb_search_concepts
│   │   ├── arxiv_tools.py    # search_arxiv, fetch_arxiv_paper, ingest_arxiv_paper
│   │   ├── arxiv_source.py   # ArxivSearchSource (feed fetch, download_pdf)
│   │   ├── pdf_utils.py      # extract_pdf_text
│   │   ├── interaction_tools.py # ask_clarification
│   │   ├── file_read.py      # read_file, diff_compare
│   │   ├── file_write.py     # write_file, edit_file
│   │   ├── glob_tool.py      # glob
│   │   ├── grep_tool.py      # grep
│   │   ├── search.py         # web_search (Tavily/Brave backends)
│   │   ├── webfetch_tool.py  # web_fetch + get_or_create_fetcher (orchestrator)
│   │   ├── memory_tools.py   # save_memory, search_memory, update_memory, delete_memory + MemoryStore
│   │   ├── _core/           # Backend-neutral tool logic
│   │   │   ├── planning.py  # save_plan/get_plan core (+ checkbox parsing)
│   │   │   └── tasks.py     # save_tasks/get_tasks core (+ progress parsing)
│   │   ├── adk/state_tools.py       # ADK-native plan/task tools (ToolContext.state)
│   │   ├── langgraph/state_tools.py # LangGraph-native plan/task tools (Command/InjectedState)
│   │   ├── sandbox/         # Stateful code-execution sandbox (sandbox_execute)
│   │   ├── shell/           # 8-layer shell security (+ os_sandbox/)
│   │   └── webfetch/        # Fetcher, converter, validator, robots, summarizer
│   ├── knowledge_base/
│   │   ├── models.py         # Document, SearchResult
│   │   ├── embeddings.py     # EmbeddingService
│   │   ├── vector_store.py   # VectorStore (FAISS)
│   │   ├── bm25_index.py     # BM25 index (+ _bm25_backends.py: bm25s / rank_bm25)
│   │   ├── concepts.py       # ConceptStore (concept pages)
│   │   ├── sidecar.py        # Markdown sidecar rendering
│   │   ├── sources.py
│   │   ├── _mocks.py         # MockEmbeddingService, MockVectorStore (+ _mock_bm25.py)
│   │   └── manager.py        # KnowledgeBaseManager
│   └── persistence/
│       └── session.py        # SessionPersistence
├── tests/
│   ├── conftest.py           # MockContext, shared fixtures
│   ├── test_*.py             # Unit tests
│   ├── tools/                # Tool-specific tests
│   ├── workflow/             # Backend-isolation / workflow tests
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
- **fix/\***: Fix branches for fixing issues. Branch from `develop`, merge back to `develop`.
- **refactor/\***: For larger refactoring changes. Branch from `develop`, merge back to `develop`.

Workflow:
1. For small fixes: commit directly to `develop`
2. For features: create `feature/<name>` (or `fix/<name>` or `refactor/<name>`) from `develop`, work there, merge back to `develop`
3. When ready to release: merge `develop` → `main` and tag the release

### What NOT to commit
- `docs/` is gitignored on purpose (see `.gitignore`). It is a scratchpad for review notes, plans, and internal analysis. **Never `git add docs/…` or suggest committing anything under `docs/`.** If a document belongs in the repo, it lives elsewhere (README, CHANGELOG, top-level `*.md`).

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
- **Tool registration**: Use `@register_tool(category=..., capabilities=..., description=...)` decorator. `capabilities=` is required — pass `EXEMPT` for tools that need no permission check or a list of `Capability(name, target_arg=...)` tuples the engine matches against rules. Tools are auto-discovered via the global `ToolRegistry`.
- **Permissions**: `workflow/permissions/` holds a framework-independent engine that evaluates declared capabilities against rules from four sources (builtin, user `~/.{app_name}/settings.json`, project `./.{app_name}/settings.json`, in-memory session). ADK + LangGraph gate tool calls via `workflow/adk/permission_plugin.py::PermissionPlugin` and `workflow/langgraph/permission_wrap.py::wrap_tool_for_permission`.
- **Service registry**: Tools access services and shared state via `get_service(key)` from `workflow.service_registry`. A single ContextVar holds a `dict[str, Any]` set by the workflow manager during processing. Complex services (KBManager, SandboxManager, MemoryStore) are lazily created; simple state (plan string, task list) lives directly in the registry dict.
- **Manager detection**: `BaseWorkflowManager._detect_required_managers()` scans each agent's tool names against the `_TOOL_SERVICE_MAP` (name → service key, in `base_manager.py`); `_ensure_managers_initialized()` then lazily instantiates only the services actually needed (KBManager, SandboxManager, MemoryStore, …). Adding a new service-backed tool means adding its name → service entry to `_TOOL_SERVICE_MAP`. (There is no `@requires` decorator.)
- **Atomic writes**: Use `atomic_write_json`/`atomic_write_text` from `file_utils.py` for file persistence.

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
