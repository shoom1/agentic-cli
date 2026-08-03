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
│   ├── settings_persistence.py # Trust-split save (PROJECT_SETTABLE_KEYS → project, rest → user config; excludes SECRET_FIELDS)
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
- **Manager detection**: tools declare their own service needs — `@register_tool(..., requires="kb_manager")` (or a tuple for several) — and the key is validated against `service_registry.KNOWN_SERVICE_KEYS` at registration (only *constructible* services are declarable; `user_kb_manager` is created together with `kb_manager`). `BaseWorkflowManager._detect_required_managers()` reads that metadata off the registry for each agent's tools (by registry identity, so `register(func, name=...)`'s original callable still declares its services); `_build_services()` then lazily constructs only the services actually needed, into a local dict that is released in full if a later constructor raises (nothing is published, so nothing else could close it). A downstream tool may request any framework-provided service without editing the framework; there is no mechanism for registering new service *types*, and no central name→service map.
- **Canonical tool names + permission identity**: `ToolDefinition.name` is the single identity. `register_tool(name="public_name")` wraps the callable so `func.__name__` is the registered name (backends derive the model-visible name from the callable). Permission gating **and tool assembly** resolve through a **registry-owned identity binding**: each `ToolRegistry` owns a `id(obj) → (weakref, definition)` map (`registry.bind_identity()`/`identify()`; the module-level `tools.registry.bind_tool_identity()`/`identify_tool()` answer for `get_registry()`). Every hit is confirmed with `is` against the weak reference, so neither a name, a class name, nor a forged `__eq__`/`__hash__` can stand in for it, and a recycled address inherits nothing. Identity is **per registry** — a tool registered into an application's own `ToolRegistry` is not one of the framework's, so it stays untouched during assembly and is denied at permission time — which is also what lets a short-lived registry, its definitions and their closures be garbage collected. Everything the framework issues is bound at its construction site: registered callables, factory service-bound variants, renamed wrappers, and the native ADK tool objects the framework builds (skill tools, in `tools/skills/toolset.py`). `workflow/adk/permission_plugin.py` unwraps `.func` only for the exact ADK types whose contract is to call it (`FunctionTool`, `LongRunningFunctionTool`), and gates genuine `McpTool` instances (`isinstance`, since ADK creates them on connect) under a synthetic `mcp` capability. Anything unbound is denied — and, in assembly (service detection, service-tool substitution, canonicalization, long-running wrapping), left exactly as the application supplied it: a plain callable named `kb_search` is not the framework's tool and must not be given its services, its service-bound variant, or its long-running contract. Substitution additionally requires the service variant to *be* that same definition (`identify_tool(variant) is definition`); factories bind each closure to the exact module-level tool it re-binds, so a tool an application has taken over keeps its own implementation.
- **One name, one tool**: `ToolRegistry.register()` raises on any name that is already registered — matching capabilities are *not* grounds for sharing one, since they say nothing about the docstring or the model-visible schema. Sharing is declared, never inferred: `declare_tool(name, ...)` (exported from `agentic_cli.tools`) declares a tool that has **no backend-neutral implementation** (`ToolDefinition.func is None`), and each backend registers its own with `register_tool(..., variant_of=name)` — same identity and permission contract, its own signature and docstring. Re-declaring the same contract is idempotent; changing its description or capabilities raises. `definition.variants` is ordered by defining module, so it never depends on import order, and assembly substitutes a variant's canonical callable via `registry.canonical_for()` (never `None`). A `replace=True` retires the previous definition's callables, and retired backend variants are excluded from `include_state_tools` injection, so the model never sees two tools with one name. That is how the ADK and LangGraph `save_plan`/`get_plan`/`save_tasks`/`get_tasks` coexist (declared in `tools/_core/state_tools.py`); previously they contested the name and import order decided the winner. A bare-name reference to a declared-only tool raises "ambiguous" rather than guessing a backend. `replace=True` takes a name over deliberately and **retires the old definition's identities**, so its callables resolve to nothing (denied, and left alone by assembly) rather than inheriting the replacement's capabilities.
- **Turn/lifecycle concurrency**: a manager runs one turn at a time — `process()`/`resume_with_job_result()` enter through `_turn_admission()` (which holds `_turn_lock`), and `initialize_services`/`reinitialize`/`cleanup` hold `_lifecycle_lock` **and** `_turn_lock`. Lock order is lifecycle → turn; a turn initializes *before* taking the turn lock, which is what keeps the two from deadlocking — and because of that a cleanup can land in between, so admission re-checks `_backend_ready()` while holding the turn lock and reinitializes once (or fails cleanly) rather than running against released resources. Initialization is transactional: services are built on a worker thread into a *local* dict and published only while the attempt still owns init (a cancelled attempt releases what the thread went on to build), and a failed attempt rolls back. A failed in-place reinitialization leaves the manager uninitialized; the controller reports `FAILED` and refuses to hand it out, but **keeps** it so a retry can revive it with its preserved (possibly in-memory) sessions intact. `WorkflowController` serializes init/reinitialize/swap/close on its own lifecycle lock, and a background init that finishes after `close()` releases its manager instead of publishing it.
- **HITL callback is context-local**: `set_input_callback()` stores into a per-manager `ContextVar`, so a second consumer installing its callback cannot capture a running turn's prompt, and one consumer's `clear_input_callback()` cannot unregister another's. `MessageProcessor` cancels and awaits its consumer task **before** clearing the callback, so no tool is left asking a question nobody owns.
- **No harness-level turn replay**: ADK persists a turn's input during invocation setup, so the CLI never re-invokes an event source. Retries belong to the provider client (`HttpRetryOptions`). `MessageProcessor` returns a typed `TurnResult`.
- **Session identity**: durable sessions are addressed by `SessionRef(app_name, user_id, session_id)` (`workflow/sessions.py`). Every session hook (`session_exists`/`list_sessions`/`delete_session`/`recent_messages`/`load_session`) takes an optional `user_id`, defaulting to `settings.default_user` only when the caller omits it. Backends without a session store leave `supports_sessions` False, and the base hooks raise `NotImplementedError` rather than answering with a misleading `False`/`[]`.
- **Active turn**: the in-flight `(user, session)` is a `ContextVar` (`workflow/sessions.py::get_active_turn`), set with a token by `_workflow_context()` — concurrent turns on one manager stay isolated and nesting restores the outer turn.
- **Resource ownership**: `cleanup()` is idempotent and awaits an async `close()` on owned resources (`BaseWorkflowManager._aclose_owned`); `WorkflowController.close()` is the single shutdown path (cancel init → shut executor → clean manager).
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

### Live LLM tests (real API calls)

Tests that hit real provider APIs use the existing framework — **don't invent new gating or key handling.**

- **Marker**: `@pytest.mark.llm`; modules set `pytestmark = [pytest.mark.llm, pytest.mark.skipif(<no key>, ...)]`.
- **Key loading is handled by the live-test framework** (`tests/integration/conftest.py`) — keys are not
  plain shell env vars, so go through pytest rather than re-deriving it.
- **Run**: `-m llm` (live; needs network — disable the Bash sandbox) or `-m 'not llm'` (offline).
  A bare `pytest` run makes real API calls when keys are available.
- **Example**: `tests/integration/test_adk_claude_live.py`.

### Live docker sandbox tests (real container runtime)

The `jupyter_docker` backend's isolation boundary is verified against a real daemon.

- **Marker**: `@pytest.mark.docker`; skipped unless docker/podman is available. The bulk of the
  backend is tested offline via the faked `ContainerRuntime` seam (incl. an end-to-end run of the
  real `driver.py` as a subprocess) — these live tests only cover what needs a real container.
- **Run**: `-m docker` (needs a container runtime) or `-m 'not llm and not docker'` (offline CI).
- **Fail-loud in CI**: set `SANDBOX_REQUIRE_DOCKER=1` so a missing/broken runtime FAILS instead of
  silently skipping (a skip reads as green). CI: `.github/workflows/ci.yml` (offline + docker jobs).
- **Example**: `tests/tools/test_sandbox_docker_live.py`.
