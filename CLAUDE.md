# Agentic CLI - Shared Framework for Agentic Applications

## Project Overview

Agentic CLI is a shared library providing the core infrastructure for building domain-specific CLI applications powered by LLM agents.

## Tech Stack

- **Language**: Python 3.12+
- **CLI UI**: `thinking-prompt` - enhanced CLI with thinking boxes and markdown
- **Workflow**: Google ADK 2 (2.12+) + LangGraph - dual orchestration backends (selectable via settings)
- **Models**: `agentic_cli.models` (provisional) - references, roles, the shipped table and the catalog, with no provider SDK (listers over `httpx`); two adapters build the SDK objects: ADK's own model classes (`Gemini`, `AnthropicLlm`, `OpenAIResponsesLlm`, plus our `OpenAICompatibleLlm`, on the `openai` SDK) and the LangChain partner packages (`langchain-google-genai`, `langchain-anthropic`, `langchain-openai`, `langchain-openrouter`; the `langgraph` extra)
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
│   ├── paths.py              # resolve_path: the one interpretation of a path argument
│   ├── logging.py
│   ├── cli/
│   │   ├── app.py            # BaseCLIApp (/settings model pickers: _model_choices)
│   │   ├── commands.py       # Command, CommandRegistry
│   │   ├── builtin_commands.py # /help, /status, /models (ModelsCommand), /sessions, ...
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
│   │   ├── settings.py       # Workflow/tool settings schema
│   │   ├── retry.py          # Rate-limit retry helpers
│   │   ├── tool_summaries.py
│   │   ├── memory_services.py # Settings → agentic_cli.memory arguments (KBs, memory store)
│   │   ├── model_services.py # Settings → agentic_cli.models arguments (keys, endpoints, roles); build_catalog
│   │   ├── model_wire.py     # What both adapters send: timeouts, placeholder key, OpenRouter headers, reasoning_body
│   │   ├── langchain_models.py # build_chat_model: ResolvedModel → LangChain chat model (profile=); OpenAICompatibleChat
│   │   ├── permissions/      # Framework-independent capability engine
│   │   │   ├── engine.py     # PermissionEngine (deny-wins, default-ASK)
│   │   │   ├── capabilities.py # Capability, EXEMPT
│   │   │   ├── matchers.py   # PathMatcher, URLMatcher, ShellMatcher, StringGlobMatcher
│   │   │   └── rules.py, store.py, prompt.py
│   │   ├── adk/              # ADK orchestrator
│   │   │   ├── manager.py    # GoogleADKWorkflowManager
│   │   │   ├── event_processor.py  # ADKEventProcessor
│   │   │   ├── model_factory.py # build_adk_model: ResolvedModel → ADK model, config, planner
│   │   │   ├── openai_compatible.py # OpenAICompatibleLlm (OpenRouter, endpoints; reasoning text)
│   │   │   ├── anthropic_llm.py # DirectAnthropicLlm: bare claude-* strings → direct API
│   │   │   ├── permission_plugin.py # PermissionPlugin (gates tool calls)
│   │   │   ├── task_progress_plugin.py # Emits TASK_PROGRESS events
│   │   │   ├── tool_error_plugin.py # Tool exception → error result (turn survives)
│   │   │   ├── context_cache_plugin.py # One Gemini cache attempt a turn
│   │   │   ├── foreign_reasoning_plugin.py # No model is sent another provider's thoughts
│   │   │   └── plugins.py    # LLM traffic logging (raw_llm_logging)
│   │   └── langgraph/        # LangGraph orchestrator
│   │       ├── manager.py    # LangGraphWorkflowManager
│   │       ├── graph_builder.py # LangGraphBuilder (graph; one chat model per agent via langchain_models)
│   │       ├── state.py
│   │       ├── permission_wrap.py # wrap_tool_for_permission
│   │       └── persistence/  # Checkpointers, stores
│   ├── tools/
│   │   ├── registry.py       # ToolRegistry, @register_tool, ToolCategory
│   │   ├── factories.py      # Service-bound tool builders (per-manager flavors)
│   │   ├── executor.py       # SafePythonExecutor (CORE_MODULES; SANDBOXED_MODULES gated on OS sandbox)
│   │   ├── execution_tools.py # execute_python
│   │   ├── _knowledge_base.py # current_knowledge_base(): the knowledge_base key
│   │   ├── knowledge_tools.py # kb_* tools: adapters over KnowledgeBase
│   │   ├── arxiv_tools.py    # search_arxiv, fetch_arxiv_paper, ingest_arxiv_paper
│   │   ├── arxiv_source.py   # ArxivSearchSource (feed fetch, download_pdf)
│   │   ├── search_sources.py # SearchSource, SearchSourceResult
│   │   ├── pdf_utils.py      # extract_pdf_text
│   │   ├── interaction_tools.py # ask_clarification
│   │   ├── file_read.py      # read_file, diff_compare
│   │   ├── file_write.py     # write_file, edit_file
│   │   ├── glob_tool.py      # glob
│   │   ├── grep_tool.py      # grep
│   │   ├── search.py         # web_search (Tavily/Brave backends)
│   │   ├── webfetch_tool.py  # web_fetch + get_or_create_fetcher (orchestrator)
│   │   ├── memory_tools.py   # save_memory, search_memory, update_memory, delete_memory (store: agentic_cli.memory)
│   │   ├── _core/           # Backend-neutral tool logic
│   │   │   ├── planning.py  # save_plan/get_plan core (+ checkbox parsing)
│   │   │   └── tasks.py     # save_tasks/get_tasks core (+ progress parsing)
│   │   ├── adk/state_tools.py       # ADK-native plan/task tools (ToolContext.state)
│   │   ├── langgraph/state_tools.py # LangGraph-native plan/task tools (Command/InjectedState)
│   │   ├── sandbox/         # Stateful code-execution sandbox (sandbox_execute)
│   │   ├── shell/           # 8-layer shell security (+ os_sandbox/)
│   │   └── webfetch/        # Fetcher, converter, validator, robots, summarizer, saved pages
│   ├── memory/               # Knowledge base + memory store; imports nothing else from agentic_cli
│   │   ├── __init__.py       # MemoryStore, MemoryItem, ForgettingPolicy, EmbeddingService, EmbeddingConfig, MockEmbeddingService
│   │   ├── _core/            # Shared, private: embeddings.py, mock_embeddings.py, io.py, log.py, model_record.py
│   │   ├── store.py          # MemoryStore
│   │   └── kb/               # Knowledge base (agentic_cli.memory.kb)
│   │       ├── manager.py    # KnowledgeBaseManager(base_dir, embedding=, summarizer=), Summarizer
│   │       ├── knowledge_base.py  # KnowledgeBase (provisional): the scopes as one, and the ingestion pipeline (ingest_text, ingest_bytes)
│   │       ├── enrich.py      # The summary/sidecar prompts and parser: one summarizer call per document
│   │       ├── models.py     # Document, SearchResult, SourceType
│   │       ├── convert.py    # convert_document (HTML main text via trafilatura, html2text fallback; pypdf), own _charset.py
│   │       ├── vector_store.py, bm25_index.py (+ _bm25_backends.py), concepts.py, sidecar.py
│   │       └── _mock_vector_store.py, _mock_bm25.py
│   ├── models/               # Model knowledge (provisional); imports nothing else from agentic_cli, no provider SDK
│   │   ├── refs.py           # parse_spec/parse_ref: provider:model, bare ids, role names
│   │   ├── endpoints.py      # ProviderKind, EndpointSpec (settings), Endpoint, build_endpoints
│   │   ├── records.py        # ModelRecord, ThinkingRule
│   │   ├── table.py          # ShippedTable, load_table() (data/table.json: family rules, numbers, fallbacks)
│   │   ├── roles.py          # RoleMap: default / fast / strong / app roles → references
│   │   ├── resolve.py        # resolve_model → ResolvedModel (thinking, max_tokens, sampling)
│   │   ├── catalog.py        # ModelCatalog: declarations + listings + table, validation, listing cache
│   │   └── discovery/        # One lister per provider kind, over httpx
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
- **Tool error handling**: All tools return `{"success": bool, ...}` dicts. Never raise `ToolError`. An exception that still escapes a tool is a bug, but it costs one call rather than the turn: on ADK, `workflow/adk/tool_error_plugin.py::ToolErrorPlugin` turns it (and a call to a tool name that does not exist) into an error dict; LangGraph's `ToolNode` runs with `handle_tool_errors=True`.
- **Tool registration**: Use `@register_tool(category=..., capabilities=..., description=...)` decorator. `capabilities=` is required — pass `EXEMPT` for tools that need no permission check or a list of `Capability(name, target_arg=...)` tuples the engine matches against rules. Tools are auto-discovered via the global `ToolRegistry`.
- **Permissions**: `workflow/permissions/` holds a framework-independent engine that evaluates declared capabilities against rules from four sources (builtin, user `~/.{app_name}/settings.json`, project `./.{app_name}/settings.json`, in-memory session). Interactive "Allow always" grants persist to `~/.{app_name}/project_grants.json` keyed by resolved project path — user-owned, never into the project settings file a repo could ship. ADK + LangGraph gate tool calls via `workflow/adk/permission_plugin.py::PermissionPlugin` and `workflow/langgraph/permission_wrap.py::wrap_tool_for_permission`. `web_fetch` saves every page it reads into `./.{app_name}/fetched/` (`tools/webfetch/saved.py`) as framework state, without a filesystem capability check: the location is fixed (the model never chooses it), a symlinked folder is refused, and cleanup only ever touches its own file names — so the builtin write deny on `.{app_name}/**` is about tools reaching into that path, not about this.
- **Permission targets vs. patterns**: a tool argument is a *target*, a rule's target is a *pattern*. Matchers canonicalize them with different methods: `canonicalize_pattern` expands `${workdir}`/`${home}`/`${app_name}` and anchors relative filesystem patterns to the project; `canonicalize_target` expands nothing and resolves a path exactly as the tool will. Any tool that takes a path argument must resolve it with `agentic_cli.paths.resolve_path` (the same function `PathMatcher.canonicalize_target` uses), so the engine judges the location the tool acts on. `tests/tools/test_path_contract.py` fails when a new filesystem tool has no contract case. Filesystem, `http.*` and shell capabilities must name their resource (`target_arg=` or a fixed `target=`): registering a tool with a targetless one raises `ValueError` (it would resolve to the wildcard, which is never remembered as a grant), and saved `http.read *`-style grants are dropped on load. Filesystem grants cover the named directory (a file's own directory), never `/` or anything above home; an `http.read` grant covers the URL's site (`engine.site_scope`: `https://host/**`, exact host, HTTPS, no IP literals or local names).
- **Service registry**: Tools access services and shared state via `get_service(key)` from `workflow.service_registry`. A single ContextVar holds a `dict[str, Any]` set by the workflow manager during processing. Complex services (KnowledgeBase, SandboxManager, MemoryStore) are lazily created; simple state (plan string, task list) lives directly in the registry dict.
- **Manager detection**: tools declare their own service needs — `@register_tool(..., requires="knowledge_base")` (or a tuple for several) — and the key is validated against `service_registry.KNOWN_SERVICE_KEYS` at registration (only *constructible* services are declarable). `BaseWorkflowManager._detect_required_managers()` reads that metadata off the registry for each agent's tools (by registry identity, so `register(func, name=...)`'s original callable still declares its services); `_build_services()` then lazily constructs only the services actually needed, into a local dict that is released in full if a later constructor raises (nothing is published, so nothing else could close it). A downstream tool may request any framework-provided service without editing the framework; there is no mechanism for registering new service *types*, and no central name→service map.
- **Canonical tool names + permission identity**: `ToolDefinition.name` is the single identity. `register_tool(name="public_name")` wraps the callable so `func.__name__` is the registered name (backends derive the model-visible name from the callable). Permission gating **and tool assembly** resolve through a **registry-owned identity binding**: each `ToolRegistry` owns a `id(obj) → (weakref, definition)` map (`registry.bind_identity()`/`identify()`; the module-level `tools.registry.bind_tool_identity()`/`identify_tool()` answer for `get_registry()`). Every hit is confirmed with `is` against the weak reference, so neither a name, a class name, nor a forged `__eq__`/`__hash__` can stand in for it, and a recycled address inherits nothing. Identity is **per registry** — a tool registered into an application's own `ToolRegistry` is not one of the framework's, so it stays untouched during assembly and is denied at permission time — which is also what lets a short-lived registry, its definitions and their closures be garbage collected. Everything the framework issues is bound at its construction site: registered callables, factory service-bound variants, renamed wrappers, and the native ADK tool objects the framework builds (skill tools, in `tools/skills/toolset.py`). `workflow/adk/permission_plugin.py` unwraps `.func` only for the exact ADK types whose contract is to call it (`FunctionTool`, `LongRunningFunctionTool`, and ADK's own `TransferToAgentTool`), and gates genuine `McpTool` instances (`isinstance`, since ADK creates them on connect) under a synthetic `mcp` capability. `TransferToAgentTool` is a `FunctionTool` **subclass** ADK auto-injects into any agent with `sub_agents`; matching only exact `FunctionTool` denied the built-in routing tool as unregistered, so delegation could not work at all. It is trusted on exactly the same terms as the others — ADK constructs it as `super().__init__(func=transfer_to_agent)` and overrides only `_get_declaration()` (to add the agent-name enum), never `run_async`, so what it invokes is still exactly `self.func`. Listing the **exact class** keeps every other subclass, forged object and same-named tool denied; no name authority is restored. Anything unbound is denied — and, in assembly (service detection, service-tool substitution, canonicalization, long-running wrapping), left exactly as the application supplied it: a plain callable named `kb_search` is not the framework's tool and must not be given its services, its service-bound variant, or its long-running contract. Substitution additionally requires the service variant to *be* that same definition (`identify_tool(variant) is definition`); factories bind each closure to the exact module-level tool it re-binds, so a tool an application has taken over keeps its own implementation.
- **One name, one tool**: `ToolRegistry.register()` raises on any name that is already registered — matching capabilities are *not* grounds for sharing one, since they say nothing about the docstring or the model-visible schema. Sharing is declared, never inferred: `declare_tool(name, ...)` (exported from `agentic_cli.tools`) declares a tool that has **no backend-neutral implementation** (`ToolDefinition.func is None`), and each backend registers its own with `register_tool(..., variant_of=name)` — same identity and permission contract, its own signature and docstring. Re-declaring the same contract is idempotent; changing its description or capabilities raises. `definition.variants` is ordered by defining module, so it never depends on import order, and assembly substitutes a variant's canonical callable via `registry.canonical_for()` (never `None`). A `replace=True` retires the previous definition's callables, and retired backend variants are excluded from `include_state_tools` injection, so the model never sees two tools with one name. That is how the ADK and LangGraph `save_plan`/`get_plan`/`save_tasks`/`get_tasks` coexist (declared in `tools/_core/state_tools.py`); previously they contested the name and import order decided the winner. A bare-name reference to a declared-only tool raises "ambiguous" rather than guessing a backend. `replace=True` takes a name over deliberately and **retires the old definition's identities**, so its callables resolve to nothing (denied, and left alone by assembly) rather than inheriting the replacement's capabilities.
- **Turn/lifecycle concurrency**: a manager runs one turn at a time — `process()`/`resume_with_job_result()` enter through `_turn_admission()` (which holds `_turn_lock`), and `initialize_services`/`reinitialize`/`cleanup` hold `_lifecycle_lock` **and** `_turn_lock`. Lock order is lifecycle → turn; a turn initializes *before* taking the turn lock, which is what keeps the two from deadlocking — and because of that a cleanup can land in between, so admission re-checks `_backend_ready()` while holding the turn lock and reinitializes once (or fails cleanly) rather than running against released resources. Initialization is transactional: services are built on a worker thread into a *local* dict and published only while the attempt still owns init (a cancelled attempt releases what the thread went on to build), and a failed attempt rolls back. A failed in-place reinitialization leaves the manager uninitialized; the controller reports `FAILED` and refuses to hand it out, but **keeps** it so a retry can revive it with its preserved (possibly in-memory) sessions intact. `WorkflowController` serializes init/reinitialize/swap/close on its own lifecycle lock, and a background init that finishes after `close()` releases its manager instead of publishing it.
- **HITL callback is context-local**: `set_input_callback()` stores into a per-manager `ContextVar`, so a second consumer installing its callback cannot capture a running turn's prompt, and one consumer's `clear_input_callback()` cannot unregister another's. `MessageProcessor` cancels and awaits its consumer task **before** clearing the callback, so no tool is left asking a question nobody owns.
- **No harness-level turn replay**: ADK persists a turn's input during invocation setup, so the CLI never re-invokes an event source. Retries belong to the provider client (`HttpRetryOptions`). `MessageProcessor` returns a typed `TurnResult`.
- **Session identity**: durable sessions are addressed by `SessionRef(app_name, user_id, session_id)` (`workflow/sessions.py`). Every session hook (`session_exists`/`list_sessions`/`delete_session`/`recent_messages`/`load_session`) takes an optional `user_id`, defaulting to `settings.default_user` only when the caller omits it. Backends without a session store leave `supports_sessions` False, and the base hooks raise `NotImplementedError` rather than answering with a misleading `False`/`[]`.
- **Active turn**: the in-flight `(user, session)` is a `ContextVar` (`workflow/sessions.py::get_active_turn`), set with a token by `_workflow_context()` — concurrent turns on one manager stay isolated and nesting restores the outer turn.
- **Resource ownership**: `cleanup()` is idempotent and awaits an async `close()` on owned resources (`BaseWorkflowManager._aclose_owned`); `WorkflowController.close()` is the single shutdown path (cancel init → shut executor → clean manager). `reinitialize(preserve_sessions=True)` (a model switch) carries the services over inside `_carrying_services()`: session grants, sandbox kernels, jobs and knowledge bases belong to the conversation, not the model, so neither the release nor a failed attempt's rollback closes them.
- **Memory package boundary**: `agentic_cli/memory/` (the knowledge base in `memory/kb/`, the memory store in `memory/store.py`, shared embeddings and atomic writes in `memory/_core/`) imports nothing from the rest of agentic-cli, and `kb` and `store` never import each other. The rest of agentic-cli uses only `agentic_cli.memory` and `agentic_cli.memory.kb`; `workflow/memory_services.py` is the one place settings become package arguments. The workflow exposes one `KnowledgeBase` (`memory.kb.knowledge_base`, provisional) over the project and user scopes, and one `EmbeddingService` shared with the memory store. `tests/memory/test_boundary.py` enforces all of it, and tests under `tests/memory/` import only the package. A knowledge base summarizes only with the `summarizer=` it was given; the workflow passes itself when `knowledge_base_summarize` is on. Importing the package loads no other agentic_cli module (`tests/memory/test_loads_alone.py`), because the package root `agentic_cli/__init__.py` loads its exports on first use (PEP 562 `__getattr__`): nothing may be imported there eagerly, since Python runs it before every `agentic_cli.*` import.
- **Atomic writes**: Use `atomic_write_json`/`atomic_write_text`/`atomic_write_bytes` from `file_utils.py` for file persistence. They create files private (0600), right for the framework's own state; a write to a file the user owns (`write_file`, `edit_file`) passes `preserve_mode=True`, which keeps an existing file's permission bits and gives a new file the process default.

### Console Output
All console output must go through `ThinkingPromptSession` methods. Never use `rich.Console` or `print()` directly.
Python warnings, and the stdlib logging `configure_logging` sets up, print to stderr, over the UI, so `BaseCLIApp` installs `cli/app.py::_install_adk_notice_hook`: ADK's experimental notices, `[EXPERIMENTAL] feature … is enabled` and `[EXPERIMENTAL] <Class>: …` (a class ADK marks experimental, such as the `GeminiContextCacheManager` a caching Gemini agent builds), are logged at debug (`adk_experimental_feature`), and every other warning goes to the previous `warnings.showwarning`; a filter on ADK's runner logger (`google_adk.google.adk.runners`) logs its "has no context_cache_config" advice at debug too (`adk_uncached_agent_transfer`). ADK gives that advice only to an app without a context cache: the ADK manager sets one (`_context_cache_config`) whenever `prompt_caching_enabled` is on and an agent runs on Claude or Gemini (for a native ADK YAML app, the default model), so the filter is for apps with caching off or no caching model. A filter on ADK's context-cache logger (`google_adk.google.adk.models.gemini_context_cache_manager`) logs a Gemini cache it could not create or delete at debug (`adk_cache_create_failed`, `adk_cache_cleanup_failed`): the turn goes on uncached. Any other ADK warning that lands over the UI is handled the same narrow way, never by silencing the whole `google_adk` logger tree.

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
- **Model listings**: `tests/conftest.py::_no_model_listings` keeps every manager's model catalog unloaded (no network, no `~/.<app>/model_cache.json`), so any well-formed model id is accepted; a test that wants listings patches `model_services.build_catalog` with `tests/models/fakes.py::ProviderAPI`
- **Provider fakes**: model requests never reach a provider. `tests/workflow/adk/fakes.py::FakeAPI` serves an ADK-built model's client (`serve()`) and records the bodies it sends; `tests/workflow/langchain_fakes.py`'s `route*` helpers send a LangChain chat model's client to it, with the SSE replies a LangGraph turn needs. A test-first run against the old code must fail offline too, so stub any client the old code would build
- **MockVectorStore** (`agentic_cli.memory.kb`) and **MockEmbeddingService** (`agentic_cli.memory`): for testing without ML dependencies
- **`kb` extra tests**: import faiss, bm25s, rank_bm25, sentence_transformers or trafilatura with `require_kb_module(name)` (`tests/memory/kb_extra.py`), not `pytest.importorskip`. Without the extra the test skips; CI's `kb-extra` job installs it and sets `KB_REQUIRE_EXTRA=1`, so a missing library fails there instead of skipping. No offline test loads an embedding model.
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
