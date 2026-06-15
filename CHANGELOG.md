# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Long-running job substrate** (`tools/jobs/`, Tier A milestone 1): typed long-running tools start detached work via an internal `JobManager` over pluggable execution backends behind one `JobBackend` interface (ships **subprocess** + **in-process**). The LLM only ever sees the tool — `JobManager` is internal infrastructure, never an LLM-facing tool, and there is no generic `job_submit`. Includes restart-safe completion (on-disk `exit_code` sentinel; subprocesses run detached with `start_new_session`), persistence under `~/.{app_name}/jobs/`, a concurrency cap + queue (`max_concurrent_jobs`, default 4), observe-only management tools (capability `jobs.manage`) — `job_status` is the recommended companion to a long-running tool (it returns state, a stdout tail, and the result once finished, so most agents need only it; `JOB_TOOLS == [job_status]`), with `job_result`/`job_logs`/`job_cancel`/`job_list` as opt-in extras (`JOB_MANAGEMENT_TOOLS`) that also power `/jobs` — a reference `run_shell_job` tool (capability `longrunning.run_shell_job` → default user-verify), `@register_tool(long_running=True)`, and a `/jobs` command (`/jobs`, `/jobs all`, `/jobs <id>`, `/jobs cancel <id>`, `/jobs clean`). Auto-ingest-on-completion (push/resume) is deferred to a later milestone.
- **Harness Jobs UI monitor** (`cli/job_monitor.py`, Tier A milestone 2): a background `JobMonitor` task — started for the lifetime of the CLI session, independent of the agent loop — periodically reconciles the `JobManager` (so detached jobs advance state with no LLM turn) and renders a live jobs segment into the status bar (`jobs: 2 running, 1 queued`), with a transient `✓`/`✗`/`⊘` note when a job finishes. The status bar is the only background-safe UI surface (`thinking_prompt` boxes are turn-oriented and `add_*` prints directly, which would corrupt the live prompt; `set_status` only invalidates the app); `WorkflowController` stays the single composer of the bar and reads the segment the monitor publishes. New `examples/jobs_demo.py` exercises it interactively.

## [0.5.3] - 2026-06-14

### Added
- **Session fact extraction now runs on exit** (`auto_extract_session_facts`): `BaseWorkflowManager.on_session_end()` is invoked from the CLI shutdown path (`BaseCLIApp._extract_session_facts_on_exit`), extracting key facts/preferences from the conversation into memory. Previously the hook had no call site. `on_session_end()` now also sources the session messages itself (from the live backend session) when called with no arguments; it remains a safe no-op without a memory store, and never blocks shutdown.

### Removed
- **Tool Reflection Store removed** (`save_reflection` tool, `ReflectionStore`, `REFLECTION_STORE`, and the `enable_tool_reflections` setting). Introduced in v0.5.0 as a "cross-session learning primitive," it shipped only as a standalone store + tool — the planned injection and session-end wiring were never completed, so it was inert from v0.5.0 through v0.5.2 (the v0.5.0 changelog's "wired via session-end hook" was inaccurate). The design also conflated mechanical failure-capture with LLM heuristic synthesis in a single opportunistic tool the model had to remember to call. Removed rather than finished; tool-failure heuristics, if wanted later, are better served by automatic capture + session-end synthesis (mirroring `auto_extract_session_facts`) or by tagged entries in the existing memory store.

### Security
- **`grep` ripgrep argument injection fixed** (`tools/grep_tool.py`): the LLM-supplied `pattern` was passed to ripgrep as a bare positional, so a value like `--pre=<cmd>` was parsed as a flag and could run an arbitrary program per searched file — remote code execution holding only `filesystem.read`. The pattern is now passed via `-e` and option parsing is terminated with `--` before the path, so neither can be interpreted as a flag.
- **`execute_python` library escape closed** (`tools/executor.py`): the AST/name allow-list cannot neutralize `numpy`/`pandas`/`scipy`/`sklearn`/`matplotlib`, which re-expose file, network, and pickle I/O through public APIs (e.g. `numpy.DataSource().open()`, `pandas.read_pickle(url)`) — an escape from the in-process restrictions (arbitrary file read, pickle-based code execution) when no OS sandbox is active. These `SANDBOXED_MODULES` are now importable **only when `os_sandbox_enabled=True`**; without it, only the pure-computation `CORE_MODULES` are available. The class docstring now states that the AST validation is defense-in-depth, not a security boundary.
- **arXiv PDF download SSRF fixed** (`tools/arxiv_source.py`): `download_pdf` issued a raw `httpx.get(follow_redirects=True)` with no URL validation or size cap on a `pdf_url` taken from a remote Atom feed — bypassing the hardened `ContentFetcher`. It now routes through `get_or_create_fetcher().fetch()`, inheriting SSRF validation (private-IP/redirect blocking), DNS-rebinding checks, and the byte cap. The arXiv API endpoints were also switched from plaintext HTTP to HTTPS to remove the on-path feed-rewrite vector.

### Changed
- **`execute_python` no longer exposes `numpy`/`pandas` (etc.) by default.** Enabling them requires `os_sandbox_enabled=True` (with `sandbox-exec` on macOS or `bwrap` on Linux). This is a deliberate, user-visible reduction of the default capability surface.

### Fixed
- **ADK session resume was a silent no-op** (`workflow/adk/manager.py`): `_inject_session_messages` appended events to the copy returned by `create_session`, so the stored session stayed empty and "Session resumed" restored nothing on the ADK backend. Now uses `append_event` with real `google.adk.events.Event` objects so events land in the stored session.
- **Corrupt sessions index no longer hides every session** (`persistence/session.py`): a corrupt or non-dict `_sessions_index.json` read as empty, so the next save reset it to a single entry and all other sessions vanished from `list_sessions()`. The index now rebuilds from the session files on any parse/shape error (and a non-dict payload no longer raises `AttributeError`).
- **`MemoryStore` is now thread-safe** (`tools/memory_tools.py`): added a lock around `store`/`update`/`delete`/`search` and the full-file `_save` (note `search` is a writer — it bumps access counters). Prevents "dictionary changed size during iteration" and lost writes when LangGraph's `ToolNode` runs tools concurrently in executor threads.
- **Atomic writes are now durable** (`file_utils.py::_atomic_write`): fsync before the rename (so a crash can't persist the rename ahead of the data and destroy the previously-good file), a unique temp filename (two concurrent writers no longer truncate a shared `.tmp`), and explicit UTF-8 (no locale-dependent `UnicodeEncodeError`). Applies to all persistence consumers (settings, sessions, memory).
- **LangGraph graph construction was completely broken** (`workflow/langgraph/graph_builder.py`): `build()` called the local `_get_tools(config)` helper in the agent-node loop before it was defined further down the function, so Python treated `_get_tools` as an unbound local and every graph build raised `UnboundLocalError`. The helper (and `_overrides`) are now defined before first use. This went unnoticed because the dev env had no `langgraph` installed, so the build tests were skipped.
- **LangGraph sqlite/postgres checkpointers were nonfunctional** (`workflow/langgraph/persistence/checkpointers.py`): `create_checkpointer` returned `SqliteSaver/PostgresSaver.from_conn_string(...)` directly, but in current `langgraph-checkpoint-*` that is a context manager (not a saver), and the sync savers can't serve the async `astream_events` path. `create_checkpointer` is now async and uses the async savers (`AsyncSqliteSaver`/`AsyncPostgresSaver`): it enters the connection context, calls `setup()`, and returns `(saver, context_manager)`; `LangGraphWorkflowManager` closes the context manager in `cleanup()`. The `langgraph` extra now also pulls `langgraph-checkpoint-sqlite`/`-postgres`.
- **LangGraph node `retry` deprecation** (`workflow/langgraph/graph_builder.py`): `graph.add_node(..., retry=...)` updated to `retry_policy=...` (`retry` is deprecated since LangGraph 0.5, removed in 2.0).
- Test suite is collectable in the default dev env again: `pytest.importorskip` guards added to `test_context_window.py`, `test_langgraph_state_tools.py`, and `test_backend_isolation.py`, which imported `langgraph`/`langchain_core`/`google.genai` unconditionally. The hybrid-search test now guards on `bm25s`, and `conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` so the real-FAISS and torch-backed tests (the `kb` extra) don't abort the interpreter on the duplicate OpenMP runtime.
- **FAISS index could be left corrupt by a crash** (`knowledge_base/vector_store.py`): `save()` wrote the index in place via `faiss.write_index(self._index, path)`, so a crash mid-write left a truncated `index.faiss` — and `faiss.read_index` raises on a torn file, bricking the whole knowledge base until it was deleted by hand. The index is now written to a temp file in the same directory and `os.replace`d into position (atomic), with the temp cleaned up on failure; the misleading "mappings first is crash-safe" comment was corrected to describe what is actually guaranteed.
- **Settings UI could persist invalid values** (`config.py::update_setting`): the non-special path used `object.__setattr__`, bypassing Pydantic entirely, so a wrong-typed or out-of-range value from the settings dialog was written straight to the instance and saved to `settings.json`, poisoning later runs. It now goes through `__pydantic_validator__.validate_assignment`, which type- and constraint-checks the value (and rejects unknown keys) before it lands — raising `ValidationError` (a `ValueError`, already handled by `apply_settings`).
- **Ctrl+C now actually cancels an in-flight workflow** (`cli/message_processor.py`): the status bar advertised "Ctrl+C: cancel", but the binding only finished the thinking boxes — the workflow kept streaming and burning tokens. The event stream is now consumed in a cancellable task that a watcher aborts when the boxes are finished out from under it (the HITL dialog window is exempted), so Ctrl+C stops the run and prints "Cancelled."
- **Quickstart tools were silently denied** (`README.md`): the README's `greet` example passed an unregistered raw callable and called registration "optional", but the permission engine (on by default) fails closed and denies any tool without a capability declaration on both backends. The quickstart now registers `greet` (`capabilities=EXEMPT`) and the docs state that registration is required for a tool to run.
- **Gemini 2.5 thinking config caused HTTP 400 on every call** (`workflow/adk/manager.py`): `_get_planner` always built `ThinkingConfig(thinking_level=...)`, but `thinking_level` is a Gemini-3-only field — Gemini 2.5 models (including the default `gemini-2.5-flash`) reject it with `400 "Thinking level is not supported for this model"`. With the default `thinking_effort=medium`, every Gemini call failed (the ADK demo was unusable out of the box). The manager now uses the numeric `thinking_budget` for Gemini 2.5 models and reserves `thinking_level` for Gemini 3. Surfaced by the new live scenario tests.

## [0.5.2] - 2026-05-01

### Security
- **`kb_ingest` split into `kb_ingest_text` / `kb_ingest_file` / `kb_ingest_url`**: the previous unified tool declared only `kb.write` (auto-allowed by the builtin `kb.*` rule) while internally reading arbitrary local paths and fetching arbitrary URLs. The split lets the permission engine gate `filesystem.read` and `http.read` at the right entry point, and the URL path now flows through the hardened `ContentFetcher` (URLValidator + manual redirect revalidation + DNS-rebinding mitigation).
- **Concept-page slug traversal blocked** (`ConceptStore`): explicit slugs are validated against a strict `[a-z0-9-]+` allowlist, and `_concept_path` asserts the resolved path stays under `base_dir`. A slug like `../../etc/passwd` is now rejected before any filesystem write.
- **Webfetch redirect handling rewritten**: `ContentFetcher.fetch` no longer follows redirects via httpx. Each `Location` is revalidated through `URLValidator` *before* the next GET, cross-host redirects are reported without contacting the new origin, same-host redirects re-check `robots.txt`, and the redirect chain is capped at 5.
- **`os_sandbox_enabled=True` now fails closed**: when shell or Python execution is requested with sandboxing on, both executors refuse to run if the resolved sandbox is `NoOpSandbox` or wrap fails — instead of silently dropping back to plain subprocess/ulimit.

### Changed
- **`kb_ingest` tool removed** in favor of the three split entry points; `KB_WRITER_TOOLS` and `make_kb_tools` updated. Internal helper renamed: `_ingest_document_with_kb` → `_ingest_text_with_kb` / `_ingest_file_with_kb` / `_ingest_url_with_kb`.
- **research_demo agent guide** updated to direct ingestion at `ingest_arxiv_paper` for arXiv papers and at the typed `kb_ingest_*` tools for everything else.

## [0.5.1] - 2026-04-18

### Added
- **Capability-based permission engine** (PR #72): Framework-independent engine that replaces the ConfirmationPlugin-based HITL. Tools declare capabilities (e.g. `filesystem.write(path=...)`, `http.read`, `shell.exec`) via `@register_tool(capabilities=...)`. Rules are evaluated from four sources — builtin defaults, user `~/.{app_name}/settings.json`, project `./.{app_name}/settings.json`, and in-memory session. When no rule matches, the user is prompted with `Allow once / for session / always (save to project) / Deny`; "always" grants persist into project settings.
- **Matchers**: `PathMatcher` (with `**` glob), `URLMatcher`, `ShellMatcher`, and `StringGlobMatcher`.
- **ADK `PermissionPlugin`**: capability gating for ADK agents.
- **LangGraph `wrap_tool_for_permission`**: per-tool permission wrapping for LangGraph `ToolNode`.
- **`permissions` and `permissions_enabled` settings** for runtime control.
- **`EXEMPT` sentinel**: opts a tool out of the permission engine (used for backend state tools and ADK `transfer_to_agent`).
- **Real BM25 backends**: `bm25s` (preferred) and `rank_bm25` fallback.

### Changed
- **`@register_tool`** now requires the `capabilities=` kwarg.
- **Default grants broadened**: `filesystem.*` auto-extends to the parent directory; `memory.*` and `kb.*` are allowed by default.
- **Ask prompt** now shows the directory-scope target.
- **Workflow init**: `_ensure_managers_initialized` runs in a worker thread to avoid blocking the event loop.
- **HITL confirmation** extracted into a backend-neutral module during the permissions migration.
- **Memory tools**: deduped registry-bound and factory-bound entry points.

### Fixed
- Matcher canonicalize/matches preserves the `*` wildcard.
- `KnowledgeBaseManager` concurrency contract tightened.

### Removed
- **`PermissionLevel`** (SAFE / CAUTION / DANGEROUS), `ConfirmationPlugin`, `_wrap_for_confirmation`, `hitl_tools`, and the `hitl_enabled` setting — superseded by the permission engine.

## [0.5.0] - 2026-04-17

### Added
- **Knowledge Base concept pages** (PR #70): Agent-authored concept/summary pages via `kb_write_concept` and `kb_search_concepts`. Title-weighted case-insensitive search, slug-based addressing, merge-on-overwrite, UTC timestamps.
- **Hybrid BM25 + vector KB search** (PR #69): `kb_search` fuses BM25 and vector results with Reciprocal Rank Fusion. New `create_bm25_index` abstraction, structure-aware document chunking.
- **Per-document KB sidecars**: Markdown sidecars written at ingest, surfaced via async `kb_read`. Lazy-regenerated on first read if missing. Append-only `ingest_log.md` audit trail and auto-maintained `index.md`. New `backfill_sidecars()` with per-document locking and in-progress guard.
- **KB tool bundles**: `KB_READER_TOOLS` and `KB_WRITER_TOOLS` for convenient agent wiring.
- **MemoryStore lifecycle** (PR #66): Embedding-backed semantic search, contradiction detection (`store_with_similarity_check`), `ForgettingPolicy` + `apply_forgetting()`, archive filtering, `update_memory` and `delete_memory` tools.
- **Tool Reflection Store**: Bounded per-tool heuristic memory (`save_reflection`) learned from failures, wired via session-end hook.
- **OS-native sandboxing** (PR #63): macOS seatbelt and Linux namespace sandboxes for shell and Python execution.
- **Dual thinking boxes** (PR #62): Persistent task-progress box alongside the ephemeral LLM-events box; task state persists across turns. New status icons (`[✓]`, `[▸]`).
- **ADK `ConfirmationPlugin` + LangGraph tool wrapper**: Framework-level HITL for `DANGEROUS` tools.
- **ADK `LLMLoggingPlugin`** replaces `LLMEventLogger` for structured LLM traffic logging. **`TaskProgressPlugin`** emits task progress events.
- **Dual-backend state tools**: Native planning/task tools for ADK (`tools/adk/state_tools.py`) and LangGraph (`tools/langgraph/state_tools.py`) built on shared `tools/_core/` logic.
- **`research_demo` is now a pip-installable console script**; new `/kb-backfill` demo command.

### Changed
- **KB tool naming**: `search_knowledge_base` → `kb_search`, `ingest_to_knowledge_base` → `kb_ingest`. Added `kb_read`, `kb_list`.
- **Service registry** (PR #65): Replaced 8 per-concern ContextVars with a unified service registry; tools reach KB, memory, sandbox, reflection stores, etc. through it. Self-contained, backend-specific tool factories.
- **ArXiv flow** (PR #67, #68): `ArxivSearchSource` is service-registered, async-safe, and handles PDF download + ID parsing. ArXiv ingestion moved to `arxiv_tools.ingest_arxiv_paper`. Search results expose `pdf_url`/`abs_url`/`src_url`. Id-indexed entry cache.
- **Workflow manager simplification** (PR #64): Private internals, removed public `graph` property, inlined native `part.thought` check, simplified initialization.
- **KB ingestion** uses async LLM summarizer; densified sidecar prompt with higher output budget; `embedding_device` autodetection.
- **`persistence/_utils.py` → `file_utils.py`** at the package top level.
- **Cross-KB search** uses RRF fusion; `PlanStore` decoupled from task progress display.

### Removed
- **`@requires` decorator** — replaced by service-registry detection.
- **`open_document` tool** (CLI-app concern, not framework) and **`unified_search` tool** (buggy, redundant).
- **`ArtifactManager`** and activity-logging dead code.
- **`context.py`**, **`task_progress.py` module**, legacy `planning_tools.py` and `task_tools.py` — state tools auto-inject per backend.
- **`ThinkingDetector`** (inlined into native `part.thought` check).
- **Old HITL infrastructure** — superseded by `ConfirmationPlugin` and the LangGraph tool wrapper.

### Fixed
- Concept timestamps use UTC with trailing `Z`; `kb_search` rejects whitespace-only queries.
- `kb_read` PDF lazy-gen fallback; drops frontmatter; adds scope.
- `KnowledgeBase.clear()` removes sidecars and rebuilds index; `delete` drops lock entry.
- KB audit log uses UTF-8 + UTC timestamps.
- Tags sentinel handling, `load_all` filter, re-embedding on update, contradiction-detection edge cases in `MemoryStore`.
- Eager detection for missing ML deps; `Content-Type` PDF detection in `web_fetch`.
- Final task-progress emission at end of event stream; full-color styling for the task box.
- Research demo: stale prompts, bundle/tool-name alignment, path-placeholder escaping, service-registry push in `/kb-backfill`.

## [0.4.4] - 2026-03-11

### Added
- **Session Save/Resume**: Persistent conversations across CLI exits with `/save` and `/resume`
- **Sandbox Executor**: Stateful multi-turn Python code execution with sandboxing and default deps
- **Context Window Management**: Native ADK and LangGraph context trimming with source-level detection
- **Context Window Visibility**: Trim detection, status bar token display, and `/status` breakdown
- **Dynamic Model Registry**: Replace static model lists with live API discovery from Google/Anthropic
- **Task Progress Display**: Rich-colored task progress in thinking box, persisted across turns

### Changed
- **Architecture**: Separated workflow+tools layer from UI/CLI layer; composable settings mixins
- **DRY/SOLID Cleanup** (PR #59, #60): Deduplicated config paths, ArXiv parsing, session persistence, artifact loading; extracted `drain_trim_events()`, `format_detail_rows()`, `format_task_checklist()`; removed dead code (`get_help()`, `FinanceResearchState`, unused settings introspection); replaced dynamic `type()` with `_SessionEvent` class; moved token-drop heuristic into `UsageTracker`
- **MessageProcessor**: Removed dead code, unused params, cleaner state init
- **HITL**: Removed dead Future-based machinery, simplified callback path
- **Ripgrep**: Cache availability check with `lru_cache`

### Fixed
- Executor import validation blocking allowed submodules
- Settings not persisting (replaced `exclude_defaults` with explicit exclusion)
- `verbose_thinking` not persisting across restarts
- Thread safety, path traversal, and TOCTOU race conditions in persistence
- VectorStore crash safety: write mappings before FAISS index
- Embeddings: update in-memory state after `ingest_document`
- `open_document`: extension allowlist, DANGEROUS permission, audit logging

### Removed
- `FinanceResearchState` (domain-specific code in shared library)
- Dead `get_help()`, `get_ui_section`, `is_ui_hidden` methods
- Dead `generate_tool_summary`, `clear_context`/`unbind_context`, model constant re-exports
- Dead `cli/settings.py` re-export shim

## [0.4.3] - 2026-02-12

### Changed
- **Overengineering Cleanup**: Removed ~550 lines of dead code — StateSnapshot, SearchSourceRegistry, unused settings/tool fields, dead ToolRegistry methods, dead command APIs
- **Inlined Resolvers**: ModelResolver and PathResolver inlined into BaseSettings (`resolvers.py` now contains only constants)
- **HITL Consolidation**: Merged `hitl/` package into single `tools/hitl_tools.py`; removed `create_checkpoint` tool
- **Persistence Hardening**: Extracted `sanitize_filename` utility, hardened atomic writes in `persistence/_utils.py`
- **Test Mocks Relocated**: MockEmbeddingService and MockVectorStore moved to `knowledge_base/_mocks.py`

### Fixed
- Broken example demos: updated stale imports in arxiv_demo, fileops_demo, shell_demo, websearch_demo
- Disruptive `logger.error` in background init bypassing CLI UI
- Research demo: removed stale README, trimmed prompt bloat

### Removed
- `create_checkpoint` tool (HITL simplified to `request_approval` only)
- `SearchSourceRegistry` class (unused abstraction)
- `StateSnapshot`, `ToolCallRecord` data classes (unused)
- 3 unused conftest test fixtures

## [0.4.2] - 2026-02-08

### Added
- **Claude Model Support**: Auto-switch to LangGraph orchestrator for Anthropic models
- **Tool Result Summaries**: Meaningful one-liners for 18 tools instead of generic "Returned N fields"

### Changed
- **LangGraph Native ToolNode**: Replaced custom tool node with LangChain's native ToolNode
- **Workflow Manager Decomposition**: Extracted cohesive collaborators from workflow managers
- **Message Processor Dispatch**: Extracted event handlers into dispatch table
- **Tool Module Split**: Split monolithic standard.py into domain-specific tool modules
- **Code Cleanup**: Deleted re-export shim, moved adk_manager into adk/, renamed context accessors
- **Atomic File Writes**: write_file and edit_file now use atomic writes

### Fixed
- `diff_compare` tool producing ugly dict repr for summary field

## [0.4.1] - 2026-02-08

### Changed

- **Python Executor Subprocess Isolation**: Moved code execution to subprocess with security hardening (resource limits, restricted builtins)

### Fixed

- Suppressed Google GenAI SDK non-text parts warning

## [0.4.0] - 2026-02-07

### Added

- **Task Management System**: `save_tasks`/`get_tasks` tools backed by `TaskStore` with flat JSON and atomic writes (`src/agentic_cli/tools/task_tools.py`)
- **Integration Tests**: Comprehensive ADK and LangGraph workflow pipeline tests (`tests/integration/`)
- **PDF Text Extraction**: arXiv PDF extraction in `web_fetch` tool, arXiv specialist sub-agent
- **Prompt Caching**: Claude model prompt caching and LLM usage tracking in LangGraph
- **Graceful 429 Handling**: Rate limit detection with user retry prompt and configurable delay

### Changed

- **Simplified Memory**: 5 tools → 2 (`save_memory`/`search_memory`), following Claude Code pattern
- **Simplified Planning**: 7 tools → 2 (`save_plan`/`get_plan`), backed by `PlanStore` (flat markdown)
- **Simplified HITL**: 5 tools → 2 (`request_approval`/`create_checkpoint`), async/blocking via workflow manager
- **Store Consolidation**: Merged store packages into corresponding tool files
- **Tools Review**: Standardized error handling (`{"success": bool}` dicts), improved descriptions, removed deprecated `ToolCategory` values
- **HITL Deadlock Fix**: Direct async callback instead of Future-based pattern for user input
- **Choice Dialog**: Switched from horizontal buttons to vertical radio list (`dropdown_dialog`)
- **Unified Tool Registration**: All 34 tools registered through consistent pattern
- **ContextVar Fix**: Fixed reset bug, deduplicated `_workflow_context`, removed dead modules
- **Task Progress Display**: Fixed stale display, auto-cleanup between messages, plan checkboxes from PlanStore
- **Research Demo**: Updated with task tools, removed app-specific tools, fixed Gemini API compatibility

### Fixed

- `ask_clarification` deadlock when prompting user during workflow
- Stale task progress display between messages
- ContextVar reset bug in workflow context

## [0.3.3] - 2026-02-04

### Added

- **Shell Security Architecture**: Modular shell executor with 8-layer defense-in-depth security
  - Input preprocessing with encoding/obfuscation detection
  - Command tokenization and classification
  - Path analysis and sandboxing
  - Risk assessment with HITL approval workflows
  - Comprehensive audit logging
  - **Note**: Shell tool is disabled by default pending security validation
- **File Operation Tools**: New categorized file tools with permission levels
  - READ tools (safe): `read_file`, `grep`, `glob`, `list_dir`, `diff_compare`
  - WRITE tools (caution): `write_file`, `edit_file`
- **Feature Demo Scripts**: New examples for arxiv, fileops, memory, planning, shell, and websearch

### Changed

- **File Operations Refactoring**: Replaced monolithic `file_manager` with distinct, categorized tools
- **Tool Registry**: Added `PermissionLevel` enum for tool categorization (SAFE, CAUTION, DANGEROUS)

### Removed

- Deprecated `file_manager` tool (replaced by new file operation tools)

## [0.3.2] - 2026-02-01

### Added

- **Web Fetch Tool**: Full-featured web content fetching with LLM summarization
  - HTML-to-Markdown conversion with html2text
  - Robots.txt compliance checking
  - SSRF protection and URL validation
  - Caching and redirect handling
- **Web Search Tool**: Pluggable web search with backend abstraction
- **arXiv Integration**: Enhanced arXiv search with rate limiting, caching, advanced query options, and paper analysis tools
- **LLM Event Logging**: Debug logging for model interactions
- **Task Progress Events**: `verbose_thinking` setting for detailed task progress display

### Changed

- **CLI Architecture Refactoring**:
  - Extracted `WorkflowController` and `MessageProcessor` from `BaseCLIApp` for better separation of concerns
  - Added `background_init` context manager to `WorkflowController`
  - Added public query methods to managers to reduce command-workflow coupling
- **BaseWorkflowManager**: Moved shared implementations from ADK/LangGraph managers to base class
- **Code Quality**: Added enums for string literals, improved `TaskGraph` encapsulation
- **Settings**: Use app-specific paths and generic settings application

### Fixed

- Circular import in workflow module
- Web search tool integration issues
- arXiv cache unbounded growth (added size limit)
- Webfetch redirect response structure alignment with spec

## [0.3.1] - 2026-01-28

### Added

- **Task Progress Display**: Thinking box now shows dynamic task progress with status icons (◐ ☐ ✓ ✗)
- `TASK_PROGRESS` event type for signaling task graph updates
- `TaskGraph.to_compact_display()` for condensed status display

### Changed

- **Background Initialization**: Workflow manager now initializes services in background, eliminating first-message lag
- Simplified LangGraph imports - removed `_import_langgraph` helper in favor of direct imports

## [0.3.0] - 2025-01-27

### Added

- **Memory System** with 3-tier architecture:
  - `WorkingMemory` - Session-scoped context with tags and serialization
  - `LongTermMemory` - Persistent memory with knowledge base references
  - `MemoryManager` - Unified interface for working and long-term memory
  - Memory tools (`working_memory_tool`, `long_term_memory_tool`)
- **Planning System**:
  - `TaskGraph` - Work plan management with dependencies and status tracking
  - `TaskStatus` enum for pending/in_progress/completed/blocked states
  - `Task` dataclass with subtasks, dependencies, and metadata
- **Human-in-the-Loop (HITL) System**:
  - `ApprovalManager` - Configurable approval gates with auto-approve patterns
  - `CheckpointManager` - Review checkpoints with continue/edit/regenerate/abort actions
  - `HITLConfig` and `ApprovalRule` for configuration
- **New Tools**:
  - `shell_executor` with safety controls (blocks dangerous commands)
  - `file_manager` with read/write/list/copy/move/delete operations
  - `diff_compare` with unified/side-by-side/summary modes
- New tool categories: `MEMORY`, `PLANNING`, `SYSTEM`
- Framework-provided tools with auto-detection
- Layered JSON settings persistence with organized mixins
- `research_demo` example showcasing memory, planning, and file operations

### Changed

- **LangGraph Module Reorganization**:
  - Moved to dedicated `workflow/langgraph/` submodule
  - Added `persistence/` for checkpointers and stores
  - Added `tools/` for shell and file search utilities
- **Simplified LangGraphWorkflowManager**:
  - Removed dead middleware module (~400 lines)
  - Use explicit provider instantiation (GenAI for Gemini, not VertexAI)
  - Simplified model creation with thinking support for Claude and Gemini
- **BaseCLIApp Simplification**:
  - Constructor-based configuration instead of method overrides
  - `AppInfo` is now a constructor parameter

### Fixed

- LangGraph no longer requires `langchain-google-vertexai` package
- Explicit provider detection prevents VertexAI initialization errors

## [0.2.0] - 2025-01-25

### Added

- LangGraph as pluggable orchestration backend (`LangGraphWorkflowManager`)
- `create_workflow_manager_from_settings()` factory function for auto-selecting orchestrator
- Thinking level support for LangGraphWorkflowManager (Anthropic and Google models)
- `log_activity` setting for optional conversation activity logging
- `hello_langgraph.py` example demonstrating LangGraph orchestration
- Comprehensive README documentation with architecture, examples, and API reference

### Changed

- Renamed `WorkflowManager` to `GoogleADKWorkflowManager` for clarity
- Replaced custom retry logic with framework built-in mechanisms (ADK HttpRetryOptions, LangGraph RetryPolicy)
- Removed `ConversationMemory` in favor of native framework session/state management
- Gemini 3 Pro thinking level now falls back to HIGH when MEDIUM is requested (Pro only supports LOW/HIGH)

### Fixed

- LangGraph compatibility issues with content block extraction
- Thinking level configuration for Gemini 3 Pro models

## [0.1.2] - 2026-01-24

### Added

- Added `apply_settings()` method to `BaseCLIApp` for centralized settings application
- Added `settings_command.py` module for cleaner separation of concerns
- Added `resolvers.py` with centralized model/embedding resolution logic
- Added `config_mixins.py` with reusable config traits (ModelConfigMixin, EmbeddingConfigMixin)
- Added `workflow/event_processor.py` for focused event handling
- Added `workflow/session_handler.py` for session management
- Added `workflow/retry.py` for retry logic with backoff
- Added `workflow/memory.py` for conversation memory management
- Added `SlashCommandCompleter` for improved CLI autocomplete (only triggers on `/`)

### Changed

- Refactored `SettingsCommand` to use `SettingsDialog` with `DropdownItem` and `InlineSelectItem`
- Extracted `SettingsCommand` from `builtin_commands.py` to dedicated `settings_command.py`
- Refactored `WorkflowManager` into focused helper classes
- Moved `llm/thinking.py` to `workflow/thinking.py`
- Settings dialog now shows descriptions for model and thinking effort options
- CLI now uses `complete_while_typing` for responsive autocomplete
- Sorted command completions alphabetically

### Fixed

- Fixed settings dialog display issues (cursor position, screen artifacts)

### Removed

- Removed `tools/resilience.py` (unused retry/circuit breaker code)
- Removed `tools/search.py` (unused web search client - use `google_search_tool` from ADK)
- Removed `WebSearchSource` from knowledge_base (use `google_search_tool` directly)
- Removed `llm/` package (consolidated into workflow)
- Removed short command aliases (h, ?, cls, st, q) to reduce completion noise
- Removed custom `DIALOG_STYLE` in favor of `thinking-prompt` built-in dialog styling

## [0.1.1] - 2025-01-20

### Added

- Initial public release with core CLI framework
- `BaseCLIApp` for building agentic CLI applications
- `WorkflowManager` for Google ADK integration
- Knowledge base with vector search
- Safe Python executor for code execution

## [0.1.0] - 2025-01-15

### Added

- Initial development release
