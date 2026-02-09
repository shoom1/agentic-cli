# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
