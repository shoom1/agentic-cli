# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
