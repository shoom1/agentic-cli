# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
