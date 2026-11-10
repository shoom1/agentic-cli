"""Google ADK-specific workflow implementation.

This submodule contains ADK-specific components:
- manager.py: GoogleADKWorkflowManager
- event_processor.py: ADKEventProcessor
- plugins.py: ADK plugins (LLMLoggingPlugin)
- log_filters.py: what ADK would print over the CLI, logged instead

Note: GoogleADKWorkflowManager is NOT re-exported here to avoid circular
imports. Import it directly from agentic_cli.workflow.adk.manager.

``LLMLoggingPlugin`` is loaded on first use (PEP 562): importing the package,
as ``log_filters`` does, loads no ADK.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_cli.workflow.adk.plugins import LLMLoggingPlugin

__all__ = [
    "LLMLoggingPlugin",
]


def __getattr__(name: str) -> Any:
    if name == "LLMLoggingPlugin":
        from agentic_cli.workflow.adk.plugins import LLMLoggingPlugin

        return LLMLoggingPlugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
