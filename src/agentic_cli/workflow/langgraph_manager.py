"""LangGraph-based Workflow Manager for agentic CLI applications.

DEPRECATED: This module has been moved to agentic_cli.workflow.langgraph.manager.
This file is kept for backward compatibility. Please update your imports to:

    from agentic_cli.workflow.langgraph import LangGraphWorkflowManager

Or:

    from agentic_cli.workflow import LangGraphWorkflowManager
"""

import warnings

# Emit deprecation warning on import
warnings.warn(
    "agentic_cli.workflow.langgraph_manager is deprecated. "
    "Use agentic_cli.workflow.langgraph.manager or agentic_cli.workflow instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location for backward compatibility
from agentic_cli.workflow.langgraph.manager import (
    LangGraphWorkflowManager,
    LangGraphManager,
    _import_langgraph,
    _import_langchain_models,
)

__all__ = [
    "LangGraphWorkflowManager",
    "LangGraphManager",
    "_import_langgraph",
    "_import_langchain_models",
]
