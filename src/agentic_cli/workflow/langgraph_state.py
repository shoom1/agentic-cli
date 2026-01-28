"""State definitions for LangGraph workflow orchestration.

DEPRECATED: This module has been moved to agentic_cli.workflow.langgraph.state.
This file is kept for backward compatibility. Please update your imports to:

    from agentic_cli.workflow.langgraph import AgentState, CheckpointData

Or:

    from agentic_cli.workflow.langgraph.state import AgentState, CheckpointData
"""

import warnings

# Emit deprecation warning on import
warnings.warn(
    "agentic_cli.workflow.langgraph_state is deprecated. "
    "Use agentic_cli.workflow.langgraph.state or agentic_cli.workflow instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location for backward compatibility
from agentic_cli.workflow.langgraph.state import (
    add_messages,
    Message,
    ToolCall,
    ToolResult,
    AgentState,
    ResearchState,
    ApprovalState,
    FinanceResearchState,
    CheckpointData,
    AgentStateType,
    ResearchStateType,
    ApprovalStateType,
    FinanceResearchStateType,
)

__all__ = [
    "add_messages",
    "Message",
    "ToolCall",
    "ToolResult",
    "AgentState",
    "ResearchState",
    "ApprovalState",
    "FinanceResearchState",
    "CheckpointData",
    "AgentStateType",
    "ResearchStateType",
    "ApprovalStateType",
    "FinanceResearchStateType",
]
