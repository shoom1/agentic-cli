"""LangGraph-specific workflow implementation.

This submodule contains all LangGraph-related components:
- manager: LangGraphWorkflowManager for orchestrating agents
- state: State definitions for LangGraph workflows
- middleware/: Native middleware wrappers (HITL, retry, shell, etc.)
- persistence/: Storage backends (checkpointers, stores)
- tools/: LangGraph-native tool implementations
"""

# Re-export main components for convenience
from agentic_cli.workflow.langgraph.manager import (
    LangGraphWorkflowManager,
    LangGraphManager,  # Alias
)
from agentic_cli.workflow.langgraph.state import (
    AgentState,
    ResearchState,
    ApprovalState,
    FinanceResearchState,
    CheckpointData,
    Message,
    ToolCall,
    ToolResult,
    add_messages,
)

__all__ = [
    # Manager
    "LangGraphWorkflowManager",
    "LangGraphManager",
    # State
    "AgentState",
    "ResearchState",
    "ApprovalState",
    "FinanceResearchState",
    "CheckpointData",
    "Message",
    "ToolCall",
    "ToolResult",
    "add_messages",
]
