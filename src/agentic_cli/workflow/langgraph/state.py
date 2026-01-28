"""State definitions for LangGraph workflow orchestration.

This module defines the state structures used by LangGraph-based workflows,
including message handling, agent state, and research/analysis workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict
import operator


def add_messages(existing: list, new: list | Any) -> list:
    """Reducer function for combining message lists.

    Handles both list and single message additions.
    """
    if isinstance(new, list):
        return existing + new
    return existing + [new]


class Message(TypedDict, total=False):
    """A message in the conversation.

    Compatible with LangChain's message format while remaining framework-agnostic.
    """

    role: Literal["user", "assistant", "system", "tool"]
    content: str
    name: str | None  # Tool name for tool messages
    tool_call_id: str | None  # For tool result messages


class ToolCall(TypedDict, total=False):
    """A tool call request."""

    id: str
    name: str
    args: dict[str, Any]


class ToolResult(TypedDict, total=False):
    """A tool call result."""

    tool_call_id: str
    name: str
    result: Any
    error: str | None
    duration_ms: int


class AgentState(TypedDict, total=False):
    """Base state for agent workflows.

    This is the minimal state required for any LangGraph workflow in the framework.
    Domain-specific workflows can extend this with additional fields.

    Fields:
        messages: Conversation history (uses add_messages reducer)
        current_agent: Name of the currently active agent
        pending_tool_calls: Tool calls waiting to be executed
        tool_results: Results from executed tool calls
        session_id: Session identifier
        user_id: User identifier
        metadata: Arbitrary metadata
    """

    messages: Annotated[list[Message], add_messages]
    current_agent: str | None
    pending_tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    session_id: str
    user_id: str
    metadata: dict[str, Any]


class ResearchState(AgentState, total=False):
    """State for research-oriented workflows.

    Extends AgentState with fields for deep research tasks including
    multi-step data gathering, analysis, and report generation.

    Fields:
        research_query: The original research query
        research_data: Collected research data from various sources
        analysis_results: Results from analysis steps
        report_draft: Generated report content
        sources: List of source URLs/references
        iteration_count: Number of research iterations completed
        max_iterations: Maximum allowed iterations
        needs_more_data: Flag indicating if more research is needed
    """

    research_query: str
    research_data: dict[str, Any]
    analysis_results: dict[str, Any]
    report_draft: str
    sources: list[str]
    iteration_count: int
    max_iterations: int
    needs_more_data: bool


class ApprovalState(TypedDict, total=False):
    """State for human-in-the-loop approval workflows.

    Used when a workflow requires human approval to proceed.

    Fields:
        approval_required: Whether approval is currently needed
        approval_request: Description of what needs approval
        approval_response: Human's response (approved/rejected/comment)
        approval_timestamp: When approval was given/denied
    """

    approval_required: bool
    approval_request: str
    approval_response: str | None
    approval_timestamp: datetime | None


class FinanceResearchState(ResearchState, ApprovalState, total=False):
    """State for finance-domain research workflows.

    Combines research state with approval state for workflows that
    require compliance approval for high-risk findings.

    Fields:
        risk_level: Assessed risk level (low/medium/high/critical)
        compliance_notes: Notes from compliance review
        regulatory_flags: List of regulatory concerns flagged
    """

    risk_level: Literal["low", "medium", "high", "critical"] | None
    compliance_notes: str | None
    regulatory_flags: list[str]


@dataclass
class CheckpointData:
    """Data structure for workflow checkpoints.

    Enables resumable workflows by storing state snapshots.
    """

    state: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checkpoint_id: str = ""
    parent_checkpoint_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "state": self.state,
            "timestamp": self.timestamp.isoformat(),
            "checkpoint_id": self.checkpoint_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointData":
        """Create from dictionary."""
        return cls(
            state=data["state"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            checkpoint_id=data.get("checkpoint_id", ""),
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
            metadata=data.get("metadata", {}),
        )


# Type aliases for cleaner type hints
AgentStateType = AgentState
ResearchStateType = ResearchState
ApprovalStateType = ApprovalState
FinanceResearchStateType = FinanceResearchState
