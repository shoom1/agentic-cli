"""Configuration for Human-in-the-Loop features."""

from dataclasses import dataclass, field


@dataclass
class ApprovalRule:
    """Rule defining when approval is required.

    Attributes:
        tool: Tool name (e.g., "shell_executor")
        operations: Specific operations requiring approval (None = all)
        auto_approve_patterns: Patterns that skip approval (glob-style)
    """

    tool: str
    operations: list[str] | None = None
    auto_approve_patterns: list[str] = field(default_factory=list)

    def matches_tool(self, tool_name: str) -> bool:
        """Check if rule applies to a tool."""
        return self.tool == tool_name

    def matches_operation(self, operation: str) -> bool:
        """Check if rule applies to an operation."""
        if self.operations is None:
            return True  # All operations
        return operation in self.operations


@dataclass
class HITLConfig:
    """Configuration for Human-in-the-Loop features.

    Attributes:
        approval_rules: Rules defining when approval is required
        checkpoint_enabled: Enable review checkpoints
        feedback_enabled: Enable inline feedback
        confidence_threshold: Threshold below which to ask for help
        confidence_visible: Show confidence scores to user
    """

    approval_rules: list[ApprovalRule] = field(default_factory=list)
    checkpoint_enabled: bool = True
    feedback_enabled: bool = True
    confidence_threshold: float = 0.75
    confidence_visible: bool = True
