"""Data models for shell security module.

Provides dataclasses for command analysis, risk assessment, and path checking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_cli.tools.shell.preprocessor import PreprocessResult


class CommandCategory(Enum):
    """Category of shell command based on inherent danger level."""

    BLOCKED = "blocked"  # Never allow
    PRIVILEGED = "privileged"  # Require explicit approval
    WRITE = "write"  # Modifies filesystem
    NETWORK = "network"  # Network access
    READ = "read"  # Read-only operations
    SAFE = "safe"  # Always allow


class RiskLevel(Enum):
    """Risk level for a command or operation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalType(Enum):
    """Type of approval required."""

    AUTO = "auto"  # Auto-approve
    PROMPT_ONCE = "prompt_once"  # Prompt once per session
    ALWAYS_PROMPT = "always_prompt"  # Always prompt
    BLOCK = "block"  # Never allow


@dataclass
class Redirect:
    """Represents a shell redirection."""

    operator: str  # e.g., ">", ">>", "<", "2>"
    target: str  # The file or descriptor


@dataclass
class CommandNode:
    """Parsed representation of a shell command."""

    command: str  # Base command (e.g., "rm")
    args: list[str] = field(default_factory=list)  # Command arguments
    redirections: list[Redirect] = field(default_factory=list)
    pipes_to: "CommandNode | None" = None
    chained_with: list[tuple[str, "CommandNode"]] = field(
        default_factory=list
    )  # [("&&", node), (";", node)]
    subshells: list["CommandNode"] = field(default_factory=list)
    background: bool = False  # Ends with &
    raw_command: str = ""  # Original command string


@dataclass
class TokenizeResult:
    """Result of tokenizing a shell command."""

    nodes: list[CommandNode]  # All command nodes (including chained)
    has_pipes: bool = False
    has_chains: bool = False  # ;, &&, ||
    has_subshells: bool = False  # $(), ``
    has_redirections: bool = False
    has_background: bool = False
    parse_errors: list[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Result of classifying a command."""

    category: CommandCategory
    command: str  # The base command
    matched_pattern: str | None = None  # Pattern that matched (for BLOCKED)
    reason: str | None = None  # Explanation


@dataclass
class PathCheck:
    """Result of checking a single path."""

    original: str  # Original path string
    resolved: Path | None = None  # Resolved absolute path (None if unresolvable)
    in_project: bool = False  # Within project directory
    in_allowed: bool = False  # Within any allowed directory
    is_sensitive: bool = False  # In sensitive system path
    traversal_detected: bool = False  # Contains ../ tricks
    is_glob: bool = False  # Contains wildcards


@dataclass
class PathAnalysisResult:
    """Result of analyzing all paths in a command."""

    paths: list[PathCheck]
    has_sensitive_paths: bool = False
    has_outside_project: bool = False
    has_traversal: bool = False
    write_paths: list[PathCheck] = field(default_factory=list)
    read_paths: list[PathCheck] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """Overall risk assessment for a command."""

    base_risk: RiskLevel  # From command category
    path_risk: RiskLevel  # From path analysis
    chaining_risk: RiskLevel  # Command complexity
    encoding_risk: RiskLevel = RiskLevel.LOW  # From encoding/obfuscation detection
    network_risk: RiskLevel = RiskLevel.LOW  # Refined network risk

    overall_risk: RiskLevel = RiskLevel.LOW  # Computed overall
    approval_required: ApprovalType = ApprovalType.AUTO
    block_reasons: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)

    # Analysis details
    classification: ClassificationResult | None = None
    path_analysis: PathAnalysisResult | None = None
    tokenize_result: TokenizeResult | None = None
    preprocess_result: "PreprocessResult | None" = None  # From encoding detection

    @property
    def is_blocked(self) -> bool:
        """Check if command should be blocked."""
        return self.approval_required == ApprovalType.BLOCK

    @property
    def is_write_operation(self) -> bool:
        """Check if this is a write operation."""
        if self.classification:
            return self.classification.category in (
                CommandCategory.WRITE,
                CommandCategory.PRIVILEGED,
            )
        return False


@dataclass
class SecurityAnalysis:
    """Complete security analysis for a shell command."""

    command: str  # Original command
    working_dir: Path  # Working directory for execution
    tokenize_result: TokenizeResult
    classifications: list[ClassificationResult]  # One per command node
    path_analysis: PathAnalysisResult
    risk_assessment: RiskAssessment

    @property
    def is_safe(self) -> bool:
        """Check if command is safe to execute without approval."""
        return self.risk_assessment.overall_risk == RiskLevel.LOW

    @property
    def is_blocked(self) -> bool:
        """Check if command should be blocked."""
        return self.risk_assessment.is_blocked
