"""Shell security module with layered defense architecture.

Provides secure shell command execution with multiple security layers:
- Layer 1: Input Preprocessing & Encoding Detection
- Layer 2: Lexical/Syntactic Analysis (tokenization)
- Layer 3: Command Classification & Filtering
- Layer 4: Path Analysis & Sandboxing
- Layer 5: Operation Risk Assessment
- Layer 6: User Approval Gateway (via HITL system)
- Layer 7: Resource Limits & Execution Sandbox
- Layer 8: Audit Logging

Usage:
    from agentic_cli.tools.shell import shell_executor

    # Safe command - auto-approved
    result = shell_executor("ls -la", working_dir="/project")

    # Risky command - returns pending_approval
    result = shell_executor("rm -rf ./build", working_dir="/project")
    if result.get("pending_approval"):
        # Handle approval flow via HITL system

    # Blocked command - returns error
    result = shell_executor("rm -rf /")
    # result["success"] == False, result["error"] contains reason

    # Obfuscated command - blocked
    result = shell_executor("echo cm0gLXJmIC8= | base64 -d | bash")
    # result["success"] == False, detected encoding blocks execution
"""

from agentic_cli.tools.shell.executor import (
    shell_executor,
    analyze_command,
    execute_with_approval,
)
from agentic_cli.tools.shell.config import (
    ShellSecurityConfig,
    get_strict_config,
    get_permissive_config,
)
from agentic_cli.tools.shell.models import (
    ApprovalType,
    ClassificationResult,
    CommandCategory,
    CommandNode,
    PathAnalysisResult,
    PathCheck,
    RiskAssessment,
    RiskLevel,
    SecurityAnalysis,
    TokenizeResult,
)
from agentic_cli.tools.shell.tokenizer import CommandTokenizer
from agentic_cli.tools.shell.classifier import CommandClassifier
from agentic_cli.tools.shell.path_analyzer import PathAnalyzer
from agentic_cli.tools.shell.risk_assessor import RiskAssessor
from agentic_cli.tools.shell.preprocessor import InputPreprocessor, PreprocessResult
from agentic_cli.tools.shell.sandbox import ExecutionSandbox, ExecutionLimits, ExecutionResult
from agentic_cli.tools.shell.audit import AuditLogger, AuditEntry, AuditConfig

__all__ = [
    # Main executor
    "shell_executor",
    "analyze_command",
    "execute_with_approval",
    # Configuration
    "ShellSecurityConfig",
    "get_strict_config",
    "get_permissive_config",
    # Analysis classes
    "CommandTokenizer",
    "CommandClassifier",
    "PathAnalyzer",
    "RiskAssessor",
    "InputPreprocessor",
    # Execution
    "ExecutionSandbox",
    "ExecutionLimits",
    "ExecutionResult",
    # Audit
    "AuditLogger",
    "AuditEntry",
    "AuditConfig",
    # Data models
    "ApprovalType",
    "ClassificationResult",
    "CommandCategory",
    "CommandNode",
    "PathAnalysisResult",
    "PathCheck",
    "PreprocessResult",
    "RiskAssessment",
    "RiskLevel",
    "SecurityAnalysis",
    "TokenizeResult",
]
