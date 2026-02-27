"""Shell executor with layered security and HITL integration.

Main entry point for secure shell command execution.
Integrates all security layers including:
- Layer 1: Input Preprocessing (encoding detection)
- Layer 2: Tokenization
- Layer 3: Classification
- Layer 4: Path Analysis
- Layer 5: Risk Assessment
- Layer 6: HITL Approval
- Layer 7: Sandboxed Execution
- Layer 8: Audit Logging

SECURITY NOTE:
    The shell tool is currently DISABLED by default while security safeguards
    are being validated. Set _SHELL_TOOL_ENABLED = True to enable execution.
"""

import time
from pathlib import Path

from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)

# =============================================================================
# SHELL TOOL DISABLED
# =============================================================================
# The shell tool is disabled until security safeguards are fully validated.
# To enable: set _SHELL_TOOL_ENABLED = True
# WARNING: Only enable after thorough security review and testing.
# =============================================================================
_SHELL_TOOL_ENABLED = False


def is_shell_enabled() -> bool:
    """Check if the shell tool is enabled.

    Returns:
        True if shell execution is enabled, False otherwise.
    """
    return _SHELL_TOOL_ENABLED
from typing import Any

from agentic_cli.tools.shell.audit import AuditLogger
from agentic_cli.tools.shell.classifier import CommandClassifier
from agentic_cli.tools.shell.config import ShellSecurityConfig, get_strict_config
from agentic_cli.tools.shell.models import (
    ApprovalType,
    CommandCategory,
    RiskLevel,
    SecurityAnalysis,
)
from agentic_cli.tools.shell.path_analyzer import PathAnalyzer
from agentic_cli.tools.shell.preprocessor import InputPreprocessor
from agentic_cli.tools.shell.risk_assessor import RiskAssessor
from agentic_cli.tools.shell.sandbox import ExecutionSandbox
from agentic_cli.tools.shell.tokenizer import CommandTokenizer
# Module-level audit logger (lazy initialization)
_audit_logger: AuditLogger | None = None


def _get_audit_logger(config: ShellSecurityConfig) -> AuditLogger | None:
    """Get or create the module-level audit logger."""
    global _audit_logger

    if not config.audit.enabled:
        return None

    if _audit_logger is None:
        _audit_logger = AuditLogger(config.audit)

    return _audit_logger


def analyze_command(
    command: str,
    working_dir: str | Path | None = None,
    project_root: str | Path | None = None,
    config: ShellSecurityConfig | None = None,
) -> SecurityAnalysis:
    """Analyze a shell command through all security layers.

    Args:
        command: The shell command to analyze.
        working_dir: Working directory for execution.
        project_root: Project root directory for sandboxing.
        config: Optional security configuration.

    Returns:
        SecurityAnalysis with complete analysis results.
    """
    # Resolve directories
    if working_dir is None:
        working_dir = Path.cwd()
    else:
        working_dir = Path(working_dir).resolve()

    if project_root is None:
        project_root = working_dir
    else:
        project_root = Path(project_root).resolve()

    # Use default strict config if not provided
    if config is None:
        config = get_strict_config()

    # Layer 1: Preprocessing (encoding detection)
    preprocess_result = None
    command_to_analyze = command
    if config.enable_preprocessing:
        preprocessor = InputPreprocessor()
        preprocess_result = preprocessor.process(command)
        # Use normalized command for further analysis if available
        if preprocess_result.normalized_command != command:
            command_to_analyze = preprocess_result.normalized_command

    # Layer 2: Tokenize
    tokenizer = CommandTokenizer()
    tokenize_result = tokenizer.tokenize(command_to_analyze)

    # Layer 3: Classify
    classifier = CommandClassifier(config)

    # First check the full command against blocked patterns (catches curl | bash etc)
    full_command_check = classifier.check_full_command(command_to_analyze)
    if full_command_check:
        classifications = [full_command_check]
    else:
        classifications = classifier.classify_all(tokenize_result.nodes)

    # Layer 4: Path Analysis
    path_analyzer = PathAnalyzer(project_root, config)
    path_analysis = path_analyzer.analyze(
        tokenize_result.nodes, classifications, working_dir
    )

    # Layer 5: Risk Assessment (now includes encoding risk)
    risk_assessor = RiskAssessor(config)
    risk_assessment = risk_assessor.assess(
        tokenize_result, classifications, path_analysis, preprocess_result
    )

    return SecurityAnalysis(
        command=command,
        working_dir=working_dir,
        tokenize_result=tokenize_result,
        classifications=classifications,
        path_analysis=path_analysis,
        risk_assessment=risk_assessment,
    )


def truncate_output(output: str, max_length: int = 50000) -> str:
    """Truncate output if it exceeds maximum length.

    Args:
        output: The output string to potentially truncate.
        max_length: Maximum allowed length.

    Returns:
        Original or truncated output with indicator.
    """
    if len(output) <= max_length:
        return output

    truncated = output[:max_length]
    return f"{truncated}\n... [OUTPUT TRUNCATED - exceeded {max_length} characters]"


def _execute_command(
    command: str,
    working_dir: Path,
    config: ShellSecurityConfig,
) -> dict[str, Any]:
    """Execute a command and return results.

    Internal function that actually runs the command using the sandbox.
    """
    # Create sandbox with configured limits
    sandbox = ExecutionSandbox(config.execution_limits)

    # Execute in sandbox
    result = sandbox.execute(command, working_dir)

    return {
        "success": result.success,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "return_code": result.return_code,
        "duration": result.duration_ms / 1000,  # Convert to seconds
        "error": result.error,
        "truncated": result.truncated,
        "resource_limit_hit": result.resource_limit_hit,
    }


@register_tool(
    category=ToolCategory.EXECUTION,
    permission_level=PermissionLevel.DANGEROUS,
    description="Execute a shell command with layered security and HITL approval",
)
def shell_executor(
    command: str,
    working_dir: str | None = None,
    timeout: int = 60,
    capture_output: bool = True,
    project_root: str | None = None,
    config: ShellSecurityConfig | None = None,
) -> dict[str, Any]:
    """Execute a shell command with layered security and HITL approval.

    Executes the given shell command through multiple security layers:
    - Layer 1: Preprocessing (encoding/obfuscation detection)
    - Layer 2: Tokenization (parse command structure)
    - Layer 3: Classification (categorize risk)
    - Layer 4: Path Analysis (sandbox boundaries)
    - Layer 5: Risk Assessment (overall risk score)
    - Layer 6: HITL Approval (user consent for risky operations)
    - Layer 7: Sandboxed Execution (resource limits)
    - Layer 8: Audit Logging

    Args:
        command: The shell command to execute.
        working_dir: Optional working directory for command execution.
        timeout: Maximum execution time in seconds (default: 60).
        capture_output: Whether to capture stdout/stderr (default: True).
        project_root: Project root for sandboxing (defaults to working_dir).
        config: Security configuration (defaults to strict mode).

    Returns:
        A dictionary containing:
            - success: bool - True if command executed successfully
            - stdout: str - Standard output from the command
            - stderr: str - Standard error from the command
            - return_code: int - Exit code of the command
            - duration: float - Execution time in seconds
            - error: str | None - Error message if failed

        If approval is required:
            - pending_approval: bool - True if waiting for approval
            - approval_request_id: str - ID to track approval
            - risk_level: str - Risk level of the command
            - risk_factors: list[str] - Reasons for risk assessment

        If command is blocked:
            - success: False
            - error: str - Reason for blocking

    Examples:
        >>> result = shell_executor("ls -la")
        >>> result["success"]
        True

        >>> result = shell_executor("rm -rf /")
        >>> result["success"]
        False
        >>> "blocked" in result["error"].lower()
        True

        >>> result = shell_executor("rm -rf ./build")
        >>> if result.get("pending_approval"):
        ...     print(f"Awaiting approval: {result['approval_request_id']}")
    """
    # Check if shell tool is enabled
    if not _SHELL_TOOL_ENABLED:
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "duration": 0.0,
            "error": (
                "Shell tool is DISABLED. Security safeguards are being validated. "
                "To enable, set _SHELL_TOOL_ENABLED = True in "
                "agentic_cli/tools/shell/executor.py after security review."
            ),
            "disabled": True,
        }

    # Resolve working directory
    if working_dir is None:
        working_dir_path = Path.cwd()
    else:
        working_dir_path = Path(working_dir).resolve()

    # Resolve project root
    if project_root is None:
        project_root_path = working_dir_path
    else:
        project_root_path = Path(project_root).resolve()

    # Use provided config or load default strict config
    if config is None:
        config = get_strict_config()

    # Apply timeout override to execution limits if specified
    if timeout != 60:  # Non-default timeout provided
        config.execution_limits.timeout_seconds = timeout

    # Get audit logger
    audit_logger = _get_audit_logger(config)

    # Run security analysis (includes preprocessing)
    analysis = analyze_command(
        command,
        working_dir=working_dir_path,
        project_root=project_root_path,
        config=config,
    )

    risk = analysis.risk_assessment

    # Extract encoding info for audit
    encoding_detected = []
    if risk.preprocess_result and risk.preprocess_result.encodings_detected:
        encoding_detected = risk.preprocess_result.encodings_detected

    # BLOCK: Critical risk - never allow
    if risk.overall_risk == RiskLevel.CRITICAL or risk.approval_required == ApprovalType.BLOCK:
        block_reasons = risk.block_reasons or ["Command blocked for security reasons"]
        blocked_result = {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "duration": 0.0,
            "error": f"Command blocked: {'; '.join(block_reasons)}",
            "risk_level": risk.overall_risk.value,
            "risk_factors": risk.risk_factors,
        }

        # Audit log the blocked command
        if audit_logger:
            audit_logger.log_command(
                command=command,
                risk_level=risk.overall_risk,
                approval_type=risk.approval_required,
                executed=False,
                working_dir=working_dir_path,
                risk_factors=risk.risk_factors,
                paths=[p.original for p in analysis.path_analysis.paths],
                blocked_reason="; ".join(block_reasons),
                encoding_detected=encoding_detected,
            )

        return blocked_result

    # AUTO-APPROVE: Low risk read-only commands
    if risk.approval_required == ApprovalType.AUTO:
        result = _execute_command(command, working_dir_path, config)

        # Audit log the execution
        if audit_logger:
            audit_logger.log_command(
                command=command,
                risk_level=risk.overall_risk,
                approval_type=risk.approval_required,
                executed=True,
                working_dir=working_dir_path,
                risk_factors=risk.risk_factors,
                paths=[p.original for p in analysis.path_analysis.paths],
                exit_code=result.get("return_code"),
                duration_ms=int(result.get("duration", 0) * 1000),
                stdout=result.get("stdout"),
                stderr=result.get("stderr"),
                encoding_detected=encoding_detected,
            )

        return result

    # APPROVAL REQUIRED: Risky command without auto-approve
    # Return pending state so the caller can handle approval in their own way
    if risk.overall_risk in (RiskLevel.HIGH, RiskLevel.MEDIUM):
        # No approval manager but risky command - return pending state
        # This allows the caller to handle approval in their own way
        pending_result = {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "duration": 0.0,
            "pending_approval": True,
            "approval_request_id": None,
            "command": command,
            "risk_level": risk.overall_risk.value,
            "risk_factors": risk.risk_factors,
            "message": "Command requires approval but no ApprovalManager configured",
            "error": None,
        }

        # Audit log the pending approval
        if audit_logger:
            audit_logger.log_command(
                command=command,
                risk_level=risk.overall_risk,
                approval_type=risk.approval_required,
                executed=False,
                working_dir=working_dir_path,
                risk_factors=risk.risk_factors,
                paths=[p.original for p in analysis.path_analysis.paths],
                encoding_detected=encoding_detected,
            )

        return pending_result

    # Execute command
    result = _execute_command(command, working_dir_path, config)

    # Audit log the execution
    if audit_logger:
        audit_logger.log_command(
            command=command,
            risk_level=risk.overall_risk,
            approval_type=risk.approval_required,
            executed=True,
            working_dir=working_dir_path,
            risk_factors=risk.risk_factors,
            paths=[p.original for p in analysis.path_analysis.paths],
            exit_code=result.get("return_code"),
            duration_ms=int(result.get("duration", 0) * 1000),
            stdout=result.get("stdout"),
            stderr=result.get("stderr"),
            encoding_detected=encoding_detected,
        )

    return result


def _get_operation_type(analysis: SecurityAnalysis) -> str:
    """Get human-readable operation type from analysis."""
    categories = [c.category for c in analysis.classifications]

    if CommandCategory.BLOCKED in categories:
        return "blocked"
    if CommandCategory.PRIVILEGED in categories:
        return "privileged"
    if CommandCategory.WRITE in categories:
        return "write"
    if CommandCategory.NETWORK in categories:
        return "network"
    if CommandCategory.READ in categories:
        return "read"
    return "safe"


def execute_with_approval(
    command: str,
    approval_id: str,
    working_dir: str | None = None,
    timeout: int = 60,
    config: ShellSecurityConfig | None = None,
    user_response: str | None = "approved",
) -> dict[str, Any]:
    """Execute a previously approved command.

    Call this after user approval to execute the command.

    Args:
        command: The shell command to execute.
        approval_id: The approval request ID (for audit trail).
        working_dir: Working directory for execution.
        timeout: Execution timeout in seconds.
        config: Security configuration.
        user_response: The user's approval response (for audit).

    Returns:
        Execution result dictionary.
    """
    # Check if shell tool is enabled
    if not _SHELL_TOOL_ENABLED:
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "duration": 0.0,
            "error": (
                "Shell tool is DISABLED. Security safeguards are being validated. "
                "To enable, set _SHELL_TOOL_ENABLED = True in "
                "agentic_cli/tools/shell/executor.py after security review."
            ),
            "disabled": True,
        }

    # Resolve working directory
    if working_dir is None:
        working_dir_path = Path.cwd()
    else:
        working_dir_path = Path(working_dir).resolve()

    if config is None:
        config = get_strict_config()

    # Apply timeout override if specified
    if timeout != 60:
        config.execution_limits.timeout_seconds = timeout

    # Get audit logger
    audit_logger = _get_audit_logger(config)

    # Execute directly (approval already granted)
    result = _execute_command(command, working_dir_path, config)

    # Add approval ID for audit trail
    result["approval_id"] = approval_id

    # Audit log the approved execution
    if audit_logger:
        audit_logger.log_command(
            command=command,
            risk_level=RiskLevel.MEDIUM,  # Assume medium since it required approval
            approval_type=ApprovalType.ALWAYS_PROMPT,
            executed=True,
            working_dir=working_dir_path,
            user_response=user_response,
            exit_code=result.get("return_code"),
            duration_ms=int(result.get("duration", 0) * 1000),
            stdout=result.get("stdout"),
            stderr=result.get("stderr"),
        )

    return result
