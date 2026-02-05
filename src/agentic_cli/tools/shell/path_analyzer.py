"""Path analyzer for sandboxing shell commands.

Layer 4: Path Analysis & Sandboxing
- Resolve paths to absolute
- Detect path traversal attacks
- Enforce project directory boundaries
- Block access to sensitive paths
"""

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_cli.tools.shell.models import (
    CommandCategory,
    CommandNode,
    PathAnalysisResult,
    PathCheck,
    ClassificationResult,
)

if TYPE_CHECKING:
    from agentic_cli.tools.shell.config import ShellSecurityConfig


# Sensitive paths that should never be written to
SENSITIVE_PATHS: list[Path] = [
    Path.home() / ".ssh",
    Path.home() / ".gnupg",
    Path.home() / ".gpg",
    Path.home() / ".aws",
    Path.home() / ".azure",
    Path.home() / ".gcloud",
    Path.home() / ".config" / "gcloud",
    Path.home() / ".kube",
    Path.home() / ".docker",
    Path("/etc"),
    Path("/usr"),
    Path("/var"),
    Path("/boot"),
    Path("/root"),
    Path("/System"),  # macOS
    Path("/Library"),  # macOS
    Path("/Applications"),  # macOS
]

# Patterns that look like path traversal
TRAVERSAL_PATTERNS = [
    re.compile(r"\.\."),  # ..
    re.compile(r"%2e%2e", re.IGNORECASE),  # URL encoded ..
    re.compile(r"%252e%252e", re.IGNORECASE),  # Double URL encoded
    re.compile(r"\x00"),  # Null byte
]

# Glob/wildcard patterns
GLOB_PATTERNS = re.compile(r"[*?\[\]]")


class PathAnalyzer:
    """Analyzes paths in shell commands for security.

    Ensures commands stay within allowed directories and don't
    access sensitive system paths.
    """

    def __init__(
        self,
        project_root: Path,
        config: "ShellSecurityConfig | None" = None,
    ):
        """Initialize path analyzer.

        Args:
            project_root: The project root directory (primary sandbox).
            config: Optional ShellSecurityConfig for allowed paths.
        """
        self.project_root = project_root.resolve()
        self.config = config

        # Build allowed paths list
        self._allowed_paths = [self.project_root]
        if config and config.allowed_paths:
            self._allowed_paths.extend(
                Path(p).expanduser().resolve() for p in config.allowed_paths
            )

        # Build sensitive paths list
        self._sensitive_paths = SENSITIVE_PATHS.copy()

    def analyze(
        self,
        nodes: list[CommandNode],
        classifications: list[ClassificationResult],
        working_dir: Path,
    ) -> PathAnalysisResult:
        """Analyze all paths in command nodes.

        Args:
            nodes: Parsed command nodes.
            classifications: Classification results for context (write vs read).
            working_dir: Working directory for relative path resolution.

        Returns:
            PathAnalysisResult with all path checks.
        """
        result = PathAnalysisResult(paths=[])
        working_dir = working_dir.resolve()

        # Build a map of command -> classification for context
        cmd_to_class = {}
        for c in classifications:
            cmd_to_class[c.command] = c

        # Extract and analyze paths from all nodes
        for node in nodes:
            self._analyze_node(node, cmd_to_class, working_dir, result)

        # Compute summary flags
        result.has_sensitive_paths = any(p.is_sensitive for p in result.paths)
        result.has_outside_project = any(
            not p.in_project and not p.in_allowed for p in result.paths
        )
        result.has_traversal = any(p.traversal_detected for p in result.paths)

        return result

    def _analyze_node(
        self,
        node: CommandNode,
        cmd_to_class: dict[str, ClassificationResult],
        working_dir: Path,
        result: PathAnalysisResult,
    ):
        """Analyze paths in a single command node."""
        classification = cmd_to_class.get(node.command.lower())
        is_write_cmd = classification and classification.category in (
            CommandCategory.WRITE,
            CommandCategory.PRIVILEGED,
        )

        # Extract paths from arguments
        for arg in node.args:
            # Skip flags
            if arg.startswith("-"):
                continue

            # Check if this looks like a path
            if self._looks_like_path(arg):
                check = self._check_path(arg, working_dir)
                result.paths.append(check)

                if is_write_cmd:
                    result.write_paths.append(check)
                else:
                    result.read_paths.append(check)

        # Check redirection targets
        for redir in node.redirections:
            if redir.target:
                check = self._check_path(redir.target, working_dir)
                result.paths.append(check)
                # Redirections are always write operations
                result.write_paths.append(check)

        # Recursively analyze subshells
        for subshell in node.subshells:
            self._analyze_node(subshell, cmd_to_class, working_dir, result)

        # Recursively analyze piped commands
        if node.pipes_to:
            self._analyze_node(node.pipes_to, cmd_to_class, working_dir, result)

        # Recursively analyze chained commands
        for _, chained in node.chained_with:
            self._analyze_node(chained, cmd_to_class, working_dir, result)

    def _looks_like_path(self, arg: str) -> bool:
        """Check if an argument looks like a file path."""
        # Absolute paths
        if arg.startswith("/"):
            return True
        # Home directory
        if arg.startswith("~"):
            return True
        # Relative paths with directory separators
        if "/" in arg or "\\" in arg:
            return True
        # Current/parent directory references
        if arg in (".", "..") or arg.startswith("./") or arg.startswith("../"):
            return True
        # Environment variables that might be paths
        if arg.startswith("$"):
            return True
        # Has file extension (heuristic)
        if "." in arg and not arg.startswith("-"):
            return True
        return False

    def _check_path(self, path_str: str, working_dir: Path) -> PathCheck:
        """Check a single path string."""
        check = PathCheck(original=path_str)

        # Check for traversal patterns
        check.traversal_detected = self._has_traversal(path_str)

        # Check for glob patterns
        check.is_glob = bool(GLOB_PATTERNS.search(path_str))

        # Try to resolve the path
        try:
            # Expand environment variables
            expanded = os.path.expandvars(path_str)
            # Expand user (~)
            expanded = os.path.expanduser(expanded)

            # Handle relative paths
            if not os.path.isabs(expanded):
                expanded = str(working_dir / expanded)

            resolved = Path(expanded).resolve()
            check.resolved = resolved

            # Check if in project
            check.in_project = self._is_in_directory(resolved, self.project_root)

            # Check if in any allowed path
            check.in_allowed = any(
                self._is_in_directory(resolved, allowed)
                for allowed in self._allowed_paths
            )

            # Check if sensitive
            check.is_sensitive = self._is_sensitive_path(resolved)

        except (ValueError, OSError):
            # Path couldn't be resolved
            check.resolved = None
            # Be conservative - treat unresolvable as potentially sensitive
            check.is_sensitive = True

        return check

    def _has_traversal(self, path_str: str) -> bool:
        """Check for path traversal patterns."""
        for pattern in TRAVERSAL_PATTERNS:
            if pattern.search(path_str):
                return True
        return False

    def _is_in_directory(self, path: Path, directory: Path) -> bool:
        """Check if a path is inside a directory."""
        try:
            path.relative_to(directory)
            return True
        except ValueError:
            return False

    def _is_sensitive_path(self, path: Path) -> bool:
        """Check if a path is sensitive."""
        for sensitive in self._sensitive_paths:
            try:
                # Check if path is inside sensitive path
                if self._is_in_directory(path, sensitive):
                    return True
                # Check if path equals sensitive path
                if path == sensitive:
                    return True
            except (ValueError, OSError):
                continue
        return False

    def get_path_risk_factors(self, result: PathAnalysisResult) -> list[str]:
        """Get human-readable risk factors from path analysis."""
        factors = []

        if result.has_traversal:
            factors.append("Path traversal detected (..)")

        if result.has_sensitive_paths:
            sensitive = [p.original for p in result.paths if p.is_sensitive]
            factors.append(f"Sensitive paths: {', '.join(sensitive[:3])}")

        if result.has_outside_project:
            outside = [
                p.original
                for p in result.paths
                if not p.in_project and not p.in_allowed
            ]
            factors.append(f"Paths outside project: {', '.join(outside[:3])}")

        # Check write paths specifically
        sensitive_writes = [p for p in result.write_paths if p.is_sensitive]
        if sensitive_writes:
            factors.append(
                f"Writing to sensitive paths: {', '.join(p.original for p in sensitive_writes[:3])}"
            )

        return factors
