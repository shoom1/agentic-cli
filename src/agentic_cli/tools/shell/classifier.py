"""Command classifier for categorizing shell commands by risk.

Layer 3: Command Classification & Filtering
- Categorize commands: BLOCKED, PRIVILEGED, WRITE, NETWORK, READ, SAFE
- Support user-configurable allow/deny lists
- Pattern matching for dangerous command combinations
"""

import re
from typing import TYPE_CHECKING

from agentic_cli.tools.shell.models import (
    ClassificationResult,
    CommandCategory,
    CommandNode,
)

if TYPE_CHECKING:
    from agentic_cli.tools.shell.config import ShellSecurityConfig


# Commands that are NEVER allowed
BLOCKED_COMMANDS: set[str] = {
    # Privilege escalation
    "sudo",
    "su",
    "doas",
    "pkexec",
    # System control
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init",
    # Destructive filesystem operations
    "mkfs",
    "mkfs.ext4",
    "mkfs.xfs",
    "mkfs.btrfs",
    "fdisk",
    "parted",
    "gdisk",
    # Dangerous utilities
    "shred",
    "wipe",
}

# Patterns that indicate BLOCKED commands (regex)
BLOCKED_PATTERNS: list[tuple[str, str]] = [
    # rm -rf / variations
    (r"rm\s+(-[rf]+\s+)+/\s*$", "rm -rf / (delete entire filesystem)"),
    (r"rm\s+(-[rf]+\s+)+/[a-zA-Z]+\s*$", "rm of system directory"),
    (r"rm\s+(-[rf]+\s+)+~/?", "rm -rf ~ (delete home directory)"),
    (r"rm\s+(-[rf]+\s+)+\$HOME", "rm -rf $HOME"),
    (r"rm\s+(-[rf]+\s+)+/Users/", "rm of /Users directory (macOS)"),
    (r"rm\s+(-[rf]+\s+)+/home/", "rm of /home directory (Linux)"),
    # rm -rf * (wildcard delete)
    (r"rm\s+-[rf]*\s+\*", "rm -rf * (wildcard delete)"),
    (r"rm\s+-[fr]*\s+\*", "rm -fr * (wildcard delete)"),
    # Fork bombs
    (r":\(\)\s*\{.*:\|:&\s*\}\s*;:", "Fork bomb"),
    (r"fork\s*\(\s*\)\s*while", "Fork loop"),
    # Direct disk writes
    (r"dd\s+.*of=/dev/[sh]d", "Direct disk write with dd"),
    (r">\s*/dev/[sh]d", "Redirect to disk device"),
    (r"cat\s+/dev/zero\s*>\s*/", "Overwrite with /dev/zero"),
    (r"cat\s+/dev/random\s*>\s*/", "Overwrite with /dev/random"),
    # Dangerous permission changes
    (r"chmod\s+(-R\s+)?777\s+/\s*$", "chmod 777 / (world writable root)"),
    (r"chmod\s+-R\s+777\s+/", "Recursive chmod 777 on system"),
    (r"chown\s+.*\s+/\s*$", "chown of root filesystem"),
    # Remote code execution via pipe
    (r"curl\s+.*\|\s*(ba)?sh", "Curl piped to shell"),
    (r"wget\s+.*\|\s*(ba)?sh", "Wget piped to shell"),
    (r"curl\s+.*\|\s*python", "Curl piped to python"),
    (r"wget\s+.*\|\s*python", "Wget piped to python"),
]

# Compile blocked patterns
_COMPILED_BLOCKED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), desc)
    for pattern, desc in BLOCKED_PATTERNS
]

# Commands requiring explicit approval (system/privilege operations)
PRIVILEGED_COMMANDS: set[str] = {
    "chmod",
    "chown",
    "chgrp",
    "mount",
    "umount",
    "systemctl",
    "service",
    "iptables",
    "ufw",
    "firewall-cmd",
    "launchctl",  # macOS
    "defaults",  # macOS system preferences
}

# Commands that modify filesystem
WRITE_COMMANDS: set[str] = {
    # File deletion
    "rm",
    "rmdir",
    "unlink",
    # File movement/copy
    "mv",
    "cp",
    # File creation
    "mkdir",
    "touch",
    "tee",
    "truncate",
    # In-place editing
    "sed",  # with -i flag
    "awk",  # with -i flag
    "patch",
    # Archive operations
    "tar",
    "zip",
    "unzip",
    "gzip",
    "gunzip",
    # Version control write ops
    "git",  # some ops are write
}

# Git subcommands that are write operations
GIT_WRITE_SUBCOMMANDS: set[str] = {
    "add",
    "commit",
    "push",
    "pull",
    "merge",
    "rebase",
    "reset",
    "checkout",
    "branch",
    "stash",
    "cherry-pick",
    "revert",
    "clean",
    "rm",
    "mv",
}

# Commands with network access
NETWORK_COMMANDS: set[str] = {
    "curl",
    "wget",
    "fetch",
    "ssh",
    "scp",
    "rsync",
    "sftp",
    "nc",
    "netcat",
    "ncat",
    "telnet",
    "ftp",
    "ping",
    "traceroute",
    "tracepath",
    "nslookup",
    "dig",
    "host",
    # Package managers with network
    "npm",
    "yarn",
    "pip",
    "pip3",
    "cargo",
    "gem",
    "composer",
    "go",
    "brew",
}

# Package manager subcommands that are safe (read-only)
PACKAGE_MANAGER_SAFE_SUBCOMMANDS: set[str] = {
    "list",
    "show",
    "info",
    "search",
    "help",
    "version",
    "--version",
    "-v",
    "--help",
    "-h",
}

# Commands that are always safe (read-only)
SAFE_COMMANDS: set[str] = {
    # Directory listing
    "ls",
    "dir",
    "tree",
    "exa",
    "lsd",
    # File viewing
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "bat",
    # Search
    "grep",
    "rg",
    "ag",
    "ack",
    "find",
    "fd",
    "locate",
    "which",
    "whereis",
    "type",
    "command",
    # Text processing (read-only)
    "wc",
    "sort",
    "uniq",
    "diff",
    "comm",
    "cut",
    "tr",
    "paste",
    "join",
    "column",
    "jq",
    "yq",
    "xq",
    # Output
    "echo",
    "printf",
    "date",
    "cal",
    # System info
    "pwd",
    "whoami",
    "hostname",
    "uname",
    "id",
    "groups",
    "env",
    "printenv",
    "set",
    # File info
    "file",
    "stat",
    "du",
    "df",
    "realpath",
    "dirname",
    "basename",
    # Help
    "man",
    "info",
    "help",
    # Version control (read-only)
    "git",  # default to safe, write ops checked separately
}

# Git subcommands that are read-only
GIT_SAFE_SUBCOMMANDS: set[str] = {
    "status",
    "log",
    "diff",
    "show",
    "branch",  # listing
    "remote",  # listing
    "tag",  # listing
    "describe",
    "rev-parse",
    "rev-list",
    "ls-files",
    "ls-tree",
    "cat-file",
    "blame",
    "shortlog",
    "whatchanged",
    "reflog",
    "config",  # reading
    "help",
    "--help",
    "-h",
    "--version",
}


class CommandClassifier:
    """Classifies shell commands by risk category.

    Supports user-configurable allow/deny lists that override defaults.
    """

    def __init__(self, config: "ShellSecurityConfig | None" = None):
        """Initialize classifier with optional config.

        Args:
            config: Optional ShellSecurityConfig for custom rules.
        """
        self.config = config

        # Build effective lists from config
        self._user_allow = set(config.allow_commands) if config else set()
        self._user_deny = set(config.deny_commands) if config else set()
        self._user_allow_patterns = (
            [re.compile(p) for p in config.allow_patterns] if config else []
        )
        self._user_deny_patterns = (
            [re.compile(p) for p in config.deny_patterns] if config else []
        )

    def classify(self, node: CommandNode) -> ClassificationResult:
        """Classify a single command node.

        Args:
            node: The CommandNode to classify.

        Returns:
            ClassificationResult with category and explanation.
        """
        command = node.command.lower() if node.command else ""
        raw = node.raw_command

        # Check user deny list first (highest priority)
        if command in self._user_deny:
            return ClassificationResult(
                category=CommandCategory.BLOCKED,
                command=command,
                reason="Command in user deny list",
            )

        # Check user deny patterns
        for pattern in self._user_deny_patterns:
            if pattern.search(raw):
                return ClassificationResult(
                    category=CommandCategory.BLOCKED,
                    command=command,
                    matched_pattern=pattern.pattern,
                    reason="Matches user deny pattern",
                )

        # Check user allow list (overrides default classification)
        if command in self._user_allow:
            return ClassificationResult(
                category=CommandCategory.SAFE,
                command=command,
                reason="Command in user allow list",
            )

        # Check user allow patterns
        for pattern in self._user_allow_patterns:
            if pattern.search(raw):
                return ClassificationResult(
                    category=CommandCategory.SAFE,
                    command=command,
                    matched_pattern=pattern.pattern,
                    reason="Matches user allow pattern",
                )

        # Check blocked commands
        if command in BLOCKED_COMMANDS:
            return ClassificationResult(
                category=CommandCategory.BLOCKED,
                command=command,
                reason=f"'{command}' is a blocked command",
            )

        # Check blocked patterns
        for pattern, description in _COMPILED_BLOCKED_PATTERNS:
            if pattern.search(raw):
                return ClassificationResult(
                    category=CommandCategory.BLOCKED,
                    command=command,
                    matched_pattern=pattern.pattern,
                    reason=description,
                )

        # Check privileged commands
        if command in PRIVILEGED_COMMANDS:
            return ClassificationResult(
                category=CommandCategory.PRIVILEGED,
                command=command,
                reason=f"'{command}' requires elevated privileges",
            )

        # Check network commands (with subcommand analysis)
        if command in NETWORK_COMMANDS:
            # Check if it's a safe subcommand (like npm list)
            if node.args and node.args[0].lower() in PACKAGE_MANAGER_SAFE_SUBCOMMANDS:
                return ClassificationResult(
                    category=CommandCategory.READ,
                    command=command,
                    reason=f"'{command} {node.args[0]}' is read-only",
                )
            return ClassificationResult(
                category=CommandCategory.NETWORK,
                command=command,
                reason=f"'{command}' has network access",
            )

        # Check write commands (with special handling for git)
        if command in WRITE_COMMANDS:
            # Special handling for git
            if command == "git":
                return self._classify_git(node)

            # Check for sed/awk with -i flag
            if command in ("sed", "awk"):
                if "-i" in node.args or any(
                    arg.startswith("-i") for arg in node.args
                ):
                    return ClassificationResult(
                        category=CommandCategory.WRITE,
                        command=command,
                        reason=f"'{command} -i' modifies files in-place",
                    )
                return ClassificationResult(
                    category=CommandCategory.READ,
                    command=command,
                    reason=f"'{command}' without -i is read-only",
                )

            return ClassificationResult(
                category=CommandCategory.WRITE,
                command=command,
                reason=f"'{command}' modifies filesystem",
            )

        # Check safe commands
        if command in SAFE_COMMANDS:
            return ClassificationResult(
                category=CommandCategory.SAFE,
                command=command,
                reason=f"'{command}' is a safe read-only command",
            )

        # Unknown command - default to READ (conservative but allows execution)
        return ClassificationResult(
            category=CommandCategory.READ,
            command=command,
            reason=f"'{command}' is not in known lists - treating as read",
        )

    def _classify_git(self, node: CommandNode) -> ClassificationResult:
        """Classify git commands based on subcommand."""
        command = "git"

        if not node.args:
            return ClassificationResult(
                category=CommandCategory.SAFE,
                command=command,
                reason="git with no subcommand (shows help)",
            )

        subcommand = node.args[0].lower()

        # Check if it's just getting help/version
        if subcommand in ("--help", "-h", "--version"):
            return ClassificationResult(
                category=CommandCategory.SAFE,
                command=command,
                reason="git help/version",
            )

        # Check safe subcommands
        if subcommand in GIT_SAFE_SUBCOMMANDS:
            # Special case: branch -d/-D is a write operation
            if subcommand == "branch" and any(
                arg in ("-d", "-D", "--delete") for arg in node.args
            ):
                return ClassificationResult(
                    category=CommandCategory.WRITE,
                    command=command,
                    reason="git branch delete modifies repository",
                )
            return ClassificationResult(
                category=CommandCategory.SAFE,
                command=command,
                reason=f"'git {subcommand}' is read-only",
            )

        # Check write subcommands
        if subcommand in GIT_WRITE_SUBCOMMANDS:
            return ClassificationResult(
                category=CommandCategory.WRITE,
                command=command,
                reason=f"'git {subcommand}' modifies repository",
            )

        # Unknown git subcommand - conservative approach
        return ClassificationResult(
            category=CommandCategory.READ,
            command=command,
            reason=f"'git {subcommand}' - unknown subcommand, treating as read",
        )

    def classify_all(self, nodes: list[CommandNode]) -> list[ClassificationResult]:
        """Classify all command nodes, including subshells and pipes.

        Args:
            nodes: List of CommandNode objects.

        Returns:
            List of ClassificationResult for all commands found.
        """
        results = []

        def classify_recursive(node: CommandNode):
            # Classify this node
            results.append(self.classify(node))

            # Classify subshells
            for subshell in node.subshells:
                classify_recursive(subshell)

            # Classify piped commands
            if node.pipes_to:
                classify_recursive(node.pipes_to)

            # Classify chained commands
            for _, chained in node.chained_with:
                classify_recursive(chained)

        for node in nodes:
            classify_recursive(node)

        return results

    def get_highest_risk_category(
        self, results: list[ClassificationResult]
    ) -> CommandCategory:
        """Get the highest risk category from multiple classifications.

        Order (highest to lowest): BLOCKED > PRIVILEGED > WRITE > NETWORK > READ > SAFE
        """
        priority = {
            CommandCategory.BLOCKED: 6,
            CommandCategory.PRIVILEGED: 5,
            CommandCategory.WRITE: 4,
            CommandCategory.NETWORK: 3,
            CommandCategory.READ: 2,
            CommandCategory.SAFE: 1,
        }

        if not results:
            return CommandCategory.SAFE

        return max(results, key=lambda r: priority[r.category]).category

    def check_full_command(self, full_command: str) -> ClassificationResult | None:
        """Check the full command string against blocked patterns.

        This catches patterns that span multiple commands (e.g., curl | bash)
        which would be split during tokenization.

        Args:
            full_command: The complete command string.

        Returns:
            ClassificationResult if blocked, None otherwise.
        """
        # Check user deny patterns first
        for pattern in self._user_deny_patterns:
            if pattern.search(full_command):
                return ClassificationResult(
                    category=CommandCategory.BLOCKED,
                    command=full_command.split()[0] if full_command.split() else "",
                    matched_pattern=pattern.pattern,
                    reason="Matches user deny pattern",
                )

        # Check built-in blocked patterns
        for pattern, description in _COMPILED_BLOCKED_PATTERNS:
            if pattern.search(full_command):
                return ClassificationResult(
                    category=CommandCategory.BLOCKED,
                    command=full_command.split()[0] if full_command.split() else "",
                    matched_pattern=pattern.pattern,
                    reason=description,
                )

        return None
