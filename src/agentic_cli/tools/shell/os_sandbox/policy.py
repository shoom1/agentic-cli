"""Sandbox policy defining what OS-level sandboxing allows.

Specifies writable paths, mandatory deny lists, and network policy.
The policy is resolved at wrap time (not config time) because working
directories change between invocations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Paths that must never be writable, regardless of policy.
# These protect shell configs, credentials, and system directories
# from agent-generated commands.
MANDATORY_DENY_WRITE: list[str] = [
    # Shell configuration
    "~/.bashrc",
    "~/.bash_profile",
    "~/.zshrc",
    "~/.zprofile",
    "~/.profile",
    # Git configuration
    "~/.gitconfig",
    "~/.git-credentials",
    # SSH and GPG
    "~/.ssh/",
    "~/.gnupg/",
    # Note: .git/hooks/ and .git/config are added dynamically
    # relative to working_dir in resolved_deny_write_paths()
    # System directories
    "/etc/",
    "/usr/",
    "/bin/",
    "/sbin/",
    # macOS system
    "/System/",
    "/Library/",
    # Agent config
    "~/.claude/",
]

# Paths that are always readable (system libraries, interpreters, etc.)
DEFAULT_READABLE: list[str] = [
    "/usr/",
    "/bin/",
    "/sbin/",
    "/lib/",
    "/lib64/",
    "/etc/",
    "/dev/null",
    "/dev/urandom",
    "/dev/zero",
    "/tmp/",
    # macOS system libraries
    "/System/Library/",
    "/Library/Frameworks/",
    "/Applications/Xcode.app/",
    # Common package manager locations
    "/opt/homebrew/",
    "/usr/local/",
]


@dataclass
class OSSandboxPolicy:
    """Policy defining what OS-level sandboxing allows.

    Attributes:
        enabled: Whether OS sandboxing is active.
        writable_paths: Additional paths the sandboxed process can write to.
            The working directory is always writable regardless of this list.
        deny_write_paths: Paths denied for writing even within writable dirs.
            Always includes MANDATORY_DENY_WRITE entries.
        deny_read_paths: Paths to hide entirely from the sandboxed process.
        allow_network: Whether network access is allowed (Phase 2).
    """

    enabled: bool = True
    writable_paths: list[str] = field(default_factory=list)
    deny_write_paths: list[str] = field(
        default_factory=lambda: list(MANDATORY_DENY_WRITE)
    )
    deny_read_paths: list[str] = field(default_factory=list)
    allow_network: bool = False

    def resolved_writable_paths(self, working_dir: Path) -> list[Path]:
        """Resolve all writable paths to absolute, always including working_dir.

        Args:
            working_dir: The command's working directory.

        Returns:
            Deduplicated list of resolved absolute paths.
        """
        paths = {working_dir.resolve()}
        for p in self.writable_paths:
            paths.add(Path(p).expanduser().resolve())
        return sorted(paths)

    def resolved_deny_write_paths(self, working_dir: Path | None = None) -> list[Path]:
        """Resolve all deny-write paths to absolute.

        Adds .git/hooks/ and .git/config relative to working_dir
        if a working directory is provided.

        Args:
            working_dir: Working directory for resolving relative deny paths.

        Returns:
            List of resolved absolute paths that must not be writable.
        """
        paths: list[Path] = []
        for p in self.deny_write_paths:
            resolved = Path(p).expanduser().resolve()
            paths.append(resolved)

        # Add git-related deny paths relative to working_dir
        if working_dir is not None:
            wd = working_dir.resolve()
            paths.append(wd / ".git" / "hooks")
            paths.append(wd / ".git" / "config")

        return paths

    def resolved_deny_read_paths(self) -> list[Path]:
        """Resolve all deny-read paths to absolute.

        Returns:
            List of resolved absolute paths to hide.
        """
        return [Path(p).expanduser().resolve() for p in self.deny_read_paths]

    def resolved_readable_paths(self) -> list[Path]:
        """Resolve default readable paths.

        Returns:
            List of resolved paths that should always be readable.
        """
        paths: list[Path] = []
        for p in DEFAULT_READABLE:
            path = Path(p).expanduser()
            # Don't resolve symlinks for system paths — they may not exist
            # on all platforms (e.g., /lib64 on macOS)
            paths.append(path)
        return paths
