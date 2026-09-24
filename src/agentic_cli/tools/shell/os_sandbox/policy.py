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

# Credential stores: never readable or writable from the sandbox, whatever the
# configured policy says. Reads are otherwise broad (interpreters need the
# filesystem), so these must be named. See also the app's own config dir,
# added per policy from ``app_name``.
PROTECTED_HOME_PATHS: list[str] = [
    "~/.ssh",
    "~/.gnupg",
    "~/.aws",
    "~/.azure",
    "~/.kube",
    "~/.config/gcloud",
    "~/.config/gh",
    "~/.docker/config.json",
    "~/.netrc",
    "~/.git-credentials",
    "~/.pypirc",
    "~/.npmrc",
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
        app_name: The application's name. Its config dir (``~/.{app_name}``,
            holding settings, permission grants and ``.env``) is hidden and
            write-protected, and ``{working_dir}/.{app_name}`` is
            write-protected.

    Whatever the configured lists hold, the resolved deny lists always include
    ``PROTECTED_HOME_PATHS``, the app's config dir, the whole ``.git``
    directory and ``.env`` of the working directory.
        strict: When True, refuse to execute if sandboxing was requested but no
            real backend is available (instead of falling back to the restricted
            in-process executor).
    """

    enabled: bool = True
    writable_paths: list[str] = field(default_factory=list)
    deny_write_paths: list[str] = field(
        default_factory=lambda: list(MANDATORY_DENY_WRITE)
    )
    deny_read_paths: list[str] = field(default_factory=list)
    allow_network: bool = False
    strict: bool = False
    app_name: str | None = None

    def resolved_writable_paths(self, working_dir: Path) -> list[Path]:
        """Resolve all writable paths to absolute.

        The working directory is writable unless it is the home directory, an
        ancestor of it, or the filesystem root: started from ``~`` the whole
        home would otherwise be writable. Explicitly configured paths are
        always included.

        Args:
            working_dir: The command's working directory.

        Returns:
            Deduplicated list of resolved absolute paths.
        """
        paths: set[Path] = set()
        wd = working_dir.resolve()
        home = Path.home().resolve()
        if not (wd == Path(wd.anchor) or home.is_relative_to(wd)):
            paths.add(wd)
        for p in self.writable_paths:
            paths.add(Path(p).expanduser().resolve())
        return sorted(paths)

    def resolved_deny_write_paths(self, working_dir: Path | None = None) -> list[Path]:
        """Resolve all deny-write paths to absolute.

        Always adds the protected home paths and the app's config dir, and,
        given a working directory, its whole ``.git`` (protecting only
        ``.git/hooks`` let the directory be moved aside and replaced) and
        ``.{app_name}``.

        Args:
            working_dir: Working directory for resolving relative deny paths.

        Returns:
            List of resolved absolute paths that must not be writable.
        """
        paths = [Path(p).expanduser().resolve() for p in self.deny_write_paths]
        paths += self._protected_home_paths()
        if working_dir is not None:
            wd = working_dir.resolve()
            paths.append(wd / ".git")
            if self.app_name:
                paths.append(wd / f".{self.app_name}")
        return _dedupe(paths)

    def resolved_deny_read_paths(self, working_dir: Path | None = None) -> list[Path]:
        """Resolve all deny-read paths to absolute.

        Always adds the protected home paths and the app's config dir, and,
        given a working directory, its ``.env``.

        Args:
            working_dir: Working directory for resolving relative deny paths.

        Returns:
            List of resolved absolute paths to hide.
        """
        paths = [Path(p).expanduser().resolve() for p in self.deny_read_paths]
        paths += self._protected_home_paths()
        if working_dir is not None:
            paths.append(working_dir.resolve() / ".env")
        return _dedupe(paths)

    def _protected_home_paths(self) -> list[Path]:
        entries = list(PROTECTED_HOME_PATHS)
        if self.app_name:
            entries.append(f"~/.{self.app_name}")
        return [Path(p).expanduser().resolve() for p in entries]

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


def _dedupe(paths: list[Path]) -> list[Path]:
    """Drop repeated paths, keeping the first occurrence's position."""
    return list(dict.fromkeys(paths))
