"""Sandbox policy defining what OS-level sandboxing allows.

Specifies writable paths, mandatory deny lists, and network policy.
The policy is resolved at wrap time (not config time) because working
directories change between invocations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agentic_cli.protected_paths import (
    APP_CONFIG_DIR,
    CREDENTIAL_STORES,
    NEVER_WRITABLE,
    WORKDIR_NEVER_READABLE,
    WORKDIR_NEVER_WRITABLE,
    ProtectedPath,
)

# The protected locations come from ``agentic_cli.protected_paths``, the lists
# the permission engine's builtin deny rules are built from, so the sandbox
# protects what the engine does.

# Paths that must never be writable: shell configs, git's config, credentials
# and system directories (a directory spelled with a trailing slash). Note:
# the working directory's ``.git`` and ``.{app_name}`` are added in
# resolved_deny_write_paths().
MANDATORY_DENY_WRITE: list[str] = [
    f"{p.location}/" if p.is_dir else p.location for p in NEVER_WRITABLE
]

# Credential stores: never readable or writable from the sandbox, whatever the
# configured policy says. Reads are otherwise broad (interpreters need the
# filesystem), so these must be named. See also the app's own config dir,
# added per policy from ``app_name``.
PROTECTED_HOME_PATHS: list[str] = [p.location for p in CREDENTIAL_STORES]

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
        strict: When True (the default), refuse to execute if sandboxing was
            requested but no real backend is available or the wrap fails. When
            False, Python execution then falls back to running without a
            sandbox, with the user's privileges.
        app_name: The application's name. Its config dir (``~/.{app_name}``,
            holding settings, permission grants and ``.env``) is hidden and
            write-protected, and ``{working_dir}/.{app_name}`` is
            write-protected.

    Whatever the configured lists hold, the resolved deny lists always include
    ``PROTECTED_HOME_PATHS``, the app's config dir, the whole ``.git``
    directory and ``.env`` of the working directory.
    """

    enabled: bool = True
    writable_paths: list[str] = field(default_factory=list)
    deny_write_paths: list[str] = field(
        default_factory=lambda: list(MANDATORY_DENY_WRITE)
    )
    deny_read_paths: list[str] = field(default_factory=list)
    allow_network: bool = False
    strict: bool = True
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
            paths += self._in_working_dir(WORKDIR_NEVER_WRITABLE, working_dir)
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
            paths += self._in_working_dir(WORKDIR_NEVER_READABLE, working_dir)
        return _dedupe(paths)

    def _protected_home_paths(self) -> list[Path]:
        entries = list(PROTECTED_HOME_PATHS)
        if self.app_name:
            entries.append(APP_CONFIG_DIR.location.format(app_name=self.app_name))
        return [Path(p).expanduser().resolve() for p in entries]

    def _in_working_dir(
        self, entries: tuple[ProtectedPath, ...], working_dir: Path
    ) -> list[Path]:
        """The working directory's protected ``entries``; one naming the app
        is left out when the policy has no ``app_name``."""
        wd = working_dir.resolve()
        return [
            wd / p.location.format(app_name=self.app_name)
            for p in entries
            if self.app_name or "{app_name}" not in p.location
        ]

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
