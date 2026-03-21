"""Generate macOS Seatbelt (SBPL) profiles for sandbox-exec.

Seatbelt uses Apple's Scheme-based profile language (SBPL) to define
sandbox policies. Key semantics:
- Profiles start with (deny default) — everything blocked unless allowed.
- Later rules override earlier ones for the same operation.
- This means deny-write rules MUST come AFTER allow-write rules to ensure
  mandatory denies can't be overridden.

Reference: Anthropic's sandbox-runtime generates similar profiles for
Claude Code's macOS sandboxing.
"""

from __future__ import annotations

from pathlib import Path


def _escape_sbpl_path(path: str) -> str:
    """Escape a path for use in SBPL string literals.

    SBPL uses double-quoted strings. Backslashes and double quotes
    need escaping.

    Args:
        path: The filesystem path to escape.

    Returns:
        Escaped path safe for SBPL string context.
    """
    return path.replace("\\", "\\\\").replace('"', '\\"')


def _subpath_rule(operation: str, path: Path, allow: bool = True) -> str:
    """Generate a subpath rule for a file operation.

    Args:
        operation: SBPL operation (e.g. "file-read*", "file-write*").
        path: The filesystem path.
        allow: Whether to allow or deny.

    Returns:
        SBPL rule string.
    """
    action = "allow" if allow else "deny"
    escaped = _escape_sbpl_path(str(path))
    return f'({action} {operation} (subpath "{escaped}"))'


def _literal_rule(operation: str, path: Path, allow: bool = True) -> str:
    """Generate a literal path rule for a file operation.

    Args:
        operation: SBPL operation.
        path: The exact file path.
        allow: Whether to allow or deny.

    Returns:
        SBPL rule string.
    """
    action = "allow" if allow else "deny"
    escaped = _escape_sbpl_path(str(path))
    return f'({action} {operation} (literal "{escaped}"))'


def generate_seatbelt_profile(
    writable_paths: list[Path],
    deny_write_paths: list[Path],
    readable_paths: list[Path],
    deny_read_paths: list[Path],
    allow_network: bool = False,
) -> str:
    """Generate a Seatbelt SBPL profile string.

    Profile structure:
    1. (version 1) + (deny default) — deny everything
    2. Process operations (exec, fork, signal)
    3. System operations (sysctl, mach IPC)
    4. Device I/O
    5. File read rules (allow system dirs + working dir + readable paths)
    6. File write rules (allow writable paths)
    7. Deny write rules (mandatory denies AFTER allows — later wins)
    8. Network rules

    Args:
        writable_paths: Paths allowed for writing.
        deny_write_paths: Paths denied for writing (overrides allows).
        readable_paths: Currently unused — retained for forward compatibility
            with Phase 2 (granular read restrictions). The broad
            ``(allow file-read* (subpath "/"))`` covers all reads.
        deny_read_paths: Paths to block from reading.
        allow_network: Whether to allow network access.

    Returns:
        Complete SBPL profile as a string.
    """
    lines: list[str] = []

    # --- Header ---
    lines.append("(version 1)")
    lines.append("(deny default)")
    lines.append("")

    # --- Process operations ---
    lines.append(";; Process operations")
    lines.append("(allow process-exec)")
    lines.append("(allow process-fork)")
    lines.append("(allow signal (target self))")
    lines.append("")

    # --- System operations ---
    lines.append(";; System operations")
    lines.append("(allow sysctl-read)")
    lines.append("")

    # --- Mach IPC for essential macOS services ---
    lines.append(";; Essential Mach IPC services")
    lines.append("(allow mach-lookup")
    lines.append('    (global-name "com.apple.system.logger")')
    lines.append('    (global-name "com.apple.system.notification_center")')
    lines.append('    (global-name "com.apple.CoreServices.coreservicesd")')
    lines.append('    (global-name "com.apple.SecurityServer")')
    lines.append('    (global-name "com.apple.lsd.mapdb")')
    lines.append('    (global-name "com.apple.coreservices.launchservicesd")')
    lines.append('    (global-name "com.apple.fonts")')
    lines.append('    (global-name "com.apple.FontObjectsServer")')
    lines.append('    (global-name "com.apple.logd")')
    lines.append(")")
    lines.append("")

    # --- POSIX shared memory and semaphores ---
    lines.append(";; POSIX IPC")
    lines.append("(allow ipc-posix-shm*)")
    lines.append("(allow ipc-posix-sem)")
    lines.append("")

    # --- Device I/O ---
    lines.append(";; Device I/O")
    lines.append('(allow file-read* (subpath "/dev"))')
    lines.append('(allow file-write* (literal "/dev/null"))')
    lines.append('(allow file-write* (literal "/dev/dfd"))')
    lines.append('(allow file-write* (literal "/dev/tty"))')
    lines.append('(allow file-ioctl (literal "/dev/tty"))')
    lines.append("")

    # --- File read: allow broadly, then deny specific paths ---
    # Programs (bash, python, etc.) need to read files throughout the
    # filesystem on startup — dynamic linker, shared libs, locale data,
    # etc. Rather than trying to enumerate every needed path (which is
    # fragile), we allow read access to the entire filesystem and then
    # deny specific sensitive paths. This matches how srt does it.
    # The real security boundary is on WRITES, not reads.
    lines.append(";; File read access (broad allow, then selective deny)")
    lines.append('(allow file-read* (subpath "/"))')
    lines.append("")

    # Deny read for hidden paths (AFTER the broad allow, so these win)
    if deny_read_paths:
        lines.append(";; Deny read access")
        for path in deny_read_paths:
            lines.append(_subpath_rule("file-read*", path, allow=False))
        lines.append("")

    # --- File write: allow writable paths ---
    lines.append(";; File write access")
    for path in writable_paths:
        lines.append(_subpath_rule("file-write*", path, allow=True))
    lines.append("")

    # --- Deny write: mandatory denies AFTER allows (later rules win) ---
    lines.append(";; Mandatory write denies (override allows)")
    for path in deny_write_paths:
        lines.append(_subpath_rule("file-write*", path, allow=False))
        # Also block unlink/rename to prevent mv-based bypass
        lines.append(_subpath_rule("file-write-unlink", path, allow=False))
    lines.append("")

    # --- Network ---
    lines.append(";; Network access")
    if allow_network:
        lines.append("(allow network*)")
    else:
        lines.append("(deny network*)")
    lines.append("")

    return "\n".join(lines)
