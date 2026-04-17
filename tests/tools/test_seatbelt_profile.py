"""Tests for Seatbelt SBPL profile generation.

Tests cover:
- Basic profile structure (version, deny default, process ops)
- Writable path rules
- Mandatory deny ordering (deny-write AFTER allow-write)
- Deny write paths (file-write* and file-write-unlink)
- Readable paths (explicit + implicit from writable)
- Deny read paths
- Network deny / allow
- Device I/O rules
- Path escaping for SBPL string literals
- Empty path lists
"""

from pathlib import Path

import pytest

from agentic_cli.tools.shell.os_sandbox.seatbelt_profile import (
    _escape_sbpl_path,
    generate_seatbelt_profile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_default(**overrides) -> str:
    """Generate a profile with sensible defaults, overriding as needed."""
    defaults = dict(
        writable_paths=[],
        deny_write_paths=[],
        readable_paths=[],
        deny_read_paths=[],
        allow_network=False,
    )
    defaults.update(overrides)
    return generate_seatbelt_profile(**defaults)


def _lines(profile: str) -> list[str]:
    """Split profile into non-empty, non-comment lines."""
    return [
        line for line in profile.splitlines()
        if line.strip() and not line.strip().startswith(";;")
    ]


# ---------------------------------------------------------------------------
# 1. Basic profile structure
# ---------------------------------------------------------------------------

class TestBasicStructure:
    """Verify the foundational profile skeleton."""

    def test_starts_with_version_and_deny_default(self):
        profile = _generate_default()
        lines = profile.splitlines()
        assert lines[0] == "(version 1)"
        assert lines[1] == "(deny default)"

    def test_contains_process_exec(self):
        profile = _generate_default()
        assert "(allow process-exec)" in profile

    def test_contains_process_fork(self):
        profile = _generate_default()
        assert "(allow process-fork)" in profile

    def test_contains_signal_self(self):
        profile = _generate_default()
        assert "(allow signal (target self))" in profile

    def test_contains_sysctl_read(self):
        profile = _generate_default()
        assert "(allow sysctl-read)" in profile


# ---------------------------------------------------------------------------
# 2. Writable paths
# ---------------------------------------------------------------------------

class TestWritablePaths:
    """Verify allow file-write* rules for writable paths."""

    def test_single_writable_path(self):
        profile = _generate_default(writable_paths=[Path("/tmp/sandbox")])
        assert '(allow file-write* (subpath "/tmp/sandbox"))' in profile

    def test_multiple_writable_paths(self):
        paths = [Path("/tmp/a"), Path("/home/user/work")]
        profile = _generate_default(writable_paths=paths)
        assert '(allow file-write* (subpath "/tmp/a"))' in profile
        assert '(allow file-write* (subpath "/home/user/work"))' in profile


# ---------------------------------------------------------------------------
# 3. Mandatory deny ordering
# ---------------------------------------------------------------------------

class TestDenyOrdering:
    """Deny-write rules MUST appear AFTER allow-write rules.

    Seatbelt uses "later rules win" semantics, so deny rules must come
    after allows to actually block access.
    """

    def test_deny_write_after_allow_write(self):
        writable = [Path("/tmp/work")]
        deny = [Path("/tmp/work/secrets")]
        profile = _generate_default(
            writable_paths=writable,
            deny_write_paths=deny,
        )
        allow_pos = profile.index('(allow file-write* (subpath "/tmp/work"))')
        deny_pos = profile.index('(deny file-write* (subpath "/tmp/work/secrets"))')
        assert deny_pos > allow_pos, (
            "deny-write must appear after allow-write for Seatbelt 'later wins' semantics"
        )

    def test_deny_read_after_allow_read(self):
        deny_read = [Path("/home/user/.ssh")]
        profile = _generate_default(deny_read_paths=deny_read)
        allow_pos = profile.index('(allow file-read* (subpath "/"))')
        deny_pos = profile.index('(deny file-read* (subpath "/home/user/.ssh"))')
        assert deny_pos > allow_pos, (
            "deny-read must appear after allow-read for Seatbelt 'later wins' semantics"
        )


# ---------------------------------------------------------------------------
# 4. Deny write paths
# ---------------------------------------------------------------------------

class TestDenyWritePaths:
    """Verify deny file-write* and file-write-unlink rules."""

    def test_deny_file_write_star(self):
        profile = _generate_default(deny_write_paths=[Path("/etc")])
        assert '(deny file-write* (subpath "/etc"))' in profile

    def test_deny_file_write_unlink(self):
        """Unlink deny prevents mv-based bypass of write protection."""
        profile = _generate_default(deny_write_paths=[Path("/etc")])
        assert '(deny file-write-unlink (subpath "/etc"))' in profile

    def test_multiple_deny_paths(self):
        paths = [Path("/etc"), Path("/usr/bin")]
        profile = _generate_default(deny_write_paths=paths)
        for p in paths:
            assert f'(deny file-write* (subpath "{p}"))' in profile
            assert f'(deny file-write-unlink (subpath "{p}"))' in profile


# ---------------------------------------------------------------------------
# 5. Readable paths
# ---------------------------------------------------------------------------

class TestReadablePaths:
    """Verify file-read* rules.

    The profile uses a broad (allow file-read* (subpath "/")) rule
    rather than enumerating individual system directories. This is
    necessary because programs (bash, python) need to read many files
    across the filesystem on startup. The real security boundary is on
    writes, not reads.
    """

    def test_broad_read_access(self):
        """All files are readable by default."""
        profile = _generate_default()
        assert '(allow file-read* (subpath "/"))' in profile

    def test_readable_paths_param_accepted(self):
        """readable_paths param is accepted (forward compat) but broad read covers it."""
        profile = _generate_default(readable_paths=[Path("/data/models")])
        # The broad read rule already covers this
        assert '(allow file-read* (subpath "/"))' in profile


# ---------------------------------------------------------------------------
# 6. Deny read paths
# ---------------------------------------------------------------------------

class TestDenyReadPaths:
    """Verify deny file-read* rules."""

    def test_deny_read_single_path(self):
        profile = _generate_default(deny_read_paths=[Path("/home/user/.ssh")])
        assert '(deny file-read* (subpath "/home/user/.ssh"))' in profile

    def test_deny_read_multiple_paths(self):
        paths = [Path("/home/user/.ssh"), Path("/home/user/.gnupg")]
        profile = _generate_default(deny_read_paths=paths)
        for p in paths:
            assert f'(deny file-read* (subpath "{p}"))' in profile

    def test_no_deny_read_section_when_empty(self):
        """When deny_read_paths is empty, the deny-read section is omitted."""
        profile = _generate_default(deny_read_paths=[])
        assert "(deny file-read*" not in profile


# ---------------------------------------------------------------------------
# 7. Network deny
# ---------------------------------------------------------------------------

class TestNetworkDeny:
    """Verify network rules when allow_network=False."""

    def test_network_denied_by_default(self):
        profile = _generate_default()
        assert "(deny network*)" in profile

    def test_network_denied_explicitly(self):
        profile = _generate_default(allow_network=False)
        assert "(deny network*)" in profile

    def test_no_allow_network_when_denied(self):
        profile = _generate_default(allow_network=False)
        assert "(allow network*)" not in profile


# ---------------------------------------------------------------------------
# 8. Network allow
# ---------------------------------------------------------------------------

class TestNetworkAllow:
    """Verify network rules when allow_network=True."""

    def test_network_allowed(self):
        profile = _generate_default(allow_network=True)
        assert "(allow network*)" in profile

    def test_no_deny_network_when_allowed(self):
        profile = _generate_default(allow_network=True)
        assert "(deny network*)" not in profile


# ---------------------------------------------------------------------------
# 9. Device I/O
# ---------------------------------------------------------------------------

class TestDeviceIO:
    """Verify /dev access rules."""

    def test_dev_readable(self):
        profile = _generate_default()
        assert '(allow file-read* (subpath "/dev"))' in profile

    def test_dev_null_writable(self):
        profile = _generate_default()
        assert '(allow file-write* (literal "/dev/null"))' in profile

    def test_dev_tty_writable(self):
        profile = _generate_default()
        assert '(allow file-write* (literal "/dev/tty"))' in profile

    def test_dev_tty_ioctl(self):
        profile = _generate_default()
        assert '(allow file-ioctl (literal "/dev/tty"))' in profile

    def test_dev_dfd_writable(self):
        profile = _generate_default()
        assert '(allow file-write* (literal "/dev/dfd"))' in profile


# ---------------------------------------------------------------------------
# 10. Path escaping
# ---------------------------------------------------------------------------

class TestPathEscaping:
    """Test _escape_sbpl_path with special characters."""

    def test_plain_path_unchanged(self):
        assert _escape_sbpl_path("/tmp/sandbox") == "/tmp/sandbox"

    def test_backslash_escaped(self):
        assert _escape_sbpl_path("/tmp/back\\slash") == "/tmp/back\\\\slash"

    def test_double_quote_escaped(self):
        assert _escape_sbpl_path('/tmp/has"quote') == '/tmp/has\\"quote'

    def test_both_backslash_and_quote(self):
        result = _escape_sbpl_path('/a\\b"c')
        assert result == '/a\\\\b\\"c'

    def test_multiple_backslashes(self):
        result = _escape_sbpl_path("/a\\\\b")
        assert result == "/a\\\\\\\\b"

    def test_escaping_used_in_profile(self):
        """Verify that path escaping is applied in generated profiles."""
        path_with_quote = Path('/tmp/has"quote')
        profile = _generate_default(writable_paths=[path_with_quote])
        assert '(allow file-write* (subpath "/tmp/has\\"quote"))' in profile


# ---------------------------------------------------------------------------
# 11. Empty paths lists
# ---------------------------------------------------------------------------

class TestEmptyPaths:
    """Verify profile is valid when path lists are empty."""

    def test_all_empty(self):
        profile = _generate_default()
        # Should still have header + process ops + network
        assert "(version 1)" in profile
        assert "(deny default)" in profile
        assert "(deny network*)" in profile

    def test_empty_writable_paths_still_has_write_section_comment(self):
        """The write section exists but has no subpath rules."""
        profile = _generate_default(writable_paths=[])
        # The section comment is present
        assert ";; File write access" in profile
        # No allow file-write* subpath rules (device rules exist but are literal)
        active = _lines(profile)
        write_subpath_rules = [
            l for l in active
            if l.startswith("(allow file-write* (subpath")
        ]
        # Only the /dev subpath rule should exist from device I/O
        # Actually, device I/O uses (subpath "/dev") for read only,
        # and literal for /dev/null etc., so no subpath write rules.
        assert len(write_subpath_rules) == 0

    def test_empty_deny_write_still_has_section_comment(self):
        profile = _generate_default(deny_write_paths=[])
        assert ";; Mandatory write denies (override allows)" in profile

    def test_empty_readable_paths_still_has_broad_read(self):
        """Even with no user-specified readable paths, broad read is present."""
        profile = _generate_default(readable_paths=[])
        assert '(allow file-read* (subpath "/"))' in profile

    def test_profile_is_nonempty_string(self):
        profile = _generate_default()
        assert isinstance(profile, str)
        assert len(profile) > 100
