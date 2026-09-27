"""``write_file`` and ``edit_file`` leave a file's permissions as they were.

Atomic writes go through a temp file created by ``mkstemp``, which is always
0600, and that temp file replaces the original. So editing a 0755 script made
it non-executable and unreadable to anyone else, and every new file was 0600.
The file tools now keep an existing file's permission bits and give a new file
the process default (0666 minus the umask), as a plain ``open()`` would. The
framework's own state files stay private (0600).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from agentic_cli.file_utils import atomic_write_text
from agentic_cli.tools.file_write import edit_file, write_file


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def umask_022():
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


@pytest.mark.parametrize("mode", [0o755, 0o644, 0o640, 0o600])
def test_edit_file_keeps_the_mode(tmp_path, mode):
    script = tmp_path / "run.sh"
    script.write_text("echo old\n")
    script.chmod(mode)

    assert edit_file(str(script), "old", "new")["success"] is True

    assert script.read_text() == "echo new\n"
    assert _mode(script) == mode


@pytest.mark.parametrize("mode", [0o755, 0o644])
def test_write_file_over_an_existing_file_keeps_the_mode(tmp_path, mode):
    target = tmp_path / "notes.txt"
    target.write_text("before\n")
    target.chmod(mode)

    assert write_file(str(target), "after\n")["success"] is True

    assert _mode(target) == mode


def test_write_file_creates_a_file_with_the_default_mode(tmp_path, umask_022):
    target = tmp_path / "new.txt"

    assert write_file(str(target), "hello\n")["success"] is True

    assert _mode(target) == 0o644


def test_setuid_and_setgid_are_not_carried_over(tmp_path):
    """The kernel clears them when a non-root process writes a file; a rewrite
    must not keep them either."""
    target = tmp_path / "tool"
    target.write_text("x\n")
    target.chmod(0o755 | stat.S_ISUID | stat.S_ISGID)

    assert write_file(str(target), "y\n")["success"] is True

    assert _mode(target) == 0o755


def test_framework_state_files_stay_private(tmp_path, umask_022):
    state = tmp_path / "grants.json"

    atomic_write_text(state, "{}")

    assert _mode(state) == 0o600
