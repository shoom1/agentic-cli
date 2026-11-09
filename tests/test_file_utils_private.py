"""``make_private_file``: a file created owner-only before a library opens it.

A session database holds every word of a conversation. Created by its
library (SQLAlchemy, aiosqlite), it got the process's umask: 0644 under 022,
in a 0755 directory, where the raw LLM log of the same words is 0600 in a
0700 one.
"""

from __future__ import annotations

import os
import stat

import pytest

from agentic_cli.file_utils import make_private_file


def _mode(path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


@pytest.fixture(autouse=True)
def umask_022():
    old = os.umask(0o022)
    yield
    os.umask(old)


def test_a_new_file_is_owner_only_in_a_new_owner_only_directory(tmp_path):
    path = tmp_path / "ws" / "sessions" / "sessions.db"

    make_private_file(path)

    assert path.read_bytes() == b""
    assert _mode(path) == 0o600
    assert _mode(path.parent) == 0o700


def test_a_directory_that_exists_is_left_as_it_is(tmp_path):
    """It may be the user's own (``sqlite_uri`` can point anywhere)."""
    folder = tmp_path / "mine"
    folder.mkdir(mode=0o755)

    make_private_file(folder / "app.db")

    assert _mode(folder) == 0o755
    assert _mode(folder / "app.db") == 0o600


def test_an_existing_file_is_tightened_and_keeps_its_content(tmp_path):
    """One an older version created 0644."""
    path = tmp_path / "sessions.db"
    path.write_bytes(b"SQLite format 3\x00")
    path.chmod(0o644)

    make_private_file(path)

    assert _mode(path) == 0o600
    assert path.read_bytes() == b"SQLite format 3\x00"


def test_an_owner_only_file_is_left_alone(tmp_path):
    path = tmp_path / "sessions.db"
    path.write_bytes(b"x")
    path.chmod(0o400)

    make_private_file(path)

    assert _mode(path) == 0o400


def test_a_symlink_and_what_it_points_at_are_left_alone(tmp_path):
    target = tmp_path / "target.db"
    target.write_bytes(b"x")
    target.chmod(0o644)
    link = tmp_path / "sessions.db"
    link.symlink_to(target)

    make_private_file(link)

    assert link.is_symlink()
    assert _mode(target) == 0o644


def test_without_fchmod_an_existing_file_is_left_as_it_is(tmp_path, monkeypatch):
    """Python 3.12 on Windows has no ``os.fchmod`` (3.13 added it), and there
    the permission bits mean nothing: a second start must still start."""
    path = tmp_path / "sessions.db"
    path.write_bytes(b"x")
    path.chmod(0o644)
    monkeypatch.delattr(os, "fchmod")

    make_private_file(path)

    assert _mode(path) == 0o644
    assert path.read_bytes() == b"x"


def test_a_file_this_process_may_not_change_is_left_as_it_is(tmp_path, monkeypatch):
    """Another user's file, or one on a read-only volume: no error."""
    path = tmp_path / "sessions.db"
    path.write_bytes(b"x")
    path.chmod(0o644)

    def refused(fd: int, mode: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "fchmod", refused)

    make_private_file(path)

    assert _mode(path) == 0o644
