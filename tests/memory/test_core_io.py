"""Atomic writes for the memory package's own files."""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from agentic_cli.memory._core import io as core_io
from agentic_cli.memory._core.io import atomic_write_json, atomic_write_text
from agentic_cli.memory._core.log import get_logger

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="symlinks and FIFOs are POSIX here")


def test_json_round_trips_and_the_file_is_private(tmp_path):
    path = tmp_path / "nested" / "data.json"

    atomic_write_json(path, {"a": [1, 2]})

    assert json.loads(path.read_text()) == {"a": [1, 2]}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_text_replaces_the_file_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "notes.md"

    atomic_write_text(path, "old")
    atomic_write_text(path, "new — ünïcode")

    assert path.read_text(encoding="utf-8") == "new — ünïcode"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.md"]


def test_a_failed_write_keeps_the_old_content(tmp_path, monkeypatch):
    path = tmp_path / "notes.md"
    atomic_write_text(path, "old")

    def _fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _fail)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(path, "new")

    assert path.read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.md"]


def test_get_logger_logs_without_host_configuration():
    get_logger("agentic_cli.memory.test").debug("memory_log_smoke", ok=True)


# -- the package's files are opened only when they are regular files ----------
# A project can ship a knowledge base, so a name in its folder may be a symlink
# to a file elsewhere, a FIFO (opening one blocks until a writer appears) or a
# link to a device such as /dev/zero (reading it never ends).


def test_a_regular_file_is_read(tmp_path):
    (tmp_path / "data.json").write_bytes(b"{}")

    assert core_io.read_regular_file(tmp_path / "data.json") == b"{}"
    assert core_io.read_if_regular(tmp_path / "data.json") == b"{}"
    assert core_io.is_regular_file(tmp_path / "data.json")


def test_a_missing_file_is_none(tmp_path):
    with pytest.raises(FileNotFoundError):
        core_io.read_regular_file(tmp_path / "missing.json")
    assert core_io.read_if_regular(tmp_path / "missing.json") is None
    assert not core_io.is_regular_file(tmp_path / "missing.json")


@posix_only
def test_a_symlink_is_not_followed(tmp_path):
    (tmp_path / "elsewhere").write_bytes(b"secret")
    (tmp_path / "data.json").symlink_to(tmp_path / "elsewhere")

    with pytest.raises(core_io.NotARegularFile):
        core_io.read_regular_file(tmp_path / "data.json")
    assert core_io.read_if_regular(tmp_path / "data.json") is None
    assert not core_io.is_regular_file(tmp_path / "data.json")


@posix_only
def test_a_fifo_is_refused_without_waiting_for_a_writer(tmp_path):
    os.mkfifo(tmp_path / "data.json")

    with pytest.raises(core_io.NotARegularFile):
        core_io.read_regular_file(tmp_path / "data.json")


def test_a_folder_is_not_a_file(tmp_path):
    (tmp_path / "data.json").mkdir()

    with pytest.raises(core_io.NotARegularFile):
        core_io.read_regular_file(tmp_path / "data.json")


def test_a_file_over_the_limit_is_refused(tmp_path):
    (tmp_path / "data.json").write_bytes(b"x" * 11)

    assert core_io.read_regular_file(tmp_path / "data.json", limit=11) == b"x" * 11
    with pytest.raises(OSError):
        core_io.read_regular_file(tmp_path / "data.json", limit=10)


def test_appending_creates_a_private_file_and_appends(tmp_path):
    path = tmp_path / "log.md"

    core_io.append_text(path, "one\n")
    core_io.append_text(path, "two ü\n")

    assert path.read_text(encoding="utf-8") == "one\ntwo ü\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@posix_only
def test_appending_never_writes_through_a_symlink(tmp_path):
    (tmp_path / "rc").write_bytes(b"# rc\n")
    (tmp_path / "log.md").symlink_to(tmp_path / "rc")

    with pytest.raises(core_io.NotARegularFile):
        core_io.append_text(tmp_path / "log.md", "line\n")
    assert (tmp_path / "rc").read_bytes() == b"# rc\n"


@posix_only
def test_appending_never_writes_through_a_dangling_symlink(tmp_path):
    (tmp_path / "log.md").symlink_to(tmp_path / "created")

    with pytest.raises(core_io.NotARegularFile):
        core_io.append_text(tmp_path / "log.md", "line\n")
    assert not (tmp_path / "created").exists()


@posix_only
def test_appending_to_a_fifo_is_refused_without_waiting_for_a_reader(tmp_path):
    os.mkfifo(tmp_path / "log.md")

    with pytest.raises(core_io.NotARegularFile):
        core_io.append_text(tmp_path / "log.md", "line\n")


@posix_only
def test_an_atomic_write_replaces_a_symlink_instead_of_following_it(tmp_path):
    (tmp_path / "elsewhere").write_bytes(b"theirs")
    (tmp_path / "data.bin").symlink_to(tmp_path / "elsewhere")

    core_io.atomic_write_bytes(tmp_path / "data.bin", b"ours")

    assert (tmp_path / "elsewhere").read_bytes() == b"theirs"
    assert not (tmp_path / "data.bin").is_symlink()
    assert (tmp_path / "data.bin").read_bytes() == b"ours"
    assert stat.S_IMODE((tmp_path / "data.bin").stat().st_mode) == 0o600


def test_creating_a_private_file_leaves_an_existing_one_as_it_is(tmp_path):
    core_io.create_private_file(tmp_path / "kb.sqlite")
    assert stat.S_IMODE((tmp_path / "kb.sqlite").stat().st_mode) == 0o600
    assert (tmp_path / "kb.sqlite").read_bytes() == b""

    (tmp_path / "kb.sqlite").write_bytes(b"theirs")
    core_io.create_private_file(tmp_path / "kb.sqlite")
    assert (tmp_path / "kb.sqlite").read_bytes() == b"theirs"


@posix_only
def test_creating_a_private_file_never_follows_a_dangling_symlink(tmp_path):
    (tmp_path / "kb.sqlite").symlink_to(tmp_path / "created")

    core_io.create_private_file(tmp_path / "kb.sqlite")

    assert not (tmp_path / "created").exists()
