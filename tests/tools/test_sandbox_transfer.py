import os
from pathlib import Path

import pytest

from agentic_cli.tools.sandbox.manager import stage_inputs


def test_stage_inputs_copies_regular_file(tmp_path):
    session = tmp_path / "session"; session.mkdir()
    src = tmp_path / "data.csv"; src.write_bytes(b"col1,col2")
    stage_inputs(session, [str(src)])
    staged = session / "inputs" / "data.csv"
    assert staged.read_bytes() == b"col1,col2" and not staged.is_symlink()


def test_stage_inputs_rejects_symlink_source(tmp_path):
    session = tmp_path / "session"; session.mkdir()
    secret = tmp_path / "secret.txt"; secret.write_bytes(b"TOP SECRET")
    link = tmp_path / "data.csv"; link.symlink_to(secret)
    with pytest.raises(ValueError):
        stage_inputs(session, [str(link)])


def test_stage_inputs_does_not_follow_dest_symlink(tmp_path):
    # Simulate a kernel that pre-planted inputs/data.csv -> a host file.
    session = tmp_path / "session"; session.mkdir()
    inputs_dir = session / "inputs"; inputs_dir.mkdir()
    outside = tmp_path / "host_secret"; outside.write_bytes(b"ORIGINAL")
    (inputs_dir / "data.csv").symlink_to(outside)
    src = tmp_path / "data.csv"; src.write_bytes(b"INPUT")
    stage_inputs(session, [str(src)])
    assert outside.read_bytes() == b"ORIGINAL"                  # not written through
    assert (inputs_dir / "data.csv").read_bytes() == b"INPUT"   # replaced by a real file
    assert not (inputs_dir / "data.csv").is_symlink()


def test_stage_inputs_duplicate_basename_errors(tmp_path):
    session = tmp_path / "session"; session.mkdir()
    a = tmp_path / "a" / "x.csv"; a.parent.mkdir(); a.write_text("1")
    b = tmp_path / "b" / "x.csv"; b.parent.mkdir(); b.write_text("2")
    with pytest.raises(ValueError):
        stage_inputs(session, [str(a), str(b)])
