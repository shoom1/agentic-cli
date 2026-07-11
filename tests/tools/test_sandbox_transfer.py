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


from types import SimpleNamespace

from agentic_cli.tools.sandbox.backends.jupyter_docker import JupyterDockerBackend


def _backend(shared: Path):
    return JupyterDockerBackend(
        settings=SimpleNamespace(sandbox_outputs_dir=str(shared), workspace_dir=str(shared))
    )


def test_collect_outputs_copies_regular_files(tmp_path):
    shared = tmp_path / "shared"
    wd = tmp_path / "wd"; (wd / "outputs").mkdir(parents=True)
    (wd / "outputs" / "plot.png").write_bytes(b"PNG")
    got = _backend(shared)._collect_outputs(wd)
    assert (shared / "plot.png").read_bytes() == b"PNG"
    assert any("plot.png" in g for g in got)


def test_collect_outputs_skips_symlink_to_host_secret(tmp_path):
    shared = tmp_path / "shared"
    wd = tmp_path / "wd"; (wd / "outputs").mkdir(parents=True)
    secret = tmp_path / "aws_credentials"; secret.write_bytes(b"AKIA-SECRET")
    (wd / "outputs" / "result.txt").symlink_to(secret)     # kernel exfil attempt
    got = _backend(shared)._collect_outputs(wd)
    assert got == []
    assert not (shared / "result.txt").exists()            # secret not copied out


def test_collect_outputs_skips_when_outputs_is_symlink(tmp_path):
    shared = tmp_path / "shared"
    wd = tmp_path / "wd"; wd.mkdir()
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir()
    (elsewhere / "x.txt").write_text("data")
    (wd / "outputs").symlink_to(elsewhere)                 # 'outputs' itself a symlink
    assert _backend(shared)._collect_outputs(wd) == []


def test_outputs_dir_is_0700(tmp_path):
    shared = tmp_path / "shared"
    base = _backend(shared)._outputs_dir()
    assert (base.stat().st_mode & 0o777) == 0o700
