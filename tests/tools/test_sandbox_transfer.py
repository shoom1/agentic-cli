from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_cli.tools.sandbox import manager
from agentic_cli.tools.sandbox.backends import jupyter_docker
from agentic_cli.tools.sandbox.manager import stage_inputs
from agentic_cli.tools.sandbox.backends.jupyter_docker import JupyterDockerBackend


def _backend(shared: Path):
    return JupyterDockerBackend(
        settings=SimpleNamespace(sandbox_outputs_dir=str(shared), workspace_dir=str(shared))
    )


def _swap_for_link(directory: Path, target: Path) -> None:
    """What a kernel thread can do between two host steps: park the real
    directory and put a symlink to a host directory in its place."""
    if not directory.is_symlink():
        directory.rename(directory.with_name(f".{directory.name}.parked"))
        directory.symlink_to(target, target_is_directory=True)


def _swap_before_each_copy(monkeypatch, module, directory: Path, target: Path) -> None:
    """Swap ``directory`` for a link to ``target`` just before every per-entry
    copy, so the race the kernel would have to win is won every time."""
    real_copy = module.copy_regular_file_no_follow

    def swap_then_copy(*args, **kwargs):
        _swap_for_link(directory, target)
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(module, "copy_regular_file_no_follow", swap_then_copy)


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


def test_stage_inputs_rejects_symlinked_inputs_dir(tmp_path):
    # Kernel plants session/inputs as a symlink to a host dir -> must be rejected,
    # not written through (O_NOFOLLOW guards only the final component).
    session = tmp_path / "session"; session.mkdir()
    outside = tmp_path / "host_dir"; outside.mkdir()
    (session / "inputs").symlink_to(outside, target_is_directory=True)
    src = tmp_path / "data.csv"; src.write_bytes(b"INPUT")
    with pytest.raises(ValueError):
        stage_inputs(session, [str(src)])
    assert list(outside.iterdir()) == []   # nothing written into the host dir


def test_collect_outputs_never_raises_on_outputs_dir_error(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    wd = tmp_path / "wd"; (wd / "outputs").mkdir(parents=True)
    (wd / "outputs" / "a.txt").write_text("x")
    b = _backend(shared)
    def boom():
        raise OSError("boom")
    monkeypatch.setattr(b, "_outputs_dir", boom)
    assert b._collect_outputs(wd) == []   # best-effort: never raises


def test_collect_outputs_reads_nothing_outside_when_outputs_is_swapped(tmp_path, monkeypatch):
    # The kernel swaps outputs/ for a link to a host directory after the host
    # looked at it: every file is still read from the session's own directory.
    shared = tmp_path / "shared"
    wd = tmp_path / "wd"; outs = wd / "outputs"; outs.mkdir(parents=True)
    outside = tmp_path / "host_secrets"; outside.mkdir()
    for name in ("a.csv", "id_rsa"):
        (outs / name).write_bytes(b"SESSION " + name.encode())
        (outside / name).write_bytes(b"OUTSIDE " + name.encode())
    _swap_before_each_copy(monkeypatch, jupyter_docker, outs, outside)
    got = _backend(shared)._collect_outputs(wd)
    assert sorted(Path(g).read_bytes() for g in got) == [b"SESSION a.csv", b"SESSION id_rsa"]
    assert sorted(p.read_bytes() for p in shared.iterdir()) == [b"SESSION a.csv", b"SESSION id_rsa"]


def test_stage_inputs_writes_nothing_outside_when_inputs_is_swapped(tmp_path, monkeypatch):
    # The kernel swaps inputs/ for a link to a host directory after the host
    # looked at it: every input still lands in the session's own directory.
    session = tmp_path / "session"; (session / "inputs").mkdir(parents=True)
    outside = tmp_path / "LaunchAgents"; outside.mkdir()
    (outside / "keep.plist").write_bytes(b"ORIGINAL")
    big = tmp_path / "big.csv"; big.write_bytes(b"BIG")
    payload = tmp_path / "agent.plist"; payload.write_bytes(b"PAYLOAD")
    _swap_before_each_copy(monkeypatch, manager, session / "inputs", outside)
    stage_inputs(session, [str(big), str(payload)])
    assert {p.name: p.read_bytes() for p in outside.iterdir()} == {"keep.plist": b"ORIGINAL"}
    parked = session / ".inputs.parked"
    assert {p.name: p.read_bytes() for p in parked.iterdir()} == {
        "big.csv": b"BIG", "agent.plist": b"PAYLOAD",
    }


def test_stage_inputs_refuses_a_dangling_inputs_link(tmp_path):
    session = tmp_path / "session"; session.mkdir()
    (session / "inputs").symlink_to(tmp_path / "nowhere", target_is_directory=True)
    src = tmp_path / "data.csv"; src.write_bytes(b"INPUT")
    with pytest.raises(ValueError, match="not a real directory"):
        stage_inputs(session, [str(src)])
    assert not (tmp_path / "nowhere").exists()
