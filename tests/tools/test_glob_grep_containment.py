"""P0-4: glob/grep patterns must not escape the permission-authorized root.

The permission engine authorizes only the ``path`` argument. A ``pattern`` /
``file_pattern`` of ``../*`` or an absolute path would let the tool read
outside the granted directory. These tests pin the containment behavior.
"""
from __future__ import annotations

import json
import subprocess

import agentic_cli.tools.glob_tool as glob_mod
import agentic_cli.tools.grep_tool as grep_mod
from agentic_cli.tools.glob_tool import glob
from agentic_cli.tools.grep_tool import grep


def test_glob_rejects_parent_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "inside.txt").write_text("x")
    (tmp_path / "secret.txt").write_text("SECRET")

    r = glob(pattern="../*", path=str(root))

    assert r["success"] is False
    assert "secret" not in str(r).lower()


def test_glob_rejects_absolute_pattern(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    r = glob(pattern="/etc/*", path=str(root))
    assert r["success"] is False


def test_glob_normal_pattern_still_works(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.py").write_text("x")
    (root / "b.txt").write_text("y")
    r = glob(pattern="*.py", path=str(root))
    assert r["success"] is True
    assert r["files"] == ["a.py"]


def test_glob_recursive_pattern_still_works(tmp_path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "deep.py").write_text("x")
    r = glob(pattern="**/*.py", path=str(root))
    assert r["success"] is True
    assert "sub/deep.py" in r["files"]


def test_glob_skips_symlink_escaping_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "passwd").write_text("SECRET")
    (root / "link").symlink_to(outside, target_is_directory=True)

    r = glob(pattern="link/*", path=str(root))

    # Pattern itself is legal, but the symlinked result resolves outside root.
    assert r["success"] is True
    assert all("passwd" not in str(f) for f in r["files"])


def test_grep_rejects_parent_escape_file_pattern(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("needle")
    (tmp_path / "secret.txt").write_text("needle SECRET")
    # Force the Python fallback — that's the path that follows ../ literally.
    monkeypatch.setattr(grep_mod, "_ripgrep_available", lambda: False)

    r = grep(pattern="needle", path=str(root), file_pattern="../*")

    assert r["success"] is False


def test_grep_python_skips_symlink_escaping_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("needle SECRET")
    (root / "link.txt").symlink_to(outside / "secret.txt")
    (root / "real.txt").write_text("needle here")
    monkeypatch.setattr(grep_mod, "_ripgrep_available", lambda: False)

    r = grep(pattern="needle", path=str(root))

    assert r["success"] is True
    files = {m["file"] for m in r["matches"]}
    # link.txt resolves to outside/secret.txt (outside root) → must be skipped,
    # even though its own path name doesn't contain "secret".
    assert not any("link.txt" in f for f in files)
    assert any("real.txt" in f for f in files)


def test_grep_ripgrep_filters_outside_root(tmp_path, monkeypatch):
    """The ripgrep path (used when rg is installed) must apply the same
    containment as the Python fallback — a followed symlink that rg reports
    with an outside-root path must be dropped."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "real.txt").write_text("needle here")
    monkeypatch.setattr(grep_mod, "_ripgrep_available", lambda: True)

    outside = tmp_path / "outside" / "secret.txt"
    inside = root / "real.txt"

    def fake_run(cmd, **kwargs):
        lines = [
            json.dumps({"type": "match", "data": {
                "path": {"text": str(outside)}, "line_number": 1,
                "lines": {"text": "needle SECRET\n"}}}),
            json.dumps({"type": "match", "data": {
                "path": {"text": str(inside)}, "line_number": 1,
                "lines": {"text": "needle here\n"}}}),
        ]
        return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(lines), stderr="")

    monkeypatch.setattr(grep_mod.subprocess, "run", fake_run)
    r = grep(pattern="needle", path=str(root))
    files = {m["file"] for m in r["matches"]}
    assert any("real.txt" in f for f in files)
    assert not any("secret.txt" in f for f in files)


def test_grep_ripgrep_scrubs_config_path_env(tmp_path, monkeypatch):
    """A host RIPGREP_CONFIG_PATH could inject --follow (defeating containment);
    it must not be inherited by the rg subprocess."""
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", "/home/user/.rgrc")
    monkeypatch.setattr(grep_mod, "_ripgrep_available", lambda: True)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grep_mod.subprocess, "run", fake_run)
    grep(pattern="x", path=str(root))
    assert captured["env"] is not None
    assert "RIPGREP_CONFIG_PATH" not in captured["env"]


def test_glob_excludes_hidden_ancestor(tmp_path):
    """include_hidden=False must drop results with a hidden ANCESTOR, not just
    a hidden basename (e.g. .hidden/secret.txt via **/*)."""
    root = tmp_path / "root"
    (root / ".hidden").mkdir(parents=True)
    (root / ".hidden" / "secret.txt").write_text("s")
    (root / "visible.txt").write_text("v")
    r = glob(pattern="**/*", path=str(root), include_hidden=False)
    assert r["success"] is True
    assert all(".hidden" not in f for f in r["files"])
    assert any("visible.txt" in f for f in r["files"])


def test_glob_caps_scanned_matches(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    for i in range(6):
        (root / f"f{i}.txt").write_text("x")
    monkeypatch.setattr(glob_mod, "_MAX_SCAN", 3)
    r = glob(pattern="*", path=str(root), max_results=100)
    assert r["success"] is True
    assert len(r["files"]) == 3   # exactly the ceiling (6 files, all pass filters)
    assert r["truncated"] is True


def test_grep_python_skips_oversized_files(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (root / "small.txt").write_text("needle here")
    (root / "big.txt").write_text("needle " + ("x" * 1000))
    monkeypatch.setattr(grep_mod, "_ripgrep_available", lambda: False)
    monkeypatch.setattr(grep_mod, "_MAX_FILE_BYTES", 100)
    r = grep(pattern="needle", path=str(root))
    files = {m["file"] for m in r["matches"]}
    assert any("small.txt" in f for f in files)
    assert not any("big.txt" in f for f in files)   # oversized file skipped


def test_grep_python_reports_truncated_when_file_cap_hit(tmp_path, monkeypatch):
    """Hitting the _MAX_FILES scan ceiling must be reported as truncated, not
    silently dropped (else a partial result claims to be complete)."""
    root = tmp_path / "root"
    root.mkdir()
    for i in range(3):
        (root / f"f{i}.txt").write_text("needle")
    monkeypatch.setattr(grep_mod, "_ripgrep_available", lambda: False)
    monkeypatch.setattr(grep_mod, "_MAX_FILES", 2)  # fewer than the 3 files
    r = grep(pattern="needle", path=str(root))
    assert r["truncated"] is True


def test_grep_python_directories_do_not_consume_file_budget(tmp_path, monkeypatch):
    """Directories must not count against _MAX_FILES — otherwise dirs before the
    files exhaust the budget and matches are silently missed."""
    root = tmp_path / "root"
    root.mkdir()
    for i in range(20):
        (root / f"dir{i}").mkdir()
    (root / "a.txt").write_text("needle")
    (root / "b.txt").write_text("needle")
    monkeypatch.setattr(grep_mod, "_ripgrep_available", lambda: False)
    monkeypatch.setattr(grep_mod, "_MAX_FILES", 2)  # exactly the 2 real files
    r = grep(pattern="needle", path=str(root))
    files = {m["file"] for m in r["matches"]}
    assert any("a.txt" in f for f in files)
    assert any("b.txt" in f for f in files)
