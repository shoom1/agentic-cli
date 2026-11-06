"""P0-4: glob/grep patterns must not escape the permission-authorized root.

The permission engine authorizes only the ``path`` argument. A ``pattern`` /
``file_pattern`` of ``../*`` or an absolute path would let the tool read
outside the granted directory. These tests pin the containment behavior.
"""
from __future__ import annotations

import json
import subprocess

import pytest

import agentic_cli.tools.glob_tool as glob_mod
import agentic_cli.tools.grep_tool as grep_mod
from agentic_cli.tools.glob_tool import glob
from agentic_cli.tools.grep_tool import grep


class _FakeProc:
    """Stand-in for a ripgrep subprocess (grep now uses Popen + a temp file)."""
    pid = 4321
    returncode = 0

    def wait(self, timeout=None):
        return 0


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

    def fake_popen(cmd, **kwargs):
        out = kwargs["stdout"]
        lines = "\n".join([
            json.dumps({"type": "match", "data": {
                "path": {"text": str(outside)}, "line_number": 1,
                "lines": {"text": "needle SECRET\n"}}}),
            json.dumps({"type": "match", "data": {
                "path": {"text": str(inside)}, "line_number": 1,
                "lines": {"text": "needle here\n"}}}),
        ]) + "\n"
        out.write(lines.encode())
        return _FakeProc()

    monkeypatch.setattr(grep_mod.subprocess, "Popen", fake_popen)
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

    def fake_popen(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakeProc()

    monkeypatch.setattr(grep_mod.subprocess, "Popen", fake_popen)
    grep(pattern="x", path=str(root))
    assert captured["env"] is not None
    assert "RIPGREP_CONFIG_PATH" not in captured["env"]


def test_grep_ripgrep_output_bounded(tmp_path, monkeypatch):
    """rg output beyond _MAX_RG_OUTPUT_BYTES is not read into memory whole; the
    result is flagged truncated."""
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "a.txt"
    inside.write_text("needle")
    monkeypatch.setattr(grep_mod, "_ripgrep_available", lambda: True)
    monkeypatch.setattr(grep_mod, "_MAX_RG_OUTPUT_BYTES", 200)

    def fake_popen(cmd, **kwargs):
        out = kwargs["stdout"]
        line = (json.dumps({"type": "match", "data": {
            "path": {"text": str(inside)}, "line_number": 1,
            "lines": {"text": "needle\n"}}}) + "\n").encode()
        for _ in range(50):  # well over the 200-byte cap
            out.write(line)
        return _FakeProc()

    monkeypatch.setattr(grep_mod.subprocess, "Popen", fake_popen)
    r = grep(pattern="needle", path=str(root))
    assert r["truncated"] is True


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


# ---------------------------------------------------------------------------
# Hidden files below the search root are not searched
# ---------------------------------------------------------------------------
#
# The engine judges only grep's ``path``, and the project directory is
# readable without a prompt, so a search of the project read its ``.env`` (and
# ``.git``, and the app's ``.{app}``) even though ``read_file(".env")`` is
# denied. ripgrep skips hidden files by default; the Python search now does
# too, and both keep doing so when a file pattern is given.

_HIDDEN = {".env", ".hidden/x", "sub/.env", "sub/.deep/y"}


def _hidden_tree(tmp_path):
    root = tmp_path / "root"
    for rel in _HIDDEN | {"a.txt", "sub/b.txt"}:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("needle SECRET\n" if rel in _HIDDEN else "needle here\n")
    # Non-hidden names for hidden files (a cloned repository can ship them:
    # git stores symlinks). ripgrep does not follow them; the Python search
    # judges where they lead.
    (root / "envlink").symlink_to(".env")
    (root / "sub" / "envlink2").symlink_to(".env")
    return root


def _found(result, root):
    from pathlib import Path

    return {Path(m["file"]).resolve().relative_to(root.resolve()).as_posix() for m in result["matches"]}



@pytest.mark.parametrize("file_pattern", [None, "*", "**/*", ".env", ".*", "x", "*.txt"])
def test_grep_python_skips_hidden_files_below_the_root(tmp_path, monkeypatch, file_pattern):
    root = _hidden_tree(tmp_path)
    monkeypatch.setattr(grep_mod, "_ripgrep_available", lambda: False)

    for mode in ("content", "files", "count"):
        r = grep(pattern="needle", path=str(root), file_pattern=file_pattern, output_mode=mode)
        assert r["success"] is True
        assert not _found(r, root) & _HIDDEN, (file_pattern, mode)
    if file_pattern in (None, "*", "**/*", "*.txt"):
        assert _found(r, root) == {"a.txt", "sub/b.txt"}
        assert r["files_searched"] == 2


def test_grep_python_still_searches_a_symlink_to_an_ordinary_file(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("needle\n")
    (root / "link.txt").symlink_to("a.txt")
    monkeypatch.setattr(grep_mod, "_ripgrep_available", lambda: False)

    files = {m["file"] for m in grep(pattern="needle", path=str(root))["matches"]}

    assert {f.rsplit("/", 1)[-1] for f in files} == {"a.txt", "link.txt"}


def test_grep_python_searches_a_hidden_root_given_explicitly(tmp_path, monkeypatch):
    """A hidden file or directory named as ``path`` is what the engine judged
    (and denies, for ``.env``), so it is searched."""
    root = _hidden_tree(tmp_path)
    nested = tmp_path / ".outer" / "project"
    (nested / "src").mkdir(parents=True)
    (nested / "src" / "c.txt").write_text("needle\n")
    monkeypatch.setattr(grep_mod, "_ripgrep_available", lambda: False)

    assert _found(grep(pattern="needle", path=str(root / ".env")), root) == {".env"}
    assert _found(grep(pattern="needle", path=str(root / ".hidden")), root) == {".hidden/x"}
    # A hidden directory above the root is no part of what is below it.
    assert _found(grep(pattern="needle", path=str(nested)), nested) == {"src/c.txt"}


@pytest.mark.parametrize("file_pattern", ["*", ".env", "**/*"])
def test_grep_ripgrep_keeps_hidden_files_out_with_a_file_pattern(tmp_path, monkeypatch, file_pattern):
    """A ``--glob`` overrides ripgrep's hidden-file filter (ripgrep 14.1:
    ``rg -g '*'`` searches ``.env`` and ``.hidden/x``), so grep excludes them
    with a later glob, which wins."""
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(grep_mod, "_ripgrep_available", lambda: True)
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(grep_mod.subprocess, "Popen", fake_popen)
    grep(pattern="needle", path=str(root), file_pattern=file_pattern)

    cmd = captured["cmd"]
    globs = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--glob"]
    assert globs == [file_pattern, "!.*"]
    assert cmd.index("--") > max(i for i, arg in enumerate(cmd) if arg == "--glob")
