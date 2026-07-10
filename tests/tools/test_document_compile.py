"""Offline tests for compile_document — subprocess and engine lookup faked."""
from __future__ import annotations

import subprocess
from pathlib import Path

from agentic_cli.tools.document import compile as mod
from agentic_cli.tools.document import compile_document


def _fake_engine(monkeypatch, name="latexmk"):
    monkeypatch.setattr(mod, "_which", lambda n: f"/usr/bin/{n}" if n == name else None)


def test_no_engine_returns_structured_error(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_which", lambda n: None)
    tex = tmp_path / "r.tex"; tex.write_text("x")
    r = compile_document(str(tex))
    assert r["success"] is False
    assert "No LaTeX engine" in r["error"]
    assert r["engine"] is None


def test_missing_source_returns_error(tmp_path):
    r = compile_document(str(tmp_path / "nope.tex"))
    assert r["success"] is False and "not found" in r["error"]


def test_success_places_pdf_and_keeps_intermediates(monkeypatch, tmp_path):
    _fake_engine(monkeypatch)
    tex = tmp_path / "r.tex"
    tex.write_text("\\documentclass{article}\\begin{document}hi\\end{document}")

    def fake_run(argv, *, cwd, env, timeout):
        (Path(cwd) / "r.pdf").write_bytes(b"%PDF-1.5 fake")
        (Path(cwd) / "r.log").write_text("output written on r.pdf")
        (Path(cwd) / "r.aux").write_text("\\relax")
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)
    out = tmp_path / "deliver" / "report.pdf"
    r = compile_document(str(tex), output_pdf=str(out), assets_dir=str(tmp_path / "assets"))
    assert r["success"] is True
    assert r["pdf_path"] == str(out)
    assert out.is_file()                             # PDF promoted to delivery dir
    assert (tmp_path / "r.aux").is_file()            # intermediates stay in build dir
    assert not (out.parent / "r.aux").exists()       # not beside the delivered PDF


def test_failure_parses_errors(monkeypatch, tmp_path):
    _fake_engine(monkeypatch)
    tex = tmp_path / "r.tex"; tex.write_text("bad")

    def fake_run(argv, *, cwd, env, timeout):
        (Path(cwd) / "r.log").write_text("! Undefined control sequence.\nl.5 \\badcmd\n")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)
    r = compile_document(str(tex))
    assert r["success"] is False
    assert any("Undefined control sequence" in e for e in r["errors"])
    assert r["pdf_path"] is None


def test_argv_never_enables_shell_escape(monkeypatch, tmp_path):
    _fake_engine(monkeypatch)
    tex = tmp_path / "r.tex"; tex.write_text("x")
    captured = {}

    def fake_run(argv, *, cwd, env, timeout):
        captured["argv"] = argv; captured["env"] = env
        (Path(cwd) / "r.pdf").write_bytes(b"%PDF")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)
    compile_document(str(tex), assets_dir="/tmp/assets")
    assert "-shell-escape" not in captured["argv"]
    assert "-halt-on-error" in captured["argv"]
    assert captured["argv"][0] == "latexmk"
    assert "/tmp/assets" in captured["env"]["TEXINPUTS"]


def test_pdflatex_uses_no_shell_escape_flag(monkeypatch, tmp_path):
    _fake_engine(monkeypatch, name="pdflatex")
    tex = tmp_path / "r.tex"; tex.write_text("x")
    captured = {}

    def fake_run(argv, *, cwd, env, timeout):
        captured["argv"] = argv
        (Path(cwd) / "r.pdf").write_bytes(b"%PDF")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)
    compile_document(str(tex))
    assert captured["argv"][0] == "pdflatex"
    assert "-no-shell-escape" in captured["argv"]


def test_timeout_returns_structured_failure(monkeypatch, tmp_path):
    _fake_engine(monkeypatch)
    tex = tmp_path / "r.tex"; tex.write_text("x")

    def fake_run(argv, *, cwd, env, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(mod, "_run", fake_run)
    r = compile_document(str(tex), timeout_s=1)
    assert r["success"] is False and "timed out" in r["error"]


# --- Fix wave 1 tests ---

def test_run_raises_file_not_found_returns_structured_error(monkeypatch, tmp_path):
    """Finding 1: _run raising FileNotFoundError must not propagate; must return failure dict."""
    _fake_engine(monkeypatch)
    tex = tmp_path / "r.tex"; tex.write_text("x")

    def fake_run(argv, *, cwd, env, timeout):
        raise FileNotFoundError("latexmk: not found")

    monkeypatch.setattr(mod, "_run", fake_run)
    r = compile_document(str(tex))
    assert r["success"] is False
    assert r["error"] is not None and len(r["error"]) > 0
    assert r["engine"] == "latexmk"
    assert r["pdf_path"] is None
    assert "duration_ms" in r


def test_pdf_copy_oserror_returns_structured_error(monkeypatch, tmp_path):
    """Finding 2: OSError during PDF delivery must not propagate; must return failure dict."""
    _fake_engine(monkeypatch)
    tex = tmp_path / "r.tex"; tex.write_text("x")
    out = tmp_path / "deliver" / "report.pdf"

    def fake_run(argv, *, cwd, env, timeout):
        (Path(cwd) / "r.pdf").write_bytes(b"%PDF-1.5 fake")
        (Path(cwd) / "r.log").write_text("output written on r.pdf")
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)
    # Delivery is a no-follow atomic write; force the atomic rename to fail.
    monkeypatch.setattr(mod.os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("disk full")))

    r = compile_document(str(tex), output_pdf=str(out))
    assert r["success"] is False
    assert "deliver" in r["error"].lower() or str(out) in r["error"]
    assert r["pdf_path"] is not None   # PDF still exists in build dir
    assert "duration_ms" in r


def test_forced_unsupported_engine_rejected(monkeypatch, tmp_path):
    """Finding 3: forcing engine='xelatex' (not in _ENGINES) must be rejected → No LaTeX engine error."""
    monkeypatch.setattr(mod, "_which", lambda n: f"/usr/bin/{n}" if n == "xelatex" else None)
    tex = tmp_path / "r.tex"; tex.write_text("x")
    r = compile_document(str(tex), engine="xelatex")
    assert r["success"] is False
    assert "No LaTeX engine" in r["error"]


# --- Fix wave — final review (I1/I2/M1/M2/M3) ---

def test_run_group_kill_on_timeout(monkeypatch):
    """I1: _run must start a new session and kill the process group on timeout."""
    import subprocess as _subprocess

    popen_kwargs: dict = {}
    killpg_calls: list = []

    class FakePopen:
        pid = 42

        def __init__(self, argv, **kwargs):
            popen_kwargs.update(kwargs)

        def communicate(self, timeout=None):
            if timeout is not None:
                raise _subprocess.TimeoutExpired([], timeout)
            # reap call after kill
            return ("", "")

    monkeypatch.setattr(mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(mod.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(mod.os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))

    try:
        mod._run(["latexmk"], cwd="/tmp", env={}, timeout=1.0)
    except _subprocess.TimeoutExpired:
        pass  # expected

    assert popen_kwargs.get("start_new_session") is True, "Popen must use start_new_session=True"
    assert len(killpg_calls) >= 1, "os.killpg must be called on timeout"


# --- P0-3 hardening: env allowlist, no-follow delivery, capability scope ---

def test_env_is_allowlisted_not_full_environ(monkeypatch, tmp_path):
    """Env inheritance leak: the TeX process must not receive host secrets;
    PATH is preserved and assets_dir still reaches TEXINPUTS."""
    _fake_engine(monkeypatch)
    monkeypatch.setenv("MY_SECRET_TOKEN", "sk-must-not-leak")
    monkeypatch.setenv("PATH", "/custom/bin")
    monkeypatch.setenv("PERL5LIB", "/opt/perl/lib")  # latexmk (Perl) needs this
    tex = tmp_path / "r.tex"; tex.write_text("x")
    captured = {}

    def fake_run(argv, *, cwd, env, timeout):
        captured["env"] = env
        (Path(cwd) / "r.pdf").write_bytes(b"%PDF")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)
    compile_document(str(tex), assets_dir="/tmp/assets")
    assert "MY_SECRET_TOKEN" not in captured["env"]
    assert captured["env"]["PATH"] == "/custom/bin"
    assert captured["env"]["PERL5LIB"] == "/opt/perl/lib"
    assert "/tmp/assets" in captured["env"]["TEXINPUTS"]


def test_delivery_does_not_follow_output_symlink(monkeypatch, tmp_path):
    """output_pdf may be an attacker-placed symlink; delivery must not write
    through it to the link target."""
    _fake_engine(monkeypatch)
    tex = tmp_path / "r.tex"; tex.write_text("x")
    outside = tmp_path / "outside.pdf"; outside.write_bytes(b"ORIGINAL")
    deliver = tmp_path / "deliver"; deliver.mkdir()
    link = deliver / "report.pdf"; link.symlink_to(outside)

    def fake_run(argv, *, cwd, env, timeout):
        (Path(cwd) / "r.pdf").write_bytes(b"%PDF-NEW")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)
    r = compile_document(str(tex), output_pdf=str(link))
    assert r["success"] is True
    assert outside.read_bytes() == b"ORIGINAL"   # link target NOT overwritten
    assert link.read_bytes() == b"%PDF-NEW"      # PDF delivered to the path
    assert not link.is_symlink()                 # symlink replaced by a real file


def test_capabilities_scope_assets_and_output():
    """document.compile alone under-authorizes: reading assets_dir and writing
    output_pdf need explicit (optional) filesystem capabilities."""
    from agentic_cli.tools.registry import get_registry

    defn = get_registry().get("compile_document")
    caps = {(c.name, c.target_arg, c.optional) for c in defn.capabilities}
    assert ("document.compile", "source_path", False) in caps
    assert ("filesystem.read", "assets_dir", True) in caps
    assert ("filesystem.write", "output_pdf", True) in caps


def test_env_passes_tex_config_but_not_texinputs(monkeypatch, tmp_path):
    """TeX's own TEX* config vars pass through (so a custom TEXMFHOME works),
    but a caller-inherited TEXINPUTS is dropped in favor of our controlled one."""
    _fake_engine(monkeypatch)
    monkeypatch.setenv("TEXMFHOME", "/home/user/texmf")
    monkeypatch.setenv("TEXINPUTS", "/evil/inputs")
    tex = tmp_path / "r.tex"; tex.write_text("x")
    captured = {}

    def fake_run(argv, *, cwd, env, timeout):
        captured["env"] = env
        (Path(cwd) / "r.pdf").write_bytes(b"%PDF")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)
    compile_document(str(tex), assets_dir="/tmp/assets")
    assert captured["env"].get("TEXMFHOME") == "/home/user/texmf"
    assert "/evil/inputs" not in captured["env"]["TEXINPUTS"]
    assert "/tmp/assets" in captured["env"]["TEXINPUTS"]


def test_delivery_preserves_readable_mode(monkeypatch, tmp_path):
    """Delivered PDF keeps the produced file's mode (copystat), not the
    mkstemp default 0600 — a report is not a secret and consumers expect it
    readable."""
    import os as _os

    _fake_engine(monkeypatch)
    tex = tmp_path / "r.tex"; tex.write_text("x")
    out = tmp_path / "deliver" / "report.pdf"

    def fake_run(argv, *, cwd, env, timeout):
        p = Path(cwd) / "r.pdf"
        p.write_bytes(b"%PDF")
        _os.chmod(p, 0o644)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)
    r = compile_document(str(tex), output_pdf=str(out))
    assert r["success"] is True
    assert out.stat().st_mode & 0o044  # group/other readable, not 0600
