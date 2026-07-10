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
    monkeypatch.setattr(mod.shutil, "copy2", lambda src, dst: (_ for _ in ()).throw(OSError("disk full")))

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
