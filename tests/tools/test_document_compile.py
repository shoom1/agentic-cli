"""Offline tests for compile_document — subprocess and engine lookup faked."""
from __future__ import annotations

import os
import signal
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


def test_success_delivers_pdf_and_isolates_intermediates(monkeypatch, tmp_path):
    _fake_engine(monkeypatch)
    tex = tmp_path / "r.tex"
    tex.write_text("\\documentclass{article}\\begin{document}hi\\end{document}")

    def fake_run(argv, *, cwd, env, timeout):
        # fake_run receives the private build dir as cwd; write artifacts there
        (Path(cwd) / "r.pdf").write_bytes(b"%PDF-1.5 fake")
        (Path(cwd) / "r.log").write_text("output written on r.pdf")
        (Path(cwd) / "r.aux").write_text("\\relax")
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)
    out = tmp_path / "deliver" / "report.pdf"
    r = compile_document(str(tex), output_pdf=str(out), assets_dir=str(tmp_path / "assets"))
    assert r["success"] is True
    assert r["pdf_path"] == str(out)
    assert out.is_file()                              # PDF promoted to delivery dir
    assert not (tmp_path / "r.aux").exists()          # intermediates NOT in the source dir
    assert not (out.parent / "r.aux").exists()        # nor beside the delivered PDF


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
    assert r["pdf_path"] is None   # build dir (with the PDF) is cleaned up on return
    assert "duration_ms" in r


def test_forced_unsupported_engine_rejected(monkeypatch, tmp_path):
    """Finding 3: forcing engine='xelatex' (not in _ENGINES) must be rejected → No LaTeX engine error."""
    monkeypatch.setattr(mod, "_which", lambda n: f"/usr/bin/{n}" if n == "xelatex" else None)
    tex = tmp_path / "r.tex"; tex.write_text("x")
    r = compile_document(str(tex), engine="xelatex")
    assert r["success"] is False
    assert "No LaTeX engine" in r["error"]


# --- Fix wave — final review (I1/I2/M1/M2/M3) ---

def test_run_group_kill_on_timeout(monkeypatch, tmp_path):
    """I1 regression: _run starts a new session and kills the process group on
    timeout (now via proc.wait, not communicate)."""
    import subprocess as _subprocess

    popen_kwargs: dict = {}
    killpg_calls: list = []

    class FakePopen:
        pid = 42

        def __init__(self, argv, **kwargs):
            popen_kwargs.update(kwargs)

        def wait(self, timeout=None):
            if timeout is not None:
                raise _subprocess.TimeoutExpired([], timeout)
            return 0

    monkeypatch.setattr(mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(mod.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(mod.os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))

    try:
        mod._run(["latexmk"], cwd=str(tmp_path), env={}, timeout=1.0)
    except _subprocess.TimeoutExpired:
        pass  # expected

    assert popen_kwargs.get("start_new_session") is True
    assert len(killpg_calls) >= 1
    assert killpg_calls[0][1] == signal.SIGKILL


def test_run_caps_captured_output(tmp_path):
    import os as _os
    import sys as _sys

    argv = [_sys.executable, "-c", "print('x' * 1_000_000)"]
    r = mod._run(argv, cwd=str(tmp_path), env={"PATH": _os.environ.get("PATH", "")}, timeout=30)
    assert r.returncode == 0
    assert len(r.stdout) <= mod._MAX_CAPTURE_BYTES + 1  # bounded (byte cap + trailing newline)
    assert r.stdout.rstrip().endswith("x")              # tail retained


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


# --- P0-3 re-review: env scope, assets_dir separator, latexmk -norc ---

def test_env_excludes_nontex_vars_starting_with_tex(monkeypatch, tmp_path):
    """`k.startswith('TEX')` is too broad — TEXT_API_TOKEN etc. must NOT leak;
    only real TeX vars (TEXMF*/known) pass."""
    _fake_engine(monkeypatch)
    monkeypatch.setenv("TEXT_API_TOKEN", "sk-must-not-leak")
    monkeypatch.setenv("TEXMFHOME", "/home/u/texmf")
    tex = tmp_path / "r.tex"; tex.write_text("x")
    captured = {}

    def fake_run(argv, *, cwd, env, timeout):
        captured["env"] = env
        (Path(cwd) / "r.pdf").write_bytes(b"%PDF")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)
    compile_document(str(tex))
    assert "TEXT_API_TOKEN" not in captured["env"]
    assert captured["env"].get("TEXMFHOME") == "/home/u/texmf"


def test_env_texmf_uses_exact_allowlist_not_prefix(monkeypatch, tmp_path):
    """A real TEXMF var passes; a TEXMF-prefixed non-var (potential secret) does not."""
    _fake_engine(monkeypatch)
    monkeypatch.setenv("TEXMFHOME", "/home/u/texmf")
    monkeypatch.setenv("TEXMF_SECRET", "leak")
    tex = tmp_path / "r.tex"; tex.write_text("x")
    captured = {}

    def fake_run(argv, *, cwd, env, timeout):
        captured["env"] = env
        (Path(cwd) / "r.pdf").write_bytes(b"%PDF")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)
    compile_document(str(tex))
    assert captured["env"].get("TEXMFHOME") == "/home/u/texmf"
    assert "TEXMF_SECRET" not in captured["env"]


def test_assets_dir_with_path_separator_rejected(monkeypatch, tmp_path):
    """assets_dir authorized as one filesystem path must not smuggle extra
    TEXINPUTS roots via os.pathsep (e.g. 'assets:/etc')."""
    import os as _os

    _fake_engine(monkeypatch)
    tex = tmp_path / "r.tex"; tex.write_text("x")
    r = compile_document(str(tex), assets_dir=f"assets{_os.pathsep}/etc")
    assert r["success"] is False
    assert "assets_dir" in r["error"].lower()


def test_latexmk_uses_norc(monkeypatch, tmp_path):
    """latexmk must run with -norc so a build-dir/home .latexmkrc (arbitrary
    Perl) is not executed."""
    _fake_engine(monkeypatch)
    tex = tmp_path / "r.tex"; tex.write_text("x")
    captured = {}

    def fake_run(argv, *, cwd, env, timeout):
        captured["argv"] = argv
        (Path(cwd) / "r.pdf").write_bytes(b"%PDF")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)
    compile_document(str(tex))
    assert "-norc" in captured["argv"]


def test_option_like_source_name_not_treated_as_flag(monkeypatch, tmp_path):
    """A source basename starting with '-' must be anchored (./) so the engine
    parses it as a file, not an option — otherwise '-pdflatex=CMD.tex' et al.
    execute arbitrary host commands (which -norc does NOT prevent)."""
    _fake_engine(monkeypatch)
    tex = tmp_path / "-pdflatex=evil.tex"
    tex.write_text("x")
    captured = {}

    def fake_run(argv, *, cwd, env, timeout):
        captured["argv"] = argv
        (Path(cwd) / (tex.stem + ".pdf")).write_bytes(b"%PDF")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)
    r = compile_document(str(tex))
    assert "-pdflatex=evil.tex" not in captured["argv"]        # never a bare option-like token
    assert "./-pdflatex=evil.tex" in captured["argv"]          # anchored as a path
    assert r["success"] is True


def test_default_delivery_to_source_dir_without_intermediates(monkeypatch, tmp_path):
    """No output_pdf → PDF lands at <source dir>/<stem>.pdf, but the build's
    intermediates never touch the source dir."""
    _fake_engine(monkeypatch)
    tex = tmp_path / "r.tex"; tex.write_text("x")

    def fake_run(argv, *, cwd, env, timeout):
        (Path(cwd) / "r.pdf").write_bytes(b"%PDF")
        (Path(cwd) / "r.aux").write_text("aux")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)
    r = compile_document(str(tex))                     # no output_pdf
    assert r["success"] is True
    assert r["pdf_path"] == str(tmp_path / "r.pdf")
    assert (tmp_path / "r.pdf").is_file()              # delivered to source dir
    assert not (tmp_path / "r.aux").exists()           # intermediate isolated in temp


# --- P0-3 hardening: tail-read .log to bound memory on runaway compiler logs ---


def test_read_log_tail_bounds_large_log(tmp_path):
    log = tmp_path / "big.log"
    log.write_text("START\n" + ("x" * 200_000) + "\n! Real error.\nEND\n")
    out = mod._read_log_tail(log, fallback="FB")
    assert "END" in out and "! Real error." in out       # tail retained
    assert "START" not in out                              # head dropped
    assert len(out) <= mod._LOG_TAIL_BYTES + 3             # bounded


def test_read_log_tail_missing_returns_fallback(tmp_path):
    assert mod._read_log_tail(tmp_path / "nope.log", fallback="FB") == "FB"


def test_read_log_tail_oserror_returns_fallback(monkeypatch, tmp_path):
    """An existing-but-unreadable log (open raises OSError) returns fallback, not raises."""
    log = tmp_path / "x.log"; log.write_text("data")

    def boom(*a, **k):
        raise PermissionError("nope")

    monkeypatch.setattr("builtins.open", boom)
    assert mod._read_log_tail(log, fallback="FB") == "FB"


def test_texinputs_roots_are_absolute_for_relative_source(monkeypatch, tmp_path):
    """Relative source_path/assets_dir must resolve to ABSOLUTE TEXINPUTS roots
    (the build runs in a temp dir, so relative roots would resolve there)."""
    _fake_engine(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "assets").mkdir()
    (tmp_path / "r.tex").write_text("x")
    captured = {}

    def fake_run(argv, *, cwd, env, timeout):
        captured["env"] = env
        (Path(cwd) / "r.pdf").write_bytes(b"%PDF")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)
    compile_document("r.tex", assets_dir="assets")  # relative paths
    entries = [e for e in captured["env"]["TEXINPUTS"].split(os.pathsep) if e]
    assert entries
    assert all(os.path.isabs(e) for e in entries)
    assert str((tmp_path / "assets").resolve()) in entries


def test_run_limits_child_file_size(monkeypatch, tmp_path):
    """A runaway child writing beyond RLIMIT_FSIZE is killed (SIGXFSZ), not
    allowed to fill the disk."""
    import os as _os
    import sys as _sys

    monkeypatch.setattr(mod, "_RLIMIT_FSIZE_BYTES", 4096)
    argv = [_sys.executable, "-c", "open('big.bin','wb').write(b'x' * 1_000_000)"]
    r = mod._run(argv, cwd=str(tmp_path), env={"PATH": _os.environ.get("PATH", "")}, timeout=30)
    assert r.returncode != 0                              # killed by the file-size limit
    assert (tmp_path / "big.bin").stat().st_size <= 4096 * 8   # capped, not 1MB
