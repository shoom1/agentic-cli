"""Compile a LaTeX document to PDF with a host TeX engine.

``compile_document`` is a narrow, permission-gated tool: it runs ``latexmk``
(preferred) or ``pdflatex`` as a guarded subprocess with the two arbitrary-code
vectors disabled — ``latexmk`` with ``-norc`` (so no ``.latexmkrc`` Perl is read
from the build directory or home) and ``pdflatex`` with ``-no-shell-escape`` (no
``\\write18``).  It runs on the host (not the container sandbox — an intentional
decoupling); OS-sandbox confinement of the build tree is deferred (spec §9).

The tool runs with a wall-clock timeout and a private temp build dir, and
returns a structured result.  It does not execute arbitrary host code by
default; a ``report_writer``-style agent uses it to turn an authored ``.tex``
into a PDF.  Build intermediates (``*.aux``, ``*.log``) are isolated in a
``tempfile.TemporaryDirectory`` and never written to the source or delivery dir;
only the final PDF is promoted via ``_deliver_no_follow``.

The subprocess receives only an allowlisted environment (``_ENV_PASSTHROUGH`` +
the ``TEXMF*``/TeX config vars), not the full host environment, so host secrets
aren't handed to the TeX process.  Delivery to ``output_pdf`` is a no-follow
atomic write (temp file + ``os.replace``) so an attacker-placed symlink at the
destination can't redirect the write.

Provisioning is host-based: the engine must be on ``PATH`` (TeX Live / MacTeX).
If neither is found the tool returns a structured error with an install hint.

The subprocess call and the engine lookup sit behind module-level seams
(``_run``, ``_which``) so the logic is unit-tested offline without a real TeX
install.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from agentic_cli.tools.registry import ToolCategory, register_tool
from agentic_cli.workflow.permissions import Capability

_ENGINES = ("latexmk", "pdflatex")
_LOG_TAIL_LINES = 40
_LOG_TAIL_BYTES = 64 * 1024
_MAX_CAPTURE_BYTES = 200_000

# Only these host env vars reach the TeX process. The tool must not hand the
# whole host environment (API keys, tokens) to a subprocess that — on the
# latexmk path — can execute arbitrary Perl from a .latexmkrc. PATH/HOME are
# needed for the engine binary and kpathsea; TEXINPUTS is set explicitly.
_ENV_PASSTHROUGH = (
    "PATH", "HOME", "TERM", "TMPDIR", "TEMP", "TMP",
    "LANG", "LC_ALL", "LC_CTYPE", "SOURCE_DATE_EPOCH",
    # latexmk is a Perl program; without its module path it can fail to load.
    "PERL5LIB", "PERLLIB",
)

# TeX's own search/config vars (kpathsea) that don't fall under the TEXMF*
# namespace. TEXINPUTS is deliberately excluded — it is set explicitly below.
_TEX_VARS = (
    "TEXFONTS", "TEXFORMATS", "TEXPOOL", "TEXPSHEADERS",
    "TEXCONFIG", "TEXDOCS", "TEXSOURCES",
)


def _build_env(assets_dir: str | None, source_dir: str | None = None) -> dict[str, str]:
    """Minimal, allowlisted environment for the TeX subprocess.

    Passes PATH/HOME/locale plus TeX's own configuration variables — the
    ``TEXMF*`` tree and a fixed set of other TeX vars — so a custom
    ``TEXMFHOME`` etc. keeps working, but not arbitrary host env (a name like
    ``TEXT_API_TOKEN`` starts with "TEX" yet is not a TeX var), and never the
    caller's ``TEXINPUTS`` (set explicitly below).

    ``assets_dir`` and ``source_dir`` become TEXINPUTS read roots so figures and
    ``\\input`` siblings resolve even though the build runs in a private temp dir.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k in _ENV_PASSTHROUGH or k.startswith("TEXMF") or k in _TEX_VARS
    }
    env.setdefault("PATH", os.defpath)
    roots = [r for r in (assets_dir, source_dir) if r]
    if roots:
        # Trailing empty entry lets kpathsea append its default search path.
        env["TEXINPUTS"] = os.pathsep.join(roots) + os.pathsep
    return env


def _deliver_no_follow(produced: Path, dest: Path) -> None:
    """Copy ``produced`` to ``dest`` without following a symlink at ``dest``.

    Writes to a private temp file in dest's directory, then atomically renames
    it over dest. ``os.replace`` swaps the destination *name*: if dest is a
    symlink the link itself is replaced (not written through), so an
    attacker-placed symlink can't redirect the write outside the intended path.

    This protects only the final path component. A symlinked ``dest.parent``
    (or an ancestor) still redirects the write; the permission engine
    canonicalizes ``output_pdf`` at check time, but a check→write window
    remains. Full parent containment is deferred (spec §9, OS-sandbox).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, open(produced, "rb") as src:
            shutil.copyfileobj(src, out)
            out.flush()
            os.fsync(out.fileno())
        # Match the produced PDF's mode/mtime (mkstemp is 0600) so the delivered
        # file has the readability a consumer expects, as the old copy2 did.
        shutil.copystat(produced, tmp_path)
        os.replace(tmp_path, dest)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _which(name: str) -> str | None:
    """Locate an executable on PATH (seam for tests)."""
    return shutil.which(name)


def _run(
    argv: list[str], *, cwd: str, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess:
    """Run a subprocess in its own process group so a timeout kills the whole
    tree (latexmk + its pdflatex grandchild), not just the direct child.
    stdout/stderr are captured to temp files and only the last
    _MAX_CAPTURE_BYTES of each are retained, bounding host memory. Seam for
    tests. POSIX (macOS/Linux), which is what the framework targets."""
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        proc = subprocess.Popen(
            argv, cwd=cwd, env=env,
            stdout=out_f, stderr=err_f,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()  # reap the killed group
            raise
        out = _tail_of_file(out_f, _MAX_CAPTURE_BYTES)
        err = _tail_of_file(err_f, _MAX_CAPTURE_BYTES)
    return subprocess.CompletedProcess(argv, proc.returncode, stdout=out, stderr=err)


def _detect_engine(engine: str | None) -> str | None:
    """Return the engine to use, or None if unavailable."""
    if engine is not None:
        return engine if (engine in _ENGINES and _which(engine)) else None
    for candidate in _ENGINES:
        if _which(candidate):
            return candidate
    return None


def _build_argv(engine: str, source: str) -> list[str]:
    """Compiler argv — never enables shell-escape.

    latexmk runs with ``-norc`` so it won't read ``.latexmkrc`` (arbitrary
    Perl) from the build directory or home; pdflatex runs with
    ``-no-shell-escape``. Neither path executes arbitrary host code by default.
    """
    if engine == "latexmk":
        return [
            "latexmk", "-norc", "-pdf", "-interaction=nonstopmode",
            "-halt-on-error", source,
        ]
    return [
        "pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
        "-halt-on-error", source,
    ]


def _safe_source_arg(name: str) -> str:
    """Anchor a source filename so a leading ``-`` can't be parsed as an engine
    option (arbitrary-exec via ``-pdflatex=CMD`` etc.). ``name`` is a basename
    and the subprocess cwd is the source's directory, so ``./`` resolves it."""
    return name if name.startswith("./") else f"./{name}"


def _parse_errors(log_text: str) -> list[str]:
    """Extract LaTeX error lines (those beginning with '!') from a log."""
    return [ln for ln in log_text.splitlines() if ln.startswith("!")]


def _tail_of_file(f, limit: int) -> str:
    """Return the last ``limit`` bytes of an open binary temp file, decoded."""
    size = f.seek(0, os.SEEK_END)
    f.seek(max(0, size - limit))
    return f.read().decode("utf-8", errors="replace")


def _read_log_tail(log_path: Path, fallback: str) -> str:
    """Return at most the last _LOG_TAIL_BYTES of the log (decoded), else
    ``fallback``. Bounds memory on a runaway compiler log."""
    try:
        if not log_path.is_file():
            return fallback
        with open(log_path, "rb") as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - _LOG_TAIL_BYTES))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return fallback


@register_tool(
    category=ToolCategory.EXECUTION,
    capabilities=[
        Capability("document.compile", target_arg="source_path"),
        # The tool also reads assets_dir and writes output_pdf when those are
        # supplied; scope them explicitly (optional → not exercised when absent).
        Capability("filesystem.read", target_arg="assets_dir", optional=True),
        Capability("filesystem.write", target_arg="output_pdf", optional=True),
    ],
    description=(
        "Compile a LaTeX source file to PDF using a host TeX engine (latexmk or "
        "pdflatex). Runs on the host: latexmk with -norc (no .latexmkrc) and "
        "pdflatex with -no-shell-escape, so it does not execute arbitrary host "
        "code by default. Returns the PDF path plus any compiler errors. "
        "Requires TeX Live/MacTeX on PATH."
    ),
)
def compile_document(
    source_path: str,
    output_pdf: str | None = None,
    assets_dir: str | None = None,
    engine: str | None = None,
    timeout_s: int = 120,
) -> dict[str, Any]:
    """Compile a LaTeX file to PDF (guarded subprocess; host TeX engine).

    Args:
        source_path: Path to the .tex file to compile.
        output_pdf: If set, the produced PDF is copied here (parents created);
            build intermediates stay in the source's directory.
        assets_dir: Directory prepended to TEXINPUTS so figures/resources resolve
            by bare name (e.g. an artifacts dir).
        engine: Force an engine ("latexmk"/"pdflatex"); default auto-detects
            (latexmk preferred).
        timeout_s: Wall-clock timeout; the process is killed on expiry.

    Returns:
        dict with success, pdf_path, engine, log_tail, errors, duration_ms, and
        (on setup/timeout failure) error.
    """
    src = Path(source_path)
    if not src.is_file():
        return {
            "success": False, "error": f"Source not found: {source_path}",
            "pdf_path": None, "engine": None, "log_tail": "", "errors": [],
            "duration_ms": 0,
        }

    if assets_dir and os.pathsep in assets_dir:
        # A path-list separator would turn one authorized filesystem.read target
        # into several TEXINPUTS search roots (e.g. "assets:/etc" also reads /etc).
        return {
            "success": False,
            "error": f"assets_dir must be a single path (no {os.pathsep!r}): {assets_dir}",
            "pdf_path": None, "engine": None, "log_tail": "", "errors": [],
            "duration_ms": 0,
        }

    chosen = _detect_engine(engine)
    if chosen is None:
        looked = engine or "/".join(_ENGINES)
        return {
            "success": False,
            "error": (
                f"No LaTeX engine on PATH (looked for {looked}). "
                "Install TeX Live or MacTeX."
            ),
            "pdf_path": None, "engine": None, "log_tail": "", "errors": [],
            "duration_ms": 0,
        }

    env = _build_env(assets_dir, source_dir=str(src.parent))
    argv = _build_argv(chosen, _safe_source_arg(src.name))
    start = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="texbuild-") as build_dir:
        build = Path(build_dir)
        try:
            shutil.copy2(src, build / src.name)
        except OSError as exc:
            return {
                "success": False, "error": f"Failed to stage source: {exc}",
                "pdf_path": None, "engine": chosen, "log_tail": "", "errors": [],
                "duration_ms": int((time.monotonic() - start) * 1000),
            }

        try:
            proc = _run(argv, cwd=str(build), env=env, timeout=float(timeout_s))
        except subprocess.TimeoutExpired:
            return {
                "success": False, "error": f"Compilation timed out after {timeout_s}s",
                "pdf_path": None, "engine": chosen, "log_tail": "", "errors": [],
                "duration_ms": int((time.monotonic() - start) * 1000),
            }
        except OSError as exc:
            return {
                "success": False, "error": f"Failed to run {chosen}: {exc}",
                "pdf_path": None, "engine": chosen, "log_tail": "", "errors": [],
                "duration_ms": int((time.monotonic() - start) * 1000),
            }
        duration_ms = int((time.monotonic() - start) * 1000)

        log_path = build / (src.stem + ".log")
        log_text = _read_log_tail(log_path, fallback=proc.stdout or "")
        log_tail = "\n".join(log_text.splitlines()[-_LOG_TAIL_LINES:])
        produced = build / (src.stem + ".pdf")
        success = proc.returncode == 0 and produced.is_file()

        if not success:
            return {
                "success": False, "pdf_path": None, "engine": chosen,
                "log_tail": log_tail, "errors": _parse_errors(log_text),
                "duration_ms": duration_ms, "error": None,
            }

        dest = Path(output_pdf) if output_pdf else (src.parent / (src.stem + ".pdf"))
        try:
            _deliver_no_follow(produced, dest)
        except OSError as exc:
            return {
                "success": False,
                "error": f"Failed to deliver PDF to {dest}: {exc}",
                "pdf_path": str(produced), "engine": chosen,
                "log_tail": log_tail, "errors": [], "duration_ms": duration_ms,
            }

    return {
        "success": True, "pdf_path": str(dest), "engine": chosen,
        "log_tail": log_tail, "errors": [], "duration_ms": duration_ms,
        "error": None,
    }
