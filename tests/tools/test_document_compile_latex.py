"""Real-engine compile — skipped unless a TeX engine is installed.

Set LATEX_REQUIRE=1 to fail (not skip) when no engine is present.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from agentic_cli.tools import compile_document

pytestmark = pytest.mark.latex

_HAS_ENGINE = shutil.which("latexmk") or shutil.which("pdflatex")
if not _HAS_ENGINE and os.environ.get("LATEX_REQUIRE") != "1":
    pytest.skip("no LaTeX engine (latexmk/pdflatex) on PATH", allow_module_level=True)


def test_compiles_minimal_document(tmp_path):
    build = tmp_path / "build"; build.mkdir()
    tex = build / "r.tex"
    tex.write_text(
        "\\documentclass{article}\n\\begin{document}\n"
        "Hello \\textbf{world}.\n\\end{document}\n"
    )
    out = tmp_path / "deliver" / "report.pdf"
    r = compile_document(str(tex), output_pdf=str(out))
    assert r["success"] is True, r
    assert Path(r["pdf_path"]).is_file() and Path(r["pdf_path"]).stat().st_size > 0
    assert out.is_file()
    # Isolation contract: the build runs in a private temp dir (cleaned up), so
    # intermediates never land in the source dir or beside the delivered PDF.
    assert not (build / "r.log").exists()            # source dir stays clean
    assert not (out.parent / "r.log").exists()       # not beside delivered PDF


# --- Lockdown of the TeX process (real engine) ------------------------------
# Paths are neutral: a temp project with an ordinary file beside it.

ENGINES = ["latexmk", "pdflatex"]


def _engine(name: str) -> str:
    if not shutil.which(name) and os.environ.get("LATEX_REQUIRE") != "1":
        pytest.skip(f"{name} not on PATH")
    return name


def _compile(tmp_path, body: str, engine: str, **kwargs) -> dict:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    tex = project / "doc.tex"
    tex.write_text(body)
    return compile_document(
        str(tex), output_pdf=str(tmp_path / "out" / "doc.pdf"), engine=engine, **kwargs,
    )


def _outside_file(tmp_path) -> Path:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    note = elsewhere / "note.tex"
    note.write_text("OUTSIDEMARKER\n")
    return note


@pytest.mark.parametrize("engine", ENGINES)
def test_shell_escape_is_disabled(tmp_path, engine):
    """\\pdfshellescape is 0 disabled, 1 enabled, 2 restricted (TeX Live's
    default, which latexmk's pdflatex used to get)."""
    r = _compile(tmp_path, (
        "\\documentclass{article}\n"
        "\\ifnum\\pdfshellescape>0 \\errmessage{SHELL-ESCAPE-ENABLED}\\fi\n"
        "\\begin{document}ok\\end{document}\n"
    ), _engine(engine))
    assert r["success"] is True, r


@pytest.mark.parametrize("engine", ENGINES)
def test_a_file_outside_the_project_cannot_be_input(tmp_path, engine):
    note = _outside_file(tmp_path)
    r = _compile(tmp_path, (
        "\\documentclass{article}\n\\begin{document}\n"
        f"\\input{{{note}}}\n"
        "\\end{document}\n"
    ), _engine(engine))
    assert r["success"] is False, r
    assert "OUTSIDEMARKER" not in json.dumps(r)
    assert not (tmp_path / "out" / "doc.pdf").exists()


@pytest.mark.parametrize("engine", ENGINES)
def test_a_file_outside_the_project_cannot_be_dumped(tmp_path, engine):
    """\\pdffiledump returns a file's bytes as hex, a read path that does not
    go through \\input."""
    note = _outside_file(tmp_path)
    r = _compile(tmp_path, (
        "\\documentclass{article}\n"
        f"\\edef\\leak{{\\pdffiledump offset 0 length 13 {{{note}}}}}\n"
        "\\errmessage{LEAK=\\leak}\n"
        "\\begin{document}x\\end{document}\n"
    ), _engine(engine))
    assert "4F5554534944454D41524B4552" not in json.dumps(r).upper()  # OUTSIDEMARKER


@pytest.mark.parametrize("engine", ENGINES)
def test_a_dotfile_in_the_project_cannot_be_input(tmp_path, engine):
    """A project's dotfiles (.env and the like) are found through TEXINPUTS."""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".private.tex").write_text("DOTFILEMARKER\n")
    r = _compile(tmp_path, (
        "\\documentclass{article}\n\\begin{document}\n"
        "\\input{.private.tex}\n"
        "\\end{document}\n"
    ), _engine(engine))
    assert r["success"] is False, r
    assert "DOTFILEMARKER" not in json.dumps(r)


@pytest.mark.parametrize("engine", ENGINES)
def test_project_and_asset_files_still_resolve(tmp_path, engine):
    """The lockdown must not break ordinary documents: siblings of the source
    and files in assets_dir are found by relative name through TEXINPUTS."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "chapter.tex").write_text("Chapter text.\n")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "extra.tex").write_text("Extra text.\n")
    r = _compile(tmp_path, (
        "\\documentclass{article}\n\\begin{document}\n"
        "\\input{chapter}\n\\input{extra}\n"
        "\\end{document}\n"
    ), _engine(engine), assets_dir=str(assets))
    assert r["success"] is True, r
