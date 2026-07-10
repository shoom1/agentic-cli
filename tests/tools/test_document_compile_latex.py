"""Real-engine compile — skipped unless a TeX engine is installed.

Set LATEX_REQUIRE=1 to fail (not skip) when no engine is present.
"""
from __future__ import annotations

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
    assert (build / "r.log").is_file()               # intermediates in build dir
    assert not (out.parent / "r.log").exists()       # not beside delivered PDF
