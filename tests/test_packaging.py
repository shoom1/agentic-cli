"""Packaging consistency regression pins.

A clean ``pip install`` uses only pyproject.toml (not environment.yml), so an
eagerly-imported third-party module missing from ``[project.dependencies]``
breaks on first import. This pins the specific modules that regressed; it is
not an exhaustive import-vs-dependency audit.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _declared_dependencies() -> list[str]:
    data = tomllib.loads(_PYPROJECT.read_text())
    return list(data.get("project", {}).get("dependencies", []))


def _dep_names() -> set[str]:
    """Normalized distribution names from the dependency specifiers."""
    names = set()
    for spec in _declared_dependencies():
        # Strip version/extras/markers: name is the leading run of allowed chars.
        name = spec.split(";")[0].strip()
        for sep in ("[", ">", "<", "=", "!", "~", " "):
            name = name.split(sep)[0]
        names.add(name.strip().lower())
    return names


def test_html2text_is_declared():
    """converter.py imports html2text at module load (tools/webfetch/converter.py)."""
    assert "html2text" in _dep_names(), (
        "html2text is imported eagerly by tools/webfetch/converter.py but is not "
        "declared in pyproject.toml [project.dependencies] — a clean pip install "
        "breaks when the webfetch tools are imported."
    )
