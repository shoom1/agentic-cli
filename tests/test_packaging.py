"""Packaging consistency: every eagerly-imported third-party module must be a
declared dependency in pyproject.toml, so a clean ``pip install`` doesn't fail
on first import (a package built from pyproject alone won't see environment.yml).
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
