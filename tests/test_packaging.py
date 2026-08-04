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


# --- Optional-capability isolation -------------------------------------------
#
# The heavyweight capability stacks live behind extras (``kb``: torch /
# sentence-transformers / faiss / bm25; ``langgraph``). Importing the base
# package — or the tool package, which the tool resolver imports to discover
# built-ins — must not require any of them, and must not drag them in.

_OPTIONAL_ROOTS = {
    "torch",
    "sentence_transformers",
    "faiss",
    "bm25s",
    "rank_bm25",
    "langgraph",
    "langchain",
    "langchain_core",
}


def _import_with_optionals_blocked(module_name: str) -> set[str]:
    """Import ``module_name`` in a subprocess with the extras blocked.

    Returns:
        The set of optional roots that ended up in ``sys.modules`` anyway.
    """
    import json
    import subprocess
    import sys

    script = f"""
import builtins, importlib, json, sys
blocked = {sorted(_OPTIONAL_ROOTS)!r}
_real = builtins.__import__
def _guard(name, *a, **kw):
    if name.split(".")[0] in blocked:
        raise ImportError("blocked optional dependency: " + name)
    return _real(name, *a, **kw)
builtins.__import__ = _guard
importlib.import_module({module_name!r})
builtins.__import__ = _real
print(json.dumps([m for m in blocked if m in sys.modules]))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert proc.returncode == 0, (
        f"importing {module_name} requires an optional dependency:\n{proc.stderr}"
    )
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


def test_base_package_imports_without_optional_extras():
    assert _import_with_optionals_blocked("agentic_cli") == set()


def test_tools_package_imports_without_optional_extras():
    """Tool auto-discovery must not pull the kb/langgraph stacks."""
    assert _import_with_optionals_blocked("agentic_cli.tools") == set()
