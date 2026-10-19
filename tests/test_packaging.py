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


def test_openai_is_declared():
    """The ADK model factory builds GPT, OpenRouter and endpoint models on ADK's
    OpenAI classes, which import ``openai``; google-adk pulls it only through
    an extra."""
    assert "openai" in _dep_names(), (
        "openai is imported by the ADK model factory (google.adk.integrations.openai) "
        "but is not declared in pyproject.toml [project.dependencies]."
    )


#: The ``langgraph`` extra's floors: Deep Agents' (0.7.0). The LangChain model
#: adapter needs what they bring: ChatAnthropic's ``reasoning_effort`` and
#: summarized adaptive thinking, Gemini's ``reasoning_effort``, the profile's
#: effort fields, and ``ChatOpenRouter``.
_LANGGRAPH_FLOORS = {
    "langchain": "1.4.6",
    "langchain-core": "1.6.11",
    "langchain-anthropic": "1.7.6",
    "langchain-google-genai": "4.4.2",
    "langchain-openai": "1.7",
    "langchain-openrouter": "0.2",
}


def test_the_langgraph_extra_has_the_model_adapter_floors():
    from packaging.requirements import Requirement

    extra = tomllib.loads(_PYPROJECT.read_text())["project"]["optional-dependencies"]
    floors = {}
    for spec in extra["langgraph"]:
        requirement = Requirement(spec)
        floors[requirement.name.lower()] = next(
            (s.version for s in requirement.specifier if s.operator == ">="), None
        )
    assert {name: floors.get(name) for name in _LANGGRAPH_FLOORS} == _LANGGRAPH_FLOORS


def test_the_model_table_ships_in_the_wheel():
    """``agentic_cli.models`` reads ``data/table.json`` at run time.

    hatchling puts every file under a listed package into the wheel, except
    what the build config excludes and what ``.gitignore`` ignores. This pins
    all three; the wheel acceptance test checks the installed file itself.
    """
    import subprocess

    root = _PYPROJECT.parent
    table = root / "src" / "agentic_cli" / "models" / "data" / "table.json"
    assert table.is_file()

    wheel = tomllib.loads(_PYPROJECT.read_text())["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert "src/agentic_cli" in wheel["packages"]
    assert not {"exclude", "only-include", "artifacts"} & set(wheel), wheel

    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(table)], cwd=root, capture_output=True
    )
    if ignored.returncode == 128:  # not a git checkout (e.g. an sdist)
        return
    assert ignored.returncode == 1, "data/table.json is git-ignored, so hatchling leaves it out"


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
    "langchain_anthropic",
    "langchain_google_genai",
    "langchain_openai",
    "langchain_openrouter",
    "openrouter",
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


def test_the_adk_model_path_imports_without_the_langgraph_extra():
    """The wire details both adapters share (``workflow/model_wire.py``) bring
    no LangChain package to an ADK-only install."""
    assert _import_with_optionals_blocked("agentic_cli.workflow.model_wire") == set()
    assert _import_with_optionals_blocked("agentic_cli.workflow.adk.model_factory") == set()
