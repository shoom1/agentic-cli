"""Every example's agents get tools the permission gate lets through.

The gate denies a tool the registry did not issue, so an example that passes
a plain function in ``AgentConfig.tools`` shows a model refused every call:
``hello_agent.py`` and ``hello_langgraph.py`` did, after identity gating
arrived. Each example is loaded in a fresh interpreter (two examples
registering a ``calculate`` would collide in one registry, and the suite's own
registry stays untouched), and every tool of every ``AgentConfig`` it defines
must resolve to a registry definition (a name, as an agents YAML file has it,
is resolved first: ``yaml_agents`` builds its configs from one).

The hello examples' ``calculate`` is registered EXEMPT, so it runs without a
prompt: it must evaluate arithmetic and nothing else.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.examples.console_smoke import STEP_TIMEOUT, child_env

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = sorted((_REPO_ROOT / "examples").glob("*.py"))
#: Examples known to define agents: the test fails if it finds none in them.
_WITH_AGENTS = {"hello_agent", "hello_langgraph", "hello_deepagents", "jobs_demo"}
_HELLO = ("hello_agent", "hello_langgraph", "hello_deepagents")

_LOAD = """
import importlib.util, json, sys

# The checkout under test first, then the suite (the child's env has no PYTHONPATH).
sys.path[:0] = [{src!r}, {repo!r}]
spec = importlib.util.spec_from_file_location({name!r}, {path!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
"""

_IMPORT = """
import importlib, json, sys

# The checkout under test first, then the suite and the examples' packages.
sys.path[:0] = [{src!r}, {repo!r}, {examples!r}]
module = importlib.import_module({name!r})
"""

_REPORT = """
from agentic_cli.tools.registry import identify_tool
from agentic_cli.tools.tool_resolver import resolve_tool
from agentic_cli.workflow.config import AgentConfig


def configs(value):
    if isinstance(value, AgentConfig):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from configs(item)


def issued(tool):
    # A name, as agents YAML has it, resolves first; then the registry must have issued it.
    if isinstance(tool, str):
        try:
            tool = resolve_tool(tool)
        except ValueError:
            return False
    return identify_tool(tool) is not None


found = [config for value in vars(module).values() for config in configs(value)]
unregistered = [
    f"{{config.name}}: {{getattr(tool, '__name__', repr(tool))}}"
    for config in found
    for tool in config.tools or []
    if not issued(tool)
]
print(json.dumps({{"configs": len(found), "unregistered": unregistered}}))
"""
_CHECK_TOOLS = _LOAD + _REPORT

_CALCULATE = _LOAD + """
expressions = {expressions!r}
print(json.dumps([module.calculate(expression) for expression in expressions]))
"""


def _run(script: str, tmp_path: Path) -> object:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=child_env(home),
        capture_output=True,
        text=True,
        timeout=STEP_TIMEOUT,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def _script(template: str, example: Path, **values) -> str:
    return template.format(
        src=str(_REPO_ROOT / "src"),
        repo=str(_REPO_ROOT),
        examples=str(_REPO_ROOT / "examples"),
        name=values.pop("name", example.stem),
        path=str(example),
        **values,
    )


def test_the_examples_with_agents_are_all_found():
    assert _WITH_AGENTS <= {example.stem for example in _EXAMPLES}


@pytest.mark.parametrize("example", _EXAMPLES, ids=[example.stem for example in _EXAMPLES])
def test_every_agent_tool_is_registered(example, tmp_path):
    if example.stem == "hello_langgraph":
        pytest.importorskip("langgraph")
    if example.stem == "hello_deepagents":
        pytest.importorskip("deepagents")

    report = _run(_script(_CHECK_TOOLS, example), tmp_path)

    if example.stem in _WITH_AGENTS:
        assert report["configs"] > 0, f"{example.name} defines no AgentConfig the test can see"
    assert report["unregistered"] == [], (
        f"{example.name}: the gate denies every call to these: {report['unregistered']}"
    )


def test_the_research_demo_agent_tools_are_registered(tmp_path):
    """research_demo is a package, which the per-file loop above skips."""
    agents = _REPO_ROOT / "examples" / "research_demo" / "agents.py"

    report = _run(_script(_IMPORT + _REPORT, agents, name="research_demo.agents"), tmp_path)

    assert report["configs"] > 0
    assert report["unregistered"] == [], report["unregistered"]


def test_the_yaml_example_agent_tools_are_registered(tmp_path):
    """yaml_agents' agents come from a YAML file, whose tools are names to resolve."""
    app = _REPO_ROOT / "examples" / "yaml_agents" / "app.py"

    report = _run(_script(_CHECK_TOOLS, app, name="yaml_agents_app"), tmp_path)

    assert report["configs"] == 2
    assert report["unregistered"] == [], report["unregistered"]


_SAFE = ["sqrt(16)", "2 + 3 * 4", "round(pi, 2)"]
_REFUSED = [
    "(-8) ** 0.5",  # complex: no provider adapter can send it back
    "(-8) ** (1 / 3)",
    "1e308 * 10",  # inf
    "1e999",
    "().__class__.__base__.__subclasses__()",
    "__import__('os').getcwd()",
    "(lambda: 1)()",
    "9 ** 9 ** 9",
    "open('x')",
]


@pytest.mark.parametrize("name", _HELLO)
def test_the_hello_calculator_evaluates_arithmetic_only(name, tmp_path):
    if name == "hello_langgraph":
        pytest.importorskip("langgraph")
    if name == "hello_deepagents":
        pytest.importorskip("deepagents")
    example = _REPO_ROOT / "examples" / f"{name}.py"

    results = _run(_script(_CALCULATE, example, expressions=_SAFE + _REFUSED), tmp_path)

    safe, refused = results[: len(_SAFE)], results[len(_SAFE):]
    assert [(r["success"], r["result"]) for r in safe] == [(True, 4.0), (True, 14), (True, 3.14)]
    assert [r["success"] for r in refused] == [False] * len(_REFUSED), refused
