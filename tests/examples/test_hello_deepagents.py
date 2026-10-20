"""``examples/hello_deepagents.py`` loads, builds its manager and runs its tools.

The example registers its three tools in the framework registry when it is
imported, so it is loaded in a fresh interpreter, never into the suite's own
process. That child loads it as a module (its ``__main__`` block, the terminal
app, does not run), builds its manager through the factory, as the CLI does,
initializes it offline (its catalog lists no models, and its agent runs on a
scripted model), and runs one turn in which the agent calls two of the tools.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.examples.console_smoke import STEP_TIMEOUT, child_env

pytest.importorskip("deepagents")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE = _REPO_ROOT / "examples" / "hello_deepagents.py"

_CHILD = """
import asyncio, importlib.util, json, sys

sys.path.insert(0, {repo!r})  # the suite's fakes, not the example's imports
spec = importlib.util.spec_from_file_location("hello_deepagents", {example!r})
hello = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hello)

from agentic_cli.models import ModelCatalog
from agentic_cli.tools.registry import identify_tool
from agentic_cli.workflow import EventType, create_workflow_manager_from_settings
from agentic_cli.workflow import model_services
from agentic_cli.workflow.deepagents import manager as manager_module
from agentic_cli.workflow.permissions import EXEMPT
from tests.workflow.deepagents.fakes import ScriptedModels, ai


class Unlisted(ModelCatalog):
    async def load(self):
        return None


model_services.build_catalog = lambda settings, endpoints, *, transport=None: Unlisted(
    endpoints=endpoints
)
manager_module.build_chat_model = ScriptedModels(assistant=[
    ai(calls=[("calculate", {{"expression": "6 * 7"}}), ("echo", {{"message": "hi"}})]),
    ai("6 * 7 is 42."),
])


async def main():
    settings = hello.get_settings()
    manager = create_workflow_manager_from_settings(hello.AGENT_CONFIGS, settings)
    await manager.initialize_services()
    try:
        events = [e async for e in manager.process("What is 6 * 7?", user_id="u")]
        found = {{
            "class": type(manager).__name__,
            "orchestrator": settings.orchestrator.value,
            "agents": sorted(manager.agent_models),
            "exempt": [
                identify_tool(tool).capabilities is EXEMPT
                for tool in hello.AGENT_CONFIGS[0].tools
            ],
            # The two calls run at once, so their results come in either order.
            "results": {{
                e.metadata["tool_name"]: e.metadata["result"]
                for e in events if e.type is EventType.TOOL_RESULT
            }},
            "text": [e.content for e in events if e.type is EventType.TEXT],
        }}
    finally:
        await manager.cleanup()
    print(json.dumps(found))  # last: the manager logs to stdout


asyncio.run(main())
"""


@pytest.fixture
def run_child(tmp_path):
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()

    def run() -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", _CHILD.format(repo=str(_REPO_ROOT), example=str(_EXAMPLE))],
            cwd=str(cwd),
            env={**child_env(home), "GOOGLE_API_KEY": "probe-key"},  # fake: nothing is sent
            capture_output=True,
            text=True,
            timeout=STEP_TIMEOUT,
        )

    return run


def test_the_example_builds_its_manager_and_runs_its_tools(run_child):
    result = run_child()

    assert result.returncode == 0, result.stderr
    found = json.loads(result.stdout.strip().splitlines()[-1])
    assert found == {
        "class": "DeepAgentsWorkflowManager",
        "orchestrator": "deepagents",
        "agents": ["assistant"],
        # Registered, so the permission gate lets them run without asking.
        "exempt": [True, True, True],
        "results": {
            "calculate": {"success": True, "expression": "6 * 7", "result": 42},
            "echo": {"success": True, "echoed": "hi", "length": 2},
        },
        "text": ["6 * 7 is 42."],
    }
    # Nothing printed over a terminal: no Python warning while it loads and runs.
    assert result.stderr == ""


def test_the_calculator_evaluates_arithmetic_only(tmp_path):
    """``calculate`` is EXEMPT, so it runs without asking: it must not run code."""
    code = (
        "import importlib.util, json; "
        f"spec = importlib.util.spec_from_file_location('hello_deepagents', {str(_EXAMPLE)!r}); "
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        "print(json.dumps([m.calculate(e) for e in "
        "['sqrt(16) + 2 ** 3', '__import__(\"os\").getcwd()', '().__class__', "
        "'9 ** 9 ** 9', '1 / 0']]))"
    )
    (tmp_path / "home").mkdir()
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(tmp_path),
        env=child_env(tmp_path / "home"),
        capture_output=True,
        text=True,
        timeout=STEP_TIMEOUT,
    )

    assert result.returncode == 0, result.stderr
    answers = json.loads(result.stdout.strip().splitlines()[-1])
    assert answers[0] == {"success": True, "expression": "sqrt(16) + 2 ** 3", "result": 12.0}
    assert [a["success"] for a in answers[1:]] == [False, False, False, False]
    assert answers[1]["error"] == "unsupported expression: __import__('os').getcwd()"
    assert answers[3]["error"] == "exponent too large"
