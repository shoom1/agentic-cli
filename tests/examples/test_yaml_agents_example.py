"""``examples/yaml_agents/app.py`` loads its agents from YAML and runs a turn.

The example's agents are defined in ``agents.yaml`` (prompts in ``prompts/``) and
its tool, ``word_count``, registers in the framework registry when ``app.py`` is
imported, so it is loaded in a fresh interpreter, never into the suite's own
process. That child loads it by path, as a script would run it (its
``__main__`` block, the terminal app, does not run), builds its manager through
the factory on Deep Agents, as the CLI does, initializes it offline (its catalog
lists no models, and each agent runs on a scripted model), and runs one turn:
the coordinator delegates to the researcher, which calls ``word_count``, and both
answer.
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
_EXAMPLE_DIR = _REPO_ROOT / "examples" / "yaml_agents"
_EXAMPLE = _EXAMPLE_DIR / "app.py"

_CHILD = """
import asyncio, importlib.util, json, sys

# The checkout under test first, then the suite (the child's env has no PYTHONPATH).
sys.path[:0] = [{src!r}, {repo!r}]
spec = importlib.util.spec_from_file_location("yaml_agents_app", {example!r})
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

from agentic_cli.models import ModelCatalog
from agentic_cli.tools.registry import identify_tool
from agentic_cli.workflow import EventType, create_workflow_manager_from_settings
from agentic_cli.workflow import model_services
from agentic_cli.workflow.deepagents import manager as manager_module
from agentic_cli.workflow.permissions import EXEMPT
from tests.workflow.deepagents.fakes import ScriptedModels, ai, task


class Unlisted(ModelCatalog):
    async def load(self):
        return None


TEXT = "The quick brown fox jumps over the lazy dog"
model_services.ModelCatalog = Unlisted  # what build_catalog builds
manager_module.build_chat_model = ScriptedModels(
    coordinator=[
        ai(calls=[task("researcher", "Count the words in: " + TEXT)]),
        ai("The researcher counted 9 words."),
    ],
    researcher=[
        ai(calls=[("word_count", {{"text": TEXT}})]),
        ai("The text has 9 words."),
    ],
)

configs = {{c.name: c for c in app.AGENT_CONFIGS}}
declared = {{
    "names": [c.name for c in app.AGENT_CONFIGS],
    "prompts": {{name: c.prompt for name, c in configs.items()}},
    "tools": {{name: list(c.tools) for name, c in configs.items()}},
    "sub_agents": {{name: list(c.sub_agents) for name, c in configs.items()}},
    "context": {{name: c.context for name, c in configs.items()}},
    "descriptions": {{name: c.description for name, c in configs.items()}},
}}


async def main():
    settings = app.Settings(orchestrator="deepagents")
    manager = create_workflow_manager_from_settings(app.AGENT_CONFIGS, settings)
    await manager.initialize_services()
    try:
        events = [e async for e in manager.process("How many words are in: " + TEXT, user_id="u")]
        found = {{
            "class": type(manager).__name__,
            "agents": sorted(manager.agent_models),
            "declared": declared,
            # The tool names resolved from the YAML are the registry's, so the gate lets them through.
            "exempt": [
                identify_tool(tool).capabilities is EXEMPT
                for tool in configs["researcher"].tools
            ],
            "results": {{
                e.metadata["tool_name"]: [e.metadata["agent"], e.metadata["result"]]
                for e in events if e.type is EventType.TOOL_RESULT
            }},
            "text": [
                [e.metadata["agent"], bool(e.metadata.get("subagent")), e.content]
                for e in events if e.type is EventType.TEXT
            ],
            "errors": [e.content for e in events if e.type is EventType.ERROR],
        }}
    finally:
        await manager.cleanup()
    print(json.dumps(found))  # last: the manager logs to stdout


asyncio.run(main())
"""


@pytest.fixture(scope="module")
def child_result(tmp_path_factory) -> subprocess.CompletedProcess:
    """The child's run, once: both tests read what the one turn found."""
    root = tmp_path_factory.mktemp("yaml_agents")
    home, cwd = root / "home", root / "cwd"
    home.mkdir()
    cwd.mkdir()
    return subprocess.run(
        [
            sys.executable, "-c",
            _CHILD.format(src=str(_REPO_ROOT / "src"), repo=str(_REPO_ROOT), example=str(_EXAMPLE)),
        ],
        cwd=str(cwd),
        env={**child_env(home), "GOOGLE_API_KEY": "probe-key"},  # fake: nothing is sent
        capture_output=True,
        text=True,
        timeout=STEP_TIMEOUT,
    )


def test_the_example_builds_its_manager_and_runs_a_delegated_turn(child_result):
    result = child_result

    assert result.returncode == 0, result.stderr[-2000:]
    found = json.loads(result.stdout.strip().splitlines()[-1])
    assert found["class"] == "DeepAgentsWorkflowManager"
    assert found["agents"] == ["coordinator", "researcher"]
    # The YAML's tool name resolved to the registered function: EXEMPT, so no prompt.
    assert found["exempt"] == [True]
    # The researcher called it; the coordinator called none of its own but the delegation.
    assert found["results"]["word_count"] == ["researcher", {"success": True, "words": 9}]
    assert found["results"]["task"][0] == "coordinator"
    # The researcher's answer goes to the coordinator (activity); only the coordinator replies.
    assert found["text"] == [
        ["researcher", True, "The text has 9 words."],
        ["coordinator", False, "The researcher counted 9 words."],
    ]
    assert found["errors"] == []
    # Nothing printed over a terminal: no Python warning while it loads and runs.
    assert result.stderr == ""


def test_the_agents_come_from_the_yaml_and_the_prompts_from_their_files(child_result):
    result = child_result

    assert result.returncode == 0, result.stderr[-2000:]
    declared = json.loads(result.stdout.strip().splitlines()[-1])["declared"]
    assert declared["names"] == ["coordinator", "researcher"]
    assert declared["sub_agents"] == {"coordinator": ["researcher"], "researcher": []}
    assert declared["context"] == {"coordinator": None, "researcher": "fresh"}
    assert declared["descriptions"]["researcher"]
    # Read from prompts/, not written into the YAML or the module.
    for name in ("coordinator", "researcher"):
        assert declared["prompts"][name] == (_EXAMPLE_DIR / "prompts" / f"{name}.md").read_text(
            encoding="utf-8"
        )
    # A tool is the registered name, resolved when the manager is built.
    assert declared["tools"]["researcher"] == ["word_count"]
