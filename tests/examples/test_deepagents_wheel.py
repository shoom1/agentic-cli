"""Acceptance smoke for the ``deepagents`` extra against the **built wheel**.

The offline suite imports the Deep Agents backend from the checkout, with
whatever the developer's environment has installed. That cannot catch an
extra whose pins do not resolve together, a pin that admits a release the
backend cannot import, or a backend module the wheel leaves out. This builds
the wheel, installs it with ``[deepagents]`` into an isolated virtualenv, and
asks that interpreter to import the backend and to build a manager through
the factory, agents and all (offline: the probe lists no models).

Opt-in on the same two axes as ``test_research_demo_wheel.py``: the ``wheel``
marker *and* ``AGENTIC_WHEEL_ACCEPTANCE=1``, because building a wheel and
installing its dependencies is slow and needs the network:

    AGENTIC_WHEEL_ACCEPTANCE=1 conda run -n agenticcli python -m pytest \
        tests/examples/test_deepagents_wheel.py -m wheel -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from tests.examples.console_smoke import STEP_TIMEOUT, child_env

_OPT_IN = os.environ.get("AGENTIC_WHEEL_ACCEPTANCE") == "1"

pytestmark = [
    pytest.mark.wheel,
    pytest.mark.skipif(
        not _OPT_IN,
        reason=(
            "wheel acceptance builds a wheel and installs its dependencies "
            "(slow + needs network); set AGENTIC_WHEEL_ACCEPTANCE=1 to run it"
        ),
    ),
]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {' '.join(cmd[:4])}…\n"
            f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
        )
    return result


@pytest.fixture(scope="module")
def installed_extra(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """The wheel, installed with ``[deepagents]`` into a throwaway virtualenv."""
    root = tmp_path_factory.mktemp("deepagents-wheel")
    dist = root / "dist"
    env_dir = root / "venv"

    _run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(dist),
         str(_REPO_ROOT)],
        cwd=str(root),  # outside the repository
    )
    wheels = sorted(dist.glob("agentic_cli-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

    venv.EnvBuilder(with_pip=True, symlinks=True).create(env_dir)
    python = env_dir / "bin" / "python"
    _run([str(python), "-m", "pip", "install", "--quiet", f"{wheels[0]}[deepagents]"],
         cwd=str(root))

    return {"python": python, "env_dir": env_dir, "root": root}


def _probe(installed: dict, code: str, tmp_path: Path) -> dict:
    """Run ``code`` in the isolated interpreter; it prints one JSON line."""
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    result = subprocess.run(
        [str(installed["python"]), "-c", code],
        cwd=str(cwd),
        env=child_env(home),
        capture_output=True,
        text=True,
        timeout=STEP_TIMEOUT,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"probe failed ({result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_backend_imports_from_the_isolated_install(installed_extra, tmp_path):
    found = _probe(
        installed_extra,
        "import json, sys, agentic_cli.workflow.deepagents as backend; "
        "import deepagents; "
        "print(json.dumps({'backend': backend.__file__, "
        "'deepagents': deepagents.__version__}))",
        tmp_path,
    )

    backend = Path(found["backend"]).resolve()
    assert backend.is_relative_to(installed_extra["env_dir"].resolve()), (
        f"the backend resolved to {backend}, outside the isolated venv"
    )
    assert not backend.is_relative_to(_REPO_ROOT), (
        f"the backend resolved into the repository checkout: {backend}"
    )
    # The extra's pin: deepagents>=0.7.28,<0.8.
    assert found["deepagents"].startswith("0.7."), found["deepagents"]


#: Builds and initializes a delegating Deep Agents manager in the child, offline:
#: its catalog lists no provider (as the offline suite's conftest arranges), and
#: building chat models sends nothing.
_BUILD_A_MANAGER = """
import asyncio, json

from agentic_cli.config import BaseSettings
from agentic_cli.models import ModelCatalog
from agentic_cli.workflow import AgentConfig, create_workflow_manager_from_settings
from agentic_cli.workflow import model_services


class Unlisted(ModelCatalog):
    async def load(self):
        return None


model_services.build_catalog = lambda settings, endpoints, *, transport=None: Unlisted(
    endpoints=endpoints
)


async def main():
    settings = BaseSettings(
        orchestrator="deepagents", session_store="memory", google_api_key="probe-key",
        default_model="google:gemini-2.5-flash",
    )
    manager = create_workflow_manager_from_settings(
        [AgentConfig(name="lead", prompt="Lead.", sub_agents=["helper"]),
         AgentConfig(name="helper", prompt="Help.", description="Helps.")],
        settings,
    )
    await manager.initialize_services()
    try:
        found = {
            "class": type(manager).__name__,
            "backend": manager.backend_type,
            "initialized": manager.is_initialized,
            "agents": sorted(manager.agent_models),
        }
    finally:
        await manager.cleanup()
    # Last: the manager logs to stdout (nothing configured logging here).
    print(json.dumps(found))


asyncio.run(main())
"""


def test_a_manager_builds_from_a_minimal_config(installed_extra, tmp_path):
    found = _probe(installed_extra, _BUILD_A_MANAGER, tmp_path)

    assert found == {
        "class": "DeepAgentsWorkflowManager",
        "backend": "deepagents",
        "initialized": True,
        "agents": ["helper", "lead"],
    }
