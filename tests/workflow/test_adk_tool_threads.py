"""On ADK, a synchronous tool runs in a worker thread, off the event loop.

Without a tool thread pool in the run's ``RunConfig``, ADK calls a
synchronous function tool on the event loop, so a ``sandbox_execute`` cell
(two minutes by default) or ``compile_document`` froze the CLI for as long as
it ran: the spinner, the job monitor and Ctrl+C. A config-built app's turns
run with ADK's pool, as Deep Agents runs synchronous tools in LangChain's
executor; a native ADK YAML app's tools are its own, and it keeps ADK's
default. Ctrl+C during such a call: ``tests/integration/test_interrupted_calls.py``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import anthropic
import pytest

pytest.importorskip("google.adk")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools.registry import register_tool  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message  # noqa: E402
from tests.workflow.test_adk_context_cache import _native_yaml  # noqa: E402
from tests.workflow.test_manager_models import route  # noqa: E402

MODEL = "claude-sonnet-4-6"

#: When ``tt_slow_sync`` last ran (by the monotonic clock), and on which thread.
RAN: dict[str, object] = {}


@register_tool(capabilities=EXEMPT, description="Works for a second (a test tool).")
def tt_slow_sync() -> dict:
    """Work for a second without yielding, as a sandbox cell does."""
    RAN["thread"] = threading.current_thread()
    RAN["start"] = time.monotonic()
    time.sleep(1.0)
    RAN["end"] = time.monotonic()
    return {"success": True}


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """A hermetic project (the cwd) and HOME, with a fake Claude key."""
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    for variable in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
    return tmp_path / "project"


@pytest.fixture
def claude(monkeypatch) -> list[dict]:
    """Claude's answers, in order."""
    script: list[dict] = []
    route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(lambda request: script.pop(0)))
    return script


def call(name: str) -> dict:
    return anthropic_message({"type": "tool_use", "id": f"toolu_{name}", "name": name, "input": {}})


def says(text: str) -> dict:
    return anthropic_message({"type": "text", "text": text})


def settings() -> BaseSettings:
    return BaseSettings(session_store="memory", default_model=MODEL)


def configured(*tools) -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(
            name="solo", prompt="Answer.", tools=list(tools), include_state_tools=False,
        )],
        settings=settings(),
    )


async def turn(manager, text: str = "Go.") -> list:
    return [event async for event in manager.process(text, "u", session_id="s")]


async def test_the_loop_runs_while_a_synchronous_tool_works(project, claude):
    claude += [call("tt_slow_sync"), says("Done.")]
    manager = configured(tt_slow_sync)
    ticks: list[float] = []

    async def tick() -> None:
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.05)

    try:
        await manager.initialize_services()
        ticker = asyncio.create_task(tick())
        try:
            await turn(manager)
        finally:
            ticker.cancel()
    finally:
        await manager.cleanup()

    during = [t for t in ticks if RAN["start"] < t < RAN["end"]]
    assert len(during) >= 10, f"the loop ticked {len(during)} times in the tool's second"
    assert RAN["thread"] is not threading.current_thread()


@pytest.mark.parametrize("native", [False, True], ids=["config-built", "native-yaml"])
async def test_every_app_gets_the_tool_thread_pool(project, tmp_path, native):
    """A native ADK YAML app's too: its synchronous tools are the framework's
    (the permission gate denies any other), which the pool is for."""
    manager = (
        GoogleADKWorkflowManager(
            agent_configs=[], settings=settings(), adk_config_path=_native_yaml(tmp_path, MODEL),
        )
        if native
        else configured()
    )
    passed: dict = {}

    async def run_async(**kwargs):
        passed.update(kwargs)
        return
        yield  # an async generator that yields no event

    try:
        await manager.initialize_services()
        manager._runner.run_async = run_async
        await turn(manager)
    finally:
        await manager.cleanup()

    pool = getattr(passed.get("run_config"), "tool_thread_pool_config", None)
    assert pool is not None and pool.max_workers == 8
