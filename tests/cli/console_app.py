"""A CLI app for the console tests: ``BaseCLIApp`` over a scripted workflow.

``tests/cli/test_console_turns.py`` runs it in a child process over a pty;
without a ``test_`` prefix, pytest does not collect it. Nothing reaches a
provider and no key is needed: the controller builds this script's manager
instead of a backend.

What the workflow answers:

- ``slow``: a thinking line ("Working on it"), then it waits, so a Ctrl+C
  lands mid-turn.
- ``ask``: a tool's free-text question ("What colour?"), then
  ``You said: <answer>``.
- ``hold``: ``Holding.``, then it waits until a file named ``release``
  appears in the working directory, and says ``Released.``; the test creates
  the file, so the turn ends on its own while the test watches the prompt.
- anything else: ``Echo: <message>``, at once.

With a file named ``init-fails`` in the working directory at startup, the
first initialization fails, and the next one (a turn retries it) waits until
a file named ``ready`` appears.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import agentic_cli.cli.workflow_controller as workflow_controller
from agentic_cli.cli import AppInfo, BaseCLIApp
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import InputType, UserInputRequest, WorkflowEvent


class ScriptedManager:
    """What the app's startup, turns and shutdown touch on a workflow manager."""

    backend_type = "scripted"
    model = "scripted:model"
    catalog = None
    agent_models: dict = {}
    job_manager = None
    # Initializations so far, of every manager the controller built.
    initializations = 0

    def __init__(self) -> None:
        self.is_initialized = False
        self._input_callback: Any = None

    async def initialize_services(self) -> None:
        ScriptedManager.initializations += 1
        if Path("init-fails").exists():
            if ScriptedManager.initializations == 1:
                raise RuntimeError("scripted initialization failure")
            while not Path("ready").exists():
                await asyncio.sleep(0.05)
        self.is_initialized = True

    async def reinitialize(self, model=None, preserve_sessions=True) -> None:
        self.is_initialized = True

    async def load_session(self, session_id, user_id=None) -> bool:
        return False

    async def cleanup(self) -> None:
        self.is_initialized = False

    def set_input_callback(self, callback) -> None:
        self._input_callback = callback

    def clear_input_callback(self) -> None:
        self._input_callback = None

    async def process(self, message: str, user_id: str, session_id: str | None = None):
        if message == "slow":
            yield WorkflowEvent.thinking("Working on it")
            await asyncio.Event().wait()  # until Ctrl+C cancels the turn
        elif message == "ask":
            answer = await self._input_callback(
                UserInputRequest(
                    request_id="q",
                    tool_name="ask_clarification",
                    prompt="What colour?",
                    input_type=InputType.TEXT,
                )
            )
            yield WorkflowEvent.text(f"You said: {answer}")
        elif message == "hold":
            yield WorkflowEvent.text("Holding.")
            while not Path("release").exists():
                await asyncio.sleep(0.05)
            yield WorkflowEvent.text("Released.")
        else:
            yield WorkflowEvent.text(f"Echo: {message}")


def main() -> None:
    workflow_controller.create_workflow_manager_from_settings = (
        lambda **kwargs: ScriptedManager()
    )
    app = BaseCLIApp(
        app_info=AppInfo(name="Console test", version="0"),
        agent_configs=[AgentConfig(name="assistant", prompt="Help.")],
        settings=BaseSettings(session_store="memory"),
    )
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
