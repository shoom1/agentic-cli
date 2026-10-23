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
- ``which``: ``Session: <the turn's session id>``.
- anything else: ``Echo: <message>``, at once, and the call's usage: a
  4,000-token prompt against a 16,000-token trigger (``ctx 4k/16k (25%)``).

``/compact`` reports that a summary is being written, waits until a file
named ``summarized`` appears, then reports its usage and what it did: 2
messages, 7,400 tokens before and 1,200 after. A session is stored once a
turn ran in it (``session_exists``), as ``/new``'s line asks.

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
from agentic_cli.workflow import live_events
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.context import CompactResult
from agentic_cli.workflow.events import InputType, UserInputRequest, WorkflowEvent
from agentic_cli.workflow.sessions import SessionRef


class ScriptedManager:
    """What the app's startup, turns and shutdown touch on a workflow manager."""

    backend_type = "scripted"
    model = "scripted:model"
    catalog = None
    agent_models: dict = {}
    job_manager = None
    supports_compact = True
    supports_sessions = True
    # Initializations so far, of every manager the controller built.
    initializations = 0

    def __init__(self) -> None:
        self.is_initialized = False
        self._input_callback: Any = None
        self._sessions: set[str | None] = set()

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

    async def session_exists(self, session_id, *, user_id=None) -> bool:
        return session_id in self._sessions

    async def cleanup(self) -> None:
        self.is_initialized = False

    def set_input_callback(self, callback) -> None:
        self._input_callback = callback

    def clear_input_callback(self) -> None:
        self._input_callback = None

    def session_ref(self, session_id=None, user_id=None) -> SessionRef:
        return SessionRef("console", user_id or "user", session_id or "default_session")

    async def compact(self, session, focus=None) -> CompactResult:
        live_events.report(WorkflowEvent.context_summarizing("assistant"))
        while not Path("summarized").exists():
            await asyncio.sleep(0.05)
        usage = WorkflowEvent.llm_usage(model=self.model, prompt_tokens=7_400,
                                        completion_tokens=300)
        usage.metadata.update(purpose="summarization", agent="assistant")
        live_events.report(usage)
        live_events.report(WorkflowEvent.context_trimmed(
            3, 2, "compact", "assistant", messages_summarized=2,
            tokens_before=7_400, tokens_after=1_200,
        ))
        return CompactResult(messages_summarized=2, tokens_before=7_400, tokens_after=1_200)

    async def process(self, message: str, user_id: str, session_id: str | None = None):
        self._sessions.add(session_id)
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
        elif message == "which":
            yield WorkflowEvent.text(f"Session: {session_id}")
        else:
            yield WorkflowEvent.text(f"Echo: {message}")
            usage = WorkflowEvent.llm_usage(model=self.model, prompt_tokens=4_000,
                                            completion_tokens=5)
            usage.metadata.update(agent="assistant", context_trigger=16_000, window_known=True)
            yield usage


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
