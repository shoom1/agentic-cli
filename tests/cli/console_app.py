"""A CLI app for the console tests: ``BaseCLIApp`` over a scripted workflow.

It runs through ``BaseCLIApp.main()``, so it takes the session flags
(``--continue``, ``--resume <id>``) as any app does, and one of its own:
``--store memory`` runs it on the memory store instead of sqlite.

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
- ``warn``: four lines the framework does not print itself, then
  ``Warned.``: a structlog warning on the event loop
  (``console_warning_on_the_loop``), one from ``asyncio.to_thread``
  (``console_warning_in_a_thread``), a standard-library one from a thread
  started without the loop's context, as an executor's is
  (``console_stdlib_warning``), and a Python warning
  (``console_python_warning``).
- ``delegate``: the main agent's sub-agent ``researcher-1`` runs once ("Find
  X.", "X is 42."), then ``The researcher found X.``; ``/agent researcher-1``
  opens its view.
- anything else: ``Echo: <message>``, at once, and the call's usage: a
  4,000-token prompt against a 16,000-token trigger (``ctx 4k/16k (25%)``).

In ``researcher-1``'s view a line is its next run (``run_instance``): ``hold``
says ``Holding.`` and waits for a line given to it (``steer_instance``), then
says ``Steered: <line>``; ``slow`` says ``Working on it.`` and waits until
Ctrl+C stops it (the user's stop: the instance is stopped); anything else
says ``Researched: <line>``. The main agent's notice after the view
(``tell_root``) is answered ``Told: <the notice's first clause>``.

``/compact`` reports that a summary is being written, waits until a file
named ``summarized`` appears, then reports its usage and what it did: 2
messages, 7,400 tokens before and 1,200 after. A session is stored once a
turn ran in it (``session_exists``), as ``/new``'s line asks, and ``/resume``
lists and reopens it: its row (the first prompt, the turns) and its messages,
each turn's prompt and the reply it echoed; ``delete_session`` drops it.

With a file named ``init-fails`` in the working directory at startup, the
first initialization fails, and the next one (a turn retries it) waits until
a file named ``ready`` appears. A ``seed.json`` there holds conversations an
earlier run stored: ``{"<id>": {"messages": [["user", "..."], ["assistant",
"..."]], "hours": 2}}``, last active that many hours ago. With a file named
``open-gate`` there, loading a stored conversation (``--continue`` opening it
once the prompt is up, holding the turn lock) writes a file named ``opening``
and waits until a file named ``open-released`` appears.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog

import agentic_cli.cli.workflow_controller as workflow_controller
from agentic_cli.cli import AppInfo, BaseCLIApp
from agentic_cli.config import BaseSettings
from agentic_cli.workflow import live_events
from agentic_cli.workflow.agents import AgentRuns
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.context import CompactResult
from agentic_cli.workflow.events import InputType, UserInputRequest, WorkflowEvent
from agentic_cli.workflow.sessions import (
    SessionInfo,
    SessionRef,
    reset_active_turn,
    session_title,
    set_active_turn,
)
from agentic_cli.workflow.usage import UsageLedgers


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
        #: The sub-agent runs ``delegate`` and a view make (real records).
        self.agent_runs = AgentRuns()
        self._ledgers = UsageLedgers()
        #: Each instance's latest envelope of a run the user started.
        self._results: dict[tuple[SessionRef, str], dict] = {}
        #: A ``hold`` run's wait for the line given to it.
        self._holding: asyncio.Future[str] | None = None
        self._sessions: set[str | None] = set()
        #: Each stored conversation's messages and when it last changed.
        self._messages: dict[str, list[dict]] = {}
        self._updated: dict[str, datetime] = {}
        seed = Path("seed.json")
        if seed.exists():
            for session_id, stored in json.loads(seed.read_text()).items():
                for role, content in stored["messages"]:
                    self._said(session_id, role, content)
                self._sessions.add(session_id)
                self._updated[session_id] = (
                    datetime.now(timezone.utc) - timedelta(hours=stored.get("hours", 0))
                )

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
        if session_id in self._messages and Path("open-gate").exists():
            Path("opening").touch()
            while not Path("open-released").exists():
                await asyncio.sleep(0.05)
        return False

    async def session_exists(self, session_id, *, user_id=None) -> bool:
        return session_id in self._sessions

    def _row(self, session_id: str) -> SessionInfo:
        prompts = [m["content"] for m in self._messages[session_id] if m["role"] == "user"]
        return SessionInfo(session_id, session_title(prompts[0]), len(prompts),
                           self._updated[session_id])

    async def session_info(self, session_id, *, user_id=None) -> SessionInfo | None:
        return self._row(session_id) if session_id in self._messages else None

    async def list_sessions(self, *, user_id=None, limit=None) -> list[SessionInfo]:
        rows = sorted((self._row(s) for s in self._messages),
                      key=lambda row: row.updated, reverse=True)
        return rows if limit is None else rows[:limit]

    async def recent_messages(self, session_id, limit=20, *, user_id=None) -> list[dict]:
        return self._messages.get(session_id, [])[-limit:]

    async def delete_session(self, session_id, *, user_id=None) -> bool:
        self._sessions.discard(session_id)
        self._updated.pop(session_id, None)
        return self._messages.pop(session_id, None) is not None

    def _said(self, session_id: str, role: str, content: str) -> None:
        self._messages.setdefault(session_id, []).append({"role": role, "content": content})
        self._updated[session_id] = datetime.now(timezone.utc)

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

    # -- sub-agents ------------------------------------------------------------------

    def _observe(self, conversation: SessionRef, event: WorkflowEvent) -> None:
        """Record ``event`` as a run reports it, under its conversation's turn."""
        token = set_active_turn(conversation)
        try:
            self.agent_runs.observe(event)
        finally:
            reset_active_turn(token)

    async def load_agents(self, conversation) -> None:
        pass

    def usage_of(self, conversation):
        return self._ledgers.ledger(conversation)

    def context_budget(self, agent_name=None):
        raise KeyError(agent_name)  # no context managed here

    def instance_result(self, instance, *, session_id=None, user_id=None):
        return self._results.get((self.session_ref(session_id), instance))

    def steer_instance(self, instance, message, *, session_id=None, user_id=None) -> bool:
        if self._holding is None or self._holding.done():
            return False
        self._holding.set_result(message)
        return True

    async def run_instance(self, instance, message, *, session_id=None, user_id=None):
        conversation = self.session_ref(session_id)  # one user here
        ref = self.agent_runs.next_run(conversation, instance, parent_run=None)
        started = WorkflowEvent.agent_run(ref, "started", title=message, input=message,
                                          started_by="user", mode="blocking")
        self._observe(conversation, started)
        yield started
        ended = WorkflowEvent.agent_run(ref, "cancelled", error="stopped by the user",
                                        cancelled_by="user", instance_status="stopped")
        try:
            if message == "hold":
                yield self._instance_text(ref, "Holding.")
                self._holding = asyncio.get_running_loop().create_future()
                text = f"Steered: {await self._holding}"
            elif message == "slow":
                yield self._instance_text(ref, "Working on it.")
                await asyncio.Event().wait()  # until Ctrl+C stops the run
            else:
                text = f"Researched: {message}"
            yield self._instance_text(ref, text)
            ended = WorkflowEvent.agent_run(ref, "completed", result=text, instance_status="idle")
        finally:
            self._observe(conversation, ended)
            self._results[(conversation, instance)] = {
                "success": ended.metadata["status"] == "completed", "agent": instance,
                "run": ref.run, "status": ended.metadata["status"],
                "result": ended.metadata.get("result", ""),
            }
        yield ended

    @staticmethod
    def _instance_text(ref, text: str) -> WorkflowEvent:
        event = WorkflowEvent.text(text)
        event.run = ref
        event.metadata.update(agent=ref.agent, subagent=True)
        return event

    async def tell_root(self, text, *, session_id=None, user_id=None):
        yield WorkflowEvent.text(f"Told: {text.split(';')[0]}")

    async def process(self, message: str, user_id: str, session_id: str | None = None):
        self._sessions.add(session_id)
        self._said(session_id, "user", message)
        if message == "delegate":
            conversation = self.session_ref(session_id)  # as the CLI names it
            ref = self.agent_runs.run_for(conversation, None, "c1", "researcher")
            self._observe(conversation, WorkflowEvent.agent_run(
                ref, "started", title="Find X.", call_id="c1",
                brief={"title": "Find X.", "objective": "Find X."},
                started_by="agent:root", mode="blocking",
            ))
            self._observe(conversation, WorkflowEvent.agent_run(
                ref, "completed", call_id="c1", result="X is 42.", instance_status="idle",
            ))
            yield WorkflowEvent.text("The researcher found X.")
        elif message == "slow":
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
        elif message == "warn":
            log = structlog.get_logger("console_app")
            log.warning("console_warning_on_the_loop")
            await asyncio.to_thread(log.warning, "console_warning_in_a_thread")
            thread = threading.Thread(
                target=logging.getLogger("console_app.library").warning,
                args=("console_stdlib_warning",),
            )
            thread.start()
            thread.join()
            warnings.warn("console_python_warning", UserWarning, stacklevel=1)
            yield WorkflowEvent.text("Warned.")
        else:
            self._said(session_id, "assistant", f"Echo: {message}")
            yield WorkflowEvent.text(f"Echo: {message}")
            usage = WorkflowEvent.llm_usage(model=self.model, prompt_tokens=4_000,
                                            completion_tokens=5)
            usage.metadata.update(agent="assistant", context_trigger=16_000, window_known=True)
            yield usage


class ConsoleApp(BaseCLIApp):
    """The console tests' app: the scripted workflow, on the default store
    (sqlite; the scripted workflow writes nothing to it), or on the memory
    store with ``--store memory``."""

    def __init__(self, store: str = "sqlite") -> None:
        super().__init__(
            app_info=AppInfo(name="Console test", version="0"),
            agent_configs=[AgentConfig(name="assistant", prompt="Help.")],
            settings=BaseSettings(session_store=store),
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--store", choices=("sqlite", "memory"), default="sqlite")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ConsoleApp":
        return cls(store=args.store)


def main() -> None:
    workflow_controller.create_workflow_manager_from_settings = (
        lambda **kwargs: ScriptedManager()
    )
    ConsoleApp.main()  # the session flags: --continue, --resume <id>


if __name__ == "__main__":
    main()
