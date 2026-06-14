"""Reusable fakes + (de)serialization for replaying WorkflowEvent streams.

This is plumbing shared by tests; it defines no test classes, so pytest does
not collect it.

Two uses:

- **Deterministic render tests** (``test_message_processor_render.py``): feed a
  known list of ``WorkflowEvent`` through ``MessageProcessor.process()`` with a
  ``RecordingSession`` in place of the real ``ThinkingPromptSession``, then
  assert on exactly what the UI was asked to render. No LLM, runs in normal CI.

- **Record / replay bridge**: a live scenario test can dump a *real* event
  stream (``events_to_dicts`` → JSON) and a render test can replay it later
  (``events_from_dicts``), so "how the CLI renders this run" becomes a
  deterministic snapshot decoupled from the non-deterministic agent run that
  produced it.
"""

from __future__ import annotations

from typing import Any

from agentic_cli.workflow.events import EventType, WorkflowEvent


class RecordingThinkingContext:
    """Stand-in for thinking_prompt's ``ThinkingContext`` that records calls.

    Every method appends to the owning session's ``calls`` list so tests can
    assert on box lifecycle (append/clear/set_title/finish) in order.
    """

    def __init__(self, session: "RecordingSession", label: str) -> None:
        self._session = session
        self.label = label
        self.finished = False

    def append(self, text: str) -> None:
        self._session.calls.append(("ctx_append", self.label, text))

    def clear(self) -> None:
        self._session.calls.append(("ctx_clear", self.label))

    def set_title(self, title: str) -> None:
        self.label = title
        self._session.calls.append(("ctx_title", self.label, title))

    def finish(self, **kwargs: Any) -> None:
        self.finished = True
        self._session.calls.append(("ctx_finish", self.label, kwargs))


class RecordingSession:
    """Minimal ``ThinkingPromptSession`` stand-in that records render calls.

    Implements only the surface ``MessageProcessor`` touches. ``calls`` is an
    ordered list of tuples (``kind``, ...payload) you can assert against, plus a
    few convenience query helpers.
    """

    def __init__(self, is_thinking: bool = True) -> None:
        self.calls: list[tuple] = []
        # MessageProcessor's Ctrl+C watcher reads this; keep it True so a fast
        # replay is never mistaken for the user cancelling.
        self.is_thinking = is_thinking

    # --- thinking boxes -----------------------------------------------------
    def start_thinking(
        self,
        status_or_title: Any = None,
        *,
        title: str | None = None,
        content_format: str = "plain",
        order: int = 0,
        **kwargs: Any,
    ) -> RecordingThinkingContext:
        # The events box passes the status callback positionally (no title);
        # the task box passes title= as a keyword.
        label = title if title is not None else "events"
        self.calls.append(
            ("start_thinking", label, {"order": order, "content_format": content_format})
        )
        return RecordingThinkingContext(self, label)

    # --- output methods -----------------------------------------------------
    def add_response(self, content: str, *, markdown: bool = False) -> None:
        self.calls.append(("response", content, markdown))

    def add_rich(self, renderable: Any) -> None:
        self.calls.append(("rich", str(renderable)))

    def add_message(self, role: str, content: str) -> None:
        self.calls.append(("message", role, content))

    def add_warning(self, content: str) -> None:
        self.calls.append(("warning", content))

    def add_error(self, content: str) -> None:
        self.calls.append(("error", content))

    def add_success(self, content: str) -> None:
        self.calls.append(("success", content))

    def set_status(self, text: str) -> None:
        self.calls.append(("status", text))

    # --- query helpers for assertions --------------------------------------
    def kinds(self) -> list[str]:
        return [c[0] for c in self.calls]

    def of(self, kind: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == kind]

    def responses(self) -> list[str]:
        return [c[1] for c in self.of("response")]

    def rich(self) -> list[str]:
        return [c[1] for c in self.of("rich")]

    def warnings(self) -> list[str]:
        return [c[1] for c in self.of("warning")]

    def errors(self) -> list[str]:
        return [c[1] for c in self.of("error")]


class ReplayWorkflow:
    """Workflow-manager stand-in that yields a fixed list of events."""

    def __init__(self, events: list[WorkflowEvent]) -> None:
        self._events = list(events)
        self.input_callback = None

    def set_input_callback(self, callback: Any) -> None:
        self.input_callback = callback

    def clear_input_callback(self) -> None:
        self.input_callback = None

    async def process(self, message: str, user_id: str, session_id: str | None = None):
        for event in self._events:
            yield event


class ReplayController:
    """``WorkflowController`` stand-in wrapping a :class:`ReplayWorkflow`."""

    def __init__(self, workflow: ReplayWorkflow) -> None:
        self.workflow = workflow
        self.status_bar_updates = 0

    async def ensure_initialized(self, ui: Any = None) -> bool:
        return True

    @property
    def is_ready(self) -> bool:
        return True

    def update_status_bar(self, ui: Any) -> None:
        self.status_bar_updates += 1


def events_to_dicts(events: list[WorkflowEvent]) -> list[dict]:
    """Serialize a WorkflowEvent stream to JSON-able dicts (drops timestamps)."""
    return [
        {"type": e.type.value, "content": e.content, "metadata": e.metadata}
        for e in events
    ]


def events_from_dicts(dicts: list[dict]) -> list[WorkflowEvent]:
    """Rebuild a WorkflowEvent stream from :func:`events_to_dicts` output."""
    return [
        WorkflowEvent(
            type=EventType(d["type"]),
            content=d.get("content", ""),
            metadata=d.get("metadata", {}),
        )
        for d in dicts
    ]
