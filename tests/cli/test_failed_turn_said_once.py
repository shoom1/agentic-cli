"""The CLI says a failed turn once, from its ``ERROR`` event, on both backends.

A turn that fails yields one ``ERROR`` (the base ``process()``'s contract).
``MessageProcessor`` renders it as it arrives: a provider's rate limit
(``metadata["rate_limited"]``) as the line the CLI said when the turn raised
one ("Rate limited: … Send the request again once the limit resets."), a
conversation another process changed with what to do next, any other error
as itself. The turn is ``FAILED`` and nothing else says it. A manager that
still raises (an app's own) is handled as before
(``tests/cli/test_turn_retry_safety.py``).

The real managers run on Anthropic's API faked offline
(``tests/workflow/test_failed_turn.py``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("deepagents")

from agentic_cli.cli.message_processor import MessageProcessor, TurnStatus  # noqa: E402
from tests.cli.dialog_fakes import autospec_session, opening_boxes  # noqa: E402
from tests.event_replay import ReplayController  # noqa: E402
from tests.workflow.test_failed_turn import (  # noqa: E402, F401
    BACKENDS,
    BAD_REQUEST,
    RAISED,
    RATE_LIMITED,
    Script,
    anthropic_error,
    hermetic,
    say,
    serve,
    solo,
)

SETTINGS = SimpleNamespace(default_user="u", verbose_thinking=False, dialog_placement="box")
NOT_RETRIED = (
    "\nThe turn was not retried: the backend accepts and persists the input before "
    "the first event, so running it again would duplicate this turn (and repeat any "
    "tool calls it made). Send the request again once the limit resets."
)


async def said_on(manager, text: str = "Hello."):
    """One turn through the CLI's message processor: its result and the UI."""
    ui = opening_boxes(autospec_session())
    result = await MessageProcessor().process(
        text, ReplayController(manager), ui, SETTINGS, session_id="s"
    )
    return result, ui


def errors_said(ui) -> list[str]:
    return [call.args[0] for call in ui.add_error.call_args_list]


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_rate_limit_is_said_once_as_before(hermetic, monkeypatch, backend):
    serve(monkeypatch, Script(RATE_LIMITED), backend)
    manager = solo(backend)
    try:
        result, ui = await said_on(manager)
    finally:
        await manager.cleanup()

    (said,) = errors_said(ui)
    assert said.startswith("Rate limited: ")
    assert said.endswith(anthropic_error(RATE_LIMITED) + NOT_RETRIED)
    ui.add_warning.assert_not_called()
    assert result.status is TurnStatus.FAILED
    assert not result.delivered
    assert "429" in result.error


@pytest.mark.parametrize("backend", BACKENDS)
async def test_another_failure_is_said_once_as_itself(hermetic, monkeypatch, backend):
    serve(monkeypatch, Script(BAD_REQUEST), backend)
    manager = solo(backend)
    try:
        result, ui = await said_on(manager)
    finally:
        await manager.cleanup()

    code = RAISED[backend][400]
    assert errors_said(ui) == [f"{anthropic_error(BAD_REQUEST)} [{code}]"]
    ui.add_warning.assert_not_called()
    assert result.status is TurnStatus.FAILED
    assert result.error == anthropic_error(BAD_REQUEST)


async def test_a_conversation_changed_elsewhere_is_said_once_with_what_to_do(
    hermetic, monkeypatch
):
    """ADK refuses to append to a session another process changed since it
    was read; the turn's error says so, and to send the message again."""
    from google.adk.errors import StaleSessionError

    serve(monkeypatch, Script(say("First."), say("Second.")), "adk")
    manager = solo("adk")
    try:
        await said_on(manager)
        service = manager._session_service
        real = service.append_event

        async def stale(session, event):
            if event.author != "user":
                raise StaleSessionError("The session has been modified in storage.")
            return await real(session, event)

        monkeypatch.setattr(service, "append_event", stale)
        result, ui = await said_on(manager, "Hello again.")
    finally:
        await manager.cleanup()

    assert errors_said(ui) == [
        "Conversation 's' was changed by another process during this turn. "
        "Send your message again to go on from what it holds now."
    ]
    assert result.status is TurnStatus.FAILED
