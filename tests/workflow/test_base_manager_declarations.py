"""What a minimal backend gets from the base without setting it itself.

``BaseWorkflowManager`` reads the current ``session_id`` (``session_ref``,
``load_session``) and the ``_open_calls`` record (``request_user_input``), and
the CLI's ``/status`` shows ``session_id``. Each backend used to set them, and
the base read them through ``getattr``/``hasattr``: a backend that set neither
silently never adopted the id ``load_session`` promises to adopt, and its
questions were never recorded as open. The base declares both.
"""

from __future__ import annotations

from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.events import UserInputRequest
from agentic_cli.workflow.interrupts import NOT_RUN, OpenCalls
from agentic_cli.workflow.sessions import (
    ToolCallRef,
    reset_active_tool_call,
    set_active_tool_call,
)
from tests.conftest import settings_double


class _Minimal(BaseWorkflowManager):
    """The abstract hooks and nothing else: no ``session_id``, no ``_open_calls``."""

    def __init__(self) -> None:
        super().__init__(agent_configs=[], settings=settings_double(default_user="user"))

    @property
    def backend_type(self) -> str:
        return "minimal"

    def _get_state_tools(self):
        return []

    async def _do_initialize(self) -> None:
        pass

    async def process(self, message, user_id, session_id=None):
        raise NotImplementedError

    async def reinitialize(self, model=None, preserve_sessions=True):
        pass

    async def cleanup(self):
        pass


class _Durable(_Minimal):
    """A backend with durable sessions: it overrides the session hooks."""

    def __init__(self, existing: set[str]) -> None:
        super().__init__()
        self._existing = existing

    async def session_exists(self, session_id, *, user_id=None) -> bool:
        return session_id in self._existing


def _request() -> UserInputRequest:
    return UserInputRequest(request_id="r", tool_name="t", prompt="p")


class TestSessionId:
    def test_it_starts_as_the_default_session(self):
        m = _Minimal()

        assert m.session_id == "default_session"
        assert m.session_ref().session_id == "default_session"

    async def test_load_session_adopts_the_id_without_durable_sessions(self):
        m = _Minimal()

        assert await m.load_session("abc") is False

        assert m.session_id == "abc"
        assert m.session_ref().session_id == "abc"

    async def test_load_session_adopts_the_id_of_a_stored_conversation(self):
        m = _Durable({"abc"})

        assert await m.load_session("abc") is True

        assert m.session_id == "abc"
        assert m.session_ref().session_id == "abc"

    async def test_load_session_adopts_the_id_of_a_new_conversation_too(self):
        m = _Durable(set())

        assert await m.load_session("fresh") is False

        assert m.session_id == "fresh"

    def test_an_explicit_id_wins_over_the_current_one(self):
        m = _Minimal()
        m.session_id = "current"

        assert m.session_ref("other").session_id == "other"
        assert m.session_ref().session_id == "current"


class TestOpenCalls:
    def test_the_base_declares_the_record(self):
        assert isinstance(_Minimal()._open_calls, OpenCalls)

    async def test_a_question_asked_inside_a_tool_call_marks_it_open(self):
        m = _Minimal()
        seen: list = []

        async def callback(request):
            # A cancel here would leave the call as it is now: at its prompt.
            seen.append(m._open_calls.answer("call-1", by_user=True))
            return "yes"

        m.set_input_callback(callback)
        token = set_active_tool_call(ToolCallRef("call-1", "ask_clarification"))
        try:
            assert await m.request_user_input(_request()) == "yes"
        finally:
            reset_active_tool_call(token)

        assert len(seen) == 1 and NOT_RUN in str(seen[0])

    async def test_the_mark_is_gone_once_the_question_is_answered(self):
        m = _Minimal()
        m.set_input_callback(lambda request: _answer("yes"))
        token = set_active_tool_call(ToolCallRef("call-1", "ask_clarification"))
        try:
            await m.request_user_input(_request())
        finally:
            reset_active_tool_call(token)

        assert NOT_RUN not in str(m._open_calls.answer("call-1", by_user=True))


async def _answer(text: str) -> str:
    return text
