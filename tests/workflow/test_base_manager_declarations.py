"""What a minimal backend gets from the base without setting it itself, and
what the base's documentation says of it.

``BaseWorkflowManager`` reads the current ``session_id`` (``session_ref``,
``load_session``) and the ``_open_calls`` record (``request_user_input``), and
the CLI's ``/status`` shows ``session_id``. Each backend used to set them, and
the base read them through ``getattr``/``hasattr``: a backend that set neither
silently never adopted the id ``load_session`` promises to adopt, and its
questions were never recorded as open. The base declares both.

The documented constructor ``app_name`` and ``initialize_services``' locks are
pinned at the end.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.events import UserInputRequest
from agentic_cli.workflow.interrupts import NOT_RUN, OpenCalls
from agentic_cli.workflow.sessions import (
    ToolCallRef,
    reset_active_tool_call,
    set_active_tool_call,
)
from agentic_cli.workflow.service_registry import PERMISSION_ENGINE
from tests.conftest import settings_double


class _Hooks(BaseWorkflowManager):
    """The abstract hooks and nothing else: no constructor, no state."""

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


class _Minimal(_Hooks):
    """A backend that sets neither ``session_id`` nor ``_open_calls``."""

    def __init__(self, app_name: str | None = None) -> None:
        super().__init__(
            agent_configs=[],
            settings=settings_double(default_user="user"),
            app_name=app_name,
        )


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


def _base(**kwargs):
    return dict(agent_configs=[], settings=settings_double(default_user="user"), **kwargs)


class TestBackendsThatDeclareThemselves:
    """A backend that provides ``session_id`` or ``_open_calls`` its own way
    keeps what it provides: the base declares a default, never overwrites."""

    async def test_values_set_before_the_base_constructor_survive_it(self):
        class SetsBeforeSuper(_Hooks):
            def __init__(self) -> None:
                self.session_id = "chosen-by-backend"
                self.record = self._open_calls = OpenCalls()
                super().__init__(**_base())

        m = SetsBeforeSuper()

        assert m.session_id == "chosen-by-backend"
        assert m.session_ref().session_id == "chosen-by-backend"
        assert m._open_calls is m.record
        await m.load_session("resumed")
        assert m.session_id == "resumed"

    async def test_a_class_level_session_id_is_the_default(self):
        class ClassDefault(_Hooks):
            session_id = "main"

            def __init__(self) -> None:
                super().__init__(**_base())

        m = ClassDefault()

        assert m.session_id == "main"
        assert m.session_ref().session_id == "main"
        await m.load_session("resumed")
        assert m.session_id == "resumed"
        assert ClassDefault.session_id == "main"

    async def test_a_session_id_property_whose_setter_needs_state_set_after_the_base_constructor(
        self,
    ):
        class Conversation:
            id = "conv-0"

        class PropertyBacked(_Hooks):
            def __init__(self) -> None:
                super().__init__(**_base())
                self.conversation = Conversation()

            @property
            def session_id(self) -> str:
                return self.conversation.id

            @session_id.setter
            def session_id(self, value: str) -> None:
                self.conversation.id = value

        m = PropertyBacked()

        assert m.session_ref().session_id == "conv-0"
        await m.load_session("resumed")
        assert m.conversation.id == "resumed"
        assert m.session_ref().session_id == "resumed"

    async def test_values_set_after_the_base_constructor_win(self):
        class SetsAfterSuper(_Hooks):
            def __init__(self) -> None:
                super().__init__(**_base())
                self.session_id = "after"
                self.record = self._open_calls = OpenCalls()

        m = SetsAfterSuper()

        assert m.session_ref().session_id == "after"
        assert m._open_calls is m.record

    def test_an_open_calls_provided_by_the_class_is_kept(self):
        class Provided(_Hooks):
            record = OpenCalls()
            _open_calls = record

            def __init__(self) -> None:
                super().__init__(**_base())

        assert Provided()._open_calls is Provided.record

    def test_each_manager_has_its_own_record_and_session_id(self):
        first, second = _Minimal(), _Minimal()

        assert first._open_calls is not second._open_calls

    async def test_adopting_an_id_leaves_other_managers_alone(self):
        first, second = _Minimal(), _Minimal()

        await first.load_session("resumed")

        assert first.session_id == "resumed"
        assert second.session_id == "default_session"
        assert BaseWorkflowManager.session_id == "default_session"

    def test_a_manager_built_without_its_constructor_has_the_default_id(self):
        with patch.object(BaseWorkflowManager, "__abstractmethods__", set()):
            m = BaseWorkflowManager.__new__(BaseWorkflowManager)

        assert m.session_id == "default_session"


class TestAppName:
    def test_it_defaults_to_the_settings_app_name(self):
        m = _Minimal()

        assert m.app_name == "test-app"
        assert m.session_ref().app_name == "test-app"

    def test_the_constructor_s_app_name_names_the_conversations_not_the_services(self):
        m = _Minimal(app_name="other")

        assert m.app_name == "other"
        assert m.session_ref("s").app_name == "other"
        # What the app keeps and builds for the tools is named by the settings.
        engine = m._build_services()[PERMISSION_ENGINE]
        assert engine._ctx.app_name == "test-app"


class TestInitializeServicesLocks:
    async def test_it_does_not_wait_for_a_running_turn(self):
        """It takes the lifecycle lock only: a turn initializes before it takes
        the turn lock, so initializing while one is held must not block."""
        m = _Minimal()

        async with m._turn_lock:
            await asyncio.wait_for(m.initialize_services(validate=False), timeout=5)

        assert m.is_initialized
