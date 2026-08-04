"""One turn at a time per manager, and lifecycle mutation waits for it.

The active session/user identity is a ContextVar, so it is already per-turn.
Everything *else* a turn touches is manager-scoped: the HITL input callback
(``set_input_callback``) and the ADK plugins' event buffers (drained by
``_run_and_stream``). Two overlapping turns would route a permission/HITL answer
to the wrong request and let one invocation drain the other's events, and a
cleanup could tear the runner down mid-stream.

``process()``/``resume_with_job_result()`` therefore hold a turn lock, and
``cleanup()``/``reinitialize()`` take it too. Initialization happens *before*
the turn lock so the two lock orders can never deadlock.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")

from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.events import (  # noqa: E402
    EventType,
    UserInputRequest,
    WorkflowEvent,
)
from tests.conftest import MockContext  # noqa: E402


class _Harness:
    """A manager whose stream is driven by the test, with no real ADK runner."""

    def __init__(self, ctx) -> None:
        self.manager = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="a", prompt="p")], settings=ctx.settings
        )
        self.manager._initialized = True
        self.manager._event_processor = SimpleNamespace(model=None)
        # Turn admission re-checks readiness while holding the turn lock, so
        # the double presents a complete backend (runner + agent + sessions).
        self.manager._session_service = SimpleNamespace()
        self.manager._runner = SimpleNamespace(name="runner")
        self.manager._root_agent = SimpleNamespace(name="a")
        self.events: list[str] = []
        self.release = asyncio.Event()
        self.entered = asyncio.Event()
        # When set, the stream asks the user a question mid-turn (as a HITL
        # tool would) once it is released.
        self.prompt_mid_turn = False

        async def _ensure() -> None:
            return None

        async def _get_or_create(user_id, session_id):
            return SimpleNamespace(id=session_id)

        async def _stream(*, session_id, user_id, new_message, run_config):
            self.events.append(f"start:{session_id}")
            self.entered.set()
            await self.release.wait()
            if self.prompt_mid_turn:
                answer = await self.manager.request_user_input(
                    UserInputRequest(
                        request_id=f"req-{session_id}",
                        tool_name="ask_clarification",
                        prompt="which?",
                    )
                )
                self.events.append(f"answer:{session_id}:{answer}")
            yield WorkflowEvent(type=EventType.TEXT, content=session_id)
            self.events.append(f"end:{session_id}")

        self.manager._ensure_initialized = _ensure
        self.manager._get_or_create_session = _get_or_create
        self.manager._run_and_stream = _stream

    async def drain(self, session_id: str) -> list[str]:
        return [
            e.content
            async for e in self.manager.process("hi", "u", session_id=session_id)
        ]


def _harness():
    ctx = MockContext(google_api_key="test-key")
    ctx.__enter__()
    return _Harness(ctx), ctx


class TestTurnSerialization:
    async def test_second_turn_waits_for_the_first(self):
        h, ctx = _harness()
        try:
            first = asyncio.create_task(h.drain("sess-a"))
            await asyncio.wait_for(h.entered.wait(), timeout=2)

            second = asyncio.create_task(h.drain("sess-b"))
            await asyncio.sleep(0.05)

            assert h.events == ["start:sess-a"], "turns overlapped"
            assert not second.done()

            h.release.set()
            assert await first == ["sess-a"]
            assert await second == ["sess-b"]
            assert h.events == [
                "start:sess-a", "end:sess-a", "start:sess-b", "end:sess-b",
            ]
        finally:
            ctx.__exit__(None, None, None)

    async def test_running_turn_keeps_its_own_hitl_callback(self):
        """A second consumer's callback must not capture the first turn's prompt.

        The turn lock only serialises ``process()``; callbacks are installed
        *before* it, so a manager-global callback attribute let the second
        consumer answer the first turn's question (and then the first
        consumer's ``clear_input_callback()`` unregistered the second's).
        The callback is therefore context-local.
        """
        h, ctx = _harness()
        try:
            h.prompt_mid_turn = True
            observed: list[str] = []

            async def _cb_a(request):
                observed.append(f"a:{request.request_id}")
                return "from-a"

            async def _cb_b(request):  # pragma: no cover - must never run
                observed.append(f"b:{request.request_id}")
                return "from-b"

            h.manager.set_input_callback(_cb_a)
            first = asyncio.create_task(h.drain("sess-a"))
            await asyncio.wait_for(h.entered.wait(), timeout=2)

            # A second consumer installs its own callback and starts a turn.
            h.manager.set_input_callback(_cb_b)
            second = asyncio.create_task(h.drain("sess-b"))
            await asyncio.sleep(0.05)
            assert h.events == ["start:sess-a"], "turns overlapped"

            h.release.set()
            await asyncio.gather(first, second)

            assert observed == [
                "a:req-sess-a",
                "b:req-sess-b",
            ], "a turn's prompt was answered by another consumer's callback"
            assert "answer:sess-a:from-a" in h.events
            assert "answer:sess-b:from-b" in h.events
        finally:
            ctx.__exit__(None, None, None)

    async def test_clearing_one_callback_does_not_unregister_another(self):
        """One consumer tidying up must not unregister a concurrent consumer."""
        h, ctx = _harness()
        try:
            ready = asyncio.Event()

            async def _cb(request):
                return "answer"

            h.manager.set_input_callback(_cb)

            async def _consumer():
                await ready.wait()
                return await h.manager.request_user_input(
                    UserInputRequest(
                        request_id="r", tool_name="t", prompt="which?"
                    )
                )

            # Created after the install, so it carries this callback.
            consumer = asyncio.create_task(_consumer())

            # A different consumer finishes its turn and clears *its* callback.
            h.manager.clear_input_callback()

            ready.set()
            assert await asyncio.wait_for(consumer, timeout=2) == "answer"
        finally:
            ctx.__exit__(None, None, None)

    async def test_resume_turn_shares_the_lock(self):
        h, ctx = _harness()
        try:
            record = SimpleNamespace(
                job_id="j1", session_id="sess-r", user_id="u", call_id="c1",
                call_name="t", tool="t", state=SimpleNamespace(value="succeeded"),
                exit_code=0, error=None,
            )

            async def _get_session(**kwargs):
                return SimpleNamespace(id="sess-r")

            h.manager._session_service = SimpleNamespace(get_session=_get_session)

            first = asyncio.create_task(h.drain("sess-a"))
            await asyncio.wait_for(h.entered.wait(), timeout=2)

            async def _drain_resume():
                return [
                    e.content
                    async for e in h.manager.resume_with_job_result(record, "ok")
                ]

            resume = asyncio.create_task(_drain_resume())
            await asyncio.sleep(0.05)
            assert h.events == ["start:sess-a"], "a resume ran during a user turn"

            h.release.set()
            await first
            await resume
        finally:
            ctx.__exit__(None, None, None)


class TestCancellationReleasesTheLock:
    async def test_cancelled_turn_frees_the_manager(self):
        h, ctx = _harness()
        try:
            first = asyncio.create_task(h.drain("sess-a"))
            await asyncio.wait_for(h.entered.wait(), timeout=2)

            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

            h.release.set()
            second = asyncio.create_task(h.drain("sess-b"))
            assert await asyncio.wait_for(second, timeout=2) == ["sess-b"]
        finally:
            ctx.__exit__(None, None, None)


class TestLifecycleWaitsForTurns:
    async def test_cleanup_does_not_run_during_a_turn(self):
        h, ctx = _harness()
        try:
            order: list[str] = []
            real_release = h.manager._release_resources

            async def _tracked(keep_session_service: bool = False):
                order.append("cleanup")
                await real_release(keep_session_service)

            h.manager._release_resources = _tracked

            first = asyncio.create_task(h.drain("sess-a"))
            await asyncio.wait_for(h.entered.wait(), timeout=2)

            cleanup = asyncio.create_task(h.manager.cleanup())
            await asyncio.sleep(0.05)
            assert order == [], "cleanup tore the backend down mid-turn"

            h.release.set()
            await first
            await cleanup
            assert order == ["cleanup"]
        finally:
            ctx.__exit__(None, None, None)

    async def test_reinitialize_does_not_run_during_a_turn(self):
        h, ctx = _harness()
        try:
            order: list[str] = []

            async def _init(validate: bool = True):
                order.append("reinit")

            h.manager._initialize_locked = _init
            h.manager._reset_model = lambda model: None

            first = asyncio.create_task(h.drain("sess-a"))
            await asyncio.wait_for(h.entered.wait(), timeout=2)

            reinit = asyncio.create_task(h.manager.reinitialize())
            await asyncio.sleep(0.05)
            assert order == [], "reinitialize ran under an active turn"

            h.release.set()
            await first
            await reinit
            assert order == ["reinit"]
        finally:
            ctx.__exit__(None, None, None)

    async def test_turn_after_cleanup_reinitializes_without_deadlock(self):
        """Lock order (lifecycle → turn) must not deadlock a turn that inits."""
        h, ctx = _harness()
        try:
            h.release.set()
            await h.manager.cleanup()

            inits: list[int] = []

            async def _ensure() -> None:
                inits.append(1)
                # A real _ensure_initialized rebuilds the backend; admission
                # verifies that it did.
                h.manager._initialized = True
                h.manager._session_service = SimpleNamespace()
                h.manager._runner = SimpleNamespace(name="runner")
                h.manager._root_agent = SimpleNamespace(name="a")

            h.manager._ensure_initialized = _ensure
            assert await asyncio.wait_for(h.drain("sess-x"), timeout=2) == ["sess-x"]
            assert inits == [1]
        finally:
            ctx.__exit__(None, None, None)
