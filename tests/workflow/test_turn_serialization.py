"""One turn at a time per manager, and lifecycle mutation waits for it.

The active session/user identity is a ContextVar, so it is already per-turn.
Everything *else* a turn touches is manager-scoped: the HITL input callback
(``set_input_callback``), the ADK plugins' event buffers (drained by
``_run_and_stream``) and Deep Agents' pending notices and open-call record.
Two overlapping turns would route a permission/HITL answer to the wrong
request and let one invocation drain the other's events, and a cleanup could
tear the backend down mid-stream.

``process()``/``resume_with_job_result()`` therefore hold a turn lock, and
``cleanup()``/``reinitialize()`` take it too. Initialization happens *before*
the turn lock so the two lock orders can never deadlock. Every test runs on
both backends that implement the contract (ADK and Deep Agents).
"""

from __future__ import annotations

import asyncio
import contextlib
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
from agentic_cli.config import get_context_settings  # noqa: E402
from agentic_cli.workflow.service_registry import get_service_registry  # noqa: E402
from agentic_cli.workflow.sessions import get_active_turn  # noqa: E402
from tests.conftest import MockContext  # noqa: E402


class _StoppedAtItsRead(Exception):
    """What a harness's store raises where ``compact`` first reads it."""


class _Harness:
    """A manager whose stream is driven by the test, with no real backend.

    Its turn holds at ``release`` (and may ask a question first), and records
    when it starts and ends in ``events``, and the text each run was given
    in ``told``. ``revive()`` presents a live backend, as an initialization
    would.
    """

    def __init__(self, manager) -> None:
        self.manager = manager
        self.events: list[str] = []
        self.told: list[str] = []
        self.release = asyncio.Event()
        self.entered = asyncio.Event()
        # When set, the stream asks the user a question mid-turn (as a HITL
        # tool would) once it is released.
        self.prompt_mid_turn = False

        async def _ensure() -> None:
            return None

        self.manager._ensure_initialized = _ensure
        self.revive()

    def revive(self) -> None:
        """Present a complete backend: turn admission re-checks it under the lock."""
        raise NotImplementedError

    async def stream(self, session_id: str):
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

    async def drain(self, session_id: str) -> list[str]:
        return [
            e.content
            async for e in self.manager.process("hi", "u", session_id=session_id)
        ]


class _ADKHarness(_Harness):
    """An ADK manager with no real runner."""

    def __init__(self, ctx) -> None:
        manager = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="a", prompt="p")], settings=ctx.settings
        )
        manager._event_processor = SimpleNamespace(model=None)
        super().__init__(manager)

        async def _get_or_create(user_id, session_id):
            return SimpleNamespace(id=session_id, state={}, events=[])

        async def _stream(*, session_id, user_id, new_message, state_delta=None):
            self.told.append(new_message.parts[0].text)
            async for event in self.stream(session_id):
                yield event

        self.manager._get_or_create_session = _get_or_create
        self.manager._run_and_stream = _stream

    def stop_compact_at_its_read(self) -> None:
        """``compact``'s read of the session records itself and stops there."""

        async def _get_session(**kwargs):
            self.events.append(f"compact:{kwargs['session_id']}")
            raise _StoppedAtItsRead

        self.manager._session_service = SimpleNamespace(get_session=_get_session)

    def revive(self) -> None:
        async def _get_session(**kwargs):
            return SimpleNamespace(id=kwargs["session_id"], events=[], state={})

        self.manager._initialized = True
        self.manager._session_service = SimpleNamespace(get_session=_get_session)
        self.manager._runner = SimpleNamespace(name="runner")
        self.manager._root_agent = SimpleNamespace(name="a")


class _DeepAgentsHarness(_Harness):
    """A Deep Agents manager with no compiled agents."""

    def __init__(self, ctx) -> None:
        from agentic_cli.workflow.deepagents import DeepAgentsWorkflowManager

        manager = DeepAgentsWorkflowManager(
            agent_configs=[AgentConfig(name="a", prompt="p")], settings=ctx.settings
        )
        super().__init__(manager)

        async def _run_turn(ref, graph_input):
            self.told.append(graph_input["messages"][0].content)
            async for event in self.stream(ref.session_id):
                yield event

        self.manager._run_turn = _run_turn

    def stop_compact_at_its_read(self) -> None:
        """``compact``'s read of the thread records itself and stops there."""

        async def _aget_state(config):
            self.events.append(f"compact:{config['metadata']['session_id']}")
            raise _StoppedAtItsRead

        self.manager._root = SimpleNamespace(name="root", aget_state=_aget_state)

    def revive(self) -> None:
        async def _exists(ref) -> bool:
            return True

        async def _outside_content(ref) -> str:
            return ""

        self.manager._initialized = True
        self.manager._sessions = SimpleNamespace(
            exists=_exists, outside_content=_outside_content
        )
        self.manager._root = SimpleNamespace(name="root")


@pytest.fixture(params=["adk", "deepagents"])
def backend(request) -> str:
    return request.param


def turn_harness(backend: str):
    if backend == "deepagents":
        pytest.importorskip("deepagents")
    ctx = MockContext(google_api_key="test-key")
    ctx.__enter__()
    harness = _DeepAgentsHarness if backend == "deepagents" else _ADKHarness
    return harness(ctx), ctx


def _job_record() -> SimpleNamespace:
    return SimpleNamespace(
        job_id="j1", session_id="sess-r", user_id="u", call_id="c1",
        call_name="t", tool="t", state=SimpleNamespace(value="succeeded"),
        exit_code=0, error=None,
    )


class TestTurnSerialization:
    async def test_second_turn_waits_for_the_first(self, backend):
        h, ctx = turn_harness(backend)
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

    async def test_running_turn_keeps_its_own_hitl_callback(self, backend):
        """A second consumer's callback must not capture the first turn's prompt.

        The turn lock only serialises ``process()``; callbacks are installed
        *before* it, so a manager-global callback attribute let the second
        consumer answer the first turn's question (and then the first
        consumer's ``clear_input_callback()`` unregistered the second's).
        The callback is therefore context-local.
        """
        h, ctx = turn_harness(backend)
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

    async def test_clearing_one_callback_does_not_unregister_another(self, backend):
        """One consumer tidying up must not unregister a concurrent consumer."""
        h, ctx = turn_harness(backend)
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

    async def test_resume_turn_shares_the_lock(self, backend):
        h, ctx = turn_harness(backend)
        try:
            first = asyncio.create_task(h.drain("sess-a"))
            await asyncio.wait_for(h.entered.wait(), timeout=2)

            async def _drain_resume():
                return [
                    e.content
                    async for e in h.manager.resume_with_job_result(_job_record(), "ok")
                ]

            resume = asyncio.create_task(_drain_resume())
            await asyncio.sleep(0.05)
            assert h.events == ["start:sess-a"], "a resume ran during a user turn"

            h.release.set()
            await first
            assert await resume == ["sess-r"]
            assert h.events == [
                "start:sess-a", "end:sess-a", "start:sess-r", "end:sess-r",
            ]
        finally:
            ctx.__exit__(None, None, None)

    async def test_compact_shares_the_lock(self, backend):
        h, ctx = turn_harness(backend)
        try:
            first = asyncio.create_task(h.drain("sess-a"))
            await asyncio.wait_for(h.entered.wait(), timeout=2)
            h.stop_compact_at_its_read()

            compact = asyncio.create_task(
                h.manager.compact(h.manager.session_ref("sess-c", "u"))
            )
            await asyncio.sleep(0.05)
            assert h.events == ["start:sess-a"], "a compact ran during a user turn"

            h.release.set()
            await first
            with pytest.raises(_StoppedAtItsRead):
                await compact
            assert h.events == ["start:sess-a", "end:sess-a", "compact:sess-c"]
        finally:
            ctx.__exit__(None, None, None)


class TestCancellationReleasesTheLock:
    async def test_cancelled_turn_frees_the_manager(self, backend):
        h, ctx = turn_harness(backend)
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


class TestAnEarlyStopFreesTheManager:
    """A consumer that closes the stream after its first event (the CLI's
    ``aclosing``) ends the turn there, in its own task and context."""

    @pytest.mark.parametrize("entry", ["process", "resume_with_job_result"])
    async def test_closing_after_the_first_event(self, backend, entry):
        h, ctx = turn_harness(backend)
        try:
            h.release.set()
            registry, settings = get_service_registry(), get_context_settings()
            if entry == "process":
                source, session = h.manager.process("hi", "u", session_id="sess-a"), "sess-a"
            else:
                source, session = h.manager.resume_with_job_result(_job_record(), "ok"), "sess-r"

            async with contextlib.aclosing(source) as events:
                async for event in events:
                    inside = (get_active_turn(), get_service_registry())
                    break

            assert event.content == session
            assert inside == (h.manager.session_ref(session, "u"), h.manager._services)
            assert h.events == [f"start:{session}"], "the run went on after the close"
            assert not h.manager._turn_lock.locked()
            assert get_active_turn() is None
            assert get_service_registry() is registry
            assert get_context_settings() is settings
            assert await asyncio.wait_for(h.drain("sess-b"), timeout=2) == ["sess-b"]
        finally:
            ctx.__exit__(None, None, None)


class TestLifecycleWaitsForTurns:
    async def test_cleanup_does_not_run_during_a_turn(self, backend):
        h, ctx = turn_harness(backend)
        try:
            order: list[str] = []
            real_release = h.manager._release_resources

            async def _tracked():
                order.append("cleanup")
                await real_release()

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

    async def test_reinitialize_does_not_run_during_a_turn(self, backend):
        h, ctx = turn_harness(backend)
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

    async def test_turn_after_cleanup_reinitializes_without_deadlock(self, backend):
        """Lock order (lifecycle → turn) must not deadlock a turn that inits."""
        h, ctx = turn_harness(backend)
        try:
            h.release.set()
            await h.manager.cleanup()

            inits: list[int] = []

            async def _ensure() -> None:
                inits.append(1)
                # A real _ensure_initialized rebuilds the backend; admission
                # verifies that it did.
                h.revive()

            h.manager._ensure_initialized = _ensure
            assert await asyncio.wait_for(h.drain("sess-x"), timeout=2) == ["sess-x"]
            assert inits == [1]
        finally:
            ctx.__exit__(None, None, None)
