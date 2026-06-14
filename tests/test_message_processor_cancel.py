"""Tests for MessageProcessor Ctrl+C cancellation (_watch_for_cancel).

thinking_prompt's Ctrl+C binding finishes all thinking boxes (flipping
``ui.is_thinking`` to False) but never cancels the coroutine driving the
workflow. ``_watch_for_cancel`` watches for that transition and cancels the
in-flight task, while leaving the HITL dialog window (``state.in_hitl``) alone.
"""

import asyncio

import pytest

from agentic_cli.cli.message_processor import MessageProcessor, _EventProcessingState


class FakeUI:
    """Minimal stand-in exposing only the public ``is_thinking`` property."""

    def __init__(self, is_thinking: bool = True) -> None:
        self.is_thinking = is_thinking


class TestWatchForCancel:
    async def test_completes_without_cancel(self):
        """Task finishing while boxes are alive returns False (no cancel)."""
        mp = MessageProcessor()
        state = _EventProcessingState()
        ui = FakeUI(is_thinking=True)

        async def work() -> None:
            await asyncio.sleep(0.05)

        task = asyncio.create_task(work())
        cancelled = await mp._watch_for_cancel(task, ui, state)

        assert cancelled is False
        assert task.done() and not task.cancelled()

    async def test_cancels_when_boxes_finished(self):
        """When boxes vanish mid-run (Ctrl+C), the task is cancelled."""
        mp = MessageProcessor()
        state = _EventProcessingState()
        ui = FakeUI(is_thinking=True)
        started = asyncio.Event()

        async def work() -> None:
            started.set()
            await asyncio.sleep(10)  # would hang without cancellation

        task = asyncio.create_task(work())

        async def simulate_ctrl_c() -> None:
            await started.wait()
            await asyncio.sleep(0.15)
            ui.is_thinking = False  # thinking_prompt's finish_all()

        flipper = asyncio.create_task(simulate_ctrl_c())
        cancelled = await mp._watch_for_cancel(task, ui, state)
        await flipper

        assert cancelled is True
        assert task.cancelled()

    async def test_hitl_window_is_not_cancelled(self):
        """No active boxes during a HITL dialog must not trigger a cancel."""
        mp = MessageProcessor()
        state = _EventProcessingState()
        state.in_hitl = True
        ui = FakeUI(is_thinking=False)  # dialog on screen, no boxes

        async def work() -> None:
            await asyncio.sleep(0.25)

        task = asyncio.create_task(work())
        cancelled = await mp._watch_for_cancel(task, ui, state)

        assert cancelled is False
        assert task.done() and not task.cancelled()

    async def test_task_exception_propagates(self):
        """An exception from the task surfaces (for rate-limit retry handling)."""
        mp = MessageProcessor()
        state = _EventProcessingState()
        ui = FakeUI(is_thinking=True)

        async def work() -> None:
            await asyncio.sleep(0.05)
            raise RuntimeError("boom")

        task = asyncio.create_task(work())
        with pytest.raises(RuntimeError, match="boom"):
            await mp._watch_for_cancel(task, ui, state)
