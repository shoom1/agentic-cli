"""Message processing for agentic CLI applications.

This module provides:
- MessageProcessor: Handles user message processing through workflow
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

from thinking_prompt import ButtonConfig, Dialog, TextField

from agentic_cli.logging import Loggers, bind_context

if TYPE_CHECKING:
    from agentic_cli.cli.usage_tracker import UsageTracker
    from agentic_cli.config import BaseSettings
    from agentic_cli.cli.workflow_controller import WorkflowController
    from agentic_cli.workflow import WorkflowEvent
    from agentic_cli.workflow.events import UserInputRequest
    from thinking_prompt import ThinkingBox, ThinkingPromptSession


logger = Loggers.cli()

# The line that says the request was stopped. Ctrl+C stops the request the
# agent is processing (thinking-prompt 0.4 cancels the input handler running
# it), and this is 0.4's own line for that. The boxes the request's
# processing opened end with it, out of the transcript, so 0.4 has none left
# to end and says nothing; the request says this line in its place, once.
CANCELLED_LINE = "Operation cancelled..."

# Under a free-text question's answer field. thinking-prompt opens a text
# field for viewing: keys typed at it before Enter are dropped.
ANSWER_HINT = "Press Enter to type your answer, Enter again to keep it"


def _richify_task_display(plain: str) -> str:
    """Convert plain task checklist to rich ANSI-colored display.

    Task box uses full-color styling (not dim grey like thinking/tool boxes):
    - Completed: green ✓, dim text (done = de-emphasised)
    - In-progress: yellow ▸, white text (active = prominent)
    - Pending: white ○, white text (upcoming = visible)
    - Cancelled: red ✗, dim text (irrelevant = de-emphasised)
    - Section headers: bold white
    """
    from thinking_prompt import rich_to_ansi

    def _esc(text: str) -> str:
        """Escape Rich markup brackets in task text."""
        return text.replace("[", "\\[")

    lines = []
    for line in plain.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("[✓]"):
            lines.append(f"{indent}[green]✓[/green] [dim]{_esc(stripped[4:])}[/dim]")
        elif stripped.startswith("[▸]"):
            lines.append(
                f"{indent}[yellow]▸[/yellow] [white]{_esc(stripped[4:])}[/white]"
            )
        elif stripped.startswith("[ ]"):
            lines.append(
                f"{indent}[white]○[/white] [white]{_esc(stripped[4:])}[/white]"
            )
        elif stripped.startswith("[-]"):
            lines.append(
                f"{indent}[red]✗[/red] [dim]{_esc(stripped[4:])}[/dim]"
            )
        else:
            lines.append(f"[bold white]{_esc(line)}[/bold white]")
    return rich_to_ansi("\n".join(lines))


# === Turn results ===


class TurnStatus(str, Enum):
    """Outcome of one processed turn.

    - ``COMPLETED`` — the event stream ran to the end.
    - ``CANCELLED`` — the user pressed Ctrl+C. ``process()`` no longer returns
      it: a cancelled turn raises ``asyncio.CancelledError``.
    - ``FAILED`` — the workflow raised; ``TurnResult.error`` says why.
    - ``UNAVAILABLE`` — the turn never started (workflow not initialized, or a
      job whose originating conversation is gone).
    """

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class TurnResult:
    """Explicit outcome of ``MessageProcessor.process``/``process_resume``.

    Callers that own durable state (the background-job coordinator) must key
    off ``delivered`` rather than "the coroutine returned without raising" —
    that is what let a failed resume be recorded as delivered and dropped.
    """

    status: TurnStatus
    error: str | None = None
    # True once the turn produced output or ran a tool, i.e. a replay of the
    # whole turn would duplicate side effects.
    partial: bool = False

    @property
    def delivered(self) -> bool:
        """True only when the turn ran to completion."""
        return self.status is TurnStatus.COMPLETED


# === Event Processing State ===


@dataclass
class _EventProcessingState:
    """Mutable state shared across event handlers during process()."""

    usage_tracker: "UsageTracker | None" = field(default=None, repr=False)
    workflow_controller: "WorkflowController | None" = field(default=None, repr=False)
    status_line: str = "Processing..."
    thinking_started: bool = False
    thinking_content: list[str] = field(default_factory=list)
    response_content: list[str] = field(default_factory=list)
    # Set once the turn emits assistant text or runs a tool: the turn is then
    # observably partial, which the caller records on a failed TurnResult (and
    # a cancel logs). (It is not a retry gate — the harness never replays a
    # turn.)
    side_effects_seen: bool = False
    # First non-recoverable ERROR event seen. The stream may keep going (the
    # backend decides), but the turn's outcome is FAILED, never delivered.
    fatal_error: str | None = None
    # Prevents double-counting when LangGraph emits both CONTEXT_TRIMMED and LLM_USAGE
    _context_trimmed_this_invocation: bool = False

    def get_status(self) -> str:
        """Return the current status line for the events thinking box."""
        return self.status_line


# === Message Processor ===


class MessageProcessor:
    """Processes user messages through the workflow.

    Handles:
    - Event stream processing from workflow
    - Thinking box state management
    - Tool result display
    - User input prompting during workflow
    - Message history tracking

    Example:
        processor = MessageProcessor()

        await processor.process(
            message="Hello",
            workflow_controller=controller,
            ui=session,
            settings=settings,
        )

        # Access history
        for msg in processor.history.get_all():
            print(f"{msg.message_type}: {msg.content}")
    """

    def __init__(self) -> None:
        """Initialize the message processor."""
        self._last_task_progress: str | None = None
        self._task_box: "ThinkingBox | None" = None
        self._last_task_content: str | None = None

    def _live_task_box(self) -> "ThinkingBox | None":
        """The task box, unless there is none or it has ended.

        A box can end without us: the session ends the boxes a handler leaves
        open, and Ctrl+C at an idle prompt clears every box. A write to an
        ended box is dropped (with a logged warning), so it counts as none.
        """
        box = self._task_box
        if box is None or box.is_finished:
            return None
        return box

    @staticmethod
    async def _workflow_ready(
        workflow_controller: "WorkflowController", ui: "ThinkingPromptSession"
    ) -> bool:
        """Wait for the workflow to be ready, as ``ensure_initialized`` does.

        Ctrl+C here stops a request still waiting for the workflow to start.
        The controller's "Waiting for initialization..." box ends with the
        wait, out of the transcript, and ``CANCELLED_LINE`` says the request
        was stopped.
        """
        try:
            return await workflow_controller.ensure_initialized(ui)
        except asyncio.CancelledError:
            ui.add_message("system", CANCELLED_LINE)
            logger.info("message_cancelled_by_user", waiting_for="initialization")
            raise

    async def process(
        self,
        message: str,
        workflow_controller: "WorkflowController",
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        usage_tracker: "UsageTracker | None" = None,
        session_id: str | None = None,
    ) -> TurnResult:
        """Process a user message through the workflow.

        Args:
            message: User message to process
            workflow_controller: Controller managing workflow lifecycle
            ui: UI session for output
            settings: Application settings
            usage_tracker: Optional tracker for accumulating LLM token usage
            session_id: Session to run the turn in. Passed explicitly so every
                turn targets the app's durable session rather than the manager's
                fallback (which would collapse unnamed runs into one session).

        Returns:
            The turn's outcome; ``TurnResult.delivered`` is True only when the
            event stream ran to completion.

        Raises:
            asyncio.CancelledError: The user pressed Ctrl+C, which stops the
                request: thinking-prompt cancels the input handler running it.
        """
        # Wait for initialization if needed
        if not await self._workflow_ready(workflow_controller, ui):
            ui.add_error(
                "Cannot process message - workflow not initialized. "
                "Please check your API keys (GOOGLE_API_KEY or ANTHROPIC_API_KEY)."
            )
            return TurnResult(
                TurnStatus.UNAVAILABLE, error="workflow not initialized"
            )

        bind_context(user_id=settings.default_user)
        logger.info("handling_message", message_length=len(message))

        def _source(workflow):
            return workflow.process(
                message=message,
                user_id=settings.default_user,
                session_id=session_id,
            )

        return await self._run_turn(
            _source, workflow_controller, ui, settings, usage_tracker
        )

    async def process_resume(
        self,
        record,
        workflow_controller: "WorkflowController",
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        usage_tracker: "UsageTracker | None" = None,
    ) -> TurnResult:
        """Resume the agent with a finished long-running job's result.

        Streams ``workflow.resume_with_job_result(record)`` through the exact
        same rendering path as a user turn (events box, tool results, token
        accounting, Ctrl+C).

        Args:
            record: The terminal JobRecord to resume from.
            workflow_controller: Controller managing workflow lifecycle.
            ui: UI session for output.
            settings: Application settings.
            usage_tracker: Optional tracker for accumulating LLM token usage.

        Returns:
            The turn's outcome. ``UNAVAILABLE`` when the workflow is not ready
            or the originating conversation is gone; the caller must not record
            the job as delivered unless ``TurnResult.delivered`` is True.

        Raises:
            asyncio.CancelledError: Ctrl+C stopped the request, as it stops a
                user turn.
        """
        if not await self._workflow_ready(workflow_controller, ui):
            return TurnResult(
                TurnStatus.UNAVAILABLE, error="workflow not initialized"
            )

        workflow = workflow_controller.workflow
        bind_context(user_id=settings.default_user)
        icon = "✓" if record.state.value == "succeeded" else "✗"

        # Resume only if the backend supports it AND the originating conversation
        # is still available (with session_store='memory' it is gone after a
        # restart, on ADK and Deep Agents alike, and a deleted one is gone too).
        # Otherwise surface a notice — the result stays reachable by job id.
        resumable = hasattr(
            workflow, "resume_with_job_result"
        ) and await workflow.can_resume(record)
        if not resumable:
            logger.info(
                "job_resume_not_resumable",
                job_id=record.job_id,
                backend=getattr(workflow, "backend_type", "?"),
            )
            ui.add_message(
                "system",
                f"{icon} Background job '{record.name}' finished "
                f"({record.state.value}) while its conversation was unavailable "
                f"— fetch the result with /jobs {record.job_id}.",
            )
            return TurnResult(
                TurnStatus.UNAVAILABLE,
                error="originating conversation is no longer available",
            )

        logger.info("resuming_job", job_id=record.job_id, state=record.state.value)
        ui.add_message(
            "system",
            f"↻ Background job '{record.name}' finished ({icon} {record.state.value}) "
            "— resuming.",
        )

        def _source(wf):
            return wf.resume_with_job_result(record)

        return await self._run_turn(
            _source, workflow_controller, ui, settings, usage_tracker
        )

    async def _run_turn(
        self,
        source_factory,
        workflow_controller: "WorkflowController",
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        usage_tracker: "UsageTracker | None" = None,
    ) -> TurnResult:
        """Drive one turn from an event-source factory through the UI.

        Shared by ``process`` (user message) and ``process_resume`` (job
        result). ``source_factory(workflow)`` returns the WorkflowEvent async
        generator to consume; everything else (events box, HITL callback,
        Ctrl+C cancel, rate-limit retry, token accounting) is identical.

        The event source is invoked **exactly once**. The harness never replays
        a turn: ADK accepts and persists the input (the user message, or a
        resumed ``FunctionResponse``) into the session while setting up the
        invocation — before the first event is yielded — so there is no point
        at which re-running ``source_factory`` is side-effect free. Even a 429
        on the first model call leaves that input in the session, and a replay
        would duplicate the turn and repeat any tool calls it made.

        Retrying belongs at the provider/model-client boundary, which can
        prove nothing was accepted: on ADK, Gemini's ``HttpRetryOptions``
        (configured in ``workflow/adk/model_factory.py``) retry transient 5xx
        inside the client, and the Anthropic and OpenAI clients retry per
        ``retry_max_attempts``.
        A rate limit that still surfaces here fails the turn explicitly.

        A cancel of this coroutine is the user's Ctrl+C, which stops the
        request the agent is processing: thinking-prompt cancels the input
        handler's task, and every request runs inside the handler (user
        turns, then job-resume turns). The cancel reaches the consumer task,
        which is awaited to completion **before** the HITL callback and turn
        state are torn down — otherwise a tool could still be running, and
        asking for input, with nothing left to answer it. The boxes the
        request's processing opened end with it, out of the transcript;
        ``CANCELLED_LINE`` says once that the request was stopped; and the
        cancel is re-raised.

        Errors reported as ``EventType.ERROR`` are rendered as they arrive. A
        recoverable one is a warning and the stream decides the outcome; a
        non-recoverable one makes the turn ``FAILED`` (``delivered`` False)
        even if the stream then ends normally, so a caller owning durable
        state does not record a failed delivery as delivered.

        Returns:
            The turn's outcome as a :class:`TurnResult`. ``partial`` reports
            whether the turn had already emitted output or run a tool.
        """
        state = _EventProcessingState(
            usage_tracker=usage_tracker,
            workflow_controller=workflow_controller,
        )
        workflow = workflow_controller.workflow
        dispatch = self._get_event_dispatch()

        # Cold start: recreate the task box from the tasks it showed when the
        # last turn ended it (see the finally below), drawn as it drew them.
        if self._live_task_box() is None and self._last_task_progress is not None:
            self._task_box = ui.start_thinking(
                title="Tasks",
                order=100,
                content_format="ansi",
                overflow="head",  # a long list keeps its first tasks in view
            )
            self._task_box.append(_richify_task_display(self._last_task_progress))

        # Events thinking box context — tracks the per-invocation events box.
        # This is a callback-driven box (no append/clear), only the callback
        # (state.get_status) drives its display.
        events_ctx: "ThinkingBox | None" = None

        def _finish_events_box() -> None:
            """Finish the events box if it is open. Idempotent by construction.

            Every path goes through this, so a box is finished exactly once:
            reopening one that was already closed (or closing one twice) is
            visible to the user as a stray or duplicated panel. Its status
            lines stay out of the transcript.
            """
            nonlocal events_ctx
            if state.thinking_started and events_ctx is not None:
                events_ctx.finish(add_to_history=False, echo_to_console=False)
            events_ctx = None
            state.thinking_started = False

        def _open_events_box() -> None:
            nonlocal events_ctx
            events_ctx = ui.start_thinking(state.get_status, content_format="ansi")
            state.thinking_started = True

        # Set up direct callback so HITL tools can prompt the user without
        # deadlocking the workflow runner.
        async def _handle_input(request: "UserInputRequest") -> str:
            _finish_events_box()
            # If the dialog raises, the turn is unwinding (cancelled, or the
            # dialog failed): reopening the events box then would leave a
            # panel on screen that nothing downstream will ever finish.
            response = await self._prompt_user_input(request, ui)
            _open_events_box()
            return response

        # The callback is context-local, so installing and clearing it here
        # affects only this turn; no token round-trip is needed (and a manager
        # implementing the older no-argument clear stays compatible).
        workflow.set_input_callback(_handle_input)
        result = TurnResult(TurnStatus.FAILED, error="turn did not run")
        proc_task: "asyncio.Task[None] | None" = None
        try:
            try:
                _open_events_box()

                # Consume the event stream in a task of its own. Awaiting it
                # passes a cancel of this coroutine (Ctrl+C) on to it, and
                # returns once it has unwound.
                async def _consume() -> None:
                    async for event in source_factory(workflow):
                        handler = dispatch.get(event.type)
                        if handler is not None:
                            await handler(
                                self, event, state, ui, settings, workflow
                            )

                proc_task = asyncio.create_task(_consume())
                await proc_task

                # Finish events box only (don't add status to history)
                _finish_events_box()

                # Ensure final token counts are reflected in status bar
                workflow_controller.update_status_bar(ui)

                if state.fatal_error is not None:
                    # The stream ended, but it reported a failure. Never
                    # "delivered".
                    logger.info("turn_failed_by_error_event")
                    result = TurnResult(
                        TurnStatus.FAILED,
                        error=state.fatal_error,
                        partial=state.side_effects_seen,
                    )
                else:
                    logger.debug("message_handled_successfully")
                    result = TurnResult(TurnStatus.COMPLETED)

            except asyncio.CancelledError:
                # Ctrl+C stopped the request. The boxes its processing
                # opened end with it, out of the transcript (the session
                # would end them into it, for a repaint to print), and the
                # line says the request was stopped: also with a question on
                # screen, when no box is open and the session says nothing.
                _finish_events_box()
                task_box = self._live_task_box()
                if task_box is not None:
                    task_box.finish(add_to_history=False, echo_to_console=False)
                # The next turn starts without a task box.
                self._task_box = None
                self._last_task_content = None
                ui.add_message("system", CANCELLED_LINE)
                workflow_controller.update_status_bar(ui)
                logger.info("message_cancelled_by_user", partial=state.side_effects_seen)
                raise

            except Exception as e:
                _finish_events_box()

                from agentic_cli.workflow.retry import is_rate_limit_error

                if is_rate_limit_error(e):
                    logger.warning("turn_rate_limited", partial=state.side_effects_seen)
                    ui.add_error(
                        f"Rate limited: {e}\n"
                        "The turn was not retried: the backend accepts and "
                        "persists the input before the first event, so running "
                        "it again would duplicate this turn (and repeat any "
                        "tool calls it made). Send the request again once the "
                        "limit resets."
                    )
                else:
                    ui.add_error(f"Workflow error: {e}")
                result = TurnResult(
                    TurnStatus.FAILED,
                    error=str(e),
                    partial=state.side_effects_seen,
                )
        finally:
            # Settle the consumer *first*: tearing the callback down under a
            # live tool would strand a HITL prompt.
            await self._settle(proc_task)
            # Then close whatever is still open: each path above closes its
            # own, but a BaseException that is not a cancel (KeyboardInterrupt)
            # takes none of them, and a box left open outlives the turn.
            _finish_events_box()
            workflow.clear_input_callback()
            # The task box ends with the turn, out of the transcript: the
            # session would end a box the turn leaves open into it, unseen
            # (with echo_thinking off), for a repaint to print. Cache its
            # plain-text content (not the richified ANSI it shows) for the
            # next turn's cold start.
            task_box = self._live_task_box()
            self._last_task_progress = (
                self._last_task_content if task_box is not None else None
            )
            if task_box is not None:
                task_box.finish(add_to_history=False, echo_to_console=False)
            self._task_box = None
        return result

    @staticmethod
    async def _settle(proc_task: "asyncio.Task[None] | None") -> None:
        """Ensure the consumer task is finished before the turn is torn down.

        A no-op on every path ``_run_turn`` takes today: awaiting the task
        settles it, a cancel included. It is the backstop for a turn torn down
        while its consumer still runs, which would keep consuming the
        workflow generator after its owner is gone.

        Uses ``asyncio.wait`` rather than awaiting the task, so neither the
        task's ``CancelledError`` nor its exception is re-raised out of a
        ``finally`` block — the original outcome must survive.
        """
        if proc_task is None or proc_task.done():
            MessageProcessor._retrieve_exception(proc_task)
            return
        proc_task.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.wait({proc_task})
        MessageProcessor._retrieve_exception(proc_task)

    @staticmethod
    def _retrieve_exception(proc_task: "asyncio.Task[None] | None") -> None:
        """Mark a finished task's exception retrieved (no 'never retrieved' log)."""
        if proc_task is None or not proc_task.done() or proc_task.cancelled():
            return
        with suppress(asyncio.InvalidStateError):
            proc_task.exception()

    async def _prompt_user_input(
        self,
        request: "UserInputRequest",
        ui: "ThinkingPromptSession",
    ) -> str:
        """Prompt user for input requested by a tool.

        A choice is a dropdown and a confirmation a yes/no question; any
        other request is free text, in a dialog with one text field and OK
        and Cancel.

        Args:
            request: The user input request from the tool.
            ui: UI session for prompting.

        Returns:
            User's response string ("" for a cancelled free-text question).
        """
        from agentic_cli.workflow.events import InputType

        tool_name = request.tool_name or "Tool"

        if request.input_type == InputType.CHOICE and request.choices:
            result = await ui.dropdown_dialog(
                title=f"{tool_name} - Input Required",
                text=request.prompt,
                options=request.choices,
                default=request.default,
            )
            return result or (request.default or "")

        elif request.input_type == InputType.CONFIRM:
            result = await ui.yes_no_dialog(
                title=f"{tool_name} - Confirmation",
                text=request.prompt,
            )
            return "yes" if result else "no"

        else:
            dialog = Dialog(
                title=f"{tool_name} - Input Required",
                body=request.prompt,
                buttons=[ButtonConfig("OK", result=True), ButtonConfig("Cancel")],
                fields=[
                    TextField(
                        "answer",
                        "Answer",
                        description=ANSWER_HINT,
                        default=request.default or "",
                        edit_width=60,
                    )
                ],
            )
            # OK returns True; Cancel and Escape return None.
            if not await ui.show_dialog(dialog):
                return ""
            return dialog.values["answer"]

    # === Event handler methods ===

    async def _handle_text(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle TEXT events — stream response to console."""
        ui.add_response(event.content, markdown=True)
        state.response_content.append(event.content)
        # Text is already on screen and in the durable session: replaying the
        # turn would emit it twice.
        state.side_effects_seen = True

    async def _handle_error(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle ERROR events — render, and fail the turn unless recoverable.

        ``WorkflowEvent.error(..., recoverable=True)`` means the backend
        handled it and is carrying on (a retried tool, a degraded feature): it
        is surfaced as a warning and the stream still decides the outcome.
        Anything else is a failure the caller must see in the ``TurnResult``,
        because a background-job coordinator keys durable state off
        ``delivered`` — an unrendered, unreported error was recorded as a
        successful delivery.
        """
        recoverable = bool(event.metadata.get("recoverable", False))
        code = event.metadata.get("error_code")
        suffix = f" [{code}]" if code else ""
        if recoverable:
            ui.add_warning(f"{event.content}{suffix}")
            state.status_line = f"! {event.content}"
            logger.warning("workflow_error_event", recoverable=True, code=code)
            return

        ui.add_error(f"{event.content}{suffix}")
        state.status_line = f"x {event.content}"
        logger.error("workflow_error_event", recoverable=False, code=code)
        if state.fatal_error is None:
            # Keep the first: later ones are usually the cascade.
            state.fatal_error = event.content

    async def _handle_thinking(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle THINKING events — update status, optionally display."""
        state.status_line = "Thinking..."
        if settings.verbose_thinking:
            ui.add_message("system", event.content)
        state.thinking_content.append(event.content)

    async def _handle_tool_call(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle TOOL_CALL events — update status line."""
        tool_name = event.metadata.get("tool_name", "unknown")
        state.status_line = f"Calling: {tool_name}"
        # A tool ran; the turn is no longer safe to replay wholesale.
        state.side_effects_seen = True
        # For the stateful executor, show the code being run (syntax-highlighted,
        # first N lines) so the run is visible, not just a status blip.
        if tool_name == "sandbox_execute":
            from agentic_cli.cli.sandbox_render import render_sandbox_code

            code = (event.metadata.get("tool_args") or {}).get("code", "")
            block = render_sandbox_code(code)
            if block is not None:
                ui.add_rich(block)

    async def _handle_tool_result(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle TOOL_RESULT events — display result summary."""
        state.side_effects_seen = True
        tool_name = event.metadata.get("tool_name", "unknown")
        success = event.metadata.get("success", True)
        duration = event.metadata.get("duration_ms")
        icon = "+" if success else "x"
        duration_str = f" ({duration}ms)" if duration else ""
        # The stateful executor gets a single combined message: a +/x header with
        # the run's output indented under a ╰ marker (mirrors the code block).
        if tool_name == "sandbox_execute":
            from agentic_cli.cli.sandbox_render import render_sandbox_result

            result = event.metadata.get("result")
            result_dict = result if isinstance(result, dict) else {}
            state.status_line = f"{icon} {tool_name}{duration_str}"
            ui.add_rich(render_sandbox_result(result_dict, success=success))
            return
        lines = event.content.split("\n")
        first_line = lines[0]
        state.status_line = f"{icon} {tool_name}: {first_line}{duration_str}"
        style = "green" if success else "red"
        # Tool names and output are data: escape them, or a path such as
        # "notes[/draft].md" is parsed as a closing tag (MarkupError, failing
        # the turn after the tool already ran) and "[bold]" text restyles it.
        from rich.markup import escape

        display = (
            f"[{style}]{icon}[/{style}] {escape(tool_name)}: "
            f"{escape(first_line)}{duration_str}"
        )
        if len(lines) > 1:
            display += "\n" + escape("\n".join(lines[1:]))
        ui.add_rich(display)

    async def _handle_code_execution(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle CODE_EXECUTION events — update status with result preview."""
        result_preview = (
            event.content[:40] + "..."
            if len(event.content) > 40
            else event.content
        )
        state.status_line = f"Result: {result_preview}"

    async def _handle_executable_code(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle EXECUTABLE_CODE events — update status with language."""
        lang = event.metadata.get("language", "python")
        state.status_line = f"Running {lang} code..."

    async def _handle_file_data(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle FILE_DATA events — update status with filename."""
        state.status_line = f"File: {event.content}"

    async def _handle_task_progress(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle TASK_PROGRESS events — manage a separate task thinking box."""
        progress = event.metadata.get("progress", {})
        total = progress.get("total", 0)
        completed = progress.get("completed", 0)

        box = self._live_task_box()

        # Destroy box when tasks are cleared or all complete
        if total == 0 or completed >= total:
            if self._task_box is not None:
                if box is not None:
                    box.finish(add_to_history=False, echo_to_console=False)
                self._task_box = None
                self._last_task_content = None
            # Update events box status line
            current_task = event.metadata.get("current_task_description")
            if current_task:
                state.status_line = f"Working on: {current_task}"
            return

        # Create task box on first progress event with work remaining, or
        # when the last one ended
        new_box = box is None
        if new_box:
            box = self._task_box = ui.start_thinking(
                title=f"Tasks: {completed}/{total}",
                order=100,
                content_format="ansi",
                overflow="head",  # a long list keeps its first tasks in view
            )
        else:
            # Update title on existing box
            box.set_title(f"Tasks: {completed}/{total}")

        # Deduplicate: skip content update if unchanged (a new box has none)
        if event.content and (new_box or event.content != self._last_task_content):
            self._last_task_content = event.content
            box.clear()
            box.append(_richify_task_display(event.content))

        # Update events box status line
        current_task = event.metadata.get("current_task_description")
        if current_task:
            state.status_line = f"Working on: {current_task}"

    async def _handle_context_trimmed(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle CONTEXT_TRIMMED events — increment counter and warn user."""
        if state.usage_tracker is not None:
            state.usage_tracker.context_trimmed_count += 1
        state._context_trimmed_this_invocation = True

        removed = event.metadata.get("messages_removed")
        source = event.metadata.get("source", "unknown")
        if removed is not None:
            ui.add_warning(
                f"Context trimmed: {removed} messages removed ({source})"
            )
        else:
            ui.add_warning(f"Context window trimmed ({source})")

    async def _handle_llm_usage(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle LLM_USAGE events — accumulate token counts and refresh status bar."""
        if state.usage_tracker is not None:
            tracker = state.usage_tracker
            prev_prompt = tracker.last_prompt_tokens
            heuristic_trimmed = tracker.record(
                event.metadata,
                context_trimmed_already=state._context_trimmed_this_invocation,
            )

            if heuristic_trimmed:
                from agentic_cli.cli.usage_tracker import format_tokens

                ui.add_warning(
                    f"Context window trimmed: {format_tokens(prev_prompt)}"
                    f" → {format_tokens(tracker.last_prompt_tokens)} tokens"
                    " (token_drop_heuristic)"
                )

            # Reset per-invocation flag for next LLM call
            state._context_trimmed_this_invocation = False

        if state.workflow_controller is not None:
            state.workflow_controller.update_status_bar(ui)

    # === Dispatch table ===

    _EVENT_DISPATCH: ClassVar[dict | None] = None

    @classmethod
    def _get_event_dispatch(cls) -> dict:
        """Get or build the EventType → handler dispatch table."""
        if cls._EVENT_DISPATCH is None:
            from agentic_cli.workflow import EventType

            cls._EVENT_DISPATCH = {
                EventType.TEXT: cls._handle_text,
                EventType.ERROR: cls._handle_error,
                EventType.THINKING: cls._handle_thinking,
                EventType.TOOL_CALL: cls._handle_tool_call,
                EventType.TOOL_RESULT: cls._handle_tool_result,
                EventType.CODE_EXECUTION: cls._handle_code_execution,
                EventType.EXECUTABLE_CODE: cls._handle_executable_code,
                EventType.FILE_DATA: cls._handle_file_data,
                EventType.TASK_PROGRESS: cls._handle_task_progress,
                EventType.CONTEXT_TRIMMED: cls._handle_context_trimmed,
                EventType.LLM_USAGE: cls._handle_llm_usage,
            }
        return cls._EVENT_DISPATCH
