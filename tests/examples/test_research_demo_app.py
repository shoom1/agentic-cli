"""Headless acceptance tests for the *composed* research-demo application.

The workflow, the renderer and the individual commands all have component
tests. What none of them covers is the wiring: a real ``ResearchDemoApp`` — its
own command registry, ``BaseCLIApp.process_input``, the real
``MessageProcessor`` — reacting to real input.

So these build the actual app and drive ``process_input()``. The only things
substituted are the two nondeterministic ones:

- the **workflow manager**, replaced by a scripted ``WorkflowEvent`` stream
  (no LLM, no network) — ``WorkflowEvent`` is the UI-independent boundary the
  framework already defines, so scripting it is not a new seam;
- the **UI session**, replaced by ``RecordingSession`` (the same stand-in
  ``test_message_processor_render.py`` uses) so assertions are on semantic
  render calls rather than terminal bytes.

Assertions are deliberately about *shape* — which kind of call happened, in
what order, was it a warning or an error — never about generated prose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agentic_cli.workflow.events import WorkflowEvent
from tests.demo_isolation import (
    assert_dotenv_isolated,
    assert_no_global_settings_leak,
    effective_env_file,
    isolated_env_file,
    make_isolated_settings,
    suppress_global_logging_config,
)
from tests.event_replay import RecordingSession, ReplayController

# Make the `examples` namespace package importable (no __init__.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.research_demo.app import ResearchDemoApp  # noqa: E402


class ScriptedWorkflow:
    """Workflow-manager stand-in driving one scripted turn per call.

    Extends what ``ReplayWorkflow`` offers with the two things these tests
    need: several distinct turns in sequence, and a turn that fails — so
    "the app is still usable afterwards" can actually be asserted.
    """

    def __init__(self, turns: list[list[WorkflowEvent] | Exception]) -> None:
        self._turns = list(turns)
        self.messages: list[str] = []
        self.session_ids: list[str | None] = []
        self.input_callback = None

    def set_input_callback(self, callback) -> None:  # noqa: ANN001
        self.input_callback = callback

    def clear_input_callback(self) -> None:
        self.input_callback = None

    async def process(self, message: str, user_id: str, session_id: str | None = None):
        self.messages.append(message)
        self.session_ids.append(session_id)
        turn = self._turns.pop(0) if self._turns else []
        if isinstance(turn, Exception):
            raise turn
        for event in turn:
            yield event


@pytest.fixture
def demo_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ResearchDemoApp:
    """A real ResearchDemoApp on an isolated HOME/cwd, with a recording UI.

    HOME and cwd are both redirected: the demo derives its user config, user
    KB and ``.env`` from ``~/.research_demo`` and its project KB and permission
    workdir from ``./.research_demo``, and neither may touch the developer's
    real config or the repo.

    The dotenv needs the third, explicit step (``make_isolated_settings``):
    ``model_config["env_file"]`` was frozen to the developer's real
    ``~/.research_demo/.env`` when this module was imported, so redirecting
    HOME cannot move it.

    The workflow controller is left as the **real** one, uninitialized — that
    is what makes ``app.workflow`` raise, which is exactly the state the
    readiness tests are about. Tests that need a turn install a scripted
    controller themselves.
    """
    home = tmp_path / "home"
    project = tmp_path / "project"
    workspace = tmp_path / "ws"
    for d in (home, project, workspace):
        d.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    settings = make_isolated_settings(
        home=home,
        workspace=workspace,
        permissions_enabled=False,
    )
    # Building a real app would otherwise reconfigure structlog globally and
    # break every later test that captures logs (see the helper's docstring).
    suppress_global_logging_config(monkeypatch)

    app = ResearchDemoApp(settings=settings)
    app.session = RecordingSession()
    app._test_home = home  # for the isolation assertions below
    return app


def _script(app: ResearchDemoApp, *turns) -> ScriptedWorkflow:
    """Install a scripted workflow behind the app's controller seam."""
    workflow = ScriptedWorkflow(list(turns))
    app._workflow_controller = ReplayController(workflow)
    return workflow


class TestFixtureIsolation:
    """The fixture must not be able to read the developer's configuration.

    Worth its own tests because the failure is silent: a headless test that
    quietly picked up a real API key from ``~/.research_demo/.env`` would still
    pass, while proving something different from what it claims.
    """

    def test_effective_dotenv_is_under_the_temporary_home(self, demo_app) -> None:
        home = demo_app._test_home
        env_file = effective_env_file(
            workspace_dir=home / "ws",
            _env_file=str(isolated_env_file(home)),
        )
        assert_dotenv_isolated(env_file, home)

    def test_the_class_default_would_not_have_been_isolated(self) -> None:
        """Pin the hazard this fixture works around.

        If the class ever stops resolving its dotenv at import time, this test
        fails and the ``_env_file`` plumbing can be simplified away.
        """
        from tests.demo_isolation import REAL_HOME, ResearchDemoSettings

        class_default = Path(str(ResearchDemoSettings.model_config["env_file"]))
        assert class_default.is_relative_to(REAL_HOME), (
            "ResearchDemoSettings no longer freezes its env_file under the real "
            f"home ({class_default}) — re-check whether _env_file is still needed"
        )

    def test_settings_do_not_leak_globally(self, demo_app) -> None:
        """Building the app must not install a process-wide settings singleton."""
        assert_no_global_settings_leak()


class TestCommandsBeforeReadiness:
    """Demo commands must degrade to a warning while the workflow comes up.

    ``app.workflow`` raises until the controller reports READY. A command that
    touched it during background init was caught by
    ``BaseCLIApp._handle_command`` and surfaced as the generic
    "Error executing command: Workflow not initialized yet" — which reads like
    a bug rather than "not yet".
    """

    async def test_memory_warns_and_does_not_error(self, demo_app):
        with pytest.raises(RuntimeError):
            _ = demo_app.workflow  # precondition: genuinely not ready

        await demo_app.process_input("/memory")

        assert demo_app.session.errors() == [], (
            f"/memory errored before readiness: {demo_app.session.errors()}"
        )
        assert len(demo_app.session.warnings()) == 1
        assert "initializing" in demo_app.session.warnings()[0].lower()

    async def test_kb_backfill_warns_and_does_not_error(self, demo_app):
        await demo_app.process_input("/kb-backfill")

        assert demo_app.session.errors() == [], (
            f"/kb-backfill errored before readiness: {demo_app.session.errors()}"
        )
        assert len(demo_app.session.warnings()) == 1
        assert "initializing" in demo_app.session.warnings()[0].lower()

    async def test_the_command_is_still_registered_and_echoed(self, demo_app):
        """A warning must come from the command, not from it being unknown."""
        await demo_app.process_input("/memory")

        echoed = [c for c in demo_app.session.calls if c[0] == "message"]
        assert ("message", "user", "/memory") in echoed
        assert not any("Unknown command" in e for e in demo_app.session.errors())


class TestMessageRouting:
    """A plain message goes through MessageProcessor and renders its events."""

    async def test_scripted_turn_is_rendered_in_order(self, demo_app):
        workflow = _script(
            demo_app,
            [
                WorkflowEvent.thinking("Considering the question."),
                WorkflowEvent.tool_call("kb_search", {"query": "decoding"}),
                WorkflowEvent.tool_result(
                    "kb_search", {"success": True}, success=True, duration_ms=5
                ),
                WorkflowEvent.text("Here is what I found."),
            ],
        )

        await demo_app.process_input("what do you know about decoding?")

        assert workflow.messages == ["what do you know about decoding?"]

        kinds = demo_app.session.kinds()
        # The user's message is echoed before anything is rendered for the turn.
        assert kinds[0] == "message"
        assert demo_app.session.calls[0][1] == "user"
        # The scripted text reached the UI as a response, after the echo.
        assert "Here is what I found." in demo_app.session.responses()
        assert kinds.index("response") > 0
        # A tool call opened the events box before the response was written.
        assert "start_thinking" in kinds
        assert kinds.index("start_thinking") < kinds.index("response")
        assert demo_app.session.errors() == []

    async def test_turn_receives_the_application_session_id(self, demo_app):
        workflow = _script(demo_app, [WorkflowEvent.text("ok")])

        await demo_app.process_input("hello")

        assert workflow.session_ids == [demo_app.session_id]
        assert demo_app.session_id, "the app must carry a durable session id"

    async def test_blank_input_is_ignored(self, demo_app):
        workflow = _script(demo_app, [WorkflowEvent.text("should not run")])

        await demo_app.process_input("   ")

        assert workflow.messages == []
        assert demo_app.session.calls == []


class TestUnknownCommand:
    async def test_unknown_command_errors_with_a_hint(self, demo_app):
        await demo_app.process_input("/definitely-not-a-command")

        errors = demo_app.session.errors()
        assert len(errors) == 1
        assert "definitely-not-a-command" in errors[0]
        # And the user is pointed somewhere useful.
        assert any(
            "/help" in c[2] for c in demo_app.session.calls if c[0] == "message"
        )

    async def test_unknown_command_does_not_reach_the_workflow(self, demo_app):
        workflow = _script(demo_app, [WorkflowEvent.text("should not run")])

        await demo_app.process_input("/nope")

        assert workflow.messages == []


class TestRecoveryAfterFailure:
    """A failed turn must not wedge the app: the next turn still works."""

    async def test_failed_turn_is_reported_then_the_next_turn_succeeds(self, demo_app):
        workflow = _script(
            demo_app,
            RuntimeError("backend exploded"),
            [WorkflowEvent.text("second turn is fine")],
        )

        await demo_app.process_input("first")
        assert demo_app.session.errors(), "a failing turn reported nothing"
        first_errors = len(demo_app.session.errors())

        await demo_app.process_input("second")

        assert workflow.messages == ["first", "second"]
        assert "second turn is fine" in demo_app.session.responses()
        assert len(demo_app.session.errors()) == first_errors, (
            "the recovery turn produced a new error"
        )

    async def test_a_command_error_does_not_wedge_the_next_turn(self, demo_app):
        """An exception inside a command is contained by _handle_command."""
        workflow = _script(demo_app, [WorkflowEvent.text("still working")])

        boom = demo_app.command_registry.get("memory")

        async def _raise(args, app):  # noqa: ANN001
            raise RuntimeError("command exploded")

        monkey = boom.execute
        boom.execute = _raise
        try:
            await demo_app.process_input("/memory")
        finally:
            boom.execute = monkey

        assert any("command exploded" in e for e in demo_app.session.errors())

        await demo_app.process_input("carry on")
        assert workflow.messages == ["carry on"]
        assert "still working" in demo_app.session.responses()
