"""Live scenario tests over the *real* research_demo agents.

These drive the actual ``AGENT_CONFIGS`` from ``examples/research_demo`` against
a real LLM and assert on observable *properties* of the resulting event stream
(which tools fired, whether they succeeded, whether text came back) — the
automated version of manually starting the demo and trying planning / arXiv /
KB. They are non-deterministic, so assertions are property-based and lenient;
they cost money and hit the network, so they are marked ``@pytest.mark.llm``
and skipped by default.

Run them explicitly:

    conda run -n agenticcli python -m pytest tests/integration/test_research_scenarios.py -v -m llm

Requires GOOGLE_API_KEY (Gemini, the demo default) or ANTHROPIC_API_KEY. These
scenarios are pinned to the **ADK** orchestrator, which is what the demo ships
with; ADK runs Claude natively via ``AnthropicLlm``, so setting
``AGENTIC_SCENARIO_MODEL=claude-...`` just changes the model, not the backend.
Set AGENTIC_RECORD_EVENTS=<dir> to dump each run's event stream to JSON
(replayable by the deterministic render tests).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

import pytest

from agentic_cli.workflow.adk.transfer_tool_description import TRANSFER_TOOL_NAME
from agentic_cli.workflow.events import EventType
from agentic_cli.workflow.factory import create_workflow_manager_from_settings
from agentic_cli.workflow.settings import OrchestratorType

from tests.demo_isolation import (
    assert_dotenv_isolated,
    assert_no_global_settings_leak,
    effective_env_file,
    isolated_env_file,
    make_isolated_settings,
)
from tests.event_replay import events_to_dicts
from tests.integration.helpers import (
    find_events,
    find_tool_calls,
    find_tool_results,
)

# Make the `examples` namespace package importable (no __init__.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from examples.research_demo.agents import AGENT_CONFIGS  # noqa: E402
from examples.research_demo.settings import ResearchDemoSettings  # noqa: E402


_has_any_key = bool(
    os.environ.get("GOOGLE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
)

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not _has_any_key, reason="No LLM API key in environment"),
]


@pytest.fixture
def research_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ResearchDemoSettings:
    """The demo's own settings, fully isolated: temp HOME, temp cwd, no gate.

    Uses ``ResearchDemoSettings`` rather than a bare ``BaseSettings`` so the
    scenarios exercise what the demo actually runs with — its ``skills_dirs``,
    its sandbox data mount and artifacts dir, and its ``research_demo``
    app_name (which is what ``./.{app_name}/`` and ``~/.{app_name}/`` are
    derived from).

    Isolation is on both axes, because the demo reads from both:

    - ``HOME`` is redirected, so the user config and user KB resolve into
      tmp_path and a developer's real config cannot change what these assert.
    - ``cwd`` is redirected, so the project KB and permission workdir
      (``./.research_demo/...``) are written under tmp_path, not into the repo.
    - the dotenv is passed explicitly (``make_isolated_settings``), because
      ``model_config["env_file"]`` was frozen to the real ``~/.research_demo/
      .env`` at import and ``HOME`` cannot move it.

    Provider credentials still arrive the way the live-test framework supplies
    them (real environment / the integration conftest) — that is the point of a
    live test. What must *not* happen is ``ResearchDemoSettings`` re-reading the
    developer's dotenv or JSON config and changing the model, orchestrator or
    workspace out from under the scenario.

    No ``set_settings()``: the manager scopes settings to itself for the whole
    turn (``BaseWorkflowManager._workflow_context`` → ``set_context_settings``,
    and ADK agent construction under ``SettingsContext``), so the global
    singleton is unnecessary — and it has no teardown, so setting it would leak
    this fixture's settings into every later test in the session.

    The permission gate is disabled so tool calls run headlessly without a
    prompt UI — these tests exercise agent behavior, not the permission UX.
    """
    home = tmp_path / "home"
    workspace = tmp_path / "research_ws"
    project = tmp_path / "project"
    for d in (home, workspace, project):
        d.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    settings = make_isolated_settings(
        home=home,
        workspace=workspace,
        permissions_enabled=False,
        # The demo ships ADK; pin it so a stray orchestrator setting in the
        # environment cannot silently move these scenarios to another backend.
        orchestrator=OrchestratorType.ADK,
    )

    # Prove the isolation rather than assume it: a live run that silently read
    # the real dotenv could pick up a different model or workspace.
    assert_dotenv_isolated(
        effective_env_file(
            workspace_dir=workspace,
            _env_file=str(isolated_env_file(home)),
        ),
        home,
    )
    assert settings.orchestrator is OrchestratorType.ADK

    yield settings

    assert_no_global_settings_leak()


def _maybe_record(events: list, name: str) -> None:
    out_dir = os.environ.get("AGENTIC_RECORD_EVENTS")
    if not out_dir:
        return
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{name}.json").write_text(
        json.dumps(events_to_dicts(events), indent=2, default=str)
    )


@contextlib.asynccontextmanager
async def conversation(
    settings: ResearchDemoSettings,
    *,
    user_id: str = "tester",
    session_id: str = "scenario-session",
):
    """One manager and one session, held open across several turns.

    The coordinator's policy is size-based: an explicit, bounded request ("search
    arXiv for two papers", "ingest this note") is executed or delegated
    immediately, while an open-ended goal — or an explicit request for a plan —
    is planned, shown and held for confirmation. The scenarios therefore ask for
    one or the other deliberately, and each asserts only its own contract.

    Multiple turns matter where state carries across them: the KB scenario
    ingests in one turn and reads back in the next, which only works because the
    manager is initialized and cleaned up exactly once and every turn reuses the
    same ``user_id`` and ``session_id``.
    """
    manager = create_workflow_manager_from_settings(
        agent_configs=AGENT_CONFIGS,
        settings=settings,
        model=os.environ.get("AGENTIC_SCENARIO_MODEL") or None,
    )

    # Auto-approve any HITL request so a live run never blocks waiting on input.
    async def _auto_input(request) -> str:  # noqa: ANN001
        return "yes, proceed"

    manager.set_input_callback(_auto_input)
    turns: list[list] = []

    async def say(message: str) -> list:
        """Send one user message; return that turn's events.

        A workflow exception propagates: these are behavioural contracts, and a
        turn that died is a failure, not something to inspect around. Only
        ``Exception`` is involved anywhere here — cancellation, Ctrl+C and
        SystemExit must keep unwinding an interrupted live run.
        """
        events: list = []
        async for event in manager.process(
            message=message, user_id=user_id, session_id=session_id
        ):
            events.append(event)
        turns.append(events)
        return events

    say.turns = turns  # type: ignore[attr-defined]
    say.all_events = lambda: [e for t in turns for e in t]  # type: ignore[attr-defined]

    try:
        yield say
    finally:
        manager.clear_input_callback()
        await manager.cleanup()


async def run_agent(
    settings: ResearchDemoSettings,
    message: str,
    *,
    session_id: str | None = None,
    record_name: str | None = None,
) -> list:
    """Run one turn through the real demo agents and return the event stream."""
    async with conversation(
        settings, session_id=session_id or "scenario-session"
    ) as say:
        events = await say(message)

    if record_name:
        _maybe_record(events, record_name)
    return events


class TestPlanningScenario:
    """Planning, on its own: an explicit plan request produces a saved plan.

    Deliberately does **not** also test delegation. Planning, delegation and KB
    behaviour are three independent properties, and bundling them meant one
    stochastic policy choice failed all three at once.
    """

    async def test_creates_and_saves_a_plan(self, research_settings):
        events = await run_agent(
            research_settings,
            "Create a short research plan (about 3 steps) for the topic "
            "'speculative decoding in LLMs' and save it with save_plan, then show me "
            "the plan. Do not ask me for confirmation — just create and show it.",
            record_name="planning",
        )

        assert find_tool_calls(events, "save_plan"), (
            "agent did not call save_plan; tool calls were: "
            f"{[c.metadata.get('tool_name') for c in find_tool_calls(events)]}"
        )
        # The plan should also be shown to the user as text.
        assert find_events(events, EventType.TEXT), "no text response produced"


def _tool_names(events: list) -> list[str]:
    return [c.metadata.get("tool_name") for c in find_tool_calls(events)]


#: A bounded, fully-specified arXiv request. Per the coordinator prompt this is
#: an "explicit, bounded operation": delegate and run it, no plan, no
#: confirmation. Shared so the delegation diagnostic and the behavioural
#: scenario exercise the same path.
BOUNDED_ARXIV_REQUEST = (
    "Search arXiv for 2 recent papers on 'speculative decoding' and list their "
    "titles. This is a small, explicit request — do it now via the "
    "arxiv_specialist; no plan and no confirmation needed."
)


#: The class name ADK's own tool description used to recommend. Asserted
#: against here (never put in the agent's prompt, where it would prime the
#: model toward the very mistake being guarded).
_TRANSFER_CLASS_NAME = "TransferToAgentTool"


def _assert_no_class_name_transfer(events: list) -> None:
    """Delegation must go through the function, never the class name."""
    called = _tool_names(events)
    assert _TRANSFER_CLASS_NAME not in called, (
        f"the model called the transfer tool by its class name: {called}"
    )
    offending = [
        str(e.content)
        for e in find_events(events, EventType.ERROR)
        if _TRANSFER_CLASS_NAME in str(e.content)
    ]
    assert not offending, f"transfer-by-class-name error surfaced: {offending}"


def _assert_no_unknown_tool_errors(events: list) -> None:
    """Every tool the model called must be one that actually exists."""
    unknown = [
        str(e.content)
        for e in find_events(events, EventType.ERROR)
        if "not found" in str(e.content)
    ]
    assert not unknown, (
        f"the model called a tool that does not exist: {unknown}; "
        f"tools actually called: {_tool_names(events)}"
    )


class TestArxivScenario:
    """Delegation: a bounded request reaches the specialist and runs.

    One turn, because the coordinator's prompt says an explicit bounded
    operation should just be done. No plan is required or asserted here — that
    is :class:`TestPlanningScenario`'s job.

    This is also the delegation regression: the transfer defect is asserted
    inside the behavioural run rather than as a separate paid diagnostic, so
    the class-name check can never be satisfied vacuously by a run in which no
    delegation was attempted — the same test requires the correct
    ``transfer_to_agent`` call.
    """

    async def test_bounded_request_delegates_and_searches(self, research_settings):
        async with conversation(research_settings) as say:
            events = await say(BOUNDED_ARXIV_REQUEST)
            _maybe_record(events, "arxiv_search")

        # Delegation happened, by the correct function name.
        assert find_tool_calls(events, TRANSFER_TOOL_NAME), (
            f"the coordinator never delegated; tools called: {_tool_names(events)}"
        )
        _assert_no_class_name_transfer(events)
        _assert_no_unknown_tool_errors(events)

        # And the specialist actually did the work.
        assert find_tool_calls(events, "search_arxiv"), (
            f"the specialist never searched arXiv; tools called: {_tool_names(events)}"
        )
        results = find_tool_results(events, "search_arxiv")
        assert any(r.metadata.get("success", True) for r in results), (
            f"search_arxiv never returned a successful result: "
            f"{[r.content for r in results]}"
        )


class TestKnowledgeBaseScenario:
    """KB: ingest a note (via the writer specialist) then read it back.

    Depends on the coordinator delegating the write to the arxiv_specialist
    (only it holds KB writer tools), so assertions stay lenient.
    """

    @pytest.fixture(autouse=True)
    def _require_embeddings(self):
        pytest.importorskip("faiss")
        pytest.importorskip("sentence_transformers")

    async def test_bounded_ingest_then_bounded_readback(self, research_settings):
        """Two bounded operations in one persistent conversation.

        Each turn is explicit and small, so per the coordinator's prompt both
        should just run — no plan-policy assertions here. The two turns share a
        manager and a session so the read-back sees what the ingest wrote.

        The ingest turn asks for the *outcome* and points at the specialist,
        rather than naming a writer tool: the coordinator holds only the KB
        readers (``kb_search``/``kb_read``/``kb_list``/``kb_search_concepts``),
        and ``kb_ingest_*`` belongs to ``arxiv_specialist``. Telling the
        coordinator to call a tool it does not have makes a correct refusal
        look like a failure.
        """
        note = (
            "Speculative decoding uses a small draft model to propose tokens that a "
            "larger target model verifies in parallel, cutting latency without "
            "changing the output distribution."
        )
        async with conversation(research_settings) as say:
            ingest_turn = await say(
                "Store this exact note in the knowledge base now. The "
                "arxiv_specialist holds the knowledge-base write tools, so hand "
                "it over. This is a small explicit request — no plan and no "
                f"confirmation needed.\n\nNote: {note}"
            )
            readback_turn = await say(
                "Now search the knowledge base for 'speculative decoding' and "
                "show me what you find. Again, just do it."
            )
            _maybe_record(say.all_events(), "kb_ingest_search")

        ingest_results = [
            r
            for r in find_events(ingest_turn, EventType.TOOL_RESULT)
            if str(r.metadata.get("tool_name", "")).startswith("kb_ingest")
        ]
        assert ingest_results, (
            f"no kb_ingest_* tool result; turn-1 tools: {_tool_names(ingest_turn)}"
        )
        assert any(r.metadata.get("success", True) for r in ingest_results), (
            f"kb_ingest never succeeded: {[r.content for r in ingest_results]}"
        )

        assert find_tool_calls(readback_turn, "kb_search") or find_tool_calls(
            readback_turn, "kb_list"
        ), f"the KB was never queried back; turn-2 tools: {_tool_names(readback_turn)}"

        # Both turns must have used real tool names. Asserted here rather than
        # as a separate paid run: a spliced name (two real tools joined into
        # one) shows up as an unknown-tool error on exactly this path.
        both_turns = ingest_turn + readback_turn
        _assert_no_unknown_tool_errors(both_turns)
        _assert_no_class_name_transfer(both_turns)
