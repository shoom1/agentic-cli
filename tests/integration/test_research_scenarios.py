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

Requires GOOGLE_API_KEY (Gemini/ADK, the demo default) or ANTHROPIC_API_KEY.
For Anthropic, also set AGENTIC_SCENARIO_MODEL=claude-... so the factory routes
to LangGraph. Set AGENTIC_RECORD_EVENTS=<dir> to dump each run's event stream
to JSON (replayable by the deterministic render tests).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from agentic_cli.config import BaseSettings, set_settings
from agentic_cli.workflow.events import EventType
from agentic_cli.workflow.factory import create_workflow_manager_from_settings

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


_has_any_key = bool(
    os.environ.get("GOOGLE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
)

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not _has_any_key, reason="No LLM API key in environment"),
]


@pytest.fixture
def research_settings(tmp_path: Path) -> BaseSettings:
    """Settings for live runs: real keys, temp workspace, permission gate off.

    The permission gate is disabled so tool calls run headlessly without a
    prompt UI — these tests exercise agent behavior, not the permission UX.
    """
    workspace = tmp_path / "research_ws"
    workspace.mkdir(parents=True)
    settings = BaseSettings(workspace_dir=workspace, permissions_enabled=False)
    set_settings(settings)
    return settings


def _maybe_record(events: list, name: str) -> None:
    out_dir = os.environ.get("AGENTIC_RECORD_EVENTS")
    if not out_dir:
        return
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{name}.json").write_text(
        json.dumps(events_to_dicts(events), indent=2, default=str)
    )


async def run_agent(
    settings: BaseSettings,
    message: str,
    *,
    session_id: str | None = None,
    record_name: str | None = None,
) -> list:
    """Run one turn through the real demo agents and return the event stream."""
    manager = create_workflow_manager_from_settings(
        agent_configs=AGENT_CONFIGS,
        settings=settings,
        model=os.environ.get("AGENTIC_SCENARIO_MODEL") or None,
    )

    # Auto-approve any HITL request so a live run never blocks waiting on input.
    async def _auto_input(request) -> str:  # noqa: ANN001
        return "yes, proceed"

    manager.set_input_callback(_auto_input)
    try:
        events: list = []
        async for event in manager.process(
            message=message, user_id="tester", session_id=session_id
        ):
            events.append(event)
    finally:
        manager.clear_input_callback()
        await manager.cleanup()

    if record_name:
        _maybe_record(events, record_name)
    return events


class TestPlanningScenario:
    """Planning: the coordinator should save and show a plan."""

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


class TestArxivScenario:
    """arXiv: the specialist should search arXiv and get results back."""

    async def test_searches_arxiv(self, research_settings):
        events = await run_agent(
            research_settings,
            "Search arXiv for 2 recent papers on 'speculative decoding' using "
            "search_arxiv and list their titles. Do not ask for confirmation.",
            record_name="arxiv_search",
        )

        calls = find_tool_calls(events, "search_arxiv")
        assert calls, "agent did not call search_arxiv"

        results = find_tool_results(events, "search_arxiv")
        assert any(r.metadata.get("success", True) for r in results), (
            "search_arxiv never returned a successful result: "
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

    async def test_ingest_text_then_search(self, research_settings):
        note = (
            "Speculative decoding uses a small draft model to propose tokens that a "
            "larger target model verifies in parallel, cutting latency without "
            "changing the output distribution."
        )
        events = await run_agent(
            research_settings,
            "Ingest the following note into the knowledge base (use the "
            "arxiv_specialist, which has KB write access), then search the knowledge "
            "base for 'speculative decoding' to confirm it is retrievable. Do not ask "
            f"for confirmation.\n\nNote: {note}",
            record_name="kb_ingest_search",
        )

        ingest_results = [
            r
            for r in find_events(events, EventType.TOOL_RESULT)
            if str(r.metadata.get("tool_name", "")).startswith("kb_ingest")
        ]
        assert ingest_results, (
            "no kb_ingest_* tool result observed; tool calls were: "
            f"{[c.metadata.get('tool_name') for c in find_tool_calls(events)]}"
        )
        assert any(r.metadata.get("success", True) for r in ingest_results), (
            "kb_ingest never succeeded: "
            f"{[r.content for r in ingest_results]}"
        )
        # And the KB was queried afterward.
        assert find_tool_calls(events, "kb_search") or find_tool_calls(
            events, "kb_list"
        ), "agent did not query the KB after ingesting"
