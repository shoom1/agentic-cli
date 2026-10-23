"""``/status``: the session, the workflow, and the main agent's context.

After Services come the main agent's rows: the agent and its model, the
model's context window (or the 128k floor in use), the input budget, the
prompt size its conversation is summarized at, and its last prompt. A
manager with no budget for its root agent (not initialized yet, LangGraph
and a native ADK YAML app, which manage no context, or one without
``context_budget`` at all) shows only the last prompt, once there is one.

The managers are real and initialized offline; nothing reaches a provider.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_cli.cli.builtin_commands import StatusCommand
from agentic_cli.cli.usage_tracker import UsageTracker, format_tokens
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from tests.cli.dialog_fakes import autospec_session

SONNET = "anthropic:claude-sonnet-4-5"  # a 200,000-token window
OPUS = "anthropic:claude-opus-4-7"  # a 1,000,000-token window
#: The user's server, serving a model whose window it does not declare.
ENDPOINTS = {"local": {"base_url": "http://localhost:8000/v1", "models": {"unknown": {}}}}


@pytest.fixture(autouse=True)
def hermetic(tmp_path, monkeypatch):
    """A temp home and project, and a fake Claude key."""
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    for variable in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")


def team(root_model: str | None = None) -> list[AgentConfig]:
    return [
        AgentConfig(name="coordinator", prompt="Coordinate.", sub_agents=["researcher"],
                    description="Leads.", model=root_model),
        AgentConfig(name="researcher", prompt="Research.", description="Finds facts."),
    ]


def used(prompt: int) -> UsageTracker:
    """A tracker whose main agent's last prompt was ``prompt`` tokens."""
    tracker = UsageTracker()
    tracker.record({"prompt_tokens": prompt, "completion_tokens": 10})
    return tracker


async def status_rows(workflow, tracker: UsageTracker | None = None) -> list[tuple[str, str]]:
    """``/status``'s rows, as (label, value)."""
    app = SimpleNamespace(
        workflow=workflow, session_id="s1", session=autospec_session(),
        usage_tracker=tracker if tracker is not None else UsageTracker(),
    )
    await StatusCommand().execute("", app)
    (panel,) = app.session.add_rich.call_args.args
    labels, values = (list(column.cells) for column in panel.renderable.columns)
    return list(zip(labels, values))


def context_rows(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The rows between Services and Session: the main agent's context."""
    labels = [label for label, _ in rows]
    return rows[labels.index("Services") + 1:labels.index("Session")]


async def test_the_main_agents_context_on_a_known_window():
    # The root runs on a model of its own: its rows are its model's, while
    # the Model row keeps the default model, as the status bar does.
    manager = GoogleADKWorkflowManager(
        agent_configs=team(root_model=OPUS),
        settings=BaseSettings(session_store="memory", default_model=SONNET),
    )
    await manager.initialize_services()
    try:
        budget = manager.context_budget()
        rows = await status_rows(manager, used(45_000))
    finally:
        await manager.cleanup()

    assert rows[0] == ("Model", SONNET)
    assert context_rows(rows) == [
        ("Main Agent", f"coordinator ({OPUS})"),
        ("Context Window", "1M"),
        ("Input Budget", format_tokens(budget.budget)),
        ("Summarize At", "200k"),
        ("Last Prompt", "45k"),
    ]


async def test_an_unknown_window_says_the_floor_is_in_use():
    manager = GoogleADKWorkflowManager(
        agent_configs=team(),
        settings=BaseSettings(session_store="memory", model_endpoints=ENDPOINTS,
                              default_model="local:unknown"),
    )
    await manager.initialize_services()
    try:
        budget = manager.context_budget()
        rows = await status_rows(manager)
    finally:
        await manager.cleanup()

    assert context_rows(rows) == [
        ("Main Agent", "coordinator (local:unknown)"),
        ("Context Window", "unknown, 128k floor in use"),
        ("Input Budget", format_tokens(budget.budget)),
        ("Summarize At", format_tokens(budget.trigger)),
        ("Last Prompt", "—"),
    ]


async def test_before_initialization_there_are_none():
    manager = GoogleADKWorkflowManager(
        agent_configs=team(), settings=BaseSettings(session_store="memory", default_model=SONNET)
    )

    rows = await status_rows(manager)

    assert ("Services", "Not initialized") in rows
    assert context_rows(rows) == []


async def test_langgraph_shows_only_the_last_prompt():
    # Deprecated, and it manages no context: no window, budget or trigger.
    pytest.importorskip("langgraph")
    from agentic_cli.workflow.langgraph import LangGraphWorkflowManager

    manager = LangGraphWorkflowManager(
        agent_configs=team(), settings=BaseSettings(session_store="memory", default_model=SONNET),
        checkpointer="memory",
    )
    await manager.initialize_services()
    try:
        before_a_call = await status_rows(manager)
        after_one = await status_rows(manager, used(45_000))
    finally:
        await manager.cleanup()

    assert context_rows(before_a_call) == []
    assert context_rows(after_one) == [("Last Prompt", "45k")]


async def test_a_broken_manager_is_not_taken_for_one_without_a_budget():
    # Only a missing context_budget means "no budget": an error inside a
    # manager's own is not hidden behind the Last Prompt row.
    def context_budget():
        raise AttributeError("a bug in the manager")

    workflow = SimpleNamespace(model=SONNET, app_name="app", session_id="s1",
                               is_initialized=True, context_budget=context_budget)

    rows = await status_rows(workflow, used(45_000))

    assert "Workflow" in [label for label, _ in rows]


async def test_a_manager_without_context_budget_still_shows_the_rest():
    workflow = SimpleNamespace(model=SONNET, app_name="app", session_id="s1",
                               is_initialized=True)

    rows = await status_rows(workflow, used(45_000))

    assert ("Services", "Initialized") in rows
    assert context_rows(rows) == [("Last Prompt", "45k")]
