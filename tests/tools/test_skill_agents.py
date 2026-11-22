"""A skill that starts an agent (``tools/skills/agents.py``).

The frontmatter key, read from skills as ADK's loader loads them; then
``SkillAgents`` over a manager's ``Delegations`` with a fake hook and no
backend: what a load may start (the type, the policy and the loading run's
depth), the warning as an agent is built, and the run a load starts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentic_cli.tools.skills import (
    SKILL_AGENT_KEY,
    SkillAgents,
    SkillStore,
    skill_agent,
    skill_started_by,
)
from agentic_cli.tools.skills import agents as agents_module
from agentic_cli.workflow import live_events
from agentic_cli.workflow.agents import RUN_COMPLETED, RUN_STARTED, AgentRuns, SubAgentPolicy
from agentic_cli.workflow.agents.brief import INHERITED_CONTEXT_NOTE, Brief, task_message
from agentic_cli.workflow.agents.delegate import RUNS_PER_TURN_ERROR, Delegations
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.live_events import LiveEvents, merged
from agentic_cli.workflow.sessions import (
    SessionRef,
    close_turn_scope,
    open_turn_scope,
    reset_active_run,
    reset_active_turn,
    runs_this_turn,
    set_active_run,
    set_active_turn,
)
from tests.loop_guard import no_blocking_calls
from tests.workflow.test_agent_delegate import answering, at

CONVERSATION = SessionRef("app", "u", "s")
BODY = "Look into the question and report what you find."
TYPES = {
    "lead": AgentConfig(name="lead", prompt="p", sub_agents=["middle"]),
    "middle": AgentConfig(name="middle", prompt="p", description="Plans."),
    "reader": AgentConfig(name="reader", prompt="p", description="Reads.",
                          inherit_context=True),
    "helper": AgentConfig(name="helper", prompt="p", description="Helps.",
                          spawnable_by=frozenset({"skill"})),
    "noter": AgentConfig(name="noter", prompt="p", description="Notes.",
                         spawnable_by=frozenset({"skill"}), inherit_context=True),
    "asked": AgentConfig(name="asked", prompt="p", description="Asked.",
                         spawnable_by=frozenset({"agent"})),
}
SKILLS = SubAgentPolicy(spawn_sources={"skill"})


def make_skill(parent: Path, name: str, frontmatter: str = "", body: str = BODY) -> Any:
    """The skill ``parent/<name>`` with ``frontmatter``'s extra lines, as ADK loads it."""
    folder = parent / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Does {name}.\n{frontmatter}---\n{body}\n",
        encoding="utf-8",
    )
    (skill,) = SkillStore().resolve([str(folder)])
    return skill


def made(hook=None, *, policy: SubAgentPolicy = SKILLS, runs: AgentRuns | None = None):
    return Delegations(
        runs=runs or AgentRuns(), policy=policy, types=TYPES,
        hook=hook or answering()[0],
    )


@pytest.fixture
def logged(monkeypatch) -> list[dict[str, Any]]:
    """Every log call of ``tools/skills/agents.py``, with its level."""
    events: list[dict[str, Any]] = []

    class Recorder:
        def __getattr__(self, level: str):
            return lambda event, **fields: events.append(
                {"event": event, "log_level": level, **fields}
            )

    monkeypatch.setattr(agents_module, "logger", Recorder())
    return events


class TestTheFrontmatterKey:
    def test_names_the_agent_at_the_top_of_the_frontmatter(self, tmp_path):
        assert SKILL_AGENT_KEY == "agent"
        assert skill_agent(make_skill(tmp_path, "research", "agent: helper\n")) == "helper"

    @pytest.mark.parametrize("frontmatter, named", [
        ("", None),
        ("agent:\n", None),
        ("metadata:\n  agent: helper\n", None),
        ("agent: 3\n", "3"),
        ("agent: [a, b]\n", "['a', 'b']"),
    ], ids=["none", "empty", "under-metadata", "a-number", "a-list"])
    def test_anything_else_names_none_or_is_taken_as_written(self, tmp_path, frontmatter, named):
        assert skill_agent(make_skill(tmp_path, "research", frontmatter)) == named

    def test_who_started_a_run_a_skill_starts(self):
        assert skill_started_by("research") == "skill:research"


class TestWhatALoadMayStart:
    @pytest.mark.parametrize("agent, root, policy, named, reason", [
        ("lead", True, SKILLS, "nobody", "is not one of this app's agent types"),
        ("lead", True, SKILLS, "asked", "this app does not let a skill start"),
        ("lead", True, SKILLS, "middle", "this app does not let a skill start"),
        ("lead", True, SubAgentPolicy(spawn_sources={"agent"}), "helper",
         "this app does not let a skill start"),
        ("middle", False, SKILLS, "helper",
         "cannot be started here: sub-agents cannot start sub-agents"),
        ("reader", False, SubAgentPolicy(spawn_sources={"skill"}, max_spawn_depth=2), "helper",
         "cannot be started here: sub-agents cannot start sub-agents"),
        ("middle", False, SubAgentPolicy(spawn_sources={"skill"}, max_spawn_depth=2), "noter",
         "cannot be started here: sub-agents cannot start sub-agents"),
    ], ids=["unknown", "type-not-for-skills", "a-static-sub-agent", "policy-not-for-skills",
            "past-the-depth", "an-inheriting-caller", "an-inheriting-type-below-the-root"])
    def test_a_skill_it_may_not_start_is_warned_of_as_the_agent_is_built(
        self, tmp_path, logged, agent, root, policy, named, reason
    ):
        skill = make_skill(tmp_path, "research", f"agent: {named}\n")
        agents = SkillAgents(agent, lambda: made(policy=policy), root=root)

        agents.check([skill])

        assert logged == [{
            "event": "skill_agent_not_startable", "log_level": "warning", "agent": agent,
            "skill": "research", "agent_type": named, "reason": reason,
        }]
        assert agents.refusal(skill, 0 if root else 1) == (
            f"The skill 'research' starts the agent '{named}', which {reason}."
        )

    @pytest.mark.parametrize("agent, root, policy, named", [
        ("lead", True, SKILLS, "helper"),
        ("lead", True, SKILLS, "noter"),
        ("middle", False, SubAgentPolicy(spawn_sources={"skill"}, max_spawn_depth=2), "helper"),
    ], ids=["the-root", "an-inheriting-type-by-the-root", "a-sub-agent-above-the-depth"])
    def test_one_it_may_start_is_not(self, tmp_path, logged, agent, root, policy, named):
        skill = make_skill(tmp_path, "research", f"agent: {named}\n")
        agents = SkillAgents(agent, lambda: made(policy=policy), root=root)

        agents.check([skill, make_skill(tmp_path, "plain")])

        assert logged == []
        assert agents.refusal(skill, 0 if root else 1) is None


async def loaded(agents: SkillAgents, skill: Any, *, caller=None, **fields: Any):
    """``agents.start(skill)`` in a turn of ``CONVERSATION``, from ``caller``'s run; its
    result, how many runs the turn counted, and what it reported."""
    live = LiveEvents()
    out: dict[str, Any] = {}

    async def source():
        token = set_active_turn(CONVERSATION)
        scope = open_turn_scope()
        run = set_active_run(caller)
        try:
            with live_events.reporting(live):
                out["result"] = await agents.start(
                    skill, call_id=fields.pop("call_id", "c1"),
                    state=fields.pop("state", {}), history=fields.pop("history", None),
                )
                out["counted"] = runs_this_turn()
        finally:
            reset_active_run(run)
            close_turn_scope(scope)
            reset_active_turn(token)
        if False:  # pragma: no cover - a source that yields nothing itself
            yield

    events = [item async for item in merged(source(), live)]
    return out, events


class TestALoadStartsARun:
    async def test_of_the_type_its_task_the_skills_body_its_result_the_envelope(self, tmp_path):
        hook, asked = answering("Found it.")
        runs = AgentRuns()
        delegations = made(hook, runs=runs)
        agents = SkillAgents("lead", lambda: delegations, root=True)

        with no_blocking_calls():
            out, events = await loaded(
                agents, make_skill(tmp_path, "research", "agent: helper\n"),
                state={"outside_content": "web_fetch"},
            )

        assert out["result"].result == {
            "success": True, "agent": "helper-1", "run": "helper-1#1",
            "status": RUN_COMPLETED, "partial": False, "result": "Found it.", "error": None,
            "skill": "research",
        }
        (run,) = asked
        assert run.message == task_message(Brief(title="research", objective=BODY))
        assert (run.ref.depth, run.ref.parent_run, run.state, run.history) == (
            1, None, {"outside_content": "web_fetch"}, None
        )
        assert events[0].metadata["status"] == RUN_STARTED
        assert (events[0].metadata["started_by"], events[0].metadata["call_id"],
                events[0].metadata["title"]) == ("skill:research", "c1", "research")
        assert out["counted"] == 1
        (record,) = runs.records(CONVERSATION)
        assert record.started_by == "skill:research"

    async def test_under_the_run_that_loads_it(self, tmp_path):
        hook, asked = answering()
        deeper = SubAgentPolicy(spawn_sources={"skill"}, max_spawn_depth=2)
        delegations = made(hook, policy=deeper)
        agents = SkillAgents("middle", lambda: delegations, root=False)

        out, _ = await loaded(
            agents, make_skill(tmp_path, "research", "agent: helper\n"), caller=at(1)
        )

        (run,) = asked
        assert (run.ref.instance, run.ref.depth, run.ref.parent_run) == (
            "helper-1", 2, "middle-1#1"
        )
        assert out["result"].result["skill"] == "research"

    async def test_an_inheriting_type_reads_the_callers_conversation(self, tmp_path):
        hook, asked = answering()
        delegations = made(hook)
        agents = SkillAgents("lead", lambda: delegations, root=True)

        await loaded(
            agents, make_skill(tmp_path, "notes", "agent: noter\n"),
            history=lambda: ["the conversation"],
        )

        (run,) = asked
        assert run.history == ["the conversation"]
        assert run.message.endswith(INHERITED_CONTEXT_NOTE)

    async def test_one_it_may_not_start_starts_none_and_counts_none(self, tmp_path):
        hook, asked = answering()
        delegations = made(hook)
        agents = SkillAgents("middle", lambda: delegations, root=False)

        out, events = await loaded(
            agents, make_skill(tmp_path, "research", "agent: helper\n"), caller=at(1)
        )

        assert out["result"].result == {
            "success": False,
            "error": "The skill 'research' starts the agent 'helper', which cannot be "
                     "started here: sub-agents cannot start sub-agents.",
        }
        assert (out["result"].run, out["result"].state, asked, events, out["counted"]) == (
            None, {}, [], [], 0
        )

    async def test_past_the_runs_a_turn_may_start_it_is_refused_as_task_is(self, tmp_path):
        hook, asked = answering()
        delegations = made(hook, policy=SubAgentPolicy(spawn_sources={"skill"},
                                                       max_runs_per_turn=1))
        agents = SkillAgents("lead", lambda: delegations, root=True)
        skill = make_skill(tmp_path, "research", "agent: helper\n")
        live = LiveEvents()
        results: list[Any] = []

        async def source():
            token = set_active_turn(CONVERSATION)
            scope = open_turn_scope()
            try:
                with live_events.reporting(live):
                    for call in ("c1", "c2"):
                        results.append(await agents.start(
                            skill, call_id=call, state={}, history=None
                        ))
            finally:
                close_turn_scope(scope)
                reset_active_turn(token)
            if False:  # pragma: no cover
                yield

        async for _ in merged(source(), live):
            pass

        first, second = results
        assert first.result["status"] == RUN_COMPLETED
        assert second.result == {
            "success": False, "error": RUNS_PER_TURN_ERROR.format(limit=1),
        }
        assert len(asked) == 1
