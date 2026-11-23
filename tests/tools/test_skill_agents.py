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
        ("middle", False, SKILLS, "helper",
         "cannot be started here: sub-agents cannot start sub-agents"),
        ("reader", False, SubAgentPolicy(spawn_sources={"skill"}, max_spawn_depth=2), "helper",
         "cannot be started here: sub-agents cannot start sub-agents"),
        ("middle", False, SubAgentPolicy(spawn_sources={"skill"}, max_spawn_depth=2), "noter",
         "cannot be started here: sub-agents cannot start sub-agents"),
    ], ids=["unknown", "type-not-for-skills", "a-static-sub-agent",
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

    def test_one_with_no_instructions_has_no_task_and_is_refused(self, tmp_path, logged):
        skill = make_skill(tmp_path, "research", "agent: helper\n", body="  ")
        agents = SkillAgents("lead", lambda: made(), root=True)
        reason = "needs the skill's instructions as its task, and this skill has none"

        agents.check([skill])

        assert logged == [{
            "event": "skill_agent_not_startable", "log_level": "warning", "agent": "lead",
            "skill": "research", "agent_type": "helper", "reason": reason,
        }]
        assert agents.refusal(skill, 0) == (
            f"The skill 'research' starts the agent 'helper', which {reason}."
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


class TestTheKeyWhileNoSkillMayStart:
    """With ``"skill"`` not in ``spawn_sources`` (the default) the key is ignored:
    the skill loads as instructions, and the build says so."""

    @pytest.mark.parametrize("policy", [SubAgentPolicy(), SubAgentPolicy(spawn_sources={"agent"})],
                             ids=["default", "agents-only"])
    @pytest.mark.parametrize("named", ["Explore", "helper"], ids=["no-such-type", "a-real-type"])
    def test_it_is_ignored_and_logged_once_at_info(self, tmp_path, logged, policy, named):
        skill = make_skill(tmp_path, "research", f"agent: {named}\n")
        agents = SkillAgents("lead", lambda: made(policy=policy), root=True)

        agents.check([skill, make_skill(tmp_path, "plain")])
        agents.check([skill])

        assert not agents.starts(skill)
        # A load treats it as instructions before asking; a start is refused.
        assert agents.refusal(skill, 0) == (
            f"The skill 'research' starts the agent '{named}', which "
            "this app does not let a skill start."
        )
        assert logged == [{
            "event": "skill_agent_ignored", "log_level": "info", "agent": "lead",
            "skill": "research", "agent_type": named,
            "message": f"The skill 'research' names the agent '{named}', but this app lets "
                       "no skill start an agent: it loads as instructions.",
        }]

    @pytest.mark.parametrize("policy", [SubAgentPolicy(), SubAgentPolicy(spawn_sources={"agent"})],
                             ids=["default", "agents-only"])
    async def test_a_start_is_refused_and_counts_none(self, tmp_path, policy):
        hook, asked = answering()
        delegations = made(hook, policy=policy)
        agents = SkillAgents("lead", lambda: delegations, root=True)

        out, events = await loaded(agents, make_skill(tmp_path, "research", "agent: helper\n"))

        assert out["result"].result == {
            "success": False,
            "error": "The skill 'research' starts the agent 'helper', which "
                     "this app does not let a skill start.",
        }
        assert (out["result"].run, asked, events, out["counted"]) == (None, [], [], 0)

    def test_where_a_skill_may_start_it_starts(self, tmp_path, logged):
        skill = make_skill(tmp_path, "research", "agent: helper\n")
        agents = SkillAgents("lead", lambda: made(), root=True)

        assert agents.starts(skill) and not agents.starts(make_skill(tmp_path, "plain"))

    def test_builds_of_one_agent_share_what_they_warned_of(self, tmp_path, logged):
        skill = make_skill(tmp_path, "research", "agent: Explore\n")
        warned: set = set()

        for _ in range(2):
            SkillAgents("lead", lambda: made(policy=SubAgentPolicy()), root=True,
                        warned=warned).check([skill])

        assert [entry["event"] for entry in logged] == ["skill_agent_ignored"]


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


class TestTheGeneralPurposeType:
    """A skill may name ``general_purpose``: with the policy's ``general_purpose``
    and ``"skill"`` among its sources, the load starts one made for the
    loading agent's type. A general-purpose agent's own loads start none."""

    GENERAL = SubAgentPolicy(spawn_sources={"skill"}, general_purpose=True)

    async def test_it_starts_one_made_for_the_loading_agents_type(self, tmp_path, logged):
        from agentic_cli.workflow.agents.spawn import general_purpose_config

        hook, asked = answering("Found it.")
        runs = AgentRuns()
        delegations = made(hook, policy=self.GENERAL, runs=runs)
        agents = SkillAgents("lead", lambda: delegations, root=True)
        skill = make_skill(tmp_path, "research", "agent: general_purpose\n")

        agents.check([skill])
        out, events = await loaded(agents, skill)

        assert logged == []
        assert out["result"].result["agent"] == "general_purpose-1"
        (run,) = asked
        assert (run.config, run.based_on) == (general_purpose_config(TYPES["lead"]), "lead")
        assert events[0].metadata["based_on"] == "lead"
        assert runs.instance(CONVERSATION, "general_purpose-1").based_on == "lead"

    @pytest.mark.parametrize("policy, reason", [
        (SKILLS, "is not one of this app's agent types"),
        (SubAgentPolicy(spawn_sources={"agent"}, general_purpose=True),
         "this app does not let a skill start"),
    ], ids=["off", "not-for-skills"])
    def test_with_the_type_off_or_not_for_skills_it_is_refused(self, tmp_path, policy, reason):
        skill = make_skill(tmp_path, "research", "agent: general_purpose\n")
        agents = SkillAgents("lead", lambda: made(policy=policy), root=True)

        assert agents.refusal(skill, 0) == (
            f"The skill 'research' starts the agent 'general_purpose', which {reason}."
        )

    @pytest.mark.parametrize("named", ["helper", "general_purpose"])
    def test_a_general_purpose_agent_starts_none_and_is_not_warned_of(
        self, tmp_path, logged, named
    ):
        deeper = SubAgentPolicy(spawn_sources={"skill"}, max_spawn_depth=3, general_purpose=True)
        skill = make_skill(tmp_path, "research", f"agent: {named}\n")
        agents = SkillAgents("general_purpose", lambda: made(policy=deeper), root=False)

        agents.check([skill])

        assert logged == []
        assert agents.refusal(skill, 1) == (
            f"The skill 'research' starts the agent '{named}', which cannot be started "
            "here: sub-agents cannot start sub-agents."
        )

    def test_past_the_depth_it_is_refused_as_any_type(self, tmp_path):
        skill = make_skill(tmp_path, "research", "agent: general_purpose\n")
        agents = SkillAgents("middle", lambda: made(policy=self.GENERAL), root=False)

        assert agents.refusal(skill, 1) == (
            "The skill 'research' starts the agent 'general_purpose', which cannot be started "
            "here: sub-agents cannot start sub-agents."
        )
