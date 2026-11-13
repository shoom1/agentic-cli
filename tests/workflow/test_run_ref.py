"""Which agent run something belongs to: ``RunRef``, and the run in progress.

A sub-agent's work (its events, its tool calls, the questions and permission
prompts it raises, the jobs it starts) belongs to its run, named within its
conversation; the root's own work belongs to no run. ``get_active_run()`` is
the run whose call is in progress: a ContextVar, as the active turn and the
active tool call are, so parallel runs never see each other's.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from agentic_cli.workflow.sessions import (
    RunRef,
    get_active_run,
    reset_active_run,
    root_run,
    set_active_run,
)

RESEARCHER = RunRef(
    agent="researcher", instance="researcher-1", run="researcher-1#1", parent_run=None, depth=1
)


class TestRunRef:
    def test_the_roots_own_work_has_no_run(self):
        root = root_run("coordinator")

        assert root == RunRef(
            agent="coordinator", instance="root", run=None, parent_run=None, depth=0
        )
        assert root.is_root

    def test_a_sub_agents_run_is_not_the_roots(self):
        assert not RESEARCHER.is_root

    def test_it_is_frozen_and_hashable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            RESEARCHER.depth = 2  # type: ignore[misc]
        assert {RESEARCHER, dataclasses.replace(RESEARCHER)} == {RESEARCHER}

    def test_it_is_exported_from_the_workflow_package(self):
        from agentic_cli import workflow

        assert workflow.RunRef is RunRef
        assert workflow.root_run is root_run
        assert workflow.get_active_run is get_active_run
        assert {"RunRef", "root_run", "get_active_run"} <= set(workflow.__all__)


class TestTheActiveRun:
    def test_none_outside_a_run(self):
        assert get_active_run() is None

    def test_reset_restores_the_run_before(self):
        leaf = RunRef(
            agent="leaf", instance="leaf-1", run="leaf-1#1", parent_run="researcher-1#1", depth=2
        )
        outer = set_active_run(RESEARCHER)
        inner = set_active_run(leaf)
        assert get_active_run() == leaf

        reset_active_run(inner)
        assert get_active_run() == RESEARCHER
        reset_active_run(outer)
        assert get_active_run() is None

    async def test_each_task_sees_only_its_own(self):
        other = dataclasses.replace(RESEARCHER, instance="researcher-2", run="researcher-2#1")

        async def run_as(ref: RunRef) -> RunRef | None:
            set_active_run(ref)
            await asyncio.sleep(0)
            return get_active_run()

        seen = await asyncio.gather(
            asyncio.create_task(run_as(RESEARCHER)), asyncio.create_task(run_as(other))
        )

        assert seen == [RESEARCHER, other]
        assert get_active_run() is None
