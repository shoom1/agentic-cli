"""Tests for planning module."""

import pytest


class TestPlanStore:
    """Tests for PlanStore class."""

    def test_empty_by_default(self):
        from agentic_cli.tools.planning_tools import PlanStore
        store = PlanStore()
        assert store.is_empty()
        assert store.get() == ""

    def test_save_and_get(self):
        from agentic_cli.tools.planning_tools import PlanStore
        store = PlanStore()
        plan = "## Plan\n- [ ] Task 1\n- [ ] Task 2"
        store.save(plan)
        assert store.get() == plan
        assert not store.is_empty()

    def test_save_overwrites(self):
        from agentic_cli.tools.planning_tools import PlanStore
        store = PlanStore()
        store.save("Plan v1")
        store.save("Plan v2")
        assert store.get() == "Plan v2"

    def test_clear(self):
        from agentic_cli.tools.planning_tools import PlanStore
        store = PlanStore()
        store.save("Some plan")
        store.clear()
        assert store.is_empty()
        assert store.get() == ""

    def test_save_empty_string(self):
        from agentic_cli.tools.planning_tools import PlanStore
        store = PlanStore()
        store.save("Has content")
        store.save("")
        assert store.is_empty()

    def test_markdown_checkboxes(self):
        """Test that markdown checkbox plan round-trips correctly."""
        from agentic_cli.tools.planning_tools import PlanStore
        store = PlanStore()
        plan = (
            "## Research Plan\n"
            "- [x] Gather data\n"
            "- [ ] Analyze results\n"
            "- [ ] Write summary"
        )
        store.save(plan)
        assert "- [x] Gather data" in store.get()
        assert "- [ ] Analyze results" in store.get()

    def test_update_checkboxes(self):
        """Test updating plan with checkbox progress."""
        from agentic_cli.tools.planning_tools import PlanStore
        store = PlanStore()
        store.save("- [ ] Task A\n- [ ] Task B")
        # Agent updates the plan with progress
        store.save("- [x] Task A\n- [ ] Task B")
        assert "- [x] Task A" in store.get()

    def test_multiline_plan(self):
        """Test complex multi-line plan."""
        from agentic_cli.tools.planning_tools import PlanStore
        store = PlanStore()
        plan = (
            "# Project Plan\n"
            "\n"
            "## Phase 1\n"
            "- [x] Setup environment\n"
            "- [ ] Define API\n"
            "\n"
            "## Phase 2\n"
            "- [ ] Implement\n"
            "- [ ] Test\n"
        )
        store.save(plan)
        assert store.get() == plan


class TestSummarizeCheckboxes:
    """Tests for _summarize_checkboxes helper."""

    def test_with_mixed_checkboxes(self):
        from agentic_cli.tools.planning_tools import _summarize_checkboxes
        content = "- [x] Done\n- [x] Also done\n- [ ] Pending\n- [ ] Also pending\n- [ ] Third pending"
        result = _summarize_checkboxes(content)
        assert result == "5 tasks: 2 done, 3 pending"

    def test_all_done(self):
        from agentic_cli.tools.planning_tools import _summarize_checkboxes
        result = _summarize_checkboxes("- [x] A\n- [x] B")
        assert result == "2 tasks: 2 done"

    def test_all_pending(self):
        from agentic_cli.tools.planning_tools import _summarize_checkboxes
        result = _summarize_checkboxes("- [ ] A\n- [ ] B")
        assert result == "2 tasks: 2 pending"

    def test_no_checkboxes(self):
        from agentic_cli.tools.planning_tools import _summarize_checkboxes
        result = _summarize_checkboxes("Just some text")
        assert result == ""

    def test_uppercase_x(self):
        from agentic_cli.tools.planning_tools import _summarize_checkboxes
        result = _summarize_checkboxes("- [X] Done\n- [ ] Pending")
        assert result == "2 tasks: 1 done, 1 pending"


class TestSavePlanSummary:
    """Tests for save_plan returning checkbox stats."""

    def test_save_plan_with_checkboxes_shows_stats(self):
        from agentic_cli.tools.planning_tools import save_plan, PlanStore
        from agentic_cli.workflow.context import set_context_task_graph

        store = PlanStore()
        token = set_context_task_graph(store)
        try:
            result = save_plan(content="- [x] A\n- [ ] B\n- [ ] C")
            assert result["success"] is True
            assert "3 tasks" in result["message"]
            assert "1 done" in result["message"]
            assert "2 pending" in result["message"]
        finally:
            token.var.reset(token)

    def test_save_plan_without_checkboxes(self):
        from agentic_cli.tools.planning_tools import save_plan, PlanStore
        from agentic_cli.workflow.context import set_context_task_graph

        store = PlanStore()
        token = set_context_task_graph(store)
        try:
            result = save_plan(content="## My Plan\nJust text, no checkboxes.")
            assert result["success"] is True
            assert result["message"] == "Plan saved"
        finally:
            token.var.reset(token)


class TestPlanStoreImport:
    """Tests for planning module exports."""

    def test_import_plan_store(self):
        from agentic_cli.tools.planning_tools import PlanStore
        assert PlanStore is not None

    def test_plan_store_only_export(self):
        import agentic_cli.tools.planning_tools as planning_tools
        assert hasattr(planning_tools, "PlanStore")
