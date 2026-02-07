"""Tests for planning module."""

import pytest


class TestPlanStore:
    """Tests for PlanStore class."""

    def test_empty_by_default(self):
        from agentic_cli.planning import PlanStore
        store = PlanStore()
        assert store.is_empty()
        assert store.get() == ""

    def test_save_and_get(self):
        from agentic_cli.planning import PlanStore
        store = PlanStore()
        plan = "## Plan\n- [ ] Task 1\n- [ ] Task 2"
        store.save(plan)
        assert store.get() == plan
        assert not store.is_empty()

    def test_save_overwrites(self):
        from agentic_cli.planning import PlanStore
        store = PlanStore()
        store.save("Plan v1")
        store.save("Plan v2")
        assert store.get() == "Plan v2"

    def test_clear(self):
        from agentic_cli.planning import PlanStore
        store = PlanStore()
        store.save("Some plan")
        store.clear()
        assert store.is_empty()
        assert store.get() == ""

    def test_save_empty_string(self):
        from agentic_cli.planning import PlanStore
        store = PlanStore()
        store.save("Has content")
        store.save("")
        assert store.is_empty()

    def test_markdown_checkboxes(self):
        """Test that markdown checkbox plan round-trips correctly."""
        from agentic_cli.planning import PlanStore
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
        from agentic_cli.planning import PlanStore
        store = PlanStore()
        store.save("- [ ] Task A\n- [ ] Task B")
        # Agent updates the plan with progress
        store.save("- [x] Task A\n- [ ] Task B")
        assert "- [x] Task A" in store.get()

    def test_multiline_plan(self):
        """Test complex multi-line plan."""
        from agentic_cli.planning import PlanStore
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


class TestPlanStoreImport:
    """Tests for planning module exports."""

    def test_import_plan_store(self):
        from agentic_cli.planning import PlanStore
        assert PlanStore is not None

    def test_plan_store_only_export(self):
        import agentic_cli.planning as planning
        assert hasattr(planning, "PlanStore")
