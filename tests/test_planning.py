"""Tests for planning module."""

import pytest

from agentic_cli.workflow.service_registry import set_service_registry, get_service_registry


@pytest.fixture
def plan_registry_ctx():
    """Provide a service registry with plan support, auto-cleanup."""
    registry = {}
    token = set_service_registry(registry)
    yield registry
    token.var.reset(token)


class TestPlanViaRegistry:
    """Tests for plan storage via service registry."""

    def test_empty_by_default(self, plan_registry_ctx):
        assert plan_registry_ctx.get("plan", "") == ""

    def test_save_and_get(self, plan_registry_ctx):
        plan = "## Plan\n- [ ] Task 1\n- [ ] Task 2"
        plan_registry_ctx["plan"] = plan
        assert plan_registry_ctx.get("plan", "") == plan

    def test_save_overwrites(self, plan_registry_ctx):
        plan_registry_ctx["plan"] = "Plan v1"
        plan_registry_ctx["plan"] = "Plan v2"
        assert plan_registry_ctx["plan"] == "Plan v2"

    def test_clear(self, plan_registry_ctx):
        plan_registry_ctx["plan"] = "Some plan"
        plan_registry_ctx["plan"] = ""
        assert plan_registry_ctx.get("plan", "") == ""

    def test_save_empty_string(self, plan_registry_ctx):
        plan_registry_ctx["plan"] = "Has content"
        plan_registry_ctx["plan"] = ""
        assert not plan_registry_ctx.get("plan", "")

    def test_markdown_checkboxes(self, plan_registry_ctx):
        """Test that markdown checkbox plan round-trips correctly."""
        plan = (
            "## Research Plan\n"
            "- [x] Gather data\n"
            "- [ ] Analyze results\n"
            "- [ ] Write summary"
        )
        plan_registry_ctx["plan"] = plan
        assert "- [x] Gather data" in plan_registry_ctx["plan"]
        assert "- [ ] Analyze results" in plan_registry_ctx["plan"]

    def test_update_checkboxes(self, plan_registry_ctx):
        """Test updating plan with checkbox progress."""
        plan_registry_ctx["plan"] = "- [ ] Task A\n- [ ] Task B"
        plan_registry_ctx["plan"] = "- [x] Task A\n- [ ] Task B"
        assert "- [x] Task A" in plan_registry_ctx["plan"]

    def test_multiline_plan(self, plan_registry_ctx):
        """Test complex multi-line plan."""
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
        plan_registry_ctx["plan"] = plan
        assert plan_registry_ctx["plan"] == plan


class TestSummarizeCheckboxes:
    """Tests for summarize_checkboxes helper."""

    def test_with_mixed_checkboxes(self):
        from agentic_cli.tools._core.planning import summarize_checkboxes
        content = "- [x] Done\n- [x] Also done\n- [ ] Pending\n- [ ] Also pending\n- [ ] Third pending"
        result = summarize_checkboxes(content)
        assert result == "5 tasks: 2 done, 3 pending"

    def test_all_done(self):
        from agentic_cli.tools._core.planning import summarize_checkboxes
        result = summarize_checkboxes("- [x] A\n- [x] B")
        assert result == "2 tasks: 2 done"

    def test_all_pending(self):
        from agentic_cli.tools._core.planning import summarize_checkboxes
        result = summarize_checkboxes("- [ ] A\n- [ ] B")
        assert result == "2 tasks: 2 pending"

    def test_no_checkboxes(self):
        from agentic_cli.tools._core.planning import summarize_checkboxes
        result = summarize_checkboxes("Just some text")
        assert result == ""

    def test_uppercase_x(self):
        from agentic_cli.tools._core.planning import summarize_checkboxes
        result = summarize_checkboxes("- [X] Done\n- [ ] Pending")
        assert result == "2 tasks: 1 done, 1 pending"
