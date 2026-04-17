"""Tests for agentic_cli.tools._core.planning — pure planning functions."""

from agentic_cli.tools._core.planning import summarize_checkboxes


class TestSummarizeCheckboxes:
    """Tests for summarize_checkboxes()."""

    def test_no_checkboxes_returns_empty(self):
        assert summarize_checkboxes("Just some text") == ""

    def test_empty_string_returns_empty(self):
        assert summarize_checkboxes("") == ""

    def test_all_pending(self):
        content = "- [ ] A\n- [ ] B\n- [ ] C"
        assert summarize_checkboxes(content) == "3 tasks: 3 pending"

    def test_all_done(self):
        content = "- [x] A\n- [x] B"
        assert summarize_checkboxes(content) == "2 tasks: 2 done"

    def test_mixed_checkboxes(self):
        content = "- [x] Done\n- [x] Also done\n- [ ] Pending\n- [ ] Also pending"
        assert summarize_checkboxes(content) == "4 tasks: 2 done, 2 pending"

    def test_uppercase_x(self):
        content = "- [X] Done\n- [ ] Pending"
        assert summarize_checkboxes(content) == "2 tasks: 1 done, 1 pending"

    def test_ignores_non_checkbox_lines(self):
        content = (
            "# Plan\n"
            "Some text\n"
            "- [x] Task 1\n"
            "- Normal bullet\n"
            "- [ ] Task 2\n"
        )
        assert summarize_checkboxes(content) == "2 tasks: 1 done, 1 pending"

    def test_indented_checkboxes(self):
        content = "  - [x] Done\n  - [ ] Pending"
        assert summarize_checkboxes(content) == "2 tasks: 1 done, 1 pending"

    def test_multiline_plan(self):
        content = (
            "## Phase 1\n"
            "- [x] Setup\n"
            "- [x] Configure\n"
            "\n"
            "## Phase 2\n"
            "- [ ] Implement\n"
            "- [ ] Test\n"
            "- [ ] Deploy\n"
        )
        result = summarize_checkboxes(content)
        assert result == "5 tasks: 2 done, 3 pending"

    def test_single_done_task(self):
        assert summarize_checkboxes("- [x] Only task") == "1 tasks: 1 done"

    def test_single_pending_task(self):
        assert summarize_checkboxes("- [ ] Only task") == "1 tasks: 1 pending"
