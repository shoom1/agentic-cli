"""Tests for tool-specific result summary formatters."""

import pytest

from agentic_cli.constants import format_size
from agentic_cli.workflow.tool_summaries import format_tool_summary


# ---------------------------------------------------------------------------
# format_size() helper
# ---------------------------------------------------------------------------


class TestFormatSize:
    def test_bytes(self):
        assert format_size(0) == "0B"
        assert format_size(512) == "512B"
        assert format_size(1023) == "1023B"

    def test_kilobytes(self):
        assert format_size(1024) == "1.0KB"
        assert format_size(1536) == "1.5KB"
        assert format_size(10240) == "10.0KB"

    def test_megabytes(self):
        assert format_size(1024 * 1024) == "1.0MB"
        assert format_size(5 * 1024 * 1024) == "5.0MB"


# ---------------------------------------------------------------------------
# format_tool_summary() — per-tool formatters
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_full_file(self):
        result = {"success": True, "content": "...", "path": "/tmp/config.py", "size": 1234}
        assert format_tool_summary("read_file", result) == "Read config.py (1.2KB)"

    def test_partial_file(self):
        result = {
            "success": True,
            "content": "...",
            "path": "/tmp/app.py",
            "size": 5000,
            "lines_read": 20,
            "total_lines": 200,
            "offset": 10,
        }
        assert format_tool_summary("read_file", result) == "Read app.py (lines 10-30 of 200)"

    def test_missing_keys_returns_none(self):
        assert format_tool_summary("read_file", {"success": True}) is None


class TestDiffCompare:
    def test_dict_summary(self):
        result = {
            "success": True,
            "diff": "...",
            "summary": {"added": 3, "removed": 1, "changed": 2},
            "similarity": 0.85,
        }
        assert format_tool_summary("diff_compare", result) == "+3 -1 ~2, 85% similar"

    def test_string_summary(self):
        result = {
            "success": True,
            "diff": "...",
            "summary": "No differences",
            "similarity": 1.0,
        }
        assert format_tool_summary("diff_compare", result) == "No differences, 100% similar"

    def test_zero_similarity(self):
        result = {
            "success": True,
            "diff": "...",
            "summary": {"added": 10, "removed": 10, "changed": 0},
            "similarity": 0.0,
        }
        assert format_tool_summary("diff_compare", result) == "+10 -10 ~0, 0% similar"


class TestGlob:
    def test_basic(self):
        result = {"success": True, "files": ["a.py", "b.py"], "count": 2, "truncated": False}
        assert format_tool_summary("glob", result) == "2 files matched"

    def test_zero_matches(self):
        result = {"success": True, "files": [], "count": 0, "truncated": False}
        assert format_tool_summary("glob", result) == "0 files matched"


class TestListDir:
    def test_basic(self):
        result = {
            "success": True,
            "path": "/tmp",
            "directories": [{"name": "src"}, {"name": "tests"}, {"name": "docs"}],
            "files": [{"name": "README.md"}, {"name": "setup.py"}],
            "total": 5,
        }
        assert format_tool_summary("list_dir", result) == "3 dirs, 2 files"

    def test_empty_dir(self):
        result = {"success": True, "path": "/tmp", "directories": [], "files": [], "total": 0}
        assert format_tool_summary("list_dir", result) == "0 dirs, 0 files"


class TestGrep:
    def test_basic(self):
        result = {
            "success": True,
            "matches": [
                {"file": "a.py", "line_number": 1, "content": "foo"},
                {"file": "b.py", "line_number": 5, "content": "foo"},
            ],
            "total_matches": 42,
            "files_searched": 100,
            "truncated": False,
        }
        assert format_tool_summary("grep", result) == "42 matches in 2 files"


class TestWriteFile:
    def test_created(self):
        result = {"success": True, "path": "/tmp/utils.py", "size": 256, "created": True}
        assert format_tool_summary("write_file", result) == "Created utils.py (256B)"

    def test_overwritten(self):
        result = {"success": True, "path": "/tmp/utils.py", "size": 1024, "created": False}
        assert format_tool_summary("write_file", result) == "Wrote utils.py (1.0KB)"


class TestEditFile:
    def test_single_replacement(self):
        result = {"success": True, "path": "/tmp/config.py", "replacements": 1, "size": 500}
        assert format_tool_summary("edit_file", result) == "1 replacement in config.py"

    def test_multiple_replacements(self):
        result = {"success": True, "path": "/tmp/config.py", "replacements": 3, "size": 500}
        assert format_tool_summary("edit_file", result) == "3 replacements in config.py"


class TestExecutePython:
    def test_with_output(self):
        result = {"success": True, "output": "Hello World\nline 2", "result": None, "error": "", "execution_time_ms": 10}
        assert format_tool_summary("execute_python", result) == "Hello World"

    def test_no_output(self):
        result = {"success": True, "output": "", "result": None, "error": "", "execution_time_ms": 42}
        assert format_tool_summary("execute_python", result) == "No output (42ms)"

    def test_whitespace_only_output(self):
        result = {"success": True, "output": "  \n  ", "result": None, "error": "", "execution_time_ms": 5}
        assert format_tool_summary("execute_python", result) == "No output (5ms)"


class TestShellExecutor:
    def test_with_stdout(self):
        result = {
            "success": True,
            "stdout": "v3.12.1\nmore info",
            "stderr": "",
            "return_code": 0,
            "duration": 0.5,
            "error": None,
            "truncated": False,
        }
        assert format_tool_summary("shell_executor", result) == "v3.12.1"

    def test_no_stdout(self):
        result = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
            "duration": 1.234,
            "error": None,
            "truncated": False,
        }
        assert format_tool_summary("shell_executor", result) == "Exit 0 (1.2s)"

    def test_nonzero_exit(self):
        result = {
            "success": False,
            "stdout": "",
            "stderr": "not found",
            "return_code": 127,
            "duration": 0.1,
            "error": "Command not found",
            "truncated": False,
        }
        assert format_tool_summary("shell_executor", result) == "Exit 127 (0.1s)"


class TestGetTasks:
    def test_basic(self):
        result = {"success": True, "tasks": [{"id": "1"}, {"id": "2"}], "count": 2}
        assert format_tool_summary("get_tasks", result) == "2 tasks"

    def test_zero(self):
        result = {"success": True, "tasks": [], "count": 0}
        assert format_tool_summary("get_tasks", result) == "0 tasks"


class TestGetPlan:
    def test_has_plan(self):
        result = {"success": True, "content": "# My Plan\n- Step 1"}
        assert format_tool_summary("get_plan", result) == "Plan retrieved"

    def test_no_plan(self):
        result = {"success": True, "content": "", "message": "No plan created yet"}
        assert format_tool_summary("get_plan", result) == "No plan created yet"


class TestSearchMemory:
    def test_basic(self):
        result = {"success": True, "query": "test", "items": [{"id": "1"}], "count": 1}
        assert format_tool_summary("search_memory", result) == "1 memories found"


class TestSearchArxiv:
    def test_basic(self):
        result = {"papers": [{"title": "Paper 1"}], "total_found": 10, "query": "attention"}
        assert format_tool_summary("search_arxiv", result) == "Found 10 papers"


class TestFetchArxivPaper:
    def test_basic(self):
        result = {"success": True, "paper": {"title": "Attention Is All You Need"}}
        assert format_tool_summary("fetch_arxiv_paper", result) == "Attention Is All You Need"

    def test_long_title_truncated(self):
        long_title = "A" * 200
        result = {"success": True, "paper": {"title": long_title}}
        summary = format_tool_summary("fetch_arxiv_paper", result)
        assert len(summary) <= 103  # 100 + "..."


class TestAnalyzeArxivPaper:
    def test_basic(self):
        result = {"success": True, "arxiv_id": "123", "url": "http://...", "analysis": "..."}
        assert format_tool_summary("analyze_arxiv_paper", result) == "Analysis complete"


class TestIngestToKnowledgeBase:
    def test_basic(self):
        result = {"success": True, "document_id": "abc", "title": "Paper", "chunks_created": 5}
        assert format_tool_summary("ingest_to_knowledge_base", result) == "Ingested 'Paper' (5 chunks)"


class TestRequestApproval:
    def test_approved(self):
        result = {"success": True, "approved": True, "reason": None}
        assert format_tool_summary("request_approval", result) == "Approved"

    def test_rejected(self):
        result = {"success": True, "approved": False, "reason": "Too risky"}
        assert format_tool_summary("request_approval", result) == "Rejected: Too risky"

    def test_rejected_no_reason(self):
        result = {"success": True, "approved": False, "reason": ""}
        assert format_tool_summary("request_approval", result) == "Rejected"


class TestCreateCheckpoint:
    def test_continue(self):
        result = {"success": True, "action": "continue", "edited_content": None, "feedback": None}
        assert format_tool_summary("create_checkpoint", result) == "User: continue"

    def test_abort(self):
        result = {"success": True, "action": "abort", "edited_content": None, "feedback": None}
        assert format_tool_summary("create_checkpoint", result) == "User: abort"


# ---------------------------------------------------------------------------
# Fallback / defensive behaviour
# ---------------------------------------------------------------------------


class TestFallback:
    def test_unknown_tool_returns_none(self):
        assert format_tool_summary("unknown_tool", {"foo": "bar"}) is None

    def test_missing_keys_returns_none(self):
        # Each formatter should gracefully return None on bad input
        for tool_name in [
            "read_file", "diff_compare", "glob", "grep",
            "write_file", "edit_file", "execute_python",
            "get_tasks", "get_plan", "search_memory",
            "search_arxiv", "fetch_arxiv_paper", "ingest_to_knowledge_base",
            "request_approval", "create_checkpoint",
        ]:
            result = format_tool_summary(tool_name, {})
            # Should be None (formatter caught the KeyError) or a valid string
            assert result is None or isinstance(result, str), f"{tool_name} failed defensively"


# ---------------------------------------------------------------------------
# ADK event_processor: diff_compare summary dict bug is fixed
# ---------------------------------------------------------------------------


class TestADKSummaryDictBugFix:
    """Verify the summary dict bug is fixed in ADK event processor."""

    def test_dict_summary_not_stringified(self):
        from agentic_cli.workflow.adk.event_processor import ADKEventProcessor

        proc = ADKEventProcessor(model="test")
        result = {
            "success": True,
            "diff": "...",
            "summary": {"added": 3, "removed": 1, "changed": 2},
            "similarity": 0.85,
        }
        summary = proc.generate_tool_summary("diff_compare", result, success=True)
        # Should NOT contain the raw dict repr
        assert "{'added'" not in summary
        assert "+3 -1 ~2, 85% similar" == summary

    def test_string_summary_still_works(self):
        from agentic_cli.workflow.adk.event_processor import ADKEventProcessor

        proc = ADKEventProcessor(model="test")
        # A tool that returns a string summary should still pass through
        result = {"summary": "All good", "other": "data"}
        summary = proc.generate_tool_summary("some_unknown_tool", result, success=True)
        assert summary == "All good"


# ---------------------------------------------------------------------------
# events.py: _format_result_content uses tool-specific formatters
# ---------------------------------------------------------------------------


class TestEventsFormatResultContent:
    def test_tool_specific_formatting(self):
        from agentic_cli.workflow.events import WorkflowEvent

        event = WorkflowEvent.tool_result(
            tool_name="read_file",
            result={"success": True, "content": "...", "path": "/tmp/foo.py", "size": 2048},
            success=True,
        )
        assert "Read foo.py" in event.content
        assert "2.0KB" in event.content

    def test_fallback_for_unknown_tool(self):
        from agentic_cli.workflow.events import WorkflowEvent

        event = WorkflowEvent.tool_result(
            tool_name="unknown_tool",
            result={"a": 1, "b": 2},
            success=True,
        )
        assert "Result: 2 items" == event.content

    def test_string_result_unchanged(self):
        from agentic_cli.workflow.events import WorkflowEvent

        event = WorkflowEvent.tool_result(
            tool_name="read_file",
            result="some string result",
            success=True,
        )
        assert event.content == "some string result"
