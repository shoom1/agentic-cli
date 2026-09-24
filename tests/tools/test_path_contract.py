"""Contract: every filesystem tool acts on exactly the path the engine checked.

For each registered tool that declares a ``filesystem.*`` capability, the
engine's resolved target for an argument must be the location the tool then
reads, writes or lists. ``~`` and relative paths are the inputs where the two
used to disagree (the engine expanded ``~`` and the tools did not).

``test_every_filesystem_tool_is_covered`` keeps this list honest: a new tool
with a filesystem capability fails it until a contract case is added here.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_cli.tools.registry import get_registry
from agentic_cli.workflow.permissions.engine import PermissionEngine
from agentic_cli.workflow.permissions.store import PermissionContext

COVERED = {
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "list_dir",
    "grep",
    "diff_compare",
    "kb_ingest_file",
    "compile_document",
    "sandbox_execute",
}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    root = tmp_path.resolve()
    home, work = root / "home", root / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    return {"home": home, "work": work}


def checked(tool: str, args: dict, env) -> list[str]:
    """The filesystem targets the engine would check for this call."""
    defn = get_registry().get(tool)
    assert defn is not None, tool
    s = MagicMock()
    s.permissions_enabled = True
    s.app_name = "contracttest"
    w = MagicMock()
    w.request_user_input = AsyncMock(return_value="Deny")
    ctx = PermissionContext(workdir=env["work"], home=env["home"], app_name="contracttest")
    engine = PermissionEngine(settings=s, workflow=w, ctx=ctx)
    return [
        rc.target for rc in engine._resolve(defn.capabilities, args)
        if rc.name.startswith("filesystem.")
    ]


def test_every_filesystem_tool_is_covered():
    import agentic_cli.tools  # noqa: F401  (registers the built-in tools)
    import agentic_cli.tools.sandbox  # noqa: F401  (lazy module)

    fs_tools = {
        d.name for d in get_registry().list_tools()
        if d.func is not None
        and getattr(d.func, "__module__", "").startswith("agentic_cli.")
        and isinstance(d.capabilities, list)
        and any(c.name.startswith("filesystem.") for c in d.capabilities)
    }
    assert fs_tools - COVERED == set(), "add a path-contract case for these tools"


class TestTildeAndRelativePaths:
    def test_read_file(self, env):
        from agentic_cli.tools.file_read import read_file
        (env["home"] / "notes.txt").write_text("from home\n")
        [target] = checked("read_file", {"path": "~/notes.txt"}, env)
        result = read_file("~/notes.txt")
        assert result["success"] is True, result
        assert result["path"] == target == str(env["home"] / "notes.txt")
        assert result["content"] == "from home\n"

    def test_write_file(self, env):
        from agentic_cli.tools.file_write import write_file
        [target] = checked("write_file", {"path": "~/new.txt"}, env)
        result = write_file("~/new.txt", "hello")
        assert result["success"] is True, result
        assert result["path"] == target == str(env["home"] / "new.txt")
        assert (env["home"] / "new.txt").read_text() == "hello"
        assert not (env["work"] / "~").exists(), "no literal '~' directory"

    def test_edit_file(self, env):
        from agentic_cli.tools.file_write import edit_file
        (env["home"] / "e.txt").write_text("old value\n")
        targets = checked("edit_file", {"path": "~/e.txt"}, env)
        result = edit_file("~/e.txt", "old", "new")
        assert result["success"] is True, result
        assert set(targets) == {result["path"]} == {str(env["home"] / "e.txt")}
        assert (env["home"] / "e.txt").read_text() == "new value\n"

    def test_glob(self, env):
        from agentic_cli.tools.glob_tool import glob
        (env["home"] / "d").mkdir()
        (env["home"] / "d" / "a.txt").write_text("x")
        [target] = checked("glob", {"path": "~/d"}, env)
        result = glob(pattern="*.txt", path="~/d")
        assert result["success"] is True, result
        assert result["path"] == target == str(env["home"] / "d")
        assert result["count"] == 1

    def test_list_dir(self, env):
        from agentic_cli.tools.glob_tool import list_dir
        (env["home"] / "d").mkdir()
        (env["home"] / "d" / "a.txt").write_text("x")
        [target] = checked("list_dir", {"path": "~/d"}, env)
        result = list_dir(path="~/d")
        assert result["success"] is True, result
        assert result["path"] == target == str(env["home"] / "d")

    def test_grep(self, env):
        from agentic_cli.tools.grep_tool import grep
        (env["home"] / "d").mkdir()
        (env["home"] / "d" / "a.txt").write_text("needle-4711\n")
        [target] = checked("grep", {"pattern": "needle-4711", "path": "~/d"}, env)
        assert target == str(env["home"] / "d")
        result = grep("needle-4711", path="~/d", output_mode="files")
        assert result["success"] is True, result
        assert result["total_matches"] >= 1, result

    def test_diff_compare_reads_the_files_it_was_checked_for(self, env):
        from agentic_cli.tools.file_read import diff_compare
        (env["home"] / "a.txt").write_text("alpha\n")
        (env["home"] / "b.txt").write_text("beta\n")
        targets = checked("diff_compare", {"source_a": "~/a.txt", "source_b": "~/b.txt"}, env)
        assert targets == [str(env["home"] / "a.txt"), str(env["home"] / "b.txt")]
        result = diff_compare("~/a.txt", "~/b.txt")
        assert result["success"] is True, result
        assert "-alpha" in result["diff"] and "+beta" in result["diff"], result["diff"]

    def test_diff_compare_still_compares_plain_text(self, env):
        from agentic_cli.tools.file_read import diff_compare
        result = diff_compare("one\n", "two\n")
        assert result["success"] is True
        assert "-one" in result["diff"] and "+two" in result["diff"]

    def test_kb_ingest_file(self, env, monkeypatch):
        from agentic_cli.tools import knowledge_tools
        (env["home"] / "doc.txt").write_bytes(b"kb bytes 4711")
        [target] = checked("kb_ingest_file", {"path": "~/doc.txt"}, env)
        assert target == str(env["home"] / "doc.txt")
        seen: dict = {}

        async def capture(kb_manager, **kw):
            seen.update(kw)
            return {"success": True}

        monkeypatch.setattr(knowledge_tools, "_finalize_ingest", capture)
        result = asyncio.run(knowledge_tools._ingest_file_with_kb(None, path="~/doc.txt"))
        assert result["success"] is True
        assert seen["file_bytes"] == b"kb bytes 4711"

    def test_compile_document_finds_the_source_it_was_checked_for(self, env):
        from agentic_cli.tools.document import compile_document
        (env["home"] / "doc.tex").write_text("\\documentclass{article}\n")
        args = {"source_path": "~/doc.tex", "engine": "no-such-tex-engine"}
        # source_path is gated as document.compile today; the tool must still
        # open the same location a path matcher would resolve.
        from agentic_cli.paths import resolve_path
        result = compile_document(**args)
        assert "Source not found" not in (result.get("error") or ""), result
        assert "No LaTeX engine" in (result.get("error") or ""), result
        assert str(resolve_path("~/doc.tex")) == str(env["home"] / "doc.tex")

    def test_compile_document_assets_dir_matches_the_checked_target(self, env):
        from agentic_cli.tools.document.compile import _build_env
        (env["home"] / "assets").mkdir()
        [target] = checked(
            "compile_document",
            {"source_path": "doc.tex", "assets_dir": "~/assets"},
            env,
        )
        roots = _build_env("~/assets")["TEXINPUTS"].split(":")
        assert target == str(env["home"] / "assets")
        assert target in roots

    def test_sandbox_inputs_stage_the_checked_file(self, env):
        from agentic_cli.tools.sandbox.manager import stage_inputs
        (env["home"] / "in.csv").write_text("a,b\n")
        [target] = checked(
            "sandbox_execute", {"code": "", "inputs": ["~/in.csv"]}, env
        )
        assert target == str(env["home"] / "in.csv")
        session = env["work"] / "session"
        session.mkdir()
        stage_inputs(session, ["~/in.csv"])
        assert (session / "inputs" / "in.csv").read_text() == "a,b\n"

    def test_relative_paths_agree(self, env):
        from agentic_cli.tools.file_read import read_file
        (env["work"] / "sub").mkdir()
        (env["work"] / "sub" / "f.txt").write_text("rel\n")
        [target] = checked("read_file", {"path": "sub/../sub/f.txt"}, env)
        result = read_file("sub/../sub/f.txt")
        assert result["path"] == target == str(env["work"] / "sub" / "f.txt")
