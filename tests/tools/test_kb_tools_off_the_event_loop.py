"""The async kb tools wait for the knowledge base off the event loop.

``kb_read`` looked the document up, and read a PDF's text, on the loop;
``kb_write_concept`` checked its sources there; ``kb_ingest_file`` read the
file there. Each lookup holds the knowledge base's folder lock, so while
another process held it (re-embedding on open, say) the whole loop waited:
the spinner, Ctrl+C and every other task. They now run on a worker thread.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from agentic_cli.memory.kb import KnowledgeBase, KnowledgeBaseManager, SourceType
from agentic_cli.memory.kb import concepts as concepts_module
from agentic_cli.tools.factories import make_kb_tools

fcntl = pytest.importorskip("fcntl")

TICK = 0.01
MAX_STALL = 0.05
HOLD = 0.5


async def _longest_stall(work) -> tuple[float, Any]:
    """Await ``work`` beside a ticker; the longest the ticker waited beyond its tick, and the result."""
    stalls: list[float] = []
    stop = asyncio.Event()

    async def tick() -> None:
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(TICK)
            now = time.perf_counter()
            stalls.append(now - last - TICK)
            last = now

    ticker = asyncio.create_task(tick())
    await asyncio.sleep(2 * TICK)
    try:
        result = await work
    finally:
        stop.set()
        await ticker
    return max(stalls), result


def _hold_folder_lock(kb_dir: Path, seconds: float) -> None:
    """Hold the folder lock from another thread for ``seconds``, as another process would."""
    held = threading.Event()

    def hold() -> None:
        fd = os.open(kb_dir / ".lock", os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            held.set()
            time.sleep(seconds)
        finally:
            os.close(fd)

    threading.Thread(target=hold, daemon=True).start()
    held.wait(5)


@pytest.fixture
def tools(tmp_path) -> dict[str, Any]:
    manager = KnowledgeBaseManager(tmp_path / "kb", use_mock=True)
    manager.ingest_document(content="Herons wade at dawn.", title="Herons", source_type=SourceType.USER)
    return {t.__name__: t for t in make_kb_tools(KnowledgeBase({"project": manager}))}


def _doc_id(tools) -> str:
    return tools["kb_list"]()["documents"][0]["id"]


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda tools, path: tools["kb_read"]("Herons", full=True), id="kb_read-full"),
        pytest.param(lambda tools, path: tools["kb_read"]("Herons"), id="kb_read-sidecar"),
        pytest.param(
            lambda tools, path: tools["kb_write_concept"]("Wading birds", "Herons wade.", [_doc_id(tools)]),
            id="kb_write_concept",
        ),
        pytest.param(lambda tools, path: tools["kb_ingest_file"](str(path)), id="kb_ingest_file"),
    ],
)
async def test_a_tool_waits_for_another_process_off_the_loop(tools, tmp_path, call):
    notes = tmp_path / "notes.md"
    notes.write_text("Otters nest in burrows.\n")
    work = call(tools, notes)  # the coroutine; nothing has run yet
    _hold_folder_lock(tmp_path / "kb", HOLD)

    stall, result = await _longest_stall(work)

    assert result["success"] is True, result
    assert stall < MAX_STALL


async def test_two_concept_pages_written_at_once_get_their_own_slugs(tools, monkeypatch):
    """The pages are written on worker threads: two with one title must still
    not pick the same free slug and write over each other."""
    real = concepts_module.ConceptStore._resolve_collision

    def slow(self, base_slug):
        slug = real(self, base_slug)
        time.sleep(0.1)  # both would pick the free slug before either writes
        return slug

    monkeypatch.setattr(concepts_module.ConceptStore, "_resolve_collision", slow)
    doc_id = _doc_id(tools)

    first, second = await asyncio.gather(
        tools["kb_write_concept"]("Wading birds", "Herons wade.", [doc_id]),
        tools["kb_write_concept"]("Wading birds", "Egrets wade too.", [doc_id]),
    )

    assert {first["slug"], second["slug"]} == {"wading-birds", "wading-birds-2"}
