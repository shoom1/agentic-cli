"""The async kb tools wait for the knowledge base off the event loop.

``kb_read`` looked the document up, and read a PDF's text, on the loop;
``kb_write_concept`` checked its sources there; ``kb_ingest_file`` read the
file there. Each call takes the knowledge base's lock, which a thread
starting a change holds while it waits for another process's change, so
while that lasted the whole loop waited: the spinner, Ctrl+C and every
other task. They now run on a worker thread.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from agentic_cli.memory.kb import KnowledgeBase, KnowledgeBaseManager, SourceType
from agentic_cli.memory.kb import concepts as concepts_module
from agentic_cli.tools.factories import make_kb_tools


TICK = 0.01
HOLD = 0.5
# A loop that waits for the knowledge base stalls for about the whole HOLD.
# A shared CI runner's own hiccups reach tens of milliseconds (57 ms seen),
# so the bound sits well between the two.
MAX_STALL = HOLD / 4


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


def _hold_the_knowledge_base(manager: KnowledgeBaseManager, seconds: float) -> None:
    """Keep ``manager`` busy for ``seconds``: another connection holds its
    database's write lock, as another process would, and a thread starting a
    change holds the manager's lock while it waits for it."""
    held = threading.Event()

    def hold() -> None:
        conn = sqlite3.connect(manager.db_path, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            held.set()
            time.sleep(seconds)
            conn.execute("ROLLBACK")
        finally:
            conn.close()

    threading.Thread(target=hold, daemon=True).start()
    assert held.wait(5)
    threading.Thread(target=manager.delete_document, args=("no-such-document",), daemon=True).start()
    time.sleep(0.05)  # that thread now holds the manager's lock, waiting


@pytest.fixture
def manager(tmp_path) -> KnowledgeBaseManager:
    manager = KnowledgeBaseManager(tmp_path / "kb", use_mock=True)
    manager.ingest_document(content="Herons wade at dawn.", title="Herons", source_type=SourceType.USER)
    return manager


@pytest.fixture
def tools(manager) -> dict[str, Any]:
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
async def test_a_tool_waits_for_another_process_off_the_loop(tools, manager, tmp_path, call):
    notes = tmp_path / "notes.md"
    notes.write_text("Otters nest in burrows.\n")
    work = call(tools, notes)  # the coroutine; nothing has run yet
    _hold_the_knowledge_base(manager, HOLD)
    started = time.perf_counter()

    stall, result = await _longest_stall(work)

    assert time.perf_counter() - started > HOLD / 2  # it did wait for the knowledge base
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
