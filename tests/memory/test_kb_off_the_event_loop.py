"""The knowledge base's async methods never block the event loop.

``KnowledgeBase._store`` called the synchronous ``ingest_document`` on the
loop (only conversion ran on a worker thread), and ``sidecar`` read the
document and its file there, so embedding every chunk (and the model's first
load), and a wait for another process's lock, froze the loop: the spinner,
Ctrl+C and every other task. Each synchronous manager call reached
from an async method now runs on a worker thread.

A ticker task measures how long the loop goes without running it, beyond its
own sleep.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

import pytest

from agentic_cli.memory import MockEmbeddingService
from agentic_cli.memory.kb import KnowledgeBase, KnowledgeBaseManager, SourceType

TICK = 0.01
# Far below what a blocked loop shows (the whole embedding, or the whole
# hold of the lock: several tenths of a second here).
MAX_STALL = 0.05
# Chunking a long text is Python code on the worker thread, which holds the
# GIL between switches: on a loaded machine the loop has waited 52 ms for it.
# A loop that embeds on its own thread stalls 0.43 s and more here.
MAX_STALL_WHILE_CHUNKING = 0.2
HOLD = 0.5


class _SlowEmbeddings(MockEmbeddingService):
    """Mock embeddings that take the time a CPU model would."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        time.sleep(0.002 * len(texts) + 0.3)
        return super().embed_batch(texts)


async def _longest_stall(work: Awaitable[Any]) -> tuple[float, Any]:
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


def _hold_write_lock(kb_dir: Path, seconds: float) -> threading.Event:
    """Hold the knowledge base's write lock for ``seconds`` from another thread, as another process would.

    SQLite's lock belongs to the connection, so another connection in this
    process waits for it as another process's would. Returns once it is held.
    """
    held = threading.Event()

    def hold() -> None:
        conn = sqlite3.connect(kb_dir / "kb.sqlite", isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            held.set()
            time.sleep(seconds)
            conn.execute("ROLLBACK")
        finally:
            conn.close()

    threading.Thread(target=hold, daemon=True).start()
    held.wait(5)
    return held


async def test_embedding_runs_off_the_loop(tmp_path):
    kb = KnowledgeBase.open({"project": tmp_path / "kb"}, embedding_service=_SlowEmbeddings())
    text = "\n\n".join(
        " ".join(f"Sentence {i} of paragraph {p} about otters." for i in range(10)) for p in range(100)
    )

    stall, found = await _longest_stall(kb.ingest_text(text, title="long"))

    assert len(found.document.chunks) > 50
    assert stall < MAX_STALL_WHILE_CHUNKING


async def test_an_ingest_waits_for_another_process_off_the_loop(tmp_path):
    kb = KnowledgeBase.open({"project": tmp_path / "kb"}, use_mock=True)
    _hold_write_lock(tmp_path / "kb", HOLD)
    started = time.perf_counter()

    stall, found = await _longest_stall(kb.ingest_text("A short note about herons.", title="short"))

    assert time.perf_counter() - started > HOLD / 2  # it did wait for the lock
    assert kb.get_document(found.document.id) is not None
    assert stall < MAX_STALL


async def test_a_sidecar_waits_off_the_loop_while_another_thread_holds_the_knowledge_base(tmp_path):
    """A thread starting a change holds the manager's lock while it waits for
    the write lock; writing a missing sidecar must not wait for it on the loop."""
    manager = KnowledgeBaseManager(tmp_path / "kb", use_mock=True)
    doc = manager.ingest_document(content="Herons wade at dawn.", title="Herons", source_type=SourceType.USER)
    manager.sidecar_path(doc.id).unlink()
    kb = KnowledgeBase({"project": manager})
    found = kb.get_document(doc.id)
    _hold_write_lock(tmp_path / "kb", HOLD)
    waiting = threading.Thread(target=manager.delete_document, args=("no-such-document",), daemon=True)
    waiting.start()
    await asyncio.sleep(0.05)  # the thread now holds the manager's lock, waiting

    stall, sidecar = await _longest_stall(kb.sidecar(found))

    waiting.join(5)
    assert "Herons" in sidecar
    assert stall < MAX_STALL


async def test_backfill_waits_off_the_loop(tmp_path):
    manager = KnowledgeBaseManager(tmp_path / "kb", use_mock=True)
    doc = manager.ingest_document(content="Herons wade at dawn.", title="Herons", source_type=SourceType.USER)
    manager.sidecar_path(doc.id).unlink()
    kb = KnowledgeBase({"project": manager})
    _hold_write_lock(tmp_path / "kb", HOLD)

    stall, [result] = await _longest_stall(kb.backfill_sidecars())

    assert (result.scope, result.written, result.error) == ("project", 1, None)
    assert stall < MAX_STALL


# -- a cancelled call ---------------------------------------------------------
#
# Cancelling a coroutine does not stop the worker thread it waits on. A
# sidecar's lock, or backfill's running flag, released by the cancelled
# coroutine let a second call start while the first one's thread still wrote:
# it found no sidecar yet and called the summarizer again.


class _CountingSummarizer:
    def __init__(self) -> None:
        self.calls = 0

    async def summarize(self, content: str, prompt: str) -> str:
        self.calls += 1
        return "SUMMARY: Herons wade at dawn."


def _slow_sidecar_writes(manager: KnowledgeBaseManager, monkeypatch) -> threading.Event:
    """Make the manager's sidecar write take 0.3 s; the event is set when one starts."""
    started = threading.Event()
    write = manager.write_sidecar_if_present

    def slow(doc, payload):
        started.set()
        time.sleep(0.3)
        return write(doc, payload)

    monkeypatch.setattr(manager, "write_sidecar_if_present", slow)
    return started


def _without_sidecar(tmp_path, summarizer) -> tuple[KnowledgeBaseManager, KnowledgeBase]:
    manager = KnowledgeBaseManager(tmp_path / "kb", use_mock=True, summarizer=summarizer)
    doc = manager.ingest_document(content="Herons wade at dawn.", title="Herons", source_type=SourceType.USER)
    manager.sidecar_path(doc.id).unlink()
    return manager, KnowledgeBase({"project": manager})


async def _cancel_once_writing(task: asyncio.Task, started: threading.Event) -> None:
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_sidecar_cancelled_while_it_writes_is_not_written_twice(tmp_path, monkeypatch):
    summarizer = _CountingSummarizer()
    manager, kb = _without_sidecar(tmp_path, summarizer)
    started = _slow_sidecar_writes(manager, monkeypatch)
    [found] = kb.list_documents()

    await _cancel_once_writing(asyncio.create_task(kb.sidecar(found)), started)
    sidecar = await kb.sidecar(found)

    assert "Herons wade at dawn." in sidecar
    assert summarizer.calls == 1


async def test_a_backfill_cancelled_while_it_writes_holds_off_the_next_until_the_write_ends(
    tmp_path, monkeypatch
):
    from agentic_cli.memory.kb import BackfillAlreadyRunning

    summarizer = _CountingSummarizer()
    manager, kb = _without_sidecar(tmp_path, summarizer)
    started = _slow_sidecar_writes(manager, monkeypatch)

    await _cancel_once_writing(asyncio.create_task(manager.backfill_sidecars()), started)
    with pytest.raises(BackfillAlreadyRunning):
        await manager.backfill_sidecars()
    for _ in range(100):  # the write ends within 0.3 s, and lets go then
        if not manager._backfill_running:
            break
        await asyncio.sleep(0.01)

    assert await manager.backfill_sidecars() == 0
    assert summarizer.calls == 1
