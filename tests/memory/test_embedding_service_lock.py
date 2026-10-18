"""One EmbeddingService serves every component, from several threads.

The knowledge base embeds on worker threads while the memory tools embed on
another, and a sentence-transformers model is not safe to call from two
threads at once. The service takes a lock around loading the model and
around ``embed_text`` and ``embed_batch``.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np

from agentic_cli.memory import EmbeddingService


class _Model:
    """A model that records how many ``encode`` calls overlap."""

    def __init__(self) -> None:
        self.active = 0
        self.most = 0
        self._guard = threading.Lock()

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, **kwargs):
        with self._guard:
            self.active += 1
            self.most = max(self.most, self.active)
        time.sleep(0.01)
        with self._guard:
            self.active -= 1
        rows = np.zeros((1 if isinstance(texts, str) else len(texts), 3), dtype="float32")
        return rows[0] if isinstance(texts, str) else rows


def _run(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_concurrent_calls_on_one_service_never_overlap():
    model = _Model()
    service = EmbeddingService(model_name="fake", device="cpu")
    service._model = model
    service._embedding_dim = 3

    def work(i: int) -> None:
        if i % 2:
            service.embed_text(f"text {i}")
        else:
            service.embed_batch([f"a {i}", f"b {i}"])

    _run([threading.Thread(target=work, args=(i,)) for i in range(8)])

    assert model.most == 1


def test_the_model_loads_once_when_threads_ask_for_it_together(monkeypatch):
    loads: list[str] = []

    class SentenceTransformer:
        def __init__(self, name, device=None):
            loads.append(name)
            time.sleep(0.02)

        def get_sentence_embedding_dimension(self):
            return 3

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=SentenceTransformer)
    )
    service = EmbeddingService(model_name="fake", device="cpu")

    _run([threading.Thread(target=lambda: service.embedding_dim) for _ in range(6)])

    assert loads == ["fake"]
    assert service.embedding_dim == 3
