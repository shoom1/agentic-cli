"""Real BM25 index backends (bm25s, rank_bm25).

Both implement the same interface as MockBM25Index so create_bm25_index()
can return any of them interchangeably. Tokenization is lowercase whitespace
split to match MockBM25Index; the scoring model is the library's own BM25.

Neither underlying library supports true incremental add/remove, so we keep
tokenized docs + chunk_ids in memory and rebuild the model lazily on search.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentic_cli.file_utils import atomic_write_json


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class _BM25BackendBase:
    _INDEX_FILE: str = ""

    def __init__(self):
        self._chunk_ids: list[str] = []
        self._tokenized: list[list[str]] = []
        self._model = None

    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    def add_documents(self, chunk_ids: list[str], texts: list[str]) -> None:
        for cid, text in zip(chunk_ids, texts):
            self._chunk_ids.append(cid)
            self._tokenized.append(_tokenize(text))
        self._model = None

    def remove_documents(self, chunk_ids: list[str]) -> None:
        remove_set = set(chunk_ids)
        keep = [
            (cid, toks)
            for cid, toks in zip(self._chunk_ids, self._tokenized)
            if cid not in remove_set
        ]
        if keep:
            self._chunk_ids, self._tokenized = map(list, zip(*keep))
        else:
            self._chunk_ids, self._tokenized = [], []
        self._model = None

    def rebuild(self, chunk_ids: list[str], texts: list[str]) -> None:
        self._chunk_ids = []
        self._tokenized = []
        self._model = None
        self.add_documents(chunk_ids, texts)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            path / self._INDEX_FILE,
            {"chunk_ids": self._chunk_ids, "tokenized": self._tokenized},
        )

    def load(self, path: Path) -> None:
        index_path = path / self._INDEX_FILE
        if not index_path.exists():
            return
        data = json.loads(index_path.read_text())
        self._chunk_ids = data["chunk_ids"]
        self._tokenized = data["tokenized"]
        self._model = None


class RankBM25Index(_BM25BackendBase):
    """BM25 backed by rank_bm25.BM25Plus (pure Python).

    BM25Plus rather than BM25Okapi because Okapi's IDF can go zero or
    negative when a term appears in most of the corpus; BM25Plus adds a
    delta offset that guarantees positive contributions on real matches.
    """

    _INDEX_FILE = "bm25_rank.json"

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._chunk_ids:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        query_set = set(query_tokens)
        if self._model is None:
            from rank_bm25 import BM25Plus

            self._model = BM25Plus(self._tokenized)
        scores = self._model.get_scores(query_tokens)
        scored: list[tuple[str, float]] = []
        for cid, doc_tokens, score in zip(
            self._chunk_ids, self._tokenized, scores
        ):
            if query_set.intersection(doc_tokens):
                scored.append((cid, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class BM25sIndex(_BM25BackendBase):
    """BM25 backed by bm25s (NumPy/C-accelerated)."""

    _INDEX_FILE = "bm25s_sidecar.json"

    def _build_model(self) -> None:
        import bm25s

        model = bm25s.BM25()
        model.index(self._tokenized, show_progress=False)
        self._model = model

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._chunk_ids:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        if self._model is None:
            self._build_model()
        k = min(top_k, len(self._chunk_ids))
        docs, scores = self._model.retrieve(
            [query_tokens], k=k, show_progress=False
        )
        results: list[tuple[str, float]] = []
        for idx, score in zip(docs[0], scores[0]):
            if score > 0:
                results.append((self._chunk_ids[int(idx)], float(score)))
        return results
