"""Mock BM25 index for testing without bm25s dependency."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from agentic_cli.file_utils import atomic_write_json


class MockBM25Index:
    """Simple BM25-like index using term frequency scoring.

    Provides the same interface as BM25Index but with no external dependencies.
    Uses a simplified TF-IDF scoring suitable for testing and small corpora.
    """

    def __init__(self):
        self._chunk_ids: list[str] = []
        self._documents: list[list[str]] = []  # tokenized docs

    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    def add_documents(self, chunk_ids: list[str], texts: list[str]) -> None:
        for chunk_id, text in zip(chunk_ids, texts):
            self._chunk_ids.append(chunk_id)
            self._documents.append(self._tokenize(text))

    def remove_documents(self, chunk_ids: list[str]) -> None:
        remove_set = set(chunk_ids)
        pairs = [
            (cid, doc)
            for cid, doc in zip(self._chunk_ids, self._documents)
            if cid not in remove_set
        ]
        if pairs:
            self._chunk_ids, self._documents = map(list, zip(*pairs))
        else:
            self._chunk_ids, self._documents = [], []

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._documents:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        n = len(self._documents)
        df: dict[str, int] = {}
        for token in query_tokens:
            df[token] = sum(1 for doc in self._documents if token in doc)

        scored: list[tuple[str, float]] = []
        for chunk_id, doc_tokens in zip(self._chunk_ids, self._documents):
            score = 0.0
            tf = Counter(doc_tokens)
            for token in query_tokens:
                if df.get(token, 0) == 0:
                    continue
                idf = math.log((n + 1) / (df[token] + 1)) + 1
                score += tf.get(token, 0) * idf
            if score > 0:
                scored.append((chunk_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def rebuild(self, chunk_ids: list[str], texts: list[str]) -> None:
        self._chunk_ids = []
        self._documents = []
        self.add_documents(chunk_ids, texts)

    def save(self, path: Path) -> None:
        data = {
            "chunk_ids": self._chunk_ids,
            "documents": self._documents,
        }
        atomic_write_json(path / "bm25_index.json", data)

    def load(self, path: Path) -> None:
        index_path = path / "bm25_index.json"
        if index_path.exists():
            data = json.loads(index_path.read_text())
            self._chunk_ids = data["chunk_ids"]
            self._documents = data["documents"]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()
