"""The search vectors one session holds: every stored vector of its model.

Loaded from the database (``chunks.vector``, normalized float32) into one
matrix, and searched with one matrix product. FAISS's flat index, which the
knowledge base used before, computes the same product; numpy gives the same
results without a second file to keep in step with the database.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def normalized(vectors) -> np.ndarray:
    """``vectors`` as float32 rows of length 1 (a zero row stays zero)."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def to_blob(vector) -> bytes:
    """How a vector is stored: little-endian float32."""
    return np.asarray(vector, dtype="<f4").tobytes()


class VectorIndex:
    """The vectors of one model, as of one ``generation`` of the database."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.generation: int | None = None
        self._rowids = np.empty(0, dtype=np.int64)
        self._documents: list[str] = []
        self._matrix = np.empty((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._documents)

    def load(self, rows: Iterable[tuple[int, str, bytes]], generation: int) -> None:
        """Hold exactly ``rows`` (rowid, document ID, vector). A vector of another size is left out."""
        size = self.dim * 4
        kept = [(rowid, doc, blob) for rowid, doc, blob in rows if isinstance(blob, bytes) and len(blob) == size]
        self._rowids = np.array([rowid for rowid, _, _ in kept], dtype=np.int64)
        self._documents = [str(doc) for _, doc, _ in kept]
        self._matrix = np.frombuffer(b"".join(blob for _, _, blob in kept), dtype="<f4").reshape(len(kept), self.dim).astype(np.float32)
        self.generation = generation

    def apply(self, generation: int, *, add: tuple[Sequence[int], Sequence[str], np.ndarray] | None = None, remove: Iterable[int] = ()) -> None:
        """Apply this session's own change, which made the database ``generation``.

        Only when nothing else changed the vectors since this index was
        loaded or last changed; otherwise the index stays as it is, and the
        next search reloads it (its generation no longer matches).
        """
        if self.generation is None or generation != self.generation + 1:
            return
        removed = np.asarray(list(remove), dtype=np.int64)
        if removed.size:
            keep = ~np.isin(self._rowids, removed)
            self._rowids, self._matrix = self._rowids[keep], self._matrix[keep]
            self._documents = [doc for doc, k in zip(self._documents, keep) if k]
        if add is not None:
            rowids, documents, vectors = add
            self._rowids = np.concatenate([self._rowids, np.asarray(rowids, dtype=np.int64)])
            self._matrix = np.vstack([self._matrix, np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)])
            self._documents.extend(str(doc) for doc in documents)
        self.generation = generation

    def ranked(self, query: np.ndarray, allowed: set[str] | None = None) -> list[tuple[int, str]]:
        """Every chunk as (rowid, document ID), most similar to ``query`` (normalized) first."""
        if not self._documents:
            return []
        scores = self._matrix @ np.asarray(query, dtype=np.float32)
        order = np.argsort(-scores, kind="stable")
        return [
            (int(self._rowids[i]), self._documents[i])
            for i in order
            if allowed is None or self._documents[i] in allowed
        ]
