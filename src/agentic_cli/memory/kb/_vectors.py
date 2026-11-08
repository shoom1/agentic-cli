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
    """The vectors of one model, as of one ``generation`` of the database.

    Each row keeps its chunk's rowid and its document as a number (``_codes``,
    an index into ``_names``), so a search keeps the rows of some documents
    without comparing strings row by row.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.generation: int | None = None
        self._rowids = np.empty(0, dtype=np.int64)
        self._codes = np.empty(0, dtype=np.int64)
        self._names: list[str] = []
        self._code_of: dict[str, int] = {}
        self._matrix = np.empty((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._rowids)

    def _encode(self, documents: Iterable[str]) -> np.ndarray:
        """Each document ID's number, a new one for an ID not seen before."""
        codes = []
        for doc in documents:
            code = self._code_of.get(doc)
            if code is None:
                code = self._code_of[doc] = len(self._names)
                self._names.append(doc)
            codes.append(code)
        return np.asarray(codes, dtype=np.int64)

    def load(self, rows: Iterable[tuple[int, str, bytes]], generation: int) -> None:
        """Hold exactly ``rows`` (rowid, document ID, vector), normalized again.

        A stored vector is not trusted to be normalized (a project can ship
        its database): one of another size, or holding NaN or infinity (or
        too large to measure), is left out, so its chunk is found by keyword
        only.
        """
        size = self.dim * 4
        kept = [(rowid, doc, blob) for rowid, doc, blob in rows if isinstance(blob, bytes) and len(blob) == size]
        matrix = np.frombuffer(b"".join(blob for _, _, blob in kept), dtype="<f4").reshape(len(kept), self.dim)
        with np.errstate(over="ignore", invalid="ignore"):
            usable = np.isfinite(np.linalg.norm(matrix, axis=1))
        self._names, self._code_of = [], {}
        self._rowids = np.array([rowid for rowid, _, _ in kept], dtype=np.int64)[usable]
        self._codes = self._encode(str(doc) for _, doc, _ in kept)[usable]
        self._matrix = normalized(matrix[usable])
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
            self._rowids, self._codes, self._matrix = self._rowids[keep], self._codes[keep], self._matrix[keep]
        if add is not None:
            rowids, documents, vectors = add
            matrix = np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)
            usable = np.isfinite(matrix).all(axis=1)
            self._rowids = np.concatenate([self._rowids, np.asarray(rowids, dtype=np.int64)[usable]])
            self._matrix = np.vstack([self._matrix, matrix[usable]])
            self._codes = np.concatenate([self._codes, self._encode(str(doc) for doc in documents)[usable]])
        self.generation = generation

    def ranked(self, query: np.ndarray, allowed: set[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Every chunk, most similar to ``query`` (normalized) first, as two
        arrays: the rowids and their document IDs. ``allowed`` keeps only
        those documents' chunks."""
        scores = self._matrix @ np.asarray(query, dtype=np.float32)
        order = np.argsort(-scores, kind="stable")
        if allowed is not None:
            kept = np.fromiter((name in allowed for name in self._names), dtype=bool, count=len(self._names))
            order = order[kept[self._codes[order]]]
        names = np.array(self._names if self._names else [""], dtype=str)
        return self._rowids[order], names[self._codes[order]]
