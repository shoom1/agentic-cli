"""MockVectorStore: a vector store without FAISS, for tests and mock mode."""

import json
from pathlib import Path

import numpy as np

from agentic_cli.file_utils import atomic_write_json


class MockVectorStore:
    """Mock vector store for testing without FAISS."""

    def __init__(
        self,
        index_path: Path,
        embedding_dim: int = 384,
    ) -> None:
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self._vectors: dict[str, list[float]] = {}

    @property
    def size(self) -> int:
        return len(self._vectors)

    def add_embeddings(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
    ) -> None:
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            self._vectors[chunk_id] = embedding

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        if not self._vectors:
            return []

        query = np.array(query_embedding)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query = query / query_norm

        results: list[tuple[str, float]] = []
        for chunk_id, embedding in self._vectors.items():
            vec = np.array(embedding)
            vec_norm = np.linalg.norm(vec)
            if vec_norm > 0:
                vec = vec / vec_norm
                score = float(np.dot(query, vec))
                results.append((chunk_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def remove_embeddings(self, chunk_ids: list[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            if chunk_id in self._vectors:
                del self._vectors[chunk_id]
                removed += 1
        return removed

    def rebuild(self) -> None:
        """No-op for mock store."""

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vectors": self._vectors,
            "embedding_dim": self.embedding_dim,
        }
        atomic_write_json(self.index_path, data)

    def load(self) -> None:
        if self.index_path.exists():
            data = json.loads(self.index_path.read_text())
            self._vectors = data["vectors"]
            self.embedding_dim = data["embedding_dim"]

    def clear(self) -> None:
        self._vectors = {}
