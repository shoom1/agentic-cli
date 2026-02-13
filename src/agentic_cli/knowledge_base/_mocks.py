"""Mock services for knowledge base testing and development without ML dependencies.

These are lightweight replacements for EmbeddingService and VectorStore
that work without sentence-transformers or FAISS installed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from agentic_cli.persistence._utils import atomic_write_json


class MockEmbeddingService:
    """Mock embedding service for testing without loading models."""

    def __init__(
        self,
        model_name: str = "mock-model",
        batch_size: int = 32,
        embedding_dim: int = 384,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed_text(self, text: str) -> list[float]:
        # Generate deterministic mock embedding based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        embedding = []
        for i in range(0, len(text_hash), 2):
            byte_val = int(text_hash[i : i + 2], 16)
            embedding.append((byte_val / 255.0) - 0.5)

        while len(embedding) < self._embedding_dim:
            embedding.extend(embedding[: self._embedding_dim - len(embedding)])

        return embedding[: self._embedding_dim]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def chunk_document(
        self,
        content: str,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> list[str]:
        if not content:
            return []

        chunks = []
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunks.append(content[start:end])
            start = end - overlap if end < len(content) else end

        return chunks


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
