"""Vector store for semantic search.

Provides FAISS-based vector storage and similarity search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import faiss


class VectorStore:
    """FAISS-based vector store for semantic search.

    Stores embeddings and provides efficient similarity search.
    Supports persistence to disk.
    """

    def __init__(
        self,
        index_path: Path,
        embedding_dim: int = 384,
    ) -> None:
        """Initialize the vector store.

        Args:
            index_path: Path to store/load the FAISS index.
            embedding_dim: Dimension of embedding vectors.
        """
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self._index: faiss.Index | None = None
        self._id_map: dict[int, str] = {}  # FAISS internal ID -> chunk_id
        self._chunk_to_faiss: dict[str, int] = {}  # chunk_id -> FAISS internal ID
        self._next_id: int = 0

        # Load existing index if available
        if self.index_path.exists():
            self.load()

    @property
    def index(self) -> faiss.Index:
        """Get the FAISS index (lazy initialization)."""
        if self._index is None:
            import faiss

            # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
            self._index = faiss.IndexFlatIP(self.embedding_dim)
        return self._index

    @property
    def size(self) -> int:
        """Get the number of vectors in the index."""
        return self.index.ntotal

    def add_embeddings(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Add embeddings to the index.

        Args:
            chunk_ids: List of chunk IDs corresponding to embeddings.
            embeddings: List of embedding vectors.

        Raises:
            ValueError: If chunk_ids and embeddings have different lengths.
        """
        if len(chunk_ids) != len(embeddings):
            raise ValueError(
                f"chunk_ids ({len(chunk_ids)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )

        if not embeddings:
            return

        # Convert to numpy and normalize for cosine similarity
        vectors = np.array(embeddings, dtype=np.float32)
        faiss_module = self._get_faiss()
        faiss_module.normalize_L2(vectors)

        # Add to index
        self.index.add(vectors)

        # Update ID mappings
        for chunk_id in chunk_ids:
            self._id_map[self._next_id] = chunk_id
            self._chunk_to_faiss[chunk_id] = self._next_id
            self._next_id += 1

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search for similar vectors.

        Args:
            query_embedding: Query embedding vector.
            top_k: Maximum number of results to return.

        Returns:
            List of (chunk_id, score) tuples, sorted by descending score.
        """
        if self.size == 0:
            return []

        # Normalize query vector
        query = np.array([query_embedding], dtype=np.float32)
        faiss_module = self._get_faiss()
        faiss_module.normalize_L2(query)

        # Search
        k = min(top_k, self.size)
        scores, indices = self.index.search(query, k)

        # Convert to results
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in self._id_map:  # -1 indicates no result
                chunk_id = self._id_map[idx]
                results.append((chunk_id, float(score)))

        return results

    def remove_embeddings(self, chunk_ids: list[str]) -> int:
        """Remove embeddings by chunk ID.

        Note: FAISS IndexFlat doesn't support direct removal.
        This marks IDs as removed; call rebuild() to actually remove.

        Args:
            chunk_ids: List of chunk IDs to remove.

        Returns:
            Number of embeddings marked for removal.
        """
        removed = 0
        for chunk_id in chunk_ids:
            if chunk_id in self._chunk_to_faiss:
                faiss_id = self._chunk_to_faiss.pop(chunk_id)
                self._id_map.pop(faiss_id, None)
                removed += 1
        return removed

    def save(self) -> None:
        """Persist index and mappings to disk."""
        if self._index is None:
            return

        import faiss

        # Ensure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(self.index_path))

        # Save ID mappings
        mappings_path = self.index_path.with_suffix(".mappings.json")
        mappings = {
            "id_map": {str(k): v for k, v in self._id_map.items()},
            "chunk_to_faiss": self._chunk_to_faiss,
            "next_id": self._next_id,
        }
        mappings_path.write_text(json.dumps(mappings, indent=2))

    def load(self) -> None:
        """Load index and mappings from disk."""
        if not self.index_path.exists():
            return

        import faiss

        # Load FAISS index
        self._index = faiss.read_index(str(self.index_path))

        # Load ID mappings
        mappings_path = self.index_path.with_suffix(".mappings.json")
        if mappings_path.exists():
            mappings = json.loads(mappings_path.read_text())
            self._id_map = {int(k): v for k, v in mappings["id_map"].items()}
            self._chunk_to_faiss = mappings["chunk_to_faiss"]
            self._next_id = mappings["next_id"]

    def clear(self) -> None:
        """Clear all vectors from the index."""
        import faiss

        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._id_map = {}
        self._chunk_to_faiss = {}
        self._next_id = 0

    def _get_faiss(self):
        """Get the faiss module (for normalization functions)."""
        import faiss

        return faiss


class MockVectorStore:
    """Mock vector store for testing without FAISS."""

    def __init__(
        self,
        index_path: Path,
        embedding_dim: int = 384,
    ) -> None:
        """Initialize mock store."""
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self._vectors: dict[str, list[float]] = {}

    @property
    def size(self) -> int:
        """Get the number of vectors."""
        return len(self._vectors)

    def add_embeddings(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Add embeddings."""
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            self._vectors[chunk_id] = embedding

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search using cosine similarity."""
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
        """Remove embeddings."""
        removed = 0
        for chunk_id in chunk_ids:
            if chunk_id in self._vectors:
                del self._vectors[chunk_id]
                removed += 1
        return removed

    def save(self) -> None:
        """Save to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vectors": self._vectors,
            "embedding_dim": self.embedding_dim,
        }
        self.index_path.write_text(json.dumps(data))

    def load(self) -> None:
        """Load from disk."""
        if self.index_path.exists():
            data = json.loads(self.index_path.read_text())
            self._vectors = data["vectors"]
            self.embedding_dim = data["embedding_dim"]

    def clear(self) -> None:
        """Clear all vectors."""
        self._vectors = {}
