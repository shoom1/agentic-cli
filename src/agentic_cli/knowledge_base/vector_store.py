"""Vector store for semantic search.

Provides FAISS-based vector storage and similarity search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from agentic_cli.persistence._utils import atomic_write_json

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
        """Remove embeddings by chunk ID and rebuild the index.

        FAISS IndexFlat doesn't support direct removal, so this marks IDs
        as removed and then rebuilds the index to free memory.

        Args:
            chunk_ids: List of chunk IDs to remove.

        Returns:
            Number of embeddings removed.
        """
        removed = 0
        for chunk_id in chunk_ids:
            if chunk_id in self._chunk_to_faiss:
                faiss_id = self._chunk_to_faiss.pop(chunk_id)
                self._id_map.pop(faiss_id, None)
                removed += 1
        if removed > 0:
            self.rebuild()
        return removed

    def rebuild(self) -> None:
        """Rebuild the FAISS index, removing any vectors whose IDs were deleted.

        This compacts the index by re-adding only vectors that still have
        valid ID mappings. Call after remove_embeddings() to free memory.
        """
        if self._index is None or not self._id_map:
            self.clear()
            return

        import faiss

        # Collect valid FAISS indices in order
        valid_ids = sorted(self._id_map.keys())
        if not valid_ids:
            self.clear()
            return

        # Extract valid vectors from the current index
        total = self._index.ntotal
        all_vectors = np.zeros((total, self.embedding_dim), dtype=np.float32)
        for i in range(total):
            all_vectors[i] = self._index.reconstruct(i)

        valid_vectors = all_vectors[valid_ids]

        # Rebuild mappings
        new_id_map: dict[int, str] = {}
        new_chunk_to_faiss: dict[str, int] = {}
        for new_id, old_id in enumerate(valid_ids):
            chunk_id = self._id_map[old_id]
            new_id_map[new_id] = chunk_id
            new_chunk_to_faiss[chunk_id] = new_id

        # Create fresh index and add valid vectors
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._index.add(np.ascontiguousarray(valid_vectors))
        self._id_map = new_id_map
        self._chunk_to_faiss = new_chunk_to_faiss
        self._next_id = len(new_id_map)

    def save(self) -> None:
        """Persist index and mappings to disk."""
        if self._index is None:
            return

        import faiss

        # Ensure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(self.index_path))

        # Save ID mappings (atomic to prevent corruption on crash)
        mappings_path = self.index_path.with_suffix(".mappings.json")
        mappings = {
            "id_map": {str(k): v for k, v in self._id_map.items()},
            "chunk_to_faiss": self._chunk_to_faiss,
            "next_id": self._next_id,
        }
        atomic_write_json(mappings_path, mappings)

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
