"""MockEmbeddingService: deterministic embeddings without ML libraries.

Hash-based vectors, for tests and for running without sentence-transformers
(``knowledge_base_use_mock``). Chunking is the real EmbeddingService's.
"""

import hashlib


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
        """Structure-aware chunking (delegates to real EmbeddingService logic)."""
        if not content or not content.strip():
            return []

        from agentic_cli.memory._core.embeddings import EmbeddingService

        blocks = EmbeddingService._split_structural_blocks(content)
        chunks = []
        for block_type, block_text in blocks:
            if block_type == "code":
                stripped = block_text.strip()
                if stripped:
                    chunks.append(stripped)
            else:
                # Create a temporary instance to access _split_sentences
                svc = EmbeddingService.__new__(EmbeddingService)
                sentences = svc._split_sentences(block_text)
                prose_chunks = EmbeddingService._merge_sentences(sentences, chunk_size, overlap)
                chunks.extend(prose_chunks)

        return [c for c in chunks if c.strip()]
