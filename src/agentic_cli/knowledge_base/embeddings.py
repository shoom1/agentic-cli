"""Embedding service for the knowledge base.

Provides text embedding generation using sentence-transformers
and document chunking for efficient retrieval.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Generates embeddings for document chunks.

    Uses sentence-transformers models for semantic embedding generation.
    Supports lazy loading to avoid loading the model until needed.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
    ) -> None:
        """Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformers model to use.
                Recommended models:
                - "all-MiniLM-L6-v2" (fast, 384 dims, good quality)
                - "all-mpnet-base-v2" (slower, 768 dims, best quality)
            batch_size: Batch size for embedding generation.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None
        self._embedding_dim: int | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Get the sentence transformer model (lazy loading)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            # Force model loading to get dimension
            _ = self.model
        return self._embedding_dim  # type: ignore[return-value]

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def chunk_document(
        self,
        content: str,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> list[str]:
        """Split document into overlapping chunks.

        Uses sentence-aware splitting to avoid breaking mid-sentence.

        Args:
            content: Document content to chunk.
            chunk_size: Target size of each chunk in characters.
            overlap: Number of characters to overlap between chunks.

        Returns:
            List of text chunks.
        """
        if not content or not content.strip():
            return []

        # Split into sentences
        sentences = self._split_sentences(content)

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk_size, finalize current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Keep some sentences for overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk, overlap)
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        # Simple sentence splitting on common terminators
        # More sophisticated splitting could use NLTK or spaCy
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_sentences(
        self,
        sentences: list[str],
        overlap_chars: int,
    ) -> list[str]:
        """Get sentences for overlap from the end of a chunk.

        Args:
            sentences: List of sentences in the chunk.
            overlap_chars: Target overlap in characters.

        Returns:
            List of sentences to include in overlap.
        """
        if not sentences:
            return []

        overlap_sentences: list[str] = []
        current_length = 0

        # Work backwards from end
        for sentence in reversed(sentences):
            if current_length >= overlap_chars:
                break
            overlap_sentences.insert(0, sentence)
            current_length += len(sentence)

        return overlap_sentences


class MockEmbeddingService:
    """Mock embedding service for testing without loading models."""

    def __init__(
        self,
        model_name: str = "mock-model",
        batch_size: int = 32,
        embedding_dim: int = 384,
    ) -> None:
        """Initialize mock service."""
        self.model_name = model_name
        self.batch_size = batch_size
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim

    def embed_text(self, text: str) -> list[float]:
        """Generate mock embedding for text."""
        import hashlib

        # Generate deterministic mock embedding based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Convert hash to floats
        embedding = []
        for i in range(0, len(text_hash), 2):
            byte_val = int(text_hash[i : i + 2], 16)
            embedding.append((byte_val / 255.0) - 0.5)

        # Pad or truncate to embedding_dim
        while len(embedding) < self._embedding_dim:
            embedding.extend(embedding[: self._embedding_dim - len(embedding)])

        return embedding[: self._embedding_dim]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for multiple texts."""
        return [self.embed_text(text) for text in texts]

    def chunk_document(
        self,
        content: str,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> list[str]:
        """Simple chunking for mock service."""
        if not content:
            return []

        chunks = []
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunks.append(content[start:end])
            start = end - overlap if end < len(content) else end

        return chunks
