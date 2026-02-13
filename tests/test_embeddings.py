"""Tests for EmbeddingService and MockEmbeddingService."""

import pytest

from agentic_cli.knowledge_base.embeddings import EmbeddingService
from agentic_cli.knowledge_base._mocks import MockEmbeddingService


class TestMockEmbeddingService:
    """Tests for MockEmbeddingService."""

    def test_init_defaults(self):
        """Test default initialization."""
        svc = MockEmbeddingService()
        assert svc.model_name == "mock-model"
        assert svc.batch_size == 32

    def test_embedding_dim(self):
        """Test embedding dimension property."""
        svc = MockEmbeddingService(embedding_dim=128)
        assert svc.embedding_dim == 128

    def test_embed_text_returns_correct_dim(self):
        """Test that embed_text returns correct dimension."""
        svc = MockEmbeddingService(embedding_dim=64)
        embedding = svc.embed_text("hello world")
        assert len(embedding) == 64
        assert all(isinstance(v, float) for v in embedding)

    def test_embed_text_deterministic(self):
        """Test that same text produces same embedding."""
        svc = MockEmbeddingService()
        e1 = svc.embed_text("hello world")
        e2 = svc.embed_text("hello world")
        assert e1 == e2

    def test_embed_text_different_texts(self):
        """Test that different texts produce different embeddings."""
        svc = MockEmbeddingService()
        e1 = svc.embed_text("hello world")
        e2 = svc.embed_text("goodbye world")
        assert e1 != e2

    def test_embed_batch_empty(self):
        """Test embed_batch with empty list."""
        svc = MockEmbeddingService()
        result = svc.embed_batch([])
        assert result == []

    def test_embed_batch_multiple(self):
        """Test embed_batch with multiple texts."""
        svc = MockEmbeddingService(embedding_dim=64)
        texts = ["alpha", "beta", "gamma"]
        embeddings = svc.embed_batch(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 64

    def test_chunk_document_empty(self):
        """Test chunking empty content."""
        svc = MockEmbeddingService()
        assert svc.chunk_document("") == []

    def test_chunk_document_short_text(self):
        """Test chunking text shorter than chunk_size."""
        svc = MockEmbeddingService()
        chunks = svc.chunk_document("Short text.", chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_chunk_document_long_text(self):
        """Test chunking text longer than chunk_size."""
        svc = MockEmbeddingService()
        content = "A" * 100
        chunks = svc.chunk_document(content, chunk_size=30, overlap=10)

        assert len(chunks) > 1
        # All content should be covered
        reconstructed = chunks[0]
        for chunk in chunks[1:]:
            reconstructed += chunk[10:]  # skip overlap
        assert len(reconstructed) >= len(content)


class TestEmbeddingServiceChunking:
    """Tests for EmbeddingService.chunk_document() (no model needed)."""

    @pytest.fixture
    def svc(self):
        """Create EmbeddingService without loading model."""
        s = EmbeddingService.__new__(EmbeddingService)
        s.model_name = "test"
        s.batch_size = 32
        s._model = None
        s._embedding_dim = None
        return s

    def test_chunk_empty_content(self, svc):
        """Test chunking empty content."""
        assert svc.chunk_document("") == []

    def test_chunk_whitespace_only(self, svc):
        """Test chunking whitespace-only content."""
        assert svc.chunk_document("   \n\t  ") == []

    def test_chunk_short_text_single_chunk(self, svc):
        """Test short text produces single chunk."""
        chunks = svc.chunk_document("Hello world.", chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_chunk_long_text_multiple_chunks(self, svc):
        """Test long text produces multiple chunks."""
        # Create content with clear sentences
        sentences = [f"Sentence number {i} is here." for i in range(20)]
        content = " ".join(sentences)

        chunks = svc.chunk_document(content, chunk_size=100, overlap=30)

        assert len(chunks) > 1
        # First chunk should contain some sentences
        assert "Sentence number 0" in chunks[0]

    def test_chunk_overlap(self, svc):
        """Test that chunks have overlapping content."""
        sentences = [f"Sentence {i} here." for i in range(10)]
        content = " ".join(sentences)

        chunks = svc.chunk_document(content, chunk_size=60, overlap=20)

        if len(chunks) >= 2:
            # Last sentence(s) of chunk 0 should appear in chunk 1
            # (overlap ensures continuity)
            words_0 = set(chunks[0].split())
            words_1 = set(chunks[1].split())
            assert words_0 & words_1, "Chunks should have overlapping words"

    def test_split_sentences(self, svc):
        """Test sentence splitting regex."""
        text = "First sentence. Second sentence! Third one? Final."
        sentences = svc._split_sentences(text)
        assert len(sentences) >= 2

    def test_get_overlap_sentences(self, svc):
        """Test overlap sentence selection."""
        sentences = ["Short.", "Medium length.", "A longer sentence here."]
        overlap = svc._get_overlap_sentences(sentences, overlap_chars=20)
        assert len(overlap) >= 1
        # Should include at least the last sentence
        assert sentences[-1] in overlap

    def test_get_overlap_sentences_empty(self, svc):
        """Test overlap with empty list."""
        assert svc._get_overlap_sentences([], 10) == []
