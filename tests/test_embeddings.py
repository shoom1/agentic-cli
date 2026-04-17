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
        """Test chunking text longer than chunk_size produces multiple chunks."""
        svc = MockEmbeddingService()
        sentences = [f"Sentence number {i} is here." for i in range(20)]
        content = " ".join(sentences)
        chunks = svc.chunk_document(content, chunk_size=60, overlap=10)

        assert len(chunks) > 1
        # All original sentences should appear somewhere in the chunks
        all_text = " ".join(chunks)
        assert "Sentence number 0" in all_text
        assert "Sentence number 19" in all_text


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

    def test_merge_sentences(self, svc):
        """Test _merge_sentences produces chunks with overlap."""
        sentences = ["Short.", "Medium length.", "A longer sentence here."]
        chunks = EmbeddingService._merge_sentences(sentences, chunk_size=30, overlap=10)
        assert len(chunks) >= 1
        # All sentences should appear somewhere across chunks
        all_text = " ".join(chunks)
        for s in sentences:
            assert s in all_text

    def test_merge_sentences_empty(self, svc):
        """Test _merge_sentences with empty list returns empty."""
        assert EmbeddingService._merge_sentences([], chunk_size=100, overlap=10) == []


class TestResolveEmbeddingDevice:
    """Tests for the device-resolution autodetect helper."""

    def test_explicit_cpu_passes_through(self):
        from agentic_cli.knowledge_base.embeddings import resolve_embedding_device
        assert resolve_embedding_device("cpu") == "cpu"

    def test_explicit_mps_passes_through(self):
        from agentic_cli.knowledge_base.embeddings import resolve_embedding_device
        assert resolve_embedding_device("mps") == "mps"

    def test_explicit_cuda_passes_through(self):
        from agentic_cli.knowledge_base.embeddings import resolve_embedding_device
        assert resolve_embedding_device("cuda") == "cuda"

    def test_auto_returns_cpu_when_torch_unavailable(self, monkeypatch):
        import builtins
        from agentic_cli.knowledge_base.embeddings import resolve_embedding_device

        real_import = builtins.__import__

        def fail_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch in this test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_torch)
        assert resolve_embedding_device("auto") == "cpu"

    def test_auto_prefers_cuda_when_available(self, monkeypatch):
        from agentic_cli.knowledge_base import embeddings as emb_mod

        class FakeBackends:
            class mps:
                @staticmethod
                def is_available():
                    return True

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

        class FakeTorch:
            cuda = FakeCuda
            backends = FakeBackends

        import sys
        monkeypatch.setitem(sys.modules, "torch", FakeTorch)
        assert emb_mod.resolve_embedding_device("auto") == "cuda"

    def test_auto_picks_mps_only_on_apple_silicon(self, monkeypatch):
        from agentic_cli.knowledge_base import embeddings as emb_mod

        class FakeBackends:
            class mps:
                @staticmethod
                def is_available():
                    return True

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeTorch:
            cuda = FakeCuda
            backends = FakeBackends

        import sys
        monkeypatch.setitem(sys.modules, "torch", FakeTorch)
        monkeypatch.setattr(emb_mod.platform, "machine", lambda: "arm64")
        assert emb_mod.resolve_embedding_device("auto") == "mps"

        # Intel Mac with the same "mps available" report should get CPU
        monkeypatch.setattr(emb_mod.platform, "machine", lambda: "x86_64")
        assert emb_mod.resolve_embedding_device("auto") == "cpu"

    def test_embedding_service_resolves_device_at_init(self, monkeypatch):
        from agentic_cli.knowledge_base import embeddings as emb_mod

        monkeypatch.setattr(emb_mod, "resolve_embedding_device", lambda p: "cpu")
        svc = EmbeddingService(device="auto")
        assert svc.device == "cpu"
