"""KnowledgeBaseManager takes a directory and plain arguments, never settings."""

from __future__ import annotations

import pytest

from agentic_cli.memory import EmbeddingConfig
from agentic_cli.memory.kb import KnowledgeBaseManager


def test_the_knowledge_base_lives_under_the_directory_it_is_given(tmp_path):
    kb = KnowledgeBaseManager(tmp_path / "kb", use_mock=True)

    assert kb.kb_dir == tmp_path / "kb"
    for sub in ("documents", "embeddings", "files"):
        assert (tmp_path / "kb" / sub).is_dir()


def test_a_directory_is_required(tmp_path, monkeypatch):
    # The old constructor defaulted to ~/.agentic/knowledge_base: keep a RED
    # run out of the real home directory.
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(TypeError):
        KnowledgeBaseManager()


def test_settings_are_not_accepted(tmp_path):
    with pytest.raises(TypeError):
        KnowledgeBaseManager(tmp_path / "kb", settings=object())


def test_the_embedding_config_is_used(tmp_path):
    kb = KnowledgeBaseManager(
        tmp_path / "kb", use_mock=True, embedding=EmbeddingConfig(model_name="m-test", batch_size=8)
    )

    assert kb.get_stats()["embedding_model"] == "m-test"


async def test_without_a_summarizer_the_preview_is_stored(tmp_path):
    kb = KnowledgeBaseManager(tmp_path / "kb", use_mock=True)

    payload = await kb.generate_sidecar_payload("Body text.", title="T")

    assert payload == {"summary": "Body text.", "claims": [], "entities": {}}


def test_the_knowledge_base_has_no_pdf_reader():
    assert not hasattr(KnowledgeBaseManager, "extract_text_from_pdf")
