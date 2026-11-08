"""Chinese, Japanese and Korean words are found by keyword (review K6, memory-kb F7).

The tokenizer kept a run of word characters as one term, and Chinese and
Japanese put no spaces between words, so 水獭住在河边的洞穴里 was one term and
水獭 (otter) matched nothing. A run of these scripts is now its overlapping
two-character terms, in the indexed text and in the query alike.
"""

from __future__ import annotations

import pytest

from agentic_cli.memory import EmbeddingService
from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType
from agentic_cli.memory.kb._tokenize import TOKENIZER, tokenize


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(EmbeddingService, "is_available", staticmethod(lambda: False))


def test_a_chinese_run_becomes_two_character_terms():
    assert tokenize("水獭住在河边") == ["水獭", "獭住", "住在", "在河", "河边"]


def test_japanese_and_korean_runs_too():
    assert tokenize("カワウソ") == ["カワ", "ワウ", "ウソ"]
    assert tokenize("수달이") == ["수달", "달이"]


def test_a_single_character_stays_one_term():
    assert tokenize("水") == ["水"]


def test_a_word_mixing_scripts_keeps_its_latin_part_whole():
    assert tokenize("Python3では") == ["python3", "では"]


def test_other_scripts_are_split_as_before():
    assert tokenize("Выдры, живут; résumé") == ["выдры", "живут", "résumé"]


def test_the_tokenizer_has_a_new_version():
    assert TOKENIZER == "words-v3"


def test_a_chinese_word_inside_a_sentence_is_found(tmp_path):
    kb = KnowledgeBaseManager(tmp_path, use_mock=True)
    kb.ingest_document(content="水獭住在河边的洞穴里。", title="otters", source_type=SourceType.USER)
    kb.ingest_document(content="苍鹭在黎明涉水。", title="herons", source_type=SourceType.USER)

    hits = kb.search("水獭")["results"]

    assert [hit["document_title"] for hit in hits] == ["otters"]
