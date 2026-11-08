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


def test_the_iteration_mark_is_part_of_its_run():
    # 々 repeats the character before it (時々, 人々, 様々); it broke a run.
    assert tokenize("時々") == ["時々"]
    assert tokenize("彼は時々来る") == ["彼は", "は時", "時々", "々来", "来る"]


@pytest.mark.parametrize(
    "char",
    [
        "\u3006",  # 〆 ideographic closing mark
        "\u3007",  # 〇 ideographic number zero
        "\u3021",  # 〡 Hangzhou numeral one
        "\u3029",  # 〩 Hangzhou numeral nine
        "\u3038",  # 〸 Hangzhou numeral ten
        "\u303b",  # 〻 vertical ideographic iteration mark
        "\u31f0",  # ㇰ small katakana ku
        "\u31ff",  # ㇿ small katakana ro
        "\U00031350",  # first of Han extension H
        "\U000323af",  # last of Han extension H
    ],
)
def test_other_han_characters_stay_in_a_run(char):
    assert tokenize("水" + char) == ["水" + char]


@pytest.mark.parametrize("char", ["\u3005", "\u3006", "\u3007"])
def test_a_run_may_start_with_the_marks(char):
    assert tokenize(char + "水") == [char + "水"]


def test_the_tokenizer_has_a_new_version():
    assert TOKENIZER == "words-v3"


def test_a_chinese_word_inside_a_sentence_is_found(tmp_path):
    kb = KnowledgeBaseManager(tmp_path, use_mock=True)
    kb.ingest_document(content="水獭住在河边的洞穴里。", title="otters", source_type=SourceType.USER)
    kb.ingest_document(content="苍鹭在黎明涉水。", title="herons", source_type=SourceType.USER)

    hits = kb.search("水獭")["results"]

    assert [hit["document_title"] for hit in hits] == ["otters"]


def test_a_word_with_the_iteration_mark_is_found_and_another_with_it_is_not(tmp_path):
    kb = KnowledgeBaseManager(tmp_path, use_mock=True)
    kb.ingest_document(content="彼は時々来る。", title="sometimes", source_type=SourceType.USER)
    kb.ingest_document(content="人々が集まる。", title="people", source_type=SourceType.USER)

    hits = kb.search("時々")["results"]

    assert [hit["document_title"] for hit in hits] == ["sometimes"]
