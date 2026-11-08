"""A session's search vectors: one matrix, kept in step with the database."""

from __future__ import annotations

import numpy as np

from agentic_cli.memory.kb._vectors import VectorIndex, normalized, to_blob


def _rows(*pairs):
    return [(rowid, doc, to_blob(normalized([vector])[0])) for rowid, doc, vector in pairs]


def test_the_most_similar_chunk_ranks_first():
    index = VectorIndex(3)
    index.load(_rows((1, "a", [1, 0, 0]), (2, "b", [0, 1, 0]), (3, "c", [1, 1, 0])), generation=1)

    assert [rowid for rowid, _ in index.ranked(normalized([[0, 1, 0.1]])[0])] == [2, 3, 1]


def test_allowed_documents_filter_the_ranking():
    index = VectorIndex(3)
    index.load(_rows((1, "a", [1, 0, 0]), (2, "b", [0, 1, 0])), generation=1)

    assert index.ranked(normalized([[0, 1, 0]])[0], allowed={"a"}) == [(1, "a")]


def test_a_vector_of_another_size_is_left_out():
    index = VectorIndex(3)
    index.load([(1, "a", to_blob(np.ones(3))), (2, "b", b"\x00" * 8)], generation=1)

    assert len(index) == 1


def test_an_own_change_applies_only_when_nothing_else_changed_in_between():
    index = VectorIndex(3)
    index.load(_rows((1, "a", [1, 0, 0])), generation=5)

    index.apply(6, add=([2], ["b"], normalized([[0, 1, 0]])))
    assert len(index) == 2 and index.generation == 6

    index.apply(8, remove=[1])  # generation 7 was another session's
    assert len(index) == 2 and index.generation == 6  # left to be reloaded


def test_removing_rows_keeps_the_rest_in_step():
    index = VectorIndex(3)
    index.load(_rows((1, "a", [1, 0, 0]), (2, "b", [0, 1, 0]), (3, "c", [0, 0, 1])), generation=1)

    index.apply(2, remove=[2])

    assert index.ranked(normalized([[0, 0, 1]])[0])[0] == (3, "c")
    assert {doc for _, doc in index.ranked(normalized([[1, 0, 0]])[0])} == {"a", "c"}
