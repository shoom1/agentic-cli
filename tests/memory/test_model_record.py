"""The record of which embedding model built an index.

A project can ship its knowledge base, so ``embeddings/model.json`` is
untrusted: anything that is not a small regular file holding a model name
and a positive dimension reads as no record, and reading never blocks.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from agentic_cli.memory import MockEmbeddingService
from agentic_cli.memory._core.model_record import (
    ModelRecord,
    read_model_record,
    record_for,
    write_model_record,
)


def test_a_record_reads_back(tmp_path):
    path = tmp_path / "model.json"

    write_model_record(path, ModelRecord("all-MiniLM-L6-v2", 384))

    assert json.loads(path.read_text()) == {"model_name": "all-MiniLM-L6-v2", "embedding_dim": 384}
    assert read_model_record(path) == ModelRecord("all-MiniLM-L6-v2", 384)


def test_a_missing_record_is_none(tmp_path):
    assert read_model_record(tmp_path / "model.json") is None


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        '{"model_name": "m"}',
        '{"model_name": 7, "embedding_dim": 384}',
        '{"model_name": "", "embedding_dim": 384}',
        '{"model_name": "m", "embedding_dim": "384"}',
        '{"model_name": "m", "embedding_dim": true}',
        '{"model_name": "m", "embedding_dim": -384}',
        '{"model_name": "m", "embedding_dim": 0}',
        '{"model_name": "' + "m" * 5000 + '", "embedding_dim": 384}',
    ],
)
def test_a_malformed_record_is_none(tmp_path, content):
    path = tmp_path / "model.json"
    path.write_text(content)

    assert read_model_record(path) is None


def test_a_directory_is_not_a_record(tmp_path):
    (tmp_path / "model.json").mkdir()

    assert read_model_record(tmp_path / "model.json") is None


def test_a_symlink_is_not_followed(tmp_path):
    target = tmp_path / "elsewhere.json"
    write_model_record(target, ModelRecord("m", 8))
    (tmp_path / "model.json").symlink_to(target)

    assert read_model_record(tmp_path / "model.json") is None


@pytest.mark.skipif(sys.platform == "win32", reason="FIFOs are POSIX")
def test_a_fifo_is_not_read(tmp_path):
    """Opening a FIFO for reading blocks until a writer appears."""
    os.mkfifo(tmp_path / "model.json")

    assert read_model_record(tmp_path / "model.json") is None


def test_the_record_for_a_service():
    assert record_for(MockEmbeddingService(model_name="m", embedding_dim=16)) == ModelRecord("m", 16)


def test_a_service_that_does_not_name_its_model_has_no_record():
    class Bare:
        def embed_text(self, text):
            return [0.0]

        def embed_batch(self, texts):
            return [[0.0] for _ in texts]

    assert record_for(Bare()) is None
