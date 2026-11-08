"""A memory file: its frontmatter, its name, its fields' rules (spec §5.2)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentic_cli.memory.store import _files as files
from agentic_cli.memory.store.models import Memory, Proposal
from tests.memory.memory_files import T0, memory_text


def _memory(**changes) -> Memory:
    values = dict(
        id="4f2a9c1e", description="Prefers pytest fixtures", content="Use fixtures.",
        type="feedback", scope="user", source="agent", conversation=None,
        created=T0, updated=T0, path=Path("x.md"),
    )
    values.update(changes)
    return Memory(**values)


class TestValues:
    @pytest.mark.parametrize("value", [
        "Prefers pytest fixtures", "2026-11-08T14:03:22Z", "4f2a9c1e", "Café au lait", "水獭 likes rivers",
    ])
    def test_a_plain_value_is_written_bare(self, value):
        assert files.write_value(value) == value

    @pytest.mark.parametrize("value", [
        "Note: the server", "a # comment", "ends with space ", " starts with space",
        "true", "No", "null", "~", "42", "3.14", "1e5", "0x1F", "", "'quoted'", '"quoted"',
        "- a dash first", "[a list]", "{a: map}", "*star",
    ])
    def test_anything_else_is_a_json_string_and_reads_back(self, value):
        written = files.write_value(value)
        assert written.startswith('"')
        assert files.read_value(written) == value

    def test_a_yaml_single_quoted_value_reads(self):
        assert files.read_value("'It''s fine'") == "It's fine"

    def test_a_bare_value_is_stripped(self):
        assert files.read_value("  spaced out  ") == "spaced out"


class TestParse:
    def test_round_trip_keeps_every_key_in_order(self):
        memory = _memory(conversation="c-1")
        text = files.render(files.memory_fields(memory, {"tags": "a, b"}), memory.content)
        fields, content = files.parse(text)
        assert list(fields) == ["id", "description", "type", "source", "conversation",
                                "created", "updated", "tags"]
        assert content == "Use fixtures."

    def test_conversation_is_left_out_when_none(self):
        text = files.render(files.memory_fields(_memory(), {}), "x")
        assert "conversation" not in text

    def test_crlf_and_a_byte_order_mark_are_read(self):
        text = "﻿" + memory_text().replace("\n", "\r\n")
        fields, content = files.parse(text)
        assert fields["id"] == "4f2a9c1e"
        assert content == "New tests use pytest fixtures."

    def test_blank_lines_and_comments_in_the_frontmatter_are_skipped(self):
        fields, _ = files.parse("---\n# mine\nid: a1\n\ndescription: d\n---\nbody\n")
        assert fields == {"id": "a1", "description": "d"}

    @pytest.mark.parametrize("text, detail", [
        ("no frontmatter\n", "the first line is not ---"),
        ("---\nid: a1\n", "no closing ---"),
        ("---\njust words\n---\n", "line 2 is not 'key: value'"),
        ("---\n  id: a1\n---\n", "line 2 is not 'key: value'"),
        ("---\nid: a1\nid: b2\n---\n", "repeats the key 'id'"),
        ('---\nid: "unclosed\n---\n', "line 2"),
    ])
    def test_a_malformed_file_says_where(self, text, detail):
        with pytest.raises(files.FormatError, match=detail):
            files.parse(text)


class TestFields:
    def test_a_file_a_person_wrote_without_dates_or_source(self):
        fields, content = files.parse("---\nid: a1\ndescription: Mine\ntype: user\n---\nText\n")
        mtime = datetime(2026, 1, 2, tzinfo=timezone.utc)
        memory, extra = files.memory_from(fields, content, scope="project", path=Path("m.md"), mtime=mtime)
        assert (memory.source, memory.created, memory.updated) == ("user", mtime, mtime)
        assert extra == {}

    @pytest.mark.parametrize("value, expected", [
        ("2026-11-08T14:03:22Z", T0),
        ("2026-11-08T16:03:22+02:00", T0),
        ("2026-11-08T14:03:22", T0),
        ("2026-11-08", datetime(2026, 11, 8, tzinfo=timezone.utc)),
    ])
    def test_dates_in_any_iso_form(self, value, expected):
        assert files.parse_time(value) == expected

    @pytest.mark.parametrize("change, field", [
        ({"id": "../etc"}, "id"),
        ({"id": "x" * 65}, "id"),
        ({"description": ""}, "description"),
        ({"description": "x" * 151}, "description"),
        ({"description": "two\nlines"}, "description"),
        ({"type": "preference"}, "type"),
        ({"source": "robot"}, "source"),
        ({"created": "yesterday"}, "created"),
    ])
    def test_a_field_that_fails_its_rule_is_named(self, change, field):
        fields = {"id": "a1", "description": "d", "type": "user", "source": "agent",
                  "created": "2026-11-08", "updated": "2026-11-08", **change}
        with pytest.raises(files.FieldError, match=f"^{field}:"):
            files.memory_from(fields, "", scope="user", path=Path("m.md"), mtime=T0)

    def test_a_proposal_round_trips(self):
        proposal = Proposal(
            id="b2c3d4e5", action="update", target="4f2a9c1e", description="Prefers tabs",
            content="Tabs now.", type="feedback", scope="project", source="extractor",
            conversation="c-9", outside_content="web_fetch", created=T0, path=Path("p.md"),
        )
        text = files.render(files.proposal_fields(proposal), proposal.content,
                            order=files.MEMORY_KEYS + files.PROPOSAL_KEYS)
        fields, content = files.parse(text)
        assert files.proposal_from(fields, content, scope="project", path=Path("p.md")) == proposal

    def test_a_new_proposal_with_a_target_is_refused(self):
        fields = {"id": "b2", "description": "d", "type": "user", "source": "extractor",
                  "created": "2026-11-08", "action": "new", "target": "a1"}
        with pytest.raises(files.FieldError, match="^target:"):
            files.proposal_from(fields, "c", scope="user", path=Path("p.md"))


class TestNames:
    @pytest.mark.parametrize("description, expected", [
        ("Prefers pytest fixtures over unittest classes for new tests",
         "prefers-pytest-fixtures-over-unittest"),
        ("水獭", "memory"),
        ("C++ & Rust!", "c-rust"),
        ("x" * 50, "x" * 40),
    ])
    def test_slug(self, description, expected):
        assert files.slug(description) == expected
        assert len(files.slug(description)) <= 40

    def test_file_name(self):
        assert files.file_name("Prefers tabs", "4f2a9c1e") == "prefers-tabs-4f2a9c1e.md"

    def test_new_ids_avoid_taken_ones(self, monkeypatch):
        draws = iter(["aaaaaaaa", "bbbbbbbb"])
        monkeypatch.setattr(files.secrets, "token_hex", lambda n: next(draws))
        assert files.new_id(lambda candidate: candidate == "aaaaaaaa") == "bbbbbbbb"

    @pytest.mark.parametrize("value, ok", [("4f2a9c1e", True), ("0b6f-uuid-like", True),
                                           ("../x", False), ("", False), ("a b", False)])
    def test_valid_id(self, value, ok):
        assert files.valid_id(value) is ok

    def test_normalized(self):
        assert files.normalized("  Prefers   PYTEST\tfixtures ") == "prefers pytest fixtures"
