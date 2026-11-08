"""Proposals: the filters, the inbox, accept and discard; the extraction prompt (spec §6.8, §8)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from agentic_cli.memory.store import _inbox
from agentic_cli.memory.store.extract import extraction_prompt, parse_candidates
from agentic_cli.memory.store.models import Candidate, DuplicateMemory, InvalidMemory
from agentic_cli.memory.store.store import MemoryStore
from tests.memory.kb_storage import LogRecorder
from tests.memory.memory_files import T0, Clock


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path, clock) -> MemoryStore:
    return MemoryStore(tmp_path / "user", tmp_path / "proj" / "memory", now=clock)


@pytest.fixture
def log(monkeypatch) -> LogRecorder:
    from agentic_cli.memory.store import store as store_module

    return LogRecorder().install(monkeypatch, store_module)


def new(description="Prefers tabs", content="Tabs, width 4.", type="feedback", scope="user") -> Candidate:
    return Candidate("new", None, description, content, type, scope)


class TestPropose:
    def test_a_proposal_is_an_inbox_file_the_model_does_not_see(self, store, tmp_path):
        proposal = store.propose(new(), conversation="c-1", outside_content="web_fetch")
        assert proposal.path == tmp_path / "user" / ".inbox" / f"{proposal.id}.md"
        assert (proposal.source, proposal.conversation, proposal.outside_content) == ("extractor", "c-1", "web_fetch")
        assert store.memories() == [] and store.search("tabs") == [] and store.index() == ""
        assert store.proposals() == [proposal]

    @pytest.mark.parametrize("candidate, reason", [
        (new(description=""), "description: empty"),
        (new(content="x" * 4001), "content: longer than 4000"),
        (new(type="preference"), "type"),
        (new(content="key AKIAIOSFODNN7EXAMPLE"), "looks like a secret"),
        (Candidate("update", "deadbeef", "d", "c", "user", "user"), "its memory does not exist"),
    ])
    def test_a_candidate_that_fails_a_rule_is_dropped_and_logged(self, store, log, candidate, reason):
        assert store.propose(candidate) is None
        [fields] = log.infos("memory_proposal_dropped")
        assert reason in fields["reason"]
        assert store.proposals() == []

    def test_no_project_scope(self, tmp_path, log):
        bare = MemoryStore(tmp_path / "user")
        assert bare.propose(new(scope="project")) is None
        assert "no project scope" in log.infos("memory_proposal_dropped")[0]["reason"]

    @pytest.mark.parametrize("candidate", [
        new(description="  PREFERS tabs  ", content="something else"),
        new(description="Something else", content="tabs,   WIDTH 4."),
    ])
    def test_a_new_candidate_that_repeats_a_memory(self, store, candidate):
        store.save("Prefers tabs", "Tabs, width 4.", "feedback", scope="user")
        assert store.propose(candidate) is None

    def test_a_new_candidate_that_repeats_a_waiting_proposal_or_a_discard(self, store):
        assert store.propose(new()) is not None
        assert store.propose(new()) is None
        store.discard(store.proposals()[0].id)
        assert store.propose(new(content="Different text")) is None

    def test_an_update_of_the_memory_itself_is_kept(self, store):
        memory = store.save("Prefers spaces", "Spaces.", "feedback", scope="user")
        proposal = store.propose(Candidate("update", memory.id, "Prefers spaces", "Tabs now.", "feedback", "project"))
        assert proposal is not None
        assert (proposal.action, proposal.target, proposal.scope) == ("update", memory.id, "user")

    def test_an_update_that_changes_nothing_or_takes_another_s_description(self, store):
        one = store.save("one", "1", "user", scope="user")
        store.save("two", "2", "user", scope="user")
        assert store.propose(Candidate("update", one.id, "ONE", "1", "user", "user")) is None
        assert store.propose(Candidate("update", one.id, "two", "1b", "user", "user")) is None

    def test_a_full_inbox(self, store, monkeypatch):
        monkeypatch.setattr(_inbox, "INBOX_MAX", 1)
        assert store.propose(new()) is not None
        assert store.propose(new(description="Other", content="Other.")) is None


class TestReview:
    def test_accepting_a_new_proposal_saves_it_with_its_provenance(self, store):
        proposal = store.propose(new(), conversation="c-1")
        memory = store.accept(proposal.id)
        assert (memory.source, memory.conversation, memory.scope) == ("extractor", "c-1", "user")
        assert store.proposals() == [] and not proposal.path.exists()

    def test_accepting_an_update_keeps_the_old_version(self, store):
        memory = store.save("Prefers spaces", "Spaces.", "feedback", scope="user")
        proposal = store.propose(Candidate("update", memory.id, "Prefers tabs", "Tabs.", "feedback", "user"))
        changed = store.accept(proposal.id)
        assert (changed.id, changed.content) == (memory.id, "Tabs.")
        assert store.history(memory.id)[0].memory.content == "Spaces."

    def test_accepting_an_update_whose_memory_was_deleted(self, store):
        memory = store.save("Prefers spaces", "Spaces.", "feedback", scope="user")
        proposal = store.propose(Candidate("update", memory.id, "Prefers tabs", "Tabs.", "feedback", "user"))
        store.delete(memory.id)
        with pytest.raises(InvalidMemory, match="its memory was deleted; discard it"):
            store.accept(proposal.id)
        assert store.proposals() == [proposal]

    def test_accepting_a_proposal_that_is_now_a_duplicate(self, store):
        proposal = store.propose(new())
        store.save("Prefers tabs", "Saved by hand.", "feedback")
        with pytest.raises(DuplicateMemory):
            store.accept(proposal.id)
        assert store.proposals() == [proposal]

    def test_discard_remembers_the_description(self, store, tmp_path):
        proposal = store.propose(new(description="Prefers TABS"))
        assert store.discard(proposal.id) is True
        assert store.discarded() == ["prefers tabs"]
        assert store.discard(proposal.id) is False
        assert (tmp_path / "user" / ".inbox" / "discarded.jsonl").is_file()

    def test_the_discarded_list_keeps_its_newest(self, store, monkeypatch):
        monkeypatch.setattr(_inbox, "DISCARDED_KEEP", 2)
        for n in range(3):
            store.discard(store.propose(new(description=f"d{n}", content=f"c{n}")).id)
        assert store.discarded() == ["d1", "d2"]

    def test_proposals_older_than_30_days_go_at_the_next_open(self, store, clock, tmp_path):
        store.propose(new())
        clock.now = T0 + timedelta(days=31)
        reopened = MemoryStore(tmp_path / "user", tmp_path / "proj" / "memory", now=clock)
        assert reopened.proposals() == []
        assert reopened.discarded() == []

    def test_proposals_list_oldest_first(self, store, clock):
        first = store.propose(new(description="a", content="1"))
        clock.now = T0 + timedelta(minutes=1)
        second = store.propose(new(description="b", content="2", scope="project"))
        assert store.proposals() == [first, second]
        assert store.proposals("project") == [second]

    def test_accept_of_an_unknown_proposal(self, store):
        with pytest.raises(InvalidMemory, match="no proposal deadbeef"):
            store.accept("deadbeef")


class TestExtraction:
    def test_the_prompt_holds_the_index_the_discards_and_the_conversation(self):
        prompt = extraction_prompt(
            [{"role": "user", "content": "Use tabs from now on."}, {"role": "assistant", "content": "OK."}],
            "<memory-index>\n- [a1] 2026-11-08 feedback: Prefers spaces\n</memory-index>",
            ["prefers emacs"],
        )
        assert "at most 5" in prompt
        assert "- [a1] 2026-11-08 feedback: Prefers spaces" in prompt
        assert "- prefers emacs" in prompt
        assert prompt.endswith("user: Use tabs from now on.\nassistant: OK.")

    def test_an_empty_index_says_so(self):
        assert "(no memories yet)" in extraction_prompt([], "", [])

    def test_parse_a_bare_array_and_a_fenced_one(self):
        text = '[{"action": "new", "description": "d", "content": "c", "type": "user", "scope": "user"}]'
        expected = [Candidate("new", None, "d", "c", "user", "user")]
        assert parse_candidates(text) == expected
        assert parse_candidates(f"Here you go:\n```json\n{text}\n```\n") == expected

    def test_parse_an_update(self):
        text = '[{"action": "update", "id": "a1", "description": "d", "content": "c", "type": "feedback", "scope": "project"}]'
        assert parse_candidates(text) == [Candidate("update", "a1", "d", "c", "feedback", "project")]

    @pytest.mark.parametrize("item", [
        {"action": "new", "description": "d", "content": "c", "type": "user"},
        {"action": "new", "description": "d", "content": "c", "type": "pref", "scope": "user"},
        {"action": "update", "description": "d", "content": "c", "type": "user", "scope": "user"},
        {"action": "new", "id": "a1", "description": "d", "content": "c", "type": "user", "scope": "user"},
        {"action": "delete", "id": "a1", "description": "d", "content": "c", "type": "user", "scope": "user"},
        {"action": "new", "description": 5, "content": "c", "type": "user", "scope": "user"},
        "not an object",
    ])
    def test_a_malformed_item_is_dropped(self, item):
        import json

        assert parse_candidates(json.dumps([item])) == []

    def test_items_past_five_are_ignored(self):
        import json

        items = [{"action": "new", "description": f"d{n}", "content": "c", "type": "user", "scope": "user"}
                 for n in range(7)]
        assert [c.description for c in parse_candidates(json.dumps(items))] == [f"d{n}" for n in range(5)]

    @pytest.mark.parametrize("text", ["", "none", '{"action": "new"}', "[1, 2", "```\nnot json\n```"])
    def test_text_that_is_not_an_array_gives_nothing(self, text):
        assert parse_candidates(text) == []
