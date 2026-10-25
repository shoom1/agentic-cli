"""One process at a time in a conversation (``cli/conversation_lock.py``).

Two CLIs on one stored conversation lose a turn: Deep Agents keeps one of
two runs that start from the same checkpoint, ADK fails the second writer.
So a CLI holds a lock on the conversation it has open (an ``flock`` the OS
releases when the process ends), and another that asks for it is told who
holds it. ``flock`` locks belong to an open file, so two locks in one
process stand for two processes here.
"""

from __future__ import annotations

import os

from agentic_cli.cli.conversation_lock import ConversationLock

KEY = ("adk", "sqlite", "app", "me")


def test_a_conversation_held_elsewhere_names_who_holds_it(tmp_path):
    mine, theirs = ConversationLock(tmp_path), ConversationLock(tmp_path)

    assert mine.take((*KEY, "abc")) is None
    assert theirs.take((*KEY, "abc")) == os.getpid()


def test_taking_another_conversation_lets_the_first_go(tmp_path):
    mine, theirs = ConversationLock(tmp_path), ConversationLock(tmp_path)
    mine.take((*KEY, "abc"))

    assert mine.take((*KEY, "xyz")) is None
    assert theirs.take((*KEY, "abc")) is None
    assert mine.take((*KEY, "abc")) == os.getpid()


def test_a_refused_take_keeps_the_conversation_held_before(tmp_path):
    mine, theirs = ConversationLock(tmp_path), ConversationLock(tmp_path)
    mine.take((*KEY, "abc"))
    theirs.take((*KEY, "xyz"))

    assert mine.take((*KEY, "xyz")) == os.getpid()
    assert theirs.take((*KEY, "abc")) == os.getpid()  # still mine


def test_taking_the_one_held_again_is_nothing(tmp_path):
    mine = ConversationLock(tmp_path)
    mine.take((*KEY, "abc"))

    assert mine.take((*KEY, "abc")) is None


def test_release_lets_it_go(tmp_path):
    mine, theirs = ConversationLock(tmp_path), ConversationLock(tmp_path)
    mine.take((*KEY, "abc"))
    mine.release()

    assert theirs.take((*KEY, "abc")) is None


def test_the_same_id_in_another_backend_is_another_conversation(tmp_path):
    mine, theirs = ConversationLock(tmp_path), ConversationLock(tmp_path)
    mine.take(("adk", "sqlite", "app", "me", "abc"))

    assert theirs.take(("deepagents", "sqlite", "app", "me", "abc")) is None


def test_its_files_are_private(tmp_path):
    ConversationLock(tmp_path / "locks").take((*KEY, "abc"))

    (path,) = (tmp_path / "locks").iterdir()
    assert path.stat().st_mode & 0o777 == 0o600


def test_a_switch_holds_both_until_it_commits(tmp_path):
    mine, theirs = ConversationLock(tmp_path), ConversationLock(tmp_path)
    mine.take((*KEY, "abc"))

    taken = mine.acquire((*KEY, "xyz"))
    assert theirs.take((*KEY, "abc")) == os.getpid()  # still held
    assert theirs.take((*KEY, "xyz")) == os.getpid()

    mine.commit(taken)
    assert theirs.take((*KEY, "abc")) is None
    assert ConversationLock(tmp_path).take((*KEY, "xyz")) == os.getpid()


def test_a_switch_that_does_not_happen_lets_the_new_one_go(tmp_path):
    mine, theirs = ConversationLock(tmp_path), ConversationLock(tmp_path)
    mine.take((*KEY, "abc"))

    mine.abandon(mine.acquire((*KEY, "xyz")))

    assert theirs.take((*KEY, "xyz")) is None
    assert ConversationLock(tmp_path).take((*KEY, "abc")) == os.getpid()
