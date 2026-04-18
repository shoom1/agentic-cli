"""Tests for rule/decision data types."""

from agentic_cli.workflow.permissions.rules import (
    AskScope,
    CheckResult,
    Effect,
    Rule,
    RuleSource,
)


class TestEnums:
    def test_effect_values(self):
        assert Effect.ALLOW.value == "allow"
        assert Effect.DENY.value == "deny"

    def test_rule_source_values(self):
        assert {s.value for s in RuleSource} == {"builtin", "user", "project", "session"}

    def test_ask_scope_values(self):
        assert {s.value for s in AskScope} == {"once", "session", "project", "deny"}


class TestRule:
    def test_fields(self):
        r = Rule("filesystem.read", "/abs/**", Effect.ALLOW, RuleSource.BUILTIN)
        assert r.capability == "filesystem.read"
        assert r.target == "/abs/**"
        assert r.effect is Effect.ALLOW
        assert r.source is RuleSource.BUILTIN

    def test_rule_is_hashable(self):
        a = Rule("filesystem.read", "/x", Effect.ALLOW, RuleSource.BUILTIN)
        b = Rule("filesystem.read", "/x", Effect.ALLOW, RuleSource.BUILTIN)
        assert a == b
        assert hash(a) == hash(b)


class TestCheckResult:
    def test_allowed(self):
        res = CheckResult(True, "rule: builtin/allow filesystem.read")
        assert res.allowed is True
        assert "allow" in res.reason

    def test_denied(self):
        res = CheckResult(False, "user denied")
        assert res.allowed is False
