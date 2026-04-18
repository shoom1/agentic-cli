"""Rule, decision, and scope types for the permission engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Effect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class RuleSource(str, Enum):
    BUILTIN = "builtin"
    USER = "user"
    PROJECT = "project"
    SESSION = "session"


class AskScope(str, Enum):
    ONCE = "once"
    SESSION = "session"
    PROJECT = "project"
    DENY = "deny"


@dataclass(frozen=True)
class Rule:
    capability: str
    target: str
    effect: Effect
    source: RuleSource


@dataclass
class CheckResult:
    allowed: bool
    reason: str
