"""How an app's sub-agents run: its sub-agent policy, and a run's limits.

An app passes a ``SubAgentPolicy`` as ``subagents=`` (to ``BaseCLIApp``, a
workflow manager or ``create_workflow_manager_from_settings``); None there is
``SubAgentPolicy()``, the policy every manager has unless it is given
another. ``AgentLimits`` are one sub-agent run's limits: the policy holds
every run's, and an agent type's own (``AgentConfig.limits``) lie over them
field by field (``AgentLimits.over``), a field left None taking the
policy's. Both are frozen, and refuse a value that is no limit as they are
made, with a ``ValueError`` naming the field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

#: Who may start an agent type at run time: an agent's ``task`` call, a skill
#: an agent loads, the user. A run records it as ``agent:<instance>``,
#: ``skill:<name>`` or ``user``.
SPAWN_SOURCES = ("agent", "skill", "user")


def _check_count(owner: str, name: str, value: object, *, optional: bool) -> None:
    """Refuse ``value`` unless it is a positive int (or None, when ``optional``).

    True and False are ints to Python, but no count: ``max_active=True``
    is a mistake, not one.
    """
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        allowed = "a positive int or None" if optional else "a positive int"
        raise ValueError(f"{owner}.{name} must be {allowed}, not {value!r}.")


def _check_seconds(owner: str, name: str, value: object) -> None:
    """Refuse ``value`` unless it is None or a positive, finite number of seconds."""
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not _is_finite(value)
        or value <= 0
    ):
        raise ValueError(
            f"{owner}.{name} must be a positive number of seconds or None, not {value!r}."
        )


def _is_finite(value: int | float) -> bool:
    """Whether ``value`` is finite as a float: an int too large for a float is not
    (``math.isfinite`` raises ``OverflowError`` for it)."""
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class AgentLimits:
    """One sub-agent run's limits; a field left None takes the policy's (on a
    type), or means no limit (on the policy).

    Attributes:
        max_model_calls: The most model calls the run may make.
        timeout: The most seconds the run may take.
        max_tokens: The most tokens the run may use: its input plus output
            tokens, as its usage counts them.
    """

    max_model_calls: int | None = None
    timeout: float | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        _check_count("AgentLimits", "max_model_calls", self.max_model_calls, optional=True)
        _check_seconds("AgentLimits", "timeout", self.timeout)
        _check_count("AgentLimits", "max_tokens", self.max_tokens, optional=True)

    def over(self, base: AgentLimits) -> AgentLimits:
        """These limits over ``base``: each field of this one that is set,
        else ``base``'s (a type's limits over the policy's).

        A field left None keeps the limit ``base`` sets; a field set
        replaces it, higher or lower. No field removes one.
        """
        own = {f.name: getattr(self, f.name) for f in fields(self)}
        return replace(base, **{name: value for name, value in own.items() if value is not None})


#: The policy's limits unless an app passes its own: a run makes at most 100 model calls.
DEFAULT_LIMITS = AgentLimits(max_model_calls=100)


@dataclass(frozen=True)
class SubAgentPolicy:
    """How an app's sub-agents run; ``SubAgentPolicy()`` is the default every manager gets.

    Attributes:
        max_active: How many sub-agent runs may work at once in a
            conversation, at any depth; the others wait for a free
            working slot (``agents/slots.py``).
        max_runs_per_turn: How many sub-agent runs one turn may start.
        limits: Every run's limits (``AgentLimits``), which an agent type's
            own lie over. A policy's own replace ``DEFAULT_LIMITS`` whole:
            ``SubAgentPolicy(limits=AgentLimits(timeout=60))`` has no
            model-call limit, while
            ``AgentLimits(timeout=60).over(DEFAULT_LIMITS)`` keeps the 100
            calls.
        max_open: How many sub-agent instances a conversation keeps open
            (``idle`` or ``stopped``) between their runs; past it, as a run
            ends, the oldest idle one (by its last run's end) is closed.
        follow_ups: Whether every agent that can have sub-agents gets
            ``message_agent``, ``stop_agent`` and ``check_agents`` beside
            ``task``; False, ``task`` alone.
        spawn_sources: Who may start an agent type at run time: a type whose
            ``spawnable_by`` names a source this set holds may be started by
            it. Empty by default (nothing is started at run time); a set,
            list or tuple is kept as a frozenset.
        max_spawn_depth: A run at depth ``d`` (the root's is 0) may start an
            agent type at run time while ``d`` is below this; the configured
            tree keeps its depth.
        general_purpose: Whether the built-in ``general_purpose`` type is
            one of the types started at run time (``agents/spawn.py``): its
            spawner's own tools but the ones that start agents, its
            spawner's model and the framework's system prompt, startable by
            every source ``spawn_sources`` enables. False (the default): no
            such type. True needs a source.
    """

    max_active: int = 4
    max_runs_per_turn: int = 20
    limits: AgentLimits = DEFAULT_LIMITS
    max_open: int = 20
    follow_ups: bool = False
    spawn_sources: frozenset[str] = frozenset()
    max_spawn_depth: int = 1
    general_purpose: bool = False

    def __post_init__(self) -> None:
        _check_count("SubAgentPolicy", "max_active", self.max_active, optional=False)
        _check_count("SubAgentPolicy", "max_runs_per_turn", self.max_runs_per_turn, optional=False)
        if not isinstance(self.follow_ups, bool):
            raise ValueError(
                f"SubAgentPolicy.follow_ups must be True or False, not {self.follow_ups!r}."
            )
        if not isinstance(self.limits, AgentLimits):
            raise ValueError(
                f"SubAgentPolicy.limits must be an AgentLimits, not {self.limits!r}."
            )
        _check_count("SubAgentPolicy", "max_open", self.max_open, optional=False)
        object.__setattr__(self, "spawn_sources", _sources(self.spawn_sources))
        _check_count("SubAgentPolicy", "max_spawn_depth", self.max_spawn_depth, optional=False)
        if not isinstance(self.general_purpose, bool):
            raise ValueError(
                f"SubAgentPolicy.general_purpose must be True or False, not "
                f"{self.general_purpose!r}."
            )
        if self.general_purpose and not self.spawn_sources:
            raise ValueError(
                "SubAgentPolicy.general_purpose needs spawn_sources: with none, nothing could "
                "start it."
            )


def _sources(value: object) -> frozenset[str]:
    """``value`` as a frozenset of sources, or a ``ValueError`` naming it."""
    if isinstance(value, (set, frozenset, list, tuple)) and all(
        isinstance(item, str) and item in SPAWN_SOURCES for item in value
    ):
        return frozenset(value)
    allowed = ", ".join(repr(source) for source in SPAWN_SOURCES)
    raise ValueError(
        f"SubAgentPolicy.spawn_sources has {value!r}: each source must be one of {allowed}."
    )


def as_policy(subagents: object) -> SubAgentPolicy:
    """The policy a ``subagents=`` argument names: itself, or ``SubAgentPolicy()`` for None.

    Every place that takes ``subagents=`` reads it through this, first, so a
    wrong value fails where it was passed rather than when a manager is
    built from it later.

    Raises:
        TypeError: ``subagents`` is neither a ``SubAgentPolicy`` nor None.
    """
    if subagents is None:
        return SubAgentPolicy()
    if isinstance(subagents, SubAgentPolicy):
        return subagents
    raise TypeError(
        f"subagents must be a SubAgentPolicy or None, not {type(subagents).__name__}."
    )
