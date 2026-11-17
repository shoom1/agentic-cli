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

        A field left None never lifts a limit ``base`` sets.
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
            conversation, at any depth.
        max_runs_per_turn: How many sub-agent runs one turn may start.
        limits: Every run's limits (``AgentLimits``), which an agent type's
            own lie over.
    """

    max_active: int = 4
    max_runs_per_turn: int = 20
    limits: AgentLimits = DEFAULT_LIMITS

    def __post_init__(self) -> None:
        _check_count("SubAgentPolicy", "max_active", self.max_active, optional=False)
        _check_count("SubAgentPolicy", "max_runs_per_turn", self.max_runs_per_turn, optional=False)
        if not isinstance(self.limits, AgentLimits):
            raise ValueError(
                f"SubAgentPolicy.limits must be an AgentLimits, not {self.limits!r}."
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
