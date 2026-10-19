"""Resolution: one agent's model, thinking, max_tokens and sampling, decided once.

``resolve_model`` combines the agent's record (from the catalog), its
generation settings and the global ``thinking_effort`` into a neutral
``ResolvedModel``. Both backend adapters read only that, so the thinking
mapping, the ``max_tokens`` rules and the sampling rule live here and nowhere
else.

The thinking vocabulary: ``none``, ``low``, ``medium``, ``high`` everywhere;
per agent also ``budget`` (with ``budget_tokens``) or a native level of the
model (``minimal``, ``xhigh``, ``max``, …), checked against its ladder.
``high`` is the model's own ``high``, never its top level.

- Gemini 2.5 (budget): ``none`` is budget 0, or the model's minimum when it
  cannot turn thinking off; low / medium / high are 4096 / 12288 / 24576;
  ``budget`` N is clamped to the model's range.
- Gemini 3+ (level): ``none`` is the lowest level; low / medium / high are
  the levels (Gemini 3 Pro: medium becomes high); ``budget`` is an error.
- Claude, adaptive: ``none`` is disabled, or effort low when the model cannot
  disable thinking (Opus 5.5, Sonnet 5.5, Fable); low / medium / high are
  adaptive thinking with that effort; ``budget`` is an error.
- Claude, budget (up to 4.5): ``none`` is disabled; low / medium / high are
  budgets 4096 / 10000 / 32000; ``budget`` N is at least 1024.
- Effort (OpenAI and OpenRouter reasoning models, endpoints declaring
  effort): ``none`` is the lowest effort on the ladder; low / medium / high
  are those efforts; ``budget`` is an error.
- Chat template (endpoints): ``none`` is ``enable_thinking: false``,
  low / medium / high ``enable_thinking: true``; ``budget`` is an error.

A model with no thinking knob ignores the setting (logged at debug).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from agentic_cli.models.endpoints import PROVIDER_KEY_ENV, Endpoint, ProviderKind
from agentic_cli.models.records import ModelRecord, ThinkingMechanism, ThinkingRule
from agentic_cli.models.refs import ModelRef, ModelSpecError, RoleRef, parse_spec

if TYPE_CHECKING:
    from agentic_cli.models.roles import RoleMap

logger = logging.getLogger(__name__)

_GENERIC_LEVELS = ("low", "medium", "high")
# Where a level sits when the allowed values are listed in a message.
_LEVEL_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
# ``budget`` with no ``budget_tokens``: the budget 0.6.x used on both providers.
_DEFAULT_BUDGET = 12288
# Anthropic: max_tokens when none is set, and the output room kept above a
# thinking budget (thinking tokens count toward max_tokens, which must exceed
# the budget).
_ANTHROPIC_DEFAULT_MAX_TOKENS = 8192
_ANTHROPIC_OUTPUT_HEADROOM = 8192
_SAMPLING_FIELDS = ("temperature", "top_p", "top_k")

_DESCRIPTIONS = {
    ThinkingMechanism.GEMINI_BUDGET: "thinking budget",
    ThinkingMechanism.GEMINI_LEVEL: "thinking levels",
    ThinkingMechanism.ANTHROPIC_ADAPTIVE: "adaptive thinking",
    ThinkingMechanism.ANTHROPIC_BUDGET: "budget thinking",
    ThinkingMechanism.EFFORT: "reasoning effort",
    ThinkingMechanism.CHAT_TEMPLATE: "chat-template thinking",
}
_BUDGET_MECHANISMS = (ThinkingMechanism.GEMINI_BUDGET, ThinkingMechanism.ANTHROPIC_BUDGET)


class ModelResolutionError(ValueError):
    """An agent's model cannot be resolved; the message names the agent."""


class ThinkingIntent(Protocol):
    """What an agent asks for: ``workflow.model_settings.ThinkingSettings``."""

    mode: str
    budget_tokens: int | None


class GenerationSettings(Protocol):
    """An agent's generation settings: ``workflow.model_settings.ModelSettings``."""

    temperature: float | None
    top_p: float | None
    top_k: int | None
    max_tokens: int | None
    stop_sequences: list[str] | None
    thinking: ThinkingIntent | None
    extra: Mapping[str, Any]


class CatalogView(Protocol):
    """The catalog as resolution sees it."""

    def record(self, ref: ModelRef) -> ModelRecord:
        """The model's record. Raises ``ModelResolutionError`` when the
        provider's listing is authoritative and does not have the model."""
        ...

    def records(self, provider: str) -> Sequence[ModelRecord]:
        """Every record the catalog holds for one provider."""
        ...


@dataclass(frozen=True)
class NativeThinking:
    """The thinking value an adapter sends, in the mechanism's own terms.

    Attributes:
        mechanism: The record's thinking mechanism.
        enabled: Whether the model will think.
        budget: Tokens, for the budget mechanisms (0 when off).
        level: The Gemini thinking level, or the effort name (adaptive Claude,
            effort mechanisms; an effort model always gets one, ``none``
            included, so the provider's own default never applies).
        label: How the UI describes it: ``adaptive, effort high``,
            ``budget 24576``, ``off``.
    """

    mechanism: ThinkingMechanism
    enabled: bool
    budget: int | None = None
    level: str | None = None
    label: str = ""


@dataclass(frozen=True)
class ResolvedModel:
    """One agent's model, decided: everything an adapter needs."""

    agent: str | None
    record: ModelRecord
    endpoint: Endpoint
    #: None: the model has no thinking knob.
    thinking: NativeThinking | None
    max_tokens: int | None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: tuple[str, ...] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    #: The sampling settings the record's rule dropped.
    dropped: tuple[str, ...] = ()

    @property
    def ref(self) -> ModelRef:
        return self.record.ref


@dataclass(frozen=True)
class _Intent:
    mode: str
    budget_tokens: int | None = None


def resolve_model(
    *,
    agent: str | None,
    spec: str | None,
    generation: GenerationSettings | None,
    thinking_effort: str,
    roles: "RoleMap",
    catalog: CatalogView,
    endpoints: Mapping[str, Endpoint],
) -> ResolvedModel:
    """Resolve one agent's model.

    Args:
        agent: The agent's name, for messages; None for the manager's own
            default model.
        spec: The agent's model: a reference, a bare id or a role. None means
            the ``default`` role.
        generation: The agent's generation settings, if any.
        thinking_effort: The global effort, used when the agent sets no
            thinking of its own.
        roles: Maps a role to a reference.
        catalog: Gives the model's record.
        endpoints: Every provider and endpoint, by name.

    Raises:
        ModelResolutionError: For a malformed spec, an unmapped role nothing
            can fill, an undefined endpoint, a provider without a key, a model
            an authoritative listing lacks, or a thinking mode the model does
            not have. The message names the agent (or the default model), the
            model and the allowed values.
    """
    who = f"Agent {agent!r}" if agent else "Default model"
    try:
        parsed = parse_spec(spec) if spec is not None else RoleRef("default")
        ref = (
            roles.resolve(parsed.role, catalog=catalog, endpoints=endpoints)
            if isinstance(parsed, RoleRef)
            else parsed
        )
    except (ModelSpecError, ModelResolutionError) as exc:
        raise ModelResolutionError(f"{who}: {exc}") from None

    endpoint = endpoints.get(ref.provider)
    if endpoint is None:
        defined = ", ".join(
            sorted(n for n, e in endpoints.items() if e.kind is ProviderKind.OPENAI_COMPATIBLE)
        )
        raise ModelResolutionError(
            f"{who}: model {str(ref)!r} names endpoint {ref.provider!r}, which is not defined in "
            f"model_endpoints (user settings); defined endpoints: {defined or 'none'}."
        )
    if not endpoint.has_credentials:
        variable = PROVIDER_KEY_ENV.get(ref.provider, "the provider's API key")
        raise ModelResolutionError(
            f"{who}: model {str(ref)!r} needs the {ref.provider} API key; set {variable}."
        )

    try:
        record = catalog.record(ref)
    except ModelResolutionError as exc:
        raise ModelResolutionError(f"{who}: {exc}") from None

    intent = _intent(generation, thinking_effort)
    thinking = _native_thinking(record, intent, who)
    max_tokens = _max_tokens(record, generation, thinking)
    sampling, dropped = _sampling(record, generation, thinking)
    stop = generation.stop_sequences if generation is not None else None
    extra = generation.extra if generation is not None else {}

    return ResolvedModel(
        agent=agent,
        record=record,
        endpoint=endpoint,
        thinking=thinking,
        max_tokens=max_tokens,
        stop_sequences=tuple(stop) if stop is not None else None,
        extra=MappingProxyType(dict(extra or {})),
        dropped=dropped,
        **sampling,
    )


def _intent(generation: GenerationSettings | None, thinking_effort: str) -> ThinkingIntent:
    if generation is not None and generation.thinking is not None:
        return generation.thinking
    return _Intent(thinking_effort)


# --- thinking ----------------------------------------------------------------


def _native_thinking(
    record: ModelRecord, intent: ThinkingIntent, who: str
) -> NativeThinking | None:
    rule = record.thinking
    mechanism = rule.mechanism
    mode = intent.mode
    if mechanism is ThinkingMechanism.NONE:
        if mode != "none":
            logger.debug(
                "Model %s has no thinking knob; thinking mode %r ignored", record.ref, mode
            )
        return None

    def unavailable() -> ModelResolutionError:
        return ModelResolutionError(
            f"{who}: thinking mode {mode!r} is not available on {record.ref} "
            f"({_DESCRIPTIONS[mechanism]}); use {_allowed(rule)}."
        )

    if mechanism in _BUDGET_MECHANISMS:
        if mode == "none":
            if rule.can_disable:
                return _off(mechanism, budget=0)
            return _budget(mechanism, rule.budget_range[0])
        if mode == "budget":
            requested = intent.budget_tokens
            if requested is None:
                requested = _DEFAULT_BUDGET
            if requested == 0 and rule.can_disable:
                return _off(mechanism, budget=0)
            return _budget(mechanism, _clamp_budget(record, requested))
        if mode in _GENERIC_LEVELS:
            budget = rule.budgets[_GENERIC_LEVELS.index(mode)]
            return _budget(mechanism, _clamp_budget(record, budget))
        raise unavailable()

    if mechanism is ThinkingMechanism.CHAT_TEMPLATE:
        if mode == "none":
            return _off(mechanism)
        if mode in _GENERIC_LEVELS:
            return NativeThinking(mechanism, enabled=True, label="thinking on")
        raise unavailable()

    # The ladder mechanisms: Gemini levels, adaptive Claude, effort.
    if mode == "none":
        if mechanism is ThinkingMechanism.EFFORT:
            return _level(mechanism, rule.levels[0])
        if rule.can_disable:
            return _off(mechanism)
        return _level(mechanism, rule.levels[0])
    level = _ladder_level(rule, mode)
    if level is None:
        raise unavailable()
    return _level(mechanism, level)


def _ladder_level(rule: ThinkingRule, mode: str) -> str | None:
    """The native level ``mode`` means on this ladder, or None."""
    if mode in rule.levels:
        return mode
    if mode in _GENERIC_LEVELS:
        return rule.generic_map.get(mode)
    return None


def _clamp_budget(record: ModelRecord, requested: int) -> int:
    """A budget within the model's range (and, for Claude, below its max output)."""
    low, high = record.thinking.budget_range
    if record.thinking.mechanism is ThinkingMechanism.ANTHROPIC_BUDGET and record.max_output:
        high = min(high, record.max_output - 1)
    budget = min(max(requested, low), high)
    if budget != requested:
        logger.debug("Thinking budget %d for %s clamped to %d", requested, record.ref, budget)
    return budget


def _off(mechanism: ThinkingMechanism, *, budget: int | None = None) -> NativeThinking:
    return NativeThinking(mechanism, enabled=False, budget=budget, label="off")


def _budget(mechanism: ThinkingMechanism, budget: int) -> NativeThinking:
    return NativeThinking(mechanism, enabled=True, budget=budget, label=f"budget {budget}")


def _level(mechanism: ThinkingMechanism, level: str) -> NativeThinking:
    if mechanism is ThinkingMechanism.ANTHROPIC_ADAPTIVE:
        label = f"adaptive, effort {level}"
    elif mechanism is ThinkingMechanism.GEMINI_LEVEL:
        label = f"level {level}"
    else:
        label = f"effort {level}"
    # An effort of "none" is the model's own way of not reasoning.
    return NativeThinking(mechanism, enabled=level != "none", level=level, label=label)


def _allowed(rule: ThinkingRule) -> str:
    """The thinking modes a model takes, for an error message."""
    if rule.mechanism in _BUDGET_MECHANISMS:
        names = ["none", *_GENERIC_LEVELS, "budget"]
    elif rule.mechanism is ThinkingMechanism.CHAT_TEMPLATE:
        names = ["none", *_GENERIC_LEVELS]
    else:
        available = {"none", *rule.levels}
        available.update(g for g in _GENERIC_LEVELS if _ladder_level(rule, g) is not None)
        known = [name for name in _LEVEL_ORDER if name in available]
        names = known + [name for name in rule.levels if name not in known]
    return ", ".join(names[:-1]) + f" or {names[-1]}" if len(names) > 1 else names[0]


# --- max_tokens and sampling -------------------------------------------------


def _max_tokens(
    record: ModelRecord, generation: GenerationSettings | None, thinking: NativeThinking | None
) -> int | None:
    """The one ``max_tokens`` both adapters send.

    The agent's value, else the provider default (Anthropic 8192); raised on
    budget-thinking Claude to the budget plus 8192 (Anthropic counts thinking
    toward max_tokens and requires it above the budget); then clamped to the
    model's max output.
    """
    max_tokens = generation.max_tokens if generation is not None and generation.max_tokens else None
    if max_tokens is None and record.kind is ProviderKind.ANTHROPIC:
        max_tokens = _ANTHROPIC_DEFAULT_MAX_TOKENS
    if (
        thinking is not None
        and thinking.enabled
        and thinking.mechanism is ThinkingMechanism.ANTHROPIC_BUDGET
    ):
        max_tokens = max(max_tokens or 0, thinking.budget + _ANTHROPIC_OUTPUT_HEADROOM)
    if max_tokens is not None and record.max_output and max_tokens > record.max_output:
        logger.debug(
            "max_tokens %d for %s clamped to %d", max_tokens, record.ref, record.max_output
        )
        max_tokens = record.max_output
    return max_tokens


def _sampling(
    record: ModelRecord, generation: GenerationSettings | None, thinking: NativeThinking | None
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """The sampling settings to send, and the names of those dropped."""
    values = {name: getattr(generation, name, None) for name in _SAMPLING_FIELDS}
    drop = record.sampling == "never" or (
        record.sampling == "without_thinking" and thinking is not None and thinking.enabled
    )
    if not drop:
        return values, ()
    dropped = tuple(name for name in _SAMPLING_FIELDS if values[name] is not None)
    if dropped:
        logger.debug(
            "Dropping %s for %s: the model does not take them here", ", ".join(dropped), record.ref
        )
    return dict.fromkeys(_SAMPLING_FIELDS), dropped
