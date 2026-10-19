"""What is known about one model: a ``ModelRecord``.

A record carries the model's numbers (context window, max output), what it
accepts (tools, image input, sampling parameters), its thinking knob as a
``ThinkingRule``, whether it returns reasoning text, and which prompt-cache
mechanism its provider offers. ``sources`` says where each value came from:
``declared`` (the user's endpoint settings), ``listing`` (the provider's model
list), ``table`` (the shipped table) or ``default`` (nothing knew; the
conservative value).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from agentic_cli.models.endpoints import ProviderKind
from agentic_cli.models.refs import ModelRef


class ThinkingMechanism(str, Enum):
    """How a model's thinking is switched and sized."""

    #: No thinking knob: the thinking setting is ignored.
    NONE = "none"
    #: Gemini 2.5: a token budget (``thinking_budget``).
    GEMINI_BUDGET = "gemini_budget"
    #: Gemini 3 and later: a discrete ``thinking_level``.
    GEMINI_LEVEL = "gemini_level"
    #: Claude 4.6 and later: adaptive thinking, sized by an effort level.
    ANTHROPIC_ADAPTIVE = "anthropic_adaptive"
    #: Claude up to 4.5: enabled thinking with ``budget_tokens``.
    ANTHROPIC_BUDGET = "anthropic_budget"
    #: OpenAI and OpenRouter reasoning models, endpoints declaring effort.
    EFFORT = "effort"
    #: Endpoint models switched by ``chat_template_kwargs.enable_thinking``.
    CHAT_TEMPLATE = "chat_template"


class CacheMechanism(str, Enum):
    """The prompt cache a model's provider offers."""

    NONE = "none"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


@dataclass(frozen=True)
class ThinkingRule:
    """A model's thinking knob.

    Attributes:
        mechanism: How thinking is switched and sized.
        levels: The native ladder, lowest first (``minimal``, ``low``, …,
            ``max``): effort names, or Gemini thinking levels.
        budgets: The budgets the generic ``low``/``medium``/``high`` map to
            (budget mechanisms).
        budget_range: The inclusive range an enabled budget is clamped to
            (budget mechanisms).
        can_disable: Whether thinking can be turned off. When it cannot,
            ``none`` means the least thinking the model allows.
        generic_map: A generic level the ladder lacks, and the native level it
            becomes (Gemini 3 Pro: ``{"medium": "high"}``).
    """

    mechanism: ThinkingMechanism = ThinkingMechanism.NONE
    levels: tuple[str, ...] = ()
    budgets: tuple[int, int, int] | None = None
    budget_range: tuple[int, int] | None = None
    can_disable: bool = True
    generic_map: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRecord:
    """Everything the model layer knows about one model of one provider."""

    provider: str
    kind: ProviderKind
    model: str
    display_name: str = ""
    #: The role tier the model fills when a role is not mapped: ``fast``,
    #: ``default``, ``strong``, or None.
    tier: str | None = None
    context_window: int | None = None
    max_output: int | None = None
    tools: bool = True
    image_input: bool = False
    thinking: ThinkingRule = field(default_factory=ThinkingRule)
    reasoning_text: bool = False
    cache: CacheMechanism = CacheMechanism.NONE
    #: When temperature, top-p and top-k may be sent: ``always``,
    #: ``without_thinking`` (dropped while thinking or effort is on), or
    #: ``never`` (the model rejects them).
    sampling: Literal["always", "without_thinking", "never"] = "always"
    #: Field name -> ``declared``, ``listing``, ``table`` or ``default``.
    sources: Mapping[str, str] = field(default_factory=dict)

    @property
    def ref(self) -> ModelRef:
        return ModelRef(self.provider, self.model)
