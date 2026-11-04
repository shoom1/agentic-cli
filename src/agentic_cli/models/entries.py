"""Model entries: what an app, a user or a project says about one model.

The ``models`` setting maps a ``provider:model`` reference to a
``ModelEntry``. Every field is optional: one that is set wins over the
provider's listing and the shipped table (``ModelCatalog`` marks it
``config``), and one left unset leaves their value.

The thinking fields (``THINKING_FIELDS``) are for a model on another
provider than Google, Anthropic and OpenAI, whose thinking knob only the
user can describe. Those three providers' thinking comes from their
listings and the table, so an entry on one of their models may not set it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_cli.models.refs import BUILTIN_PROVIDERS, ModelRef, ModelSpecError, parse_spec

#: The fields only a model on a configured provider may set.
THINKING_FIELDS: tuple[str, ...] = ("thinking", "effort_levels", "reasoning_text")
#: Effort levels when an entry says thinking "effort" and names none.
DEFAULT_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high")

_LEVEL = re.compile(r"[a-z][a-z0-9_]*")


class ModelEntry(BaseModel):
    """One ``models`` entry: what is known about the model, field by field."""

    model_config = ConfigDict(extra="forbid")

    context_window: int | None = Field(default=None, ge=1)
    max_output: int | None = Field(default=None, ge=1)
    tools: bool | None = None
    image_input: bool | None = None
    #: ``chat_template``: thinking switched by ``chat_template_kwargs.enable_thinking``
    #: (Qwen3 and similar on vLLM); ``effort``: a reasoning effort parameter.
    thinking: Literal["none", "chat_template", "effort"] | None = None
    #: The effort ladder, lowest first (``DEFAULT_EFFORT_LEVELS`` when thinking
    #: is ``effort`` and this is unset).
    effort_levels: list[str] | None = None
    reasoning_text: bool | None = None

    @field_validator("effort_levels")
    @classmethod
    def _check_levels(cls, levels: list[str] | None) -> list[str] | None:
        if levels is None:
            return None
        if not levels:
            raise ValueError("effort_levels must name at least one level")
        bad = [level for level in levels if not _LEVEL.fullmatch(level)]
        if bad:
            raise ValueError(f"effort level(s) {bad} must be lowercase names")
        if len(set(levels)) != len(levels):
            raise ValueError("effort_levels must not repeat a level")
        return levels


def parse_entry_key(key: str) -> ModelRef:
    """The model a ``models`` key names: a ``provider:model`` reference only.

    The key is split on its first colon, as references are, so
    ``openrouter:qwen/qwen3-235b-a22b:free`` keeps its ``:free``. A bare model
    id is refused even where a reference may be one, so that one model has
    one key.

    Raises:
        ModelSpecError: For a bare model id (the message gives its full
            form), or anything else that is not a reference; the message
            names the key.
    """
    try:
        parsed = parse_spec(key)
    except ModelSpecError:
        parsed = None
    if isinstance(parsed, ModelRef):
        if ":" in key:
            return parsed
        raise ModelSpecError(f"models key {key!r} needs its provider: write it as {parsed}.")
    raise ModelSpecError(
        f"models key {key!r} is not a provider:model reference, such as "
        "anthropic:claude-sonnet-5-5."
    )


def check_entries(entries: Mapping[str, ModelEntry]) -> dict[str, ModelEntry]:
    """``entries``, once every key is a reference and every entry fits its provider.

    Raises:
        ValueError: Naming the key, for a key ``parse_entry_key`` refuses, or
            for a thinking field set on a Google, Anthropic or OpenAI model
            (set: given, even as null).
    """
    for key, entry in entries.items():
        ref = parse_entry_key(key)
        if ref.provider in BUILTIN_PROVIDERS and entry.model_fields_set & set(THINKING_FIELDS):
            raise ValueError(
                f"models[{key!r}]: thinking, effort_levels and reasoning_text are only for a "
                "model on a model_providers provider; the built-in providers' thinking comes "
                "from their listings and the table."
            )
    return dict(entries)
