"""Backend-neutral per-agent model parameters.

``ModelSettings`` carries generation parameters (temperature, top-p/k, max
tokens, stop sequences, thinking/reasoning effort) for a single agent,
independent of the orchestration backend. Each workflow manager translates
these neutral fields into its backend's native config:

- ADK   -> ``google.genai.types.GenerateContentConfig`` + a ``BuiltInPlanner``
           (for the thinking config).
- LangGraph (deferred) -> ``init_chat_model`` keyword arguments.

The ``extra`` dict is an escape hatch for provider-specific parameters that
have no neutral field; each backend filters it to keys it understands.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# Thinking modes: a lowercase name. ``none`` turns thinking off wherever the
# model allows it; ``low``/``medium``/``high`` are the model's own levels;
# ``budget`` uses an explicit token budget (see ``ThinkingSettings.budget_tokens``).
# Any other name is a native level of the model (``minimal``, ``xhigh``, ``max``
# on models that have them), checked against the model's ladder when the
# workflow resolves its agents (``agentic_cli.models.resolve_model``).
ThinkingMode = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]


class ThinkingSettings(BaseModel):
    """Per-agent thinking/reasoning configuration.

    Attributes:
        mode: ``none``, ``low``, ``medium``, ``high``, ``budget`` (an explicit
            token budget), or a native level of the agent's model such as
            ``xhigh``; resolving the agent's model rejects a level the model
            lacks.
        budget_tokens: Explicit thinking-token budget (only used when
            ``mode == "budget"``).
    """

    model_config = ConfigDict(protected_namespaces=())

    mode: ThinkingMode = "none"
    budget_tokens: int | None = Field(default=None, ge=0)


class ModelSettings(BaseModel):
    """Backend-neutral generation parameters for a single agent.

    All fields are optional; a ``None`` value means "leave the backend/model
    default in place". Field names are neutral — backends map them to native
    names (e.g. ``max_tokens`` -> ADK ``max_output_tokens``).

    Attributes:
        temperature: Sampling temperature.
        top_p: Nucleus-sampling probability mass.
        top_k: Top-k sampling cutoff.
        max_tokens: Maximum output tokens.
        stop_sequences: Sequences that stop generation.
        thinking: Thinking/reasoning configuration.
        extra: Provider-specific parameters passed through to the backend
            (filtered per-backend to recognised keys).
    """

    model_config = ConfigDict(protected_namespaces=())

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    thinking: ThinkingSettings | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
