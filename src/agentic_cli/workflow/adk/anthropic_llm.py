"""Bare ``claude-*`` model strings resolve to the direct Anthropic API.

ADK's ``LLMRegistry`` maps a ``claude-*`` model string to ``Claude``, which
serves Claude from Vertex AI (and needs GOOGLE_CLOUD_PROJECT/LOCATION). The
agents agentic-cli builds never go through the registry: the model factory
(``workflow/adk/model_factory.py``) gives each one an ``AnthropicLlm``
instance. A native ADK agent YAML (``adk_config_path``) still names its model
as a string, so ``register_direct_anthropic()`` points ``claude-*`` at
``DirectAnthropicLlm``, ADK's direct-API ``AnthropicLlm`` under a name of its
own. ADK builds it with the class defaults and its own client, keyed by
``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.models.registry import LLMRegistry
from pydantic import ConfigDict

from agentic_cli.logging import Loggers

logger = Loggers.workflow()

# Replaces ADK's own entry for the same pattern (Claude on Vertex AI).
_CLAUDE_PATTERNS = [r"claude-.*"]


class DirectAnthropicLlm(AnthropicLlm):
    """Claude on the direct Anthropic API, for model strings ADK resolves.

    Fields it does not have are refused rather than ignored: 0.6's ``effort``,
    ``extra_params``, ``request_timeout`` and ``max_retries`` would otherwise
    be dropped without a word.
    """

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def supported_models(cls) -> list[str]:
        return list(_CLAUDE_PATTERNS)


def register_direct_anthropic() -> None:
    """Resolve ``claude-*`` strings to ``DirectAnthropicLlm`` via ``LLMRegistry``.

    Idempotent; clears the ``resolve()`` cache so already-resolved names pick
    up the change.
    """
    LLMRegistry.register(DirectAnthropicLlm)
    cache_clear = getattr(LLMRegistry.resolve, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    logger.debug("direct_anthropic_registered", patterns=_CLAUDE_PATTERNS)
