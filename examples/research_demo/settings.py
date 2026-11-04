"""Settings for the Research Demo application.

The demo ships a short list of models (``models``), the way an app built on
agentic-cli offers its users a choice of models. The ``/settings`` model
pickers offer the models of the list whose provider has a key (and the model
each role resolves to):

- Google: ``GOOGLE_API_KEY``, for Gemini 3.5 Flash-Lite and Gemini 3.8 Flash.
- Anthropic: ``ANTHROPIC_API_KEY``, for Claude Haiku 5.5, Claude Sonnet 5.5
  and Claude Opus 5.5.

The list holds no Gemini Pro (Google's is a preview, which a later release
retires), so with only a Google key the Strong Model picker shows the
newest Pro the workflow computes for the ``strong`` role.

Put the keys in ``~/.research_demo/.env`` (``GOOGLE_API_KEY=...``, one per
line) or in the environment; a ``.env`` in the working directory is never
read for them.

The demo defines no other provider: OpenRouter, or an OpenAI-compatible
service, is yours to define in ``~/.research_demo/settings.json`` (a
project's settings file cannot), with the models you want from it. OpenRouter
and three of its models, the key in ``OPENROUTER_API_KEY``::

    {
      "model_providers": {
        "openrouter": {
          "kind": "openrouter",
          "base_url": "https://openrouter.ai/api/v1",
          "api_key_env": "OPENROUTER_API_KEY"
        }
      },
      "models": {
        "openrouter:qwen/qwen3.7-plus": {},
        "openrouter:deepseek/deepseek-v4-pro": {},
        "openrouter:moonshotai/kimi-k3": {}
      }
    }

An OpenAI-compatible service (Together, Fireworks, a local vLLM) takes no
``kind``: the README's "Configured providers" has examples.

To pick a model, open ``/settings``: Default Model offers the list, and the
pick is saved to the project's ``./.research_demo/settings.json``. ``/models``
shows the list, with what is known about each model and why one is not
offered; ``/models <text>`` searches every provider's full listing, such as
``/models llama``, to find an id to add to ``models``.

Without a ``default_model``, the demo runs on the list's default-tier model:
Gemini 3.8 Flash with a Google key, else Claude Sonnet 5.5 with an Anthropic
key. A configured provider's model has no tier, so with only such a
provider's key, choose the default model too: pick one in ``/settings``,
which works while the workflow is down, or add
``"default_model": "openrouter:qwen/qwen3.7-plus"`` to
``~/.research_demo/settings.json``.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from agentic_cli import BaseSettings
from agentic_cli.models import ModelEntry

# The framework's field whose default the demo overrides. A field declared
# again replaces its title and description, so they are carried over.
_MODELS = BaseSettings.model_fields["models"]


class ResearchDemoSettings(BaseSettings):
    """Settings for the Research Demo application.

    Demonstrates all P0/P1 features with memory, planning, and HITL.

    Settings are loaded from (highest precedence first):

    1. Constructor arguments
    2. Environment variables (``RESEARCH_DEMO_*`` prefix)
    3. Project config ``./.research_demo/settings.json`` — **untrusted**: a
       cloned repo can ship one, so only an explicit allowlist of benign keys
       is honoured and security-sensitive keys are dropped with a warning
    4. User config ``~/.research_demo/settings.json`` (trusted)
    5. ``~/.research_demo/.env`` — trusted because the path is absolute; a
       cwd-relative ``.env`` would be filtered like the project config, so put
       API keys here or in real environment variables
    6. Field defaults (including the ones set in ``model_post_init`` below)

    JSON sources are only consulted when the file exists. ``models`` and
    ``model_providers`` are the exception to "highest precedence wins": the
    sources' entries are added to the demo's, field by field, and none is
    removed (``model_providers``, which the demo leaves empty, from the user
    config and the environment only).
    """

    model_config = SettingsConfigDict(
        env_file=str(Path.home() / ".research_demo" / ".env"),
        env_prefix="RESEARCH_DEMO_",
        extra="ignore",
    )

    # Two Gemini models (fast, default tier) and three Claude models (fast,
    # default, strong). Each entry is empty: the listings and the shipped
    # table say what each model takes.
    models: dict[str, ModelEntry] = Field(
        default_factory=lambda: {
            "google:gemini-3.5-flash-lite": ModelEntry(),
            "google:gemini-3.8-flash": ModelEntry(),
            "anthropic:claude-haiku-5-5": ModelEntry(),
            "anthropic:claude-sonnet-5-5": ModelEntry(),
            "anthropic:claude-opus-5-5": ModelEntry(),
        },
        title=_MODELS.title,
        description=_MODELS.description,
    )

    def __init__(self, **kwargs):
        kwargs.setdefault("app_name", "research_demo")
        kwargs.setdefault("workspace_dir", Path.home() / ".research_demo")
        super().__init__(**kwargs)

    def model_post_init(self, __context):
        """Override verbose_thinking default without blocking JSON persistence.

        kwargs.setdefault would inject at init level (highest priority),
        overriding saved JSON values. model_post_init runs after all sources
        are resolved, and model_fields_set tracks which fields were explicitly
        set by any source (env, JSON, init kwargs). We only apply our custom
        default when no source provided a value.
        """
        if "verbose_thinking" not in self.model_fields_set:
            object.__setattr__(self, "verbose_thinking", False)
        if "sandbox_data_mounts" not in self.model_fields_set:
            data_dir = Path(__file__).parent / "data"
            object.__setattr__(self, "sandbox_data_mounts", [f"{data_dir}:samples"])
        if "sandbox_outputs_dir" not in self.model_fields_set:
            object.__setattr__(self, "sandbox_outputs_dir", str(Path(self.workspace_dir) / "artifacts"))
        if "skills_dirs" not in self.model_fields_set:
            object.__setattr__(
                self, "skills_dirs", [str(Path(__file__).parent / "skills")]
            )
