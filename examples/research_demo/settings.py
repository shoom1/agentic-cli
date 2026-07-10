"""Settings for the Research Demo application."""

from pathlib import Path

from pydantic_settings import SettingsConfigDict

from agentic_cli import BaseSettings


class ResearchDemoSettings(BaseSettings):
    """Settings for the Research Demo application.

    Demonstrates all P0/P1 features with memory, planning, and HITL.

    Settings are loaded from (in order of precedence):
    1. Environment variables (RESEARCH_DEMO_* prefix)
    2. Project config (./settings.json)
    3. User config (~/.research_demo/settings.json)
    4. .env file (~/.research_demo/.env)
    5. Default values
    """

    model_config = SettingsConfigDict(
        env_file=str(Path.home() / ".research_demo" / ".env"),
        env_prefix="RESEARCH_DEMO_",
        extra="ignore",
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
