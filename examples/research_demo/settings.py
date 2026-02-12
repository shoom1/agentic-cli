"""Settings for the Research Demo application."""

from pathlib import Path

from pydantic import Field
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

    app_name: str = Field(default="research_demo")
    workspace_dir: Path = Field(default=Path.home() / ".research_demo")

    def __init__(self, **kwargs):
        # Override verbose_thinking default to False (concise mode)
        kwargs.setdefault("verbose_thinking", False)
        super().__init__(**kwargs)
