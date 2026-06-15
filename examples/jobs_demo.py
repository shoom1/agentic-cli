"""Jobs Demo - long-running background jobs with a live status-bar monitor.

Showcases the job-control substrate end to end:

1. The agent has ``run_shell_job`` (a typed long-running tool) plus the job
   management tools, so it can launch detached shell jobs and check on them.
2. The harness ``JobMonitor`` keeps a live ``jobs: N running``/``✓ done``
   segment in the status bar — independent of the agent loop — so jobs stay
   visible while you're idle at the prompt.
3. ``/jobs`` lists/inspects/cancels/cleans jobs.

Run with:
    conda run -n agenticcli python examples/jobs_demo.py

Things to try once it's up (watch the status bar, bottom of the screen):
    > run `sleep 30; echo finished` in the background
    > start three jobs: sleep 10, sleep 20, sleep 30
    > what's the status of my jobs?
    > /jobs
    > /jobs all
    > /jobs <id>

API keys are read from ``~/.research_demo/.env`` (shared with the research
demo) or from standard environment variables (GOOGLE_API_KEY / ANTHROPIC_API_KEY).
"""

import asyncio
from functools import lru_cache
from pathlib import Path

from pydantic_settings import SettingsConfigDict
from rich.panel import Panel
from rich.text import Text

from agentic_cli import BaseCLIApp, BaseSettings
from agentic_cli.cli import AppInfo
from agentic_cli.tools import JOB_MANAGEMENT_TOOLS, run_shell_job
from agentic_cli.workflow import AgentConfig


# =============================================================================
# Settings
# =============================================================================


class Settings(BaseSettings):
    """Settings for the Jobs demo.

    Reuses the research demo's ``.env`` (``~/.research_demo/.env``) so API keys
    only have to live in one place. API keys may also be set via standard
    environment variables (GOOGLE_API_KEY / ANTHROPIC_API_KEY).
    """

    model_config = SettingsConfigDict(
        env_file=str(Path.home() / ".research_demo" / ".env"),
        extra="ignore",
    )

    def __init__(self, **kwargs):
        kwargs.setdefault("app_name", "jobs_demo")
        kwargs.setdefault("workspace_dir", Path.home() / ".jobs_demo")
        # Cap concurrent jobs low so the queue is easy to observe in the demo.
        kwargs.setdefault("max_concurrent_jobs", 2)
        # Permissions off for a frictionless demo — every run_shell_job would
        # otherwise prompt for approval (it declares a default-ASK capability).
        kwargs.setdefault("permissions_enabled", False)
        super().__init__(**kwargs)


@lru_cache
def get_settings() -> Settings:
    return Settings()


# =============================================================================
# Agent & App
# =============================================================================

AGENT_PROMPT = """You are a build/ops assistant that runs background jobs.

When the user asks you to run something that may take a while (builds, tests,
downloads, sleeps), use `run_shell_job` to start it as a detached background
job and immediately report the returned job_id. Do NOT block waiting for it.

To check on a job, call `job_status(job_id)` — it returns the state, a stdout
tail, and (once finished) the result. Use `job_list` to enumerate jobs and
`job_cancel` to stop one.

Tell the user they can watch live progress in the status bar at the bottom of
the screen, and inspect jobs anytime with the /jobs command.
"""


AGENT_CONFIGS = [
    AgentConfig(
        name="ops_assistant",
        prompt=AGENT_PROMPT,
        tools=[run_shell_job, *JOB_MANAGEMENT_TOOLS],
        description="Runs and monitors long-running background shell jobs",
    ),
]


def _create_app_info() -> AppInfo:
    text = Text()
    text.append("Jobs Demo\n\n", style="bold cyan")
    text.append("Long-running background jobs + live status-bar monitor.\n\n", style="dim")
    text.append("Try: ", style="dim")
    text.append("run `sleep 30; echo done` in the background\n", style="white")
    text.append("Then watch the status bar, or type ", style="dim")
    text.append("/jobs", style="white")
    return AppInfo(
        name="Jobs Demo",
        version="0.1.0",
        welcome_message=lambda: Panel(text, border_style="cyan"),
        echo_thinking=False,
    )


if __name__ == "__main__":
    app = BaseCLIApp(
        app_info=_create_app_info(),
        agent_configs=AGENT_CONFIGS,
        settings=get_settings(),
    )
    asyncio.run(app.run())
