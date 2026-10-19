"""An ``app_name`` ADK cannot run is refused when the ADK manager is built.

ADK 2's ``App`` validates its name: letters, digits, ``_`` and ``-``, starting
with a letter, and never ``user``. ``Runner(app_name=...)`` validated nothing,
so an app could carry a name ADK now rejects, and it would fail only on the
first turn, inside initialization. The manager checks it with ADK's own rule
at construction, with an error that says which setting is wrong.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.adk")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.factory import create_workflow_manager_from_settings  # noqa: E402


@pytest.mark.parametrize(
    "name, rule",
    [
        ("my.app", "must start with a letter"),
        ("1app", "must start with a letter"),
        ("user", "reserved"),
    ],
)
def test_an_app_name_adk_rejects_is_refused_at_construction(name, rule, mock_context):
    with pytest.raises(ValueError) as exc:
        GoogleADKWorkflowManager(
            agent_configs=[], settings=mock_context.settings, app_name=name
        )

    message = str(exc.value)
    assert message.startswith(f"app_name '{name}' cannot be used with the ADK backend: ")
    assert rule in message


@pytest.mark.parametrize(
    "name", ["research_demo", "airesearcher_cli", "agentic_web", "agentic_cli"]
)
def test_the_app_names_in_use_construct(name, mock_context):
    manager = GoogleADKWorkflowManager(
        agent_configs=[], settings=mock_context.settings, app_name=name
    )

    assert manager.app_name == name


def test_an_app_name_from_the_settings_is_checked_too(tmp_path):
    settings = BaseSettings(workspace_dir=tmp_path, app_name="my app")

    with pytest.raises(ValueError, match="app_name 'my app' cannot be used with the ADK backend"):
        create_workflow_manager_from_settings([], settings)
