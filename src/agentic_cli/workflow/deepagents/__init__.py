"""The Deep Agents orchestrator (``orchestrator = "deepagents"``).

Each ``AgentConfig`` becomes one LangChain ``create_agent`` graph, and a
coordinator delegates to its sub-agents through the ``task`` tool of Deep
Agents' ``SubAgentMiddleware``. Our own middleware gates every tool call and
turns a raising tool into an error result, so the framework's permission
engine, tools and models are the same as on the other backends.

Installed by the ``deepagents`` extra: ``pip install 'agentic-cli[deepagents]'``.
"""

# Imported eagerly, and only here: without the extra this package fails to
# import at once, and the backend registry (workflow/factory.py) turns that
# ImportError into the extra's install hint. No module outside this package
# imports Deep Agents.
import deepagents  # noqa: F401
import langchain.agents  # noqa: F401

from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.deepagents.summarization import install_history_filters  # noqa: E402

# Deep Agents' "older messages will not be recoverable" notice is wrong for
# this backend (the checkpoint keeps them); it would print over the UI.
install_history_filters()

__all__ = ["DeepAgentsWorkflowManager"]
