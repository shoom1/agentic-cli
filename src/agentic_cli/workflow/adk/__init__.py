"""Google ADK-specific workflow implementation.

This submodule contains ADK-specific components:
- manager.py: GoogleADKWorkflowManager (in adk_manager.py)
- llm_event_logger.py: LLM traffic logging for debugging

Future structure:
- middleware/: ADK middleware wrappers
- persistence/: ADK-specific storage backends
- tools/: ADK-native tool implementations
"""

# Re-export the existing manager for convenience
from agentic_cli.workflow.adk_manager import GoogleADKWorkflowManager
from agentic_cli.workflow.adk.llm_event_logger import LLMEventLogger

__all__ = [
    "GoogleADKWorkflowManager",
    "LLMEventLogger",
]
