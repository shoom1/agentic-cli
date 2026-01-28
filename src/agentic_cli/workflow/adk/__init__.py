"""Google ADK-specific workflow implementation.

This submodule is a placeholder for future ADK-specific components.
Currently, ADK implementation remains in workflow/adk_manager.py.

Future structure:
- manager.py: GoogleADKWorkflowManager (refactored)
- middleware/: ADK middleware wrappers
- persistence/: ADK-specific storage backends
- tools/: ADK-native tool implementations
"""

# Re-export the existing manager for convenience
from agentic_cli.workflow.adk_manager import GoogleADKWorkflowManager

__all__ = [
    "GoogleADKWorkflowManager",
]
