"""User interaction tools for agentic workflows.

Provides tools for requesting clarification and input from the user.
"""

from typing import Any

from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)


@register_tool(
    category=ToolCategory.INTERACTION,
    permission_level=PermissionLevel.SAFE,
    description="Ask the user a clarifying question and wait for their response. Use this when requirements are ambiguous or you need user input to proceed.",
)
async def ask_clarification(
    question: str,
    options: list[str] | None = None,
) -> dict[str, Any]:
    """Ask the user for clarification.

    This tool pauses execution and requests input from the user via
    the CLI. It uses the workflow context to emit a USER_INPUT_REQUIRED
    event that the CLI will handle.

    Args:
        question: The question to ask
        options: Optional list of suggested answers (shown as choices)

    Returns:
        Dictionary with the user's response
    """
    import uuid
    from agentic_cli.config import get_context_workflow
    from agentic_cli.workflow.events import UserInputRequest

    workflow = get_context_workflow()

    if workflow is None:
        # Fallback for when not running within a workflow context
        return {
            "question": question,
            "options": options or [],
            "error": "No workflow context available for user interaction",
            "response": None,
        }

    from agentic_cli.workflow.events import InputType

    # Create user input request
    request = UserInputRequest(
        request_id=str(uuid.uuid4()),
        tool_name="ask_clarification",
        prompt=question,
        input_type=InputType.CHOICE if options else InputType.TEXT,
        choices=options,
    )

    # Request user input (this will block until CLI provides response)
    response = await workflow.request_user_input(request)

    return {
        "question": question,
        "options": options or [],
        "response": response,
        "summary": f"User responded: {response[:50]}{'...' if len(response) > 50 else ''}",
    }
