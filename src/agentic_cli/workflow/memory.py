"""Conversation memory with automatic summarization.

Provides intelligent context management for long conversations by maintaining
a summary of older messages plus recent messages verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from agentic_cli.workflow.manager import WorkflowManager


@dataclass
class ConversationMemory:
    """Manages conversation history with automatic summarization.

    When conversation exceeds a threshold, older messages are summarized
    to maintain context while staying within token limits.

    Attributes:
        max_recent_messages: Number of recent messages to keep verbatim
        summarize_threshold: Trigger summarization when messages exceed this count
        summary: Condensed summary of older conversation
        messages: Full message history
    """

    max_recent_messages: int = 10
    summarize_threshold: int = 20
    summary: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to history.

        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

    def should_summarize(self) -> bool:
        """Check if summarization is needed based on message count."""
        return len(self.messages) > self.summarize_threshold

    def get_context_messages(self) -> list[dict[str, Any]]:
        """Get messages to send to agent: summary + recent.

        Returns:
            List of messages with optional summary prefix and recent messages
        """
        recent = self.messages[-self.max_recent_messages:]

        if self.summary:
            summary_msg = {
                "role": "system",
                "content": f"Summary of earlier conversation:\n{self.summary}",
            }
            return [summary_msg] + recent

        return recent

    def get_full_history(self) -> list[dict[str, Any]]:
        """Get complete message history without summarization.

        Returns:
            All messages in history
        """
        return list(self.messages)

    async def summarize(
        self,
        summarizer: Callable[[list[dict[str, Any]]], Awaitable[str]],
    ) -> None:
        """Summarize older messages, keep recent ones.

        Args:
            summarizer: Async function that takes messages and returns summary text
        """
        if not self.should_summarize():
            return

        # Messages to summarize (everything except recent)
        to_summarize = self.messages[:-self.max_recent_messages]

        # Include previous summary in summarization for continuity
        if self.summary:
            to_summarize.insert(0, {
                "role": "system",
                "content": f"Previous summary: {self.summary}",
            })

        # Generate new summary
        self.summary = await summarizer(to_summarize)

        # Keep only recent messages
        self.messages = self.messages[-self.max_recent_messages:]

    def clear(self) -> None:
        """Clear all messages and summary."""
        self.messages = []
        self.summary = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize memory state to dictionary."""
        return {
            "max_recent_messages": self.max_recent_messages,
            "summarize_threshold": self.summarize_threshold,
            "summary": self.summary,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationMemory":
        """Deserialize memory state from dictionary."""
        return cls(
            max_recent_messages=data.get("max_recent_messages", 10),
            summarize_threshold=data.get("summarize_threshold", 20),
            summary=data.get("summary"),
            messages=data.get("messages", []),
        )


SUMMARIZATION_PROMPT = """Summarize this conversation concisely, preserving:
- Key facts and decisions made
- Important context and constraints mentioned
- Any action items or next steps discussed
- Technical details that may be referenced later

Keep the summary focused and under 500 words.

Conversation:
"""


async def create_summarizer(workflow: "WorkflowManager") -> Callable:
    """Create a summarizer function using the workflow's model.

    Args:
        workflow: WorkflowManager instance to use for LLM calls

    Returns:
        Async function that summarizes a list of messages
    """
    async def summarize(messages: list[dict[str, Any]]) -> str:
        # Build prompt from messages
        prompt = SUMMARIZATION_PROMPT
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            prompt += f"\n{role}: {content}\n"

        # Use workflow's generate_simple method for summarization
        response = await workflow.generate_simple(prompt, max_tokens=500)
        return response

    return summarize
