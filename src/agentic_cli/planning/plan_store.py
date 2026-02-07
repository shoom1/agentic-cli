"""Simple plan store for agent work plans.

Stores a markdown plan string that the agent manages in-context.
The agent uses markdown checkboxes (- [ ] / - [x]) for task tracking.
The framework just persists the string — no DAG machinery needed.

Example:
    >>> store = PlanStore()
    >>> store.save("## Research Plan\\n- [ ] Gather data\\n- [ ] Analyze results")
    >>> print(store.get())
    ## Research Plan
    - [ ] Gather data
    - [ ] Analyze results
"""


class PlanStore:
    """Simple string store for agent plans.

    Follows the same pattern as MemoryStore — minimal backing store
    that lets the LLM handle structure in-context.
    """

    def __init__(self) -> None:
        """Initialize an empty plan store."""
        self._content: str = ""

    def save(self, content: str) -> None:
        """Save or overwrite the plan content.

        Args:
            content: Markdown plan string (use checkboxes for tasks).
        """
        self._content = content

    def get(self) -> str:
        """Get the current plan content.

        Returns:
            The plan string, or empty string if no plan exists.
        """
        return self._content

    def is_empty(self) -> bool:
        """Check if the plan store has content.

        Returns:
            True if no plan has been saved.
        """
        return not self._content

    def clear(self) -> None:
        """Clear the plan content."""
        self._content = ""
