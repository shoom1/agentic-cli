"""Planning module for managing agent work plans.

This module provides PlanStore for persisting agent-managed markdown plans.
The agent writes plans with checkboxes; the framework just stores the string.

Example:
    >>> from agentic_cli.planning import PlanStore
    >>> store = PlanStore()
    >>> store.save("- [ ] Gather data\\n- [ ] Analyze results")
    >>> print(store.get())
"""

from agentic_cli.planning.plan_store import PlanStore

__all__ = [
    "PlanStore",
]
