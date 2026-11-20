"""The base manager's ``tell_root``: a backend without a notice hook cannot tell its root."""

from __future__ import annotations

import pytest

from agentic_cli.workflow.base_manager import BaseWorkflowManager


async def test_a_backend_without_a_notice_hook_raises_as_iteration_starts():
    class Bare:
        _deliver_notice = BaseWorkflowManager._deliver_notice

    with pytest.raises(NotImplementedError, match="cannot tell its root agent a notice"):
        async for _ in BaseWorkflowManager.tell_root(Bare(), "A notice."):  # type: ignore[arg-type]
            pass
