"""Tests for rate-limit detection helpers (workflow/retry.py)."""

from __future__ import annotations

from agentic_cli.workflow.retry import is_rate_limit_error


class _Err(Exception):
    def __init__(self, msg: str = "", *, status_code=None, code=None) -> None:
        super().__init__(msg)
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


def test_gemini_code_429_detected():
    assert is_rate_limit_error(_Err(code=429)) is True


def test_resource_exhausted_string_detected():
    assert is_rate_limit_error(_Err("RESOURCE_EXHAUSTED: quota")) is True


def test_anthropic_status_code_429_detected():
    """anthropic.RateLimitError carries .status_code, not .code."""
    assert is_rate_limit_error(_Err(status_code=429)) is True


def test_non_rate_limit_error_not_detected():
    assert is_rate_limit_error(_Err("boom", status_code=500)) is False
    assert is_rate_limit_error(_Err("plain error")) is False
