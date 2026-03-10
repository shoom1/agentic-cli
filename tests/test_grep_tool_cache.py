"""Tests for grep tool ripgrep caching."""

from unittest.mock import patch
from agentic_cli.tools.grep_tool import _ripgrep_available


class TestRipgrepCache:
    def test_ripgrep_check_cached_across_calls(self):
        """Second call should not spawn subprocess."""
        _ripgrep_available.cache_clear()
        with patch("agentic_cli.tools.grep_tool.subprocess.run") as mock_run:
            mock_run.return_value = None
            result1 = _ripgrep_available()
            result2 = _ripgrep_available()
            assert result1 is True
            assert result2 is True
            assert mock_run.call_count == 1
        _ripgrep_available.cache_clear()

    def test_ripgrep_cache_clear_allows_recheck(self):
        """Cache can be cleared to re-check."""
        _ripgrep_available.cache_clear()
        with patch("agentic_cli.tools.grep_tool.subprocess.run") as mock_run:
            mock_run.return_value = None
            _ripgrep_available()
            _ripgrep_available.cache_clear()
            _ripgrep_available()
            assert mock_run.call_count == 2
        _ripgrep_available.cache_clear()
