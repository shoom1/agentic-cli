"""WebFetch tool components."""

from agentic_cli.tools.webfetch.validator import URLValidator, ValidationResult
from agentic_cli.tools.webfetch.robots import RobotsTxtChecker
from agentic_cli.tools.webfetch.converter import HTMLToMarkdown
from agentic_cli.tools.webfetch.fetcher import ContentFetcher, FetchResult, RedirectInfo
from agentic_cli.tools.webfetch.summarizer import (
    LLMSummarizer,
    FAST_MODEL_MAP,
    get_fast_model,
    build_summarize_prompt,
)

# Lazy imports for web_fetch to avoid circular imports
_lazy_imports = {
    "web_fetch": "agentic_cli.tools.webfetch_tool",
    "get_or_create_fetcher": "agentic_cli.tools.webfetch_tool",
}


def __getattr__(name: str):
    """Lazy import for web_fetch to avoid circular imports."""
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "URLValidator",
    "ValidationResult",
    "RobotsTxtChecker",
    "HTMLToMarkdown",
    "ContentFetcher",
    "FetchResult",
    "RedirectInfo",
    "LLMSummarizer",
    "FAST_MODEL_MAP",
    "get_fast_model",
    "build_summarize_prompt",
    "web_fetch",
    "get_or_create_fetcher",
]
