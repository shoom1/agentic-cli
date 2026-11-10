"""What ADK would print over a terminal UI, logged at debug instead.

ADK writes notices and failure tracebacks to stderr, through Python's
warnings and stdlib logging, where the CLI's prompt is drawn. Some are noise
for the user (an experimental feature in use, advice the app cannot act on, a
cache it could not create), and the rest are told to the user otherwise (a
failed model call as the turn's error). ``install_log_filters`` logs each at
debug, matched narrowly, and leaves every other warning and record as it was:
never the whole ``google_adk`` logger tree.

``BaseCLIApp.run()`` installs them, once per process, so a library user who
never runs the CLI keeps Python's own handling. The filters attach to ADK's
loggers by name: importing this module imports no ADK, and installing them
loads none either (a Deep Agents app never does).
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Callable

from agentic_cli.logging import LOG_STREAM, Loggers

logger = Loggers.workflow()

# ADK 2 announces each experimental feature it uses with a UserWarning, in one
# of two forms: a feature flag ("[EXPERIMENTAL] feature
# FeatureName.JSON_SCHEMA_FOR_FUNC_DECL is enabled."), or a class or function
# marked experimental ("[EXPERIMENTAL] GeminiContextCacheManager: This feature
# is experimental and may change ...", from a Gemini agent's first turn while
# prompt caching is on).
_ADK_EXPERIMENTAL_NOTICE = re.compile(
    r"\[EXPERIMENTAL\] (?:feature (?P<feature>\S+) is enabled|(?P<name>\w+): )"
)

# ADK 2's runner warns, through stdlib logging, that an app whose agents can
# transfer to each other has no context_cache_config: a native ADK YAML app
# with sub-agents (agents built from configs delegate and never transfer). The
# ADK manager sets one while prompt caching is on and a model caches, so the
# advice comes only when the user turned caching off or no model caches: it
# has nothing left to tell them.
_ADK_RUNNER_LOGGER = "google_adk.google.adk.runners"
_ADK_UNCACHED_TRANSFER = "can transfer between agents but has no context_cache_config"


class _AdkUncachedTransferFilter(logging.Filter):
    """Turns ADK's context-cache advice into a debug log entry."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if _ADK_UNCACHED_TRANSFER not in message:
            return True
        logger.debug("adk_uncached_agent_transfer", message=message)
        return False


# While prompt caching is on, ADK's Gemini context cache warns through stdlib
# logging when it cannot create a cache or delete an old one (an expired cache
# is already gone on Google's side). The turn goes on uncached, so the warning
# has nothing for the user mid-conversation.
_ADK_CACHE_LOGGER = "google_adk.google.adk.models.gemini_context_cache_manager"
_ADK_CACHE_FAILURES = (
    ("Failed to create cache", "adk_cache_create_failed"),
    ("Failed to cleanup cache", "adk_cache_cleanup_failed"),
)


class _AdkCacheFailureFilter(logging.Filter):
    """Turns ADK's context-cache failure warnings into debug log entries."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for prefix, event in _ADK_CACHE_FAILURES:
            if message.startswith(prefix):
                logger.debug(event, message=message)
                return False
        return True


def _overflow_plugin_failed() -> str:
    """How ADK's plugin manager starts the error it wraps around one the
    overflow plugin raised (``overflow_plugin.OVERFLOW_PLUGIN_FAILED``).

    Read when a record comes, from the plugin's module: ADK runs by then
    (the record is its plugin manager's), and this module imports none of it.
    """
    from agentic_cli.workflow.adk.overflow_plugin import OVERFLOW_PLUGIN_FAILED

    return OVERFLOW_PLUGIN_FAILED


# ADK logs a failure's traceback through stdlib logging where the user is
# told otherwise. A summary that failed: ADK goes on with the whole history,
# and the summarizer itself warns the user once a turn (it never lets its own
# failure reach ADK, so this is a backstop). A model call that failed: the
# user reads its error as an ERROR event or as the turn's error. ADK logs one
# twice for the main agent, from the node and from the runner ("Root node %s
# failed.", matched exactly: "Root node %s was cancelled …" passes), and a
# failed overflow retry once more, from the plugin manager. ADK also warns on
# every request of a session holding a call without a result, which it drops:
# the next turn answers such a call (workflow/adk/interrupted_calls_plugin.py),
# so the warning has nothing for the user either. Each entry: the logger, what
# a record's message starts with or (``msg=``) its exact format, and the debug
# event it becomes.
_ADK_FAILURES = (
    ("google_adk.google.adk.apps.compaction", {"prefix": "Event compaction failed"},
     "adk_compaction_failed"),
    ("google_adk.google.adk.workflow._node_runner", {"prefix": "Node execution failed"},
     "adk_node_failed"),
    ("google_adk.google.adk.runners", {"msg": "Root node %s failed."}, "adk_root_node_failed"),
    ("google_adk.google.adk.plugins.plugin_manager",
     {"prefix": _overflow_plugin_failed}, "adk_overflow_plugin_failed"),
    ("google_adk.google.adk.flows.llm_flows.tools._rearranger",
     {"prefix": "Dropping function calls with no matching function response"},
     "adk_dropped_open_calls"),
)


class _AdkFailureFilter(logging.Filter):
    """Turns ADK's failure tracebacks on one logger into a debug log entry.

    A record is matched by the start of its message (``prefix``: the text, or
    a function that returns it) or by its format string itself (``msg``).
    """

    def __init__(
        self,
        event: str,
        *,
        prefix: str | Callable[[], str] | None = None,
        msg: str | None = None,
    ) -> None:
        super().__init__()
        self._event = event
        self._prefix = prefix
        self._msg = msg

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if self._msg is not None:
            matched = record.msg == self._msg
        else:
            prefix = self._prefix() if callable(self._prefix) else self._prefix
            matched = prefix is not None and message.startswith(prefix)
        if not matched:
            return True
        error = record.exc_info[1] if record.exc_info else None
        logger.debug(
            self._event,
            message=message,
            error_type=type(error).__name__ if error is not None else None,
            error=str(error) if error is not None else None,
        )
        return False


def install_log_filters() -> None:
    """Log ADK's notices instead of printing them.

    While the prompt runs, the log lines ``configure_logging`` sets up print
    above it (``cli/app.py::_log_lines_above_prompt``), and so do Python's
    warnings: this hook has Python's own hook write them to the same stream
    (another hook, such as logging's ``captureWarnings``, gets them as given).
    The hook logs ADK's experimental-feature notices at debug and passes
    every other warning to the hook it replaced; a filter on ADK's runner
    logger does the same for its context-cache advice, one on its
    context-cache logger for a cache it could not create or delete, one each
    on its compaction, node, runner and plugin-manager loggers for the
    traceback of a summary, a model call or an overflow retry that failed,
    and one on its rearranger for a call without a result it drops. Only the
    CLI installs them, as it runs (``BaseCLIApp.run``), so library users and
    the test suite keep Python's own handling. Installing twice changes
    nothing.
    """
    for name, kind in (
        (_ADK_RUNNER_LOGGER, _AdkUncachedTransferFilter),
        (_ADK_CACHE_LOGGER, _AdkCacheFailureFilter),
    ):
        adk_logger = logging.getLogger(name)
        if not any(isinstance(f, kind) for f in adk_logger.filters):
            adk_logger.addFilter(kind())
    for name, match, event in _ADK_FAILURES:
        adk_logger = logging.getLogger(name)
        if not any(isinstance(f, _AdkFailureFilter) for f in adk_logger.filters):
            adk_logger.addFilter(_AdkFailureFilter(event, **match))

    previous = warnings.showwarning
    if getattr(previous, "_logs_adk_notices", False):
        return

    def showwarning(message, category, filename, lineno, file=None, line=None):
        notice = _ADK_EXPERIMENTAL_NOTICE.match(str(message))
        if notice and issubclass(category, UserWarning):
            feature = (notice["feature"] or notice["name"]).removeprefix("FeatureName.")
            logger.debug("adk_experimental_feature", feature=feature)
            return
        if file is None and previous is warnings._showwarning_orig:
            file = LOG_STREAM
        previous(message, category, filename, lineno, file, line)

    showwarning._logs_adk_notices = True
    warnings.showwarning = showwarning
