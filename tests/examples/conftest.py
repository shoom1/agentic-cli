"""Shared fixtures for tests/examples.

Sets ADK feature flags required for the examples test suite.
"""

import os

# ADK >= 1.36 requires SNAKE_CASE_SKILL_NAME feature flag to accept
# underscore-style skill names (e.g. report_writer).  Set before any test
# imports google.adk so the flag is seen by every call to is_feature_enabled.
os.environ.setdefault("ADK_ENABLE_SNAKE_CASE_SKILL_NAME", "1")
