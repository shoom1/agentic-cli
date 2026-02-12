"""Model and path constants for configuration.

Provides model lists and default values used by BaseSettings.
"""

# Available models by provider
GOOGLE_MODELS = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

ANTHROPIC_MODELS = [
    "claude-opus-4-5",
    "claude-opus-4",
    "claude-sonnet-4-5",
    "claude-sonnet-4",
]

ALL_MODELS = GOOGLE_MODELS + ANTHROPIC_MODELS

# Thinking effort levels (for models that support it)
THINKING_EFFORT_LEVELS = ["none", "low", "medium", "high"]

# Default models per provider
DEFAULT_GOOGLE_MODEL = "gemini-3-flash-preview"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"
