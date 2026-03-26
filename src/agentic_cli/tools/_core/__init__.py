"""Shared core logic for tools — pure functions, no framework imports."""

# Tool function names that need backend-specific replacement
STATE_TOOL_NAMES = frozenset({"save_plan", "get_plan", "save_tasks", "get_tasks"})

# Tool function names that need service binding via factories
SERVICE_TOOL_NAMES = frozenset({
    "save_memory", "search_memory",
    "search_knowledge_base", "ingest_document",
    "read_document", "list_documents", "open_document",
    "web_fetch",
    "sandbox_execute",
    "ask_clarification",
})
