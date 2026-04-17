"""Verify the kb_* tool name surface."""


def test_kb_search_exported():
    from agentic_cli.tools import kb_search  # noqa: F401


def test_kb_list_exported():
    from agentic_cli.tools import kb_list  # noqa: F401


def test_kb_ingest_exported():
    from agentic_cli.tools import kb_ingest  # noqa: F401


def test_old_search_name_gone():
    import agentic_cli.tools as t
    assert not hasattr(t, "search_knowledge_base")


def test_old_list_name_gone():
    import agentic_cli.tools as t
    assert not hasattr(t, "list_documents")


def test_old_ingest_name_gone():
    import agentic_cli.tools as t
    assert not hasattr(t, "ingest_document")


def test_kb_read_exported():
    from agentic_cli.tools import kb_read  # noqa: F401


def test_old_read_name_gone():
    import agentic_cli.tools as t
    assert not hasattr(t, "read_document")
