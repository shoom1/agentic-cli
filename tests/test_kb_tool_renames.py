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


def test_unified_search_dropped():
    import agentic_cli.tools as t
    assert not hasattr(t, "unified_search")


def test_open_document_dropped():
    import agentic_cli.tools as t
    assert not hasattr(t, "open_document")


def test_kb_write_concept_exported():
    from agentic_cli.tools import kb_write_concept  # noqa: F401


def test_kb_search_concepts_exported():
    from agentic_cli.tools import kb_search_concepts  # noqa: F401


def test_reader_bundle_includes_search_concepts():
    from agentic_cli.tools import (
        KB_READER_TOOLS,
        kb_search,
        kb_read,
        kb_list,
        kb_search_concepts,
    )
    assert KB_READER_TOOLS == [kb_search, kb_read, kb_list, kb_search_concepts]


def test_writer_bundle_includes_write_concept():
    from agentic_cli.tools import (
        KB_WRITER_TOOLS,
        KB_READER_TOOLS,
        kb_ingest,
        kb_write_concept,
    )
    assert KB_WRITER_TOOLS == [*KB_READER_TOOLS, kb_ingest, kb_write_concept]
