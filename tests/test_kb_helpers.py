"""Tests for knowledge tools helpers."""

from unittest.mock import MagicMock, patch


class TestFindDocumentInKBs:
    def test_found_in_main_kb(self):
        from agentic_cli.tools.knowledge_tools import _find_document_in_kbs

        doc = MagicMock()
        main_kb = MagicMock()
        main_kb.find_document.return_value = doc

        def mock_get_service(key):
            if key == "kb_manager":
                return main_kb
            if key == "user_kb_manager":
                return None
            return None

        with patch("agentic_cli.tools.knowledge_tools.get_service", side_effect=mock_get_service):
            result_doc, result_kb = _find_document_in_kbs("test-id")

        assert result_doc is doc
        assert result_kb is main_kb

    def test_fallback_to_user_kb(self):
        from agentic_cli.tools.knowledge_tools import _find_document_in_kbs

        doc = MagicMock()
        main_kb = MagicMock()
        main_kb.find_document.return_value = None
        user_kb = MagicMock()
        user_kb.find_document.return_value = doc

        def mock_get_service(key):
            if key == "kb_manager":
                return main_kb
            if key == "user_kb_manager":
                return user_kb
            return None

        with patch("agentic_cli.tools.knowledge_tools.get_service", side_effect=mock_get_service):
            result_doc, result_kb = _find_document_in_kbs("test-id")

        assert result_doc is doc
        assert result_kb is user_kb

    def test_not_found_in_either(self):
        from agentic_cli.tools.knowledge_tools import _find_document_in_kbs

        main_kb = MagicMock()
        main_kb.find_document.return_value = None
        user_kb = MagicMock()
        user_kb.find_document.return_value = None

        def mock_get_service(key):
            if key == "kb_manager":
                return main_kb
            if key == "user_kb_manager":
                return user_kb
            return None

        with patch("agentic_cli.tools.knowledge_tools.get_service", side_effect=mock_get_service):
            result_doc, result_kb = _find_document_in_kbs("test-id")

        assert result_doc is None
        assert result_kb is main_kb

    def test_user_kb_same_as_main_skips_fallback(self):
        from agentic_cli.tools.knowledge_tools import _find_document_in_kbs

        main_kb = MagicMock()
        main_kb.find_document.return_value = None

        def mock_get_service(key):
            return main_kb  # same instance for both

        with patch("agentic_cli.tools.knowledge_tools.get_service", side_effect=mock_get_service):
            result_doc, result_kb = _find_document_in_kbs("test-id")

        assert result_doc is None
        assert main_kb.find_document.call_count == 1
