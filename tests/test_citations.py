"""
Unit tests for citation extraction and formatting.
"""
import pytest
from langchain_core.documents import Document

from rag.prompts import (
    format_context_with_sources,
    extract_citations,
    create_rag_prompt
)

class TestCitations:
    """Tests for citation functionality."""

    def test_format_context_empty(self):
        """Test formatting with no documents."""
        context = format_context_with_sources([])
        assert context == "No relevant context found."

    def test_format_context_with_documents(self):
        """Test formatting with documents."""
        docs = [
            Document(
                page_content="Test content 1",
                metadata={'filename': 'test1.txt', 'chunk_id': 0}
            ),
            Document(
                page_content="Test content 2",
                metadata={'filename': 'test2.txt', 'chunk_id': 1}
            )
        ]

        context = format_context_with_sources(docs)

        assert 'Document 1' in context
        assert 'Document 2' in context
        assert 'test1.txt' in context
        assert 'test2.txt' in context
        assert 'Test content 1' in context
        assert 'Test content 2' in context

    def test_extract_citations_empty(self):
        """Test citation extraction with no documents."""
        citations = extract_citations([])
        assert citations == []

    def test_extract_citations_with_documents(self):
        """Test citation extraction with documents."""
        docs = [
            Document(
                page_content="Test content",
                metadata={'filename': 'test.txt', 'chunk_id': 0}
            )
        ]

        citations = extract_citations(docs)

        assert len(citations) == 1
        assert citations[0]['id'] == 1
        assert citations[0]['source'] == 'test.txt'
        assert citations[0]['chunk_id'] == 0
        assert 'snippet' in citations[0]

    def test_extract_citations_snippet_truncation(self):
        """Test that long content is truncated in snippets."""
        docs = [
            Document(
                page_content="A" * 300,
                metadata={'filename': 'test.txt', 'chunk_id': 0}
            )
        ]

        citations = extract_citations(docs)
        snippet = citations[0]['snippet']

        assert len(snippet) <= 203
        assert snippet.endswith('...')

    def test_create_rag_prompt(self):
        """Test RAG prompt template creation."""
        prompt = create_rag_prompt()

        assert prompt is not None
        assert 'context' in prompt.input_variables
        assert 'question' in prompt.input_variables

    def test_rag_prompt_formatting(self):
        """Test RAG prompt formatting with values."""
        prompt = create_rag_prompt()

        formatted = prompt.format(
            context="Test context",
            question="What is this?"
        )

        assert 'Test context' in formatted
        assert 'What is this?' in formatted

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
