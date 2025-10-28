"""
Unit tests for ingestion module.
"""
import pytest
from pathlib import Path
from langchain_core.documents import Document

from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker

class TestDocumentLoader:
    """Tests for DocumentLoader class."""

    def test_loader_initialization(self):
        """Test loader initialization with default formats."""
        loader = DocumentLoader()
        assert 'pdf' in loader.supported_formats
        assert 'html' in loader.supported_formats
        assert 'md' in loader.supported_formats
        assert 'txt' in loader.supported_formats

    def test_loader_custom_formats(self):
        """Test loader with custom supported formats."""
        loader = DocumentLoader(supported_formats=['txt', 'md'])
        assert loader.supported_formats == ['txt', 'md']
        assert 'pdf' not in loader.supported_formats

    def test_unsupported_format_error(self):
        """Test error handling for unsupported file formats."""
        loader = DocumentLoader()
        with pytest.raises((ValueError, FileNotFoundError)):
            loader.load_document("test.docx")

    def test_file_not_found_error(self):
        """Test error handling for non-existent files."""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_document("nonexistent_file.pdf")

    def test_get_document_stats_empty(self):
        """Test stats for empty document list."""
        loader = DocumentLoader()
        stats = loader.get_document_stats([])
        assert stats['total'] == 0

class TestDocumentChunker:
    """Tests for DocumentChunker class."""

    def test_chunker_initialization(self):
        """Test chunker initialization with default parameters."""
        chunker = DocumentChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_chunker_custom_parameters(self):
        """Test chunker with custom parameters."""
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100

    def test_chunk_text(self):
        """Test text chunking."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a test. " * 20
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
        assert all('chunk_id' in chunk.metadata for chunk in chunks)

    def test_chunk_documents(self):
        """Test document chunking."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        docs = [
            Document(page_content="Test content " * 20, metadata={"source": "test.txt"})
        ]
        chunks = chunker.chunk_documents(docs)

        assert len(chunks) > 0
        assert all('chunk_id' in chunk.metadata for chunk in chunks)
        assert all('source' in chunk.metadata for chunk in chunks)

    def test_get_chunk_stats(self):
        """Test chunk statistics calculation."""
        chunker = DocumentChunker()
        chunks = [
            Document(page_content="Test " * 10, metadata={}),
            Document(page_content="Test " * 20, metadata={})
        ]
        stats = chunker.get_chunk_stats(chunks)

        assert stats['total_chunks'] == 2
        assert stats['avg_chunk_size'] > 0
        assert stats['min_chunk_size'] > 0
        assert stats['max_chunk_size'] > 0

    def test_get_chunk_stats_empty(self):
        """Test stats for empty chunk list."""
        chunker = DocumentChunker()
        stats = chunker.get_chunk_stats([])
        assert stats['total_chunks'] == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
