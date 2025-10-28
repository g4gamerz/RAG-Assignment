"""
Unit tests for retrieval module.
"""
import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from rag.retriever import PineconeRetriever

class TestPineconeRetriever:
    """Tests for PineconeRetriever class."""

    @patch('rag.retriever.genai')
    @patch('rag.retriever.Pinecone')
    def test_retriever_initialization(self, mock_pinecone, mock_genai):
        """Test retriever initialization."""
        retriever = PineconeRetriever(
            google_api_key="test_key",
            pinecone_api_key="test_key",
            top_k=5
        )

        assert retriever.top_k == 5
        assert retriever.score_threshold == 0.7

    @patch('rag.retriever.genai')
    @patch('rag.retriever.Pinecone')
    def test_retriever_custom_threshold(self, mock_pinecone, mock_genai):
        """Test retriever with custom score threshold."""
        retriever = PineconeRetriever(
            google_api_key="test_key",
            pinecone_api_key="test_key",
            score_threshold=0.8
        )

        assert retriever.score_threshold == 0.8

    def test_get_retrieval_stats_empty(self):
        """Test stats for empty results."""
        with patch('rag.retriever.genai'), patch('rag.retriever.Pinecone'):
            retriever = PineconeRetriever(
                google_api_key="test_key",
                pinecone_api_key="test_key"
            )
            stats = retriever.get_retrieval_stats([])

            assert stats['retrieved_count'] == 0
            assert stats['avg_score'] == 0.0

    def test_get_retrieval_stats_with_results(self):
        """Test stats calculation for results."""
        with patch('rag.retriever.genai'), patch('rag.retriever.Pinecone'):
            retriever = PineconeRetriever(
                google_api_key="test_key",
                pinecone_api_key="test_key"
            )

            results = [
                Document(page_content="Test 1", metadata={'score': 0.9, 'source': 'test1.txt'}),
                Document(page_content="Test 2", metadata={'score': 0.8, 'source': 'test2.txt'})
            ]

            stats = retriever.get_retrieval_stats(results)

            assert stats['retrieved_count'] == 2
            assert stats['avg_score'] == 0.85
            assert stats['min_score'] == 0.8
            assert stats['max_score'] == 0.9
            assert stats['unique_sources'] == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
