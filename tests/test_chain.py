"""
Unit tests for RAG chain.
"""
import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from rag.chain import RAGChain

class TestRAGChain:
    """Tests for RAGChain class."""

    @patch('rag.chain.PineconeRetriever')
    @patch('rag.chain.genai')
    def test_chain_initialization(self, mock_genai, mock_retriever):
        """Test RAG chain initialization."""
        chain = RAGChain(
            google_api_key="test_key",
            temperature=0.5
        )

        assert chain.temperature == 0.5
        assert chain.max_tokens == 1024

    @patch('rag.chain.PineconeRetriever')
    @patch('rag.chain.genai')
    def test_calculate_confidence_score_no_docs(self, mock_genai, mock_retriever):
        """Test confidence score with no documents."""
        chain = RAGChain(google_api_key="test_key")
        score = chain.calculate_confidence_score("Question?", "Answer", [])

        assert 0.0 <= score <= 1.0

    @patch('rag.chain.PineconeRetriever')
    @patch('rag.chain.genai')
    def test_calculate_confidence_score_with_docs(self, mock_genai, mock_retriever):
        """Test confidence score with documents."""
        chain = RAGChain(google_api_key="test_key")

        docs = [
            Document(page_content="Test content", metadata={'score': 0.9}),
            Document(page_content="More content", metadata={'score': 0.8})
        ]

        score = chain.calculate_confidence_score(
            "What is this?",
            "This is a test answer with good length [Source: test.txt]",
            docs
        )

        assert 0.0 <= score <= 1.0
        assert score > 0.5

    @patch('rag.chain.PineconeRetriever')
    @patch('rag.chain.genai')
    def test_calculate_confidence_score_uncertain_answer(self, mock_genai, mock_retriever):
        """Test confidence score with uncertain answer."""
        chain = RAGChain(google_api_key="test_key")

        docs = [Document(page_content="Test", metadata={'score': 0.9})]

        score = chain.calculate_confidence_score(
            "What is this?",
            "I don't have enough information to answer this question.",
            docs
        )

        assert 0.0 <= score < 0.5

    @patch('rag.chain.PineconeRetriever')
    @patch('rag.chain.genai')
    def test_batch_ask(self, mock_genai, mock_retriever):
        """Test batch question answering."""
        mock_retriever_instance = Mock()
        mock_retriever_instance.retrieve.return_value = []

        chain = RAGChain(
            retriever=mock_retriever_instance,
            google_api_key="test_key"
        )

        questions = ["Question 1?", "Question 2?"]
        responses = chain.batch_ask(questions)

        assert len(responses) == 2
        assert all('question' in r for r in responses)
        assert all('answer' in r for r in responses)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
