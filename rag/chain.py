"""
Complete RAG chain combining retrieval and generation with citations.
"""
import os
import re
from typing import Dict, List
from dotenv import load_dotenv
from google import genai
import yaml

from rag.retriever import PineconeRetriever
from rag.prompts import (
    create_rag_prompt,
    format_context_with_sources,
    extract_citations,
    RAG_SYSTEM_PROMPT
)

load_dotenv()

class RAGChain:
    """Complete RAG pipeline for question answering with citations."""

    def __init__(
        self,
        retriever: PineconeRetriever = None,
        google_api_key: str = None,
        generation_model: str = None,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ):
        """
        Initialize the RAG chain.

        Args:
            retriever: PineconeRetriever instance
            google_api_key: Google API key for Gemini
            generation_model: Gemini model name for generation
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        self.retriever = retriever or PineconeRetriever.from_config()

        self.google_api_key = google_api_key or os.getenv('GOOGLE_API_KEY')
        self.client = genai.Client(api_key=self.google_api_key)

        self.generation_model = generation_model or os.getenv('GENERATION_MODEL', 'gemini-2.5-flash')
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.rag_prompt = create_rag_prompt()

    @classmethod
    def from_config(cls, config_path: str = "config.yaml"):
        """
        Create RAG chain from configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            RAGChain instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        gen_config = config.get('generation', {})
        retriever = PineconeRetriever.from_config(config_path)

        return cls(
            retriever=retriever,
            generation_model=gen_config.get('model', 'gemini-2.5-flash'),
            temperature=gen_config.get('temperature', 0.3),
            max_tokens=gen_config.get('max_tokens', 1024)
        )

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using Gemini with provided context.

        Args:
            question: User question
            context: Retrieved context

        Returns:
            Generated answer text
        """
        prompt = self.rag_prompt.format(
            context=context,
            question=question
        )

        full_prompt = f"{RAG_SYSTEM_PROMPT}\n\n{prompt}"

        response = self.client.models.generate_content(
            model=self.generation_model,
            contents=full_prompt
        )

        return response.text

    def calculate_confidence_score(
        self,
        question: str,
        answer: str,
        retrieved_docs: List
    ) -> float:
        """
        Calculate confidence score for the answer.

        Args:
            question: Original question
            answer: Generated answer
            retrieved_docs: List of retrieved documents

        Returns:
            Confidence score between 0 and 1
        """
        score = 0.0

        if retrieved_docs:
            avg_retrieval_score = sum(
                doc.metadata.get('score', 0) for doc in retrieved_docs
            ) / len(retrieved_docs)
            score += avg_retrieval_score * 0.65

        answer_words = len(answer.split())
        if 10 <= answer_words <= 300:
            score += 0.25
        elif answer_words < 10:
            score += 0.15

        uncertainty_phrases = [
            "i don't have enough information",
            "not enough information",
            "cannot answer",
            "insufficient context",
            "don't know"
        ]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            score *= 0.5

        if len(retrieved_docs) >= 5:
            score += 0.20
        elif len(retrieved_docs) >= 3:
            score += 0.15

        return min(round(score, 3), 1.0)

    def ask(
        self,
        question: str,
        top_k: int = None,
        include_sources: bool = True
    ) -> Dict:
        """
        Answer a question using RAG pipeline.

        Args:
            question: User question
            top_k: Number of documents to retrieve (override default)
            include_sources: Whether to include source documents in response

        Returns:
            Dictionary containing answer, citations, confidence, and metadata
        """
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)

        if not retrieved_docs:
            return {
                "question": question,
                "answer": "I don't have enough information to answer this question based on the available knowledge base.",
                "citations": [],
                "confidence_score": 0.0,
                "retrieved_docs_count": 0,
                "sources": []
            }

        context = format_context_with_sources(retrieved_docs)

        answer = self.generate_answer(question, context)

        citations = extract_citations(retrieved_docs)

        confidence = self.calculate_confidence_score(question, answer, retrieved_docs)

        response = {
            "question": question,
            "answer": answer,
            "citations": citations,
            "confidence_score": confidence,
            "retrieved_docs_count": len(retrieved_docs),
            "avg_retrieval_score": round(
                sum(doc.metadata.get('score', 0) for doc in retrieved_docs) / len(retrieved_docs),
                3
            )
        }

        if include_sources:
            response["sources"] = [
                {
                    "filename": doc.metadata.get('filename'),
                    "chunk_id": doc.metadata.get('chunk_id'),
                    "score": doc.metadata.get('score'),
                    "content": doc.page_content
                }
                for doc in retrieved_docs
            ]

        return response

    def batch_ask(self, questions: List[str]) -> List[Dict]:
        """
        Answer multiple questions in batch.

        Args:
            questions: List of questions

        Returns:
            List of response dictionaries
        """
        responses = []
        for question in questions:
            response = self.ask(question, include_sources=False)
            responses.append(response)

        return responses

if __name__ == "__main__":
    rag_chain = RAGChain.from_config()

    test_questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the applications of deep learning?"
    ]

    print("RAG Chain Test\n" + "="*50)

    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)

        try:
            response = rag_chain.ask(question, include_sources=False)

            print(f"Answer: {response['answer']}\n")
            print(f"Confidence: {response['confidence_score']}")
            print(f"Retrieved Docs: {response['retrieved_docs_count']}")

            if response['citations']:
                print("\nCitations:")
                for citation in response['citations']:
                    print(f"  [{citation['id']}] {citation['source']} (Chunk {citation['chunk_id']})")
                    print(f"      Snippet: {citation['snippet'][:100]}...")

        except Exception as e:
            print(f"Error: {e}")
            print("Make sure to run ingestion first to populate the knowledge base!")

        print("\n" + "="*50)
