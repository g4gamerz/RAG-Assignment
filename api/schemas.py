"""
Pydantic schemas for API request and response validation.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    """Request schema for asking a question."""

    question: str = Field(
        ...,
        description="The question to ask the RAG system",
        min_length=3,
        max_length=500,
        example="What is artificial intelligence?"
    )
    top_k: Optional[int] = Field(
        default=5,
        description="Number of documents to retrieve",
        ge=1,
        le=20
    )
    include_sources: Optional[bool] = Field(
        default=False,
        description="Whether to include full source documents in response"
    )

class Citation(BaseModel):
    """Citation information for a source."""

    id: int = Field(..., description="Citation ID")
    source: str = Field(..., description="Source filename")
    chunk_id: Any = Field(..., description="Chunk identifier")
    snippet: str = Field(..., description="Text snippet from source")

class SourceDocument(BaseModel):
    """Full source document information."""

    filename: str = Field(..., description="Source filename")
    chunk_id: Any = Field(..., description="Chunk identifier")
    score: float = Field(..., description="Relevance score")
    content: str = Field(..., description="Full content of the chunk")

class QuestionResponse(BaseModel):
    """Response schema for question answering."""

    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(..., description="List of source citations")
    confidence_score: float = Field(
        ...,
        description="Confidence score (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    retrieved_docs_count: int = Field(..., description="Number of documents retrieved")
    avg_retrieval_score: Optional[float] = Field(
        default=None,
        description="Average retrieval score"
    )
    sources: Optional[List[SourceDocument]] = Field(
        default=None,
        description="Full source documents (if requested)"
    )

class BatchQuestionRequest(BaseModel):
    """Request schema for batch question answering."""

    questions: List[str] = Field(
        ...,
        description="List of questions to ask",
        min_length=1,
        max_length=10
    )

class BatchQuestionResponse(BaseModel):
    """Response schema for batch question answering."""

    results: List[QuestionResponse] = Field(..., description="List of answers")
    total_questions: int = Field(..., description="Total number of questions processed")

class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="API status")
    pinecone_connected: bool = Field(..., description="Pinecone connection status")
    total_vectors: int = Field(..., description="Total vectors in index")
    message: Optional[str] = Field(default=None, description="Additional message")

class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
