"""
FastAPI application for RAG system.
Provides endpoints for question answering with citations.
"""
import os
from typing import Dict
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

from api.schemas import (
    QuestionRequest,
    QuestionResponse,
    BatchQuestionRequest,
    BatchQuestionResponse,
    HealthResponse,
    ErrorResponse
)
from rag.chain import RAGChain

load_dotenv()

app = FastAPI(
    title="RAG Agent API",
    description="Retrieval-Augmented Generation API with citations and grounding",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    rag_chain = RAGChain.from_config()
    print("[OK] RAG chain initialized successfully")
except Exception as e:
    print(f"[X] Error initializing RAG chain: {e}")
    rag_chain = None

@app.get("/", tags=["Root"])
async def root() -> Dict:
    """Root endpoint with API information."""
    return {
        "message": "RAG Agent API",
        "version": "1.0.0",
        "endpoints": {
            "ask": "/rag/ask",
            "batch": "/rag/batch",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns status of the API and Pinecone connection.
    """
    try:
        if rag_chain is None:
            return HealthResponse(
                status="unhealthy",
                pinecone_connected=False,
                total_vectors=0,
                message="RAG chain not initialized"
            )

        stats = rag_chain.retriever.get_index_stats()

        return HealthResponse(
            status="healthy",
            pinecone_connected=True,
            total_vectors=stats.get('total_vectors', 0),
            message="All systems operational"
        )

    except Exception as e:
        return HealthResponse(
            status="degraded",
            pinecone_connected=False,
            total_vectors=0,
            message=f"Error: {str(e)}"
        )

@app.post(
    "/rag/ask",
    response_model=QuestionResponse,
    tags=["RAG"],
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    """
    Ask a question to the RAG system.

    Returns an answer with citations, confidence score, and optionally source documents.
    """
    try:
        if rag_chain is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="RAG chain not initialized. Please check server configuration."
            )

        response = rag_chain.ask(
            question=request.question,
            top_k=request.top_k,
            include_sources=request.include_sources
        )

        return QuestionResponse(**response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )

@app.post(
    "/rag/batch",
    response_model=BatchQuestionResponse,
    tags=["RAG"],
    status_code=status.HTTP_200_OK
)
async def ask_batch_questions(request: BatchQuestionRequest) -> BatchQuestionResponse:
    """
    Ask multiple questions in batch.

    Processes multiple questions and returns answers for all.
    """
    try:
        if rag_chain is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="RAG chain not initialized"
            )

        responses = rag_chain.batch_ask(request.questions)

        return BatchQuestionResponse(
            results=responses,
            total_questions=len(responses)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing batch questions: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

def start_server(host: str = None, port: int = None):
    """
    Start the FastAPI server.

    Args:
        host: Host address (default from env or 0.0.0.0)
        port: Port number (default from env or 8000)
    """
    host = host or os.getenv('API_HOST', '0.0.0.0')
    port = port or int(os.getenv('API_PORT', 8000))

    print(f"\n🚀 Starting RAG API server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Docs: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs\n")

    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()
