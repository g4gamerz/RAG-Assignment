"""
Prompt templates for RAG system with grounded generation and citations.
"""
from langchain_core.prompts import PromptTemplate

RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based ONLY on the provided context.

Your responsibilities:
1. Answer questions accurately using ONLY the information from the provided context
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question based on the provided context."
3. Do not make up or infer information that is not explicitly stated in the context
4. Be concise and clear in your responses
5. Provide ONLY the answer text without any citation markers, source references, or document IDs

Format your response:
- Provide a clear, direct answer to the question
- DO NOT include citations like [Source: filename] or [Document X] in your answer
- DO NOT mention source files or chunk IDs in your response
- Write the answer as a natural, flowing text without any reference markers
- The citations will be displayed separately, so focus only on answering the question clearly
"""

RAG_PROMPT_TEMPLATE = """Context information from the knowledge base:

{context}

---

Question: {question}

Instructions:
- Answer the question based ONLY on the context provided above
- DO NOT include any citations, source references, or document markers in your answer
- Provide a clean, natural answer without any [Source: ...] or [Document X] markers
- If the context is insufficient, state that clearly
- Be accurate and do not hallucinate information

Answer:"""

QUERY_REFINEMENT_PROMPT = """Given the following user question, generate a more effective search query to retrieve relevant information from a knowledge base.

Original Question: {question}

Generate an improved search query that:
- Captures the key concepts and intent
- Removes unnecessary words
- Is optimized for semantic search

Improved Query:"""

CONFIDENCE_PROMPT = """Based on the following answer and the context it was generated from, rate the confidence level of this answer on a scale of 0.0 to 1.0.

Context: {context}

Answer: {answer}

Consider:
- How well the context supports the answer
- Whether the answer is directly stated or inferred
- The specificity and completeness of the answer

Confidence Score (0.0-1.0):"""

def create_rag_prompt() -> PromptTemplate:
    """
    Create the main RAG prompt template.

    Returns:
        PromptTemplate configured for RAG
    """
    return PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

def create_query_refinement_prompt() -> PromptTemplate:
    """
    Create query refinement prompt template.

    Returns:
        PromptTemplate for query refinement
    """
    return PromptTemplate(
        template=QUERY_REFINEMENT_PROMPT,
        input_variables=["question"]
    )

def create_confidence_prompt() -> PromptTemplate:
    """
    Create confidence scoring prompt template.

    Returns:
        PromptTemplate for confidence scoring
    """
    return PromptTemplate(
        template=CONFIDENCE_PROMPT,
        input_variables=["context", "answer"]
    )

def format_context_with_sources(retrieved_docs: list) -> str:
    """
    Format retrieved documents with source information.

    Args:
        retrieved_docs: List of retrieved documents with metadata

    Returns:
        Formatted context string with source citations
    """
    if not retrieved_docs:
        return "No relevant context found."

    formatted_context = []

    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get('filename', 'Unknown')
        chunk_id = doc.metadata.get('chunk_id', 'N/A')

        context_block = f"""[Document {i}]
Source: {source}
Chunk ID: {chunk_id}
Content: {doc.page_content}
---"""
        formatted_context.append(context_block)

    return "\n\n".join(formatted_context)

def extract_citations(retrieved_docs: list) -> list:
    """
    Extract citation information from retrieved documents.

    Args:
        retrieved_docs: List of retrieved documents

    Returns:
        List of citation dictionaries
    """
    citations = []

    for i, doc in enumerate(retrieved_docs, 1):
        citation = {
            'id': i,
            'source': doc.metadata.get('filename', 'Unknown'),
            'chunk_id': doc.metadata.get('chunk_id', 'N/A'),
            'snippet': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        }
        citations.append(citation)

    return citations

if __name__ == "__main__":
    prompt = create_rag_prompt()
    print("RAG Prompt Template:")
    print(prompt.template)

    from langchain_core.documents import Document

    sample_docs = [
        Document(
            page_content="Python is a high-level programming language known for its simplicity.",
            metadata={"filename": "python_intro.pdf", "chunk_id": 0}
        ),
        Document(
            page_content="Python supports multiple programming paradigms including procedural and object-oriented.",
            metadata={"filename": "python_guide.pdf", "chunk_id": 5}
        )
    ]

    print("\n\nFormatted Context:")
    print(format_context_with_sources(sample_docs))

    print("\n\nExtracted Citations:")
    citations = extract_citations(sample_docs)
    for citation in citations:
        print(citation)
