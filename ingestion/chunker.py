"""
Text chunking with configurable size and overlap.
"""
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import yaml

class DocumentChunker:
    """Split documents into chunks for embedding."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        """
        Initialize the document chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            separators: List of separators to use for splitting (hierarchical)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    @classmethod
    def from_config(cls, config_path: str = "config.yaml"):
        """
        Create chunker from configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            DocumentChunker instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        doc_config = config.get('document_processing', {})
        return cls(
            chunk_size=doc_config.get('chunk_size', 1000),
            chunk_overlap=doc_config.get('chunk_overlap', 200),
            separators=doc_config.get('separators')
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of Document objects to chunk

        Returns:
            List of chunked Document objects with preserved metadata
        """
        chunks = self.text_splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        return chunks

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Document]:
        """
        Chunk a single text string.

        Args:
            text: Text string to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunked Document objects
        """
        metadata = metadata or {}
        chunks = self.text_splitter.split_text(text)

        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy()
            doc_metadata['chunk_id'] = i
            doc_metadata['chunk_size'] = len(chunk)

            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))

        return documents

    def get_chunk_stats(self, chunks: List[Document]) -> Dict:
        """
        Get statistics about chunks.

        Args:
            chunks: List of chunked Document objects

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {"total_chunks": 0}

        chunk_sizes = [len(chunk.page_content) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) // len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_characters": sum(chunk_sizes)
        }

if __name__ == "__main__":
    chunker = DocumentChunker.from_config()

    sample_text = """
    Artificial Intelligence (AI) is transforming the world.
    It enables machines to learn from experience and perform human-like tasks.

    Machine Learning is a subset of AI that focuses on algorithms that improve through experience.
    Deep Learning uses neural networks with multiple layers to process complex patterns.

    Natural Language Processing allows computers to understand human language.
    """ * 10  # Repeat to create larger text

    chunks = chunker.chunk_text(sample_text, metadata={"source": "example"})
    stats = chunker.get_chunk_stats(chunks)

    print("Chunk Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nFirst chunk preview:")
    print(chunks[0].page_content[:200] + "...")
