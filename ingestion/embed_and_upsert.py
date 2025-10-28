"""
Generate embeddings using Pinecone hosted models and upsert to Pinecone.
"""
import os
from typing import List, Dict
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_core.documents import Document
import yaml

load_dotenv()

class EmbeddingUpserter:
    """Generate embeddings and upsert to Pinecone vector database."""

    def __init__(
        self,
        pinecone_api_key: str = None,
        index_name: str = None,
        embedding_model: str = None,
        batch_size: int = 100
    ):
        """
        Initialize the embedding upserter.

        Args:
            pinecone_api_key: Pinecone API key
            index_name: Name of Pinecone index
            embedding_model: Pinecone hosted embedding model name
            batch_size: Number of documents to process in each batch
        """
        self.pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        self.index_name = index_name or os.getenv('PINECONE_INDEX_NAME', 'rag-knowledge-base')
        self.index = self.pc.Index(self.index_name)

        self.embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 'llama-text-embed-v2')
        self.batch_size = batch_size

    @classmethod
    def from_config(cls, config_path: str = "config.yaml"):
        """
        Create upserter from configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            EmbeddingUpserter instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return cls(
            embedding_model=config['embeddings']['model'],
            batch_size=100
        )

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using Pinecone hosted model.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        result = self.pc.inference.embed(
            model=self.embedding_model,
            inputs=[text],
            parameters={"input_type": "passage", "dimension": 768}
        )
        return result[0].values

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        result = self.pc.inference.embed(
            model=self.embedding_model,
            inputs=texts,
            parameters={"input_type": "passage", "dimension": 768}
        )
        return [embedding.values for embedding in result]

    def prepare_vectors(self, documents: List[Document]) -> List[Dict]:
        """
        Prepare documents as vectors for Pinecone upsert.

        Args:
            documents: List of Document objects with content and metadata

        Returns:
            List of vector dictionaries for Pinecone
        """
        vectors = []

        for i, doc in enumerate(documents):
            embedding = self.generate_embedding(doc.page_content)

            metadata = {
                'text': doc.page_content[:1000],
                'source': doc.metadata.get('source', 'unknown'),
                'filename': doc.metadata.get('filename', 'unknown'),
                'chunk_id': doc.metadata.get('chunk_id', i),
                'chunk_size': doc.metadata.get('chunk_size', len(doc.page_content))
            }

            vector = {
                'id': f"doc_{i}_{int(time.time())}",
                'values': embedding,
                'metadata': metadata
            }
            vectors.append(vector)

        return vectors

    def upsert_documents(self, documents: List[Document], show_progress: bool = True) -> Dict:
        """
        Upsert documents to Pinecone in batches.

        Args:
            documents: List of Document objects to upsert
            show_progress: Whether to show progress during upsert

        Returns:
            Dictionary with upsert statistics
        """
        if not documents:
            return {"error": "No documents to upsert"}

        total_docs = len(documents)
        upserted_count = 0
        start_time = time.time()

        if show_progress:
            print(f"Starting upsert of {total_docs} documents...")

        for i in range(0, total_docs, self.batch_size):
            batch = documents[i:i + self.batch_size]
            vectors = self.prepare_vectors(batch)

            self.index.upsert(vectors=vectors)
            upserted_count += len(vectors)

            if show_progress:
                progress = (upserted_count / total_docs) * 100
                print(f"Progress: {upserted_count}/{total_docs} ({progress:.1f}%)")

        elapsed_time = time.time() - start_time

        stats = {
            "total_documents": total_docs,
            "upserted_count": upserted_count,
            "elapsed_time_seconds": round(elapsed_time, 2),
            "docs_per_second": round(upserted_count / elapsed_time, 2)
        }

        if show_progress:
            print(f"\n[OK] Upsert completed!")
            print(f"  - Total documents: {stats['total_documents']}")
            print(f"  - Time elapsed: {stats['elapsed_time_seconds']}s")
            print(f"  - Speed: {stats['docs_per_second']} docs/sec")

        return stats

    def get_index_stats(self) -> Dict:
        """
        Get statistics about the Pinecone index.

        Returns:
            Dictionary with index statistics
        """
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.get('total_vector_count', 0),
            "dimension": stats.get('dimension', 0),
            "index_fullness": stats.get('index_fullness', 0)
        }

    def delete_all_vectors(self, confirm: bool = False):
        """
        Delete all vectors from the index (use with caution).

        Args:
            confirm: Must be True to proceed with deletion
        """
        if not confirm:
            print("⚠ Set confirm=True to delete all vectors")
            return

        self.index.delete(delete_all=True)
        print("[OK] All vectors deleted from index")

if __name__ == "__main__":
    upserter = EmbeddingUpserter.from_config()

    stats = upserter.get_index_stats()
    print("Current Index Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    sample_docs = [
        Document(
            page_content="Artificial Intelligence is transforming technology.",
            metadata={"source": "example.txt", "filename": "example.txt", "chunk_id": 0}
        ),
        Document(
            page_content="Machine Learning enables computers to learn from data.",
            metadata={"source": "example.txt", "filename": "example.txt", "chunk_id": 1}
        )
    ]

    print("\nUpserting sample documents...")
    result = upserter.upsert_documents(sample_docs)
