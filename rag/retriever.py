"""
Retriever for fetching relevant context from Pinecone vector database.
"""
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_core.documents import Document
import yaml

load_dotenv()

class PineconeRetriever:
    """Retrieve relevant documents from Pinecone vector database."""

    def __init__(
        self,
        pinecone_api_key: str = None,
        index_name: str = None,
        embedding_model: str = None,
        top_k: int = 5,
        score_threshold: float = 0.7
    ):
        """
        Initialize the retriever.

        Args:
            pinecone_api_key: Pinecone API key
            index_name: Name of Pinecone index
            embedding_model: Pinecone hosted embedding model name
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
        """
        self.pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        self.index_name = index_name or os.getenv('PINECONE_INDEX_NAME', 'rag-knowledge-base')
        self.index = self.pc.Index(self.index_name)

        self.embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 'llama-text-embed-v2')
        self.top_k = top_k
        self.score_threshold = score_threshold

    @classmethod
    def from_config(cls, config_path: str = "config.yaml"):
        """
        Create retriever from configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            PineconeRetriever instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        retrieval_config = config.get('retrieval', {})
        embedding_config = config.get('embeddings', {})

        return cls(
            embedding_model=embedding_config.get('model'),
            top_k=retrieval_config.get('top_k', 5),
            score_threshold=retrieval_config.get('score_threshold', 0.7)
        )

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for search query.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        result = self.pc.inference.embed(
            model=self.embedding_model,
            inputs=[query],
            parameters={"input_type": "query", "dimension": 768}
        )
        return result[0].values

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query text
            top_k: Override default top_k value
            filter_dict: Optional metadata filter for Pinecone

        Returns:
            List of relevant Document objects with scores
        """
        query_embedding = self.generate_query_embedding(query)

        k = top_k or self.top_k
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            filter=filter_dict
        )

        documents = []
        for match in results.get('matches', []):
            if match['score'] < self.score_threshold:
                continue

            metadata = match.get('metadata', {})
            metadata['score'] = match['score']

            doc = Document(
                page_content=metadata.get('text', ''),
                metadata=metadata
            )
            documents.append(doc)

        return documents

    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve documents with detailed score information.

        Args:
            query: Search query text
            top_k: Override default top_k value

        Returns:
            List of dictionaries with document and score info
        """
        documents = self.retrieve(query, top_k)

        results = []
        for doc in documents:
            result = {
                'document': doc,
                'score': doc.metadata.get('score', 0.0),
                'source': doc.metadata.get('source', 'unknown'),
                'filename': doc.metadata.get('filename', 'unknown'),
                'chunk_id': doc.metadata.get('chunk_id', 'N/A'),
                'content_preview': doc.page_content[:200] + "..."
            }
            results.append(result)

        return results

    def get_retrieval_stats(self, results: List[Document]) -> Dict:
        """
        Get statistics about retrieval results.

        Args:
            results: List of retrieved documents

        Returns:
            Dictionary with retrieval statistics
        """
        if not results:
            return {
                "retrieved_count": 0,
                "avg_score": 0.0
            }

        scores = [doc.metadata.get('score', 0.0) for doc in results]
        sources = set(doc.metadata.get('source', 'unknown') for doc in results)

        return {
            "retrieved_count": len(results),
            "avg_score": round(sum(scores) / len(scores), 3),
            "min_score": round(min(scores), 3),
            "max_score": round(max(scores), 3),
            "unique_sources": len(sources)
        }

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

if __name__ == "__main__":
    retriever = PineconeRetriever.from_config()

    test_query = "What is artificial intelligence?"
    print(f"Query: {test_query}\n")

    try:
        results = retriever.retrieve_with_scores(test_query)

        if results:
            print(f"Retrieved {len(results)} documents:\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. Score: {result['score']:.3f}")
                print(f"   Source: {result['filename']}")
                print(f"   Preview: {result['content_preview'][:100]}...")
                print()

            docs = [r['document'] for r in results]
            stats = retriever.get_retrieval_stats(docs)
            print("Retrieval Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print("No relevant documents found. Make sure to run ingestion first!")

    except Exception as e:
        print(f"Error during retrieval: {e}")
        print("Make sure Pinecone index is initialized and populated.")
