"""
Initialize Pinecone index for RAG system.
Run this script once to create the Pinecone index.
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import yaml

load_dotenv()

def init_pinecone_index():
    """Create Pinecone index if it doesn't exist."""

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

    index_name = os.getenv('PINECONE_INDEX_NAME', 'rag-knowledge-base')
    dimension = config['pinecone']['dimension']
    metric = config['pinecone']['metric']

    # Check if index exists
    existing_indexes = pc.list_indexes()
    index_names = [idx['name'] for idx in existing_indexes]

    if index_name in index_names:
        print(f"[OK] Index '{index_name}' already exists.")
    else:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"[OK] Index '{index_name}' created successfully!")

    # Get index stats
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"\nIndex Stats:")
    print(f"  - Total vectors: {stats['total_vector_count']}")
    print(f"  - Dimension: {stats['dimension']}")
    print(f"  - Index fullness: {stats.get('index_fullness', 0)}")

if __name__ == "__main__":
    init_pinecone_index()
