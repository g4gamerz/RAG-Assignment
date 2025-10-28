"""
Convenience script to run document ingestion pipeline.
Place documents in the 'data/' folder before running.
"""
import sys
from pathlib import Path

from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embed_and_upsert import EmbeddingUpserter

def main():
    """Run the complete ingestion pipeline."""

    print("="*70)
    print("RAG SYSTEM - DOCUMENT INGESTION")
    print("="*70)

    data_dir = Path("data")
    if not data_dir.exists():
        print("\n[X] Error: 'data/' directory not found")
        print("Creating 'data/' directory...")
        data_dir.mkdir()
        print("[OK] Directory created")
        print("\nPlease add documents to the 'data/' folder and run again.")
        return 1

    files = list(data_dir.rglob('*'))
    doc_files = [f for f in files if f.is_file() and f.suffix.lower() in ['.pdf', '.html', '.md', '.txt']]

    if not doc_files:
        print("\n[X] Error: No documents found in 'data/' directory")
        print("Supported formats: PDF, HTML, Markdown (.md), Text (.txt)")
        print("\nPlease add documents and run again.")
        return 1

    print(f"\n[OK] Found {len(doc_files)} document(s) to process")

    try:
        print("\n" + "-"*70)
        print("STEP 1: Loading Documents")
        print("-"*70)

        loader = DocumentLoader()
        documents = loader.load_directory("data/")

        if not documents:
            print("[X] No documents were successfully loaded")
            return 1

        stats = loader.get_document_stats(documents)
        print("\nDocument Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\n" + "-"*70)
        print("STEP 2: Chunking Documents")
        print("-"*70)

        chunker = DocumentChunker.from_config()
        chunks = chunker.chunk_documents(documents)

        chunk_stats = chunker.get_chunk_stats(chunks)
        print("\nChunk Statistics:")
        for key, value in chunk_stats.items():
            print(f"  {key}: {value}")

        print("\n" + "-"*70)
        print("STEP 3: Generating Embeddings & Upserting to Pinecone")
        print("-"*70)

        upserter = EmbeddingUpserter.from_config()
        result = upserter.upsert_documents(chunks, show_progress=True)

        print("\n" + "-"*70)
        print("STEP 4: Verification")
        print("-"*70)

        index_stats = upserter.get_index_stats()
        print("\nPinecone Index Statistics:")
        for key, value in index_stats.items():
            print(f"  {key}: {value}")

        print("\n" + "="*70)
        print("[OK] INGESTION COMPLETE!")
        print("="*70)
        print("\nYour knowledge base is ready for queries.")
        print("\nNext steps:")
        print("  - Start FastAPI: python -m api.app")
        print("  - Start Streamlit: streamlit run streamlit_app.py")
        print("  - Run evaluation: python eval/eval_ragas.py")

        return 0

    except Exception as e:
        print(f"\n[X] Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
