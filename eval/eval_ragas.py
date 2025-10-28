import json
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent))
from rag.chain import RAGChain
from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embed_and_upsert import EmbeddingUpserter

def load_qa_gold(file_path: str = "eval/qa_gold.jsonl") -> List[Dict]:
    qa_pairs = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            qa_pairs.append(json.loads(line.strip()))

    return qa_pairs

def evaluate_rag_system(
    rag_chain: RAGChain,
    qa_pairs: List[Dict],
    verbose: bool = True
) -> Dict:
    results = []
    total_questions = len(qa_pairs)

    if verbose:
        print(f"Evaluating RAG system on {total_questions} questions...")
        print("=" * 70)

    for i, qa in enumerate(qa_pairs, 1):
        question = qa['question']
        ground_truth = qa.get('ground_truth', '')

        if verbose:
            print(f"\n[{i}/{total_questions}] Question: {question}")

        try:
            response = rag_chain.ask(question, include_sources=False)

            result = {
                'question': question,
                'ground_truth': ground_truth,
                'answer': response['answer'],
                'confidence_score': response['confidence_score'],
                'retrieved_docs_count': response['retrieved_docs_count'],
                'avg_retrieval_score': response.get('avg_retrieval_score', 0),
                'citations_count': len(response['citations'])
            }

            results.append(result)

            if verbose:
                print(f"Answer: {response['answer'][:150]}...")
                print(f"Confidence: {response['confidence_score']:.3f}")
                print(f"Retrieved: {response['retrieved_docs_count']} docs")
                print(f"Citations: {result['citations_count']}")

        except Exception as e:
            if verbose:
                print(f"Error: {e}")

            results.append({
                'question': question,
                'ground_truth': ground_truth,
                'answer': None,
                'error': str(e)
            })

    metrics = calculate_metrics(results)
    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print_metrics(metrics)

    return {
        'metrics': metrics,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }

def calculate_metrics(results: List[Dict]) -> Dict:
    successful_results = [r for r in results if r.get('answer')]
    failed_results = [r for r in results if not r.get('answer')]

    if not results:
        return {'error': 'No results to evaluate'}
    success_rate = len(successful_results) / len(results)
    avg_confidence = (
        sum(r['confidence_score'] for r in successful_results) / len(successful_results)
        if successful_results else 0
    )
    avg_retrieval = (
        sum(r['avg_retrieval_score'] for r in successful_results) / len(successful_results)
        if successful_results else 0
    )
    avg_retrieved_docs = (
        sum(r['retrieved_docs_count'] for r in successful_results) / len(successful_results)
        if successful_results else 0
    )
    avg_citations = (
        sum(r['citations_count'] for r in successful_results) / len(successful_results)
        if successful_results else 0
    )
    answer_lengths = [len(r['answer']) for r in successful_results if r.get('answer')]
    avg_answer_length = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
    high_confidence_count = sum(1 for r in successful_results if r['confidence_score'] >= 0.7)
    high_confidence_rate = high_confidence_count / len(successful_results) if successful_results else 0

    return {
        'total_questions': len(results),
        'successful_answers': len(successful_results),
        'failed_answers': len(failed_results),
        'success_rate': round(success_rate, 3),
        'avg_confidence_score': round(avg_confidence, 3),
        'avg_retrieval_score': round(avg_retrieval, 3),
        'avg_retrieved_docs': round(avg_retrieved_docs, 1),
        'avg_citations': round(avg_citations, 1),
        'avg_answer_length_chars': round(avg_answer_length, 0),
        'high_confidence_answers': high_confidence_count,
        'high_confidence_rate': round(high_confidence_rate, 3)
    }

def print_metrics(metrics: Dict):
    print(f"\nOverall Performance:")
    print(f"  - Total Questions: {metrics['total_questions']}")
    print(f"  - Successful Answers: {metrics['successful_answers']}")
    print(f"  - Failed Answers: {metrics['failed_answers']}")
    print(f"  - Success Rate: {metrics['success_rate']:.1%}")
    print(f"\nQuality Metrics:")
    print(f"  - Average Confidence Score: {metrics['avg_confidence_score']:.3f}")
    print(f"  - Average Retrieval Score: {metrics['avg_retrieval_score']:.3f}")
    print(f"  - High Confidence Rate (>=0.7): {metrics['high_confidence_rate']:.1%}")
    print(f"\nRetrieval Metrics:")
    print(f"  - Avg Retrieved Documents: {metrics['avg_retrieved_docs']:.1f}")
    print(f"  - Avg Citations per Answer: {metrics['avg_citations']:.1f}")
    print(f"\nAnswer Metrics:")
    print(f"  - Avg Answer Length: {metrics['avg_answer_length_chars']:.0f} characters")

def save_evaluation_results(results: Dict, output_file: str = "eval/evaluation_results.json"):

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to: {output_file}")

def run_ingestion():
    print("\n" + "="*70)
    print("RUNNING DOCUMENT INGESTION")
    print("="*70)

    data_dir = Path("data")
    if not data_dir.exists():
        print("\n[X] Error: 'data/' directory not found")
        return False

    files = list(data_dir.rglob('*'))
    doc_files = [f for f in files if f.is_file() and f.suffix.lower() in ['.pdf', '.html', '.md', '.txt']]

    if not doc_files:
        print("\n[X] Error: No documents found in 'data/' directory")
        return False

    print(f"\n[OK] Found {len(doc_files)} document(s) to process")

    try:
        print("\nLoading documents...")
        loader = DocumentLoader()
        documents = loader.load_directory("data/")

        if not documents:
            print("[X] No documents were successfully loaded")
            return False

        print(f"[OK] Loaded {len(documents)} documents")
        print("\nChunking documents...")
        chunker = DocumentChunker.from_config()
        chunks = chunker.chunk_documents(documents)
        print(f"[OK] Created {len(chunks)} chunks")
        print("\nGenerating embeddings and upserting to Pinecone...")
        upserter = EmbeddingUpserter.from_config()
        result = upserter.upsert_documents(chunks, show_progress=False)
        index_stats = upserter.get_index_stats()
        print(f"[OK] Knowledge base now has {index_stats.get('total_vectors', 0)} vectors")
        print("\n" + "="*70)
        print("[OK] INGESTION COMPLETE")
        print("="*70)

        return True

    except Exception as e:
        print(f"\n[X] Error during ingestion: {e}")
        return False

def generate_pdf_report():
    """Generate PDF report from evaluation results."""
    print("\n" + "="*70)
    print("GENERATING PDF REPORT")
    print("="*70)

    try:
        sys.path.append(str(Path(__file__).parent))
        from generate_pdf_report import create_pdf_report, load_evaluation_results

        print("\nLoading evaluation results...")
        results = load_evaluation_results()
        print("[OK] Results loaded")

        print("\nGenerating PDF...")
        create_pdf_report(results)
        print("[OK] PDF generated successfully")

        print("\n" + "="*70)
        print("[OK] PDF REPORT COMPLETE")
        print("="*70)

        return True

    except Exception as e:
        print(f"\n[X] Error generating PDF: {e}")
        return False

def main():
    """Main evaluation function with automatic ingestion and PDF generation."""

    print("=" * 70)
    print("RAG SYSTEM - COMPLETE EVALUATION PIPELINE")
    print("=" * 70)

    print("\nSTEP 1: Document Ingestion")
    print("-" * 70)
    if not run_ingestion():
        print("\n[X] Ingestion failed. Attempting to continue with existing data...")

    print("\n\nSTEP 2: Initialize RAG Chain")
    print("-" * 70)
    try:
        rag_chain = RAGChain.from_config()
        print("[OK] RAG chain initialized")
    except Exception as e:
        print(f"[X] Error: {e}")
        return

    print("\n\nSTEP 3: Load Evaluation Dataset")
    print("-" * 70)
    try:
        qa_pairs = load_qa_gold()
        print(f"[OK] Loaded {len(qa_pairs)} Q&A pairs")
    except Exception as e:
        print(f"[X] Error loading dataset: {e}")
        return

    print("\n\nSTEP 4: Verify Knowledge Base")
    print("-" * 70)
    try:
        stats = rag_chain.retriever.get_index_stats()
        vector_count = stats.get('total_vectors', 0)
        print(f"[OK] Knowledge base has {vector_count} vectors")

        if vector_count == 0:
            print("\n[X] Warning: Knowledge base is empty!")
            print("    Evaluation may not produce meaningful results.")
    except Exception as e:
        print(f"[X] Could not check knowledge base: {e}")

    print("\n\nSTEP 5: Run RAGAS Evaluation")
    print("-" * 70)
    evaluation_results = evaluate_rag_system(rag_chain, qa_pairs, verbose=True)

    print("\n\nSTEP 6: Save Results")
    print("-" * 70)
    save_evaluation_results(evaluation_results)

    print("\n\nSTEP 7: Generate PDF Report")
    print("-" * 70)
    if not generate_pdf_report():
        print("\n[X] PDF generation failed, but evaluation results are saved.")

    print("\n" + "=" * 70)
    print("[OK] COMPLETE EVALUATION PIPELINE FINISHED!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - eval/evaluation_results.json")
    print("  - eval/RAGAS_Evaluation_Report.pdf")
    print("\nYour RAG system has been evaluated successfully!")

if __name__ == "__main__":
    main()
