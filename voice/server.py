"""
Voice interaction server for real-time RAG queries.
STT → RAG → TTS pipeline with <2.5s latency goal.
"""
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from voice.stt import SpeechToText
from voice.tts import TextToSpeech, FastTTS
from rag.chain import RAGChain
from dotenv import load_dotenv

load_dotenv()


class VoiceRAGServer:
    """Voice interaction server for RAG system."""

    def __init__(
        self,
        stt_engine: str = "google",
        use_fast_tts: bool = True
    ):
        """
        Initialize voice RAG server.

        Args:
            stt_engine: STT engine to use ("whisper" or "google")
            use_fast_tts: Whether to use fast TTS with chunking
        """
        print("Initializing Voice RAG Server...")

        # Initialize components
        self.stt = SpeechToText(engine=stt_engine)
        print("[OK] STT initialized")

        if use_fast_tts:
            self.tts = FastTTS()
        else:
            self.tts = TextToSpeech()
        print("[OK] TTS initialized")

        self.rag_chain = RAGChain.from_config()
        print("[OK] RAG chain initialized")

        # Performance tracking
        self.latency_target = 2.5  # seconds

    def process_voice_query(
        self,
        timeout: int = 5,
        phrase_time_limit: int = 10
    ) -> Optional[dict]:
        """
        Process a single voice query through the full pipeline.

        Args:
            timeout: Seconds to wait for speech
            phrase_time_limit: Max seconds for phrase

        Returns:
            Dictionary with results and timing info
        """
        start_time = time.time()

        # Step 1: Speech-to-Text
        print("\n🎤 Listening for question...")
        stt_start = time.time()
        question = self.stt.listen_from_microphone(
            timeout=timeout,
            phrase_time_limit=phrase_time_limit
        )
        stt_time = time.time() - stt_start

        if not question:
            print("[X] No question detected")
            return None

        print(f"[OK] Question: {question}")
        print(f"   STT time: {stt_time:.2f}s")

        # Step 2: RAG Processing
        print("\n🔍 Processing through RAG...")
        rag_start = time.time()
        try:
            response = self.rag_chain.ask(question, include_sources=False)
            answer = response['answer']
            confidence = response['confidence_score']
        except Exception as e:
            print(f"[X] RAG error: {e}")
            answer = "I encountered an error processing your question."
            confidence = 0.0

        rag_time = time.time() - rag_start

        print(f"[OK] Answer: {answer[:100]}...")
        print(f"   RAG time: {rag_time:.2f}s")
        print(f"   Confidence: {confidence:.2f}")

        # Step 3: Text-to-Speech
        print("\n🔊 Speaking answer...")
        tts_start = time.time()

        if isinstance(self.tts, FastTTS):
            self.tts.speak_fast(answer)
        else:
            self.tts.speak(answer, play_audio=True)

        tts_time = time.time() - tts_start

        print(f"[OK] TTS time: {tts_time:.2f}s")

        # Total time
        total_time = time.time() - start_time

        # Results
        result = {
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'timing': {
                'stt_seconds': round(stt_time, 2),
                'rag_seconds': round(rag_time, 2),
                'tts_seconds': round(tts_time, 2),
                'total_seconds': round(total_time, 2)
            },
            'latency_target_met': total_time <= self.latency_target
        }

        # Print summary
        print("\n" + "="*60)
        print("📊 Performance Summary")
        print("="*60)
        print(f"Total latency: {total_time:.2f}s")
        print(f"Target: {self.latency_target}s")

        if result['latency_target_met']:
            print("[OK] Target met!")
        else:
            print(f"⚠ Exceeded by {total_time - self.latency_target:.2f}s")

        print(f"\nBreakdown:")
        print(f"  STT:  {stt_time:.2f}s ({stt_time/total_time*100:.1f}%)")
        print(f"  RAG:  {rag_time:.2f}s ({rag_time/total_time*100:.1f}%)")
        print(f"  TTS:  {tts_time:.2f}s ({tts_time/total_time*100:.1f}%)")

        return result

    def run_interactive(self):
        """Run interactive voice session."""
        print("\n" + "="*60)
        print("🎤 Voice RAG Interactive Mode")
        print("="*60)
        print("\nInstructions:")
        print("- Speak your question when prompted")
        print("- Press Ctrl+C to exit")
        print(f"- Target latency: <{self.latency_target}s")
        print("\n" + "="*60)

        try:
            while True:
                print("\n\n" + "▶"*30)
                result = self.process_voice_query()

                if result is None:
                    print("\n⏭️  No query detected, ready for next question...")

                input("\n Press Enter to continue (or Ctrl+C to exit)...")

        except KeyboardInterrupt:
            print("\n\n👋 Exiting voice RAG server...")
            return


def main():
    """Main entry point for voice server."""
    import argparse

    parser = argparse.ArgumentParser(description="Voice RAG Server")
    parser.add_argument(
        "--stt-engine",
        choices=["whisper", "google"],
        default="google",
        help="STT engine to use"
    )
    parser.add_argument(
        "--no-fast-tts",
        action="store_true",
        help="Disable fast TTS"
    )

    args = parser.parse_args()

    try:
        server = VoiceRAGServer(
            stt_engine=args.stt_engine,
            use_fast_tts=not args.no_fast_tts
        )

        server.run_interactive()

    except Exception as e:
        print(f"\n[X] Error: {e}")
        print("\nMake sure:")
        print("1. Microphone is connected and working")
        print("2. RAG system is properly configured")
        print("3. Knowledge base is populated")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
