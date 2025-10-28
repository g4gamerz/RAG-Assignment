"""
Speech-to-Text using LemonFox.ai Whisper API.
Fast, cloud-based transcription with <2.5s latency.
"""
import requests
import time
import os
from typing import Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class LemonFoxSTT:
    """Speech-to-Text using LemonFox.ai API."""

    def __init__(self, api_key: str = None):
        """
        Initialize LemonFox STT.

        Args:
            api_key: LemonFox.ai API key (defaults to env variable)
        """
        self.api_key = api_key or os.getenv('LEMONFOX_API_KEY')
        self.api_url = "https://api.lemonfox.ai/v1/audio/transcriptions"

        if not self.api_key:
            raise ValueError("LemonFox API key not found. Set LEMONFOX_API_KEY environment variable.")

    def transcribe(
        self,
        audio_path: str,
        language: str = "english",
        response_format: str = "json"
    ) -> Dict:
        """
        Transcribe audio file to text using LemonFox API.

        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)
            language: Language (default: "english")
            response_format: Response format (json, vtt, srt)

        Returns:
            Dictionary with transcription results and timing info
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "language": language,
            "response_format": response_format
        }

        # Upload local file
        files = {
            "file": open(str(audio_path), "rb")
        }

        # Transcribe with timing
        start_time = time.time()

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                files=files,
                data=data,
                timeout=10  # 10 second timeout
            )

            transcription_time = time.time() - start_time

            # Close file
            files["file"].close()

            if response.status_code == 200:
                result = response.json()

                return {
                    "text": result.get("text", "").strip(),
                    "transcription_time": transcription_time,
                    "meets_latency_target": transcription_time < 2.5,
                    "api": "lemonfox",
                    "status": "success"
                }
            else:
                return {
                    "text": "",
                    "transcription_time": transcription_time,
                    "meets_latency_target": False,
                    "api": "lemonfox",
                    "status": "error",
                    "error": f"API Error {response.status_code}: {response.text}"
                }

        except requests.exceptions.Timeout:
            return {
                "text": "",
                "transcription_time": time.time() - start_time,
                "meets_latency_target": False,
                "api": "lemonfox",
                "status": "error",
                "error": "Request timeout (>10s)"
            }
        except Exception as e:
            return {
                "text": "",
                "transcription_time": time.time() - start_time,
                "meets_latency_target": False,
                "api": "lemonfox",
                "status": "error",
                "error": str(e)
            }
        finally:
            # Ensure file is closed
            if 'files' in locals() and files.get("file"):
                try:
                    files["file"].close()
                except:
                    pass

    def transcribe_fast(self, audio_path: str) -> str:
        """
        Fast transcription that only returns text.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        result = self.transcribe(audio_path)
        return result.get("text", "")

    def get_stats(self) -> Dict:
        """Get API statistics."""
        return {
            "api": "lemonfox",
            "api_key_configured": bool(self.api_key),
            "latency_target_ms": 2500,
            "cloud_based": True
        }


if __name__ == "__main__":
    # Example usage and speed test
    import sys

    print("=" * 60)
    print("LEMONFOX WHISPER API - SPEED TEST")
    print("=" * 60)

    try:
        stt = LemonFoxSTT()
        print(f"✓ API Key configured: {stt.api_key[:10]}...")

        # Test with a sample audio file if provided
        if len(sys.argv) > 1:
            audio_file = sys.argv[1]
            print(f"\nTranscribing: {audio_file}")

            result = stt.transcribe(audio_file)

            if result["status"] == "success":
                print(f"\n✓ Transcription Time: {result['transcription_time']:.3f}s")
                print(f"✓ Meets <2.5s target: {result['meets_latency_target']}")
                print(f"✓ Text: {result['text'][:100]}...")

                if result['meets_latency_target']:
                    print(f"\n✅ API meets latency requirement!")
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
        else:
            stats = stt.get_stats()
            print(f"\nAPI: {stats['api']}")
            print(f"Cloud-based: {stats['cloud_based']}")
            print(f"Target latency: {stats['latency_target_ms']}ms")
            print("\nTo test transcription:")
            print("  python lemonfox_stt.py <audio_file.mp3>")

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure LEMONFOX_API_KEY is set in .env file")

    print("\n" + "=" * 60)
