"""
Text-to-Speech using LemonFox.ai API.
Converts text responses to speech audio.
"""
import requests
import os
from typing import Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
import time

load_dotenv()


class LemonFoxTTS:
    """Text-to-Speech using LemonFox.ai API."""

    def __init__(self, api_key: str = None):
        """
        Initialize LemonFox TTS.

        Args:
            api_key: LemonFox.ai API key (defaults to env variable)
        """
        self.api_key = api_key or os.getenv('LEMONFOX_API_KEY')
        self.api_url = "https://api.lemonfox.ai/v1/audio/speech"

        if not self.api_key:
            raise ValueError("LemonFox API key not found. Set LEMONFOX_API_KEY environment variable.")

    def synthesize(
        self,
        text: str,
        voice: str = "sarah",
        response_format: str = "mp3",
        output_path: Optional[str] = None,
        stream: bool = False
    ) -> Dict:
        """
        Convert text to speech using LemonFox API.

        Args:
            text: Text to convert to speech
            voice: Voice to use (sarah, john, etc.)
            response_format: Audio format (mp3, wav, opus)
            output_path: Path to save audio file (optional)
            stream: Enable streaming response (faster perceived latency)

        Returns:
            Dictionary with synthesis results
        """
        if not text or not text.strip():
            return {
                "status": "error",
                "error": "No text provided"
            }

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "input": text,
            "voice": voice,
            "response_format": response_format
        }

        start_time = time.time()

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=15,  # 15 second timeout
                stream=stream  # Enable streaming
            )

            if response.status_code == 200:
                if stream:
                    # Stream audio chunks
                    audio_chunks = []
                    for chunk in response.iter_content(chunk_size=4096):
                        if chunk:
                            audio_chunks.append(chunk)
                    audio_content = b''.join(audio_chunks)
                else:
                    # Get complete response
                    audio_content = response.content

                synthesis_time = time.time() - start_time

                # Save audio if output path provided
                if output_path:
                    with open(output_path, "wb") as f:
                        f.write(audio_content)
                    saved_path = output_path
                else:
                    saved_path = None

                return {
                    "status": "success",
                    "audio_content": audio_content,
                    "saved_path": saved_path,
                    "synthesis_time": synthesis_time,
                    "format": response_format,
                    "voice": voice,
                    "text_length": len(text),
                    "streamed": stream
                }
            else:
                synthesis_time = time.time() - start_time
                return {
                    "status": "error",
                    "error": f"API Error {response.status_code}: {response.text}",
                    "synthesis_time": synthesis_time
                }

        except requests.exceptions.Timeout:
            return {
                "status": "error",
                "error": "Request timeout (>15s)",
                "synthesis_time": time.time() - start_time
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "synthesis_time": time.time() - start_time
            }

    def synthesize_to_file(self, text: str, output_path: str, voice: str = "sarah") -> bool:
        """
        Synthesize text to audio file.

        Args:
            text: Text to convert
            output_path: Path to save audio file
            voice: Voice to use

        Returns:
            True if successful, False otherwise
        """
        result = self.synthesize(text, voice=voice, output_path=output_path)
        return result["status"] == "success"

    def get_stats(self) -> Dict:
        """Get TTS API statistics."""
        return {
            "api": "lemonfox",
            "api_key_configured": bool(self.api_key),
            "cloud_based": True,
            "supported_formats": ["mp3", "wav", "opus"],
            "available_voices": ["sarah", "john"]
        }


if __name__ == "__main__":
    # Example usage and test
    import sys

    print("=" * 60)
    print("LEMONFOX TTS API - TEST")
    print("=" * 60)

    try:
        tts = LemonFoxTTS()
        print(f"✓ API Key configured: {tts.api_key[:10]}...")

        # Test with sample text
        test_text = "Hello! This is a test of the LemonFox text to speech API."

        if len(sys.argv) > 1:
            test_text = " ".join(sys.argv[1:])

        print(f"\nSynthesizing: '{test_text}'")

        result = tts.synthesize(test_text, output_path="test_speech.mp3")

        if result["status"] == "success":
            print(f"\n✓ Synthesis Time: {result['synthesis_time']:.3f}s")
            print(f"✓ Voice: {result['voice']}")
            print(f"✓ Format: {result['format']}")
            print(f"✓ Saved to: {result['saved_path']}")
            print(f"\n✅ Audio file created successfully!")
        else:
            print(f"\n❌ Error: {result.get('error', 'Unknown error')}")

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure LEMONFOX_API_KEY is set in .env file")

    print("\n" + "=" * 60)
