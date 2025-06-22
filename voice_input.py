"""
TermiAI - Voice Input Module
Handles voice recording and transcription using Faster-Whisper
"""

import os
import asyncio
import tempfile
import threading
from pathlib import Path
from typing import Optional

try:
    import pyaudio
    import wave
    from faster_whisper import WhisperModel
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install faster-whisper pyaudio")
    exit(1)


class VoiceProcessor:
    """Handles voice recording and transcription with Faster-Whisper"""

    def __init__(self, config: dict):
        """Initialize voice processor with Faster-Whisper"""
        self.config = config
        self.model_size = config.get('whisper_model_size', 'tiny.en')
        self.device = config.get('whisper_device', 'cpu')
        self.compute_type = config.get('whisper_compute_type', 'int8')

        # Audio recording settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.recording = False

        # Initialize Whisper model
        print(f"🔧 Loading Whisper model: {self.model_size}")
        try:
            self.whisper_model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            print("✅ Faster-Whisper model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            raise

    def __del__(self):
        """Cleanup PyAudio resources"""
        if hasattr(self, 'audio'):
            self.audio.terminate()

    async def listen_and_transcribe(self) -> Optional[str]:
        """Record audio and transcribe using Faster-Whisper"""
        try:
            # Record audio
            audio_file = await self._record_audio()
            if not audio_file:
                return None

            # Transcribe audio
            text = await self._transcribe_audio(audio_file)

            # Cleanup temp file
            try:
                os.unlink(audio_file)
            except:
                pass

            return text

        except Exception as e:
            print(f"❌ Voice processing error: {e}")
            return None

    async def _record_audio(self) -> Optional[str]:
        """Record audio from microphone"""
        try:
            # Create temporary file for recording
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_filename = temp_file.name
            temp_file.close()

            # Record audio in a separate thread
            recording_complete = threading.Event()
            audio_data = []

            def record():
                try:
                    stream = self.audio.open(
                        format=self.audio_format,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk_size
                    )

                    print("🔴 Recording... (press Enter to finish)")
                    self.recording = True

                    while self.recording:
                        data = stream.read(self.chunk_size, exception_on_overflow=False)
                        audio_data.append(data)

                    stream.stop_stream()
                    stream.close()
                    recording_complete.set()

                except Exception as e:
                    print(f"❌ Recording error: {e}")
                    recording_complete.set()

            # Start recording thread
            record_thread = threading.Thread(target=record)
            record_thread.daemon = True
            record_thread.start()

            # Wait for user to press Enter (non-blocking)
            await self._wait_for_enter()
            self.recording = False

            # Wait for recording to complete
            recording_complete.wait(timeout=1.0)

            if not audio_data:
                print("❌ No audio data recorded")
                return None

            # Save audio to WAV file
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(audio_data))

            print("⏹️  Recording finished")
            return temp_filename

        except Exception as e:
            print(f"❌ Audio recording error: {e}")
            return None

    async def _wait_for_enter(self):
        """Wait for user to press Enter (async)"""
        loop = asyncio.get_event_loop()

        def wait_for_input():
            try:
                input()  # Wait for Enter
            except:
                pass

        await loop.run_in_executor(None, wait_for_input)

    async def _transcribe_audio(self, audio_file: str) -> Optional[str]:
        """Transcribe audio file using Faster-Whisper"""
        try:
            print("🔄 Transcribing audio...")

            # Run transcription in executor to avoid blocking
            loop = asyncio.get_event_loop()

            def transcribe():
                segments, info = self.whisper_model.transcribe(
                    audio_file,
                    language="en",
                    vad_filter=True,  # Voice Activity Detection
                    vad_parameters=dict(min_silence_duration_ms=500)
                )

                # Combine all segments into single text
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text.strip())

                return " ".join(text_parts).strip()

            text = await loop.run_in_executor(None, transcribe)

            if text:
                print(f"✅ Transcription complete: '{text}'")
                return text
            else:
                print("❌ No speech detected in audio")
                return None

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None

    def test_microphone(self) -> bool:
        """Test if microphone is working"""
        try:
            # Try to open microphone stream
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            # Read a small amount of data
            data = stream.read(self.chunk_size, exception_on_overflow=False)

            stream.stop_stream()
            stream.close()

            return len(data) > 0

        except Exception as e:
            print(f"❌ Microphone test failed: {e}")
            return False

    def list_audio_devices(self):
        """List available audio input devices"""
        print("\n🎤 Available Audio Input Devices:")
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"   {i}: {device_info['name']} (Channels: {device_info['maxInputChannels']})")

    def get_model_info(self) -> dict:
        """Get information about the loaded Whisper model"""
        return {
            'model_size': self.model_size,
            'device': self.device,
            'compute_type': self.compute_type,
            'sample_rate': self.sample_rate,
            'channels': self.channels
        }


# Utility function for testing
async def test_voice_processor():
    """Test the voice processor functionality"""
    config = {
        'whisper_model_size': 'tiny.en',
        'whisper_device': 'cpu',
        'whisper_compute_type': 'int8'
    }

    processor = VoiceProcessor(config)

    # Test microphone
    if not processor.test_microphone():
        print("❌ Microphone test failed")
        return

    print("✅ Microphone test passed")
    processor.list_audio_devices()

    # Test transcription
    print("\n🎙️  Testing voice transcription...")
    text = await processor.listen_and_transcribe()

    if text:
        print(f"✅ Transcription successful: '{text}'")
    else:
        print("❌ Transcription failed")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_voice_processor())