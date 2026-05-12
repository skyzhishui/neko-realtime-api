"""Audio Buffer Manager - buffers PCM16 audio and encodes to WAV."""
import io
import wave
import base64
import numpy as np
import logging

logger = logging.getLogger("realtime-server")


class AudioBufferManager:
    """Manages audio buffering, WAV encoding, and base64 conversion."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.chunks: list[bytes] = []
        self.image_chunks: list[str] = []  # base64 jpeg

    def append_audio(self, pcm16_b64: str):
        """Append base64-encoded PCM16 audio."""
        raw = base64.b64decode(pcm16_b64)
        self.chunks.append(raw)

    def append_audio_raw(self, pcm16_bytes: bytes):
        """Append raw PCM16 audio bytes."""
        self.chunks.append(pcm16_bytes)

    def clear_audio(self):
        """Clear audio buffer."""
        self.chunks.clear()

    def append_image(self, image_b64: str):
        """Append base64-encoded JPEG image."""
        self.image_chunks.append(image_b64)

    def clear_images(self):
        """Clear image buffer."""
        self.image_chunks.clear()

    def get_full_pcm(self) -> bytes:
        """Get complete PCM16 audio."""
        return b"".join(self.chunks)

    def get_wav_b64(self) -> str:
        """Encode buffer audio as WAV + base64 (for Omni input_audio)."""
        pcm = self.get_full_pcm()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm)
        return base64.b64encode(buf.getvalue()).decode()

    def get_wav_bytes(self) -> bytes:
        """Encode buffer audio as WAV bytes."""
        pcm = self.get_full_pcm()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm)
        return buf.getvalue()

    def get_duration_ms(self) -> float:
        """Get buffer audio duration in ms."""
        total_samples = len(self.get_full_pcm()) // 2  # PCM16 = 2 bytes/sample
        return total_samples / self.sample_rate * 1000

    def reset(self):
        """Reset all buffers."""
        self.chunks.clear()
        self.image_chunks.clear()
