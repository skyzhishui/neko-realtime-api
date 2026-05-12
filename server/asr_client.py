"""SenseVoice ASR Client - HTTP/WS transcription."""
import aiohttp
import io
import wave
import json
import logging
from typing import AsyncIterator

logger = logging.getLogger("realtime-server")


class SenseVoiceASRClient:
    """SenseVoice ASR client.
    
    HTTP: POST http://localhost:8082/v1/audio/transcriptions
    WS:   ws://localhost:8082/ws/asr
    """

    def __init__(self, base_url: str = "http://localhost:8082"):
        self.base_url = base_url

    async def transcribe_http(
        self, pcm_bytes: bytes, sample_rate: int = 16000, language: str = "zh"
    ) -> str:
        """HTTP transcription, returns full text.
        
        Latency: ~260ms (2s audio)
        """
        wav_bytes = self._pcm_to_wav(pcm_bytes, sample_rate)
        
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            data = aiohttp.FormData()
            data.add_field(
                "file", wav_bytes, filename="audio.wav", content_type="audio/wav"
            )
            data.add_field("model", "SenseVoiceSmall")
            data.add_field("language", language)
            
            try:
                async with session.post(
                    f"{self.base_url}/v1/audio/transcriptions",
                    data=data,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"ASR error {resp.status}: {error_text[:500]}")
                        return ""
                    result = await resp.json()
                    return result.get("text", "")
            except Exception as e:
                logger.error(f"ASR HTTP error: {e}")
                return ""

    async def transcribe_ws(
        self, pcm_chunks: AsyncIterator[bytes]
    ) -> str:
        """WebSocket streaming transcription.
        
        Sends int16 PCM binary stream, receives JSON transcription results.
        
        ⚠️ Note: SenseVoice GPU inference is serialized, concurrent requests queue up.
        """
        import websockets
        
        full_text = ""
        ws_url = f"ws://{self.base_url.replace('http://', '').replace('https://', '')}/ws/asr"
        
        try:
            async with websockets.connect(ws_url) as ws:
                async for chunk in pcm_chunks:
                    await ws.send(chunk)  # Send int16 PCM binary
                # Send end signal
                await ws.send(b"")
                
                async for msg in ws:
                    result = json.loads(msg)
                    if result.get("status") == "success":
                        full_text = result.get("text", "")
                    if result.get("is_final", False):
                        break
        except Exception as e:
            logger.error(f"ASR WS error: {e}")
        
        return full_text

    def _pcm_to_wav(self, pcm_bytes: bytes, sample_rate: int) -> bytes:
        """Convert PCM bytes to WAV format."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()
