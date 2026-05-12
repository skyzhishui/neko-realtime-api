"""GSV-TTS-Lite Pipeline - HTTP SSE streaming + punctuation-based sentence splitting.

Architecture: mirrors TTSPipeline (qwen3-tts) but targets GSV-TTS-Lite's
/tts/stream SSE endpoint instead of Qwen3-TTS's /v1/audio/speech.

Core flow:
1. LLM outputs text delta -> SentenceSplitter detects complete sentences
2. For each sentence, POST to /tts/stream (SSE)
3. Parse SSE events: extract base64 audio from "audio" events
4. Decode base64 -> raw int16 PCM 32kHz -> resample to 24kHz -> yield PCM bytes

Output audio format: PCM16 24kHz mono (resampled from GSV's native 32kHz)
"""

import aiohttp
import asyncio
import base64
import json
import logging
import time
import numpy as np
from typing import AsyncIterator

from .tts_pipeline import SentenceSplitter

logger = logging.getLogger("realtime-server")


class GsvTtsPipeline:
    """GSV-TTS-Lite HTTP SSE streaming pipeline.

    Follows the same sentence-splitting pattern as TTSPipeline, but calls
    GSV-TTS-Lite's personal_api /tts/stream endpoint.

    Key differences from Qwen3-TTS TTSPipeline:
    - GSV uses SSE (Server-Sent Events) instead of raw HTTP streaming
    - Audio is base64-encoded in SSE "audio" events (not raw WAV stream)
    - GSV outputs 32kHz int16 PCM (needs resample to 24kHz for pipeline)
    - GSV requires speaker_audio, prompt_audio, prompt_text (voice cloning)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        speaker_audio: str = "",
        prompt_audio: str = "",
        prompt_text: str = "",
        sample_rate_out: int = 24000,
        timeout_s: int = 30,
        speed: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.speaker_audio = speaker_audio
        self.prompt_audio = prompt_audio
        self.prompt_text = prompt_text
        self.sample_rate_out = sample_rate_out
        self.gsv_sample_rate = 32000  # GSV-TTS-Lite fixed output 32kHz
        self.timeout_s = timeout_s
        self.speed = speed

        # Pre-compute resample ratio
        self._resample_ratio = self.gsv_sample_rate / self.sample_rate_out

    def _resample_32k_to_24k(self, pcm_int16: bytes) -> bytes:
        """Resample 32kHz int16 PCM to 24kHz int16 PCM using linear interpolation."""
        if not pcm_int16:
            return b""

        samples = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32)
        new_len = int(len(samples) * self.sample_rate_out / self.gsv_sample_rate)
        if new_len <= 0:
            return b""

        # Linear interpolation
        indices = np.linspace(0, len(samples) - 1, new_len)
        floor_idx = np.floor(indices).astype(int)
        ceil_idx = np.minimum(floor_idx + 1, len(samples) - 1)
        frac = indices - floor_idx

        resampled = samples[floor_idx] * (1 - frac) + samples[ceil_idx] * frac
        resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
        return resampled.tobytes()

    async def stream_tts(
        self,
        text: str,
    ) -> AsyncIterator[bytes]:
        """Single sentence GSV-TTS SSE streaming, yields PCM16 24kHz bytes.

        Calls POST /tts/stream and parses SSE events:
        - event: audio -> decode base64 audio data -> resample 32k->24k -> yield
        - event: done -> finish
        - event: error -> log and return
        """
        payload = {
            "text": text,
            "speaker_audio": self.speaker_audio,
            "prompt_audio": self.prompt_audio,
            "prompt_text": self.prompt_text,
            "is_cut_text": False,  # We already split sentences ourselves
            "speed": self.speed,
        }

        t_tts_sent = time.time()
        first_chunk_logged = False
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(
                    f"{self.base_url}/tts/stream",
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"GSV-TTS error {resp.status}: {error_text[:500]}")
                        return

                    # Parse SSE stream manually (avoid aiohttp line-size limit of 128KB)
                    # base64 audio data in SSE can easily exceed that limit
                    sse_buffer = b""
                    current_event = None
                    async for chunk in resp.content.iter_any():
                        sse_buffer += chunk
                        # Process complete SSE messages (delimited by \n\n)
                        while b"\n\n" in sse_buffer:
                            event_raw, sse_buffer = sse_buffer.split(b"\n\n", 1)
                            lines = event_raw.decode("utf-8", errors="replace").split("\n")

                            for line in lines:
                                line = line.rstrip("\r")
                                if line.startswith("event: "):
                                    current_event = line[7:]
                                elif line.startswith("data: "):
                                    if current_event == "audio":
                                        try:
                                            data = json.loads(line[6:])
                                            audio_b64 = data.get("audio", "")
                                            if not audio_b64:
                                                continue

                                            # Decode base64 -> raw int16 PCM 32kHz
                                            pcm_32k = base64.b64decode(audio_b64)

                                            # Resample 32kHz -> 24kHz
                                            pcm_24k = self._resample_32k_to_24k(pcm_32k)

                                            if pcm_24k:
                                                if not first_chunk_logged:
                                                    latency_ms = (time.time() - t_tts_sent) * 1000
                                                    logger.info(
                                                        f'[TRACE] gsv_tts_first_chunk: latency={latency_ms:.1f}ms, '
                                                        f'text="{text[:30]}"'
                                                    )
                                                    first_chunk_logged = True
                                                yield pcm_24k

                                        except (json.JSONDecodeError, Exception) as e:
                                            logger.warning(f"GSV-TTS SSE audio parse error: {e}")
                                            continue

                                    elif current_event == "error":
                                        try:
                                            data = json.loads(line[6:])
                                            logger.error(f'GSV-TTS SSE error: {data.get("error", "unknown")}')
                                        except json.JSONDecodeError:
                                            logger.error(f"GSV-TTS SSE error (raw): {line[6:]}")
                                        return

                                    elif current_event == "done":
                                        return

            except asyncio.CancelledError:
                logger.info("[GSV-TTS] stream_tts cancelled")
                raise
            except Exception as e:
                logger.error(f"GSV-TTS streaming error: {e}")
