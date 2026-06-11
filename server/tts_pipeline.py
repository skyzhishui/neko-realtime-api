"""TTS Pipeline - HTTP streaming + punctuation-based sentence splitting + WebSocket streaming."""
import aiohttp
import re
import asyncio
import time
import json
import base64
import logging
from typing import AsyncIterator


logger = logging.getLogger("realtime-server")


class TTSPipeline:
    """Qwen3-TTS HTTP streaming pipeline.
    
    Core flow:
    1. LLM outputs text delta -> buffer
    2. SentenceSplitter detects complete sentences -> split
    3. For each sentence, initiate HTTP streaming TTS request
    4. Receive PCM stream -> yield PCM 24kHz directly (no WAV header)
    
    Output audio format: PCM16 24kHz mono (raw binary stream)
    
    Supports voice cloning via ref_audio/ref_text.
    """

    # Sentence splitting punctuation
    SENTENCE_ENDINGS = re.compile(r"([。！？；\n])")
    SUB_SENTENCE_ENDINGS = re.compile(r"([，、：,])")

    # Punctuation/whitespace pattern: matches all common CJK/Latin punctuation and whitespace
    PUNCTUATION_WHITESPACE_RE = re.compile(
        r"["
        r"\s"                          # whitespace (space, tab, newline, etc.)
        r"\u3000"                       # fullwidth space
        r"\u3001\u3002"                 # 、。
        r"\uff01\uff08\uff09\uff0c"     # ！（），
        r"\uff1a\uff1b\uff1f"           # ：；？
        r"\u201c\u201d\u2018\u2019"     # ""''
        r"\u3010\u3011"                 # 【】
        r"\u300a\u300b"                 # 《》
        r"\u2026\u2014\u00b7"           # …—·
        r"!,\\.?;:'\"()\\[\\]{}<>"
        r"]+"
    )

    def __init__(
        self,
        base_url: str = "http://localhost:8091/v1",
        voice: str = "Vivian",
        sample_rate_out: int = 24000,
        timeout_s: int = 15,
        api_key: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ):
        self.base_url = base_url
        self.voice = voice
        self.sample_rate_out = sample_rate_out
        self.tts_sample_rate = 24000  # Qwen3-TTS fixed output 24kHz
        self.timeout_s = timeout_s
        self.api_key = api_key
        self._session: aiohttp.ClientSession | None = None

        # Voice cloning (ref_audio/ref_text)
        self.ref_audio = self._resolve_ref_audio(ref_audio)
        self.ref_text = ref_text

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=10, keepalive_timeout=30)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    @staticmethod
    def _is_punctuation_only(text: str) -> bool:
        """Check if text contains only punctuation/whitespace with no actual content."""
        stripped = TTSPipeline.PUNCTUATION_WHITESPACE_RE.sub("", text)
        return len(stripped) == 0

    @staticmethod
    def _resolve_ref_audio(ref_audio: str | None) -> str | None:
        """Resolve ref_audio to a format suitable for the API.

        If ref_audio is a local file path (no scheme like http://, https://, data:, file://),
        read the file and convert to a base64 data URL.
        Otherwise, return as-is (already an HTTP URL, data URL, or file:// URI).
        """
        if ref_audio is None:
            return None

        # If it already has a scheme, return as-is
        if "://" in ref_audio:
            return ref_audio

        # Treat as a local file path
        from pathlib import Path
        path = Path(ref_audio)
        if not path.exists():
            logger.warning(f"ref_audio file not found: {ref_audio}, using as-is")
            return ref_audio

        try:
            audio_bytes = path.read_bytes()
            b64 = base64.b64encode(audio_bytes).decode("utf-8")
            suffix = path.suffix.lower().lstrip(".")
            mime_map = {
                "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
                "ogg": "audio/ogg", "aac": "audio/aac", "pcm": "audio/pcm",
            }
            mime_type = mime_map.get(suffix, "audio/wav")
            data_url = f"data:{mime_type};base64,{b64}"
            logger.info(f"ref_audio: loaded local file {ref_audio} ({len(audio_bytes)} bytes) -> base64 data URL")
            return data_url
        except Exception as e:
            logger.error(f"Failed to read ref_audio file {ref_audio}: {e}")
            return ref_audio

    async def stream_tts(
        self,
        text: str,
        instructions: str = "",
        language: str = "Chinese",
    ) -> AsyncIterator[bytes]:
        """Single sentence HTTP streaming TTS, yields PCM16 24kHz bytes (raw stream, no WAV header)."""
        payload = {
            "model": "Qwen3-TTS",
            "voice": self.voice,
            "input": text,
            "response_format": "pcm",
            "stream": True,
            "language": language,
        }
        if instructions:
            payload["instructions"] = instructions

        # Voice cloning
        if self.ref_audio:
            payload["ref_audio"] = self.ref_audio
            if self.ref_text:
                payload["ref_text"] = self.ref_text

        t_tts_sent = time.time()
        first_chunk_logged = False
        session = await self._get_session()
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            async with session.post(
                f"{self.base_url}/audio/speech",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout_s),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"TTS error {resp.status}: {error_text[:500]}")
                    return
                
                async for chunk in resp.content.iter_chunked(4096):
                    if chunk:
                        if not first_chunk_logged:
                            latency_ms = (time.time() - t_tts_sent) * 1000
                            logger.info(f'[TRACE] tts_first_chunk: latency={latency_ms:.1f}ms, text="{text[:30]}"')
                            first_chunk_logged = True
                        yield chunk

        except asyncio.CancelledError:
            logger.info("[Qwen3-TTS] stream_tts cancelled")
            raise
        except Exception as e:
            logger.error(f"TTS streaming error: {e}")

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None



class SentenceSplitter:
    """Punctuation-based sentence splitter for LLM text delta stream."""

    def __init__(self, min_sub_sentence_len: int = 6):
        self.buffer = ""
        self.min_sub_sentence_len = min_sub_sentence_len

    def add_text(self, text: str) -> list[str]:
        """Add text delta, return list of complete sentences."""
        self.buffer += text
        sentences = []

        while True:
            match = TTSPipeline.SENTENCE_ENDINGS.search(self.buffer)
            if not match:
                break
            end_pos = match.end()
            sentence = self.buffer[:end_pos].strip()
            if sentence and not TTSPipeline._is_punctuation_only(sentence):
                sentences.append(sentence)
            self.buffer = self.buffer[end_pos:]

        if len(self.buffer) >= self.min_sub_sentence_len:
            match = TTSPipeline.SUB_SENTENCE_ENDINGS.search(self.buffer)
            if match:
                end_pos = match.end()
                sentence = self.buffer[:end_pos].strip()
                if sentence and len(sentence) >= self.min_sub_sentence_len and not TTSPipeline._is_punctuation_only(sentence):
                    sentences.append(sentence)
                    self.buffer = self.buffer[end_pos:]

        return sentences

    def flush(self) -> list[str]:
        """Return remaining buffer content."""
        sentences = []
        if self.buffer.strip():
            sentence = self.buffer.strip()
            if not TTSPipeline._is_punctuation_only(sentence):
                sentences.append(sentence)
        self.buffer = ""
        return sentences

    def reset(self):
        """Reset buffer."""
        self.buffer = ""
