"""TTS Pipeline - HTTP streaming + punctuation-based sentence splitting + WebSocket streaming."""
import aiohttp
import re
import asyncio
import time
import json
import logging
from typing import AsyncIterator


logger = logging.getLogger("realtime-server")


class TTSPipeline:
    """Qwen3-TTS HTTP streaming pipeline.
    
    Core flow:
    1. LLM outputs text delta -> buffer
    2. SentenceSplitter detects complete sentences -> split
    3. For each sentence, initiate HTTP streaming TTS request
    4. Receive WAV stream -> skip 44-byte header -> yield PCM 24kHz directly
    
    Output audio format: PCM16 24kHz mono
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
        r"\u201c\u201d\u2018\u2019"     # “”‘’
        r"\u3010\u3011"                 # 【】
        r"\u300a\u300b"                 # 《》
        r"\u2026\u2014\u00b7"           # …—·
        r"!,\\.?;:'\"()\\[\\]{}<>"
        r"]+"
    )

    def __init__(
        self,
        base_url: str = "http://localhost:8091",
        voice: str = "Vivian",
        sample_rate_out: int = 24000,
        timeout_s: int = 15,
    ):
        self.base_url = base_url
        self.voice = voice
        self.sample_rate_out = sample_rate_out
        self.tts_sample_rate = 24000  # Qwen3-TTS fixed output 24kHz
        self.timeout_s = timeout_s
        self._session: aiohttp.ClientSession | None = None

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

    async def stream_tts(
        self,
        text: str,
        instructions: str = "",
        language: str = "Chinese",
    ) -> AsyncIterator[bytes]:
        """Single sentence HTTP streaming TTS, yields PCM16 24kHz bytes."""
        payload = {
            "model": "Qwen3-TTS",
            "voice": self.voice,
            "input": text,
            "response_format": "wav",
            "stream": True,
            "language": language,
        }
        if instructions:
            payload["instructions"] = instructions

        t_tts_sent = time.time()
        first_chunk_logged = False
        wav_header_skipped = False
        leftover = b""
        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/audio/speech",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout_s),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"TTS error {resp.status}: {error_text[:500]}")
                    return
                
                async for chunk in resp.content.iter_chunked(4096):
                    if not wav_header_skipped:
                        if len(chunk) > 44:
                            chunk = chunk[44:]
                        else:
                            continue
                        wav_header_skipped = True
                    
                    chunk = leftover + chunk
                    leftover = b""

                    if len(chunk) % 2 != 0:
                        leftover = chunk[-1:]
                        chunk = chunk[:-1]

                    if chunk:
                        if not first_chunk_logged:
                            latency_ms = (time.time() - t_tts_sent) * 1000
                            logger.info(f'[TRACE] tts_first_chunk: latency={latency_ms:.1f}ms, text="{text[:30]}"')
                            first_chunk_logged = True
                        yield chunk

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
