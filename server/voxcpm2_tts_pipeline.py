"""VoxCPM2 TTS Pipeline - HTTP raw PCM streaming + punctuation-based sentence splitting.

Architecture: mirrors TTSPipeline (qwen3-tts) but targets VoxCPM2's
OpenAI-compatible /audio/speech endpoint (base_url includes /v1 prefix).

Core flow:
1. LLM outputs text delta -> SentenceSplitter detects complete sentences
2. For each sentence, POST to /audio/speech (OpenAI-compatible, base_url includes /v1)
3. Read raw PCM16 48kHz audio stream (chunked transfer)
4. Resample 48kHz -> 24kHz via 2:1 decimation -> yield PCM16 bytes

Key differences from Qwen3-TTS TTSPipeline:
- VoxCPM2 returns raw PCM16 stream (content-type: audio/pcm), no WAV header
- VoxCPM2 outputs 48kHz PCM16 (needs 2:1 decimation to 24kHz for pipeline)
- Uses "input" field in request body (OpenAI-compatible)
- No SSE framing — direct chunked binary stream

Output audio format: PCM16 24kHz mono (resampled from VoxCPM2's native 48kHz)
"""

import aiohttp
import asyncio
import base64
import json
import logging
import struct
import time
from pathlib import Path
from typing import AsyncIterator

import websockets

from .tts_pipeline import SentenceSplitter
from ._ref_audio_safety import RefAudioConfig, read_ref_audio_safely, ref_audio_config_from_app_config

logger = logging.getLogger("realtime-server")


class VoxCpm2TtsPipeline:
    """VoxCPM2 HTTP raw PCM streaming pipeline.

    Follows the same sentence-splitting pattern as TTSPipeline, but calls
    VoxCPM2's OpenAI-compatible /audio/speech endpoint (base_url includes /v1 prefix).

    Key differences from Qwen3-TTS TTSPipeline:
    - VoxCPM2 returns raw PCM16 stream (no WAV header, no SSE framing)
    - VoxCPM2 outputs 48kHz PCM16 (needs 2:1 decimation to 24kHz)
    - Uses "input" field in request body (not "text")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8093/v1",
        voice: str = "default",
        timeout_s: float = 30.0,
        api_key: str = "",
        ref_audio: str | None = None,
        ref_text: str | None = None,
        app_config: object | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.voice = voice
        self.timeout_s = timeout_s
        self.api_key = api_key
        self.voxcpm_sample_rate = 48000  # VoxCPM2 fixed output 48kHz
        self._session: aiohttp.ClientSession | None = None

        # Safety config for ref_audio resolution
        if app_config is not None:
            self._ref_audio_cfg = ref_audio_config_from_app_config(app_config)
        else:
            self._ref_audio_cfg = RefAudioConfig()

        # Voice cloning parameters
        # If ref_audio is a local file path, read it and convert to base64 data URL
        self.ref_audio = self._resolve_ref_audio(ref_audio)
        self.ref_text = ref_text

    def _resolve_ref_audio(self, ref_audio: str | None) -> str | None:
        """Resolve ref_audio to a format suitable for the API.

        If ref_audio is a local file path (no scheme like http://, https://, data:, file://),
        read the file safely (path-traversal guard) and convert to a base64 data URL.
        If ref_audio is an HTTP(S) URL, validate against SSRF rules before returning as-is.
        If ref_audio is a data: URL, return as-is.
        """
        if ref_audio is None:
            return None

        # data: URLs are inline — no I/O risk, pass through
        if ref_audio.startswith("data:"):
            return ref_audio

        # HTTP(S) URL — validate against SSRF rules
        if "://" in ref_audio and ref_audio.startswith(("http://", "https://")):
            from ._ref_audio_safety import validate_remote_url
            try:
                validate_remote_url(ref_audio, self._ref_audio_cfg)
            except ValueError as e:
                logger.error(f"ref_audio URL rejected: {e}")
                raise
            return ref_audio

        # Any other scheme (file://, ftp://, etc.) is rejected
        if "://" in ref_audio:
            raise ValueError(
                f"ref_audio URL scheme rejected (only http/https/data allowed): {ref_audio}"
            )

        # Local file path — resolve safely
        try:
            audio_bytes = read_ref_audio_safely(ref_audio, self._ref_audio_cfg)
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"ref_audio local file rejected: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to read ref_audio file {ref_audio}: {e}")
            raise

        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        # Guess MIME type from extension
        suffix = Path(ref_audio).suffix.lower().lstrip(".")
        mime_map = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
            "aac": "audio/aac",
            "webm": "audio/webm",
            "mp4": "audio/mp4",
            "m4a": "audio/mp4",
        }
        mime_type = mime_map.get(suffix, "audio/wav")
        data_url = f"data:{mime_type};base64,{b64}"
        logger.info(f"ref_audio: loaded local file {ref_audio} ({len(audio_bytes)} bytes) -> base64 data URL")
        return data_url

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with connection pooling."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=10, keepalive_timeout=30)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    @staticmethod
    def _resample_48k_to_24k(pcm_data: bytes) -> bytes:
        """Resample 48kHz PCM16 mono to 24kHz via 2:1 decimation.

        Takes every other sample point (simple decimation without filtering).
        Input: PCM16 little-endian signed short (2 bytes per sample) at 48kHz
        Output: PCM16 little-endian signed short at 24kHz

        Args:
            pcm_data: Raw PCM16 48kHz mono bytes

        Returns:
            Raw PCM16 24kHz mono bytes
        """
        if not pcm_data:
            return b""

        # Handle odd-length PCM data (leftover byte) — skip it
        data_len = len(pcm_data)
        if data_len % 2 != 0:
            pcm_data = pcm_data[:-1]
            data_len -= 1

        if data_len == 0:
            return b""

        # Unpack all PCM16 samples, take every other sample, repack
        num_samples = data_len // 2
        samples = struct.unpack(f"<{num_samples}h", pcm_data)
        decimated = samples[::2]
        return struct.pack(f"<{len(decimated)}h", *decimated)

    async def stream_tts(
        self,
        text: str,
        language: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Single sentence VoxCPM2 raw PCM streaming TTS, yields PCM16 24kHz bytes.

        Calls POST /audio/speech and reads raw PCM16 48kHz chunked stream.
        VoxCPM2 returns content-type: audio/pcm with chunked transfer encoding.

        Args:
            text: Text to synthesize
            language: Language code, or None to let VoxCPM2 auto-detect (default None)

        Yields:
            PCM16 24kHz mono audio bytes
        """
        payload = {
            "model": "voxcpm2",
            "input": text,
            "voice": self.voice,
            "response_format": "pcm",
            "stream": True,
        }
        if language is not None:
            payload["language"] = language

        # Voice cloning parameters
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
                    logger.error(f"VoxCPM2-TTS error {resp.status}: {error_text[:500]}")
                    return

                # VoxCPM2 returns raw PCM16 48kHz stream (no WAV header, no SSE framing)
                # Read chunked stream and resample 48kHz -> 24kHz
                # ---- 跨 chunk PCM16 + decimation phase 对齐 (P0 fix B 方案) ----
                # 双重对齐要求：
                #   1. 字节对齐：_resample_48k_to_24k 内部会 trim 奇数字节，跨
                #      chunk 调用时 trim 掉的字节会让下一 chunk 整体错位 1 byte。
                #   2. decimation phase 对齐：函数无状态地做 samples[::2]，每次
                #      都从 chunk[0] 开始取偶数索引。如果上个 chunk 输出了奇数
                #      个 sample，下个 chunk 的 phase 会翻转 (sample[1,3,5,...]
                #      代替 [0,2,4,...])，前几毫秒正常，之后整段杂音。
                # 解决：leftover 缓冲到 **4 字节 (2 sample) 边界**，保证每次
                # resample 输入字节数为 4 的倍数 → sample 数为偶数 → decimation
                # 输出永远以 sample[0] 起点，phase 跨 chunk 一致。
                leftover = b""
                async for chunk in resp.content.iter_chunked(4096):
                    if not chunk:
                        continue
                    buf = leftover + chunk
                    rem = len(buf) % 4
                    if rem:
                        leftover = buf[-rem:]
                        buf = buf[:-rem]
                    else:
                        leftover = b""
                    if not buf:
                        # chunk 累计 < 4 byte，全部入 leftover
                        continue

                    # Resample 48kHz -> 24kHz (输入恒为 4 字节倍数 → 偶数 sample)
                    pcm_24k = self._resample_48k_to_24k(buf)

                    if pcm_24k:
                        if not first_chunk_logged:
                            latency_ms = (time.time() - t_tts_sent) * 1000
                            logger.info(
                                f'[TRACE] voxcpm2_tts_first_chunk: latency={latency_ms:.1f}ms, '
                                f'text="{text[:30]}"'
                            )
                            first_chunk_logged = True
                        yield pcm_24k
                # 流末尾若仍有 1-3 byte residual，丢弃 (不足 1 个完整 24kHz sample)
                if leftover:
                    logger.debug(
                        f"[VoxCPM2-TTS] discarded {len(leftover)} trailing byte(s) at stream end"
                    )

        except asyncio.CancelledError:
            logger.info("[VoxCPM2-TTS] stream_tts cancelled")
            raise
        except Exception as e:
            logger.error(f"VoxCPM2-TTS streaming error: {e}")

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


class VoxCpm2TtsWsPipeline:
    """VoxCPM2 WebSocket streaming pipeline (character-by-character).

    Same protocol flow as TTSWebSocketPipeline, but:
    - session.config sends model="voxcpm2"
    - No language parameter
    - Supports ref_audio/ref_text for voice cloning
    - Output is 48kHz PCM16, needs 2:1 decimation to 24kHz

    Core flow:
    1. Connect to WS endpoint, send session.config
    2. As LLM produces text deltas, immediately send each delta via input.text
    3. When LLM finishes, send input.done
    4. Receive binary PCM chunks (48kHz S16_LE mono), decimate to 24kHz
    5. Yield 24kHz PCM chunks to upper layer for playback
    """

    def __init__(
        self,
        ws_url: str = "ws://localhost:8093/v1/audio/speech/stream",
        voice: str = "default",
        sample_rate: int = 24000,
        timeout_s: int = 15,
        api_key: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        app_config: object | None = None,
    ):
        self.ws_url = ws_url
        self.voice = voice
        self.sample_rate = sample_rate
        self.timeout_s = timeout_s
        self.api_key = api_key

        # Safety config for ref_audio resolution
        if app_config is not None:
            self._ref_audio_cfg = ref_audio_config_from_app_config(app_config)
        else:
            self._ref_audio_cfg = RefAudioConfig()

        # Voice cloning — resolve local file paths to base64 data URLs
        self.ref_audio = self._resolve_ref_audio(ref_audio)
        self.ref_text = ref_text

        self._ws = None
        self._connected = False
        self._input_finished = False
        self._session_done = False
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._error: str | None = None
        self._receive_task: asyncio.Task | None = None
        self._total_sentences = 0

    def _resolve_ref_audio(self, ref_audio: str | None) -> str | None:
        """Resolve ref_audio safely (delegates to VoxCpm2TtsPipeline logic)."""
        # Reuse the same safe resolution logic as the HTTP pipeline
        pipeline = VoxCpm2TtsPipeline.__new__(VoxCpm2TtsPipeline)
        pipeline._ref_audio_cfg = self._ref_audio_cfg
        return pipeline._resolve_ref_audio(ref_audio)

    async def connect(self):
        """Establish WebSocket connection and send session.config.

        Must be called before sending any text. This should be invoked
        at the same time as sending the ASR result to LLM, to overlap
        connection setup with LLM processing.
        """
        try:
            extra_headers = {}
            if self.api_key:
                extra_headers["Authorization"] = f"Bearer {self.api_key}"
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self.ws_url,
                    max_size=None,  # Allow large binary frames
                    ping_interval=20,
                    ping_timeout=10,
                    additional_headers=extra_headers if extra_headers else None,
                ),
                timeout=self.timeout_s,
            )
            self._connected = True

            # Send session.config as the first message
            config_msg = {
                "type": "session.config",
                "model": "voxcpm2",
                "voice": self.voice,
                "response_format": "pcm",
                "stream_audio": True,
                "split_granularity": "sentence",
            }
            # Voice cloning parameters
            if self.ref_audio:
                config_msg["ref_audio"] = self.ref_audio
                if self.ref_text:
                    config_msg["ref_text"] = self.ref_text

            await self._ws.send(json.dumps(config_msg))
            logger.info(f'[TRACE] voxcpm2_ws_connected: voice={self.voice}, url={self.ws_url}, '
                        f'clone={self.ref_audio is not None}')

            # Start background receiver task
            self._receive_task = asyncio.create_task(self._receive_loop())

        except Exception as e:
            logger.error(f"VoxCPM2 WS connect error: {e}")
            self._connected = False
            raise

    async def send_text_delta(self, text: str):
        """Send incremental text to TTS.

        Each LLM text delta is immediately forwarded to TTS.

        Raises:
            RuntimeError: If the WS connection is broken or send fails.
        """
        if not self._connected or self._ws is None:
            raise RuntimeError("VoxCPM2 WS not connected, cannot send text delta")

        try:
            msg = {"type": "input.text", "text": text}
            await self._ws.send(json.dumps(msg))
        except Exception as e:
            logger.error(f"VoxCPM2 WS send_text_delta error: {e}")
            self._error = str(e)
            raise RuntimeError(f"VoxCPM2 WS send_text_delta failed: {e}") from e

    async def finish_input(self):
        """Signal that all text has been sent (input.done).

        Raises:
            RuntimeError: If the WS connection is broken or send fails.
        """
        if not self._connected or self._ws is None:
            raise RuntimeError("VoxCPM2 WS not connected, cannot finish input")

        try:
            msg = {"type": "input.done"}
            await self._ws.send(json.dumps(msg))
            self._input_finished = True
            logger.info('[TRACE] voxcpm2_ws_input_done: all text sent to TTS')
        except Exception as e:
            logger.error(f"VoxCPM2 WS finish_input error: {e}")
            self._error = str(e)
            raise RuntimeError(f"VoxCPM2 WS finish_input failed: {e}") from e

    async def _receive_loop(self):
        """Background task to receive messages from VoxCPM2 WS server.

        Puts decimated PCM chunks (24kHz) into the audio queue and handles
        control messages. Binary frames are 48kHz PCM16 and are decimated
        to 24kHz before being queued.
        """
        # ---- 跨 frame PCM16 + decimation phase 对齐 (P0 fix B 方案) ----
        # 与 HTTP 路径同理：_resample_48k_to_24k 是无状态函数，跨 frame 调用
        # 必须保证每次输入 (1) 字节为偶数 (2) sample 数为偶数。统一缓冲到 4
        # 字节 (2 sample) 边界，避免奇数字节 trim 错位 + decimation phase 翻转。
        leftover = b""
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    # Binary PCM chunk (S16_LE 48kHz mono from VoxCPM2)
                    # Decimate 48kHz -> 24kHz before queueing
                    buf = leftover + message
                    rem = len(buf) % 4
                    if rem:
                        leftover = buf[-rem:]
                        buf = buf[:-rem]
                    else:
                        leftover = b""
                    if not buf:
                        continue
                    pcm_24k = VoxCpm2TtsPipeline._resample_48k_to_24k(buf)
                    if pcm_24k:
                        await self._audio_queue.put(pcm_24k)
                elif isinstance(message, str):
                    # JSON control message
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")

                        if msg_type == "audio.start":
                            self._total_sentences = max(
                                self._total_sentences,
                                data.get("sentence_index", 0) + 1
                            )
                            logger.debug(
                                f"VoxCPM2 WS audio.start: sentence_index={data.get('sentence_index')}, "
                                f"text=\"{data.get('sentence_text', '')[:30]}\""
                            )

                        elif msg_type == "audio.done":
                            logger.debug(
                                f"VoxCPM2 WS audio.done: sentence_index={data.get('sentence_index')}, "
                                f"total_bytes={data.get('total_bytes', 0)}"
                            )

                        elif msg_type == "session.done":
                            self._session_done = True
                            self._total_sentences = data.get("total_sentences", self._total_sentences)
                            await self._audio_queue.put(None)  # Sentinel: no more audio
                            logger.info(
                                f'[TRACE] voxcpm2_ws_session_done: total_sentences={self._total_sentences}'
                            )
                            break

                        elif msg_type == "error":
                            error_msg = data.get("message", "Unknown error")
                            logger.error(f"VoxCPM2 WS server error: {error_msg}")
                            self._error = error_msg
                            await self._audio_queue.put(None)  # Sentinel
                            break

                    except json.JSONDecodeError:
                        logger.warning(f"VoxCPM2 WS invalid JSON: {message[:200]}")

        except websockets.exceptions.ConnectionClosed as e:
            if not self._session_done:
                logger.warning(f"VoxCPM2 WS connection closed: code={e.code}, reason={e.reason}")
                if not self._error:
                    self._error = f"Connection closed: {e.code}"
                await self._audio_queue.put(None)  # Ensure consumer doesn't hang
        except Exception as e:
            logger.error(f"VoxCPM2 WS receive error: {e}")
            self._error = str(e)
            await self._audio_queue.put(None)
        finally:
            # 流末尾若仍有 1-3 byte residual，丢弃 (不足 1 个完整 24kHz sample)
            if leftover:
                logger.debug(
                    f"[VoxCPM2-WS] discarded {len(leftover)} trailing byte(s) at stream end"
                )

    async def receive_audio(self) -> AsyncIterator[bytes]:
        """Async iterator that yields PCM chunks from VoxCPM2 TTS.

        Yields raw PCM data (S16_LE 24kHz mono), already decimated from 48kHz.
        Returns when session.done is received or an error occurs.
        """
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                # Sentinel: session done or error
                break
            yield chunk

    async def close(self):
        """Close the WebSocket connection and clean up."""
        self._connected = False

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        # Drain any remaining items in the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info('[TRACE] voxcpm2_ws_closed')

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    @property
    def total_sentences(self) -> int:
        return self._total_sentences
