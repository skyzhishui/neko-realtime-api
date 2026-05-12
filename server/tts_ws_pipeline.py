"""TTS WebSocket Pipeline - Character-by-character streaming TTS via WebSocket."""
import asyncio
import json
import time
import logging
from typing import AsyncIterator

import websockets

logger = logging.getLogger("realtime-server")


class TTSWebSocketPipeline:
    """Qwen3-TTS WebSocket streaming pipeline.
    
    Core flow (character-by-character streaming):
    1. Connect to WS endpoint, send session.config
    2. As LLM produces text deltas, immediately send each delta via input.text
    3. When LLM finishes, send input.done
    4. Receive binary PCM chunks (S16_LE 24kHz mono) asynchronously
    5. Yield PCM chunks to upper layer for playback
    
    This pipeline eliminates the sentence-splitting bottleneck of the HTTP mode,
    achieving much lower first-audio latency by starting TTS as soon as the
    first LLM token arrives.
    """

    def __init__(
        self,
        ws_url: str = "ws://localhost:8091/v1/audio/speech/stream",
        voice: str = "Vivian",
        sample_rate: int = 24000,
        language: str = "Chinese",
        timeout_s: int = 15,
    ):
        self.ws_url = ws_url
        self.voice = voice
        self.sample_rate = sample_rate
        self.language = language
        self.timeout_s = timeout_s

        self._ws = None
        self._connected = False
        self._input_finished = False
        self._session_done = False
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._error: str | None = None
        self._receive_task: asyncio.Task | None = None
        self._total_sentences = 0

    async def connect(self):
        """Establish WebSocket connection and send session.config.
        
        Must be called before sending any text. This should be invoked
        at the same time as sending the ASR result to LLM, to overlap
        connection setup with LLM processing.
        """
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self.ws_url,
                    max_size=None,  # Allow large binary frames
                    ping_interval=20,
                    ping_timeout=10,
                ),
                timeout=self.timeout_s,
            )
            self._connected = True

            # Send session.config as the first message
            config_msg = {
                "type": "session.config",
                "voice": self.voice,
                "response_format": "pcm",
                "stream_audio": True,
                "sample_rate": self.sample_rate,
                "language": self.language,
            }
            await self._ws.send(json.dumps(config_msg))
            logger.info(f'[TRACE] tts_ws_connected: voice={self.voice}, url={self.ws_url}')

            # Start background receiver task
            self._receive_task = asyncio.create_task(self._receive_loop())

        except Exception as e:
            logger.error(f"TTS WS connect error: {e}")
            self._connected = False
            raise

    async def send_text_delta(self, text: str):
        """Send incremental text to TTS.
        
        Each LLM text delta is immediately forwarded to TTS.
        
        Raises:
            RuntimeError: If the WS connection is broken or send fails.
        """
        if not self._connected or self._ws is None:
            raise RuntimeError("TTS WS not connected, cannot send text delta")

        try:
            msg = {"type": "input.text", "text": text}
            await self._ws.send(json.dumps(msg))
        except Exception as e:
            logger.error(f"TTS WS send_text_delta error: {e}")
            self._error = str(e)
            raise RuntimeError(f"TTS WS send_text_delta failed: {e}") from e

    async def finish_input(self):
        """Signal that all text has been sent (input.done).
        
        Raises:
            RuntimeError: If the WS connection is broken or send fails.
        """
        if not self._connected or self._ws is None:
            raise RuntimeError("TTS WS not connected, cannot finish input")

        try:
            msg = {"type": "input.done"}
            await self._ws.send(json.dumps(msg))
            self._input_finished = True
            logger.info('[TRACE] tts_ws_input_done: all text sent to TTS')
        except Exception as e:
            logger.error(f"TTS WS finish_input error: {e}")
            self._error = str(e)
            raise RuntimeError(f"TTS WS finish_input failed: {e}") from e

    async def _receive_loop(self):
        """Background task to receive messages from TTS WS server.
        
        Puts PCM chunks into the audio queue and handles control messages.
        """
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    # Binary PCM chunk (S16_LE 24kHz mono, no WAV header)
                    await self._audio_queue.put(message)
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
                                f"TTS WS audio.start: sentence_index={data.get('sentence_index')}, "
                                f"text=\"{data.get('sentence_text', '')[:30]}\""
                            )

                        elif msg_type == "audio.done":
                            logger.debug(
                                f"TTS WS audio.done: sentence_index={data.get('sentence_index')}, "
                                f"total_bytes={data.get('total_bytes', 0)}"
                            )

                        elif msg_type == "session.done":
                            self._session_done = True
                            self._total_sentences = data.get("total_sentences", self._total_sentences)
                            await self._audio_queue.put(None)  # Sentinel: no more audio
                            logger.info(
                                f'[TRACE] tts_ws_session_done: total_sentences={self._total_sentences}'
                            )
                            break

                        elif msg_type == "error":
                            error_msg = data.get("message", "Unknown error")
                            logger.error(f"TTS WS server error: {error_msg}")
                            self._error = error_msg
                            await self._audio_queue.put(None)  # Sentinel
                            break

                    except json.JSONDecodeError:
                        logger.warning(f"TTS WS invalid JSON: {message[:200]}")

        except websockets.exceptions.ConnectionClosed as e:
            if not self._session_done:
                logger.warning(f"TTS WS connection closed: code={e.code}, reason={e.reason}")
                if not self._error:
                    self._error = f"Connection closed: {e.code}"
                await self._audio_queue.put(None)  # Ensure consumer doesn't hang
        except Exception as e:
            logger.error(f"TTS WS receive error: {e}")
            self._error = str(e)
            await self._audio_queue.put(None)

    async def receive_audio(self) -> AsyncIterator[bytes]:
        """Async iterator that yields PCM chunks from TTS.
        
        Yields raw PCM data (S16_LE 24kHz mono).
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

        logger.info('[TRACE] tts_ws_closed')

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    @property
    def total_sentences(self) -> int:
        return self._total_sentences
