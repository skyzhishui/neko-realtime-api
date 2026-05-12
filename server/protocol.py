"""Protocol Adapter - encapsulates Server → Client event sending."""
import json
import base64
import logging
from fastapi import WebSocket

logger = logging.getLogger("realtime-server")


class ProtocolAdapter:
    """Encapsulates Server → Client event sending with Qwen-style Realtime protocol."""

    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self._response_counter = 0

    async def send(self, event: dict):
        """Send raw event as JSON."""
        try:
            await self.ws.send_text(json.dumps(event, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"Failed to send WebSocket event: {e}")

    async def send_error(self, msg: str):
        await self.send({"type": "error", "error": msg})

    async def send_speech_started(self):
        await self.send({"type": "input_audio_buffer.speech_started"})

    async def send_speech_stopped(self):
        await self.send({"type": "input_audio_buffer.speech_stopped"})

    async def send_input_transcript(self, transcript: str):
        await self.send({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": transcript,
        })

    async def send_response_created(self) -> str:
        resp_id = f"resp_{self._response_counter}"
        self._response_counter += 1
        await self.send({
            "type": "response.created",
            "response": {"id": resp_id},
        })
        await self.send({
            "type": "response.output_item.added",
            "item": {"id": f"item_{resp_id}"},
        })
        return resp_id

    async def send_audio_delta(self, pcm_bytes: bytes):
        """Send audio delta. Output is 24kHz PCM16."""
        b64 = base64.b64encode(pcm_bytes).decode()
        await self.send({"type": "response.audio.delta", "delta": b64})

    async def send_transcript_delta(self, text: str):
        """Send transcript delta."""
        await self.send({"type": "response.audio_transcript.delta", "delta": text})

    async def send_text_delta(self, text: str):
        """Send text delta."""
        await self.send({"type": "response.text.delta", "delta": text})

    async def send_transcript_done(self):
        await self.send({"type": "response.audio_transcript.done"})

    async def send_response_done(self, resp_id: str, usage: dict | None = None):
        await self.send({
            "type": "response.done",
            "response": {
                "id": resp_id,
                "model": "qwen-realtime",
                "usage": usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
        })
