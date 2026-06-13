"""Protocol Adapter - encapsulates Server → Client event sending."""
import json
import base64
import logging
import uuid
from fastapi import WebSocket

logger = logging.getLogger("realtime-server")


def _build_session_object(session_config) -> dict:
    """Build an OpenAI-Realtime-API-compliant session object.
    
    Args:
        session_config: A SessionConfig instance (from session.py).
    """
    return {
        "id": f"sess_{uuid.uuid4().hex[:24]}",
        "object": "realtime.session",
        "model": getattr(session_config, "_model", "local-qwen-omni"),
        "modalities": getattr(session_config, "modalities", ["text", "audio"]),
        "voice": getattr(session_config, "voice", "Vivian"),
        "input_audio_format": getattr(session_config, "input_audio_format", "pcm16"),
        "output_audio_format": getattr(session_config, "output_audio_format", "pcm16"),
        "instructions": getattr(session_config, "instructions", ""),
        "turn_detection": getattr(session_config, "turn_detection", {}),
        "tools": getattr(session_config, "tools", []) or [],
        "tool_choice": "auto",
        "temperature": getattr(session_config, "temperature", 0.7),
        "max_response_output_tokens": "inf",
    }


def _build_usage_object(usage: dict | None = None) -> dict:
    """Build an OpenAI-Realtime-API-compliant usage object.
    
    If usage is provided with partial keys, fill in missing ones with zeros.
    If usage is None, return a full zero-usage object.
    """
    template = {
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "input_token_details": {
            "text_tokens": 0,
            "audio_tokens": 0,
            "cached_tokens": 0,
        },
        "output_token_details": {
            "text_tokens": 0,
            "audio_tokens": 0,
        },
    }
    if usage is None:
        return template
    result = template.copy()
    for key in ("total_tokens", "input_tokens", "output_tokens"):
        if key in usage:
            result[key] = usage[key]
    if "input_token_details" in usage and isinstance(usage["input_token_details"], dict):
        for k in ("text_tokens", "audio_tokens", "cached_tokens"):
            if k in usage["input_token_details"]:
                result["input_token_details"][k] = usage["input_token_details"][k]
    if "output_token_details" in usage and isinstance(usage["output_token_details"], dict):
        for k in ("text_tokens", "audio_tokens"):
            if k in usage["output_token_details"]:
                result["output_token_details"][k] = usage["output_token_details"][k]
    return result


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

    async def send_error(self, msg: str, error_type: str = "invalid_request_error",
                          code: str | None = None, event_id: str | None = None,
                          param: str | None = None):
        """Send an error event in OpenAI Realtime API format.

        The OpenAI spec requires `error` to be an object, not a string:
        ``{"type": "error", "error": {"type", "code", "message", "param", "event_id"}}``.

        Args:
            msg: Human-readable error message.
            error_type: One of "invalid_request_error", "server_error", etc.
                Defaults to "invalid_request_error" since most current callers
                pass user-input-validation errors.
            code: Optional machine-readable error code (e.g. "rate_limit_exceeded").
            event_id: Optional ID of the client event that caused the error.
            param: Optional parameter name that caused the error.
        """
        # Heuristic: if msg starts with "invalid_request_error:" or "server_error:",
        # honor the prefix instead of the default error_type.
        if isinstance(msg, str):
            for prefix, etype in (("invalid_request_error:", "invalid_request_error"),
                                   ("server_error:", "server_error")):
                if msg.startswith(prefix):
                    error_type = etype
                    msg = msg[len(prefix):].strip()
                    break
        error_obj: dict = {"type": error_type, "message": msg}
        if code is not None:
            error_obj["code"] = code
        if event_id is not None:
            error_obj["event_id"] = event_id
        if param is not None:
            error_obj["param"] = param
        await self.send({"type": "error", "error": error_obj})

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
        await self.send({"type": "response.output_audio.delta", "delta": b64})

    async def send_transcript_delta(self, text: str):
        """Send transcript delta."""
        await self.send({"type": "response.output_audio_transcript.delta", "delta": text})

    async def send_text_delta(self, text: str):
        """Send text delta."""
        await self.send({"type": "response.output_text.delta", "delta": text})

    async def send_transcript_done(self):
        await self.send({"type": "response.output_audio_transcript.done"})

    async def send_response_done(self, resp_id: str, status: str = "completed",
                                   status_details: dict | None = None, usage: dict | None = None):
        """Send response.done event with OpenAI-Realtime-API-compliant payload.
        
        Args:
            resp_id: Response ID (e.g. "resp_0").
            status: One of "completed", "cancelled", "incomplete", "failed".
            status_details: Optional status details object.
            usage: Optional usage dict; missing keys will be filled with zeros.
        """
        response_obj = {
            "id": resp_id,
            "object": "realtime.response",
            "model": "qwen-realtime",
            "status": status,
            "usage": _build_usage_object(usage),
        }
        if status_details is not None:
            response_obj["status_details"] = status_details
        await self.send({
            "type": "response.done",
            "response": response_obj,
        })

    async def send_session_created(self, session_config=None):
        """Send session.created event to client with full session object."""
        session_obj = _build_session_object(session_config) if session_config else _build_session_object(type("SC", (), {})())
        await self.send({"type": "session.created", "session": session_obj})

    async def send_session_updated(self, session_config=None):
        """Send session.updated event to client with full session object."""
        session_obj = _build_session_object(session_config) if session_config else _build_session_object(type("SC", (), {})())
        await self.send({"type": "session.updated", "session": session_obj})

    # ── Tool calling events ─────────────────────────────────────────

    async def send_function_call_item_added(self, call_id: str, name: str, resp_id: str, output_index: int):
        """Send response.output_item.added for a function_call item."""
        await self.send({
            "type": "response.output_item.added",
            "response_id": resp_id,
            "output_index": output_index,
            "item": {
                "type": "function_call",
                "id": f"fc_{call_id}",
                "call_id": call_id,
                "name": name,
                "arguments": "",
            },
        })

    async def send_function_call_arguments_delta(self, call_id: str, name: str, delta: str):
        """Stream function call arguments incrementally."""
        await self.send({
            "type": "response.function_call_arguments.delta",
            "call_id": call_id,
            "name": name,
            "delta": delta,
        })

    async def send_function_call_arguments_done(self, call_id: str, name: str, arguments: str, resp_id: str, output_index: int):
        """Signal that function call arguments are complete."""
        await self.send({
            "type": "response.function_call_arguments.done",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "response_id": resp_id,
            "output_index": output_index,
        })
