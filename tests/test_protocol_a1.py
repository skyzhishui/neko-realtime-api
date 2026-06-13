"""Tests for A1 protocol additions: client-compatibility events.

Verifies:
- send_transcript_done MUST carry the ``transcript`` field
  (neko-fork client reads ``event["transcript"]``)
- send_input_audio_buffer_committed exists and emits the right event type
- send_response_done propagates the configured model
  (neko-fork TokenTracker reads ``response.model``)
- send_error response-conflict messages contain ``response_already_active``
  for content-fallback routing
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from server.protocol import ProtocolAdapter


class _FakeWS:
    """Minimal websocket stub capturing every send_text payload."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture()
def adapter() -> tuple[ProtocolAdapter, _FakeWS]:
    ws = _FakeWS()
    return ProtocolAdapter(ws), ws  # type: ignore[arg-type]


def test_transcript_done_carries_transcript(adapter):
    pa, ws = adapter
    _run(pa.send_transcript_done("hello world"))
    assert len(ws.sent) == 1
    ev = ws.sent[0]
    assert ev["type"] == "response.output_audio_transcript.done"
    # Critical: client (omni_realtime_client.py) reads event["transcript"]
    # at the streaming-fallback branch (line 2793).
    assert ev["transcript"] == "hello world"


def test_transcript_done_default_empty(adapter):
    pa, ws = adapter
    _run(pa.send_transcript_done())
    assert ws.sent[0]["transcript"] == ""


def test_input_audio_buffer_committed_basic(adapter):
    pa, ws = adapter
    _run(pa.send_input_audio_buffer_committed())
    assert ws.sent[0] == {"type": "input_audio_buffer.committed"}


def test_input_audio_buffer_committed_with_ids(adapter):
    pa, ws = adapter
    _run(pa.send_input_audio_buffer_committed(item_id="item_5", previous_item_id="item_4"))
    assert ws.sent[0] == {
        "type": "input_audio_buffer.committed",
        "item_id": "item_5",
        "previous_item_id": "item_4",
    }


def test_response_done_uses_provided_model(adapter):
    pa, ws = adapter
    _run(pa.send_response_done("resp_0", status="completed",
                                usage={"total_tokens": 10}, model="gpt-4o-realtime"))
    response = ws.sent[0]["response"]
    # Critical: neko-fork TokenTracker keys per-model usage on response.model.
    assert response["model"] == "gpt-4o-realtime"
    assert response["status"] == "completed"
    assert response["usage"]["total_tokens"] == 10


def test_response_done_default_model(adapter):
    pa, ws = adapter
    _run(pa.send_response_done("resp_0"))
    # Default must NOT be the previous hardcoded "qwen-realtime".
    assert ws.sent[0]["response"]["model"] == "local-qwen-omni"


def test_error_object_format(adapter):
    pa, ws = adapter
    _run(pa.send_error("something broke", error_type="server_error",
                        code="internal", event_id="ev_42"))
    ev = ws.sent[0]
    assert ev["type"] == "error"
    # OpenAI Realtime spec: error MUST be an object, not a string.
    assert isinstance(ev["error"], dict)
    assert ev["error"]["type"] == "server_error"
    assert ev["error"]["code"] == "internal"
    assert ev["error"]["event_id"] == "ev_42"
    assert ev["error"]["message"] == "something broke"


def test_error_response_conflict_keyword(adapter):
    """Response-conflict errors must contain ``response_already_active`` for
    client-side content fallback (omni_realtime_client.py line 1985-1990)."""
    pa, ws = adapter
    _run(pa.send_error("response_already_active: a response is already in progress",
                        event_id="ev_inj_1"))
    ev = ws.sent[0]
    assert "response_already_active" in ev["error"]["message"]
    assert ev["error"]["event_id"] == "ev_inj_1"


def test_error_prefix_routing_invalid_request(adapter):
    pa, ws = adapter
    _run(pa.send_error("invalid_request_error: bad input"))
    ev = ws.sent[0]
    assert ev["error"]["type"] == "invalid_request_error"
    assert ev["error"]["message"] == "bad input"


def test_session_created_has_full_object(adapter):
    pa, ws = adapter
    cfg = SimpleNamespace(
        _model="local-qwen-omni",
        modalities=["text", "audio"],
        voice="hutao",
        input_audio_format="pcm16",
        output_audio_format="pcm16",
        instructions="be brief",
        turn_detection={"type": "server_vad"},
        tools=[],
        temperature=0.7,
    )
    _run(pa.send_session_created(cfg))
    ev = ws.sent[0]
    assert ev["type"] == "session.created"
    s = ev["session"]
    assert s["object"] == "realtime.session"
    assert s["model"] == "local-qwen-omni"
    assert s["voice"] == "hutao"
    assert s["modalities"] == ["text", "audio"]
