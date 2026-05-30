"""Omni Audio Client - HTTP streaming client for Qwen3-Omni."""
import aiohttp
import json
import logging
import time
from typing import AsyncIterator, Literal, TypedDict

logger = logging.getLogger("realtime-server")


# ── Typed stream events ─────────────────────────────────────────────


class TextDeltaEvent(TypedDict):
    type: Literal["text"]
    delta: str


class ToolCallDeltaEvent(TypedDict):
    type: Literal["tool_call"]
    index: int
    call_id: str
    name: str | None  # only present on first chunk for this index
    arguments_delta: str


class ToolCallDoneEvent(TypedDict):
    type: Literal["tool_call_done"]
    index: int
    call_id: str
    name: str
    arguments: str  # accumulated full arguments


class FinishEvent(TypedDict):
    type: Literal["finish"]
    finish_reason: str  # "stop", "tool_calls", etc.


ChatStreamEvent = TextDeltaEvent | ToolCallDeltaEvent | ToolCallDoneEvent | FinishEvent


# ── Client ───────────────────────────────────────────────────────────


class OmniAudioClient:
    """HTTP streaming client for Qwen3-Omni.
    
    Endpoint: POST http://localhost:8000/v1/chat/completions
    """

    def __init__(self, base_url: str = "http://localhost:8000", model: str = "Qwen3-Omni", api_key: str | None = None):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=10, keepalive_timeout=30)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def stream_chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        max_tokens: int = 4096,
        timeout_s: int = 30,
        tools: list[dict] | None = None,
        enable_search: bool = False,
    ) -> AsyncIterator[ChatStreamEvent]:
        """Stream chat completion, yielding typed events.
        
        Events:
            TextDeltaEvent      — text content delta
            ToolCallDeltaEvent  — function call arguments fragment
            ToolCallDoneEvent   — function call arguments complete
            FinishEvent         — stream finished (finish_reason)
        
        Messages format:
        [
            {"role": "system", "content": "instructions..."},
            {"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": "<wav_b64>", "format": "wav"}},
                {"type": "image_url", "image_url": "data:image/jpeg;base64,..."},
                {"type": "text", "text": "描述这张图片"}
            ]}
        ]
        """
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if enable_search:
            payload["enable_search"] = True

        # [TRACE] 记录请求发送时间，用于计算首 token 延迟
        t_request_sent = time.time()
        first_token_logged = False

        # Accumulate tool call arguments per index
        tool_call_accumulators: dict[int, dict] = {}  # index -> {call_id, name, arguments}
        finish_reason = "stop"

        session = await self._get_session()
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Omni API error {resp.status}: {error_text[:500]}")
                    raise Exception(f"Omni API error {resp.status}: {error_text[:200]}")
                
                async for line in resp.content:
                    line = line.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        # Only emit tool_call_done here if we haven't already
                        # (i.e. finish_reason was NOT "tool_calls" — those were
                        #  emitted inline when finish_reason was detected).
                        # Also only emit FinishEvent if not already emitted.
                        if finish_reason != "tool_calls":
                            for idx in sorted(tool_call_accumulators.keys()):
                                tc = tool_call_accumulators[idx]
                                yield ToolCallDoneEvent(
                                    type="tool_call_done",
                                    index=idx,
                                    call_id=tc["call_id"],
                                    name=tc["name"],
                                    arguments=tc["arguments"],
                                )
                            yield FinishEvent(type="finish", finish_reason=finish_reason)
                        # else: ToolCallDoneEvent + FinishEvent already emitted
                        # when finish_reason=="tool_calls" was detected inline.
                        break
                    try:
                        chunk = json.loads(data)
                        choice = chunk.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        fr = choice.get("finish_reason")
                        if fr:
                            finish_reason = fr

                        # Handle text content
                        content = delta.get("content", "")
                        if content:
                            if not first_token_logged:
                                first_token_logged = True
                                latency_ms = (time.time() - t_request_sent) * 1000
                                first_content_preview = content[:20].replace("\n", " ")
                                logger.info(f'[TRACE] llm_first_token: latency={latency_ms:.0f}ms, first_content="{first_content_preview}"')
                            yield TextDeltaEvent(type="text", delta=content)

                        # Handle tool calls
                        tool_calls = delta.get("tool_calls")
                        if tool_calls:
                            for tc in tool_calls:
                                idx = tc.get("index", 0)
                                func = tc.get("function", {})

                                if idx not in tool_call_accumulators:
                                    tool_call_accumulators[idx] = {
                                        "call_id": tc.get("id", f"call_{idx}"),
                                        "name": func.get("name", ""),
                                        "arguments": "",
                                    }
                                    # First chunk carries id + name (arguments may be empty)
                                    # Yield immediately so session layer gets the name
                                    yield ToolCallDeltaEvent(
                                        type="tool_call",
                                        index=idx,
                                        call_id=tool_call_accumulators[idx]["call_id"],
                                        name=func.get("name"),  # present on first chunk
                                        arguments_delta="",
                                    )

                                args_delta = func.get("arguments", "")
                                if args_delta:
                                    tool_call_accumulators[idx]["arguments"] += args_delta
                                    yield ToolCallDeltaEvent(
                                        type="tool_call",
                                        index=idx,
                                        call_id=tool_call_accumulators[idx]["call_id"],
                                        name=None,  # only on first chunk
                                        arguments_delta=args_delta,
                                    )

                        # If finish_reason indicates tool_calls, emit done events
                        if fr == "tool_calls":
                            for idx in sorted(tool_call_accumulators.keys()):
                                tc = tool_call_accumulators[idx]
                                yield ToolCallDoneEvent(
                                    type="tool_call_done",
                                    index=idx,
                                    call_id=tc["call_id"],
                                    name=tc["name"],
                                    arguments=tc["arguments"],
                                )
                            yield FinishEvent(type="finish", finish_reason=fr)

                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse Omni SSE chunk: {data[:200]}")
                        continue
        except aiohttp.ClientError as e:
            logger.error(f"Omni client error: {e}")
            raise

    def build_audio_message(self, wav_b64: str) -> dict:
        """Build audio input message (recommended format)."""
        return {
            "type": "input_audio",
            "input_audio": {
                "data": wav_b64,
                "format": "wav",
            },
        }

    def build_image_message(self, image_b64: str) -> dict:
        """Build image input message."""
        return {
            "type": "image_url",
            #"image_url": f"data:image/jpeg;base64,{image_b64}",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"
            },
        }

    def build_text_message(self, text: str) -> dict:
        """Build text input message."""
        return {
            "type": "text",
            "text": text,
        }

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
