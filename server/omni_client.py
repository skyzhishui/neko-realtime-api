"""Omni Audio Client - HTTP streaming client for Qwen3-Omni."""
import aiohttp
import json
import logging
import time
from typing import AsyncIterator

logger = logging.getLogger("realtime-server")


class OmniAudioClient:
    """HTTP streaming client for Qwen3-Omni.
    
    Endpoint: POST http://localhost:8000/v1/chat/completions
    """

    def __init__(self, base_url: str = "http://localhost:8000", model: str = "Qwen3-Omni"):
        self.base_url = base_url
        self.model = model
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
    ) -> AsyncIterator[str]:
        """Stream chat completion, yielding text deltas.
        
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
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
        }

        # [TRACE] 记录请求发送时间，用于计算首 token 延迟
        t_request_sent = time.time()
        first_token_logged = False

        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
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
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            # [TRACE] 首 token 埋点
                            if not first_token_logged:
                                first_token_logged = True
                                latency_ms = (time.time() - t_request_sent) * 1000
                                first_content_preview = content[:20].replace("\n", " ")
                                logger.info(f'[TRACE] llm_first_token: latency={latency_ms:.0f}ms, first_content="{first_content_preview}"')
                            yield content
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
