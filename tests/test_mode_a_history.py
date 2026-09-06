"""Mode A 转录历史回归测试。

Background:
- Bug（CodeRabbit PR #1 review 指出）：``_process_mode_a`` 先把本次
  transcript append 进 ``self.conversation``，再调用
  ``_build_omni_messages``。由于 ``_build_base_messages`` 会展开
  ``self.conversation[-20:]``，且 ``_build_omni_messages`` 又把同一句
  transcript 作为本轮 user 消息追加，同一句转录会在发给 LLM 的
  messages 中出现两次。
- Fix：先基于现有历史构建 messages，再 append 本次转录。

本套件用桩替换 ASR/LLM/TTS，仅真实运行 ``_process_mode_a`` /
``_build_omni_messages`` / ``_build_base_messages``，断言：
1. 发给 LLM 的 messages 中本次转录恰好出现一次；
2. 会话历史仍按序保留用户转录与助手回复。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from server.session import RealtimeSession


# --------------------------------------------------------------------------- #
# Fakes / Helpers
# --------------------------------------------------------------------------- #


class _FakeProtocol:
    """捕获 send_* 调用的最小协议桩。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def send_input_transcript(self, transcript: str) -> None:
        self.calls.append("input_transcript")

    async def send_response_created(self) -> str:
        self.calls.append("response_created")
        return "resp_1"

    async def send_transcript_done(self, transcript: str = "") -> None:
        self.calls.append("transcript_done")

    async def send_response_done(self, resp_id: str, **kw) -> None:
        self.calls.append("response_done")

    async def send_error(self, message: str) -> None:
        self.calls.append("error")


class _FakeLocalASR:
    """固定返回一句转录文本，无情绪/事件。"""

    async def transcribe_with_details(self, pcm: bytes, **kw):
        return ("今天天气怎么样", "", "")


class _FakeOmniClient:
    """提供 build_*_message 所需的最小方法。"""

    @staticmethod
    def build_text_message(text: str) -> dict:
        return {"type": "text", "text": text}

    @staticmethod
    def build_audio_message(b64: str) -> dict:
        return {"type": "input_audio", "input_audio": {"data": b64}}

    @staticmethod
    def build_image_message(b64: str) -> dict:
        return {"type": "input_image", "image_url": {"url": b64}}


class _FakeConfig:
    """覆盖 _process_mode_a 读取到的配置键。"""

    def get(self, *path, default=None):
        if path == ("vad", "sample_rate"):
            return 16000
        if path == ("services", "asr", "language"):
            return "zh"
        return default


def _make_session(captured_messages: list, tools: list[dict] | None = None):
    """构建最小 RealtimeSession 桩并绑定真实方法。

    Args:
        captured_messages: 捕获发给 LLM 的 messages 的列表。
        tools: 透传给 session_config.tools 的工具定义（None=关闭
            tool loop），用于覆盖 tools 开启时的会话历史行为。
    """
    fake = SimpleNamespace()
    fake.protocol = _FakeProtocol()
    fake.local_asr = _FakeLocalASR()
    fake.asr_client = None
    fake.config = _FakeConfig()
    fake.omni_client = _FakeOmniClient()
    fake.conversation = [{"role": "user", "content": "上一轮的提问"}]
    fake.session_config = SimpleNamespace(
        tools=tools,
        instructions="你是一个助手",
        _model="test-model",
    )
    fake.interruption = SimpleNamespace(set_generating=lambda flag: None)
    fake._current_resp_id = None

    async def _capture_llm_to_tts(messages, resp_id):
        captured_messages.append(messages)
        return "今天天气晴朗"

    fake._stream_llm_to_tts = _capture_llm_to_tts

    async def _noop_drain():
        return None

    fake._await_audio_drain = _noop_drain

    async def _noop_cleanup():
        return None

    fake._cleanup_audio_tasks_on_error = _noop_cleanup

    # 通过描述符协议绑定真实方法
    fake._process_mode_a = RealtimeSession._process_mode_a.__get__(fake, type(fake))
    fake._build_omni_messages = (
        RealtimeSession._build_omni_messages.__get__(fake, type(fake))
    )
    fake._build_base_messages = (
        RealtimeSession._build_base_messages.__get__(fake, type(fake))
    )
    return fake


def _flatten_messages(messages: list) -> str:
    """把 messages 里所有文本内容拍平成一个字符串，便于计数。"""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for piece in content:
                if isinstance(piece, dict) and piece.get("text"):
                    parts.append(piece["text"])
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_current_transcript_appears_once_in_llm_messages():
    """本次转录在发给 LLM 的 messages 中必须恰好出现一次。

    回归场景：修复前先 append 再 build，``_build_base_messages`` 展开
    历史时会带上刚 append 的同一句转录，加上 ``_build_omni_messages``
    追加的本轮 user 消息，出现两次。
    """
    captured: list = []
    fake = _make_session(captured)

    await fake._process_mode_a(b"\x00" * 3200, duration_ms=100.0)

    assert len(captured) == 1, "LLM 应恰好被调用一次"
    flat = _flatten_messages(captured[0])
    assert flat.count("今天天气怎么样") == 1, (
        f"本次转录不应重复出现在 LLM messages 中: {flat}"
    )
    # 历史消息仍要随 messages 一起发给 LLM（上下文连续性）
    assert "上一轮的提问" in flat


@pytest.mark.asyncio
async def test_conversation_history_order_after_mode_a():
    """Mode A 结束后历史按序保留：旧消息 → 本次用户转录 → 助手回复。"""
    captured: list = []
    fake = _make_session(captured)

    await fake._process_mode_a(b"\x00" * 3200, duration_ms=100.0)

    assert fake.conversation == [
        {"role": "user", "content": "上一轮的提问"},
        {"role": "user", "content": "今天天气怎么样"},
        {"role": "assistant", "content": "今天天气晴朗"},
    ]


@pytest.mark.asyncio
async def test_transcript_retained_with_tools_enabled():
    """tools 开启时本次用户转录仍必须进入会话历史。

    回归场景：曾有 ``if transcript and not self.session_config.tools``
    的守卫，导致 tool loop 会话的历史里缺失所有用户发言
    （tool loop 只回写 assistant/tool 消息）。本用例在 tools 开启
    且 tool loop 正常接管 assistant 回写的情况下，断言当前用户
    转录按序保留。
    """
    captured: list = []
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询天气",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    fake = _make_session(captured, tools=tools)

    await fake._process_mode_a(b"\x00" * 3200, duration_ms=100.0)

    # 本次转录恰好出现一次（tools 路径同样不得重复）
    flat = _flatten_messages(captured[0])
    assert flat.count("今天天气怎么样") == 1, (
        f"tools 开启时本次转录不应重复出现: {flat}"
    )
    # tool loop 接管 assistant 回写（full_text and not tools → 不追加）
    assert fake.conversation == [
        {"role": "user", "content": "上一轮的提问"},
        {"role": "user", "content": "今天天气怎么样"},
    ]
