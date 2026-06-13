"""RealtimeSession - core session logic for each WebSocket connection.

重写要点（全双工打断逻辑 v5）：
1. 打断时立即取消整个 pipeline（通过 asyncio.Task.cancel() 传播 CancelledError）
2. 去掉 should_cancel() 轮询检查，改用 CancelledError 传播
3. 去掉 _vad_speech_started_after_interruption 标志
4. 去掉 force_speaking_state() / sync_state_for_interruption() 调用
5. 打断时直接 vad.reset() 清空所有状态
6. 打断时发送 response.done(cancelled) 给客户端
7. 简化 _process_speech_input 的 finally 块
"""
import asyncio
import base64
import io
import json
import logging
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator

import numpy as np

from .protocol import ProtocolAdapter
from .vad import SileroVADModule
from .audio_buffer import AudioBufferManager
from .mode_router import ModeRouter
from .omni_client import OmniAudioClient, ChatStreamEvent, TextDeltaEvent, ToolCallDeltaEvent, ToolCallDoneEvent, FinishEvent
from .asr_client import SenseVoiceASRClient
from .local_asr import LocalASREngine
from .tts_pipeline import TTSPipeline, SentenceSplitter
from .tts_ws_pipeline import TTSWebSocketPipeline
from .gsv_tts_pipeline import GsvTtsPipeline
from .voxcpm2_tts_pipeline import VoxCpm2TtsPipeline, VoxCpm2TtsWsPipeline
from .text_processor import InlineDirectiveStripper
from .interruption import InterruptionHandler
from .config import ServerConfig

logger = logging.getLogger("realtime-server")

# ── Input size limits (P0-2) ──────────────────────────────────────────
MAX_AUDIO_FRAME_B64_BYTES = 5 * 1024 * 1024       # 5 MB base64 per audio frame
MAX_IMAGE_B64_BYTES = 10 * 1024 * 1024             # 10 MB base64 per image
MAX_CONVERSATION_ITEMS = 200                        # max items in conversation
MAX_SESSION_AUDIO_SECONDS = 600.0                   # cumulative input audio per session


class SessionConfig:
    """Per-session configuration, updated by session.update events."""

    def __init__(self, config: ServerConfig | None = None):
        self.instructions: str = ""
        self.modalities: list[str] = ["text", "audio"]
        self.voice: str = config.get("services", "tts", "voice", default="Vivian") if config else "Vivian"
        self.input_audio_format: str = "pcm16"
        self.output_audio_format: str = "pcm16"
        self.input_audio_transcription: dict = {"model": "gummy-realtime-v1"}
        self.turn_detection: dict = {
            "type": "server_vad",
            "threshold": config.get("vad", "threshold", default=0.5) if config else 0.5,
            "prefix_padding_ms": config.get("vad", "prefix_padding_ms", default=300) if config else 200,
            "silence_duration_ms": config.get("vad", "silence_timeout_ms", default=500) if config else 500,
        }
        self.temperature: float = 0.7
        self.repetition_penalty: float = 1.2
        self.tools: list[dict] | None = None  # Qwen format: [{"type":"function","function":{...}}]
        self.enable_search: bool = False

    def update(self, session_data: dict):
        """Update config from session.update event data."""
        if "instructions" in session_data:
            self.instructions = session_data["instructions"]
        if "modalities" in session_data:
            self.modalities = session_data["modalities"]
        #暂不支持客户端自定义音色
        #if "voice" in session_data:
        #    self.voice = session_data["voice"]
        if "input_audio_format" in session_data:
            self.input_audio_format = session_data["input_audio_format"]
        if "output_audio_format" in session_data:
            self.output_audio_format = session_data["output_audio_format"]
        if "input_audio_transcription" in session_data:
            self.input_audio_transcription = session_data["input_audio_transcription"]
        if "turn_detection" in session_data:
            td = session_data["turn_detection"]
            self.turn_detection.update(td)
        if "temperature" in session_data:
            self.temperature = float(session_data["temperature"])
        if "repetition_penalty" in session_data:
            self.repetition_penalty = float(session_data["repetition_penalty"])
        if "tools" in session_data:
            self.tools = session_data["tools"] or None
        if "enable_search" in session_data:
            self.enable_search = bool(session_data["enable_search"])
        # tools 和 enable_search 互斥（Qwen 约束）
        if self.tools and self.enable_search:
            logger.warning("tools and enable_search are mutually exclusive; disabling enable_search")
            self.enable_search = False


class RealtimeSession:
    """Manages a single WebSocket realtime session.
    
    全双工打断逻辑 v5：
    
    打断触发：VAD 检测到用户语音（speech_started）且当前正在响应时，立即触发打断：
    1. 取消 _active_pipeline_task（asyncio.Task.cancel()，让 CancelledError 传播）
    2. 发送 response.done(cancelled) 给客户端
    3. 清空 audio_buffer
    4. VAD 状态完全重置（vad.reset()）
    5. 当前帧写入 audio_buffer（作为新语音的起始）
    6. VAD 继续正常处理后续帧
    
    pipeline 取消：利用 Python asyncio 的 CancelledError 机制：
    - _stream_llm_to_tts_http 和 _stream_llm_to_tts_ws 中所有 await 点都会收到 CancelledError
    - 在 CancelledError handler 中关闭 TTS HTTP/WS 连接
    - 不再使用 should_cancel() 轮询，改用 CancelledError 传播
    """

    def __init__(self, websocket, model: str, config: ServerConfig):
        self.ws = websocket
        self.model = model
        self.config = config
        
        # Protocol adapter
        self.protocol = ProtocolAdapter(websocket)
        
        # Session config — stash model name so protocol can use it
        self.session_config = SessionConfig(config=config)
        self.session_config._model = model
        
        # ── Security limits (P0-2) ──────────────────────────────────
        # Override defaults from config.security if present
        _sec = config.get("security", default={})
        self._max_audio_frame_b64 = _sec.get("max_audio_frame_b64", MAX_AUDIO_FRAME_B64_BYTES) if isinstance(_sec, dict) else MAX_AUDIO_FRAME_B64_BYTES
        self._max_image_b64 = _sec.get("max_image_b64", MAX_IMAGE_B64_BYTES) if isinstance(_sec, dict) else MAX_IMAGE_B64_BYTES
        self._max_conversation_items = _sec.get("max_conversation_items", MAX_CONVERSATION_ITEMS) if isinstance(_sec, dict) else MAX_CONVERSATION_ITEMS
        self._max_session_audio_seconds = _sec.get("max_session_audio_seconds", MAX_SESSION_AUDIO_SECONDS) if isinstance(_sec, dict) else MAX_SESSION_AUDIO_SECONDS
        self._total_input_audio_seconds: float = 0.0
        
        # VAD - use ModelManager preloaded models if available, otherwise fallback to direct init
        from .model_manager import ModelManager
        manager = ModelManager.get_instance()
        if manager is not None:
            self.vad = manager.create_vad_module(config)
            logger.info("[Session] VAD created from preloaded ModelManager (lightweight init)")
        else:
            # Fallback: ModelManager not preloaded (should not happen in normal startup)
            logger.warning("[Session] ModelManager not preloaded, falling back to direct VAD init (slower)")
            self.vad = SileroVADModule(
                threshold=config.get("vad", "threshold", default=0.5),
                silence_ms=config.get("vad", "silence_timeout_ms", default=500),
                prefix_padding_ms=config.get("vad", "prefix_padding_ms", default=300),
                sample_rate=config.get("vad", "sample_rate", default=16000),
                silero_model_path=config.get("vad", "silero_model_path", default=None),
                min_speech_duration_ms=config.get("vad", "min_speech_duration_ms", default=200),
                max_audio_duration_ms=config.get("vad", "max_audio_duration_ms", default=30000),
                smart_turn_enabled=config.get("vad", "smart_turn_enabled", default=True),
                smart_turn_path=config.get("vad", "smart_turn_path", default=None),
                smart_turn_threshold=config.get("vad", "smart_turn_threshold", default=0.5),
            )
        
        # Audio buffer
        self.audio_buffer = AudioBufferManager(
            sample_rate=config.get("vad", "sample_rate", default=16000)
        )
        
        # Mode router
        self.mode_router = ModeRouter({
            "default_mode": config.get("realtime_server", "default_mode", default="omni_audio"),
            "omni_error_threshold": config.get("realtime_server", "omni_error_threshold", default=3),
            "fallback_duration_s": config.get("realtime_server", "fallback_duration_s", default=60),
        })
        
        # Clients
        self.omni_client = OmniAudioClient(
            base_url=config.get("services", "omni", "base_url", default="http://localhost:8000/v1"),
            model=config.get("services", "omni", "model", default="Qwen3-Omni-30B-A3B-Instruct-AWQ-8bit"),
            api_key=config.get("services", "omni", "api_key", default=None),
        )
        
        self.asr_client = SenseVoiceASRClient(
            base_url=config.get("services", "asr", "base_url", default="http://localhost:8082/v1"),
        )
        
        # Local ASR engine (singleton, shared across sessions)
        self.local_asr: LocalASREngine | None = None
        if config.get("services", "asr", "local_asr", default=False):
            asr_model_path = config.get("services", "asr", "asr_model_path", default=None)
            asr_device = config.get("services", "asr", "device", default="cuda")
            # Use existing singleton if available, otherwise create new one
            existing = LocalASREngine.get_instance()
            if existing is not None:
                self.local_asr = existing
                logger.info("[Session] Using existing LocalASREngine singleton")
            else:
                logger.info(f"[Session] Initializing LocalASREngine with model_path={asr_model_path}, device={asr_device}")
                self.local_asr = LocalASREngine(model_path=asr_model_path, device=asr_device)
        
        # TTS model routing: services.tts.model determines which pipeline to use
        # Qwen3-TTS (default) | voxcpm2 | gsv-tts-lite
        tts_model = config.get("services", "tts", "model", default="Qwen3-TTS")

        if tts_model == "voxcpm2":
            self.tts_pipeline = VoxCpm2TtsPipeline(
                base_url=config.get("services", "tts", "base_url", default="http://localhost:8093/v1"),
                voice=config.get("services", "tts", "voice", default="default"),
                timeout_s=config.get("services", "tts", "timeout_s", default=30),
                api_key=config.get("services", "tts", "api_key", default=""),
                ref_audio=config.get("services", "tts", "ref_audio", default=None),
                ref_text=config.get("services", "tts", "ref_text", default=None),
            )
            self.directive_stripper = InlineDirectiveStripper()
            self.tts_mode = config.get("services", "tts", "mode", default="http")
            self.gsv_tts_pipeline = None
            logger.info(f"[Session] VoxCPM2 TTS enabled: base_url={self.tts_pipeline.base_url}, mode={self.tts_mode}")

        elif tts_model == "gsv-tts-lite":
            self.tts_pipeline = TTSPipeline(
                base_url=config.get("services", "tts", "base_url", default="http://localhost:8091/v1"),
                voice=self.session_config.voice,
                sample_rate_out=config.get("tts_pipeline", "output_sample_rate", default=24000),
                timeout_s=config.get("services", "tts", "timeout_s", default=15),
                api_key=config.get("services", "tts", "api_key", default=None),
            )
            self.gsv_tts_pipeline = GsvTtsPipeline(
                base_url=config.get("services", "gsv_tts", "base_url", default="http://localhost:8001"),
                speaker_audio=config.get("services", "gsv_tts", "speaker_audio", default=""),
                prompt_audio=config.get("services", "gsv_tts", "prompt_audio", default=""),
                prompt_text=config.get("services", "gsv_tts", "prompt_text", default=""),
                sample_rate_out=config.get("tts_pipeline", "output_sample_rate", default=24000),
                timeout_s=config.get("services", "gsv_tts", "timeout_s", default=30),
                speed=config.get("services", "gsv_tts", "speed", default=1.0),
            )
            self.directive_stripper = None
            self.tts_mode = "http"  # GSV-TTS-Lite forced HTTP mode
            logger.info(f"[Session] GSV-TTS-Lite enabled: base_url={self.gsv_tts_pipeline.base_url}")

        else:  # Qwen3-TTS (default)
            self.tts_pipeline = TTSPipeline(
                base_url=config.get("services", "tts", "base_url", default="http://localhost:8091/v1"),
                voice=self.session_config.voice,
                sample_rate_out=config.get("tts_pipeline", "output_sample_rate", default=24000),
                timeout_s=config.get("services", "tts", "timeout_s", default=15),
                api_key=config.get("services", "tts", "api_key", default=None),
                ref_audio=config.get("services", "tts", "ref_audio", default=None),
                ref_text=config.get("services", "tts", "ref_text", default=None),
            )
            self.directive_stripper = None
            self.gsv_tts_pipeline = None
            self.tts_mode = config.get("services", "tts", "mode", default="http")
        self.tts_ws_url = config.get("services", "tts", "ws_url", default="ws://localhost:8091/v1/audio/speech/stream")
        # C1 fix: 删除 self.tts_voice/tts_language/tts_sample_rate/tts_timeout_s
        # voice 统一从 session_config.voice 读取，language/sample_rate/timeout 从 config 读取
        
        # Interruption handler (simplified - only cancel flag + protocol response)
        self.interruption = InterruptionHandler()
        
        # Conversation history
        self.conversation: list[dict] = []
        
        # State
        self._is_responding = False
        self._current_resp_id: str | None = None
        self._response_lock = asyncio.Lock()
        
        # ---- 语音期间音频累积状态 ----
        self._is_speech_active = False
        
        # ---- 音频chunk计数 ----
        self._audio_chunk_count = 0
        
        # ---- 活跃的pipeline task跟踪 ----
        # 用于实现"打断时立即取消整个pipeline"的机制
        # 打断时调用 _cancel_active_pipeline() 取消旧pipeline
        self._active_pipeline_task: asyncio.Task | None = None

        # ---- VAD offload executor ----
        self._vad_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vad")

        # ---- Tool calling state ----
        self._pending_tool_results: dict[str, str] = {}        # call_id -> output
        self._tool_result_events: dict[str, asyncio.Event] = {} # call_id -> Event (signaled when result arrives)
        self._tool_result_timeout_s = 60  # max wait for client to send tool result

    async def handle_event(self, event: dict):
        """Handle incoming client event."""
        event_type = event.get("type", "")
        event_id = event.get("event_id", "")
        
        try:
            if event_type == "session.update":
                await self._handle_session_update(event)
            elif event_type == "input_audio_buffer.append":
                await self._handle_audio_append(event)
            elif event_type == "input_audio_buffer.clear":
                await self._handle_audio_clear(event)
            elif event_type == "input_image_buffer.append":
                await self._handle_image_append(event)
            elif event_type == "conversation.item.create":
                await self._handle_conversation_item_create(event)
            elif event_type == "response.create":
                await self._handle_response_create(event)
            elif event_type == "response.cancel":
                await self._handle_response_cancel(event)
            else:
                logger.warning(f"Unknown event type: {event_type}")
        except Exception as e:
            logger.error(f"Error handling event {event_type}: {e}", exc_info=True)
            await self.protocol.send_error("internal error")

    async def _handle_session_update(self, event: dict):
        """Handle session.update event."""
        session_data = event.get("session", {})
        # 不更新客户端session的voice
        tts_voice = self.session_config.voice
        self.session_config.update(session_data)
        self.session_config.voice = tts_voice
        td = self.session_config.turn_detection
        if "silence_duration_ms" in td:
            self.vad.update_silence_config(td["silence_duration_ms"])
        # session.update 时同步 voice 到 tts_pipeline
        self.tts_pipeline.voice = self.session_config.voice
        logger.info(f"Session updated: voice={self.session_config.voice}, "
                     f"threshold={self.vad.threshold}, "
                     f"silence_ms={self.vad.silence_duration_ms}, "
                     f"vad_backend={self.vad.vad_backend}, "
                     f"smart_turn_enabled={self.vad.smart_turn_enabled}, "
                     f"tools={[t['function']['name'] for t in self.session_config.tools] if self.session_config.tools else None}")

        await self.protocol.send_session_updated(self.session_config)

    async def _handle_audio_append(self, event: dict):
        """Handle input_audio_buffer.append event.
        
        全双工打断逻辑 v5：
        
        核心原则：打断 = 全链路立即丢弃旧数据 + 立即处理新语音
        
        打断触发（VAD speech_started 且 _is_responding）：
        1. _cancel_active_pipeline("user_interrupt")  # 取消旧pipeline
        2. 发送 response.done(cancelled) 给客户端
        3. _is_responding = False
        4. audio_buffer.clear_audio()
        5. vad.reset()  # 完全重置VAD状态
        6. 把当前帧写入 audio_buffer（作为新语音的起始）
        7. _is_speech_active = True
        8. 发送 speech_started 给客户端
        9. VAD 继续正常处理后续帧
        
        speech_stopped 处理：
        1. _is_speech_active = False
        2. 发送 speech_stopped 给客户端
        3. 取消旧pipeline（安全起见）
        4. 启动新pipeline
        """
        audio_b64 = event.get("audio", "")
        if not audio_b64:
            return
        
        # P0-2: Enforce per-frame audio size limit
        if len(audio_b64) > self._max_audio_frame_b64:
            logger.warning(f"Audio frame too large: {len(audio_b64)} > {self._max_audio_frame_b64} bytes, dropping")
            await self.protocol.send_error("invalid_request_error: audio frame exceeds maximum allowed size")
            return
        
        self._audio_chunk_count += 1
        if self._audio_chunk_count % 100 == 1:
            logger.debug(f"Audio stream active: chunk #{self._audio_chunk_count}, {len(audio_b64)} chars base64")
        
        # Decode audio
        try:
            pcm_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            logger.warning(f"Failed to decode audio: {e}")
            return
        
        # P0-2: Track cumulative input audio seconds and reject if over limit
        frame_duration_s = len(pcm_bytes) / 2 / 16000  # PCM16 = 2 bytes/sample, 16kHz
        self._total_input_audio_seconds += frame_duration_s
        if self._total_input_audio_seconds > self._max_session_audio_seconds:
            logger.warning(f"Session cumulative audio exceeded limit: {self._total_input_audio_seconds:.1f}s > {self._max_session_audio_seconds:.1f}s")
            await self.protocol.send_error("invalid_request_error: session cumulative input audio duration exceeded")
            self._total_input_audio_seconds -= frame_duration_s  # revert since we're dropping
            return
        
        # VAD processing (offloaded to avoid blocking event loop)
        loop = asyncio.get_running_loop()
        vad_events = await loop.run_in_executor(self._vad_executor, self.vad.process, pcm_bytes)
        
        for vad_event in vad_events:
            if vad_event == "speech_started":
                if self._is_responding:
                    # ---- 全双工打断逻辑 v5 ----
                    logger.info("[Session] VAD detected speech during response, triggering immediate interruption")
                    
                    # 1. 取消旧的pipeline task（CancelledError会传播到LLM/TTS的每一层）
                    await self._cancel_active_pipeline("user_interrupt")
                    
                    # 2. 发送 response.done(cancelled) 给客户端
                    if self._current_resp_id:
                        await self.protocol.send_response_done(self._current_resp_id, status="cancelled")
                        self._current_resp_id = None
                    
                    # 3. 在 _response_lock 内设置 _is_responding = False（防止与 speech_stopped 竞态）
                    async with self._response_lock:
                        self._is_responding = False
                    
                    # 4. 写入 prefix_audio（speech_started之前的~300ms，不写则吞字）
                    prefix_audio = self.vad.get_prefix_audio()
                    if prefix_audio:
                        self.audio_buffer.append_audio_raw(prefix_audio)
                        prefix_ms = len(prefix_audio) / 2 / (self.vad.sample_rate / 1000)
                        logger.info(
                            f"[Session] Interruption: prefix padding {prefix_ms:.0f}ms "
                            f"({len(prefix_audio)} bytes) written to audio_buffer"
                        )
                    
                    # 5. 设 _is_speech_active=True，让后续帧继续写入audio_buffer直到VAD自然触发speech_stopped
                    self._is_speech_active = True
                    
                    # 6. 通知客户端语音开始（与后续speech_stopped配对）
                    await self.protocol.send_speech_started()
                    
                    logger.info("[Session] Interruption completed, VAD continues naturally")
                else:
                    # ---- 非响应期间的正常 speech_started 处理 ----
                    # 正常语音开始
                    prefix_audio = self.vad.get_prefix_audio()
                    if prefix_audio:
                        self.audio_buffer.append_audio_raw(prefix_audio)
                        prefix_ms = len(prefix_audio) / 2 / (self.vad.sample_rate / 1000)
                        logger.info(
                            f"[Session] speech_started: prefix padding {prefix_ms:.0f}ms "
                            f"({len(prefix_audio)} bytes) written to audio_buffer"
                        )
                    self._is_speech_active = True
                    
                    await self.protocol.send_speech_started()
                
            elif vad_event == "speech_stopped":
                self._is_speech_active = False
                
                await self.protocol.send_speech_stopped()
                
                async with self._response_lock:
                    if self._is_responding:
                        continue
                    self._is_responding = True
                    # 取消旧的pipeline task，确保同一时间只有一个pipeline在处理
                    await self._cancel_active_pipeline("new_speech")
                    # 启动新的pipeline task
                    self._active_pipeline_task = asyncio.create_task(self._process_speech_input())
            
            elif vad_event == "max_duration_reached":
                self._is_speech_active = False
                logger.info("[Session] 最大音频时长已达，强制触发 LLM 处理")
        
        # ---- 条件写入 audio_buffer ----
        if self._is_speech_active:
            self.audio_buffer.append_audio_raw(pcm_bytes)

    async def _cancel_active_pipeline(self, reason: str):
        """取消当前活跃的pipeline task。
        
        通过 asyncio.Task.cancel() 触发 CancelledError 传播到整个pipeline。
        不在打断路径上 await 被取消的 task，避免 ONNX 推理中途的同步等待。
        """
        if self._active_pipeline_task is not None and not self._active_pipeline_task.done():
            logger.info(
                f"[Session] Cancelling active pipeline task (reason={reason}), "
                f"task_done={self._active_pipeline_task.done()}"
            )
            task = self._active_pipeline_task
            self._active_pipeline_task = None
            task.cancel()
            # Fire-and-forget: await the task in background to ensure
            # CancelledError is consumed and resources are cleaned up,
            # but don't block the interruption path.
            async def _cleanup_cancelled_task(t: asyncio.Task):
                try:
                    await t
                except asyncio.CancelledError:
                    logger.info("[Session] Active pipeline task cancelled successfully (background cleanup)")
                except Exception as e:
                    logger.warning(f"[Session] Active pipeline task cancellation error (background cleanup): {e}")
            
            asyncio.create_task(_cleanup_cancelled_task(task))
        
        # 同时确保interruption状态也被清理
        # 这样pipeline内部的 should_cancel() 检查也能生效（作为辅助检查）
        if self.interruption.is_generating:
            logger.info(f"[Session] Setting interruption cancel flag (reason={reason})")
            self.interruption.set_generating(False)

    async def _handle_audio_clear(self, event: dict):
        """Handle input_audio_buffer.clear event."""
        self.audio_buffer.clear_audio()
        self.vad.reset()
        self._is_speech_active = False

    async def _handle_image_append(self, event: dict):
        """Handle input_image_buffer.append event."""
        image_b64 = event.get("image", "")
        if image_b64:
            # P0-2: Enforce per-image size limit
            if len(image_b64) > self._max_image_b64:
                logger.warning(f"Image too large: {len(image_b64)} > {self._max_image_b64} bytes, dropping")
                await self.protocol.send_error("invalid_request_error: image exceeds maximum allowed size")
                return
            self.audio_buffer.append_image(image_b64)

    async def _handle_conversation_item_create(self, event: dict):
        """Handle conversation.item.create event."""
        item = event.get("item", {})
        item_type = item.get("type", "")
        if item_type == "message":
            role = item.get("role", "user")
            # P0-7: Reject system role
            if role == "system":
                await self.protocol.send_error("invalid_request_error: role 'system' not allowed in conversation.item.create")
                return
            content = item.get("content", [])
            msg = {"role": role, "content": content}
            self.conversation.append(msg)
            # P0-2: Enforce conversation length limit (rolling window)
            if len(self.conversation) > self._max_conversation_items:
                dropped = len(self.conversation) - self._max_conversation_items
                self.conversation = self.conversation[dropped:]
                logger.warning(f"Conversation exceeded {self._max_conversation_items} items, dropped {dropped} oldest")
            logger.info(f"Added conversation item: role={role}")
        elif item_type == "function_call_output":
            call_id = item.get("call_id", "")
            output = item.get("output", "")
            if call_id:
                self._pending_tool_results[call_id] = output
                # Signal waiting tool loop that result is available
                if call_id in self._tool_result_events:
                    self._tool_result_events[call_id].set()
                logger.info(f"Stored tool result for call_id={call_id} ({len(output)} chars)")

    async def _handle_response_create(self, event: dict):
        """Handle response.create event."""
        async with self._response_lock:
            if self._is_responding:
                logger.warning("Response already in progress, ignoring response.create")
                return
            self._is_responding = True
        
        if self.audio_buffer.get_full_pcm():
            self._active_pipeline_task = asyncio.create_task(self._process_speech_input())
        elif self.conversation:
            self._active_pipeline_task = asyncio.create_task(self._process_text_input())
        else:
            # 无音频且无对话历史时，重置状态避免卡死
            logger.warning("response.create with no audio buffer and no conversation, resetting state")
            async with self._response_lock:
                self._is_responding = False

    async def _handle_response_cancel(self, event: dict):
        """Handle response.cancel event."""
        await self._cancel_active_pipeline("client_cancel")
        if self._current_resp_id:
            await self.protocol.send_response_done(self._current_resp_id, status="cancelled")
            self._current_resp_id = None
        async with self._response_lock:
            self._is_responding = False

    # ----------------------------------------------------------------
    # 公共流式管线方法：LLM → TTS 流式处理（支持 tool calling）
    # ----------------------------------------------------------------

    def _llm_stream_kwargs(self) -> dict:
        """Common kwargs for omni_client.stream_chat calls."""
        return {
            "temperature": self.session_config.temperature,
            "repetition_penalty": self.session_config.repetition_penalty,
            "max_tokens": self.config.get("services", "omni", "max_tokens", default=4096),
            "timeout_s": self.config.get("services", "omni", "timeout_s", default=30),
            "tools": self.session_config.tools,
            "enable_search": self.session_config.enable_search,
        }

    async def _stream_llm_to_tts(self, messages: list[dict], resp_id: str):
        """公共 LLM→TTS 流式管线逻辑（带 tool calling 支持）。

        如果 session 配置了 tools，走 _stream_llm_with_tool_loop；
        否则走纯文本→TTS 直通路径。

        Returns:
            str: LLM 生成的完整文本（用于写入 conversation history）
        """
        if self.session_config.tools:
            return await self._stream_llm_with_tool_loop(messages, resp_id)

        if self.tts_mode == "ws":
            return await self._stream_llm_events_to_tts_ws(messages, resp_id)
        else:
            return await self._stream_llm_events_to_tts_http(messages, resp_id)

    # ---- Tool calling lifecycle ----

    async def _stream_llm_with_tool_loop(self, messages: list[dict], resp_id: str) -> str:
        """LLM→TTS with tool calling support.

        Loops: LLM call → if finish_reason=="tool_calls" → emit delta/done events
        → wait for client tool results → append tool messages → re-invoke LLM
        → until finish_reason=="stop".

        Returns the final text response (from the last LLM call that produced text).
        """
        MAX_TOOL_ROUNDS = 5
        full_response_text = ""
        current_messages = list(messages)  # copy to avoid mutating original
        output_index = 0
        all_tool_calls: list[dict] = []  # collect across rounds for conversation history

        for round_num in range(MAX_TOOL_ROUNDS):
            # Stream LLM, routing text→TTS and tool_calls→protocol events
            text_parts: list[str] = []
            tool_calls_seen: dict[int, dict] = {}  # index -> {call_id, name, arguments}
            finish_reason = "stop"
            has_tool_calls = False

            if self.tts_mode == "ws":
                round_text, round_tc, round_fr = await self._stream_llm_events_to_tts_ws(
                    current_messages, resp_id, output_index,
                )
            else:
                round_text, round_tc, round_fr = await self._stream_llm_events_to_tts_http(
                    current_messages, resp_id, output_index,
                )

            text_parts.append(round_text)
            tool_calls_seen = round_tc
            finish_reason = round_fr
            output_index += len(tool_calls_seen)

            if finish_reason != "tool_calls" or not tool_calls_seen:
                # Normal completion
                full_response_text = "".join(text_parts)
                break

            has_tool_calls = True
            logger.info(f"[Tool Loop] Round {round_num}: {len(tool_calls_seen)} tool call(s), waiting for results")

            # 1. Build assistant message with tool_calls for Chat Completions
            assistant_tool_calls = []
            for idx in sorted(tool_calls_seen.keys()):
                tc = tool_calls_seen[idx]
                assistant_tool_calls.append({
                    "id": tc["call_id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                })
                all_tool_calls.append({
                    "id": tc["call_id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                })

            round_text_str = round_text if isinstance(round_text, str) else ""
            current_messages.append({
                "role": "assistant",
                "content": round_text_str or None,
                "tool_calls": assistant_tool_calls,
            })

            # 2. Wait for all tool results from client
            for idx in sorted(tool_calls_seen.keys()):
                tc = tool_calls_seen[idx]
                call_id = tc["call_id"]

                if call_id in self._pending_tool_results:
                    result = self._pending_tool_results.pop(call_id)
                else:
                    # Create event and wait (main event loop continues processing)
                    evt = asyncio.Event()
                    self._tool_result_events[call_id] = evt
                    try:
                        await asyncio.wait_for(evt.wait(), timeout=self._tool_result_timeout_s)
                    except asyncio.TimeoutError:
                        logger.warning(f"[Tool Loop] Timeout waiting for tool result call_id={call_id}")
                        await self.protocol.send_error(f"Tool result timeout for {tc['name']}")
                        del self._tool_result_events[call_id]
                        full_response_text = "".join(text_parts)
                        # 为当前超时的 tool call 补充 placeholder error result，
                        # 确保 conversation history 中每个 tool_call 都有对应 tool result
                        current_messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": f"error: tool result timeout for {tc['name']}",
                        })
                        break
                    except asyncio.CancelledError:
                        # Pipeline cancelled while waiting for tool result — clean up
                        logger.info(f"[Tool Loop] Cancelled while waiting for tool result call_id={call_id}")
                        del self._tool_result_events[call_id]
                        raise
                    del self._tool_result_events[call_id]
                    result = self._pending_tool_results.pop(call_id, "")

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                })
                logger.info(f"[Tool Loop] Received result for {tc['name']} (call_id={call_id}, {len(result)} chars)")
            else:
                # 3. Re-invoke LLM — loop continues
                resp_id = await self.protocol.send_response_created()
                self._current_resp_id = resp_id
                continue
            # If we broke out of the inner for loop (timeout), exit outer loop too
            break

        # Update conversation history
        if all_tool_calls:
            self.conversation.append({
                "role": "assistant",
                "content": full_response_text or None,
                "tool_calls": all_tool_calls,
            })
            # Tool result messages are already in current_messages but not self.conversation;
            # add them for future turns
            for msg in current_messages:
                if msg.get("role") == "tool" and msg not in self.conversation:
                    self.conversation.append(msg)
        else:
            if full_response_text:
                self.conversation.append({"role": "assistant", "content": full_response_text})

        return full_response_text

    # ---- Event-based LLM streaming with TTS routing (HTTP mode) ----

    async def _stream_llm_events_to_tts_http(
        self, messages: list[dict], resp_id: str, output_index: int = 0,
    ) -> tuple[str, dict[int, dict], str]:
        """Stream LLM ChatStreamEvents → TTS for text, protocol events for tool_calls (HTTP mode).

        Returns: (full_text, tool_calls_seen dict, finish_reason)
        """
        self.tts_pipeline.voice = self.session_config.voice
        full_text = ""
        tool_calls_seen: dict[int, dict] = {}
        finish_reason = "stop"
        has_tool_calls = False

        splitter = SentenceSplitter(
            min_sub_sentence_len=self.config.get("tts_pipeline", "min_sub_sentence_len", default=6)
        )
        max_concurrent = self.config.get("tts_pipeline", "max_concurrent_tts", default=2)
        tts_semaphore = asyncio.Semaphore(max_concurrent)

        # Determine active TTS pipeline for this session
        tts_model = self.config.get("services", "tts", "model", default="Qwen3-TTS")
        active_tts_pipeline = self.gsv_tts_pipeline if tts_model == "gsv-tts-lite" else self.tts_pipeline

        sentence_audio_queues: list[asyncio.Queue[bytes | None]] = []
        tts_tasks: list[asyncio.Task] = []
        all_sentences_enqueued = False

        async def _process_sentence_tts(sentence: str, queue: asyncio.Queue):
            async with tts_semaphore:
                try:
                    if tts_model == "gsv-tts-lite":
                        async for pcm_chunk in active_tts_pipeline.stream_tts(sentence):
                            await queue.put(pcm_chunk)
                    elif tts_model == "voxcpm2":
                        async for pcm_chunk in active_tts_pipeline.stream_tts(sentence):
                            await queue.put(pcm_chunk)
                    else:  # Qwen3-TTS
                        async for pcm_chunk in active_tts_pipeline.stream_tts(
                            sentence, instructions="",
                        ):
                            await queue.put(pcm_chunk)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"TTS error for sentence: {e}", exc_info=True)
                finally:
                    await queue.put(None)

        async def _drain_loop():
            drain_idx = 0
            while True:
                if drain_idx < len(sentence_audio_queues):
                    q = sentence_audio_queues[drain_idx]
                    while True:
                        chunk = await q.get()
                        if chunk is None:
                            break
                        await self.protocol.send_audio_delta(chunk)
                    drain_idx += 1
                elif all_sentences_enqueued:
                    break
                else:
                    await asyncio.sleep(0.005)

        drain_task = asyncio.create_task(_drain_loop())

        try:
            async for event in self.omni_client.stream_chat(
                messages=messages, **self._llm_stream_kwargs(),
            ):
                if event["type"] == "text":
                    text_delta = event["delta"]
                    full_text += text_delta
                    # Directive stripping: raw text → TTS (preserves directives),
                    # stripped text → client transcript (removes directives for voxcpm2)
                    if self.directive_stripper:
                        clean_text = self.directive_stripper.feed(text_delta)
                    else:
                        clean_text = text_delta
                    if clean_text:
                        await self.protocol.send_transcript_delta(clean_text)
                    # Only send text to TTS if no tool calls detected yet in this round
                    if not has_tool_calls:
                        sentences = splitter.add_text(text_delta)
                        for sentence in sentences:
                            logger.info(f'[TRACE] tts_sentence: text="{sentence}", sentence_len={len(sentence)}')
                            q = asyncio.Queue()
                            sentence_audio_queues.append(q)
                            task = asyncio.create_task(_process_sentence_tts(sentence, q))
                            tts_tasks.append(task)

                elif event["type"] == "tool_call":
                    has_tool_calls = True
                    idx = event["index"]
                    call_id = event["call_id"]
                    name = event.get("name")
                    args_delta = event["arguments_delta"]

                    if idx not in tool_calls_seen:
                        tool_calls_seen[idx] = {"call_id": call_id, "name": name or "", "arguments": ""}
                        logger.info(f"[Tool Call] function_call started: name={name or '?'}, call_id={call_id}, index={idx}")
                        await self.protocol.send_function_call_item_added(
                            call_id, name or "", resp_id, output_index + idx,
                        )
                    if name:
                        tool_calls_seen[idx]["name"] = name
                    tool_calls_seen[idx]["arguments"] += args_delta

                    await self.protocol.send_function_call_arguments_delta(
                        call_id, tool_calls_seen[idx]["name"], args_delta,
                    )

                elif event["type"] == "tool_call_done":
                    tc = tool_calls_seen.get(event["index"], {})
                    if tc:
                        await self.protocol.send_function_call_arguments_done(
                            tc["call_id"], tc["name"], tc["arguments"],
                            resp_id, output_index + event["index"],
                        )

                elif event["type"] == "finish":
                    finish_reason = event["finish_reason"]

            # Flush remaining text to TTS (only if no tool calls)
            if not has_tool_calls:
                for sentence in splitter.flush():
                    logger.info(f'[TRACE] tts_sentence: text="{sentence}", sentence_len={len(sentence)}(flush)')
                    q = asyncio.Queue()
                    sentence_audio_queues.append(q)
                    task = asyncio.create_task(_process_sentence_tts(sentence, q))
                    tts_tasks.append(task)

            all_sentences_enqueued = True
            await drain_task
            if tts_tasks:
                await asyncio.gather(*tts_tasks, return_exceptions=True)

        except asyncio.CancelledError:
            logger.info("[Session] HTTP TTS pipeline cancelled (CancelledError propagated)")
            all_sentences_enqueued = True
            drain_task.cancel()
            for task in tts_tasks:
                if not task.done():
                    task.cancel()
            raise

        return full_text, tool_calls_seen, finish_reason

    # ---- Event-based LLM streaming with TTS routing (WS mode) ----

    async def _stream_llm_events_to_tts_ws(
        self, messages: list[dict], resp_id: str, output_index: int = 0,
    ) -> tuple[str, dict[int, dict], str]:
        """Stream LLM ChatStreamEvents → TTS for text, protocol events for tool_calls (WS mode).

        Returns: (full_text, tool_calls_seen dict, finish_reason)
        """
        t0 = time.time()
        full_text = ""
        tool_calls_seen: dict[int, dict] = {}
        finish_reason = "stop"
        has_tool_calls = False

        tts_model_ws = self.config.get("services", "tts", "model", default="Qwen3-TTS")
        if tts_model_ws == "voxcpm2":
            ws_pipeline = VoxCpm2TtsWsPipeline(
                ws_url=self.tts_ws_url,
                voice=self.session_config.voice,
                sample_rate=self.config.get("services", "tts", "sample_rate", default=24000),
                timeout_s=self.config.get("services", "tts", "timeout_s", default=15),
                api_key=self.config.get("services", "tts", "api_key", default=None),
                ref_audio=self.tts_pipeline.ref_audio if hasattr(self.tts_pipeline, 'ref_audio') else None,
                ref_text=self.tts_pipeline.ref_text if hasattr(self.tts_pipeline, 'ref_text') else None,
            )
        else:
            ws_pipeline = TTSWebSocketPipeline(
                ws_url=self.tts_ws_url,
                voice=self.session_config.voice,
                sample_rate=self.config.get("services", "tts", "sample_rate", default=24000),
                language=self.config.get("services", "tts", "language", default="Chinese"),
                timeout_s=self.config.get("services", "tts", "timeout_s", default=15),
                api_key=self.config.get("services", "tts", "api_key", default=None),
                ref_audio=self.tts_pipeline.ref_audio if hasattr(self.tts_pipeline, 'ref_audio') else None,
                ref_text=self.tts_pipeline.ref_text if hasattr(self.tts_pipeline, 'ref_text') else None,
            )

        tts_first_chunk_received = False
        tts_receive_done = asyncio.Event()
        tts_error: list[str | None] = [None]

        async def _receive_and_send_audio():
            nonlocal tts_first_chunk_received
            try:
                async for pcm_chunk in ws_pipeline.receive_audio():
                    if not tts_first_chunk_received:
                        tts_first_chunk_received = True
                        latency_ms = (time.time() - t0) * 1000
                        logger.info(f'[TRACE] tts_ws_first_chunk: latency={latency_ms:.1f}ms, chunk_size={len(pcm_chunk)}')
                    await self.protocol.send_audio_delta(pcm_chunk)
            except asyncio.CancelledError:
                logger.info("[Session] TTS WS receive_and_send_audio cancelled")
                raise
            except Exception as e:
                tts_error[0] = str(e)
                logger.error(f"TTS WS receive_and_send error: {e}", exc_info=True)
            finally:
                tts_receive_done.set()

        receive_task = None
        try:
            await ws_pipeline.connect()
            logger.info(f'[TRACE] tts_ws_connected: latency={(time.time() - t0) * 1000:.1f}ms')
            receive_task = asyncio.create_task(_receive_and_send_audio())

            llm_first_token = False
            tts_first_text_sent = False

            async for event in self.omni_client.stream_chat(
                messages=messages, **self._llm_stream_kwargs(),
            ):
                if event["type"] == "text":
                    text_delta = event["delta"]
                    full_text += text_delta
                    # Directive stripping for voxcpm2
                    if self.directive_stripper:
                        clean_text = self.directive_stripper.feed(text_delta)
                    else:
                        clean_text = text_delta
                    if clean_text:
                        await self.protocol.send_transcript_delta(clean_text)

                    if not llm_first_token:
                        llm_first_token = True
                        latency_ms = (time.time() - t0) * 1000
                        logger.info(f'[TRACE] llm_first_token: latency={latency_ms:.1f}ms')

                    # Only send text to TTS WS if no tool calls detected
                    if not has_tool_calls:
                        await ws_pipeline.send_text_delta(text_delta)
                        if not tts_first_text_sent:
                            tts_first_text_sent = True
                            latency_ms = (time.time() - t0) * 1000
                            logger.info(f'[TRACE] tts_ws_text_sent: latency={latency_ms:.1f}ms, text="{text_delta[:20]}"')

                elif event["type"] == "tool_call":
                    has_tool_calls = True
                    idx = event["index"]
                    call_id = event["call_id"]
                    name = event.get("name")
                    args_delta = event["arguments_delta"]

                    if idx not in tool_calls_seen:
                        tool_calls_seen[idx] = {"call_id": call_id, "name": name or "", "arguments": ""}
                        logger.info(f"[Tool Call] function_call started: name={name or '?'}, call_id={call_id}, index={idx}")
                        await self.protocol.send_function_call_item_added(
                            call_id, name or "", resp_id, output_index + idx,
                        )
                    if name:
                        tool_calls_seen[idx]["name"] = name
                    tool_calls_seen[idx]["arguments"] += args_delta

                    await self.protocol.send_function_call_arguments_delta(
                        call_id, tool_calls_seen[idx]["name"], args_delta,
                    )

                elif event["type"] == "tool_call_done":
                    tc = tool_calls_seen.get(event["index"], {})
                    if tc:
                        await self.protocol.send_function_call_arguments_done(
                            tc["call_id"], tc["name"], tc["arguments"],
                            resp_id, output_index + event["index"],
                        )

                elif event["type"] == "finish":
                    finish_reason = event["finish_reason"]

            # Finish TTS input (only if we were sending text to TTS)
            if not has_tool_calls:
                await ws_pipeline.finish_input()
                await receive_task
                if tts_error[0]:
                    raise RuntimeError(f"TTS receive error: {tts_error[0]}")
                latency_ms = (time.time() - t0) * 1000
                logger.info(f'[TRACE] tts_ws_done: latency={latency_ms:.1f}ms, total_sentences={ws_pipeline.total_sentences}')

        except asyncio.CancelledError:
            logger.info("[Session] TTS WS pipeline cancelled by interruption")
            if receive_task and not receive_task.done():
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
            await ws_pipeline.close()
            raise
        except Exception as e:
            logger.error(f"TTS WS pipeline error: {e}", exc_info=True)
            raise
        finally:
            if ws_pipeline.is_connected:
                await ws_pipeline.close()

        return full_text, tool_calls_seen, finish_reason

    async def _process_speech_input(self):
        """Process speech input after VAD speech_stopped.
        
        重写要点（v5）：
        - 简化 finally 块：只做 _is_responding = False 和 interruption.set_generating(False)
        - 不再调用 vad.sync_state_for_interruption()（已删除该方法）
        - 不再调用 vad.reset()（VAD状态由打断逻辑在 _handle_audio_append 中管理）
        - 被取消时不发送 response.done（由取消方负责发送）
        """
        # 先从audio_buffer取出所有音频数据保存到局部变量
        # 这样后续任何清空操作都不会影响已保存的数据
        captured_pcm = self.audio_buffer.get_full_pcm()
        captured_duration_ms = self.audio_buffer.get_duration_ms()
        captured_images = list(self.audio_buffer.image_chunks)  # 也保存图片数据
        
        # 清空audio_buffer（数据已保存到局部变量，不会丢失）
        self.audio_buffer.clear_audio()
        self.audio_buffer.clear_images()
        
        my_task = asyncio.current_task()
        
        try:
            self._is_speech_active = False
            
            mode = self.mode_router.get_mode()
            
            if mode == ModeRouter.MODE_B:
                await self._process_mode_b(captured_pcm, captured_duration_ms, captured_images)
            else:
                await self._process_mode_a(captured_pcm, captured_duration_ms, captured_images)
        except asyncio.CancelledError:
            logger.info("[Session] _process_speech_input cancelled, pipeline interrupted")
            raise
        except Exception as e:
            logger.error(f"Error processing speech input: {e}", exc_info=True)
            await self.protocol.send_error("internal error")
        finally:
            if self._active_pipeline_task is my_task:
                self._is_responding = False
                self._active_pipeline_task = None
            self.interruption.set_generating(False)

    async def _process_mode_b(self, captured_pcm: bytes, duration_ms: float, captured_images: list[str] | None = None):
        """Mode B: Omni audio input + TTS output.
        
        Args:
            captured_pcm: 已从audio_buffer取出的PCM音频数据
            duration_ms: 音频时长(ms)
            captured_images: 已从audio_buffer取出的图片数据
        """
        logger.info("Processing with Mode B (Omni audio)")
        
        logger.info(f"Audio buffer duration: {duration_ms:.0f}ms")
        
        # 将captured_pcm转为WAV base64
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.vad.sample_rate)
            wf.writeframes(captured_pcm)
        wav_b64 = base64.b64encode(buf.getvalue()).decode()
        
        messages = self._build_omni_messages(wav_b64, is_audio=True, captured_images=captured_images)
        
        resp_id = await self.protocol.send_response_created()
        self._current_resp_id = resp_id
        self.interruption.set_generating(True)
        
        try:
            logger.info(f"[TRACE] audio_to_llm: duration={duration_ms:.0f}ms, mode=B")
            
            full_text = await self._stream_llm_to_tts(messages, resp_id)
            
            # Only append to conversation if tool loop isn't handling it
            if full_text and not self.session_config.tools:
                self.conversation.append({"role": "assistant", "content": full_text})
            
            self.mode_router.report_omni_success()
            
        except asyncio.CancelledError:
            # 被用户打断，不继续处理
            logger.info("[Session] Mode B pipeline cancelled by interruption")
            raise
        except Exception as e:
            logger.error(f"Mode B error: {e}", exc_info=True)
            self.mode_router.report_omni_error()
            
            if self.mode_router.get_mode() == ModeRouter.MODE_A:
                logger.info("Falling back to Mode A after Omni error")
                await self.protocol.send_error("Omni error, falling back to ASR mode")
            else:
                await self.protocol.send_error("internal error")
            if self._current_resp_id:
                await self.protocol.send_response_done(self._current_resp_id, status="failed")
                self._current_resp_id = None
            return
        
        await self.protocol.send_transcript_done()
        await self.protocol.send_response_done(resp_id, status="completed")
        self._current_resp_id = None

    async def _process_mode_a(self, captured_pcm: bytes, duration_ms: float, captured_images: list[str] | None = None):
        """Mode A: ASR + LLM text + TTS output.
        
        Args:
            captured_pcm: 已从audio_buffer取出的PCM音频数据
            duration_ms: 音频时长(ms)
            captured_images: 已从audio_buffer取出的图片数据
        """
        logger.info("Processing with Mode A (ASR + LLM)")
        
        logger.info(f"Audio buffer duration: {duration_ms:.0f}ms")
        
        try:
            if self.local_asr is not None:
                # Use local ASR engine (SenseVoiceSmall on GPU)
                logger.info("[Mode A] Using local ASR engine for transcription")
                transcript = await self.local_asr.transcribe(
                    captured_pcm,
                    sample_rate=self.config.get("vad", "sample_rate", default=16000),
                    language=self.config.get("services", "asr", "language", default="zh"),
                )
            else:
                # Use remote ASR client
                logger.info("[Mode A] Using remote ASR client for transcription")
                transcript = await self.asr_client.transcribe_http(
                    captured_pcm,
                    sample_rate=self.config.get("vad", "sample_rate", default=16000),
                    language=self.config.get("services", "asr", "language", default="zh"),
                )
        except asyncio.CancelledError:
            # 被用户打断
            logger.info("[Session] Mode A ASR cancelled by interruption")
            raise
        except Exception as e:
            logger.error(f"ASR error: {e}", exc_info=True)
            await self.protocol.send_error("internal error")
            return
        
        logger.info(f"ASR transcript: {transcript}")
        
        if transcript:
            await self.protocol.send_input_transcript(transcript)
        
        messages = self._build_omni_messages(transcript, is_audio=False, captured_images=captured_images)
        
        resp_id = await self.protocol.send_response_created()
        self._current_resp_id = resp_id
        self.interruption.set_generating(True)
        
        try:
            logger.info(f"[TRACE] audio_to_llm: duration={duration_ms:.0f}ms, mode=A")
            
            full_text = await self._stream_llm_to_tts(messages, resp_id)
            
            # Only append to conversation if tool loop isn't handling it
            if full_text and not self.session_config.tools:
                self.conversation.append({"role": "assistant", "content": full_text})
            
        except asyncio.CancelledError:
            # 被用户打断
            logger.info("[Session] Mode A LLM pipeline cancelled by interruption")
            raise
        except Exception as e:
            logger.error(f"Mode A LLM error: {e}", exc_info=True)
            await self.protocol.send_error("internal error")
            if self._current_resp_id:
                await self.protocol.send_response_done(self._current_resp_id, status="failed")
                self._current_resp_id = None
            return
        
        await self.protocol.send_transcript_done()
        await self.protocol.send_response_done(resp_id, status="completed")
        self._current_resp_id = None

    async def _process_text_input(self):
        """Process text-only conversation input."""
        # _is_responding 已由调用方在 _response_lock 内原子性设置
        
        my_task = asyncio.current_task()
        
        try:
            messages = self._build_text_messages()
            
            resp_id = await self.protocol.send_response_created()
            self._current_resp_id = resp_id
            self.interruption.set_generating(True)
            
            try:
                full_text = await self._stream_llm_to_tts(messages, resp_id)
                
                # Only append to conversation if tool loop isn't handling it
                if full_text and not self.session_config.tools:
                    self.conversation.append({"role": "assistant", "content": full_text})
                
                await self.protocol.send_transcript_done()
                await self.protocol.send_response_done(resp_id, status="completed")
                self._current_resp_id = None
                
            except Exception as e:
                logger.error(f"Text input error: {e}", exc_info=True)
                await self.protocol.send_error("internal error")
                if self._current_resp_id:
                    await self.protocol.send_response_done(self._current_resp_id, status="failed")
                    self._current_resp_id = None
        finally:
            if self._active_pipeline_task is my_task:
                self._is_responding = False
                self._active_pipeline_task = None
            self.interruption.set_generating(False)

    def _build_base_messages(self) -> list[dict]:
        """Build base messages (system + conversation history) shared by all modes."""
        messages = []
        
        if self.session_config.instructions:
            messages.append({
                "role": "system",
                "content": self.session_config.instructions,
            })
        
        for msg in self.conversation[-10:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Tool role messages (from tool calling results)
            if role == "tool":
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                })
                continue
            
            # Assistant messages with tool_calls
            if role == "assistant" and msg.get("tool_calls"):
                messages.append({
                    "role": "assistant",
                    "content": content if isinstance(content, str) and content else None,
                    "tool_calls": msg["tool_calls"],
                })
                continue
            
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "input_text":
                            text_parts.append(part.get("text", ""))
                if text_parts:
                    messages.append({"role": role, "content": " ".join(text_parts)})
        
        return messages

    def _build_omni_messages(self, audio_or_text: str, is_audio: bool, captured_images: list[str] | None = None) -> list[dict]:
        """Build messages for Omni API call.
        
        Args:
            audio_or_text: WAV base64 (is_audio=True) 或文本 (is_audio=False)
            is_audio: 是否为音频输入
            captured_images: 已保存的图片数据（避免从已清空的audio_buffer取）
        """
        messages = self._build_base_messages()
        
        user_content = []
        
        if is_audio:
            user_content.append(self.omni_client.build_audio_message(audio_or_text))
        else:
            user_content.append(self.omni_client.build_text_message(audio_or_text))
        
        # 使用传入的captured_images而不是从audio_buffer取
        # 因为audio_buffer可能已被清空
        images = captured_images if captured_images is not None else []
        for img_b64 in images:
            user_content.append(self.omni_client.build_image_message(img_b64))
        
        if user_content:
            messages.append({"role": "user", "content": user_content})
        
        return messages

    def _build_text_messages(self) -> list[dict]:
        """Build messages for text-only Omni API call."""
        return self._build_base_messages()

    async def cleanup(self):
        """Clean up session resources."""
        logger.info("Cleaning up session")
        
        # 清理时也取消活跃的pipeline task
        if self._active_pipeline_task is not None and not self._active_pipeline_task.done():
            self._active_pipeline_task.cancel()
            try:
                await self._active_pipeline_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._active_pipeline_task = None
        
        if self._current_resp_id:
            await self.protocol.send_response_done(self._current_resp_id, status="cancelled")
            self._current_resp_id = None
        
        self._is_responding = False
        self._is_speech_active = False
        self.audio_buffer.reset()
        self.vad.reset()
        self.conversation.clear()
        
        await self.omni_client.close()
        await self.asr_client.close()
        await self.tts_pipeline.close()
        if self.gsv_tts_pipeline:
            await self.gsv_tts_pipeline.close()
        
        # Flush directive stripper on session cleanup
        if self.directive_stripper:
            self.directive_stripper.reset()
        
        self._vad_executor.shutdown(wait=False)
