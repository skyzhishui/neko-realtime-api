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
from .omni_client import OmniAudioClient
from .asr_client import SenseVoiceASRClient
from .local_asr import LocalASREngine
from .tts_pipeline import TTSPipeline, SentenceSplitter
from .tts_ws_pipeline import TTSWebSocketPipeline
from .gsv_tts_pipeline import GsvTtsPipeline
from .interruption import InterruptionHandler
from .config import ServerConfig

logger = logging.getLogger("realtime-server")


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
        
        # Session config
        self.session_config = SessionConfig(config=config)
        
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
            base_url=config.get("services", "omni", "base_url", default="http://localhost:8000"),
            model=config.get("services", "omni", "model", default="Qwen3-Omni-30B-A3B-Instruct-AWQ-8bit"),
        )
        
        self.asr_client = SenseVoiceASRClient(
            base_url=config.get("services", "asr", "base_url", default="http://localhost:8082"),
        )
        
        # Local ASR engine (singleton, shared across sessions)
        self.local_asr: LocalASREngine | None = None
        if config.get("services", "asr", "local_asr", default=False):
            asr_model_path = config.get("services", "asr", "asr_model_path", default=None)
            # Use existing singleton if available, otherwise create new one
            existing = LocalASREngine.get_instance()
            if existing is not None:
                self.local_asr = existing
                logger.info("[Session] Using existing LocalASREngine singleton")
            else:
                logger.info(f"[Session] Initializing LocalASREngine with model_path={asr_model_path}")
                self.local_asr = LocalASREngine(model_path=asr_model_path)
        
        self.tts_pipeline = TTSPipeline(
            base_url=config.get("services", "tts", "base_url", default="http://localhost:8091"),
            voice=self.session_config.voice,  # C1 fix: 使用 session_config.voice
            sample_rate_out=config.get("tts_pipeline", "output_sample_rate", default=24000),
            timeout_s=config.get("services", "tts", "timeout_s", default=15),
        )
        
        # GSV-TTS-Lite pipeline (optional, voice cloning TTS)
        self.gsv_tts_enabled = config.get("services", "gsv_tts", "enabled", default=False)
        if self.gsv_tts_enabled:
            self.gsv_tts_pipeline = GsvTtsPipeline(
                base_url=config.get("services", "gsv_tts", "base_url", default="http://localhost:8001"),
                speaker_audio=config.get("services", "gsv_tts", "speaker_audio", default=""),
                prompt_audio=config.get("services", "gsv_tts", "prompt_audio", default=""),
                prompt_text=config.get("services", "gsv_tts", "prompt_text", default=""),
                sample_rate_out=config.get("tts_pipeline", "output_sample_rate", default=24000),
                timeout_s=config.get("services", "gsv_tts", "timeout_s", default=30),
                speed=config.get("services", "gsv_tts", "speed", default=1.0),
            )
            logger.info(f"[Session] GSV-TTS-Lite enabled: base_url={self.gsv_tts_pipeline.base_url}")
        else:
            self.gsv_tts_pipeline = None
        
        # TTS mode: "http" (default, sentence-based) or "ws" (character-by-character streaming)
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
            await self.protocol.send_error(f"Error processing {event_type}: {str(e)}")

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
                     f"smart_turn_enabled={self.vad.smart_turn_enabled}")

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
        
        self._audio_chunk_count += 1
        if self._audio_chunk_count % 100 == 1:
            logger.debug(f"Audio stream active: chunk #{self._audio_chunk_count}, {len(audio_b64)} chars base64")
        
        # Decode audio
        try:
            pcm_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            logger.warning(f"Failed to decode audio: {e}")
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
                        await self.protocol.send_response_done(self._current_resp_id)
                        self._current_resp_id = None
                    
                    # 3. 设置 _is_responding = False
                    self._is_responding = False
                   
                    # 4. 把预缓冲数据存下来
                    prefix_audio = self.vad.get_prefix_audio()

                    # 5. 清空旧的 audio_buffer
                    self.audio_buffer.clear_audio()
                    
                    # 6. 先写回预缓冲音频（用户语音起始前的~300ms），再写当前帧                    
                    if prefix_audio:
                        self.audio_buffer.append_audio_raw(prefix_audio)
                    self.audio_buffer.append_audio_raw(pcm_bytes)
                    
                    # 7. 完全重置VAD状态（清空所有内部状态，包括LSTM、prefix_chunks等）
                    self.vad.reset()
                    logger.info(
                        f"[Session] Interruption: current frame ({len(pcm_bytes)} bytes) "
                        f"written to audio_buffer as speech start"
                    )
                    
                    # 7. 设置语音活跃状态
                    self._is_speech_active = True
                    
                    # 8. 发送 speech_started 给客户端
                    await self.protocol.send_speech_started()
                    
                    # 9. 初始化 Smart Turn 音频缓冲
                    if self.vad.smart_turn_enabled and self.vad._smart_turn is not None:
                        all_audio_pcm = self.audio_buffer.get_full_pcm()
                        if len(all_audio_pcm) > 0:
                            turn_audio_np = np.frombuffer(all_audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                            self.vad._turn_audio_chunks = [turn_audio_np.copy()]
                            
                            #turn_ms = len(turn_audio_np) / self.vad.sample_rate * 1000
                            #logger.info(
                            #    f"[Session] Interruption: Smart Turn buffer initialized "
                            #    f"with {len(turn_audio_np)} samples ({turn_ms:.0f}ms)"
                            #)
                    
                    logger.info("[Session] Interruption completed, VAD reset, ready for new speech input")
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
            self.audio_buffer.append_image(image_b64)

    async def _handle_conversation_item_create(self, event: dict):
        """Handle conversation.item.create event."""
        item = event.get("item", {})
        if item.get("type") == "message":
            role = item.get("role", "user")
            content = item.get("content", [])
            msg = {"role": role, "content": content}
            self.conversation.append(msg)
            logger.info(f"Added conversation item: role={role}")

    async def _handle_response_create(self, event: dict):
        """Handle response.create event."""
        async with self._response_lock:
            if self._is_responding:
                logger.warning("Response already in progress, ignoring response.create")
                return
            self._is_responding = True
        
        if self.audio_buffer.get_full_pcm():
            asyncio.create_task(self._process_speech_input())
        elif self.conversation:
            asyncio.create_task(self._process_text_input())

    async def _handle_response_cancel(self, event: dict):
        """Handle response.cancel event."""
        await self._cancel_active_pipeline("client_cancel")
        if self._current_resp_id:
            await self.protocol.send_response_done(self._current_resp_id)
            self._current_resp_id = None
        self._is_responding = False

    # ----------------------------------------------------------------
    # 公共流式管线方法：LLM → TTS 流式处理
    # ----------------------------------------------------------------

    async def _stream_llm_to_tts(self, messages: list[dict], resp_id: str):
        """公共 LLM→TTS 流式管线逻辑。

        根据 tts_mode 选择不同管线：
        - "http": 分句 HTTP 模式（原有逻辑，SentenceSplitter → stream_tts）
        - "ws": 逐字流式 WS 模式（LLM token 即时送 TTS，更低延迟）

        Args:
            messages: 发送给 LLM 的消息列表
            resp_id: 当前响应 ID（用于打断处理）

        Returns:
            str: LLM 生成的完整文本（用于写入 conversation history）
        """
        if self.tts_mode == "ws":
            return await self._stream_llm_to_tts_ws(messages, resp_id)
        else:
            return await self._stream_llm_to_tts_http(messages, resp_id)

    async def _stream_llm_to_tts_http(self, messages: list[dict], resp_id: str):
        """HTTP 模式：分句 → 逐句 HTTP TTS（流水线并行版）。
        
        改进：句子级流水线 — 句子 N+1 的 TTS 请求在句子 N 音频接收期间即可启动，
        通过 asyncio.Semaphore 限制并发数，音频按句子顺序输出。
        
        关键设计：
        1. TTS 任务通过 audio_ready_event 通知 drain 有新音频可用，
           不再仅依赖 LLM delta 触发 drain（LLM 结束后 drain 仍可被唤醒）。
        2. Phase 3 的阻塞 drain 对 CancelledError 有容错：
           被取消时仍尝试非阻塞地排空已就绪的音频，避免音频丢失。
        3. 队列设置 maxsize 防止 TTS 产出过快导致内存膨胀。
        """
        self.tts_pipeline.voice = self.session_config.voice
        full_text = ""
        splitter = SentenceSplitter(
            min_sub_sentence_len=self.config.get("tts_pipeline", "min_sub_sentence_len", default=6)
        )
        max_concurrent = self.config.get("tts_pipeline", "max_concurrent_tts", default=2)
        tts_semaphore = asyncio.Semaphore(max_concurrent)

        # 每个句子的音频队列（按句子顺序），有界队列防止内存膨胀
        sentence_audio_queues: list[asyncio.Queue[bytes | None]] = []
        # 跟踪所有 TTS 任务
        tts_tasks: list[asyncio.Task] = []
        # 下一个要 drain 的句子队列索引
        next_drain_idx = 0
        # TTS 产出音频时触发，让 Phase 1 的 drain 能及时响应
        audio_ready_event = asyncio.Event()

        async def _process_sentence_tts(sentence: str, queue: asyncio.Queue):
            """单个句子的 TTS 处理：获取 semaphore → 请求 TTS → 推入队列。"""
            async with tts_semaphore:
                tts_pipe = self.gsv_tts_pipeline if self.gsv_tts_enabled else self.tts_pipeline
                try:
                    async for pcm_chunk in tts_pipe.stream_tts(
                        sentence,
                        **({} if self.gsv_tts_enabled else {"instructions": ""}),
                    ):
                        await queue.put(pcm_chunk)
                        audio_ready_event.set()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"TTS error for sentence: {e}", exc_info=True)
                finally:
                    await queue.put(None)
                    audio_ready_event.set()

        async def _drain_ready_queues():
            """非阻塞地消费已就绪的音频队列，按句子顺序输出。
            
            对当前 drain 指针指向的队列，持续取 chunk 直到队列暂时为空
            或收到 sentinel（None）。遇到空队列时立即返回，不阻塞等待。
            """
            nonlocal next_drain_idx
            while next_drain_idx < len(sentence_audio_queues):
                q = sentence_audio_queues[next_drain_idx]
                while True:
                    try:
                        chunk = q.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    if chunk is None:
                        break
                    await self.protocol.send_audio_delta(chunk)
                next_drain_idx += 1

        async def _drain_remaining_blocking():
            """阻塞等待并排空所有剩余队列（Phase 3）。
            
            对 CancelledError 容错：被取消时仍尝试非阻塞地发送已就绪的音频，
            而不是直接丢弃。这确保即使 pipeline 被打断（如用户新一轮语音），
            已产出的 TTS 音频仍能送达客户端。
            """
            nonlocal next_drain_idx
            while next_drain_idx < len(sentence_audio_queues):
                q = sentence_audio_queues[next_drain_idx]
                while True:
                    try:
                        chunk = await q.get()
                    except asyncio.CancelledError:
                        logger.info(
                            f"[Session] Phase 3 drain cancelled at sentence {next_drain_idx}, "
                            f"draining available chunks non-blocking"
                        )
                        while True:
                            try:
                                remaining = q.get_nowait()
                            except asyncio.QueueEmpty:
                                return
                            if remaining is None:
                                break
                            try:
                                await self.protocol.send_audio_delta(remaining)
                            except (asyncio.CancelledError, Exception):
                                return
                        next_drain_idx += 1
                        continue
                    if chunk is None:
                        break
                    await self.protocol.send_audio_delta(chunk)
                next_drain_idx += 1

        try:
            # Phase 1: LLM streaming + pipelined TTS + event-driven audio drain
            async for text_delta in self.omni_client.stream_chat(
                messages=messages,
                temperature=self.session_config.temperature,
                repetition_penalty=self.session_config.repetition_penalty,
                max_tokens=self.config.get("services", "omni", "max_tokens", default=4096),
                timeout_s=self.config.get("services", "omni", "timeout_s", default=30),
            ):
                full_text += text_delta
                await self.protocol.send_transcript_delta(text_delta)

                sentences = splitter.add_text(text_delta)
                for sentence in sentences:
                    logger.info(f'[TRACE] tts_sentence: text="{sentence}", sentence_len={len(sentence)}')
                    q = asyncio.Queue(maxsize=50)
                    sentence_audio_queues.append(q)
                    task = asyncio.create_task(_process_sentence_tts(sentence, q))
                    tts_tasks.append(task)

                # 每次 LLM delta 后 drain 已就绪的队列
                await _drain_ready_queues()

                # 如果 TTS 已产出音频但 LLM delta 间隔较长，
                # 通过 event 等待一小段时间让 drain 有机会处理
                if audio_ready_event.is_set():
                    audio_ready_event.clear()
                    await _drain_ready_queues()

            # Phase 2: Flush remaining sentences from splitter
            for sentence in splitter.flush():
                logger.info(f'[TRACE] tts_sentence: text="{sentence}", sentence_len={len(sentence)}(flush)')
                q = asyncio.Queue(maxsize=50)
                sentence_audio_queues.append(q)
                task = asyncio.create_task(_process_sentence_tts(sentence, q))
                tts_tasks.append(task)

            logger.info(
                f"[Session] Phase 3 starting: {len(sentence_audio_queues)} sentences, "
                f"next_drain_idx={next_drain_idx}"
            )

            # Phase 3: 阻塞等待所有剩余队列完成（LLM 已结束，TTS task 全部启动）
            await _drain_remaining_blocking()

            # Wait for all TTS tasks to complete (they should be done after queues drained)
            if tts_tasks:
                await asyncio.gather(*tts_tasks, return_exceptions=True)

        except asyncio.CancelledError:
            # Pipeline 被取消（用户打断），CancelledError 已传播到此处
            logger.info("[Session] HTTP TTS pipeline cancelled (CancelledError propagated)")
            # Cancel all pending TTS tasks
            for task in tts_tasks:
                if not task.done():
                    task.cancel()
            raise

        return full_text

    async def _stream_llm_to_tts_ws(self, messages: list[dict], resp_id: str):
        """WS 模式：逐字流式 TTS，LLM token 即时送 TTS。
        
        重写要点：
        - 去掉所有 should_cancel() 检查
        - 利用 CancelledError 传播实现即时取消
        - 在 CancelledError handler 中关闭 TTS WS 连接
        - receive_task 也需要取消

        核心流程（并行优化版）：
        1. ASR结果送LLM时，同时建立WS连接（发送session.config）
        2. LLM出首token后，立即开始送文本给TTS
        3.【并行】TTS音频接收与LLM流式输出同时进行，首chunk出来马上播放
        4. LLM输出完成后，发送input.done
        5. 等待TTS音频全部接收完毕
        """
        t0 = time.time()  # 基准时间：ASR结果送LLM的时刻
        full_text = ""
        ws_pipeline = TTSWebSocketPipeline(
            ws_url=self.tts_ws_url,
            voice=self.session_config.voice,  # C1 fix: 使用 session_config.voice
            sample_rate=self.config.get("services", "tts", "sample_rate", default=24000),
            language=self.config.get("services", "tts", "language", default="Chinese"),
            timeout_s=self.config.get("services", "tts", "timeout_s", default=15),
        )

        # 并行音频接收任务的状态
        tts_first_chunk_received = False
        tts_receive_done = asyncio.Event()
        tts_error: list[str | None] = [None]  # 用list包装以便在闭包中修改

        async def _receive_and_send_audio():
            """并行任务：从TTS WS接收音频块并立即发送给客户端。"""
            nonlocal tts_first_chunk_received
            try:
                async for pcm_chunk in ws_pipeline.receive_audio():
                    # 埋点：TTS首chunk
                    if not tts_first_chunk_received:
                        tts_first_chunk_received = True
                        latency_ms = (time.time() - t0) * 1000
                        logger.info(f'[TRACE] tts_ws_first_chunk: latency={latency_ms:.1f}ms, chunk_size={len(pcm_chunk)}')

                    await self.protocol.send_audio_delta(pcm_chunk)
            except asyncio.CancelledError:
                # 被取消时，直接退出，不继续接收音频
                logger.info("[Session] TTS WS receive_and_send_audio cancelled")
                raise
            except Exception as e:
                tts_error[0] = str(e)
                logger.error(f"TTS WS receive_and_send error: {e}", exc_info=True)
            finally:
                tts_receive_done.set()

        receive_task = None
        try:
            # 1. 建连：ASR结果送LLM时就建连
            await ws_pipeline.connect()
            logger.info(f'[TRACE] tts_ws_connected: latency={(time.time() - t0) * 1000:.1f}ms')

            # 2. 启动并行音频接收任务（LLM输出前就启动，确保首chunk出来立刻消费）
            receive_task = asyncio.create_task(_receive_and_send_audio())

            # 3. LLM 流式输出 + 即时送 TTS（与音频接收并行）
            llm_first_token = False
            tts_first_text_sent = False

            async for text_delta in self.omni_client.stream_chat(
                messages=messages,
                temperature=self.session_config.temperature,
                repetition_penalty=self.session_config.repetition_penalty,
                max_tokens=self.config.get("services", "omni", "max_tokens", default=4096),
                timeout_s=self.config.get("services", "omni", "timeout_s", default=30),
            ):
                full_text += text_delta
                await self.protocol.send_transcript_delta(text_delta)

                # 埋点：LLM首token
                if not llm_first_token:
                    llm_first_token = True
                    latency_ms = (time.time() - t0) * 1000
                    logger.info(f'[TRACE] llm_first_token: latency={latency_ms:.1f}ms')

                # 即时送文本给TTS WS
                await ws_pipeline.send_text_delta(text_delta)

                # 埋点：首次送TTS文本
                if not tts_first_text_sent:
                    tts_first_text_sent = True
                    latency_ms = (time.time() - t0) * 1000
                    logger.info(f'[TRACE] tts_ws_text_sent: latency={latency_ms:.1f}ms, text="{text_delta[:20]}"')

            # 4. LLM输出完成，发送input.done
            await ws_pipeline.finish_input()

            # 5. 等待TTS音频全部接收完毕
            await receive_task

            if tts_error[0]:
                raise RuntimeError(f"TTS receive error: {tts_error[0]}")

            # 埋点：TTS播放完成
            latency_ms = (time.time() - t0) * 1000
            logger.info(
                f'[TRACE] tts_ws_done: latency={latency_ms:.1f}ms, '
                f'total_sentences={ws_pipeline.total_sentences}'
            )

        except asyncio.CancelledError:
            # Pipeline 被取消（用户打断），立即关闭所有连接
            logger.info("[Session] TTS WS pipeline cancelled by interruption")
            # 取消接收任务
            if receive_task and not receive_task.done():
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
            # 关闭WS连接（在finally中也会关闭，这里提前关闭确保音频不再发送）
            await ws_pipeline.close()
            raise  # 重新抛出，让 _process_speech_input 的 except CancelledError 处理
        except Exception as e:
            logger.error(f"TTS WS pipeline error: {e}", exc_info=True)
            raise
        finally:
            # 确保WS连接被关闭
            if ws_pipeline.is_connected:
                await ws_pipeline.close()

        return full_text

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
        
        try:
            self._is_speech_active = False
            
            mode = self.mode_router.get_mode()
            
            if mode == ModeRouter.MODE_B:
                await self._process_mode_b(captured_pcm, captured_duration_ms, captured_images)
            else:
                await self._process_mode_a(captured_pcm, captured_duration_ms, captured_images)
        except asyncio.CancelledError:
            # Pipeline 被取消（用户打断），不发送 response.done（由取消方负责）
            logger.info("[Session] _process_speech_input cancelled, pipeline interrupted")
            raise
        except Exception as e:
            logger.error(f"Error processing speech input: {e}", exc_info=True)
            await self.protocol.send_error(f"Processing error: {str(e)}")
        finally:
            self._is_responding = False
            self.interruption.set_generating(False)
            self._is_speech_active = False

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
            
            if full_text:
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
                await self.protocol.send_error(f"Mode B error: {str(e)}")
            # 异常时不发送 transcript_done / response.done
            # 客户端通过 error 事件感知错误
            self._current_resp_id = None
            return
        
        await self.protocol.send_transcript_done()
        await self.protocol.send_response_done(resp_id)
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
            await self.protocol.send_error(f"ASR error: {str(e)}")
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
            
            if full_text:
                self.conversation.append({"role": "assistant", "content": full_text})
            
        except asyncio.CancelledError:
            # 被用户打断
            logger.info("[Session] Mode A LLM pipeline cancelled by interruption")
            raise
        except Exception as e:
            logger.error(f"Mode A LLM error: {e}", exc_info=True)
            await self.protocol.send_error(f"Mode A LLM error: {str(e)}")
            # 异常时不发送 transcript_done / response_done
            self._current_resp_id = None
            return
        
        await self.protocol.send_transcript_done()
        await self.protocol.send_response_done(resp_id)
        self._current_resp_id = None

    async def _process_text_input(self):
        """Process text-only conversation input."""
        # _is_responding 已由调用方在 _response_lock 内原子性设置
        
        try:
            messages = self._build_text_messages()
            
            resp_id = await self.protocol.send_response_created()
            self._current_resp_id = resp_id
            self.interruption.set_generating(True)
            
            try:
                full_text = await self._stream_llm_to_tts(messages, resp_id)
                
                if full_text:
                    self.conversation.append({"role": "assistant", "content": full_text})
                
                await self.protocol.send_transcript_done()
                await self.protocol.send_response_done(resp_id)
                self._current_resp_id = None
                
            except Exception as e:
                logger.error(f"Text input error: {e}", exc_info=True)
                await self.protocol.send_error(f"Processing error: {str(e)}")
        finally:
            self._is_responding = False
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
            await self.protocol.send_response_done(self._current_resp_id)
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
        
        self._vad_executor.shutdown(wait=False)
