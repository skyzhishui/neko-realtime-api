"""Silero VAD Module for voice activity detection + Smart Turn v3.2 endpoint detection.

VAD 推理优先级：ONNX → PyTorch（降级）
- ONNX Runtime 推理延迟更低，优先使用
- ONNX 加载/推理失败时降级到 PyTorch
- 不使用能量检测降级方案

新增策略：
- min_speech_duration: 最短语音时长限制，避免噪声误触发
- max_audio_duration: 最大音频时长硬限制，超时强制截断
- silence_timeout: 静音超时，检测到持续静音后判定用户说完
- Smart Turn v3.2: 静音超时后二次判断话轮是否结束（Complete/Incomplete）

模型路径配置（统一目录方式）：
- silero_model_path: 指向包含 silero_vad.onnx 和 silero_vad.jit 的目录
  未配置时默认通过 modelscope 缓存目录获取
- smart_turn_path: 指向包含 smart-turn-v3.2-gpu.onnx 的目录
  未配置时默认通过 modelscope 缓存目录获取
- 目标文件不存在时自动通过 modelscope 下载

VAD 埋点日志：
- speech_started: INFO 级别，打印 speech_prob 和累积音频时长
- speech_stopped: INFO 级别，打印静音持续时长和总语音时长
- max_duration_reached: INFO 级别，打印当前累积音频时长
- Smart Turn 判断: INFO 级别，打印 prediction/probability/耗时
- VAD prob 周期性输出: DEBUG 级别，每隔约2秒(60帧)输出 avg_prob 和 speech_ratio
"""
import numpy as np
from collections import deque
import logging
import time
import os

logger = logging.getLogger("realtime-server")


def _ensure_model_downloaded(model_id: str, target_dir: str, required_files: list[str]) -> str:
    """确保模型文件已下载到指定目录。
    
    如果目标目录下所有 required_files 都存在，则跳过下载；
    否则通过 modelscope snapshot_download 下载到 target_dir。
    
    Args:
        model_id: modelscope 模型 ID，如 "manyeyes/silero-vad-onnx"
        target_dir: 目标目录路径
        required_files: 必须存在的文件名列表
        
    Returns:
        target_dir 的绝对路径
    """
    target_dir = os.path.abspath(target_dir)
    
    # 检查所有必需文件是否已存在
    all_exist = all(
        os.path.isfile(os.path.join(target_dir, f)) for f in required_files
    )
    
    if all_exist:
        logger.info(f"[ModelDownload] 所有必需文件已存在，跳过下载: {target_dir}")
        return target_dir
    
    # 需要下载
    missing = [f for f in required_files if not os.path.isfile(os.path.join(target_dir, f))]
    logger.info(
        f"[ModelDownload] 缺少文件 {missing}，正在从 modelscope 下载 "
        f"model_id={model_id} → {target_dir}"
    )
    
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        downloaded_dir = snapshot_download(
            model_id=model_id,
            local_dir=target_dir,
        )
        logger.info(f"[ModelDownload] 下载完成: {downloaded_dir}")
        return downloaded_dir
    except Exception as e:
        logger.error(f"[ModelDownload] modelscope 下载失败: {e}")
        # 如果下载失败但目录已存在部分文件，仍然返回目录路径
        # 让后续加载逻辑自行处理
        if os.path.isdir(target_dir):
            return target_dir
        raise


def _get_modelscope_cache_dir(model_id: str) -> str:
    """获取 modelscope 缓存目录路径。
    
    使用 modelscope 的默认缓存机制获取模型目录。
    如果本地已缓存则直接返回路径，否则返回默认缓存路径。
    
    Args:
        model_id: modelscope 模型 ID
        
    Returns:
        模型缓存目录路径
    """
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        # local_files_only=True: 只从本地缓存获取，不下载
        cached_dir = snapshot_download(
            model_id=model_id,
            local_files_only=True,
        )
        return cached_dir
    except Exception:
        # 本地没有缓存，返回 modelscope 默认缓存路径
        try:
            from modelscope.hub.file_download import get_model_cache_dir
            return get_model_cache_dir(model_id)
        except ImportError:
            pass
        # 最终 fallback
        cache_root = os.path.expanduser("~/.cache/modelscope/hub")
        return os.path.join(cache_root, model_id)


class SmartTurnDetector:
    """Smart Turn v3.2 话轮结束检测器。
    
    在 VAD 检测到静音超时后，对整段语音做二次判断：
    - Complete (probability > threshold): 用户说完了，触发 speech_stopped
    - Incomplete (probability <= threshold): 用户在思考/犹豫，继续等待
    
    技术规格：
    - 基座: Whisper Tiny (~8M参数) + 线性分类器
    - 输入: 16kHz mono PCM float32, 最多8秒（不足前补零，超长截取末尾8秒）
    - 输出: logits → sigmoid概率, >0.5 = Complete, <=0.5 = Incomplete
    - ONNX输入名: "input_features", 形状 (1, 80, 800) — Whisper mel spectrogram (chunk_length=8)
    - 特征提取器: WhisperFeatureExtractor(chunk_length=8)
    - 注意: chunk_length=8 产生 (1,80,800), chunk_length=30 产生 (1,80,3000)
      模型实际需要 (1,80,800)，不要混淆！
    """
    
    # ONNX Provider 优先级
    _PROVIDER_PRIORITY = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    # 最大输入音频长度（秒）
    MAX_AUDIO_SECONDS = 8
    # 目标采样率
    TARGET_SAMPLE_RATE = 16000
    # 最大样本数
    MAX_SAMPLES = MAX_AUDIO_SECONDS * TARGET_SAMPLE_RATE  # 128000
    # 模型输入特征形状 (batch, n_mels, time_frames)
    FEATURE_SHAPE = (1, 80, 800)
    
    # modelscope 模型 ID
    _MODELSCOPE_MODEL_ID = "pipecat-ai/smart-turn-v3"
    # 必需文件
    _REQUIRED_FILES = ["smart-turn-v3.2-gpu.onnx"]
    
    def __init__(self, model_path: str | None = None, threshold: float = 0.5):
        """初始化 Smart Turn 检测器。
        
        Args:
            model_path: Smart Turn 模型目录路径。目录下应有 smart-turn-v3.2-gpu.onnx。
                        为 None 时使用 modelscope 缓存目录。
            threshold: 话轮结束判断阈值，sigmoid 输出 > threshold 为 Complete。
        
        Raises:
            RuntimeError: 模型加载失败时抛出异常
        """
        self.threshold = threshold
        self._session = None
        self._feature_extractor = None
        self._provider = None
        self._model_path = None
        
        # 确定模型目录
        if model_path:
            model_dir = model_path
        else:
            model_dir = _get_modelscope_cache_dir(self._MODELSCOPE_MODEL_ID)
        
        # 确保模型文件已下载
        model_dir = _ensure_model_downloaded(
            model_id=self._MODELSCOPE_MODEL_ID,
            target_dir=model_dir,
            required_files=self._REQUIRED_FILES,
        )
        
        # 加载特征提取器
        self._load_feature_extractor()
        
        # 加载 ONNX 模型
        self._load_model(model_dir)
        
        logger.info(
            f"[SmartTurn] initialized: model={self._model_path}, "
            f"provider={self._provider}, threshold={threshold}"
        )
    
    @classmethod
    def from_preloaded(
        cls,
        onnx_session,
        feature_extractor,
        provider: str | None = None,
        model_path: str | None = None,
        threshold: float = 0.5,
    ) -> "SmartTurnDetector":
        """Create a SmartTurnDetector from pre-loaded model artifacts.

        This skips model download and loading, using shared artifacts from ModelManager.
        The ONNX session and feature extractor are shared (thread-safe for inference),
        only the threshold is per-instance.

        Args:
            onnx_session: Pre-loaded onnxruntime.InferenceSession (shared, read-only)
            feature_extractor: Pre-loaded WhisperFeatureExtractor (shared, stateless)
            provider: ONNX provider name (e.g. "CUDAExecutionProvider")
            model_path: Model file path (for logging)
            threshold: Prediction threshold

        Returns:
            SmartTurnDetector instance with shared model artifacts
        """
        instance = cls.__new__(cls)
        instance.threshold = threshold
        instance._session = onnx_session
        instance._feature_extractor = feature_extractor
        instance._provider = provider
        instance._model_path = model_path
        
        logger.info(
            f"[SmartTurn] initialized from preloaded: model={model_path}, "
            f"provider={provider}, threshold={threshold} (shared artifacts)"
        )
        return instance
    
    def _load_feature_extractor(self):
        """加载 WhisperFeatureExtractor。"""
        try:
            from transformers import WhisperFeatureExtractor
            # chunk_length=8 → 产生 (1, 80, 800) 特征，匹配模型输入
            self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)
            logger.info("[SmartTurn] WhisperFeatureExtractor loaded (chunk_length=8)")
        except Exception as e:
            raise RuntimeError(
                f"[SmartTurn] WhisperFeatureExtractor 加载失败: {e}. "
                f"请安装 transformers: pip install transformers"
            )
    
    def _load_model(self, model_dir: str):
        """加载 Smart Turn ONNX 模型。
        
        从 model_dir 目录下查找 smart-turn-v3.2-gpu.onnx 文件。
        Provider 优先级: CUDAExecutionProvider → CPUExecutionProvider
        
        Raises:
            RuntimeError: 模型加载失败时抛出异常
        """
        try:
            import onnxruntime
        except ImportError:
            raise RuntimeError(
                "[SmartTurn] onnxruntime 未安装. 请安装: pip install onnxruntime-gpu"
            )
        
        # 构建模型文件路径
        onnx_path = os.path.join(model_dir, "smart-turn-v3.2-gpu.onnx")
        
        if not os.path.isfile(onnx_path):
            raise RuntimeError(
                f"[SmartTurn] ONNX 模型文件不存在: {onnx_path}"
            )
        
        # 尝试不同 Provider
        errors = []
        for provider in self._PROVIDER_PRIORITY:
            try:
                available_providers = onnxruntime.get_available_providers()
                if provider not in available_providers:
                    errors.append(f"{onnx_path} + {provider}: provider 不可用")
                    continue
                
                providers = [provider]
                # 始终添加 CPU 作为 fallback provider
                if provider != "CPUExecutionProvider" and "CPUExecutionProvider" in available_providers:
                    providers.append("CPUExecutionProvider")
                
                session = onnxruntime.InferenceSession(onnx_path, providers=providers)
                
                # 验证输入
                input_info = session.get_inputs()
                if len(input_info) != 1 or input_info[0].name != "input_features":
                    raise RuntimeError(
                        f"ONNX 输入异常: 期望 input_features, 实际 {[i.name for i in input_info]}"
                    )
                
                # 验证输入形状维度（动态维度可能是字符串，只验证维度数量）
                actual_shape = tuple(input_info[0].shape)
                if len(actual_shape) != 3:
                    raise RuntimeError(
                        f"ONNX 输入形状维度异常: 期望3维, 实际{len(actual_shape)}维"
                    )
                
                # 语义验证：用正确形状的零输入测试
                test_input = np.zeros(self.FEATURE_SHAPE, dtype=np.float32)
                test_out = session.run(None, {"input_features": test_input})
                if len(test_out) < 1:
                    raise RuntimeError("ONNX 输出为空")
                
                # 输出名是 "logits", shape (1, 1)
                logits_val = float(test_out[0].flatten()[0])
                # logits 经过 sigmoid 后应在 [0, 1]
                sigmoid_val = 1.0 / (1.0 + np.exp(-logits_val))
                if not (0.0 <= sigmoid_val <= 1.0):
                    raise RuntimeError(
                        f"ONNX 语义验证失败: logits={logits_val:.4f}, sigmoid={sigmoid_val:.4f} 不在 [0,1] 范围"
                    )
                
                # 加载成功
                self._session = session
                self._provider = provider
                self._model_path = onnx_path
                logger.info(
                    f"[SmartTurn] ✅ 模型加载成功: {onnx_path}, provider={provider}, "
                    f"test_logits={logits_val:.4f}, sigmoid={sigmoid_val:.4f}"
                )
                return
                
            except Exception as e:
                errors.append(f"{onnx_path} + {provider}: {e}")
                continue
        
        # 所有 Provider 均失败
        error_detail = "\n  - ".join(errors)
        raise RuntimeError(
            f"[SmartTurn] ❌ 模型加载失败:\n  - {error_detail}\n"
            f"请确认模型文件存在且 onnxruntime-gpu 已正确安装"
        )
    
    def predict_endpoint(self, audio_float32: np.ndarray) -> dict:
        """对一段音频预测话轮是否结束。
        
        Args:
            audio_float32: float32 numpy array, 16kHz mono, 值范围 [-1, 1]
        
        Returns:
            dict: {
                "prediction": 0 | 1,  # 0=Incomplete, 1=Complete
                "probability": float,  # sigmoid(logits) 输出, >0.5 为 Complete
                "label": "Complete" | "Incomplete",
                "inference_ms": float,  # ONNX推理耗时(ms)
            }
        """
        t_start = time.perf_counter()
        
        # 1. 音频预处理: 截取末尾8秒 + 前补零
        processed = self._preprocess_audio(audio_float32)
        
        # 2. 特征提取: Whisper mel spectrogram → shape (1, 80, 800)
        features = self._extract_features(processed)
        
        # 3. ONNX 推理
        t_infer_start = time.perf_counter()
        outputs = self._session.run(None, {"input_features": features})
        t_infer_end = time.perf_counter()
        inference_ms = (t_infer_end - t_infer_start) * 1000
        
        # 4. 解析输出: logits → sigmoid → probability
        logits = float(outputs[0].flatten()[0])
        prob = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        # 确保概率在合理范围
        prob = max(0.0, min(1.0, prob))
        
        prediction = 1 if prob > self.threshold else 0
        label = "Complete" if prediction == 1 else "Incomplete"
        
        total_ms = (time.perf_counter() - t_start) * 1000
        
        logger.info(
            f"[SmartTurn] prediction={label}, probability={prob:.4f}, "
            f"logits={logits:.4f}, threshold={self.threshold}, "
            f"inference={inference_ms:.1f}ms, total={total_ms:.1f}ms, "
            f"audio_len={len(audio_float32)/self.TARGET_SAMPLE_RATE:.2f}s"
        )
        
        return {
            "prediction": prediction,
            "probability": prob,
            "label": label,
            "inference_ms": inference_ms,
        }
    
    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """音频预处理: 截取末尾8秒 + 不足8秒前补零。
        
        Args:
            audio: float32 numpy array, 16kHz mono
        
        Returns:
            float32 numpy array, 长度恰好为 MAX_SAMPLES (128000)
        """
        if len(audio) > self.MAX_SAMPLES:
            # 超长: 截取末尾8秒
            audio = audio[-self.MAX_SAMPLES:]
        
        if len(audio) < self.MAX_SAMPLES:
            # 不足: 前补零
            pad_len = self.MAX_SAMPLES - len(audio)
            audio = np.concatenate([
                np.zeros(pad_len, dtype=np.float32),
                audio
            ])
        
        return audio
    
    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """提取 Whisper mel spectrogram 特征。
        
        Args:
            audio: float32 numpy array, 长度 MAX_SAMPLES (128000)
        
        Returns:
            float32 numpy array, shape (1, 80, 800)
        """
        features = self._feature_extractor(
            audio,
            sampling_rate=self.TARGET_SAMPLE_RATE,
            return_tensors="np",
        ).input_features
        
        # 确保形状正确: chunk_length=8 应产生 (1, 80, 800)
        if features.shape != self.FEATURE_SHAPE:
            # 如果特征提取器返回的形状不对，尝试调整
            logger.warning(
                f"[SmartTurn] 特征形状异常: 期望 {self.FEATURE_SHAPE}, "
                f"实际 {features.shape}, 尝试调整"
            )
            if features.ndim == 3 and features.shape[0] == 1 and features.shape[1] == 80:
                target_frames = self.FEATURE_SHAPE[2]
                if features.shape[2] < target_frames:
                    pad = np.zeros(
                        (1, 80, target_frames - features.shape[2]),
                        dtype=np.float32
                    )
                    features = np.concatenate([features, pad], axis=2)
                elif features.shape[2] > target_frames:
                    features = features[:, :, :target_frames]
            elif features.ndim == 2:
                features = features.reshape(1, 80, -1)
                target_frames = self.FEATURE_SHAPE[2]
                if features.shape[2] < target_frames:
                    pad = np.zeros(
                        (1, 80, target_frames - features.shape[2]),
                        dtype=np.float32
                    )
                    features = np.concatenate([features, pad], axis=2)
                elif features.shape[2] > target_frames:
                    features = features[:, :, :target_frames]
        
        return features.astype(np.float32)


class SileroVADModule:
    """Silero VAD based voice activity detection + Smart Turn endpoint detection.
    
    Processes PCM16 16kHz audio frames and detects speech_started/speech_stopped events.
    Buffers small incoming frames until reaching Silero's minimum chunk size (512 samples).
    
    VAD backend priority: ONNX → PyTorch (fallback)
    
    模型路径配置（统一目录方式）：
    - silero_model_path: 指向包含 silero_vad.onnx 和 silero_vad.jit 的目录
      ONNX 和 PyTorch 降级都从这个目录加载
    - smart_turn_path: 指向包含 smart-turn-v3.2-gpu.onnx 的目录
    
    Smart Turn integration:
    - When silence_timeout is reached, Smart Turn v3.2 is called as a second-pass check
    - If Smart Turn judges "Complete" → trigger speech_stopped (user finished speaking)
    - If Smart Turn judges "Incomplete" → reset silence counter, continue waiting (user is thinking)
    - Smart Turn only runs on silence_timeout, does NOT affect real-time VAD flow
    """
    
    CHUNK_SIZE = 512  # 32ms @ 16kHz, Silero VAD minimum processing unit
    
    # VAD prob 周期性输出间隔（帧数），约2秒 = 60帧 @ 32ms/帧
    _PROB_LOG_INTERVAL_FRAMES = 60
    
    # modelscope 模型 ID
    _SILERO_MODELSCOPE_MODEL_ID = "manyeyes/silero-vad-v6-onnx"
    # silero 必需文件
    _SILERO_REQUIRED_FILES = ["silero_vad.onnx"]
    
    def __init__(
        self,
        threshold: float = 0.7,
        silence_ms: int = 500,
        prefix_padding_ms: int = 300,
        sample_rate: int = 16000,
        silero_model_path: str | None = None,
        min_speech_duration_ms: int = 300,
        max_audio_duration_ms: int = 30000,
        smart_turn_enabled: bool = True,
        smart_turn_path: str | None = None,
        smart_turn_threshold: float = 0.5,
    ):
        self.threshold = threshold
        self.silence_duration_ms = silence_ms  # unified parameter (see M3)
        self.prefix_padding_ms = prefix_padding_ms
        self.sample_rate = sample_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_audio_duration_ms = max_audio_duration_ms
        self.smart_turn_enabled = smart_turn_enabled
        self.smart_turn_threshold = smart_turn_threshold
        
        # ---- 确定 silero 模型目录 ----
        if silero_model_path:
            silero_dir = silero_model_path
        else:
            silero_dir = _get_modelscope_cache_dir(self._SILERO_MODELSCOPE_MODEL_ID)
        
        # 确保 silero 模型文件已下载
        self._silero_model_dir = _ensure_model_downloaded(
            model_id=self._SILERO_MODELSCOPE_MODEL_ID,
            target_dir=silero_dir,
            required_files=self._SILERO_REQUIRED_FILES,
        )
        
        # ---- 确定 smart_turn 模型目录 ----
        if smart_turn_path:
            smart_turn_dir = smart_turn_path
        else:
            smart_turn_dir = _get_modelscope_cache_dir(SmartTurnDetector._MODELSCOPE_MODEL_ID)
        
        # smart_turn 模型下载在 SmartTurnDetector 初始化时处理
        self._smart_turn_model_dir = smart_turn_dir
        
        # ---- VAD 后端状态 ----
        self._onnx_session = None   # ONNX InferenceSession
        self._model = None          # PyTorch Silero VAD 模型
        self._vad_backend = "none"  # 当前使用的后端: "onnx" / "pytorch" / "none"
        
        # ONNX 流式推理状态
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context_size = 64 if sample_rate == 16000 else 32
        self._context = np.zeros((1, self._context_size), dtype=np.float32)
        
        # ---- 语音检测状态 ----
        self.is_speaking = False
        
        # min_speech_duration: 语音开始后必须持续的帧数
        self._min_speech_frames = int(
            min_speech_duration_ms / (self.CHUNK_SIZE / sample_rate * 1000)
        )
        self._speech_candidate_frames = 0  # 语音候选帧计数
        self._speech_confirmed = False      # 语音是否已确认（超过最短时长）
        
        # max_audio_duration: 最大音频累积帧数
        self._max_audio_frames = int(
            max_audio_duration_ms / (self.CHUNK_SIZE / sample_rate * 1000)
        )
        self._total_speech_frames = 0  # 说话期间累积帧数
        
        # silence_timeout: 静音超时帧数
        self._silence_timeout_frames = int(
            silence_ms / (self.CHUNK_SIZE / sample_rate * 1000)
        )
        self._silence_timeout_count = 0  # 说话后静音帧计数
        
        # Prefix retention: audio kept before speech starts
        self.prefix_chunks = deque(
            maxlen=int(prefix_padding_ms / (self.CHUNK_SIZE / sample_rate * 1000))
        )
        
        # Buffer for accumulating small frames to reach CHUNK_SIZE
        self._sample_buffer = np.array([], dtype=np.float32)
        
        # ---- Smart Turn 话轮检测 ----
        self._smart_turn = None  # SmartTurnDetector 实例
        self._turn_audio_buffer = np.array([], dtype=np.float32)  # 语音期音频累积
        
        if self.smart_turn_enabled:
            self._init_smart_turn()
        
        # ---- VAD 埋点日志状态 ----
        self._frame_count = 0              # 总处理帧数（用于周期性输出）
        self._prob_window = deque(maxlen=60)  # 概率滑动窗口（用于周期性输出 avg_prob）
        # _prob_window maxlen=60 handled by deque (约2秒)
        self._speech_frame_count_in_window = 0  # 窗口内语音帧数（用于 speech_ratio）
        self._speech_start_time = None     # 语音开始时间戳
        self._speech_start_prob = 0.0      # 语音确认时的 speech_prob
        self._last_silence_start_time = None  # 说话中最后一次检测到静音的时间戳
        
        # ---- 加载模型 ----
        self._load_model()
        
        logger.info(
            f"VAD initialized: backend={self._vad_backend}, threshold={threshold}, "
            f"silence_duration_ms={silence_ms}, "
            f"min_speech_duration_ms={min_speech_duration_ms}, "
            f"max_audio_duration_ms={max_audio_duration_ms}, "
            f"silence_timeout_ms={silence_ms}, "
            f"prefix_chunks_maxlen={self.prefix_chunks.maxlen}, "
            f"smart_turn_enabled={self.smart_turn_enabled}, "
            f"silero_model_dir={self._silero_model_dir}, "
            f"smart_turn_model_dir={self._smart_turn_model_dir}"
        )
    
    @classmethod
    def from_preloaded(
        cls,
        onnx_session,
        pytorch_model,
        vad_backend: str,
        silero_model_dir: str,
        smart_turn_onnx_session=None,
        smart_turn_feature_extractor=None,
        smart_turn_provider: str | None = None,
        smart_turn_model_path: str | None = None,
        smart_turn_threshold: float = 0.5,
        # Per-session config
        threshold: float = 0.7,
        silence_ms: int = 500,
        prefix_padding_ms: int = 300,
        sample_rate: int = 16000,
        min_speech_duration_ms: int = 300,
        max_audio_duration_ms: int = 30000,
        smart_turn_enabled: bool = True,
    ) -> "SileroVADModule":
        """Create a lightweight SileroVADModule from pre-loaded model artifacts.

        This skips model download and loading, using shared artifacts from ModelManager.
        Only per-session state (LSTM state, counters, buffers) is initialized.

        Thread safety:
        - ONNX session: shared (read-only inference, thread-safe)
        - PyTorch model: deep-copied per session (reset_states() is NOT thread-safe)
        - SmartTurnDetector: created from shared ONNX session + feature extractor

        Args:
            onnx_session: Pre-loaded ONNX InferenceSession (shared)
            pytorch_model: Pre-loaded PyTorch model (will be deep-copied)
            vad_backend: "onnx" / "pytorch" / "none"
            silero_model_dir: Silero model directory path
            smart_turn_onnx_session: Pre-loaded Smart Turn ONNX session (shared)
            smart_turn_feature_extractor: Pre-loaded WhisperFeatureExtractor (shared)
            smart_turn_provider: Smart Turn ONNX provider name
            smart_turn_model_path: Smart Turn model file path
            smart_turn_threshold: Smart Turn prediction threshold
            threshold: VAD speech probability threshold
            silence_ms: Silence duration for endpoint detection
            prefix_padding_ms: Audio kept before speech starts
            sample_rate: Audio sample rate
            min_speech_duration_ms: Minimum speech duration
            max_audio_duration_ms: Maximum audio duration
            smart_turn_enabled: Whether Smart Turn is enabled

        Returns:
            SileroVADModule instance with shared model artifacts
        """
        instance = cls.__new__(cls)
        
        # ---- Per-session config ----
        instance.threshold = threshold
        instance.silence_duration_ms = silence_ms
        instance.prefix_padding_ms = prefix_padding_ms
        instance.sample_rate = sample_rate
        instance.min_speech_duration_ms = min_speech_duration_ms
        instance.max_audio_duration_ms = max_audio_duration_ms
        instance.smart_turn_enabled = smart_turn_enabled
        instance.smart_turn_threshold = smart_turn_threshold
        
        # ---- Model directory (for logging) ----
        instance._silero_model_dir = silero_model_dir
        instance._smart_turn_model_dir = None  # Not needed for preloaded
        
        # ---- Shared model artifacts ----
        instance._onnx_session = onnx_session
        
        # PyTorch model: deep copy for thread safety (reset_states is NOT thread-safe)
        if pytorch_model is not None:
            try:
                import copy
                instance._model = copy.deepcopy(pytorch_model)
                instance._model.eval()
                instance._model.reset_states()
            except Exception as e:
                logger.warning(f"[VAD] PyTorch model deep copy failed: {e}, using shared reference (NOT thread-safe)")
                instance._model = pytorch_model
        else:
            instance._model = None
        
        instance._vad_backend = vad_backend
        
        # ---- Per-session ONNX state ----
        instance._context_size = 64 if sample_rate == 16000 else 32
        instance._state = np.zeros((2, 1, 128), dtype=np.float32)
        instance._context = np.zeros((1, instance._context_size), dtype=np.float32)
        
        # ---- Per-session speech detection state ----
        instance.is_speaking = False
        
        instance._min_speech_frames = int(
            min_speech_duration_ms / (cls.CHUNK_SIZE / sample_rate * 1000)
        )
        instance._speech_candidate_frames = 0
        instance._speech_confirmed = False
        
        instance._max_audio_frames = int(
            max_audio_duration_ms / (cls.CHUNK_SIZE / sample_rate * 1000)
        )
        instance._total_speech_frames = 0
        
        instance._silence_timeout_frames = int(
            silence_ms / (cls.CHUNK_SIZE / sample_rate * 1000)
        )
        instance._silence_timeout_count = 0
        
        instance.prefix_chunks = deque(
            maxlen=int(prefix_padding_ms / (cls.CHUNK_SIZE / sample_rate * 1000))
        )
        
        instance._sample_buffer = np.array([], dtype=np.float32)
        
        # ---- Smart Turn (from preloaded artifacts) ----
        instance._turn_audio_buffer = np.array([], dtype=np.float32)
        
        if smart_turn_enabled and smart_turn_onnx_session is not None:
            instance._smart_turn = SmartTurnDetector.from_preloaded(
                onnx_session=smart_turn_onnx_session,
                feature_extractor=smart_turn_feature_extractor,
                provider=smart_turn_provider,
                model_path=smart_turn_model_path,
                threshold=smart_turn_threshold,
            )
            logger.info("[VAD] ✅ Smart Turn v3.2 initialized from preloaded artifacts (shared)")
        elif smart_turn_enabled and smart_turn_onnx_session is None:
            logger.warning("[VAD] Smart Turn enabled but ONNX session not preloaded, disabling")
            instance._smart_turn = None
            instance.smart_turn_enabled = False
        else:
            instance._smart_turn = None
        
        # ---- VAD 埋点日志状态 ----
        instance._frame_count = 0
        instance._prob_window = deque(maxlen=60)
        instance._speech_frame_count_in_window = 0
        instance._speech_start_time = None
        instance._speech_start_prob = 0.0
        instance._last_silence_start_time = None
        
        logger.info(
            f"VAD initialized (from preloaded): backend={instance._vad_backend}, "
            f"threshold={threshold}, silence_duration_ms={silence_ms}, "
            f"min_speech_duration_ms={min_speech_duration_ms}, "
            f"max_audio_duration_ms={max_audio_duration_ms}, "
            f"smart_turn_enabled={instance.smart_turn_enabled}, "
            f"silero_model_dir={silero_model_dir}"
        )
        
        return instance
    
    @staticmethod
    def _generate_speech_like_signal_static(sample_rate: int = 16000) -> np.ndarray:
        """Generate a speech-like signal for model semantic validation (static version).

        This is a static version of _generate_speech_like_signal that can be
        called without an instance, used by ModelManager for startup validation.

        Args:
            sample_rate: Audio sample rate (default 16000)

        Returns:
            float32 numpy array, shape (1, 512), 值范围 [-1, 1]
        """
        chunk_size = 512
        t = np.linspace(0, chunk_size / sample_rate, chunk_size, dtype=np.float32)
        signal = (
            np.sin(2 * np.pi * 200 * t) * 0.4 +   # 基频
            np.sin(2 * np.pi * 400 * t) * 0.25 +   # 二次谐波
            np.sin(2 * np.pi * 600 * t) * 0.1 +    # 三次谐波
            np.random.randn(chunk_size).astype(np.float32) * 0.05  # 微量噪声
        )
        signal = np.clip(signal, -1.0, 1.0)
        return signal.reshape(1, -1)
    
    def _init_smart_turn(self):
        """初始化 Smart Turn 检测器。失败则直接报错（不降级）。"""
        try:
            self._smart_turn = SmartTurnDetector(
                model_path=self._smart_turn_model_dir,
                threshold=self.smart_turn_threshold,
            )
            logger.info("[VAD] ✅ Smart Turn v3.2 已集成，静音超时后将进行话轮二次判断")
        except Exception as e:
            logger.error(f"[VAD] ❌ Smart Turn 初始化失败: {e}")
            raise RuntimeError(
                f"Smart Turn v3.2 初始化失败，无法降级运行: {e}"
            )
    
    def _load_model(self):
        """加载 VAD 模型，优先级：ONNX → PyTorch
        
        所有模型文件从 self._silero_model_dir 目录加载：
        - ONNX: silero_vad.onnx
        - PyTorch: silero_vad.jit (通过 torch.hub.load source="local")
        """
        # 1. 尝试 ONNX
        if self._try_load_onnx():
            self._vad_backend = "onnx"
            logger.info("✅ VAD backend: ONNX Runtime (优先)")
            return
        
        # 2. 降级到 PyTorch
        if self._try_load_pytorch():
            self._vad_backend = "pytorch"
            logger.info("✅ VAD backend: PyTorch (ONNX 不可用，降级)")
            return
        
        # 3. 都失败
        self._vad_backend = "none"
        logger.error(
            "❌ VAD 模型加载失败！ONNX 和 PyTorch 均不可用。"
            "请安装 onnxruntime (pip install onnxruntime) 或确认 PyTorch + silero-vad 模型可用"
        )
    
    def _try_load_onnx(self) -> bool:
        """尝试加载 ONNX Silero VAD 模型
        
        从 self._silero_model_dir 目录下加载 silero_vad.onnx
        
        Returns:
            True 如果加载并验证成功
        """
        onnx_path = os.path.join(self._silero_model_dir, "silero_vad.onnx")
        
        if not os.path.isfile(onnx_path):
            logger.info(f"[VAD] ONNX 模型文件不存在: {onnx_path}，跳过 ONNX 加载")
            return False
        
        try:
            import onnxruntime
        except ImportError:
            logger.info("[VAD] onnxruntime 未安装，跳过 ONNX 加载 (pip install onnxruntime)")
            return False
        
        try:
            logger.info(f"[VAD] 正在加载 ONNX 模型: {onnx_path}")
            self._onnx_session = onnxruntime.InferenceSession(onnx_path)
            
            # 语义验证：用合成语音信号测试
            test_audio = self._generate_speech_like_signal()
            test_state = np.zeros((2, 1, 128), dtype=np.float32)
            test_context = np.zeros((1, self._context_size), dtype=np.float32)
            test_sr = np.array(self.sample_rate, dtype=np.int64)
            
            test_input = np.concatenate([test_context, test_audio], axis=1)
            ort_outs = self._onnx_session.run(
                None,
                {"input": test_input, "state": test_state, "sr": test_sr}
            )
            
            if len(ort_outs) != 2:
                raise RuntimeError(f"ONNX 输出数量异常: {len(ort_outs)}")
            
            prob_val = ort_outs[0].item()
            state_shape = ort_outs[1].shape
            if state_shape != (2, 1, 128):
                raise RuntimeError(f"ONNX state 形状异常: {state_shape}, 期望 (2,1,128)")
            
            if prob_val < 0.01:
                raise RuntimeError(
                    f"ONNX 语义验证失败: 合成语音 prob={prob_val:.6f} < 0.01，"
                    f"可能是 onnxruntime LSTM 兼容性问题"
                )
            
            logger.info(
                f"[VAD] ✅ ONNX 模型验证通过: {onnx_path} "
                f"(测试prob={prob_val:.4f})"
            )
            return True
            
        except Exception as e:
            logger.warning(f"[VAD] ONNX 模型加载/验证失败: {e}")
            logger.warning("[VAD] 将降级到 PyTorch 推理")
            self._onnx_session = None
            return False
    
    def _try_load_pytorch(self) -> bool:
        """尝试加载 PyTorch Silero VAD 模型
        
        从 self._silero_model_dir 目录下加载 silero_vad.jit (通过 torch.jit.load)
        
        Returns:
            True 如果加载并验证成功
        """
        jit_path = os.path.join(self._silero_model_dir, "silero_vad.jit")
        
        if not os.path.isfile(jit_path):
            logger.info(f"[VAD] PyTorch 模型文件不存在: {jit_path}，跳过 PyTorch 加载")
            return False
        
        try:
            import torch
        except ImportError:
            logger.info("[VAD] PyTorch 未安装，跳过 PyTorch 加载")
            return False
        
        try:
            logger.info(f"[VAD] 正在通过 torch.jit.load 加载 Silero VAD: {jit_path}")
            self._model = torch.jit.load(jit_path)
            self._model.eval()
            
            # 验证 PyTorch 模型推理
            self._model.reset_states()
            silence = torch.zeros(512, dtype=torch.float32)
            with torch.no_grad():
                silence_prob = self._model(silence, self.sample_rate).item()
            
            # 生成类语音信号验证
            speech_signal = self._generate_speech_like_signal().flatten()
            self._model.reset_states()
            with torch.no_grad():
                speech_prob = self._model(torch.from_numpy(speech_signal), self.sample_rate).item()
            
            self._model.reset_states()
            
            if speech_prob < 0.01:
                logger.warning(
                    f"[VAD] PyTorch VAD 验证异常: 合成语音 prob={speech_prob:.6f}"
                )
                self._model = None
                return False
            
            logger.info(
                f"[VAD] ✅ PyTorch 模型验证通过 "
                f"(静音prob={silence_prob:.4f}, 语音prob={speech_prob:.4f})"
            )
            return True
            
        except Exception as e:
            logger.warning(f"[VAD] PyTorch 模型加载失败: {e}")
            self._model = None
            return False
    
    def _generate_speech_like_signal(self) -> np.ndarray:
        """生成类语音信号用于模型语义验证
        
        Returns:
            float32 numpy array, shape (1, 512), 值范围 [-1, 1]
        """
        return self._generate_speech_like_signal_static(self.sample_rate)
    
    @property
    def silence_timeout_ms(self) -> int:
        """Alias for silence_duration_ms (unified parameter, see M3)."""
        return self.silence_duration_ms

    @silence_timeout_ms.setter
    def silence_timeout_ms(self, value: int):
        """Alias for silence_duration_ms (unified parameter, see M3)."""
        self.silence_duration_ms = value

    def update_silence_config(self, silence_ms: int):
        """Update silence detection configuration (called by session.update).

        Updates all silence-related parameters atomically:
        - silence_duration_ms: the configured silence duration
        - silence_timeout_ms: same as silence_duration_ms (unified parameter, see M3)
        - _silence_timeout_frames: calculated from silence_duration_ms
        """
        self.silence_duration_ms = silence_ms
        # silence_timeout_ms is now an alias for silence_duration_ms (see M3 property)
        self._silence_timeout_frames = int(
            silence_ms / (self.CHUNK_SIZE / self.sample_rate * 1000)
        )

    def reset(self):
        """Reset VAD state."""
        self.is_speaking = False
        self.prefix_chunks.clear()
        self._sample_buffer = np.array([], dtype=np.float32)
        
        # min_speech_duration 状态重置
        self._speech_candidate_frames = 0
        self._speech_confirmed = False
        
        # max_audio_duration 状态重置
        self._total_speech_frames = 0
        
        # silence_timeout 状态重置
        self._silence_timeout_count = 0
        
        # Smart Turn 音频缓冲重置
        self._turn_audio_buffer = np.array([], dtype=np.float32)
        
        # ONNX 状态重置
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)
        
        # PyTorch 状态重置
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass
        
        # VAD 埋点状态重置（保留 _frame_count 以维持周期性输出的连续性）
        self._prob_window.clear()
        self._speech_frame_count_in_window = 0
        self._speech_start_time = None
        self._speech_start_prob = 0.0
        self._last_silence_start_time = None
    
    def _detect_frame(self, chunk: np.ndarray) -> float:
        """对单个 CHUNK_SIZE 帧进行 VAD 检测
        
        优先级：ONNX → PyTorch
        
        Args:
            chunk: float32 numpy array, shape (512,), 值范围 [-1, 1]
            
        Returns:
            语音概率 0.0-1.0
        """
        if self._onnx_session is not None:
            return self._detect_onnx(chunk)
        elif self._model is not None:
            return self._detect_pytorch(chunk)
        else:
            logger.error("[VAD] 无可用 VAD 后端！返回 0.0")
            return 0.0
    
    def _detect_onnx(self, chunk: np.ndarray) -> float:
        """ONNX Silero VAD 检测（首选方案）"""
        try:
            frame = chunk.reshape(1, -1)  # shape: (1, 512)
            
            # 拼接 context + frame → shape (1, 576)
            x = np.concatenate([self._context, frame], axis=1)
            
            ort_inputs = {
                "input": x,
                "state": self._state,
                "sr": np.array(self.sample_rate, dtype=np.int64),
            }
            ort_outs = self._onnx_session.run(None, ort_inputs)
            
            if len(ort_outs) < 2:
                raise RuntimeError(f"ONNX 输出数量异常: {len(ort_outs)}")
            
            new_state = ort_outs[1]
            if new_state.shape != (2, 1, 128):
                raise RuntimeError(f"ONNX state 形状异常: {new_state.shape}")
            
            self._state = new_state
            prob = ort_outs[0].item()
            self._context = x[:, -self._context_size:]
            
            return prob
            
        except Exception as e:
            logger.warning(f"[VAD] ONNX 推理异常，降级到 PyTorch: {e}")
            self._onnx_session = None
            self._vad_backend = "pytorch" if self._model is not None else "none"
            # 重置 ONNX 状态
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            self._context = np.zeros((1, self._context_size), dtype=np.float32)
            # 降级到 PyTorch
            if self._model is not None:
                return self._detect_pytorch(chunk)
            return 0.0
    
    def _detect_pytorch(self, chunk: np.ndarray) -> float:
        """PyTorch Silero VAD 检测（降级方案）"""
        try:
            import torch
            audio_tensor = torch.from_numpy(chunk)
            with torch.no_grad():
                prob = self._model(audio_tensor, self.sample_rate).item()
            return prob
        except Exception as e:
            logger.error(f"[VAD] PyTorch 推理异常: {e}")
            return 0.0
    
    def _update_prob_stats(self, speech_prob: float):
        """更新 VAD prob 统计窗口，并在达到间隔时输出周期性日志
        
        Args:
            speech_prob: 当前帧的语音概率
        """
        self._frame_count += 1
        self._prob_window.append(speech_prob)
        if speech_prob >= self.threshold:
            self._speech_frame_count_in_window += 1
        
        # deque(maxlen=60) 自动维护窗口大小，无需手动 pop
        # 但需要检查被自动丢弃的元素来更新 _speech_frame_count_in_window
        if len(self._prob_window) == self._prob_window.maxlen:
            # 下一个 append 会自动丢弃最老的元素
            oldest = self._prob_window[0]
            if oldest >= self.threshold:
                self._speech_frame_count_in_window -= 1
        
        # 周期性输出（每 _PROB_LOG_INTERVAL_FRAMES 帧输出一次）
        if self._frame_count % self._PROB_LOG_INTERVAL_FRAMES == 0:
            if self._prob_window:
                avg_prob = sum(self._prob_window) / len(self._prob_window)
                speech_ratio = self._speech_frame_count_in_window / len(self._prob_window)
                frame_ms = self.CHUNK_SIZE / self.sample_rate * 1000
                elapsed_s = self._frame_count * frame_ms / 1000
                logger.debug(
                    f"[VAD] prob stats @ {elapsed_s:.1f}s: "
                    f"avg_prob={avg_prob:.4f}, speech_ratio={speech_ratio:.2f}, "
                    f"threshold={self.threshold}, speaking={self.is_speaking}, "
                    f"backend={self._vad_backend}"
                )
    
    def _check_smart_turn(self) -> bool:
        """调用 Smart Turn 判断话轮是否结束。
        
        Returns:
            True = Complete (用户说完了，应触发 speech_stopped)
            False = Incomplete (用户还在思考，应继续等待)
        """
        if self._smart_turn is None:
            # Smart Turn 未启用，默认 Complete
            return True
        
        if len(self._turn_audio_buffer) == 0:
            # 没有累积音频，默认 Complete
            logger.warning("[SmartTurn] _turn_audio_buffer 为空，默认判定 Complete")
            return True
        
        result = self._smart_turn.predict_endpoint(self._turn_audio_buffer)
        
        if result["prediction"] == 1:
            # Complete: 用户说完了
            logger.info(
                f"[SmartTurn] ✅ Complete (prob={result['probability']:.4f}), "
                f"触发 speech_stopped"
            )
            return True
        else:
            # Incomplete: 用户还在思考/犹豫
            logger.info(
                f"[SmartTurn] ⏳ Incomplete (prob={result['probability']:.4f}), "
                f"重置静音计数，继续等待"
            )
            return False
    
    def process(self, pcm16_bytes: bytes) -> list[str]:
        """Process a frame of PCM16 audio, return event list.
        
        Events: "speech_started" | "speech_stopped" | "max_duration_reached"
        
        Buffers incoming frames and processes them in CHUNK_SIZE (512 samples) blocks.
        Handles frames smaller than CHUNK_SIZE by accumulating until enough samples.
        
        VAD 策略：
        1. min_speech_duration: 语音开始后必须持续超过最短时长才确认
        2. silence_timeout: 确认说话后，持续静音超时则判定说完
           - 如果 smart_turn_enabled: 静音超时后调用 Smart Turn 二次判断
             - Complete → 触发 speech_stopped
             - Incomplete → 重置静音计数，继续等待
        3. max_audio_duration: 累积音频超过最大时长强制截断
        """
        if len(pcm16_bytes) < 2:
            return []
        
        # Ensure even byte length
        if len(pcm16_bytes) % 2 != 0:
            pcm16_bytes = pcm16_bytes[:-1]
        
        audio_np = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Append to buffer
        self._sample_buffer = np.concatenate([self._sample_buffer, audio_np])
        
        events = []
        frame_ms = self.CHUNK_SIZE / self.sample_rate * 1000  # 32ms per frame
        
        # Process complete chunks from buffer
        while len(self._sample_buffer) >= self.CHUNK_SIZE:
            chunk = self._sample_buffer[:self.CHUNK_SIZE]
            self._sample_buffer = self._sample_buffer[self.CHUNK_SIZE:]
            
            # VAD 检测
            speech_prob = self._detect_frame(chunk)
            
            # 更新 prob 统计（用于周期性输出）
            self._update_prob_stats(speech_prob)
            
            # ---- 状态机逻辑 ----
            
            if not self.is_speaking:
                # === 未说话状态 ===
                self.prefix_chunks.append(chunk.copy())
                
                if speech_prob >= self.threshold:
                    # 检测到可能的语音开始
                    self._speech_candidate_frames += 1
                    
                    if not self._speech_confirmed and self._speech_candidate_frames >= self._min_speech_frames:
                        # 语音持续超过 min_speech_duration，确认说话开始
                        self._speech_confirmed = True
                        self.is_speaking = True
                        self._total_speech_frames = self._speech_candidate_frames
                        self._silence_timeout_count = 0
                        
                        # Smart Turn: 开始累积语音期音频（包含 prefix 中的候选帧）
                        if self.smart_turn_enabled and self._smart_turn is not None:
                            prefix_np = np.concatenate(list(self.prefix_chunks))
                            self._turn_audio_buffer = prefix_np.copy()
                        
                        # 埋点: speech_started
                        self._speech_start_time = time.monotonic()
                        self._speech_start_prob = speech_prob
                        accumulated_ms = self._speech_candidate_frames * frame_ms
                        logger.info(
                            f"[VAD] speech_started: prob={speech_prob:.4f}, "
                            f"accumulated_audio={accumulated_ms:.0f}ms "
                            f"({self._speech_candidate_frames} frames), "
                            f"threshold={self.threshold}"
                        )
                        
                        events.append("speech_started")
                else:
                    # 语音概率低于阈值，重置候选帧
                    self._speech_candidate_frames = 0
                    self._speech_confirmed = False
            
            else:
                # === 说话中状态 ===
                self._total_speech_frames += 1
                
                # Smart Turn: 累积语音期音频（所有帧，包括静音帧，保持连续性）
                if self.smart_turn_enabled and self._smart_turn is not None:
                    self._turn_audio_buffer = np.concatenate([self._turn_audio_buffer, chunk])
                
                if speech_prob >= self.threshold:
                    # 仍在说话
                    self._silence_timeout_count = 0
                    self._last_silence_start_time = None
                else:
                    # 检测到静音
                    if self._last_silence_start_time is None:
                        self._last_silence_start_time = time.monotonic()
                    self._silence_timeout_count += 1
                
                # 检查 silence_timeout: 说话后持续静音超时 → Smart Turn 二次判断
                if self._silence_timeout_count >= self._silence_timeout_frames:
                    
                    if self.smart_turn_enabled and self._smart_turn is not None:
                        # Smart Turn 二次判断
                        is_complete = self._check_smart_turn()
                        
                        if is_complete:
                            # Complete: 用户说完了 → 触发 speech_stopped
                            silence_duration_ms = self._silence_timeout_count * frame_ms
                            total_speech_ms = self._total_speech_frames * frame_ms
                            logger.info(
                                f"[VAD] speech_stopped: silence_duration={silence_duration_ms:.0f}ms "
                                f"({self._silence_timeout_count} frames), "
                                f"total_speech={total_speech_ms:.0f}ms "
                                f"({self._total_speech_frames} frames), "
                                f"reason=silence_timeout+smart_turn_complete"
                            )
                            
                            self.is_speaking = False
                            self._speech_candidate_frames = 0
                            self._speech_confirmed = False
                            self._total_speech_frames = 0
                            self._silence_timeout_count = 0
                            self._speech_start_time = None
                            self._last_silence_start_time = None
                            self._turn_audio_buffer = np.array([], dtype=np.float32)
                            events.append("speech_stopped")
                        else:
                            # Incomplete: 用户还在思考 → 重置静音计数，继续等待
                            self._silence_timeout_count = 0
                            self._last_silence_start_time = None
                            # 注意: 不重置 _turn_audio_buffer，继续累积
                            # 注意: 不触发任何事件
                    else:
                        # Smart Turn 未启用，直接触发 speech_stopped（原始逻辑）
                        silence_duration_ms = self._silence_timeout_count * frame_ms
                        total_speech_ms = self._total_speech_frames * frame_ms
                        logger.info(
                            f"[VAD] speech_stopped: silence_duration={silence_duration_ms:.0f}ms "
                            f"({self._silence_timeout_count} frames), "
                            f"total_speech={total_speech_ms:.0f}ms "
                            f"({self._total_speech_frames} frames), "
                            f"reason=silence_timeout"
                        )
                        
                        self.is_speaking = False
                        self._speech_candidate_frames = 0
                        self._speech_confirmed = False
                        self._total_speech_frames = 0
                        self._silence_timeout_count = 0
                        self._speech_start_time = None
                        self._last_silence_start_time = None
                        events.append("speech_stopped")
                
                # 检查 max_audio_duration: 累积音频超过最大时长 → 强制截断
                elif self._total_speech_frames >= self._max_audio_frames:
                    # 埋点: max_duration_reached
                    total_speech_ms = self._total_speech_frames * frame_ms
                    logger.info(
                        f"[VAD] max_duration_reached: "
                        f"accumulated_audio={total_speech_ms:.0f}ms "
                        f"({self._total_speech_frames} frames), "
                        f"max_limit={self.max_audio_duration_ms}ms"
                    )
                    
                    self.is_speaking = False
                    self._speech_candidate_frames = 0
                    self._speech_confirmed = False
                    self._total_speech_frames = 0
                    self._silence_timeout_count = 0
                    self._speech_start_time = None
                    self._last_silence_start_time = None
                    self._turn_audio_buffer = np.array([], dtype=np.float32)
                    events.append("speech_stopped")
                    events.append("max_duration_reached")
        
        return events
    
    def get_prefix_audio(self) -> bytes:
        """Get prefix audio as PCM16 bytes (audio before speech started)."""
        if not self.prefix_chunks:
            return b""
        prefix_np = np.concatenate(list(self.prefix_chunks))
        return (prefix_np * 32768.0).astype(np.int16).tobytes()
    
    def get_turn_audio(self) -> bytes:
        """Get the current turn accumulated audio as PCM16 bytes.
        
        Returns the audio data accumulated during the current speech turn.
        Used after interruption to immediately obtain the speech data
        that has been accumulated so far.
        
        Returns:
            PCM16 bytes of the current speech turn, or empty bytes if no speech.
        """
        if len(self._turn_audio_buffer) == 0:
            return b""
        return (self._turn_audio_buffer * 32768.0).astype(np.int16).tobytes()

    @property
    def vad_backend(self) -> str:
        """返回当前 VAD 后端描述"""
        return self._vad_backend
