"""Local ASR Engine - SenseVoiceSmall inference via sherpa-onnx.

Provides a singleton LocalASREngine that loads the SenseVoiceSmall ONNX model
once at startup and shares it across all sessions. Inference is serialized via
asyncio.Lock + run_in_executor to ensure thread safety.

Model download:
    If model_path is provided but lacks required files (model_q8.onnx / model.onnx
    + tokens.txt), automatically downloads from ModelScope:
        xiaowangge/sherpa-onnx-sense-voice-small
"""
import asyncio
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("realtime-server")

_TARGET_SAMPLE_RATE = 16000
_MODELSCOPE_MODEL_ID = "xiaowangge/sherpa-onnx-sense-voice-small"
# Required files: Q8 model (preferred) or full model, plus tokens
_REQUIRED_FILES_Q8 = ["model_q8.onnx", "tokens.txt"]
_REQUIRED_FILES_FULL = ["model.onnx", "tokens.txt"]


def _ensure_model_downloaded(model_id: str, target_dir: str, required_files: list[str]) -> str:
    """Ensure model files are downloaded to the target directory.

    If all required_files exist under target_dir, skip download.
    Otherwise, download from ModelScope via snapshot_download.

    Args:
        model_id: ModelScope model ID, e.g. "xiaowangge/sherpa-onnx-sense-voice-small"
        target_dir: Target directory path for the model
        required_files: List of required file names

    Returns:
        Absolute path of the target directory
    """
    target_dir = os.path.abspath(target_dir)

    # Check if all required files exist
    all_exist = all(
        os.path.isfile(os.path.join(target_dir, f)) for f in required_files
    )

    if all_exist:
        logger.info(f"[LocalASR] All required files exist, skip download: {target_dir}")
        return target_dir

    # Need to download
    missing = [f for f in required_files if not os.path.isfile(os.path.join(target_dir, f))]
    logger.info(
        f"[LocalASR] Missing files {missing}, downloading from ModelScope "
        f"model_id={model_id} -> {target_dir}"
    )

    try:
        from modelscope.hub.snapshot_download import snapshot_download
        downloaded_dir = snapshot_download(
            model_id=model_id,
            local_dir=target_dir,
        )
        logger.info(f"[LocalASR] Download complete: {downloaded_dir}")
        return downloaded_dir
    except Exception as e:
        logger.error(f"[LocalASR] ModelScope download failed: {e}")
        if os.path.isdir(target_dir):
            return target_dir
        raise


def _build_prefixed_text(text: str, emotion: str, event: str) -> str:
    """Build text with emotion and event tags prefixed.

    sherpa-onnx SenseVoice result.text is already clean (no <|...|> markers),
    so no cleaning is needed. This function prepends emotion and event tags
    to make downstream consumers (LLM, TTS, etc.) aware of emotion/event context.

    Filtering: <|EMO_UNKNOWN|> is skipped (no information value).
    All other tags (including <|Speech|>) are preserved per user preference.

    Examples:
        <|HAPPY|><|Speech|>今天天气真好。
        <|HAPPY|><|Laughter|>太有意思了。
        <|Speech|>今天的会议定在下午3点。
        <|Laughter|>哈哈哈。

    Args:
        text: Clean transcription text from result.text.
        emotion: Raw emotion tag, e.g. '<|HAPPY|>'. May be empty string.
        event: Raw event tag, e.g. '<|Speech|>'. May be empty string.

    Returns:
        Text with tags prefixed, e.g. '<|HAPPY|><|Speech|>文本内容'.
    """
    prefix = ""
    if emotion and emotion != "<|EMO_UNKNOWN|>":
        prefix += emotion
    if event:
        prefix += event
    return prefix + text


class LocalASREngine:
    """Local ASR engine using SenseVoiceSmall via sherpa-onnx.

    This is designed as a global singleton: load once at startup, share across
    all sessions. Inference is serialized with asyncio.Lock to guarantee
    thread safety.

    Usage:
        engine = LocalASREngine(model_path="/path/to/sherpa-onnx-sense-voice-small")
        text = await engine.transcribe(pcm_bytes, sample_rate=16000)
    """

    _instance: Optional["LocalASREngine"] = None

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        """Initialize and load the SenseVoiceSmall ONNX model via sherpa-onnx.

        Args:
            model_path: Local directory path for the model. Must contain
                model_q8.onnx (or model.onnx) and tokens.txt. If the directory
                exists but lacks required files, they will be automatically
                downloaded from ModelScope (xiaowangge/sherpa-onnx-sense-voice-small).
                Defaults to ./models/sherpa-onnx-sense-voice-small (config-driven).
            device: Device for model inference. "cpu" or "cuda".
                Note: "cuda" requires sherpa-onnx compiled with GPU support.
                Default "cpu".
        """
        import sherpa_onnx

        _DEFAULT_MODEL_PATH = "./models/sherpa-onnx-sense-voice-small"  # User must set or rely on auto-download

        if model_path is None:
            model_dir = _DEFAULT_MODEL_PATH
            logger.info("[LocalASR] No model_path specified, using default")
        else:
            model_dir = os.path.abspath(os.path.expanduser(model_path))

        # Ensure required model files are present, download if missing.
        # Try Q8 requirements first (preferred, smaller download), then full model.
        q8_exists = all(
            os.path.isfile(os.path.join(model_dir, f)) for f in _REQUIRED_FILES_Q8
        )
        full_exists = all(
            os.path.isfile(os.path.join(model_dir, f)) for f in _REQUIRED_FILES_FULL
        )

        if q8_exists or full_exists:
            logger.info(f"[LocalASR] Model files found in: {model_dir}")
        else:
            logger.info(
                f"[LocalASR] Required model files not found in {model_dir}, "
                f"downloading from ModelScope: {_MODELSCOPE_MODEL_ID}"
            )
            # Download Q8 model (preferred - smaller, ~228MB vs ~894MB)
            model_dir = _ensure_model_downloaded(
                model_id=_MODELSCOPE_MODEL_ID,
                target_dir=model_dir,
                required_files=_REQUIRED_FILES_Q8,
            )

        # Determine ONNX model file: prefer Q8 (smaller, faster), fallback to full
        q8_model = os.path.join(model_dir, "model_q8.onnx")
        full_model = os.path.join(model_dir, "model.onnx")
        tokens_file = os.path.join(model_dir, "tokens.txt")

        if os.path.isfile(q8_model):
            model_file = q8_model
            logger.info(f"[LocalASR] Using Q8 quantized model: {q8_model}")
        elif os.path.isfile(full_model):
            model_file = full_model
            logger.info(f"[LocalASR] Using full model: {full_model}")
        else:
            raise FileNotFoundError(
                f"No ONNX model found in {model_dir}. "
                f"Expected model_q8.onnx or model.onnx"
            )

        if not os.path.isfile(tokens_file):
            raise FileNotFoundError(f"tokens.txt not found in {model_dir}")

        # Map device to sherpa-onnx provider
        provider = "cuda" if device == "cuda" else "cpu"
        num_threads = max(2, os.cpu_count() // 2) if provider == "cpu" else 1

        logger.info(
            f"[LocalASR] Loading model from: {model_file}, "
            f"provider: {provider}, num_threads: {num_threads}"
        )

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_file,
            tokens=tokens_file,
            provider=provider,
            num_threads=num_threads,
            use_itn=True,
            sample_rate=_TARGET_SAMPLE_RATE,
            language="auto",
            debug=False,
        )
        logger.info("[LocalASR] Model loaded successfully")

        # Warm up with 1-second silence audio
        logger.info("[LocalASR] Warming up model with 1s silence...")
        dummy_audio = np.random.randn(_TARGET_SAMPLE_RATE).astype(np.float32) * 0.01
        stream = self._recognizer.create_stream()
        stream.accept_waveform(_TARGET_SAMPLE_RATE, dummy_audio)
        self._recognizer.decode_stream(stream)
        logger.info("[LocalASR] Model warm-up complete")

        # Lock will be lazily initialized in transcribe() on the main event loop
        self._lock: asyncio.Lock | None = None

        # Store as singleton
        LocalASREngine._instance = self

    @classmethod
    def get_instance(cls) -> Optional["LocalASREngine"]:
        """Get the singleton instance, or None if not initialized."""
        return cls._instance

    async def transcribe(
        self,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
        language: str = "auto",
    ) -> str:
        """Transcribe PCM audio bytes to text with emotion/event tags prefixed.

        DEPRECATED: This method is kept for backward compatibility.
        Use transcribe_with_details() instead to get (text, emotion, event) separately.

        Args:
            pcm_bytes: Raw PCM audio data (int16, mono).
            sample_rate: Sample rate of the input audio.
            language: Language code for transcription (default: "auto").

        Returns:
            Text with emotion and event tags prefixed, e.g.:
                '<|HAPPY|><|Speech|>今天天气真好。'
            <|EMO_UNKNOWN|> is filtered out; all other tags are preserved.
            Returns empty string on failure.
        """
        details = await self.transcribe_with_details(pcm_bytes, sample_rate, language)
        if details:
            text, emotion, event = details
            return _build_prefixed_text(text, emotion, event)
        return ""

    async def transcribe_with_details(
        self,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
        language: str = "auto",
    ) -> tuple[str, str, str] | None:
        """Transcribe PCM audio bytes and return detailed results.

        Returns a tuple of (text, emotion, event) separately, allowing callers to:
        - Send clean text to LLM (without tags in user message)
        - Send clean text to client transcript (without tags)
        - Inject emotion/event context into system prompt

        This is the preferred method for integrations that need to separate
        the transcription text from its emotion/event metadata.

        Args:
            pcm_bytes: Raw PCM audio data (int16, mono).
            sample_rate: Sample rate of the input audio.
            language: Language code for transcription (default: "auto").

        Returns:
            Tuple of (text, emotion, event) where:
            - text: Clean transcription text without any tags
            - emotion: Emotion tag like '<|HAPPY|>' or empty string
            - event: Event tag like '<|Speech|>' or empty string
            Returns None on failure.
        """
        # Convert int16 PCM bytes to float32 numpy array
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Resample if needed (using librosa + soxr_mq for efficiency)
        if sample_rate != _TARGET_SAMPLE_RATE:
            import librosa
            audio_float = librosa.resample(
                audio_float,
                orig_sr=sample_rate,
                target_sr=_TARGET_SAMPLE_RATE,
                res_type="soxr_mq",
            )

        if self._lock is None:
            self._lock = asyncio.Lock()
        # Run inference in executor with lock (thread-safe, non-blocking)
        loop = asyncio.get_running_loop()
        async with self._lock:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: self._decode(audio_float),
                )
            except Exception as e:
                logger.error(f"[LocalASR] Inference error: {e}", exc_info=True)
                return None

        # Return the tuple directly: (text, emotion, event)
        return result if result else None

    def _decode(self, audio_float: np.ndarray) -> tuple[str, str, str]:
        """Synchronous decode method for run_in_executor.

        sherpa-onnx SenseVoice result fields:
            - result.text:    Clean text (no <|...|> markers)
            - result.emotion: Emotion tag like '<|HAPPY|>' (may be empty)
            - result.event:   Event tag like '<|Speech|>' or '<|Laughter|>'

        Args:
            audio_float: float32 numpy array, values in [-1, 1].

        Returns:
            Tuple of (text, emotion, event). emotion/event may be empty strings.
        """
        stream = self._recognizer.create_stream()
        stream.accept_waveform(_TARGET_SAMPLE_RATE, audio_float)
        self._recognizer.decode_stream(stream)
        result = stream.result
        # emotion/event 字段需要较新的 sherpa-onnx；旧版本没有这两个属性时
        # 降级为空字符串，避免 AttributeError 让整个转写失败
        emotion = getattr(result, "emotion", "") or ""
        event = getattr(result, "event", "") or ""
        return result.text, emotion, event
