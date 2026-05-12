"""Local ASR Engine - SenseVoiceSmall inference on local GPU.

Provides a singleton LocalASREngine that loads the SenseVoiceSmall model once
at startup and shares it across all sessions. Inference is serialized via
asyncio.Lock + run_in_executor to ensure thread safety.
"""
import asyncio
import logging
import os
import re
from typing import Optional

import numpy as np

logger = logging.getLogger("realtime-server")

# Target sample rate for SenseVoiceSmall
_TARGET_SAMPLE_RATE = 16000


def _clean_text(text: str) -> str:
    """Clean SenseVoice output by removing special markers like <|...|>."""
    text = re.sub(r'<\|.*?\|>', '', text)
    return text.strip()


class LocalASREngine:
    """Local ASR engine using SenseVoiceSmall via FunASR.

    This is designed as a global singleton: load once at startup, share across
    all sessions. GPU inference is serialized with asyncio.Lock to guarantee
    thread safety.

    Usage:
        engine = LocalASREngine(model_path=None)  # uses modelscope default
        text = await engine.transcribe(pcm_bytes, sample_rate=16000)
    """

    _instance: Optional["LocalASREngine"] = None

    def __init__(self, model_path: str | None = None):
        """Initialize and load the SenseVoiceSmall model.

        Args:
            model_path: Local directory path for the model. If None, uses
                modelscope default cache ("iic/SenseVoiceSmall"). If a path
                is given but doesn't contain model files, downloads from
                modelscope to that path first.
        """
        import torch
        from funasr import AutoModel

        # Determine the model identifier for AutoModel
        if model_path is None:
            model_id = "iic/SenseVoiceSmall"
            logger.info("[LocalASR] No model_path specified, using modelscope default: iic/SenseVoiceSmall")
        else:
            # Check if the path exists and contains model files
            expanded_path = os.path.expanduser(model_path)
            if os.path.isdir(expanded_path) and os.listdir(expanded_path):
                model_id = expanded_path
                logger.info(f"[LocalASR] Using local model path: {expanded_path}")
            else:
                # Download from modelscope
                logger.info(f"[LocalASR] Model path {expanded_path} is empty or does not exist, downloading from modelscope...")
                try:
                    from modelscope import snapshot_download
                    snapshot_download("iic/SenseVoiceSmall", cache_dir=expanded_path)
                    model_id = expanded_path
                    logger.info(f"[LocalASR] Model downloaded to: {expanded_path}")
                except Exception as e:
                    logger.error(f"[LocalASR] Failed to download model: {e}")
                    raise

        # Load model onto GPU
        logger.info(f"[LocalASR] Loading model from: {model_id}")
        self._model = AutoModel(
            model=model_id,
            device="cuda",
            disable_update=True,
            torch_dtype="float16",
        )
        logger.info("[LocalASR] Model loaded successfully")

        # Warm up with 1-second silence audio
        logger.info("[LocalASR] Warming up model with 1s silence...")
        dummy_audio = np.random.randn(_TARGET_SAMPLE_RATE).astype(np.float32) * 0.01
        self._model.generate(input=dummy_audio, language="zh", use_itn=True)
        logger.info("[LocalASR] Model warm-up complete")

        # Inference lock for thread safety
        self._lock = asyncio.Lock()

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
        language: str = "zh",
    ) -> str:
        """Transcribe PCM audio bytes to text.

        Args:
            pcm_bytes: Raw PCM audio data (int16, mono).
            sample_rate: Sample rate of the input audio.
            language: Language code for transcription (default: "zh").

        Returns:
            Transcribed text string, or empty string on failure.
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

        # Run inference in executor with lock (thread-safe, non-blocking)
        loop = asyncio.get_running_loop()
        async with self._lock:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: self._model.generate(
                        input=audio_float, language=language, use_itn=True
                    ),
                )
            except Exception as e:
                logger.error(f"[LocalASR] Inference error: {e}", exc_info=True)
                return ""

        # Parse result
        if result and len(result) > 0:
            raw_text = result[0].get("text", "")
            cleaned = _clean_text(raw_text)
            return cleaned

        return ""
