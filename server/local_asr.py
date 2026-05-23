"""Local ASR Engine - SenseVoiceSmall inference via FunASR.

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

    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        """Initialize and load the SenseVoiceSmall model.

        Args:
            model_path: Local directory path for the model. If None, uses
                modelscope default cache ("iic/SenseVoiceSmall"). If a path
                is given but doesn't contain model files, downloads from
                modelscope to that path first.
            device: Device for model inference, "cuda" or "cpu". Default "cuda".
        """
        from funasr import AutoModel

        _MODELSCOPE_ID = "iic/SenseVoiceSmall"

        if model_path is None:
            model_id = _MODELSCOPE_ID
            logger.info("[LocalASR] No model_path specified, using modelscope default")
        else:
            # FunASR's download_model() uses os.path.exists/os.path.join internally
            # to parse configuration.json and resolve model file paths.
            # Relative paths break this on Windows, so always resolve to absolute.
            expanded_path = os.path.abspath(os.path.expanduser(model_path))

            # Check if the path already contains model files (config.yaml or configuration.json)
            has_model_files = (
                os.path.isfile(os.path.join(expanded_path, "configuration.json"))
                or os.path.isfile(os.path.join(expanded_path, "config.yaml"))
            )

            if has_model_files:
                model_id = expanded_path
                logger.info(f"[LocalASR] Using local model path: {expanded_path}")
            else:
                # Directory doesn't exist, is empty, or lacks model files —
                # check for snapshot subdirectory first, then download if needed.
                snapshot_subdir = os.path.join(expanded_path, _MODELSCOPE_ID.replace("/", os.sep))
                if os.path.isfile(os.path.join(snapshot_subdir, "configuration.json")) or \
                   os.path.isfile(os.path.join(snapshot_subdir, "config.yaml")):
                    model_id = snapshot_subdir
                    logger.info(f"[LocalASR] Found model in snapshot subdirectory: {snapshot_subdir}")
                else:
                    # No model files found — download via modelscope
                    logger.info(f"[LocalASR] Model files not found in {expanded_path}, downloading from modelscope...")
                    try:
                        from modelscope import snapshot_download
                        snapshot_download(_MODELSCOPE_ID, cache_dir=expanded_path)
                        # snapshot_download stores under cache_dir/<model_id>/
                        snapshot_subdir = os.path.join(expanded_path, _MODELSCOPE_ID.replace("/", os.sep))
                        model_id = snapshot_subdir if os.path.isdir(snapshot_subdir) else expanded_path
                        logger.info(f"[LocalASR] Model downloaded to: {model_id}")
                    except Exception as e:
                        logger.error(f"[LocalASR] Failed to download model: {e}")
                        raise

        logger.info(f"[LocalASR] Loading model from: {model_id}, device: {device}")
        self._model = AutoModel(
            model=model_id,
            device=device,
            disable_update=True,
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
