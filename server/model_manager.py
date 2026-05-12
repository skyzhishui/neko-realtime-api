"""Global Model Manager - Preload VAD and Smart Turn models at startup.

Manages the lifecycle of heavy model artifacts (ONNX sessions, PyTorch models,
feature extractors) so they are loaded once at startup and shared across sessions.

Thread safety:
- ONNX InferenceSession: thread-safe for read-only inference, can be shared
- PyTorch model: reset_states() is NOT thread-safe, each session needs its own copy
- WhisperFeatureExtractor: stateless, can be shared
- SmartTurnDetector ONNX session: thread-safe for read-only inference, can be shared
"""
import logging
import time

import numpy as np

from .vad import SileroVADModule, SmartTurnDetector, _ensure_model_downloaded, _get_modelscope_cache_dir

logger = logging.getLogger("realtime-server")


class ModelManager:
    """Global singleton that preloads and manages VAD + Smart Turn models.

    Usage:
        # At startup (in lifespan):
        manager = ModelManager()
        manager.preload_models(config)

        # When creating a session:
        vad = manager.create_vad_module(config)
    """

    _instance: "ModelManager | None" = None

    def __init__(self):
        # Silero VAD model artifacts
        self._onnx_session = None       # onnxruntime.InferenceSession (shared)
        self._pytorch_model = None      # torch.jit.RecursiveScriptModule (per-session copy needed)
        self._vad_backend = "none"      # "onnx" / "pytorch" / "none"
        self._silero_model_dir = None

        # Smart Turn model artifacts
        self._smart_turn_onnx_session = None    # onnxruntime.InferenceSession (shared)
        self._smart_turn_feature_extractor = None  # WhisperFeatureExtractor (shared, stateless)
        self._smart_turn_provider = None
        self._smart_turn_model_path = None

        # Smart Turn config
        self._smart_turn_threshold = 0.5

        # Preload status
        self._preloaded = False

    @classmethod
    def get_instance(cls) -> "ModelManager | None":
        """Get the global ModelManager instance, or None if not yet created."""
        return cls._instance

    def preload_models(self, config):
        """Preload all VAD and Smart Turn models.

        This should be called once at startup, in a thread executor to avoid
        blocking the event loop.

        Args:
            config: ServerConfig instance
        """
        t_start = time.perf_counter()
        logger.info("[ModelManager] Starting model preloading...")

        # ---- 1. Resolve Silero model directory + download check ----
        silero_model_path = config.get("vad", "silero_model_path", default=None)
        if silero_model_path:
            silero_dir = silero_model_path
        else:
            silero_dir = _get_modelscope_cache_dir(SileroVADModule._SILERO_MODELSCOPE_MODEL_ID)

        self._silero_model_dir = _ensure_model_downloaded(
            model_id=SileroVADModule._SILERO_MODELSCOPE_MODEL_ID,
            target_dir=silero_dir,
            required_files=SileroVADModule._SILERO_REQUIRED_FILES,
        )
        logger.info(f"[ModelManager] Silero model dir resolved: {self._silero_model_dir}")

        # ---- 2. Load Silero VAD model (ONNX preferred, PyTorch fallback) ----
        self._load_silero_model()

        # ---- 3. Resolve Smart Turn model directory + download check ----
        smart_turn_enabled = config.get("vad", "smart_turn_enabled", default=True)
        self._smart_turn_threshold = config.get("vad", "smart_turn_threshold", default=0.5)

        if smart_turn_enabled:
            smart_turn_path = config.get("vad", "smart_turn_path", default=None)
            if smart_turn_path:
                smart_turn_dir = smart_turn_path
            else:
                smart_turn_dir = _get_modelscope_cache_dir(SmartTurnDetector._MODELSCOPE_MODEL_ID)

            # Ensure model files are downloaded
            smart_turn_dir = _ensure_model_downloaded(
                model_id=SmartTurnDetector._MODELSCOPE_MODEL_ID,
                target_dir=smart_turn_dir,
                required_files=SmartTurnDetector._REQUIRED_FILES,
            )
            logger.info(f"[ModelManager] Smart Turn model dir resolved: {smart_turn_dir}")

            # ---- 4. Load Smart Turn model artifacts ----
            self._load_smart_turn_model(smart_turn_dir)
        else:
            logger.info("[ModelManager] Smart Turn disabled, skipping preload")

        self._preloaded = True
        elapsed = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"[ModelManager] ✅ Model preloading complete in {elapsed:.0f}ms "
            f"(vad_backend={self._vad_backend}, "
            f"smart_turn={'loaded' if self._smart_turn_onnx_session else 'disabled'})"
        )

        # Register as global singleton
        ModelManager._instance = self

    def _load_silero_model(self):
        """Load Silero VAD model (ONNX preferred, PyTorch fallback)."""
        import os

        # Try ONNX first
        onnx_path = os.path.join(self._silero_model_dir, "silero_vad.onnx")
        if os.path.isfile(onnx_path):
            try:
                import onnxruntime
                logger.info(f"[ModelManager] Loading Silero ONNX model: {onnx_path}")
                self._onnx_session = onnxruntime.InferenceSession(onnx_path)

                # Semantic validation
                sample_rate = 16000
                context_size = 64
                test_audio = SileroVADModule._generate_speech_like_signal_static(sample_rate)
                test_state = np.zeros((2, 1, 128), dtype=np.float32)
                test_context = np.zeros((1, context_size), dtype=np.float32)
                test_sr = np.array(sample_rate, dtype=np.int64)

                test_input = np.concatenate([test_context, test_audio], axis=1)
                ort_outs = self._onnx_session.run(
                    None,
                    {"input": test_input, "state": test_state, "sr": test_sr}
                )

                if len(ort_outs) != 2:
                    raise RuntimeError(f"ONNX output count abnormal: {len(ort_outs)}")

                prob_val = ort_outs[0].item()
                state_shape = ort_outs[1].shape
                if state_shape != (2, 1, 128):
                    raise RuntimeError(f"ONNX state shape abnormal: {state_shape}")

                if prob_val < 0.01:
                    raise RuntimeError(
                        f"ONNX semantic validation failed: synthetic speech prob={prob_val:.6f} < 0.01"
                    )

                self._vad_backend = "onnx"
                logger.info(
                    f"[ModelManager] ✅ Silero ONNX model loaded and validated "
                    f"(test_prob={prob_val:.4f})"
                )
                return

            except Exception as e:
                logger.warning(f"[ModelManager] Silero ONNX load/validation failed: {e}")
                self._onnx_session = None

        # Fallback to PyTorch
        jit_path = os.path.join(self._silero_model_dir, "silero_vad.jit")
        if os.path.isfile(jit_path):
            try:
                import torch
                logger.info(f"[ModelManager] Loading Silero PyTorch model: {jit_path}")
                self._pytorch_model = torch.jit.load(jit_path)
                self._pytorch_model.eval()

                # Validate
                self._pytorch_model.reset_states()
                silence = torch.zeros(512, dtype=torch.float32)
                with torch.no_grad():
                    silence_prob = self._pytorch_model(silence, 16000).item()

                speech_signal = SileroVADModule._generate_speech_like_signal_static(16000).flatten()
                self._pytorch_model.reset_states()
                with torch.no_grad():
                    speech_prob = self._pytorch_model(torch.from_numpy(speech_signal), 16000).item()

                self._pytorch_model.reset_states()

                if speech_prob < 0.01:
                    logger.warning(
                        f"[ModelManager] PyTorch VAD validation abnormal: speech_prob={speech_prob:.6f}"
                    )
                    self._pytorch_model = None
                else:
                    self._vad_backend = "pytorch"
                    logger.info(
                        f"[ModelManager] ✅ Silero PyTorch model loaded and validated "
                        f"(silence_prob={silence_prob:.4f}, speech_prob={speech_prob:.4f})"
                    )
                return

            except Exception as e:
                logger.warning(f"[ModelManager] Silero PyTorch load failed: {e}")
                self._pytorch_model = None

        self._vad_backend = "none"
        logger.error(
            "[ModelManager] ❌ Silero VAD model loading failed! "
            "Neither ONNX nor PyTorch backend available."
        )

    def _load_smart_turn_model(self, model_dir: str):
        """Load Smart Turn ONNX model + WhisperFeatureExtractor."""
        import os

        # Load feature extractor
        try:
            from transformers import WhisperFeatureExtractor
            self._smart_turn_feature_extractor = WhisperFeatureExtractor(chunk_length=8)
            logger.info("[ModelManager] Smart Turn WhisperFeatureExtractor loaded (chunk_length=8)")
        except Exception as e:
            logger.error(f"[ModelManager] Smart Turn WhisperFeatureExtractor load failed: {e}")
            return

        # Load ONNX model
        onnx_path = os.path.join(model_dir, "smart-turn-v3.2-gpu.onnx")
        if not os.path.isfile(onnx_path):
            logger.error(f"[ModelManager] Smart Turn ONNX model not found: {onnx_path}")
            return

        try:
            import onnxruntime
        except ImportError:
            logger.error("[ModelManager] onnxruntime not installed, cannot load Smart Turn")
            return

        provider_priority = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        errors = []

        for provider in provider_priority:
            try:
                available_providers = onnxruntime.get_available_providers()
                if provider not in available_providers:
                    errors.append(f"{provider}: provider not available")
                    continue

                providers = [provider]
                if provider != "CPUExecutionProvider" and "CPUExecutionProvider" in available_providers:
                    providers.append("CPUExecutionProvider")

                session = onnxruntime.InferenceSession(onnx_path, providers=providers)

                # Validate input
                input_info = session.get_inputs()
                if len(input_info) != 1 or input_info[0].name != "input_features":
                    raise RuntimeError(
                        f"ONNX input abnormal: expected input_features, got {[i.name for i in input_info]}"
                    )

                actual_shape = tuple(input_info[0].shape)
                if len(actual_shape) != 3:
                    raise RuntimeError(
                        f"ONNX input shape dimensions abnormal: expected 3D, got {len(actual_shape)}D"
                    )

                # Semantic validation with zero input
                test_input = np.zeros(SmartTurnDetector.FEATURE_SHAPE, dtype=np.float32)
                test_out = session.run(None, {"input_features": test_input})
                if len(test_out) < 1:
                    raise RuntimeError("ONNX output is empty")

                logits_val = float(test_out[0].flatten()[0])
                sigmoid_val = 1.0 / (1.0 + np.exp(-logits_val))
                if not (0.0 <= sigmoid_val <= 1.0):
                    raise RuntimeError(
                        f"ONNX semantic validation failed: logits={logits_val:.4f}, sigmoid={sigmoid_val:.4f}"
                    )

                self._smart_turn_onnx_session = session
                self._smart_turn_provider = provider
                self._smart_turn_model_path = onnx_path
                logger.info(
                    f"[ModelManager] ✅ Smart Turn ONNX model loaded: {onnx_path}, "
                    f"provider={provider}, test_logits={logits_val:.4f}, sigmoid={sigmoid_val:.4f}"
                )
                return

            except Exception as e:
                errors.append(f"{provider}: {e}")
                continue

        error_detail = "\n  - ".join(errors)
        logger.error(
            f"[ModelManager] ❌ Smart Turn ONNX model loading failed:\n  - {error_detail}"
        )

    def create_vad_module(self, config) -> SileroVADModule:
        """Create a lightweight SileroVADModule that reuses preloaded models.

        This should be called when a new WebSocket session is established.
        The returned VAD module shares the preloaded model artifacts but has
        its own independent per-session state (LSTM state, counters, etc.).

        Args:
            config: ServerConfig instance

        Returns:
            SileroVADModule instance with shared model artifacts
        """
        if not self._preloaded:
            raise RuntimeError(
                "ModelManager.preload_models() must be called before create_vad_module()"
            )

        return SileroVADModule.from_preloaded(
            onnx_session=self._onnx_session,
            pytorch_model=self._pytorch_model,
            vad_backend=self._vad_backend,
            silero_model_dir=self._silero_model_dir,
            smart_turn_onnx_session=self._smart_turn_onnx_session,
            smart_turn_feature_extractor=self._smart_turn_feature_extractor,
            smart_turn_provider=self._smart_turn_provider,
            smart_turn_model_path=self._smart_turn_model_path,
            smart_turn_threshold=self._smart_turn_threshold,
            # Per-session config from ServerConfig
            threshold=config.get("vad", "threshold", default=0.5),
            silence_ms=config.get("vad", "silence_timeout_ms", default=500),
            prefix_padding_ms=config.get("vad", "prefix_padding_ms", default=300),
            sample_rate=config.get("vad", "sample_rate", default=16000),
            min_speech_duration_ms=config.get("vad", "min_speech_duration_ms", default=200),
            max_audio_duration_ms=config.get("vad", "max_audio_duration_ms", default=30000),
            smart_turn_enabled=config.get("vad", "smart_turn_enabled", default=True),
        )
