"""Mode Router - decides between Mode B (Omni audio) and Mode A (ASR+LLM text)."""
import time
import logging

logger = logging.getLogger("realtime-server")


class ModeRouter:
    """Decides whether to use Mode B (Omni audio input) or Mode A (ASR + LLM text).
    
    Switching conditions:
    - Default: Mode B
    - Omni request timeout/error → temporary fallback to Mode A
    - Omni consecutive errors > threshold → persistent fallback to Mode A
    - Manual config to force mode
    """

    MODE_B = "omni_audio"  # Default: Omni audio input + TTS
    MODE_A = "asr_llm"     # Fallback: ASR + LLM text + TTS

    def __init__(self, config: dict):
        self.mode = config.get("default_mode", self.MODE_B)
        self.omni_error_count = 0
        self.omni_error_threshold = config.get("omni_error_threshold", 3)
        self.fallback_duration = config.get("fallback_duration_s", 60)
        self.fallback_until = 0  # timestamp

    def get_mode(self) -> str:
        """Get current mode."""
        if self.mode == self.MODE_A:
            return self.MODE_A
        if time.time() < self.fallback_until:
            return self.MODE_A
        return self.MODE_B

    def report_omni_error(self):
        """Report an Omni error, may trigger fallback."""
        self.omni_error_count += 1
        logger.warning(f"Omni error count: {self.omni_error_count}/{self.omni_error_threshold}")
        if self.omni_error_count >= self.omni_error_threshold:
            self.fallback_until = time.time() + self.fallback_duration
            logger.warning(
                f"Omni error threshold reached, falling back to Mode A for {self.fallback_duration}s"
            )

    def report_omni_success(self):
        """Report an Omni success, reset error count."""
        self.omni_error_count = 0

    def force_mode(self, mode: str):
        """Force a specific mode."""
        if mode in (self.MODE_A, self.MODE_B):
            self.mode = mode
            logger.info(f"Mode forced to: {mode}")
