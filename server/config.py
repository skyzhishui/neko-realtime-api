"""Configuration loader for LocalOmniRealtimeServer."""
import yaml
import os
from typing import Any


_DEFAULT_CONFIG = {
    "realtime_server": {
        "host": "0.0.0.0",
        "port": 8765,
        "auth_enabled": False,
        "auth_token": "local-no-key-needed",
        "default_mode": "omni_audio",
        "omni_error_threshold": 3,
        "fallback_duration_s": 60,
    },
    "vad": {
        "engine": "silero",
        "threshold": 0.5,
        "silence_timeout_ms": 500,
        "min_speech_duration_ms": 200,
        "max_audio_duration_ms": 30000,
        "prefix_padding_ms": 300,
        "sample_rate": 16000,
        "smart_turn_enabled": True,
        "smart_turn_threshold": 0.5,
        "silero_model_path": "./models/silero-vad",
        "smart_turn_path": "./models/smart-turn-v3",
    },
    "services": {
        "omni": {
            "base_url": "http://localhost:8000",
            "model": "Qwen3-Omni",
            "timeout_s": 30,
            "max_tokens": 4096,
            "temperature": 0.7,
            "repetition_penalty": 1.2,
        },
        "asr": {
            "base_url": "http://localhost:8082",
            "model": "SenseVoiceSmall",
            "language": "zh",
            "timeout_s": 10,
            "use_ws": False,
            "local_asr": False,
            "asr_model_path": None,
        },
        "tts": {
            "base_url": "http://localhost:8091",
            "ws_url": "ws://localhost:8091/v1/audio/speech/stream",
            "mode": "http",
            "model": "Qwen3-TTS",
            "voice": "Vivian",
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
            "language": "Chinese",
            "sample_rate": 24000,
            "timeout_s": 15,
        },
        "gsv_tts": {
            "enabled": False,
            "base_url": "http://localhost:8001",
            "speaker_audio": "",
            "prompt_audio": "",
            "prompt_text": "",
            "sample_rate": 32000,
            "timeout_s": 30,
            "speed": 1.0,
        },
    },
    "tts_pipeline": {
        "min_sub_sentence_len": 6,
        "max_concurrent_tts": 2,
        "resample_output": True,
        "output_sample_rate": 16000,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, returning new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ServerConfig:
    """Server configuration singleton."""

    _instance: "ServerConfig | None" = None

    def __init__(self, config_path: str | None = None):
        self._data = _DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
            self._data = _deep_merge(self._data, file_config)

    @classmethod
    def load(cls, config_path: str | None = None) -> "ServerConfig":
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get nested config value by dot path keys."""
        obj = self._data
        for key in keys:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return default
        return obj

    @property
    def host(self) -> str:
        return self.get("realtime_server", "host", default="0.0.0.0")

    @property
    def port(self) -> int:
        return self.get("realtime_server", "port", default=8765)

    @property
    def auth_enabled(self) -> bool:
        return self.get("realtime_server", "auth_enabled", default=False)

    @property
    def auth_token(self) -> str:
        return self.get("realtime_server", "auth_token", default="local-no-key-needed")

    @property
    def default_mode(self) -> str:
        return self.get("realtime_server", "default_mode", default="omni_audio")

    def __repr__(self) -> str:
        return f"ServerConfig({self._data})"
