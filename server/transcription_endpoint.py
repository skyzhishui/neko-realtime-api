"""OpenAI 兼容的 HTTP ASR 端点：POST /v1/audio/transcriptions。

参考实现来自 10.0.1.93:/home/skyzhishui/asr_server.py，但本模块复用
Realtime 服务已加载的 LocalASREngine 单例（避免重复加载模型），并在
local_asr=false 时通过 SenseVoiceASRClient 转发到远程 ASR 服务。

请求:
    POST /v1/audio/transcriptions
    Content-Type: multipart/form-data
    Headers (可选): Authorization: Bearer <auth_token>
    Form fields:
        file:     音频文件（WAV/MP3/OGG/FLAC 等 soundfile 支持的格式，
                  或原始 int16 PCM 字节流）
        model:    模型名（兼容字段，忽略，始终使用本地/远程配置）
        language: 语言代码（zh/en/ja/ko/auto，默认 auto）

响应 (JSON):
    {"text": "<识别文本>"}
"""
from __future__ import annotations

import asyncio
import hmac
import io
import logging
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from .config import ServerConfig

logger = logging.getLogger("realtime-server")

_TARGET_SAMPLE_RATE = 16000


def warmup_audio_pipeline() -> None:
    """启动时预热 soundfile 解码 + librosa.resample 流水线。

    librosa.resample（soxr 后端）首次调用会触发 lazy import 与 JIT 初始化，
    实测在 24kHz->16kHz 上首次耗时约 900ms，后续 <1ms。预热后可消除该抖动。
    """
    try:
        import soundfile as sf  # noqa: F401
        import librosa
    except ImportError as e:
        logger.warning(f"[transcription] warmup skipped: {e}")
        return

    # 0.5s 假音频，做一次完整的 24k->16k 重采样
    dummy = np.zeros(12000, dtype=np.float32)
    librosa.resample(dummy, orig_sr=24000, target_sr=_TARGET_SAMPLE_RATE, res_type="soxr_mq")
    logger.info("[transcription] audio decode + resample pipeline warmed up")


# ---------------------------------------------------------------------------
# 认证
# ---------------------------------------------------------------------------
def _resolve_auth_settings(config: ServerConfig) -> tuple[bool, str]:
    """从配置解析 (auth_enabled, auth_token)。

    认证开关解析顺序：security.auth_enabled -> realtime_server.auth_enabled -> True
    Token 解析顺序：security.auth_token -> realtime_server.auth_token -> ""
    """
    sec_auth = config.get("security", "auth_enabled", default=None)
    legacy_auth = config.get("realtime_server", "auth_enabled", default=None)
    if sec_auth is not None:
        auth_enabled = bool(sec_auth)
    elif legacy_auth is not None:
        auth_enabled = bool(legacy_auth)
    else:
        auth_enabled = True

    raw_token = config.get("security", "auth_token", default=None)
    if raw_token is None or str(raw_token).strip() == "":
        raw_token = config.get("realtime_server", "auth_token", default="")
    auth_token = "" if raw_token is None else str(raw_token)
    return auth_enabled, auth_token


def _check_bearer_auth(
    request: Request, auth_enabled: bool, auth_token: str
) -> None:
    """与 WebSocket 端点保持一致的 Bearer 认证逻辑。

    auth_enabled 与 auth_token 在路由注册时一次性解析，避免每次请求重读配置。
    """
    if not auth_enabled:
        return

    auth_header = request.headers.get("authorization", "")
    provided_token = auth_header[7:] if auth_header.startswith("Bearer ") else ""

    if (
        not provided_token
        or not auth_token
        or not hmac.compare_digest(
            provided_token.encode("utf-8"), auth_token.encode("utf-8")
        )
    ):
        logger.warning(
            "[transcription] Unauthorized request: missing or invalid bearer token"
        )
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# 音频解码
# ---------------------------------------------------------------------------
def _decode_audio_to_float32(content: bytes) -> tuple[np.ndarray, int]:
    """把上传的音频字节解码为 float32 mono numpy 数组。

    解码优先级：
      1) soundfile：覆盖 WAV/FLAC/OGG/MP3 等通用格式
      2) 兜底：直接当作 int16 PCM 处理（采样率默认 16k）

    Returns:
        (audio_float32, sample_rate) - audio_float32 范围 [-1, 1]，单声道
    """
    # 路径 1: soundfile 解码
    try:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(content), dtype="float32", always_2d=False)
        # 多声道 -> 单声道平均
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32, copy=False)
        return audio, int(sr)
    except Exception as e:
        logger.debug(f"[transcription] soundfile decode failed, falling back to raw PCM: {e}")

    # 路径 2: 兜底 - 当作原始 int16 PCM
    if len(content) % 2 != 0:
        # 奇数字节，截掉最后一个
        content = content[:-1]
    audio_int16 = np.frombuffer(content, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0
    return audio_float, _TARGET_SAMPLE_RATE


def _resample_if_needed(audio: np.ndarray, sr: int) -> np.ndarray:
    """需要时把 audio 重采样到 16kHz。"""
    if sr == _TARGET_SAMPLE_RATE:
        return audio
    try:
        import librosa

        return librosa.resample(
            audio, orig_sr=sr, target_sr=_TARGET_SAMPLE_RATE, res_type="soxr_mq"
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"librosa not installed but resampling required ({sr}Hz -> {_TARGET_SAMPLE_RATE}Hz)",
        ) from e


# ---------------------------------------------------------------------------
# 推理：本地优先 + 远程兜底
# ---------------------------------------------------------------------------
async def _transcribe_local(
    audio_float: np.ndarray, language: str
) -> Optional[str]:
    """通过 LocalASREngine 单例做本地推理。返回 None 表示引擎未加载。"""
    from .local_asr import LocalASREngine

    engine = LocalASREngine.get_instance()
    if engine is None:
        return None

    # 把 float32 转回 int16 PCM bytes，复用现有的 transcribe 接口（已包含
    # asyncio.Lock + run_in_executor 的线程安全保证）。
    audio_int16 = np.clip(audio_float * 32768.0, -32768, 32767).astype(np.int16)
    pcm_bytes = audio_int16.tobytes()

    return await engine.transcribe(
        pcm_bytes=pcm_bytes,
        sample_rate=_TARGET_SAMPLE_RATE,
        language=language,
    )


async def _transcribe_remote(
    audio_float: np.ndarray, language: str, base_url: str, timeout_s: int
) -> Optional[str]:
    """通过远程 ASR 服务做转写。base_url 例: http://10.0.1.93:8082/v1。

    返回 None 表示请求失败。
    """
    audio_int16 = np.clip(audio_float * 32768.0, -32768, 32767).astype(np.int16)
    pcm_bytes = audio_int16.tobytes()

    # SenseVoiceASRClient.base_url 期望不带 /v1 前缀，但内部会拼 /audio/transcriptions
    # 这里直接走 aiohttp，避免对客户端 URL 约定的耦合。
    import aiohttp
    import wave

    # PCM -> WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_TARGET_SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    wav_bytes = buf.getvalue()

    # 拼接出最终 URL
    url = base_url.rstrip("/")
    if not url.endswith("/audio/transcriptions"):
        url = f"{url}/audio/transcriptions"

    data = aiohttp.FormData()
    data.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
    data.add_field("model", "SenseVoiceSmall")
    data.add_field("language", language)

    try:
        timeout = aiohttp.ClientTimeout(total=max(1, timeout_s))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=data) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.error(
                        f"[transcription] remote ASR error {resp.status}: {err[:300]}"
                    )
                    return None
                result = await resp.json()
                return result.get("text", "")
    except Exception as e:
        logger.error(f"[transcription] remote ASR request failed: {e}")
        return None


# ---------------------------------------------------------------------------
# FastAPI 路由注册
# ---------------------------------------------------------------------------
def register_transcription_routes(app: FastAPI) -> None:
    """把 OpenAI 兼容的 /v1/audio/transcriptions 路由挂到 FastAPI app 上。

    配置在路由注册时一次性读取并闭包到 endpoint 内，请求路径上不再访问
    ServerConfig，避免每次请求重复 lookup 配置项。
    """
    config = ServerConfig.load()
    auth_enabled, auth_token = _resolve_auth_settings(config)
    local_enabled = bool(config.get("services", "asr", "local_asr", default=False))
    remote_base_url = config.get("services", "asr", "base_url", default=None)
    remote_timeout_s = int(config.get("services", "asr", "timeout_s", default=10))

    @app.post("/v1/audio/transcriptions")
    async def transcribe_endpoint(  # noqa: D401 - FastAPI route
        request: Request,
        file: UploadFile = File(...),
        model: str = Form("SenseVoiceSmall"),  # 兼容字段，忽略
        language: str = Form("auto"),
    ):
        """OpenAI 兼容的语音转文字端点。

        - 认证：与 WebSocket 端点共用 Bearer Token (security.auth_token)
        - 引擎选择：local_asr=true 时优先本地；否则转发到 services.asr.base_url；
          两者皆不可用时返回 503。
        """
        _check_bearer_auth(request, auth_enabled, auth_token)

        # 读取并解码音频
        try:
            content = await file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")

        if not content:
            raise HTTPException(status_code=400, detail="Empty audio payload")

        audio_float, sr = _decode_audio_to_float32(content)
        if audio_float.size == 0:
            raise HTTPException(status_code=400, detail="Decoded audio is empty")

        audio_float = _resample_if_needed(audio_float, sr)

        # 本地优先（启动时已确定 local_enabled）
        text: Optional[str] = None

        if local_enabled:
            text = await _transcribe_local(audio_float, language)
            if text is None:
                logger.warning(
                    "[transcription] local_asr=true but LocalASREngine "
                    "instance not available, falling back to remote"
                )

        # 远程兜底（local_asr=false，或本地未加载）
        if text is None and remote_base_url:
            text = await _transcribe_remote(
                audio_float, language, remote_base_url, remote_timeout_s
            )

        if text is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "ASR backend unavailable: enable services.asr.local_asr=true "
                    "or set services.asr.base_url to a remote ASR endpoint."
                ),
            )

        return {"text": text}
