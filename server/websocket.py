"""WebSocket Server for LocalOmniRealtimeServer."""
import asyncio
import hmac
import json
import logging
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

from .session import RealtimeSession
from .config import ServerConfig

logger = logging.getLogger("realtime-server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle: pre-load models at startup."""
    # --- Startup ---
    config = ServerConfig.load()
    
    # 1. Pre-load VAD + Smart Turn models (via ModelManager)
    from .model_manager import ModelManager
    logger.info("[Startup] Pre-loading VAD and Smart Turn models via ModelManager...")
    manager = ModelManager()
    # Model loading is synchronous and may take several seconds
    # (model download check + ONNX/PyTorch loading + validation),
    # so run it in a thread to avoid blocking the event loop during startup.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, manager.preload_models, config)
    logger.info("[Startup] VAD and Smart Turn models pre-loaded \u2713")
    
    # 2. Pre-load LocalASR if configured
    if config.get("services", "asr", "local_asr", default=False):
        from .local_asr import LocalASREngine
        asr_model_path = config.get("services", "asr", "asr_model_path", default=None)
        asr_device = config.get("services", "asr", "device", default="cuda")
        logger.info(f"[Startup] local_asr=true, pre-loading LocalASREngine (device={asr_device})...")
        # LocalASREngine.__init__ is synchronous and may take several seconds
        # (model loading + warm-up), so run it in a thread to avoid blocking
        # the event loop during startup.
        await loop.run_in_executor(None, LocalASREngine, asr_model_path, asr_device)
        logger.info("[Startup] LocalASREngine pre-loaded and warmed up \u2713")
    else:
        logger.info("[Startup] local_asr=false, skipping ASR pre-load")

    # 3. 预热 HTTP ASR 端点的音频解码/重采样流水线（首次 librosa.resample
    #    在 24k->16k 上耗时 ~900ms，预热后可消除该抖动）
    from .transcription_endpoint import warmup_audio_pipeline
    await loop.run_in_executor(None, warmup_audio_pipeline)

    yield  # application is running

    # --- Shutdown ---
    logger.info("[Shutdown] Server shutting down")


app = FastAPI(title="LocalOmniRealtimeServer", lifespan=lifespan)

# 挂载 OpenAI 兼容的 ASR HTTP 端点 POST /v1/audio/transcriptions
# 复用启动时预加载的 LocalASREngine（local_asr=true 时）或转发到远程 ASR
from .transcription_endpoint import register_transcription_routes  # noqa: E402

register_transcription_routes(app)


@app.websocket("/v1/realtime")
async def realtime_endpoint(
    websocket: WebSocket,
    model: str = Query(default="local-qwen-omni"),
):
    """WebSocket endpoint for Realtime API."""
    config = ServerConfig.load()

    # P0-fix (Oracle blocker #2): unified auth source.
    # Prefer security.* (new canonical location) over realtime_server.* (legacy).
    # Default to True when neither is set — secure-by-default.
    sec_auth = config.get("security", "auth_enabled", default=None)
    legacy_auth = config.get("realtime_server", "auth_enabled", default=None)
    if sec_auth is not None:
        auth_enabled = bool(sec_auth)
    elif legacy_auth is not None:
        auth_enabled = bool(legacy_auth)
    else:
        auth_enabled = True

    # P0-fix (Oracle blocker #4): tolerate None / non-string token (YAML null → Python None).
    raw_token = config.get("security", "auth_token", default=None)
    if raw_token is None or str(raw_token).strip() == "":
        raw_token = config.get("realtime_server", "auth_token", default="")
    auth_token = "" if raw_token is None else str(raw_token)

    if auth_enabled:
        # P0-fix (Oracle blocker #3): correct empty-allowlist semantics.
        # security.allowed_origins == []  → allow ALL origins (matches yaml comment)
        # security.allowed_origins == [non-empty]  → strict allowlist
        # Server-to-server clients with no Origin header always pass this check.
        allowed_origins = config.get("security", "allowed_origins", default=[])
        if not isinstance(allowed_origins, list):
            allowed_origins = []
        origin = websocket.headers.get("origin", "")
        if origin and len(allowed_origins) > 0:
            # Strict allowlist: origin must match exactly
            if origin not in allowed_origins:
                logger.warning(f"Rejected WebSocket from disallowed origin: {origin}")
                await websocket.close(code=4401, reason="Origin not allowed")
                return
        # else: empty allowlist (allow all) OR no Origin header (server-to-server) → pass

        # Token validation using constant-time comparison
        auth = websocket.headers.get("authorization", "")
        provided_token = auth[7:] if auth.startswith("Bearer ") else ""
        # Both sides forced to bytes; auth_token is now guaranteed to be a str
        if not provided_token or not auth_token or not hmac.compare_digest(
            provided_token.encode("utf-8"), auth_token.encode("utf-8")
        ):
            logger.warning("Unauthorized connection attempt: missing or invalid bearer token")
            await websocket.close(code=4401, reason="Unauthorized")
            return
    
    await websocket.accept()
    logger.info(f"WebSocket connected, model={model}")
    
    # Create session
    session = RealtimeSession(websocket, model, config)

    # Notify client that session has been created
    await session.protocol.send_session_created(session.session_config)

    # Main event loop
    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except Exception:
                # Connection closed
                break
            
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as e:
                error_id = str(uuid.uuid4())[:8]
                logger.warning(f"[{error_id}] Invalid JSON received: {e}")
                await session.protocol.send_error(f"Invalid JSON (ref {error_id})")
                continue
            
            await session.handle_event(event)
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        error_id = str(uuid.uuid4())[:8]
        logger.error(f"[{error_id}] WebSocket error: {e}", exc_info=True)
    finally:
        await session.cleanup()
        logger.info("Session cleaned up")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "LocalOmniRealtimeServer"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "LocalOmniRealtimeServer",
        "version": "0.1.0",
        "websocket_endpoint": "/v1/realtime",
    }
