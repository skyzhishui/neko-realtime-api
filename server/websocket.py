"""WebSocket Server for LocalOmniRealtimeServer."""
import asyncio
import json
import logging
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
        logger.info("[Startup] local_asr=true, pre-loading LocalASREngine...")
        # LocalASREngine.__init__ is synchronous and may take several seconds
        # (model loading + warm-up), so run it in a thread to avoid blocking
        # the event loop during startup.
        await loop.run_in_executor(None, LocalASREngine, asr_model_path)
        logger.info("[Startup] LocalASREngine pre-loaded and warmed up \u2713")
    else:
        logger.info("[Startup] local_asr=false, skipping ASR pre-load")

    yield  # application is running

    # --- Shutdown ---
    logger.info("[Shutdown] Server shutting down")


app = FastAPI(title="LocalOmniRealtimeServer", lifespan=lifespan)


@app.websocket("/v1/realtime")
async def realtime_endpoint(
    websocket: WebSocket,
    model: str = Query(default="local-qwen-omni"),
):
    """WebSocket endpoint for Realtime API."""
    config = ServerConfig.load()
    
    # Optional authentication
    if config.auth_enabled:
        auth = websocket.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != config.auth_token:
            await websocket.close(code=1008, reason="Unauthorized")
            logger.warning("Unauthorized connection attempt")
            return
    
    await websocket.accept()
    logger.info(f"WebSocket connected, model={model}")
    
    # Create session
    session = RealtimeSession(websocket, model, config)
    
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
                logger.warning(f"Invalid JSON received: {e}")
                await session.protocol.send_error(f"Invalid JSON: {str(e)}")
                continue
            
            await session.handle_event(event)
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
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
