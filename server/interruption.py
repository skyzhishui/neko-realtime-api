"""Interruption Handler - simplified cancel flag mechanism.

重写要点：
- 去掉 task 跟踪（pipeline 取消由 session 的 _active_pipeline_task.cancel() 负责）
- 只保留 _cancel_flag（asyncio.Event）作为纯标志机制
- should_cancel() 保留作为辅助检查（某些场景下 CancelledError 可能被吞掉）
- handle_interruption 简化为：设置取消标志 + 发送 response.done
"""
import asyncio
import logging

logger = logging.getLogger("realtime-server")


class InterruptionHandler:
    """Simplified interruption mechanism using a cancel flag.
    
    The actual pipeline cancellation is handled by session._cancel_active_pipeline()
    which calls asyncio.Task.cancel() to propagate CancelledError through the
    entire pipeline (LLM → TTS → audio output).
    
    This handler only provides:
    1. A cancel flag (asyncio.Event) for should_cancel() checks
    2. A helper to send response.done(cancelled) to the client
    """

    def __init__(self):
        self._is_generating = False
        self._cancel_event = asyncio.Event()

    @property
    def is_generating(self) -> bool:
        return self._is_generating

    def set_generating(self, value: bool):
        """Set the generating state and update cancel flag accordingly."""
        self._is_generating = value
        if value:
            self._cancel_event.clear()
        else:
            self._cancel_event.set()

    def should_cancel(self) -> bool:
        """Check if current generation should be cancelled.
        
        This is a supplementary check for cases where CancelledError
        might be swallowed by library code (e.g., aiohttp, websockets).
        The primary cancellation mechanism is asyncio.Task.cancel().
        """
        return self._cancel_event.is_set()

    async def handle_interruption(self, protocol, resp_id: str):
        """Handle interruption: set cancel flag + send response.done(cancelled) to client.
        
        The actual pipeline task cancellation is handled by the caller
        (session._cancel_active_pipeline), which calls asyncio.Task.cancel().
        This method only handles the protocol-level response.
        """
        if not self._is_generating:
            return

        logger.info(f"[Interruption] Handling interruption for response {resp_id}")

        # Set cancel flag
        self._is_generating = False
        self._cancel_event.set()

        # Send response.done to client (cancelled)
        try:
            await protocol.send_response_done(resp_id)
        except Exception as e:
            logger.warning(f"[Interruption] Failed to send response.done during interruption: {e}")

    def reset(self):
        """Reset handler state."""
        self._is_generating = False
        self._cancel_event.set()
