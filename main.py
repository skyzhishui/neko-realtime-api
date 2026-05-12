#!/usr/bin/env python3
"""LocalOmniRealtimeServer - Main entry point."""
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.config import ServerConfig
import uvicorn


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


def main():
    setup_logging()
    logger = logging.getLogger("realtime-server")
    
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    config = ServerConfig.load(config_path)
    
    logger.info(f"Starting LocalOmniRealtimeServer on {config.host}:{config.port}")
    logger.info(f"Default mode: {config.default_mode}")
    logger.info(f"Auth enabled: {config.auth_enabled}")
    
    # Run server
    uvicorn.run(
        "server.websocket:app",
        host=config.host,
        port=config.port,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=60,
    )


if __name__ == "__main__":
    main()
