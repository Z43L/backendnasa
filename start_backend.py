#!/usr/bin/env python3
"""
Backend Startup Script for Seismic AI System

This script initializes and starts all backend services for the complete
Seismic AI system including:
- FastAPI server with all endpoints
- AI agents (weather, sea level, seismic detection)
- Model inference services
- Alert monitoring system
- Real-time seismic detection

Usage:
    python start_backend.py [--host HOST] [--port PORT] [--reload]

Environment Variables:
    SEISMIC_AI_HOST: Server host (default: 127.0.0.1)
    SEISMIC_AI_PORT: Server port (default: 8000)
    SEISMIC_AI_RELOAD: Enable auto-reload (default: false)
    SEISMIC_AI_LOG_LEVEL: Logging level (default: info)
"""

import os
import sys
import time
import signal
import logging
import argparse
from pathlib import Path
from typing import Optional

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_logging(log_level: str = "info"):
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(current_dir / "logs" / "backend_startup.log")
        ]
    )

    # Reduce noise from some libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)

    return logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available."""
    logger = logging.getLogger(__name__)

    required_modules = [
        "fastapi", "uvicorn", "torch", "numpy", "pydantic"
    ]

    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            logger.debug(f"✅ {module} available")
        except ImportError:
            missing_modules.append(module)
            logger.warning(f"❌ {module} not available")

    if missing_modules:
        logger.error(f"Missing required modules: {', '.join(missing_modules)}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        return False

    logger.info("✅ All core dependencies available")
    return True

def initialize_backend_components():
    """Initialize all backend components before starting the server."""
    logger = logging.getLogger(__name__)

    logger.info("🚀 Initializing Seismic AI Backend Components...")

    components_status = {}

    # Test model loading
    try:
        logger.info("Testing model loading...")
        from model_loader import model_loader
        available_models = model_loader.get_available_models()
        logger.info(f"📊 Available models: {available_models}")
        components_status["model_loader"] = True
    except Exception as e:
        logger.warning(f"⚠️ Model loader initialization failed: {e}")
        components_status["model_loader"] = False

    # Test inference service
    try:
        logger.info("Testing inference service...")
        from inference_service import SeismicInferenceService
        inference_service = SeismicInferenceService()
        model_status = inference_service.get_model_status()
        logger.info(f"🤖 Inference service models: {model_status}")
        components_status["inference_service"] = True
    except Exception as e:
        logger.warning(f"⚠️ Inference service initialization failed: {e}")
        components_status["inference_service"] = False

    # Test alert system
    try:
        logger.info("Testing alert system...")
        from alert_system import alert_manager, initialize_monitoring
        initialize_monitoring()
        logger.info("🚨 Alert system initialized")
        components_status["alert_system"] = True
    except Exception as e:
        logger.warning(f"⚠️ Alert system initialization failed: {e}")
        components_status["alert_system"] = False

    # Test seismic detection agent
    try:
        logger.info("Testing seismic detection agent...")
        from realtime_seismic_detector import RealTimeSeismicDetector
        seismic_detector = RealTimeSeismicDetector()
        stats = seismic_detector.get_stats()
        logger.info(f"🔔 Seismic detector ready - Zones: {stats.get('monitoring_zones', 0)}")
        components_status["seismic_detector"] = True
    except Exception as e:
        logger.warning(f"⚠️ Seismic detector initialization failed: {e}")
        components_status["seismic_detector"] = False

    # Test AI agents
    try:
        logger.info("Testing AI agents...")
        agents_status = {}

        # Weather agent
        try:
            from simple_weather_agent import SimpleWeatherAgent
            weather_agent = SimpleWeatherAgent()
            if weather_agent.initialize():
                agents_status["weather_agent"] = True
                logger.info("✅ Weather agent initialized")
            else:
                agents_status["weather_agent"] = False
                logger.warning("⚠️ Weather agent initialization failed")
        except Exception as e:
            agents_status["weather_agent"] = False
            logger.warning(f"⚠️ Weather agent import failed: {e}")

        # Sea level agent
        try:
            from sea_level_analyzer import SeaLevelAnalyzer
            sea_level_agent = SeaLevelAnalyzer()
            agents_status["sea_level_agent"] = True
            logger.info("✅ Sea level agent initialized")
        except Exception as e:
            agents_status["sea_level_agent"] = False
            logger.warning(f"⚠️ Sea level agent import failed: {e}")

        components_status["ai_agents"] = agents_status

    except Exception as e:
        logger.warning(f"⚠️ AI agents testing failed: {e}")
        components_status["ai_agents"] = {"error": str(e)}

    # Summary
    successful_components = sum(1 for status in components_status.values()
                               if status is True or (isinstance(status, dict) and any(status.values())))

    total_components = len(components_status)

    logger.info(f"📊 Component initialization complete: {successful_components}/{total_components} successful")

    if successful_components == total_components:
        logger.info("🎉 All backend components initialized successfully!")
    elif successful_components >= total_components * 0.7:  # At least 70% success
        logger.info("⚠️ Backend partially initialized - some components may not work")
    else:
        logger.warning("💥 Backend initialization heavily degraded - check dependencies")

    return components_status

def create_logs_directory():
    """Create logs directory if it doesn't exist."""
    logs_dir = current_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir

def start_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start the FastAPI server with uvicorn."""
    logger = logging.getLogger(__name__)

    logger.info(f"🌐 Starting FastAPI server on {host}:{port}")

    # Import here to avoid circular imports
    import uvicorn

    # Server configuration
    config = uvicorn.Config(
        "simple_api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="warning",  # Reduce uvicorn noise
        access_log=False,     # Disable access logs for cleaner output
        server_header=False,  # Don't expose server info
        date_header=False     # Don't expose date
    )

    server = uvicorn.Server(config)

    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        server.should_exit = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info("🚀 Seismic AI Backend Server Starting...")
        logger.info(f"📡 API Documentation: http://{host}:{port}/docs")
        logger.info(f"🔍 Health Check: http://{host}:{port}/health")
        logger.info("=" * 60)

        server.run()

    except Exception as e:
        logger.error(f"💥 Server startup failed: {e}")
        raise

def main():
    """Main entry point for backend startup."""
    parser = argparse.ArgumentParser(description="Seismic AI Backend Startup Script")
    parser.add_argument("--host", default=os.getenv("SEISMIC_AI_HOST", "127.0.0.1"),
                       help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=int(os.getenv("SEISMIC_AI_PORT", "8000")),
                       help="Server port (default: 8000)")
    parser.add_argument("--reload", action="store_true",
                       default=os.getenv("SEISMIC_AI_RELOAD", "false").lower() == "true",
                       help="Enable auto-reload for development")
    parser.add_argument("--log-level", default=os.getenv("SEISMIC_AI_LOG_LEVEL", "info"),
                       choices=["debug", "info", "warning", "error", "critical"],
                       help="Logging level (default: info)")
    parser.add_argument("--skip-init", action="store_true",
                       help="Skip component initialization (faster startup)")

    args = parser.parse_args()

    # Setup logging first
    logger = setup_logging(args.log_level)

    print("🔔 Seismic AI Backend Startup Script")
    print("=" * 50)
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🔄 Reload: {args.reload}")
    print(f"📝 Log Level: {args.log_level}")
    print()

    # Create logs directory
    create_logs_directory()

    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed. Cannot start backend.")
        sys.exit(1)

    # Initialize components (unless skipped)
    if not args.skip_init:
        components_status = initialize_backend_components()

        # Check if critical components are available
        critical_components = ["inference_service", "alert_system"]
        critical_ok = all(components_status.get(comp, False) for comp in critical_components)

        if not critical_ok:
            logger.warning("⚠️ Some critical components failed to initialize")
            logger.warning("The server will start but some features may not work")
    else:
        logger.info("⏭️ Skipping component initialization (--skip-init)")
        components_status = {}

    # Small delay to let components settle
    time.sleep(1)

    try:
        # Start the server
        start_server(args.host, args.port, args.reload)

    except KeyboardInterrupt:
        logger.info("🛑 Server shutdown requested by user")
    except Exception as e:
        logger.error(f"💥 Fatal error during server startup: {e}")
        sys.exit(1)

    logger.info("👋 Seismic AI Backend shutdown complete")

if __name__ == "__main__":
    main()