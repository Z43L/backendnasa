#!/usr/bin/env python3
"""
Quick Backend Development Server

Simple script to start the backend in development mode with auto-reload.
Optimized for development workflow.

Usage:
    python dev_server.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start development server."""
    backend_dir = Path(__file__).parent

    print("🚀 Starting Seismic AI Backend - Development Mode")
    print("=" * 55)
    print("🔄 Auto-reload enabled")
    print("📝 Logs will be shown in console")
    print("🛑 Press Ctrl+C to stop")
    print()

    # Change to backend directory
    os.chdir(backend_dir)

    # Start server with development settings
    cmd = [
        sys.executable, "start_backend.py",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload",
        "--log-level", "info"
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Development server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()