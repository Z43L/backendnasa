#!/usr/bin/env python3
"""
Backend Health Check Script

Checks the status of all backend components and services.
Useful for monitoring and troubleshooting.

Usage:
    python check_backend.py [--json] [--quiet]
"""

import os
import sys
import json
import requests
import argparse
from pathlib import Path
from datetime import datetime

def check_server_health(host: str = "127.0.0.1", port: int = 8000, timeout: int = 5):
    """Check if the server is responding."""
    try:
        url = f"http://{host}:{port}/health"
        response = requests.get(url, timeout=timeout)

        if response.status_code == 200:
            data = response.json()
            return {
                "status": "running",
                "response_time": response.elapsed.total_seconds(),
                "health_data": data
            }
        else:
            return {
                "status": "error",
                "error": f"HTTP {response.status_code}",
                "response": response.text[:200]
            }

    except requests.exceptions.ConnectionError:
        return {"status": "not_running", "error": "Connection refused"}
    except requests.exceptions.Timeout:
        return {"status": "timeout", "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def check_models():
    """Check available models."""
    try:
        from model_loader import model_loader
        models = model_loader.get_available_models()
        return {"status": "ok", "models": models}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def check_agents():
    """Check AI agents status."""
    agents_status = {}

    # Weather agent
    try:
        from backend.simple_weather_agent import SimpleWeatherAgent
        weather_agent = SimpleWeatherAgent()
        agents_status["weather_agent"] = weather_agent.initialize()
    except Exception as e:
        agents_status["weather_agent"] = f"Error: {str(e)}"

    # Sea level agent
    try:
        from backend.sea_level_analyzer import SeaLevelAnalyzer
        sea_level_agent = SeaLevelAnalyzer()
        agents_status["sea_level_agent"] = True
    except Exception as e:
        agents_status["sea_level_agent"] = f"Error: {str(e)}"

    # Seismic detector
    try:
        from realtime_seismic_detector import RealTimeSeismicDetector
        seismic_detector = RealTimeSeismicDetector()
        stats = seismic_detector.get_stats()
        agents_status["seismic_detector"] = True
        agents_status["seismic_stats"] = stats
    except Exception as e:
        agents_status["seismic_detector"] = f"Error: {str(e)}"

    return agents_status

def check_dependencies():
    """Check if required dependencies are installed."""
    required_modules = [
        "fastapi", "uvicorn", "torch", "numpy", "pydantic",
        "requests", "scipy", "matplotlib", "xarray", "cartopy"
    ]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    return {
        "total": len(required_modules),
        "missing": missing,
        "available": len(required_modules) - len(missing)
    }

def print_status_report(results, json_output=False, quiet=False):
    """Print status report."""
    if json_output:
        print(json.dumps(results, indent=2, default=str))
        return

    if quiet:
        # Only print essential info
        server_status = results["server"]["status"]
        if server_status == "running":
            print("✅ Backend is running")
        else:
            print(f"❌ Backend status: {server_status}")
        return

    print("🔍 Seismic AI Backend Health Check")
    print("=" * 45)
    print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Server status
    server = results["server"]
    print("🌐 Server Status:")
    if server["status"] == "running":
        print("  ✅ Running"        print(".3f"    else:
        print(f"  ❌ {server['status'].replace('_', ' ').title()}")
        if "error" in server:
            print(f"     Error: {server['error']}")
    print()

    # Dependencies
    deps = results["dependencies"]
    print("📦 Dependencies:")
    print(f"  Available: {deps['available']}/{deps['total']}")
    if deps['missing']:
        print(f"  ❌ Missing: {', '.join(deps['missing'])}")
    else:
        print("  ✅ All dependencies available")
    print()

    # Models
    models = results["models"]
    print("🤖 Models:")
    if models["status"] == "ok":
        total_models = sum(len(model_list) for model_list in models["models"].values())
        print(f"  ✅ {total_models} models available")
        for model_type, model_names in models["models"].items():
            if model_names:
                print(f"     {model_type}: {len(model_names)} models")
    else:
        print(f"  ❌ Model loading error: {models['error']}")
    print()

    # Agents
    agents = results["agents"]
    print("🎯 AI Agents:")
    for agent_name, status in agents.items():
        if agent_name == "seismic_stats":
            continue
        if isinstance(status, bool):
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {agent_name.replace('_', ' ').title()}")
        elif isinstance(status, str) and status.startswith("Error:"):
            print(f"  ❌ {agent_name.replace('_', ' ').title()}: {status}")
        else:
            print(f"  ⚠️  {agent_name.replace('_', ' ').title()}: {status}")

    # Seismic stats
    if "seismic_stats" in agents:
        stats = agents["seismic_stats"]
        print("\n📊 Seismic Detector Stats:")
        print(f"     Zones: {stats.get('monitoring_zones', 0)}")
        print(f"     Total scans: {stats.get('total_scans', 0)}")
        print(f"     Alerts: {stats.get('alerts_generated', 0)}")

    print()
    print("=" * 45)

    # Overall assessment
    issues = []

    if server["status"] != "running":
        issues.append("Server not running")

    if deps["missing"]:
        issues.append(f"{len(deps['missing'])} missing dependencies")

    if models["status"] != "ok":
        issues.append("Model loading issues")

    failed_agents = sum(1 for status in agents.values()
                       if isinstance(status, str) and status.startswith("Error:"))
    if failed_agents > 0:
        issues.append(f"{failed_agents} failed agents")

    if not issues:
        print("🎉 Backend Status: HEALTHY - All systems operational")
    else:
        print(f"⚠️  Backend Status: ISSUES DETECTED")
        for issue in issues:
            print(f"     • {issue}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Backend Health Check")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode - only essential output")
    parser.add_argument("--timeout", type=int, default=5, help="Request timeout in seconds")

    args = parser.parse_args()

    # Collect all status information
    results = {
        "timestamp": datetime.now().isoformat(),
        "server": check_server_health(args.host, args.port, args.timeout),
        "dependencies": check_dependencies(),
        "models": check_models(),
        "agents": check_agents()
    }

    # Print report
    print_status_report(results, args.json, args.quiet)

if __name__ == "__main__":
    main()