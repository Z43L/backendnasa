#!/usr/bin/env python3
"""
Seismic AI System Health Check

Verifies that all components of the Seismic AI system are working correctly.
"""

import requests
import json
import sys
import time
from typing import Dict, List, Any

class SystemHealthChecker:
    """Comprehensive health checker for Seismic AI system."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.results = []

    def log_result(self, test_name: str, success: bool, message: str, details: Dict = None):
        """Log a test result."""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "details": details or {}
        }
        self.results.append(result)
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")

    def make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Make HTTP request with error handling."""
        try:
            url = f"{self.base_url}{endpoint}"
            if method.upper() == "GET":
                response = self.session.get(url, timeout=10)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=10)
            else:
                return {"error": f"Unsupported method: {method}"}

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response"}

    def check_server_connectivity(self) -> bool:
        """Check if server is reachable."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                self.log_result("Server Connectivity", True, "Server is reachable")
                return True
            else:
                self.log_result("Server Connectivity", False, f"Server returned status {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Server Connectivity", False, f"Cannot connect to server: {e}")
            return False

    def check_health_endpoint(self) -> bool:
        """Check system health endpoint."""
        health = self.make_request("GET", "/health")

        if "error" in health:
            self.log_result("Health Endpoint", False, f"Health check failed: {health['error']}")
            return False

        status = health.get("status")
        if status == "healthy":
            models_loaded = len(health.get("models", {}))
            active_alerts = health.get("system", {}).get("monitoring", {}).get("active_alerts", 0)

            self.log_result("Health Endpoint", True, "System is healthy",
                          {"models_loaded": models_loaded, "active_alerts": active_alerts})
            return True
        else:
            self.log_result("Health Endpoint", False, f"System status: {status}")
            return False

    def check_weather_prediction(self) -> bool:
        """Test weather prediction endpoint."""
        test_data = {
            "coordinates": {"lat": 35.0, "lon": -118.0},
            "hours_ahead": 24
        }

        result = self.make_request("POST", "/predict/weather", test_data)

        if "error" in result:
            self.log_result("Weather Prediction", False, f"Weather prediction failed: {result['error']}")
            return False

        if "prediction" in result:
            precipitation = result["prediction"].get("precipitation", 0)
            self.log_result("Weather Prediction", True, f"Weather prediction successful: {precipitation:.2f} mm/h")
            return True
        else:
            self.log_result("Weather Prediction", False, "Invalid weather prediction response")
            return False

    def check_air_quality_prediction(self) -> bool:
        """Test air quality prediction endpoint."""
        test_data = {"lat": 35.0, "lon": -118.0}

        result = self.make_request("POST", "/predict/air-quality", test_data)

        if "error" in result:
            self.log_result("Air Quality Prediction", False, f"Air quality prediction failed: {result['error']}")
            return False

        if "air_quality" in result:
            aqi = result["air_quality"].get("aqi", "N/A")
            level = result["air_quality"].get("level", "Unknown")
            self.log_result("Air Quality Prediction", True, f"Air quality: AQI {aqi} ({level})")
            return True
        else:
            self.log_result("Air Quality Prediction", False, "Invalid air quality response")
            return False

    def check_seismic_risk(self) -> bool:
        """Test seismic risk assessment endpoint."""
        test_data = {"lat": 35.0, "lon": -118.0}

        result = self.make_request("POST", "/monitor/check/seismic", test_data)

        if "error" in result:
            self.log_result("Seismic Risk Assessment", False, f"Seismic risk check failed: {result['error']}")
            return False

        if "risk_assessment" in result:
            risk_level = result["risk_assessment"].get("risk_level", "Unknown")
            risk_score = result["risk_assessment"].get("risk_score", 0)
            self.log_result("Seismic Risk Assessment", True,
                          f"Seismic risk: {risk_level.upper()} ({risk_score:.3f})")
            return True
        else:
            self.log_result("Seismic Risk Assessment", False, "Invalid seismic risk response")
            return False

    def check_integrated_analysis(self) -> bool:
        """Test integrated analysis endpoint."""
        test_data = {
            "coordinates": {"lat": 35.0, "lon": -118.0},
            "analysis_type": "full"
        }

        result = self.make_request("POST", "/analyze/integrated", test_data)

        if "error" in result:
            self.log_result("Integrated Analysis", False, f"Integrated analysis failed: {result['error']}")
            return False

        if "components" in result:
            components = result["components"]
            available_components = [k for k, v in components.items() if isinstance(v, dict) and "error" not in v]
            self.log_result("Integrated Analysis", True,
                          f"Integrated analysis successful: {len(available_components)} components available",
                          {"available_components": available_components})
            return True
        else:
            self.log_result("Integrated Analysis", False, "Invalid integrated analysis response")
            return False

    def check_alert_system(self) -> bool:
        """Test alert creation and retrieval."""
        # Create test alert
        alert_data = {
            "type": "test",
            "level": "low",
            "title": "System Health Check Test",
            "message": "This is a test alert from the health checker",
            "coordinates": {"lat": 35.0, "lon": -118.0},
            "data": {"test": True}
        }

        create_result = self.make_request("POST", "/monitor/alert", alert_data)

        if "error" in create_result:
            self.log_result("Alert System", False, f"Alert creation failed: {create_result['error']}")
            return False

        alert_id = create_result.get("alert_id")
        if not alert_id:
            self.log_result("Alert System", False, "Alert creation did not return alert_id")
            return False

        # Check active alerts
        alerts_result = self.make_request("GET", "/monitor/alerts/active")

        if "error" in alerts_result:
            self.log_result("Alert System", False, f"Active alerts check failed: {alerts_result['error']}")
            return False

        active_alerts = alerts_result.get("alerts", [])
        alert_found = any(alert.get("id") == alert_id for alert in active_alerts)

        if alert_found:
            self.log_result("Alert System", True, f"Alert system working: {len(active_alerts)} active alerts")
            return True
        else:
            self.log_result("Alert System", False, "Created alert not found in active alerts")
            return False

    def check_api_documentation(self) -> bool:
        """Check if API documentation is accessible."""
        try:
            docs_response = requests.get(f"{self.base_url}/docs", timeout=5)
            redoc_response = requests.get(f"{self.base_url}/redoc", timeout=5)
            openapi_response = requests.get(f"{self.base_url}/openapi.json", timeout=5)

            docs_ok = docs_response.status_code == 200
            redoc_ok = redoc_response.status_code == 200
            openapi_ok = openapi_response.status_code == 200

            if docs_ok and redoc_ok and openapi_ok:
                self.log_result("API Documentation", True, "All documentation endpoints accessible")
                return True
            else:
                issues = []
                if not docs_ok: issues.append("docs")
                if not redoc_ok: issues.append("redoc")
                if not openapi_ok: issues.append("openapi.json")

                self.log_result("API Documentation", False, f"Documentation issues: {', '.join(issues)}")
                return False

        except Exception as e:
            self.log_result("API Documentation", False, f"Documentation check failed: {e}")
            return False

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        print("🔍 Seismic AI System Health Check")
        print("=" * 40)

        checks = [
            ("Server Connectivity", self.check_server_connectivity),
            ("Health Endpoint", self.check_health_endpoint),
            ("Weather Prediction", self.check_weather_prediction),
            ("Air Quality Prediction", self.check_air_quality_prediction),
            ("Seismic Risk Assessment", self.check_seismic_risk),
            ("Integrated Analysis", self.check_integrated_analysis),
            ("Alert System", self.check_alert_system),
            ("API Documentation", self.check_api_documentation),
        ]

        passed = 0
        total = len(checks)

        for check_name, check_func in checks:
            print(f"\n🧪 Testing {check_name}...")
            if check_func():
                passed += 1

        print("\n" + "=" * 40)
        print("📊 Health Check Summary")
        print("=" * 40)

        success_rate = (passed / total) * 100

        if success_rate == 100:
            print(f"🎉 ALL CHECKS PASSED ({passed}/{total}) - System is fully operational!")
            overall_status = "excellent"
        elif success_rate >= 75:
            print(f"✅ MOSTLY HEALTHY ({passed}/{total}) - Minor issues detected")
            overall_status = "good"
        elif success_rate >= 50:
            print(f"⚠️  PARTIALLY WORKING ({passed}/{total}) - Significant issues")
            overall_status = "warning"
        else:
            print(f"❌ SYSTEM ISSUES ({passed}/{total}) - Critical problems detected")
            overall_status = "critical"

        return {
            "overall_status": overall_status,
            "passed": passed,
            "total": total,
            "success_rate": success_rate,
            "results": self.results,
            "timestamp": time.time()
        }

def main():
    """Main function."""
    checker = SystemHealthChecker()

    try:
        summary = checker.run_all_checks()

        # Save detailed results
        with open("health_check_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: health_check_results.json")

        # Exit with appropriate code
        if summary["overall_status"] in ["excellent", "good"]:
            print("\n🚀 System is ready for use!")
            sys.exit(0)
        else:
            print("\n🔧 System needs attention. Check the results above.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Health check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Health check failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()