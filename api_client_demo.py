#!/usr/bin/env python3
"""
Seismic AI API Client Example

Complete example showing how to interact with the Seismic AI Complete API Server.
"""

import requests
import json
import time
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np

class SeismicAIClient:
    """Client for interacting with the Seismic AI API Server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Make HTTP request to API."""
        url = f"{self.base_url}{endpoint}"

        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=30)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=30)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return {"error": str(e)}

    def get_health(self) -> Dict[str, Any]:
        """Get system health status."""
        return self._make_request("GET", "/health")

    def predict_seismic(self, seismic_data: np.ndarray) -> Dict[str, Any]:
        """Predict seismic activity."""
        data = {
            "data": seismic_data.tolist(),
            "shape": list(seismic_data.shape)
        }
        return self._make_request("POST", "/predict/seismic", data)

    def predict_weather(self, lat: float, lon: float, hours_ahead: int = 24) -> Dict[str, Any]:
        """Predict weather for coordinates."""
        data = {
            "coordinates": {"lat": lat, "lon": lon},
            "hours_ahead": hours_ahead
        }
        return self._make_request("POST", "/predict/weather", data)

    def predict_air_quality(self, lat: float, lon: float) -> Dict[str, Any]:
        """Predict air quality for coordinates."""
        data = {"lat": lat, "lon": lon}
        return self._make_request("POST", "/predict/air-quality", data)

    def analyze_integrated(self, lat: float, lon: float, analysis_type: str = "full") -> Dict[str, Any]:
        """Perform integrated analysis."""
        data = {
            "coordinates": {"lat": lat, "lon": lon},
            "analysis_type": analysis_type
        }
        return self._make_request("POST", "/analyze/integrated", data)

    def create_alert(self, alert_type: str, level: str, title: str, message: str,
                    lat: float = None, lon: float = None, alert_data: Dict = None) -> Dict[str, Any]:
        """Create a monitoring alert."""
        data = {
            "type": alert_type,
            "level": level,
            "title": title,
            "message": message
        }

        if lat is not None and lon is not None:
            data["coordinates"] = {"lat": lat, "lon": lon}

        if alert_data:
            data["data"] = alert_data

        return self._make_request("POST", "/monitor/alert", data)

    def get_active_alerts(self) -> Dict[str, Any]:
        """Get active alerts."""
        return self._make_request("GET", "/monitor/alerts/active")

    def check_seismic_risk(self, lat: float, lon: float) -> Dict[str, Any]:
        """Check seismic risk for coordinates."""
        data = {"lat": lat, "lon": lon}
        return self._make_request("POST", "/monitor/check/seismic", data)

def demo_basic_functionality():
    """Demonstrate basic API functionality."""
    print("🌟 Seismic AI API Client Demo")
    print("=" * 40)

    client = SeismicAIClient()

    # Check system health
    print("\n1. System Health Check")
    health = client.get_health()
    if "error" not in health:
        print(f"✅ System Status: {health.get('status', 'unknown')}")
        print(f"📊 Active Alerts: {health.get('system', {}).get('monitoring', {}).get('active_alerts', 0)}")
        print(f"🤖 Models Loaded: {len(health.get('models', {}))}")
    else:
        print(f"❌ Health check failed: {health['error']}")
        return

    # Test coordinates (California fault zone)
    test_lat, test_lon = 35.0, -118.0

    # Weather prediction
    print("\n2. Weather Prediction")
    weather = client.predict_weather(test_lat, test_lon, 24)
    if "error" not in weather:
        print(f"✅ Weather prediction successful")
        print(f"   Coordinates: {test_lat}, {test_lon}")
        print(f"   Hours ahead: {weather.get('prediction', {}).get('hours_ahead', 'N/A')}")
    else:
        print(f"❌ Weather prediction failed: {weather['error']}")

    # Air quality prediction
    print("\n3. Air Quality Prediction")
    air_quality = client.predict_air_quality(test_lat, test_lon)
    if "error" not in air_quality:
        aq_data = air_quality.get('air_quality', {})
        print(f"✅ Air Quality: AQI {aq_data.get('aqi', 'N/A')} - {aq_data.get('level', 'Unknown')}")
    else:
        print(f"❌ Air quality prediction failed: {air_quality['error']}")

    # Seismic risk check
    print("\n4. Seismic Risk Assessment")
    seismic_risk = client.check_seismic_risk(test_lat, test_lon)
    if "error" not in seismic_risk:
        risk_data = seismic_risk.get('risk_assessment', {})
        print(f"✅ Seismic Risk: {risk_data.get('risk_level', 'Unknown').upper()}")
        print(f"   Risk Score: {risk_data.get('risk_score', 0):.3f}")
    else:
        print(f"❌ Seismic risk check failed: {seismic_risk['error']}")

    # Integrated analysis
    print("\n5. Integrated Analysis")
    integrated = client.analyze_integrated(test_lat, test_lon, "full")
    if "error" not in integrated:
        print(f"✅ Integrated analysis completed")
        components = integrated.get('components', {})
        for component, data in components.items():
            if isinstance(data, dict) and 'error' not in data:
                print(f"   {component.upper()}: Available")
            else:
                print(f"   {component.upper()}: Not available")
    else:
        print(f"❌ Integrated analysis failed: {integrated['error']}")

    # Create a test alert
    print("\n6. Alert Creation")
    alert = client.create_alert(
        alert_type="seismic",
        level="medium",
        title="Test Seismic Alert",
        message="This is a test alert generated by the API client demo",
        lat=test_lat,
        lon=test_lon,
        alert_data={"test": True, "magnitude": 3.2}
    )
    if "error" not in alert:
        print(f"✅ Alert created: {alert.get('alert_id', 'Unknown ID')}")
    else:
        print(f"❌ Alert creation failed: {alert['error']}")

    # Check active alerts
    print("\n7. Active Alerts Check")
    alerts = client.get_active_alerts()
    if "error" not in alerts:
        active_alerts = alerts.get('alerts', [])
        print(f"📊 Active alerts: {len(active_alerts)}")
        for alert in active_alerts[:3]:  # Show first 3
            print(f"   - {alert.get('title', 'Unknown')}: {alert.get('level', 'unknown').upper()}")
    else:
        print(f"❌ Active alerts check failed: {alerts['error']}")

def demo_seismic_prediction():
    """Demonstrate seismic prediction with mock data."""
    print("\n🧪 Seismic Prediction Demo")
    print("-" * 30)

    client = SeismicAIClient()

    # Create mock seismic data (1 sample, 30 timesteps, 50x50 grid)
    seismic_data = np.random.randn(1, 30, 50, 50).astype(np.float32) * 0.1

    print(f"📊 Mock seismic data shape: {seismic_data.shape}")

    prediction = client.predict_seismic(seismic_data)

    if "error" not in prediction:
        print("✅ Seismic prediction successful!")
        predictions = prediction.get('predictions', {})
        if predictions:
            print(f"   Predictions shape: {predictions.get('shape', 'Unknown')}")
            print(f"   Classes: {predictions.get('class_names', [])}")
        else:
            print("   No predictions returned")
    else:
        print(f"❌ Seismic prediction failed: {prediction['error']}")

def demo_monitoring_scenario():
    """Demonstrate a complete monitoring scenario."""
    print("\n📡 Complete Monitoring Scenario")
    print("-" * 35)

    client = SeismicAIClient()

    # Define monitoring locations
    locations = [
        {"name": "California Coast", "lat": 36.7783, "lon": -119.4179},
        {"name": "Mediterranean", "lat": 40.0, "lon": 4.0},
        {"name": "Pacific Ring", "lat": -20.0, "lon": -175.0}
    ]

    print("🔍 Monitoring locations:")
    for loc in locations:
        print(f"   📍 {loc['name']}: {loc['lat']}, {loc['lon']}")

    # Perform integrated analysis for each location
    for loc in locations:
        print(f"\n🏠 Analyzing {loc['name']}...")

        # Integrated analysis
        analysis = client.analyze_integrated(loc['lat'], loc['lon'], 'full')

        if "error" not in analysis:
            components = analysis.get('components', {})

            # Check for potential issues
            issues = []

            if 'seismic' in components and 'risk_score' in components['seismic']:
                risk_score = components['seismic']['risk_score']
                if risk_score > 0.7:
                    issues.append(f"High seismic risk ({risk_score:.2f})")

            if 'weather' in components and 'precipitation' in components['weather']:
                precip = components['weather']['precipitation']
                if precip > 5.0:
                    issues.append(f"Heavy precipitation ({precip:.1f} mm/h)")

            if issues:
                # Create alert for issues
                alert = client.create_alert(
                    alert_type="integrated",
                    level="high",
                    title=f"Environmental Alert - {loc['name']}",
                    message=f"Critical conditions detected: {'; '.join(issues)}",
                    lat=loc['lat'],
                    lon=loc['lon'],
                    alert_data={"issues": issues, "location": loc['name']}
                )

                if "error" not in alert:
                    print(f"🚨 Alert created for {loc['name']}")
                else:
                    print(f"❌ Failed to create alert for {loc['name']}")
            else:
                print(f"✅ {loc['name']} conditions normal")
        else:
            print(f"❌ Analysis failed for {loc['name']}: {analysis['error']}")

    # Final status
    alerts = client.get_active_alerts()
    active_count = len(alerts.get('alerts', [])) if "error" not in alerts else 0

    print(f"\n📊 Monitoring complete. Active alerts: {active_count}")

def main():
    """Main demo function."""
    try:
        # Basic functionality demo
        demo_basic_functionality()

        # Seismic prediction demo
        demo_seismic_prediction()

        # Complete monitoring scenario
        demo_monitoring_scenario()

        print("\n🎉 Demo completed successfully!")
        print("📚 Check the API documentation at http://127.0.0.1:8000/docs")

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()