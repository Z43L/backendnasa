#!/usr/bin/env python3
"""
Seismic AI Alert System

Real-time monitoring and alert system for seismic activity,
environmental conditions, and predictive analytics.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    SEISMIC = "seismic"
    WEATHER = "weather"
    AIR_QUALITY = "air_quality"
    SEA_LEVEL = "sea_level"
    INTEGRATED = "integrated"

@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    alert_type: AlertType
    level: AlertLevel
    title: str
    message: str
    coordinates: Optional[Dict[str, float]] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: str = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.max_history_size = 1000

    def create_alert(self, alert_type: AlertType, level: AlertLevel, title: str,
                    message: str, coordinates: Optional[Dict[str, float]] = None,
                    data: Optional[Dict[str, Any]] = None) -> Alert:
        """Create a new alert."""
        alert_id = f"alert_{int(time.time() * 1000)}"

        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            level=level,
            title=title,
            message=message,
            coordinates=coordinates,
            data=data
        )

        self.alerts[alert_id] = alert
        self.active_alerts[alert_id] = alert

        logger.warning(f"🚨 Alert created: {alert_type.value} - {level.value} - {title}")

        # Trigger notification (in real implementation)
        asyncio.create_task(self._notify_alert(alert))

        return alert

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert."""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.acknowledged = True
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now().isoformat()

        # Move to history
        del self.active_alerts[alert_id]
        self.alert_history.append(alert)

        # Keep history size manageable
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]

        logger.info(f"✅ Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return [asdict(alert) for alert in self.active_alerts.values()]

    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get alert history."""
        return [asdict(alert) for alert in self.alert_history[-limit:]]

    def get_alert_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get alert by ID."""
        alert = self.alerts.get(alert_id)
        return asdict(alert) if alert else None

    async def _notify_alert(self, alert: Alert):
        """Send alert notifications."""
        # In a real implementation, this would:
        # - Send emails
        # - Send SMS
        # - Trigger webhooks
        # - Update dashboards
        # - Send to monitoring systems

        logger.info(f"📤 Sending notifications for alert {alert.alert_id}")

        # Simulate notification delay
        await asyncio.sleep(0.1)

        # Here you would implement actual notification logic
        pass

class SeismicMonitor:
    """Monitors seismic activity and generates alerts."""

    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.monitoring_zones = []
        self.risk_thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.95
        }

    def add_monitoring_zone(self, name: str, lat: float, lon: float,
                           risk_level: str = "medium"):
        """Add a zone to monitor."""
        zone = {
            "name": name,
            "coordinates": {"lat": lat, "lon": lon},
            "risk_level": risk_level,
            "last_check": None,
            "alert_count": 0
        }
        self.monitoring_zones.append(zone)
        logger.info(f"📍 Added monitoring zone: {name} at {lat}, {lon}")

    def check_seismic_risk(self, lat: float, lon: float) -> Dict[str, Any]:
        """Check seismic risk for coordinates."""
        # Mock seismic risk assessment
        # In real implementation, this would use actual seismic data
        base_risk = 0.1  # Base seismic risk

        # Add some randomness based on location
        location_factor = abs(lat) / 90.0  # Higher risk near poles (for demo)
        risk_score = min(base_risk + location_factor * 0.4 + np.random.random() * 0.3, 1.0)

        # Determine risk level
        if risk_score >= self.risk_thresholds["critical"]:
            level = AlertLevel.CRITICAL
        elif risk_score >= self.risk_thresholds["high"]:
            level = AlertLevel.HIGH
        elif risk_score >= self.risk_thresholds["medium"]:
            level = AlertLevel.MEDIUM
        else:
            level = AlertLevel.LOW

        return {
            "risk_score": risk_score,
            "risk_level": level.value,
            "alert_threshold": self.risk_thresholds[level.value],
            "coordinates": {"lat": lat, "lon": lon}
        }

    def monitor_zones(self):
        """Monitor all zones for seismic activity."""
        for zone in self.monitoring_zones:
            risk_data = self.check_seismic_risk(
                zone["coordinates"]["lat"],
                zone["coordinates"]["lon"]
            )

            # Check if alert should be triggered
            if risk_data["risk_score"] >= self.risk_thresholds[risk_data["risk_level"]]:
                self.alert_manager.create_alert(
                    alert_type=AlertType.SEISMIC,
                    level=AlertLevel(risk_data["risk_level"]),
                    title=f"Seismic Risk Alert - {zone['name']}",
                    message=f"High seismic risk detected in {zone['name']}. Risk score: {risk_data['risk_score']:.2f}",
                    coordinates=zone["coordinates"],
                    data=risk_data
                )
                zone["alert_count"] += 1

            zone["last_check"] = datetime.now().isoformat()

class WeatherMonitor:
    """Monitors weather conditions and generates alerts."""

    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.weather_thresholds = {
            "heavy_rain": 10.0,  # mm/hour
            "high_wind": 30.0,   # km/h
            "extreme_temp": 35.0 # Celsius
        }

    def check_weather_conditions(self, lat: float, lon: float) -> Dict[str, Any]:
        """Check weather conditions for coordinates."""
        # Mock weather assessment
        conditions = {
            "precipitation": np.random.exponential(2.0),  # mm/hour
            "wind_speed": np.random.normal(15, 5),       # km/h
            "temperature": np.random.normal(25, 8),      # Celsius
            "coordinates": {"lat": lat, "lon": lon}
        }

        alerts = []

        # Check for heavy rain
        if conditions["precipitation"] >= self.weather_thresholds["heavy_rain"]:
            alerts.append({
                "type": "heavy_rain",
                "level": AlertLevel.HIGH,
                "title": "Heavy Rainfall Alert",
                "message": f"Heavy rainfall detected: {conditions['precipitation']:.1f} mm/hour"
            })

        # Check for high winds
        if conditions["wind_speed"] >= self.weather_thresholds["high_wind"]:
            alerts.append({
                "type": "high_wind",
                "level": AlertLevel.MEDIUM,
                "title": "High Wind Alert",
                "message": f"High winds detected: {conditions['wind_speed']:.1f} km/h"
            })

        # Check for extreme temperatures
        if abs(conditions["temperature"]) >= self.weather_thresholds["extreme_temp"]:
            alerts.append({
                "type": "extreme_temp",
                "level": AlertLevel.MEDIUM,
                "title": "Extreme Temperature Alert",
                "message": f"Extreme temperature: {conditions['temperature']:.1f}°C"
            })

        return {
            "conditions": conditions,
            "alerts": alerts
        }

    def monitor_weather(self, lat: float, lon: float):
        """Monitor weather for specific coordinates."""
        weather_data = self.check_weather_conditions(lat, lon)

        for alert_info in weather_data["alerts"]:
            self.alert_manager.create_alert(
                alert_type=AlertType.WEATHER,
                level=alert_info["level"],
                title=alert_info["title"],
                message=alert_info["message"],
                coordinates={"lat": lat, "lon": lon},
                data=weather_data["conditions"]
            )

class AirQualityMonitor:
    """Monitors air quality and generates alerts."""

    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.aqi_thresholds = {
            "unhealthy": 150,
            "very_unhealthy": 200,
            "hazardous": 300
        }

    def check_air_quality(self, lat: float, lon: float) -> Dict[str, Any]:
        """Check air quality for coordinates."""
        # Mock AQI assessment
        aqi = np.random.uniform(0, 400)
        conditions = {
            "aqi": aqi,
            "coordinates": {"lat": lat, "lon": lon}
        }

        # Determine alert level
        if aqi >= self.aqi_thresholds["hazardous"]:
            level = AlertLevel.CRITICAL
            title = "Hazardous Air Quality Alert"
            message = f"Dangerous air quality levels: AQI {aqi:.0f}"
        elif aqi >= self.aqi_thresholds["very_unhealthy"]:
            level = AlertLevel.HIGH
            title = "Very Unhealthy Air Quality Alert"
            message = f"Very unhealthy air quality: AQI {aqi:.0f}"
        elif aqi >= self.aqi_thresholds["unhealthy"]:
            level = AlertLevel.MEDIUM
            title = "Unhealthy Air Quality Alert"
            message = f"Unhealthy air quality: AQI {aqi:.0f}"
        else:
            return {"conditions": conditions, "alert": None}

        return {
            "conditions": conditions,
            "alert": {
                "level": level,
                "title": title,
                "message": message
            }
        }

    def monitor_air_quality(self, lat: float, lon: float):
        """Monitor air quality for coordinates."""
        aq_data = self.check_air_quality(lat, lon)

        if aq_data["alert"]:
            self.alert_manager.create_alert(
                alert_type=AlertType.AIR_QUALITY,
                level=aq_data["alert"]["level"],
                title=aq_data["alert"]["title"],
                message=aq_data["alert"]["message"],
                coordinates={"lat": lat, "lon": lon},
                data=aq_data["conditions"]
            )

# Global instances
alert_manager = AlertManager()
seismic_monitor = SeismicMonitor(alert_manager)
weather_monitor = WeatherMonitor(alert_manager)
air_quality_monitor = AirQualityMonitor(alert_manager)

# Initialize some monitoring zones
def initialize_monitoring():
    """Initialize monitoring system with default zones."""
    # Add some example monitoring zones
    seismic_monitor.add_monitoring_zone("California Fault Zone", 35.0, -118.0, "high")
    seismic_monitor.add_monitoring_zone("Mediterranean Coast", 40.0, 4.0, "medium")
    seismic_monitor.add_monitoring_zone("Pacific Ring", -20.0, -175.0, "high")

    logger.info("📊 Monitoring system initialized with sample zones")

if __name__ == "__main__":
    # Initialize monitoring
    initialize_monitoring()

    # Example usage
    print("🔔 Seismic AI Alert System Demo")
    print("===============================")

    # Test seismic monitoring
    print("\n🔍 Testing Seismic Monitoring...")
    risk = seismic_monitor.check_seismic_risk(35.0, -118.0)
    print(f"Seismic risk at California: {risk}")

    # Test weather monitoring
    print("\n🌦️ Testing Weather Monitoring...")
    weather_monitor.monitor_weather(35.0, -118.0)

    # Test air quality monitoring
    print("\n💨 Testing Air Quality Monitoring...")
    air_quality_monitor.monitor_air_quality(35.0, -118.0)

    # Show active alerts
    print("\n🚨 Active Alerts:")
    alerts = alert_manager.get_active_alerts()
    for alert in alerts:
        print(f"  - {alert['alert_type'].upper()}: {alert['title']}")

    print(f"\n📈 Total active alerts: {len(alerts)}")