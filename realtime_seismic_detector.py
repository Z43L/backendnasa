#!/usr/bin/env python3
"""
Real-Time Seismic Detection Agent

Agente inteligente para detectar seísmos en tiempo real usando modelos de IA.
Genera alertas JSON con coordenadas, posibilidad de seísmo (0.0-10.0) e intensidad.
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from inference_service import SeismicInferenceService
from alert_system import AlertManager, AlertType, AlertLevel

logger = logging.getLogger(__name__)

@dataclass
class SeismicAlert:
    """Estructura de datos para alertas sísmicas."""
    alert_id: str
    timestamp: str
    coordinates: Dict[str, float]  # {"lat": float, "lon": float}
    seismic_probability: float  # 0.0 - 10.0
    intensity: float  # Magnitud/intensidad del seísmo
    risk_level: str  # "low", "medium", "high", "critical"
    confidence: float  # 0.0 - 1.0
    data: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        """Convertir a JSON string."""
        return json.dumps(asdict(self), indent=2, default=str)

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return asdict(self)

class RealTimeSeismicDetector:
    """
    Agente para detección de seísmos en tiempo real.

    Monitorea datos sísmicos continuamente y genera alertas cuando detecta
    actividad sísmica con coordenadas GPS específicas.
    """

    def __init__(self, alert_manager: Optional[AlertManager] = None):
        """
        Inicializar el detector de seísmos.

        Args:
            alert_manager: Gestor de alertas opcional para integración
        """
        self.inference_service = SeismicInferenceService()
        self.alert_manager = alert_manager or AlertManager()

        # Configuración de monitoreo
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_zones: List[Dict[str, Any]] = []

        # Umbrales de detección
        self.probability_threshold = 3.0  # Umbral mínimo para alertas (0-10)
        self.intensity_threshold = 2.0    # Umbral mínimo de intensidad
        self.monitoring_interval = 5.0    # Segundos entre verificaciones

        # Historial de detecciones
        self.detection_history: List[SeismicAlert] = []
        self.max_history_size = 1000

        # Estadísticas
        self.stats = {
            "total_scans": 0,
            "alerts_generated": 0,
            "false_positives": 0,
            "last_scan_time": None
        }

        logger.info("🔔 Real-Time Seismic Detector initialized")

    def add_monitoring_zone(self, name: str, lat: float, lon: float,
                           radius_km: float = 50.0, priority: str = "medium"):
        """
        Agregar zona de monitoreo.

        Args:
            name: Nombre de la zona
            lat: Latitud
            lon: Longitud
            radius_km: Radio de monitoreo en kilómetros
            priority: Prioridad ("low", "medium", "high", "critical")
        """
        zone = {
            "name": name,
            "coordinates": {"lat": lat, "lon": lon},
            "radius_km": radius_km,
            "priority": priority,
            "last_scan": None,
            "alert_count": 0,
            "active": True
        }

        self.monitoring_zones.append(zone)
        logger.info(f"📍 Added monitoring zone: {name} at {lat:.4f}, {lon:.4f} (radius: {radius_km}km)")

    def remove_monitoring_zone(self, name: str) -> bool:
        """
        Remover zona de monitoreo.

        Args:
            name: Nombre de la zona a remover

        Returns:
            True si se removió exitosamente
        """
        for i, zone in enumerate(self.monitoring_zones):
            if zone["name"] == name:
                del self.monitoring_zones[i]
                logger.info(f"🗑️ Removed monitoring zone: {name}")
                return True
        return False

    def start_monitoring(self):
        """Iniciar monitoreo en tiempo real."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("🚀 Real-time seismic monitoring started")

    def stop_monitoring(self):
        """Detener monitoreo en tiempo real."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        logger.info("⏹️ Real-time seismic monitoring stopped")

    def _monitoring_loop(self):
        """Loop principal de monitoreo."""
        logger.info("🔄 Starting monitoring loop...")

        while self.monitoring_active:
            try:
                self._scan_all_zones()
                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _scan_all_zones(self):
        """Escanear todas las zonas de monitoreo."""
        self.stats["total_scans"] += 1
        self.stats["last_scan_time"] = datetime.now().isoformat()

        for zone in self.monitoring_zones:
            if not zone["active"]:
                continue

            try:
                alert = self.scan_zone(zone)
                if alert:
                    self._process_alert(alert, zone)

            except Exception as e:
                logger.error(f"Error scanning zone {zone['name']}: {e}")

            zone["last_scan"] = datetime.now().isoformat()

    def scan_zone(self, zone: Dict[str, Any]) -> Optional[SeismicAlert]:
        """
        Escanear una zona específica para actividad sísmica.

        Args:
            zone: Información de la zona a escanear

        Returns:
            SeismicAlert si se detecta actividad, None en caso contrario
        """
        lat, lon = zone["coordinates"]["lat"], zone["coordinates"]["lon"]

        # Generar datos sísmicos simulados para la zona
        # En implementación real, aquí se obtendrían datos de sensores GPS, acelerómetros, etc.
        seismic_data = self._generate_seismic_data(lat, lon)

        # Usar el modelo de clasificación sísmica
        try:
            result = self.inference_service.predict_seismic_state(seismic_data)

            if not result:
                return None

            # Extraer probabilidades
            probabilities = np.array(result["probabilities"])  # [precursor, normal, post-earthquake]

            # Calcular posibilidad de seísmo (0.0-10.0)
            # Usamos la probabilidad del estado post-earthquake como indicador
            post_earthquake_prob = probabilities[0][2] if len(probabilities.shape) > 1 else probabilities[2]
            seismic_probability = post_earthquake_prob * 10.0  # Escalar a 0-10

            # Calcular intensidad basada en la magnitud de la deformación
            intensity = self._calculate_intensity(seismic_data)

            # Determinar nivel de riesgo
            risk_level = self._calculate_risk_level(seismic_probability, intensity)

            # Verificar umbrales
            if seismic_probability >= self.probability_threshold and intensity >= self.intensity_threshold:
                # Crear alerta
                alert_id = f"seismic_{int(time.time() * 1000)}_{zone['name']}"

                alert = SeismicAlert(
                    alert_id=alert_id,
                    timestamp=datetime.now().isoformat(),
                    coordinates={"lat": lat, "lon": lon},
                    seismic_probability=round(seismic_probability, 2),
                    intensity=round(intensity, 2),
                    risk_level=risk_level,
                    confidence=post_earthquake_prob,
                    data={
                        "zone_name": zone["name"],
                        "raw_probabilities": probabilities.tolist(),
                        "seismic_data_shape": seismic_data.shape,
                        "model_used": "seismic_classification"
                    }
                )

                return alert

        except Exception as e:
            logger.error(f"Error in seismic prediction for zone {zone['name']}: {e}")

        return None

    def _generate_seismic_data(self, lat: float, lon: float) -> np.ndarray:
        """
        Generar datos sísmicos simulados para una ubicación.

        En implementación real, esto obtendría datos de:
        - Redes GPS
        - Acelerómetros
        - Datos satelitales (InSAR)
        - Estaciones sísmicas

        Args:
            lat: Latitud
            lon: Longitud

        Returns:
            Array de datos sísmicos [batch, channels, height, width]
        """
        # Simular datos basados en ubicación
        # En zonas de alta actividad sísmica, mayor probabilidad de detectar seísmos

        # Factores de riesgo por ubicación (simplificado)
        location_risk = abs(lat) / 90.0  # Más riesgo cerca de los polos (ejemplo)

        # Agregar variabilidad temporal
        time_factor = np.sin(time.time() * 0.001) * 0.3 + 0.7

        # Generar datos de deformación simulados
        base_deformation = location_risk * time_factor

        # Crear secuencia temporal de 30 frames, 50x50 grid
        sequence = []
        for t in range(30):
            # Deformación con componente temporal
            temporal_variation = np.sin(t * 0.2) * 0.1
            frame_deformation = base_deformation + temporal_variation

            # Crear patrón de deformación espacial
            x, y = np.meshgrid(np.arange(50), np.arange(50))
            # Centro de deformación ligeramente desplazado
            center_x, center_y = 25 + np.random.normal(0, 5), 25 + np.random.normal(0, 5)

            # Patrón de deformación gaussiano
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            deformation_pattern = frame_deformation * np.exp(-distance**2 / (2 * 10**2))

            # Agregar ruido
            noise = np.random.normal(0, 0.01, (50, 50))
            frame = deformation_pattern + noise

            sequence.append(frame)

        # Convertir a formato esperado por el modelo [batch, seq_len, height, width]
        seismic_sequence = np.array(sequence)  # [30, 50, 50]
        seismic_sequence = seismic_sequence[np.newaxis, ...]  # [1, 30, 50, 50]

        return seismic_sequence

    def _calculate_intensity(self, seismic_data: np.ndarray) -> float:
        """
        Calcular intensidad del seísmo basada en los datos.

        Args:
            seismic_data: Datos sísmicos

        Returns:
            Intensidad (magnitud aproximada)
        """
        # Calcular la magnitud máxima de deformación
        max_deformation = np.abs(seismic_data).max()

        # Convertir deformación a magnitud sísmica aproximada
        # Fórmula simplificada: magnitud ≈ log10(deformación * factor)
        if max_deformation > 0:
            intensity = 2.0 + np.log10(max_deformation * 1000)  # Factor de escala arbitrario
            intensity = np.clip(intensity, 0.0, 10.0)  # Limitar rango
        else:
            intensity = 0.0

        return intensity

    def _calculate_risk_level(self, probability: float, intensity: float) -> str:
        """
        Calcular nivel de riesgo basado en probabilidad e intensidad.

        Args:
            probability: Probabilidad de seísmo (0-10)
            intensity: Intensidad del seísmo

        Returns:
            Nivel de riesgo ("low", "medium", "high", "critical")
        """
        risk_score = (probability / 10.0) * (intensity / 10.0)

        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.3:
            return "medium"
        else:
            return "low"

    def _process_alert(self, alert: SeismicAlert, zone: Dict[str, Any]):
        """
        Procesar una alerta detectada.

        Args:
            alert: Alerta sísmica detectada
            zone: Información de la zona
        """
        # Agregar al historial
        self.detection_history.append(alert)
        if len(self.detection_history) > self.max_history_size:
            self.detection_history = self.detection_history[-self.max_history_size:]

        # Actualizar estadísticas
        self.stats["alerts_generated"] += 1
        zone["alert_count"] += 1

        # Crear alerta en el sistema de alertas
        alert_level = AlertLevel(alert.risk_level.upper())

        self.alert_manager.create_alert(
            alert_type=AlertType.SEISMIC,
            level=alert_level,
            title=f"🚨 Seismic Alert - {zone['name']}",
            message=f"Seismic activity detected. Probability: {alert.seismic_probability:.1f}/10, "
                   f"Intensity: {alert.intensity:.1f}",
            coordinates=alert.coordinates,
            data=alert.to_dict()
        )

        # Log de la alerta
        logger.warning(f"🚨 SEISMIC ALERT: {alert.to_json()}")

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Obtener alertas activas en formato JSON.

        Returns:
            Lista de alertas activas
        """
        active_alerts = self.alert_manager.get_active_alerts()
        return [alert for alert in active_alerts if alert.get("alert_type") == "seismic"]

    def get_detection_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtener historial de detecciones.

        Args:
            limit: Número máximo de detecciones a retornar

        Returns:
            Lista de detecciones recientes
        """
        recent_detections = self.detection_history[-limit:]
        return [alert.to_dict() for alert in recent_detections]

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del detector.

        Returns:
            Diccionario con estadísticas
        """
        return {
            **self.stats,
            "monitoring_zones": len(self.monitoring_zones),
            "active_zones": len([z for z in self.monitoring_zones if z["active"]]),
            "total_detections": len(self.detection_history)
        }

    def manual_scan(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Escaneo manual de una ubicación específica.

        Args:
            lat: Latitud
            lon: Longitud

        Returns:
            Resultado del escaneo en formato JSON, o None si no hay detección
        """
        # Crear zona temporal
        temp_zone = {
            "name": f"manual_scan_{lat:.4f}_{lon:.4f}",
            "coordinates": {"lat": lat, "lon": lon},
            "radius_km": 10.0,
            "priority": "high"
        }

        # Escanear
        alert = self.scan_zone(temp_zone)

        if alert:
            return alert.to_dict()

        return None

# Funciones de utilidad para uso standalone
def create_seismic_detector() -> RealTimeSeismicDetector:
    """Crear instancia del detector de seísmos."""
    return RealTimeSeismicDetector()

def quick_seismic_scan(lat: float, lon: float) -> Dict[str, Any]:
    """
    Escaneo rápido de una ubicación.

    Args:
        lat: Latitud
        lon: Longitud

    Returns:
        Resultado del escaneo
    """
    detector = create_seismic_detector()
    result = detector.manual_scan(lat, lon)

    if result:
        return result
    else:
        return {
            "status": "no_seismic_activity",
            "coordinates": {"lat": lat, "lon": lon},
            "message": "No seismic activity detected at this location",
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Demo del detector
    print("🔔 Real-Time Seismic Detection Agent Demo")
    print("=" * 50)

    # Crear detector
    detector = create_seismic_detector()

    # Agregar zonas de monitoreo
    detector.add_monitoring_zone("Zona_A", 19.4326, -99.1332, radius_km=100.0)  # Ciudad de México
    detector.add_monitoring_zone("Zona_B", 35.6895, 139.6917, radius_km=50.0)  # Tokio
    detector.add_monitoring_zone("Zona_C", 37.7749, -122.4194, radius_km=75.0)  # San Francisco

    # Escaneo manual de ejemplo
    print("\n🔍 Manual scan example:")
    result = detector.manual_scan(19.4326, -99.1332)  # Ciudad de México
    if result:
        print("🚨 Seismic activity detected!")
        print(json.dumps(result, indent=2))
    else:
        print("✅ No seismic activity detected")

    print(f"\n📊 Detector stats: {detector.get_stats()}")