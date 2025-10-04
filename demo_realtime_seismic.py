#!/usr/bin/env python3
"""
Demo completo del Agente de Detección de Seísmos en Tiempo Real

Muestra todas las funcionalidades del sistema de detección sísmica.
"""

import asyncio
import json
import time
import threading
from datetime import datetime

from realtime_seismic_detector import RealTimeSeismicDetector, quick_seismic_scan

def demo_manual_scans():
    """Demo de escaneos manuales en diferentes ubicaciones."""
    print("🔍 Demo: Escaneos Manuales")
    print("=" * 40)

    # Ubicaciones de prueba (ciudades con diferente riesgo sísmico)
    test_locations = [
        {"name": "Ciudad de México", "lat": 19.4326, "lon": -99.1332},
        {"name": "Tokio", "lat": 35.6895, "lon": 139.6917},
        {"name": "San Francisco", "lat": 37.7749, "lon": -122.4194},
        {"name": "Nueva York", "lat": 40.7128, "lon": -74.0060},
        {"name": "Santiago de Chile", "lat": -33.4489, "lon": -70.6693},
    ]

    for location in test_locations:
        print(f"\n📍 Escaneando: {location['name']}")
        result = quick_seismic_scan(location['lat'], location['lon'])

        if result.get("status") == "no_seismic_activity":
            print("✅ Sin actividad sísmica detectada")
        else:
            print("🚨 ¡Actividad sísmica detectada!")
            print(f"   Probabilidad: {result['seismic_probability']}/10.0")
            print(f"   Intensidad: {result['intensity']:.2f}")
            print(f"   Nivel de riesgo: {result['risk_level']}")

        time.sleep(0.5)  # Pequeña pausa entre escaneos

def demo_realtime_monitoring():
    """Demo de monitoreo en tiempo real."""
    print("\n🔄 Demo: Monitoreo en Tiempo Real")
    print("=" * 40)

    # Crear detector
    detector = RealTimeSeismicDetector()

    # Agregar zonas de monitoreo
    zones = [
        {"name": "Pacifico_Ring", "lat": 35.6895, "lon": 139.6917, "radius": 200.0},  # Tokio
        {"name": "Andes_Region", "lat": -33.4489, "lon": -70.6693, "radius": 150.0},  # Santiago
        {"name": "California_Fault", "lat": 37.7749, "lon": -122.4194, "radius": 100.0},  # San Francisco
    ]

    for zone in zones:
        detector.add_monitoring_zone(
            zone["name"],
            zone["lat"],
            zone["lon"],
            radius_km=zone["radius"]
        )

    print(f"📊 Zonas de monitoreo configuradas: {len(zones)}")

    # Iniciar monitoreo
    detector.start_monitoring()
    print("🚀 Monitoreo en tiempo real iniciado...")

    # Monitorear por 30 segundos
    start_time = time.time()
    scan_count = 0

    try:
        while time.time() - start_time < 30:  # 30 segundos de demo
            time.sleep(2)  # Actualización cada 2 segundos

            # Obtener estadísticas
            stats = detector.get_stats()
            alerts = detector.get_active_alerts()

            if stats["total_scans"] > scan_count:
                scan_count = stats["total_scans"]
                print(f"🔄 Escaneo #{scan_count} completado - Alertas activas: {len(alerts)}")

                # Mostrar alertas nuevas
                if alerts:
                    for alert in alerts[-3:]:  # Mostrar últimas 3 alertas
                        data = alert.get("data", {})
                        if isinstance(data, str):
                            try:
                                data = json.loads(data)
                            except:
                                data = {}

                        print(f"  🚨 {alert['title']}")
                        print(f"     Ubicación: {alert['coordinates']}")
                        print(f"     Probabilidad: {data.get('seismic_probability', 'N/A')}/10.0")
                        print(f"     Intensidad: {data.get('intensity', 'N/A')}")

    except KeyboardInterrupt:
        print("\n⏹️ Demo interrumpido por usuario")

    finally:
        # Detener monitoreo
        detector.stop_monitoring()
        print("⏹️ Monitoreo detenido")

        # Mostrar resumen final
        final_stats = detector.get_stats()
        history = detector.get_detection_history(limit=10)

        print("\n📊 Resumen Final:")        print(f"   Total de escaneos: {final_stats['total_scans']}")
        print(f"   Alertas generadas: {final_stats['alerts_generated']}")
        print(f"   Detecciones totales: {final_stats['total_detections']}")
        print(f"   Historial de detecciones: {len(history)}")

        if history:
            print("\n🔔 Últimas detecciones:")            for detection in history[-3:]:
                print(f"   {detection['timestamp'][:19]} - {detection['coordinates']} - Prob: {detection['seismic_probability']}/10.0")

def demo_json_output():
    """Demo del formato JSON de salida."""
    print("\n📄 Demo: Formato JSON de Salida")
    print("=" * 40)

    # Simular detección
    result = quick_seismic_scan(19.4326, -99.1332)  # Ciudad de México

    if "seismic_probability" in result:
        print("🚨 Alerta Sísmica Detectada - JSON Output:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("✅ Sin actividad sísmica - JSON Output:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

def demo_api_integration():
    """Demo de integración con API."""
    print("\n🔗 Demo: Integración con API")
    print("=" * 40)

    print("Este agente puede integrarse fácilmente con APIs REST/GraphQL:")
    print()

    # Ejemplos de endpoints
    api_endpoints = {
        "POST /api/seismic/scan": {
            "description": "Escaneo manual de coordenadas",
            "request": {"lat": 19.4326, "lon": -99.1332},
            "response": {"seismic_probability": 4.3, "intensity": 4.36, "coordinates": {"lat": 19.4326, "lon": -99.1332}}
        },
        "GET /api/seismic/alerts": {
            "description": "Obtener alertas activas",
            "response": [{"alert_id": "seismic_123456", "coordinates": {"lat": 19.4326, "lon": -99.1332}, "seismic_probability": 7.2}]
        },
        "POST /api/seismic/zones": {
            "description": "Agregar zona de monitoreo",
            "request": {"name": "Nueva_Zona", "lat": 40.7128, "lon": -74.0060, "radius_km": 50.0}
        },
        "GET /api/seismic/stats": {
            "description": "Obtener estadísticas del sistema",
            "response": {"total_scans": 150, "alerts_generated": 12, "monitoring_zones": 5}
        }
    }

    for endpoint, details in api_endpoints.items():
        print(f"🔗 {endpoint}")
        print(f"   {details['description']}")

        if 'request' in details:
            print(f"   📤 Request: {json.dumps(details['request'], indent=6)}")

        if 'response' in details:
            print(f"   📥 Response: {json.dumps(details['response'], indent=6)}")

        print()

def main():
    """Demo completo del sistema de detección sísmica."""
    print("🔔 Sistema Completo de Detección de Seísmos en Tiempo Real")
    print("=" * 60)
    print("Este demo muestra todas las funcionalidades del agente sísmico:")
    print("• Detección en tiempo real con coordenadas GPS")
    print("• Alertas JSON con posibilidad (0.0-10.0) e intensidad")
    print("• Monitoreo continuo de múltiples zonas")
    print("• Integración con modelos de IA de clasificación sísmica")
    print()

    # Ejecutar demos
    demo_manual_scans()
    demo_json_output()
    demo_realtime_monitoring()
    demo_api_integration()

    print("\n🎉 Demo completado exitosamente!")
    print("\n💡 El agente está listo para:")
    print("   • Monitoreo sísmico 24/7")
    print("   • Alertas automáticas por email/SMS/webhooks")
    print("   • Integración con sistemas de emergencia")
    print("   • Análisis predictivo de riesgos sísmicos")

if __name__ == "__main__":
    main()