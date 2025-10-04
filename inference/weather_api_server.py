#!/usr/bin/env python3
"""
Weather API Server - Servidor Flask para Predicciones Meteorológicas
====================================================================

API REST que mantiene el agente meteorológico activo y responde a peticiones
para predecir lluvia en cualquier coordenada del mundo.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import time
from datetime import datetime
import threading
import logging
from typing import Dict, List, Tuple, Optional

# Importar nuestro agente meteorológico
from simple_weather_agent import SimpleWeatherAgent

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)

# Configurar CORS para permitir peticiones desde cualquier frontend
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Variable global para mantener el agente activo
weather_agent = None
agent_initialized = False
initialization_error = None

class WeatherAPIServer:
    """Clase para manejar el servidor de API meteorológica."""
    
    def __init__(self):
        """Inicializar servidor."""
        self.agent = None
        self.initialized = False
        self.error = None
        
    def initialize_agent(self):
        """Inicializar el agente meteorológico."""
        global weather_agent, agent_initialized, initialization_error
        
        try:
            logger.info("🚀 Inicializando agente meteorológico...")
            weather_agent = SimpleWeatherAgent()
            
            if weather_agent.initialize():
                agent_initialized = True
                initialization_error = None
                logger.info("✅ Agente meteorológico inicializado exitosamente")
                return True
            else:
                raise Exception("Error inicializando agente")
                
        except Exception as e:
            error_msg = f"Error inicializando agente: {str(e)}"
            logger.error(f"❌ {error_msg}")
            initialization_error = error_msg
            agent_initialized = False
            return False

def validate_coordinates(lat: float, lon: float) -> Tuple[bool, str]:
    """Validar coordenadas geográficas."""
    try:
        lat = float(lat)
        lon = float(lon)
        
        if not (-90 <= lat <= 90):
            return False, f"Latitud inválida: {lat}. Debe estar entre -90 y 90"
        
        if not (-180 <= lon <= 180):
            return False, f"Longitud inválida: {lon}. Debe estar entre -180 y 180"
        
        return True, ""
        
    except (ValueError, TypeError):
        return False, "Coordenadas deben ser números válidos"

def create_error_response(message: str, status_code: int = 400) -> Tuple[Dict, int]:
    """Crear respuesta de error estándar."""
    return {
        "success": False,
        "error": {
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "status_code": status_code
        }
    }, status_code

def create_success_response(data: Dict, message: str = "Predicción exitosa") -> Dict:
    """Crear respuesta de éxito estándar."""
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.now().isoformat(),
        "api_version": "1.0"
    }

# =================== ENDPOINTS DE LA API ===================

@app.route('/', methods=['GET'])
def home():
    """Endpoint de información general."""
    return jsonify({
        "service": "Weather Prediction API",
        "version": "1.0",
        "description": "API para predicciones meteorológicas usando IA",
        "status": "active" if agent_initialized else "initializing",
        "endpoints": {
            "health": "/api/health",
            "predict_point": "/api/predict/point",
            "predict_multiple": "/api/predict/multiple",
            "predict_area": "/api/predict/area",
            "stats": "/api/stats"
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Verificar estado del servicio."""
    if not agent_initialized:
        return jsonify({
            "status": "unhealthy",
            "agent_ready": False,
            "error": initialization_error,
            "timestamp": datetime.now().isoformat()
        }), 503
    
    return jsonify({
        "status": "healthy",
        "agent_ready": True,
        "uptime": "running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/predict/point', methods=['POST', 'GET'])
def predict_single_point():
    """Predecir lluvia en un punto específico."""
    if not agent_initialized:
        return create_error_response("Agente meteorológico no inicializado", 503)
    
    try:
        # Obtener parámetros (POST con JSON o GET con query params)
        if request.method == 'POST':
            data = request.get_json()
            if not data:
                return create_error_response("JSON requerido en el body")
            
            lat = data.get('latitude') or data.get('lat')
            lon = data.get('longitude') or data.get('lon')
            
        else:  # GET
            lat = request.args.get('latitude') or request.args.get('lat')
            lon = request.args.get('longitude') or request.args.get('lon')
        
        if lat is None or lon is None:
            return create_error_response("Parámetros 'latitude' y 'longitude' requeridos")
        
        # Validar coordenadas
        valid, error_msg = validate_coordinates(lat, lon)
        if not valid:
            return create_error_response(error_msg)
        
        lat, lon = float(lat), float(lon)
        
        # Realizar predicción
        start_time = time.time()
        resultado = weather_agent.api['predict_point'](lat, lon)
        prediction_time = time.time() - start_time
        
        # Estructurar respuesta
        response_data = {
            "prediction": resultado,
            "performance": {
                "prediction_time_ms": round(prediction_time * 1000, 2),
                "coordinates_processed": 1
            }
        }
        
        return jsonify(create_success_response(
            response_data, 
            f"Predicción exitosa para ({lat:.4f}, {lon:.4f})"
        ))
        
    except Exception as e:
        logger.error(f"Error en predicción de punto: {e}")
        return create_error_response(f"Error interno: {str(e)}", 500)

@app.route('/api/predict/multiple', methods=['POST'])
def predict_multiple_points():
    """Predecir lluvia en múltiples puntos."""
    if not agent_initialized:
        return create_error_response("Agente meteorológico no inicializado", 503)
    
    try:
        data = request.get_json()
        if not data:
            return create_error_response("JSON requerido en el body")
        
        coordinates = data.get('coordinates', [])
        if not coordinates:
            return create_error_response("Array 'coordinates' requerido")
        
        if len(coordinates) > 100:  # Límite de seguridad
            return create_error_response("Máximo 100 coordenadas por petición")
        
        # Validar todas las coordenadas
        validated_coords = []
        for i, coord in enumerate(coordinates):
            if not isinstance(coord, (list, tuple)) or len(coord) != 2:
                return create_error_response(f"Coordenada {i}: formato inválido. Use [lat, lon]")
            
            lat, lon = coord[0], coord[1]
            valid, error_msg = validate_coordinates(lat, lon)
            if not valid:
                return create_error_response(f"Coordenada {i}: {error_msg}")
            
            validated_coords.append((float(lat), float(lon)))
        
        # Realizar predicciones
        start_time = time.time()
        resultados = weather_agent.api['predict_multiple'](validated_coords)
        prediction_time = time.time() - start_time
        
        # Estructurar respuesta
        response_data = {
            "predictions": resultados,
            "summary": {
                "total_points": len(resultados),
                "average_rain_probability": round(
                    sum(r['prediction']['rain_probability_0_10'] for r in resultados) / len(resultados), 2
                ),
                "high_probability_count": sum(
                    1 for r in resultados if r['prediction']['rain_probability_0_10'] > 7
                )
            },
            "performance": {
                "prediction_time_ms": round(prediction_time * 1000, 2),
                "coordinates_processed": len(resultados),
                "avg_time_per_point_ms": round((prediction_time * 1000) / len(resultados), 2)
            }
        }
        
        return jsonify(create_success_response(
            response_data,
            f"Predicción exitosa para {len(resultados)} puntos"
        ))
        
    except Exception as e:
        logger.error(f"Error en predicción múltiple: {e}")
        return create_error_response(f"Error interno: {str(e)}", 500)

@app.route('/api/predict/area', methods=['POST'])
def predict_area_grid():
    """Predecir lluvia en un área geográfica."""
    if not agent_initialized:
        return create_error_response("Agente meteorológico no inicializado", 503)
    
    try:
        data = request.get_json()
        if not data:
            return create_error_response("JSON requerido en el body")
        
        # Obtener parámetros del área
        lat_min = data.get('lat_min')
        lat_max = data.get('lat_max')
        lon_min = data.get('lon_min')
        lon_max = data.get('lon_max')
        resolution = data.get('resolution', 10)
        
        if any(x is None for x in [lat_min, lat_max, lon_min, lon_max]):
            return create_error_response("Parámetros requeridos: lat_min, lat_max, lon_min, lon_max")
        
        # Validar parámetros
        try:
            lat_min, lat_max = float(lat_min), float(lat_max)
            lon_min, lon_max = float(lon_min), float(lon_max)
            resolution = int(resolution)
        except (ValueError, TypeError):
            return create_error_response("Parámetros deben ser números válidos")
        
        if lat_min >= lat_max:
            return create_error_response("lat_min debe ser menor que lat_max")
        
        if lon_min >= lon_max:
            return create_error_response("lon_min debe ser menor que lon_max")
        
        if not (5 <= resolution <= 50):
            return create_error_response("Resolución debe estar entre 5 y 50")
        
        # Validar coordenadas
        for lat in [lat_min, lat_max]:
            valid, error_msg = validate_coordinates(lat, 0)
            if not valid:
                return create_error_response(f"Latitud inválida: {error_msg}")
        
        for lon in [lon_min, lon_max]:
            valid, error_msg = validate_coordinates(0, lon)
            if not valid:
                return create_error_response(f"Longitud inválida: {error_msg}")
        
        # Realizar predicción de área
        start_time = time.time()
        resultado = weather_agent.api['predict_area'](
            (lat_min, lat_max), 
            (lon_min, lon_max), 
            resolution
        )
        prediction_time = time.time() - start_time
        
        # Añadir información de rendimiento
        resultado['performance'] = {
            "prediction_time_ms": round(prediction_time * 1000, 2),
            "total_grid_points": resolution * resolution,
            "avg_time_per_point_ms": round((prediction_time * 1000) / (resolution * resolution), 2)
        }
        
        return jsonify(create_success_response(
            resultado,
            f"Predicción de área exitosa ({resolution}x{resolution} puntos)"
        ))
        
    except Exception as e:
        logger.error(f"Error en predicción de área: {e}")
        return create_error_response(f"Error interno: {str(e)}", 500)

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Obtener estadísticas del servicio."""
    if not agent_initialized:
        return create_error_response("Agente meteorológico no inicializado", 503)
    
    try:
        # Obtener información del agente
        stats = {
            "service_info": {
                "name": "Weather Prediction API",
                "version": "1.0",
                "agent_type": "SimpleWeatherAgent",
                "data_source": "NASA_IMERG"
            },
            "data_info": {
                "spatial_resolution": f"{weather_agent.spatial_shape[0]}x{weather_agent.spatial_shape[1]}",
                "temporal_sequences": weather_agent.weather_data['shape_info']['sequences'],
                "temporal_frames": weather_agent.weather_data['shape_info']['temporal_frames'],
                "spatial_points": weather_agent.weather_data['shape_info']['spatial_points']
            },
            "prediction_stats": {
                "value_range": {
                    "min": weather_agent.weather_data['stats']['min'],
                    "max": weather_agent.weather_data['stats']['max'],
                    "mean": weather_agent.weather_data['stats']['mean'],
                    "std": weather_agent.weather_data['stats']['std']
                },
                "temporal_weights_count": len(weather_agent.temporal_weights)
            },
            "api_capabilities": {
                "max_multiple_coordinates": 100,
                "max_area_resolution": 50,
                "supported_formats": ["JSON"],
                "cors_enabled": True
            },
            "status": {
                "initialized": agent_initialized,
                "ready": True,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return jsonify(create_success_response(stats, "Estadísticas obtenidas exitosamente"))
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        return create_error_response(f"Error interno: {str(e)}", 500)

# Endpoint adicional para facilitar pruebas desde el navegador
@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Endpoint de prueba rápida."""
    if not agent_initialized:
        return create_error_response("Agente meteorológico no inicializado", 503)
    
    try:
        # Hacer una predicción de prueba en Ciudad de México
        test_coords = (19.4326, -99.1332)
        resultado = weather_agent.api['predict_point'](test_coords[0], test_coords[1])
        
        return jsonify(create_success_response({
            "test_location": "Ciudad de México",
            "coordinates": test_coords,
            "prediction": resultado
        }, "Test exitoso"))
        
    except Exception as e:
        return create_error_response(f"Error en test: {str(e)}", 500)

# =================== MANEJO DE ERRORES ===================

@app.errorhandler(404)
def not_found(error):
    """Manejar rutas no encontradas."""
    return jsonify({
        "success": False,
        "error": {
            "message": "Endpoint no encontrado",
            "status_code": 404,
            "available_endpoints": [
                "/api/health",
                "/api/predict/point",
                "/api/predict/multiple", 
                "/api/predict/area",
                "/api/stats",
                "/api/test"
            ]
        }
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Manejar métodos no permitidos."""
    return jsonify({
        "success": False,
        "error": {
            "message": "Método HTTP no permitido",
            "status_code": 405
        }
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Manejar errores internos."""
    return jsonify({
        "success": False,
        "error": {
            "message": "Error interno del servidor",
            "status_code": 500
        }
    }), 500

# =================== INICIALIZACIÓN Y SERVIDOR ===================

def initialize_server():
    """Inicializar el servidor con el agente meteorológico."""
    print("🌦️ WEATHER API SERVER - INICIALIZANDO")
    print("=" * 40)
    
    server = WeatherAPIServer()
    if server.initialize_agent():
        print("✅ Servidor inicializado exitosamente")
        print("🌐 Endpoints disponibles:")
        print("   GET  /                     - Información general")
        print("   GET  /api/health           - Estado del servicio")
        print("   POST /api/predict/point    - Predicción en punto")
        print("   GET  /api/predict/point    - Predicción en punto (query)")
        print("   POST /api/predict/multiple - Predicción múltiple")
        print("   POST /api/predict/area     - Predicción de área")
        print("   GET  /api/stats            - Estadísticas")
        print("   GET  /api/test             - Endpoint de prueba")
        print()
        return True
    else:
        print("❌ Error inicializando servidor")
        return False

if __name__ == '__main__':
    # Configurar el servidor
    print("🚀 INICIANDO WEATHER API SERVER")
    print("=" * 35)
    
    if initialize_server():
        print("🌐 Servidor Flask iniciado en:")
        print("   URL: http://localhost:5001")
        print("   API: http://localhost:5001/api/")
        print()
        print("🔗 Ejemplos de uso:")
        print("   curl http://localhost:5001/api/health")
        print("   curl http://localhost:5001/api/test")
        print()
        print("📝 Para detener el servidor: Ctrl+C")
        print("=" * 50)
        
        # Iniciar servidor Flask
        app.run(
            host='0.0.0.0',  # Accesible desde cualquier IP
            port=5001,  # Usar puerto 5001 para evitar conflictos
            debug=False,  # Cambiar a True para desarrollo
            threaded=True   # Soporte para múltiples peticiones simultáneas
        )
    else:
        print("❌ No se pudo inicializar el servidor")
        exit(1)