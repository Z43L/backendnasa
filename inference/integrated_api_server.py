 #!/usr/bin/env python3
"""
API Integrada: Meteorología + Nivel del Mar
==========================================

Extensión de la API meteorológica para incluir análisis de nivel del mar.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import time
from datetime import datetime, timedelta
import threading
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

# Importar sistemas existentes
from simple_weather_agent import SimpleWeatherAgent
from sea_level_analyzer import SeaLevelAnalyzer

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales para agentes
weather_agent = None
sea_level_agent = None
agents_initialized = False
initialization_error = None

def initialize_integrated_agents():
    """Inicializar ambos agentes."""
    global weather_agent, sea_level_agent, agents_initialized, initialization_error
    
    try:
        logger.info("🚀 Inicializando agentes integrados...")
        
        # Inicializar agente meteorológico
        logger.info("⛈️ Inicializando agente meteorológico...")
        weather_agent = SimpleWeatherAgent()
        weather_success = weather_agent.initialize()
        
        # Inicializar analizador de nivel del mar
        logger.info("🌊 Inicializando analizador de nivel del mar...")
        sea_level_agent = SeaLevelAnalyzer()
        
        if weather_success:
            agents_initialized = True
            initialization_error = None
            logger.info("✅ Agentes integrados inicializados exitosamente")
            return True
        else:
            raise Exception("Error inicializando agente meteorológico")
            
    except Exception as e:
        error_msg = f"Error inicializando agentes: {str(e)}"
        logger.error(f"❌ {error_msg}")
        initialization_error = error_msg
        agents_initialized = False
        return False

def create_integrated_api():

    @app.route('/api/precipitation/real', methods=['GET'])
    def get_precipitation_real():
        """Devuelve datos reales de precipitación IMERG para una región y tiempo usando OPeNDAP/xarray."""
        if not agents_initialized:
            return create_error_response("Agentes no inicializados", 503)
        # Parámetros: bbox (west,south,east,north), date (YYYY-MM-DD)
        try:
            bbox = request.args.get('bbox')
            date = request.args.get('date', datetime.utcnow().strftime('%Y-%m-%d'))
            if bbox:
                bbox = [float(x) for x in bbox.split(',')]
            else:
                # Por defecto: España
                bbox = [-9.5, 36, 3.5, 44]
            # Construir URL OPeNDAP IMERG (ajustar año/mes/día)
            year = int(date[:4])
            month = int(date[5:7])
            day = int(date[8:10])
            imerg_url = f"https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHH.07/{year}/{month:02d}/3B-HHR.MS.MRG.3IMERG.{year}{month:02d}{day:02d}-S000000-E002959.0000.V07B.HDF5.nc4"
            import xarray as xr
            ds = xr.open_dataset(imerg_url)
            # Seleccionar lat/lon
            lat = ds['lat']
            lon = ds['lon']
            lat_mask = (lat >= bbox[1]) & (lat <= bbox[3])
            lon_mask = (lon >= bbox[0]) & (lon <= bbox[2])
            lats = lat.values[lat_mask]
            lons = lon.values[lon_mask]
            # Seleccionar primer tiempo disponible
            precip = ds['precipitationCal'].isel(time=0, lat=lat_mask, lon=lon_mask).values
            precip_grid = precip.tolist()
            result = {
                'bbox': bbox,
                'date': date,
                'precip_grid': precip_grid,
                'lat': lats.tolist(),
                'lon': lons.tolist(),
                'units': ds['precipitationCal'].attrs.get('units', 'mm/hr'),
                'data_source': imerg_url
            }
            ds.close()
            return jsonify(create_success_response(result, "Datos reales de precipitación extraídos correctamente"))
        except Exception as e:
            logger.error(f"Error extrayendo precipitación real: {e}")
            return create_error_response(f"Error extrayendo datos reales de precipitación: {str(e)}", 500)
    """Crear API integrada."""
    app = Flask(__name__)
    
    # Configurar CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
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
    
    def create_success_response(data: Dict, message: str = "Operación exitosa") -> Dict:
        """Crear respuesta de éxito estándar."""
        return {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "api_version": "2.0_integrated"
        }
    
    # =================== ENDPOINTS EXISTENTES ===================

    @app.route('/api/sealevel/ssha', methods=['GET'])
    def get_sea_level_ssha():
        """Devuelve datos reales de anomalía del nivel del mar (SSHA) para una región y periodo."""
        if not agents_initialized or not sea_level_agent:
            return create_error_response("Analizador de nivel del mar no disponible", 503)

        # Parámetros: region_key, start_date, end_date, dataset_key
        region_key = request.args.get('region', 'global')
        start_date = request.args.get('start_date', '2023-01-01')
        end_date = request.args.get('end_date', '2023-12-31')
        dataset_key = request.args.get('dataset', 'merged_alt_l4')

        try:
            # Llama a la función real (no simulada) del analizador
            result = sea_level_agent.analyze_sea_level_region(
                region_key=region_key,
                start_date=start_date,
                end_date=end_date,
                dataset_key=dataset_key,
                use_real_data=True  # Debe estar implementado en el analizador
            )
            if not result or 'ssha_grid' not in result:
                return create_error_response("No se pudieron obtener datos reales de SSHA para la región y periodo especificados.", 404)

            # Devuelve solo la matriz SSHA y metadatos mínimos para el frontend
            response_data = {
                "region": result.get("region", {}),
                "period": result.get("analysis_period", {}),
                "ssha_grid": result["ssha_grid"],  # Matriz 2D de valores SSHA
                "lat": result.get("lat", []),      # Vector de latitudes
                "lon": result.get("lon", []),      # Vector de longitudes
                "units": result.get("units", "m")
            }
            return jsonify(create_success_response(response_data, "Datos SSHA reales extraídos correctamente"))
        except Exception as e:
            logger.error(f"Error extrayendo SSHA real: {e}")
            return create_error_response(f"Error extrayendo datos reales de SSHA: {str(e)}", 500)
    
    @app.route('/', methods=['GET'])
    def home():
        """Endpoint de información general."""
        return jsonify({
            "service": "Integrated Weather & Sea Level API",
            "version": "2.0",
            "description": "API integrada para predicciones meteorológicas y análisis de nivel del mar",
            "status": "active" if agents_initialized else "initializing",
            "capabilities": {
                "weather_prediction": "Predicciones de lluvia globales",
                "sea_level_analysis": "Análisis de tendencias del nivel del mar",
                "integrated_analysis": "Análisis combinado clima-océano"
            },
            "endpoints": {
                "health": "/api/health",
                "weather": {
                    "predict_point": "/api/weather/predict/point",
                    "predict_multiple": "/api/weather/predict/multiple",
                    "predict_area": "/api/weather/predict/area"
                },
                "sea_level": {
                    "analyze_region": "/api/sealevel/analyze/region",
                    "compare_regions": "/api/sealevel/compare/regions",
                    "list_regions": "/api/sealevel/regions"
                },
                "integrated": {
                    "coastal_analysis": "/api/integrated/coastal",
                    "climate_ocean_report": "/api/integrated/report"
                }
            },
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Verificar estado del servicio."""
        if not agents_initialized:
            return jsonify({
                "status": "unhealthy",
                "weather_agent": False,
                "sea_level_agent": False,
                "error": initialization_error,
                "timestamp": datetime.now().isoformat()
            }), 503
        
        return jsonify({
            "status": "healthy",
            "weather_agent": True,
            "sea_level_agent": True,
            "capabilities": ["weather_prediction", "sea_level_analysis", "integrated_analysis"],
            "timestamp": datetime.now().isoformat()
        })
    
    # =================== ENDPOINTS METEOROLÓGICOS ===================
    
    @app.route('/api/weather/predict/point', methods=['POST', 'GET'])
    def weather_predict_point():
        """Predicción meteorológica en punto."""
        if not agents_initialized or not weather_agent:
            return create_error_response("Agente meteorológico no disponible", 503)
        
        try:
            # Obtener parámetros
            if request.method == 'POST':
                data = request.get_json()
                if not data:
                    return create_error_response("JSON requerido")
                lat = data.get('latitude') or data.get('lat')
                lon = data.get('longitude') or data.get('lon')
            else:
                lat = request.args.get('latitude') or request.args.get('lat')
                lon = request.args.get('longitude') or request.args.get('lon')
            
            if lat is None or lon is None:
                return create_error_response("Parámetros 'latitude' y 'longitude' requeridos")
            
            lat, lon = float(lat), float(lon)
            
            # Validar coordenadas
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                return create_error_response("Coordenadas fuera de rango")
            
            # Realizar predicción
            start_time = time.time()
            resultado = weather_agent.api['predict_point'](lat, lon)
            prediction_time = time.time() - start_time
            
            response_data = {
                "prediction": resultado,
                "performance": {
                    "prediction_time_ms": round(prediction_time * 1000, 2)
                }
            }
            
            return jsonify(create_success_response(
                response_data,
                f"Predicción meteorológica exitosa para ({lat:.4f}, {lon:.4f})"
            ))
            
        except Exception as e:
            logger.error(f"Error en predicción meteorológica: {e}")
            return create_error_response(f"Error interno: {str(e)}", 500)
    
    # =================== ENDPOINTS NIVEL DEL MAR ===================
    
    @app.route('/api/sealevel/regions', methods=['GET'])
    def list_sea_level_regions():
        """Listar regiones disponibles para análisis de nivel del mar."""
        if not agents_initialized or not sea_level_agent:
            return create_error_response("Analizador de nivel del mar no disponible", 503)
        
        try:
            regions_info = []
            for key, region in sea_level_agent.regions.items():
                regions_info.append({
                    "key": key,
                    "name": region['name'],
                    "latitude_range": region['lat'],
                    "longitude_range": region['lon'],
                    "description": f"Región {region['name']} ({region['lat'][0]}° a {region['lat'][1]}° lat, {region['lon'][0]}° a {region['lon'][1]}° lon)"
                })
            
            return jsonify(create_success_response({
                "regions": regions_info,
                "total_regions": len(regions_info)
            }, "Regiones obtenidas exitosamente"))
            
        except Exception as e:
            logger.error(f"Error listando regiones: {e}")
            return create_error_response(f"Error interno: {str(e)}", 500)
    
    @app.route('/api/sealevel/analyze/region', methods=['POST'])
    def analyze_sea_level_region():
        """Analizar nivel del mar en una región."""
        if not agents_initialized or not sea_level_agent:
            return create_error_response("Analizador de nivel del mar no disponible", 503)
        
        try:
            data = request.get_json()
            if not data:
                return create_error_response("JSON requerido")
            
            region_key = data.get('region')
            start_date = data.get('start_date', '2023-01-01')
            end_date = data.get('end_date', '2023-12-31')
            
            if not region_key:
                return create_error_response("Parámetro 'region' requerido")
            
            if region_key not in sea_level_agent.regions:
                available_regions = list(sea_level_agent.regions.keys())
                return create_error_response(
                    f"Región '{region_key}' no válida. Regiones disponibles: {available_regions}"
                )
            
            # Realizar análisis
            start_time = time.time()
            resultado = sea_level_agent.analyze_sea_level_region(
                region_key, start_date, end_date
            )
            analysis_time = time.time() - start_time
            
            if not resultado:
                return create_error_response("Error en análisis de nivel del mar", 500)
            
            # Añadir información de rendimiento
            resultado['performance'] = {
                "analysis_time_ms": round(analysis_time * 1000, 2),
                "data_source": "simulated_realistic"
            }
            
            return jsonify(create_success_response(
                resultado,
                f"Análisis de nivel del mar completado para {sea_level_agent.regions[region_key]['name']}"
            ))
            
        except Exception as e:
            logger.error(f"Error en análisis de nivel del mar: {e}")
            return create_error_response(f"Error interno: {str(e)}", 500)
    
    # =================== ENDPOINTS INTEGRADOS ===================
    
    @app.route('/api/integrated/coastal', methods=['POST'])
    def integrated_coastal_analysis():
        """Análisis integrado costero: meteorología + nivel del mar."""
        if not agents_initialized or not weather_agent or not sea_level_agent:
            return create_error_response("Agentes no disponibles", 503)
        
        try:
            data = request.get_json()
            if not data:
                return create_error_response("JSON requerido")
            
            # Coordenadas costeras
            coordinates = data.get('coordinates', [])
            region_key = data.get('region', 'mediterranean')
            
            if not coordinates:
                return create_error_response("Parámetro 'coordinates' requerido")
            
            # Análisis meteorológico
            start_time = time.time()
            logger.info("Realizando predicciones meteorológicas...")
            weather_results = weather_agent.api['predict_multiple'](coordinates)
            
            # Análisis de nivel del mar
            logger.info("Analizando nivel del mar...")
            sea_level_results = sea_level_agent.analyze_sea_level_region(region_key)
            
            total_time = time.time() - start_time
            
            # Análisis integrado
            integrated_analysis = {
                "coastal_risk_assessment": create_coastal_risk_assessment(
                    weather_results, sea_level_results, coordinates
                ),
                "weather_predictions": weather_results,
                "sea_level_analysis": sea_level_results,
                "integration_summary": {
                    "total_coordinates": len(coordinates),
                    "region_analyzed": sea_level_agent.regions[region_key]['name'],
                    "analysis_date": datetime.now().isoformat(),
                    "combined_indicators": calculate_combined_indicators(
                        weather_results, sea_level_results
                    )
                },
                "performance": {
                    "total_analysis_time_ms": round(total_time * 1000, 2),
                    "weather_prediction_time": "included_in_total",
                    "sea_level_analysis_time": "included_in_total"
                }
            }
            
            return jsonify(create_success_response(
                integrated_analysis,
                "Análisis costero integrado completado"
            ))
            
        except Exception as e:
            logger.error(f"Error en análisis integrado: {e}")
            return create_error_response(f"Error interno: {str(e)}", 500)
    
    def create_coastal_risk_assessment(weather_results, sea_level_results, coordinates):
        """Crear evaluación de riesgo costero."""
        risk_assessment = {
            "overall_risk_level": "medio",
            "risk_factors": [],
            "recommendations": [],
            "coordinate_risks": []
        }
        
        # Analizar riesgo meteorológico
        if weather_results:
            avg_rain_prob = np.mean([r['prediction']['rain_probability_0_10'] for r in weather_results])
            if avg_rain_prob > 7:
                risk_assessment["risk_factors"].append("Alta probabilidad de precipitación")
                risk_assessment["recommendations"].append("Monitorear drenaje costero")
        
        # Analizar riesgo nivel del mar
        if sea_level_results and 'statistics' in sea_level_results:
            trend = sea_level_results['statistics']['trend_mm_per_year']
            if trend > 3:
                risk_assessment["risk_factors"].append("Tendencia ascendente del nivel del mar")
                risk_assessment["recommendations"].append("Implementar medidas de adaptación costera")
        
        # Evaluar riesgo por coordenada
        for i, (coord, weather_pred) in enumerate(zip(coordinates, weather_results)):
            rain_prob = weather_pred['prediction']['rain_probability_0_10']
            
            # Riesgo combinado simplificado
            if rain_prob > 8:
                risk_level = "alto"
            elif rain_prob > 6:
                risk_level = "medio"
            else:
                risk_level = "bajo"
            
            risk_assessment["coordinate_risks"].append({
                "coordinate": coord,
                "rain_probability": rain_prob,
                "risk_level": risk_level,
                "factors": ["precipitación"] if rain_prob > 6 else []
            })
        
        # Determinar riesgo general
        high_risk_coords = sum(1 for r in risk_assessment["coordinate_risks"] if r["risk_level"] == "alto")
        total_coords = len(risk_assessment["coordinate_risks"])
        
        if high_risk_coords > total_coords * 0.5:
            risk_assessment["overall_risk_level"] = "alto"
        elif high_risk_coords > total_coords * 0.2:
            risk_assessment["overall_risk_level"] = "medio"
        else:
            risk_assessment["overall_risk_level"] = "bajo"
        
        return risk_assessment
    
    def calculate_combined_indicators(weather_results, sea_level_results):
        """Calcular indicadores combinados."""
        indicators = {
            "precipitation_index": 0,
            "sea_level_trend_index": 0,
            "combined_coastal_index": 0
        }
        
        # Índice de precipitación
        if weather_results:
            avg_rain = np.mean([r['prediction']['rain_probability_0_10'] for r in weather_results])
            indicators["precipitation_index"] = round(avg_rain / 10, 2)  # Normalizar 0-1
        
        # Índice de tendencia del nivel del mar
        if sea_level_results and 'statistics' in sea_level_results:
            trend = sea_level_results['statistics']['trend_mm_per_year']
            # Normalizar tendencia (asumiendo rango típico -10 a +10 mm/año)
            normalized_trend = max(0, min(1, (trend + 10) / 20))
            indicators["sea_level_trend_index"] = round(normalized_trend, 2)
        
        # Índice combinado
        indicators["combined_coastal_index"] = round(
            (indicators["precipitation_index"] + indicators["sea_level_trend_index"]) / 2, 2
        )
        
        return indicators
    
    return app

def run_integrated_server():
    """Ejecutar servidor integrado."""
    print("🌍 INICIANDO SERVIDOR API INTEGRADO")
    print("=" * 40)
    print("⛈️ Meteorología + 🌊 Nivel del Mar")
    print()
    
    if initialize_integrated_agents():
        print("✅ Agentes inicializados exitosamente")
        
        app = create_integrated_api()
        
        print("🌐 Endpoints disponibles:")
        print("   Meteorología:")
        print("     POST /api/weather/predict/point")
        print("   Nivel del Mar:")
        print("     GET  /api/sealevel/regions")
        print("     POST /api/sealevel/analyze/region")
        print("   Integrados:")
        print("     POST /api/integrated/coastal")
        print()
        print("🚀 Servidor iniciado en:")
        print("   URL: http://localhost:5002")
        print("   API: http://localhost:5002/api/")
        print()
        
        app.run(
            host='0.0.0.0',
            port=5002,
            debug=False,
            threaded=True
        )
    else:
        print("❌ No se pudieron inicializar los agentes")
        exit(1)

if __name__ == '__main__':
    run_integrated_server()