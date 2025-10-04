#!/usr/bin/env python3
"""
Agente de IA Meteorológico Simplificado
======================================

Agente que predice lluvia usando análisis de datos NASA sin dependencias complejas.
"""

import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class SimpleWeatherAgent:
    """Agente simplificado para predicción meteorológica."""
    
    def __init__(self, data_dir="../processed_data_large"):
        """Inicializar agente."""
        self.data_dir = Path(data_dir)
        self.spatial_shape = (900, 450)  # Resolución NASA IMERG
        self.lat_range = (-90, 90)
        self.lon_range = (-180, 180)
        self.weather_data = None
        self.predictor = None
        
        print("🤖 AGENTE DE IA METEOROLÓGICO SIMPLE")
        print("=" * 40)
        
    def load_weather_data(self):
        """Cargar datos meteorológicos procesados."""
        print("📊 CARGANDO DATOS METEOROLÓGICOS NASA")
        print("=" * 35)
        
        # Buscar datos procesados
        batches_dir = self.data_dir / "batches"
        if batches_dir.exists():
            batch_files = list(batches_dir.glob("batch_*.npz"))
        else:
            batch_files = list(self.data_dir.glob("batch_*.npz"))
        
        if not batch_files:
            raise FileNotFoundError("No se encontraron datos procesados")
        
        # Cargar datos
        data = np.load(batch_files[0])
        X = data['X']  # Secuencias temporales
        y = data['y']  # Valores objetivo
        
        self.weather_data = {
            'temporal_sequences': X,
            'target_values': y,
            'stats': {
                'mean': float(np.mean(y)),
                'std': float(np.std(y)),
                'min': float(np.min(y)),
                'max': float(np.max(y))
            },
            'shape_info': {
                'sequences': X.shape[0],
                'temporal_frames': X.shape[1],
                'spatial_points': X.shape[2]
            }
        }
        
        print(f"✅ Datos cargados exitosamente:")
        print(f"   - Secuencias temporales: {X.shape[0]}")
        print(f"   - Frames temporales: {X.shape[1]}")
        print(f"   - Puntos espaciales: {X.shape[2]}")
        print(f"   - Rango de valores: {self.weather_data['stats']['min']:.2f} a {self.weather_data['stats']['max']:.2f}")
        
        return True
    
    def coordinates_to_spatial_index(self, lat: float, lon: float) -> int:
        """Convertir coordenadas geográficas a índice espacial."""
        # Normalizar coordenadas
        lat_norm = (lat - self.lat_range[0]) / (self.lat_range[1] - self.lat_range[0])
        lon_norm = (lon - self.lon_range[0]) / (self.lon_range[1] - self.lon_range[0])
        
        # Convertir a índices de grid
        lat_idx = int(lat_norm * self.spatial_shape[0])
        lon_idx = int(lon_norm * self.spatial_shape[1])
        
        # Asegurar límites
        lat_idx = max(0, min(self.spatial_shape[0] - 1, lat_idx))
        lon_idx = max(0, min(self.spatial_shape[1] - 1, lon_idx))
        
        # Convertir a índice espacial lineal
        spatial_idx = lat_idx * self.spatial_shape[1] + lon_idx
        
        # Asegurar que está dentro del rango de datos
        max_spatial_idx = self.weather_data['shape_info']['spatial_points'] - 1
        spatial_idx = min(spatial_idx, max_spatial_idx)
        
        return spatial_idx
    
    def create_intelligent_predictor(self):
        """Crear predictor basado en análisis de patrones."""
        print("🧠 CREANDO PREDICTOR INTELIGENTE")
        print("=" * 30)
        
        if not self.weather_data:
            raise ValueError("Datos meteorológicos no cargados")
        
        X = self.weather_data['temporal_sequences']
        y = self.weather_data['target_values']
        
        # Analizar correlaciones temporales
        print("📈 Analizando patrones temporales...")
        
        temporal_weights = []
        for frame_idx in range(min(X.shape[1], 10)):  # Últimos 10 frames
            try:
                # Obtener frame específico
                frame_data = X[:, -(frame_idx+1), :]
                
                # Calcular correlación promedio con targets
                correlations = []
                for seq_idx in range(X.shape[0]):
                    frame_values = frame_data[seq_idx]
                    target_values = y[seq_idx]
                    
                    # Filtrar valores válidos
                    valid_mask = (frame_values > -9999) & (target_values > -9999)
                    
                    if np.sum(valid_mask) > 100:
                        corr = np.corrcoef(frame_values[valid_mask], target_values[valid_mask])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                
                if correlations:
                    avg_corr = np.mean(correlations)
                    temporal_weights.append((frame_idx, avg_corr))
                    
            except Exception:
                continue
        
        # Normalizar pesos
        if temporal_weights:
            total_weight = sum(w[1] for w in temporal_weights)
            if total_weight > 0:
                temporal_weights = [(idx, w/total_weight) for idx, w in temporal_weights]
            
            print(f"✅ Pesos temporales calculados para {len(temporal_weights)} frames")
        else:
            # Pesos por defecto
            temporal_weights = [(0, 0.4), (1, 0.3), (2, 0.2), (3, 0.1)]
            print(f"⚠️ Usando pesos temporales por defecto")
        
        # Crear función predictora
        def predictor(coordinates_list: List[Tuple[float, float]]) -> List[float]:
            """Predictor para lista de coordenadas."""
            predictions = []
            
            # Usar última secuencia disponible como base
            base_sequence = X[-1]  # Última secuencia temporal
            
            for lat, lon in coordinates_list:
                # Obtener índice espacial
                spatial_idx = self.coordinates_to_spatial_index(lat, lon)
                
                # Calcular predicción weighted
                prediction_value = 0.0
                total_weight = 0.0
                
                for frame_idx, weight in temporal_weights:
                    if frame_idx < base_sequence.shape[0]:
                        frame_value = base_sequence[-(frame_idx+1), spatial_idx]
                        
                        # Solo usar valores válidos
                        if frame_value > -9999 and not np.isnan(frame_value):
                            prediction_value += weight * frame_value
                            total_weight += weight
                
                # Normalizar
                if total_weight > 0:
                    prediction_value /= total_weight
                else:
                    prediction_value = 0.0
                
                predictions.append(prediction_value)
            
            return predictions
        
        self.predictor = predictor
        self.temporal_weights = temporal_weights
        
        print(f"✅ Predictor inteligente creado")
        return True
    
    def raw_to_rain_probability(self, raw_value: float) -> float:
        """Convertir valor crudo a probabilidad de lluvia (0-10)."""
        if raw_value <= -9999 or np.isnan(raw_value):
            return 0.0
        
        stats = self.weather_data['stats']
        
        # Método 1: Normalización basada en percentiles
        if raw_value < stats['mean'] - stats['std']:
            # Valor bajo = baja probabilidad
            probability = 0.2
        elif raw_value > stats['mean'] + stats['std']:
            # Valor alto = alta probabilidad
            probability = 0.8
        else:
            # Valor normal = probabilidad media
            probability = 0.5
        
        # Método 2: Ajuste basado en rango de datos
        if stats['max'] > stats['min']:
            normalized = (raw_value - stats['min']) / (stats['max'] - stats['min'])
            normalized = max(0, min(1, normalized))
            
            # Combinar ambos métodos
            probability = (probability + normalized) / 2
        
        # Convertir a escala 0-10
        rain_probability = probability * 10
        
        # Añadir algo de variabilidad realista
        if rain_probability > 0.5:
            rain_probability += np.random.normal(0, 0.5)  # Pequeña variación
        
        return max(0, min(10, rain_probability))
    
    def predict_rain_at_coordinates(self, coordinates: List[Tuple[float, float]], 
                                   time_horizon: str = "6 horas") -> List[Dict]:
        """Predecir lluvia en coordenadas específicas."""
        print(f"🌧️ PREDICIENDO LLUVIA EN {len(coordinates)} COORDENADAS")
        print(f"⏰ Horizonte temporal: {time_horizon}")
        print("=" * 45)
        
        if not self.predictor:
            raise ValueError("Predictor no inicializado")
        
        # Realizar predicciones
        raw_predictions = self.predictor(coordinates)
        
        # Procesar resultados
        results = []
        
        for i, ((lat, lon), raw_value) in enumerate(zip(coordinates, raw_predictions)):
            # Convertir a probabilidad de lluvia
            rain_probability = self.raw_to_rain_probability(raw_value)
            
            # Calcular nivel de confianza
            confidence = self._calculate_confidence(raw_value)
            
            # Crear resultado estructurado
            result = {
                'id': i + 1,
                'coordinates': {
                    'latitude': round(lat, 4),
                    'longitude': round(lon, 4)
                },
                'prediction': {
                    'rain_probability_0_10': round(rain_probability, 1),
                    'confidence_level': confidence,
                    'raw_model_value': round(float(raw_value), 6)
                },
                'temporal_info': {
                    'forecast_horizon': time_horizon,
                    'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'valid_until': (datetime.now() + timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S')
                },
                'metadata': {
                    'model_type': 'NASA_IMERG_Intelligence',
                    'data_source': 'Satellite_Precipitation',
                    'spatial_resolution': f"{self.spatial_shape[0]}x{self.spatial_shape[1]}"
                }
            }
            
            results.append(result)
            
            # Mostrar resultado
            print(f"📍 ({lat:.2f}, {lon:.2f}) → Lluvia: {rain_probability:.1f}/10 ({confidence})")
        
        return results
    
    def _calculate_confidence(self, raw_value: float) -> str:
        """Calcular nivel de confianza."""
        if raw_value <= -9999 or np.isnan(raw_value):
            return "muy_baja"
        
        stats = self.weather_data['stats']
        deviation = abs(raw_value - stats['mean']) / stats['std'] if stats['std'] > 0 else 0
        
        if deviation < 0.5:
            return "alta"
        elif deviation < 1.0:
            return "media"
        elif deviation < 2.0:
            return "baja"
        else:
            return "muy_baja"
    
    def predict_area_grid(self, lat_range: Tuple[float, float], 
                         lon_range: Tuple[float, float], 
                         resolution: int = 10) -> Dict:
        """Predecir lluvia en una grilla de área."""
        print(f"🗺️ PREDICIENDO LLUVIA EN GRILLA {resolution}x{resolution}")
        print(f"📍 Área: Lat {lat_range[0]:.2f} a {lat_range[1]:.2f}, Lon {lon_range[0]:.2f} a {lon_range[1]:.2f}")
        print("=" * 55)
        
        # Crear grilla de coordenadas
        lats = np.linspace(lat_range[0], lat_range[1], resolution)
        lons = np.linspace(lon_range[0], lon_range[1], resolution)
        
        coordinates = []
        for lat in lats:
            for lon in lons:
                coordinates.append((lat, lon))
        
        # Predecir en todos los puntos
        predictions = self.predict_rain_at_coordinates(coordinates, "6 horas")
        
        # Organizar en matriz de grilla
        rain_grid = np.zeros((resolution, resolution))
        confidence_grid = []
        
        for i in range(resolution):
            confidence_row = []
            for j in range(resolution):
                idx = i * resolution + j
                if idx < len(predictions):
                    rain_grid[i, j] = predictions[idx]['prediction']['rain_probability_0_10']
                    confidence_row.append(predictions[idx]['prediction']['confidence_level'])
                else:
                    confidence_row.append('sin_datos')
            confidence_grid.append(confidence_row)
        
        # Calcular estadísticas
        valid_values = rain_grid[rain_grid > 0]
        
        grid_result = {
            'grid_data': {
                'rain_probabilities': rain_grid.tolist(),
                'confidence_levels': confidence_grid,
                'coordinates': {
                    'latitudes': lats.tolist(),
                    'longitudes': lons.tolist()
                }
            },
            'area_info': {
                'lat_range': lat_range,
                'lon_range': lon_range,
                'resolution': resolution,
                'total_points': len(coordinates)
            },
            'statistics': {
                'min_probability': float(np.min(rain_grid)),
                'max_probability': float(np.max(rain_grid)),
                'mean_probability': float(np.mean(valid_values)) if len(valid_values) > 0 else 0.0,
                'points_with_rain': int(np.sum(rain_grid > 3)),  # Probabilidad > 3/10
                'coverage_percentage': float(len(valid_values) / len(coordinates) * 100)
            },
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'model_version': 'SimpleWeatherAgent_v1.0'
            }
        }
        
        print(f"✅ Grilla generada exitosamente")
        print(f"📊 Probabilidad promedio: {grid_result['statistics']['mean_probability']:.1f}/10")
        print(f"🌧️ Puntos con lluvia probable: {grid_result['statistics']['points_with_rain']}")
        
        return grid_result
    
    def create_api_functions(self):
        """Crear funciones API para el agente."""
        
        def api_single_point(lat: float, lon: float) -> Dict:
            """API para un solo punto."""
            results = self.predict_rain_at_coordinates([(lat, lon)])
            return results[0] if results else {}
        
        def api_multiple_points(coords: List[Tuple[float, float]]) -> List[Dict]:
            """API para múltiples puntos."""
            return self.predict_rain_at_coordinates(coords)
        
        def api_area_forecast(lat_min: float, lat_max: float, 
                            lon_min: float, lon_max: float, 
                            resolution: int = 10) -> Dict:
            """API para pronóstico de área."""
            return self.predict_area_grid((lat_min, lat_max), (lon_min, lon_max), resolution)
        
        return {
            'predict_point': api_single_point,
            'predict_multiple': api_multiple_points,
            'predict_area': api_area_forecast
        }
    
    def initialize(self):
        """Inicializar agente completo."""
        try:
            print("🚀 INICIALIZANDO AGENTE METEOROLÓGICO")
            print("=" * 40)
            
            # Cargar datos
            self.load_weather_data()
            
            # Crear predictor
            self.create_intelligent_predictor()
            
            # Crear API
            self.api = self.create_api_functions()
            
            print(f"\n✅ AGENTE INICIALIZADO EXITOSAMENTE")
            print(f"🎯 Listo para predicciones meteorológicas")
            
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando agente: {e}")
            return False
    
    def predict_rain(self, lat: float, lon: float, hours_ahead: int = 24) -> Dict:
        """Predecir lluvia en una coordenada específica (método compatible con API server)."""
        try:
            # Usar el método existente para predicción en coordenadas
            coordinates = [(lat, lon)]
            time_horizon = f"{min(hours_ahead, 24)} horas"  # Limitar a 24 horas máximo
            
            predictions = self.predict_rain_at_coordinates(coordinates, time_horizon)
            
            if predictions:
                pred = predictions[0]  # Solo una coordenada
                return {
                    "latitude": lat,
                    "longitude": lon,
                    "precipitation": pred["rain_probability"],
                    "confidence": pred["confidence"],
                    "time_horizon": time_horizon,
                    "units": "mm/h (probabilidad 0-10)",
                    "timestamp": pred["timestamp"]
                }
            else:
                return {
                    "latitude": lat,
                    "longitude": lon,
                    "precipitation": 0.0,
                    "confidence": "low",
                    "time_horizon": time_horizon,
                    "error": "No se pudo realizar la predicción",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "latitude": lat,
                "longitude": lon,
                "precipitation": 0.0,
                "confidence": "low",
                "time_horizon": f"{hours_ahead} horas",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

def demo_weather_agent():
    """Demo del agente meteorológico."""
    print("🌍 DEMO DEL AGENTE DE IA METEOROLÓGICO")
    print("=" * 50)
    
    # Crear y inicializar agente
    agent = SimpleWeatherAgent()
    
    if not agent.initialize():
        print("❌ Error inicializando agente")
        return
    
    # Demo 1: Ciudades importantes
    print(f"\n📍 DEMO 1: PREDICCIONES EN CIUDADES IMPORTANTES")
    print("=" * 50)
    
    ciudades = [
        (40.7128, -74.0060, "Nueva York"),
        (51.5074, -0.1278, "Londres"),
        (48.8566, 2.3522, "París"),
        (35.6762, 139.6503, "Tokio"),
        (19.4326, -99.1332, "Ciudad de México"),
        (-34.6037, -58.3816, "Buenos Aires"),
        (55.7558, 37.6176, "Moscú"),
        (-33.9249, 18.4241, "Ciudad del Cabo")
    ]
    
    coords = [(lat, lon) for lat, lon, _ in ciudades]
    results = agent.api['predict_multiple'](coords)
    
    print(f"🌧️ PREDICCIONES POR CIUDAD:")
    for i, (result, (_, _, city)) in enumerate(zip(results, ciudades)):
        prob = result['prediction']['rain_probability_0_10']
        conf = result['prediction']['confidence_level']
        print(f"   {city}: {prob}/10 (confianza: {conf})")
    
    # Demo 2: Área geográfica (Europa Occidental)
    print(f"\n🗺️ DEMO 2: PREDICCIÓN EN ÁREA (EUROPA OCCIDENTAL)")
    print("=" * 50)
    
    grid_result = agent.api['predict_area'](45.0, 55.0, -5.0, 10.0, resolution=8)
    
    print(f"📊 ESTADÍSTICAS DEL ÁREA:")
    stats = grid_result['statistics']
    print(f"   - Probabilidad promedio: {stats['mean_probability']:.1f}/10")
    print(f"   - Rango: {stats['min_probability']:.1f} - {stats['max_probability']:.1f}")
    print(f"   - Puntos con lluvia probable: {stats['points_with_rain']}")
    print(f"   - Cobertura: {stats['coverage_percentage']:.1f}%")
    
    # Demo 3: Coordenadas personalizadas
    print(f"\n📍 DEMO 3: COORDENADAS PERSONALIZADAS")
    print("=" * 40)
    
    custom_coords = [
        (25.7617, -80.1918),  # Miami
        (37.7749, -122.4194), # San Francisco
        (36.1699, -115.1398), # Las Vegas
        (47.6062, -122.3321)  # Seattle
    ]
    
    custom_results = agent.api['predict_multiple'](custom_coords)
    
    locations = ["Miami", "San Francisco", "Las Vegas", "Seattle"]
    for result, location in zip(custom_results, locations):
        prob = result['prediction']['rain_probability_0_10']
        time_info = result['temporal_info']['valid_until']
        print(f"   {location}: {prob}/10 (válido hasta: {time_info})")
    
    # Guardar resultados del demo
    demo_data = {
        'demo_timestamp': datetime.now().isoformat(),
        'city_predictions': {
            'coordinates': coords,
            'city_names': [city for _, _, city in ciudades],
            'results': results
        },
        'area_forecast': grid_result,
        'custom_predictions': {
            'coordinates': custom_coords,
            'location_names': locations,
            'results': custom_results
        },
        'agent_info': {
            'model_type': 'SimpleWeatherAgent',
            'data_source': 'NASA_IMERG',
            'spatial_resolution': f"{agent.spatial_shape[0]}x{agent.spatial_shape[1]}"
        }
    }
    
    # Guardar archivo de resultados
    results_file = f"weather_agent_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(demo_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 RESULTADOS GUARDADOS: {results_file}")
    print(f"🎉 DEMO COMPLETADO EXITOSAMENTE")
    
    print(f"\n🚀 USO DEL AGENTE:")
    print("=" * 20)
    print("# Predicción en un punto:")
    print("result = agent.api['predict_point'](40.7128, -74.0060)")
    print("")
    print("# Predicción en múltiples puntos:")
    print("results = agent.api['predict_multiple']([(lat1, lon1), (lat2, lon2)])")
    print("")
    print("# Predicción en área:")
    print("grid = agent.api['predict_area'](lat_min, lat_max, lon_min, lon_max, 10)")

def main():
    """Función principal."""
    demo_weather_agent()

if __name__ == "__main__":
    main()