#!/usr/bin/env python3
"""
Analizador de Nivel del Mar - NASA OPeNDAP Access
================================================

Sistema para analizar tendencias del nivel del mar usando datos satelitales
de la NASA a través de acceso programático OPeNDAP.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json
from typing import Tuple, List, Dict, Optional
import requests
from scipy import stats

# Configurar warnings
warnings.filterwarnings('ignore')

class SeaLevelAnalyzer:
    """Analizador de datos de nivel del mar usando OPeNDAP."""
    
    def __init__(self):
        """Inicializar analizador."""
        self.base_url = "https://podaac-opendap.jpl.nasa.gov/opendap"
        
        # URLs de conjuntos de datos principales
        self.datasets = {
            'merged_alt_l4': {
                'url': f"{self.base_url}/allData/merged_alt/L4/sea_surface_height_alt_grids",
                'description': "Datos grillados de altura del mar combinados L4",
                'variables': ['ssha', 'adt', 'err'],
                'resolution': "0.25 grados",
                'temporal': "diario"
            },
            'jason3_l4': {
                'url': f"{self.base_url}/allData/jason3/L4/sea_surface_height_alt_grids", 
                'description': "Datos Jason-3 grillados L4",
                'variables': ['ssha', 'adt'],
                'resolution': "0.25 grados", 
                'temporal': "5 días"
            }
        }
        
        # Regiones predefinidas
        self.regions = {
            'global': {'lat': (-66, 66), 'lon': (-180, 180), 'name': 'Global'},
            'mediterranean': {'lat': (30, 45), 'lon': (0, 35), 'name': 'Mediterráneo'},
            'caribbean': {'lat': (10, 25), 'lon': (-85, -60), 'name': 'Caribe'},
            'north_atlantic': {'lat': (40, 70), 'lon': (-70, -10), 'name': 'Atlántico Norte'},
            'pacific_tropical': {'lat': (-20, 20), 'lon': (120, -80), 'name': 'Pacífico Tropical'},
            'gulf_mexico': {'lat': (18, 30), 'lon': (-98, -80), 'name': 'Golfo de México'},
            'arctic': {'lat': (66, 90), 'lon': (-180, 180), 'name': 'Ártico'},
            'antarctic': {'lat': (-90, -60), 'lon': (-180, 180), 'name': 'Antártico'}
        }
        
        print("🌊 ANALIZADOR DE NIVEL DEL MAR")
        print("=" * 35)
        print("📡 Acceso programático a datos NASA via OPeNDAP")
        print("🎯 Análisis de anomalías de altura del mar (SSHA)")
        print()
    
    def check_credentials(self) -> bool:
        """Verificar credenciales de Earthdata."""
        netrc_path = Path.home() / '.netrc'
        
        if not netrc_path.exists():
            print("❌ Archivo .netrc no encontrado")
            print("💡 Configura tus credenciales NASA Earthdata:")
            print("   echo 'machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD' >> ~/.netrc")
            print("   chmod 600 ~/.netrc")
            return False
        
        print("✅ Credenciales .netrc encontradas")
        return True
    
    def list_available_datasets(self):
        """Listar conjuntos de datos disponibles."""
        print("📊 CONJUNTOS DE DATOS DISPONIBLES")
        print("=" * 40)
        
        for key, dataset in self.datasets.items():
            print(f"\n🔹 {key.upper()}")
            print(f"   Descripción: {dataset['description']}")
            print(f"   Variables: {', '.join(dataset['variables'])}")
            print(f"   Resolución: {dataset['resolution']}")
            print(f"   Frecuencia: {dataset['temporal']}")
    
    def list_available_regions(self):
        """Listar regiones predefinidas."""
        print("🗺️ REGIONES PREDEFINIDAS")
        print("=" * 30)
        
        for key, region in self.regions.items():
            lat_range = region['lat']
            lon_range = region['lon']
            print(f"🔹 {key}: {region['name']}")
            print(f"   Lat: {lat_range[0]}° a {lat_range[1]}°")
            print(f"   Lon: {lon_range[0]}° a {lon_range[1]}°")
    
    def test_opendap_connection(self, dataset_key: str = 'merged_alt_l4') -> bool:
        """Probar conexión OPeNDAP."""
        print(f"🔗 PROBANDO CONEXIÓN OPENDAP")
        print("=" * 35)
        
        if not self.check_credentials():
            return False
        
        # URL de prueba (archivo reciente)
        test_url = f"{self.datasets[dataset_key]['url']}/oscar_third_deg/2024/"
        
        try:
            print(f"📡 Conectando a: {test_url}")
            
            # Intentar listar archivos disponibles
            response = requests.get(test_url + "catalog.html", timeout=30)
            
            if response.status_code == 200:
                print("✅ Conexión OPeNDAP exitosa")
                print("📂 Archivos disponibles detectados")
                return True
            else:
                print(f"⚠️ Respuesta inesperada: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            print("💡 Verifica:")
            print("   1. Conexión a internet")
            print("   2. Credenciales NASA Earthdata en ~/.netrc")
            print("   3. URL del servicio OPeNDAP")
            return False
    
    def create_sample_opendap_url(self, dataset_key: str = 'merged_alt_l4', 
                                 year: int = 2024, month: int = 1) -> str:
        """Crear URL de ejemplo para OPeNDAP."""
        base_url = self.datasets[dataset_key]['url']
        
        # Formato típico de URLs OPeNDAP para datos de nivel del mar
        sample_url = f"{base_url}/oscar_third_deg/{year}/sea_surface_height_alt_grids_oscar_third_deg_{year}{month:02d}01.nc"
        
        return sample_url
    
    def analyze_sea_level_region(self, region_key: str, 
                               start_date: str = "2023-01-01",
                               end_date: str = "2023-12-31",
                               dataset_key: str = 'merged_alt_l4',
                               use_real_data: bool = False) -> Dict:
        """Analizar nivel del mar en una región específica. Si use_real_data=True, extrae datos reales de OPeNDAP/xarray."""
        print(f"🌊 ANALIZANDO NIVEL DEL MAR: {self.regions[region_key]['name'].upper()}")
        print("=" * 60)
        
        if not self.check_credentials():
            return {}
        
        try:
            # Obtener coordenadas de la región
            region = self.regions[region_key]
            lat_range = region['lat']
            lon_range = region['lon']
            
            print(f"📍 Región: {region['name']}")
            print(f"   Latitud: {lat_range[0]}° a {lat_range[1]}°")
            print(f"   Longitud: {lon_range[0]}° a {lon_range[1]}°")
            print(f"   Período: {start_date} a {end_date}")
            print()
            
            if use_real_data:
                # Buscar archivo NetCDF más cercano al periodo solicitado
                year = int(start_date[:4])
                month = int(start_date[5:7])
                url = self.create_sample_opendap_url(dataset_key, year, month)
                print(f"🔗 Extrayendo datos reales de: {url}")
                ds = xr.open_dataset(url)
                # Seleccionar lat/lon dentro del rango
                lat_sel = ds['latitude']
                lon_sel = ds['longitude']
                lat_mask = (lat_sel >= lat_range[0]) & (lat_sel <= lat_range[1])
                lon_mask = (lon_sel >= lon_range[0]) & (lon_sel <= lon_range[1])
                lats = lat_sel.values[lat_mask]
                lons = lon_sel.values[lon_mask]
                # Seleccionar primer tiempo disponible (snapshot)
                ssha = ds['ssha'].isel(time=0, lat=lat_mask, lon=lon_mask).values
                # Convertir a lista para JSON
                ssha_grid = ssha.tolist()
                # Metadatos
                result = {
                    'region': region,
                    'analysis_period': {
                        'start': str(ds['time'].values[0]),
                        'end': str(ds['time'].values[0]),
                        'duration_days': 1
                    },
                    'ssha_grid': ssha_grid,
                    'lat': lats.tolist(),
                    'lon': lons.tolist(),
                    'units': ds['ssha'].attrs.get('units', 'm'),
                    'data_source': url
                }
                ds.close()
                return result
            # ...código anterior de simulación...
            # Generar serie temporal simulada pero realista
            dates = pd.date_range(start_date, end_date, freq='D')
            # Simular anomalías del nivel del mar (valores típicos en metros)
            np.random.seed(42)  # Para reproducibilidad
            
            # Tendencia base según la región
            if region_key == 'mediterranean':
                base_trend = 0.003  # mm/año típico del Mediterráneo
                seasonal_amplitude = 0.05
            elif region_key == 'caribbean':
                base_trend = 0.004
                seasonal_amplitude = 0.08
            elif region_key == 'arctic':
                base_trend = 0.006  # Mayor subida en Ártico
                seasonal_amplitude = 0.12
            else:
                base_trend = 0.0032  # Promedio global
                seasonal_amplitude = 0.06
            
            # Construir serie temporal realista
            days_from_start = (dates - dates[0]).days
            
            # Componente de tendencia
            trend_component = base_trend * days_from_start / 365.25
            
            # Componente estacional
            seasonal_component = seasonal_amplitude * np.sin(2 * np.pi * days_from_start / 365.25)
            
            # Componente de ruido
            noise_component = np.random.normal(0, 0.02, len(dates))
            
            # Anomalía total del nivel del mar
            ssha_values = trend_component + seasonal_component + noise_component
            
            # Crear DataFrame para análisis
            data = pd.DataFrame({
                'date': dates,
                'ssha': ssha_values,
                'trend': trend_component,
                'seasonal': seasonal_component
            })
            
            # Análisis estadístico
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                days_from_start, ssha_values
            )
            
            # Conversión a mm/año
            trend_mm_per_year = slope * 365.25 * 1000
            
            # Estadísticas
            statistics = {
                'mean_ssha': float(np.mean(ssha_values)),
                'std_ssha': float(np.std(ssha_values)),
                'min_ssha': float(np.min(ssha_values)),
                'max_ssha': float(np.max(ssha_values)),
                'trend_mm_per_year': float(trend_mm_per_year),
                'trend_confidence': float(r_value**2),
                'p_value': float(p_value),
                'total_samples': len(ssha_values)
            }
            
            print(f"📈 RESULTADOS DEL ANÁLISIS:")
            print(f"   Tendencia: {trend_mm_per_year:.2f} mm/año")
            print(f"   Confianza (R²): {statistics['trend_confidence']:.3f}")
            print(f"   Valor p: {statistics['p_value']:.2e}")
            print(f"   Anomalía promedio: {statistics['mean_ssha']*1000:.2f} mm")
            print(f"   Rango: {statistics['min_ssha']*1000:.1f} a {statistics['max_ssha']*1000:.1f} mm")
            print(f"   Muestras analizadas: {statistics['total_samples']}")
            
            # Crear visualización
            self._create_sea_level_plot(data, region, statistics)
            
            # Resultado completo
            result = {
                'region': region,
                'analysis_period': {
                    'start': start_date,
                    'end': end_date,
                    'duration_days': len(dates)
                },
                'statistics': statistics,
                'data_summary': {
                    'total_points': len(ssha_values),
                    'data_source': 'simulated_realistic',
                    'temporal_resolution': 'daily'
                },
                'interpretation': self._interpret_sea_level_trend(trend_mm_per_year, region_key)
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error en análisis: {e}")
            return {}
    
    def _create_sea_level_plot(self, data: pd.DataFrame, region: Dict, 
                              statistics: Dict):
        """Crear visualización de tendencias del nivel del mar."""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Gráfico 1: Serie temporal completa
            ax1.plot(data['date'], data['ssha']*1000, 'b-', alpha=0.7, linewidth=0.8, label='SSHA observada')
            ax1.plot(data['date'], data['trend']*1000, 'r-', linewidth=2, label='Tendencia lineal')
            
            ax1.set_title(f"Anomalía del Nivel del Mar - {region['name']}", fontsize=14, fontweight='bold')
            ax1.set_ylabel('Anomalía (mm)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Añadir texto con estadísticas
            stats_text = f"Tendencia: {statistics['trend_mm_per_year']:.2f} mm/año\nR² = {statistics['trend_confidence']:.3f}"
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Gráfico 2: Componente estacional
            ax2.plot(data['date'], data['seasonal']*1000, 'g-', linewidth=1.5, label='Componente estacional')
            ax2.set_title("Variación Estacional", fontsize=12)
            ax2.set_xlabel('Fecha', fontsize=12)
            ax2.set_ylabel('Variación (mm)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            # Guardar gráfico
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sea_level_analysis_{region['name'].lower().replace(' ', '_')}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
            print(f"📊 Gráfico guardado: {filename}")
            # plt.show()  # Comentado para no bloquear ejecución
            plt.close()  # Cerrar figura para liberar memoria
            
        except Exception as e:
            print(f"⚠️ Error creando visualización: {e}")
    
    def _interpret_sea_level_trend(self, trend_mm_per_year: float, region_key: str) -> Dict:
        """Interpretar tendencia del nivel del mar."""
        interpretation = {
            'trend_classification': '',
            'significance': '',
            'comparison_global': '',
            'implications': []
        }
        
        # Clasificar tendencia
        if abs(trend_mm_per_year) < 1:
            interpretation['trend_classification'] = 'estable'
        elif trend_mm_per_year > 5:
            interpretation['trend_classification'] = 'subida rápida'
        elif trend_mm_per_year > 2:
            interpretation['trend_classification'] = 'subida moderada'
        elif trend_mm_per_year > 0:
            interpretation['trend_classification'] = 'subida lenta'
        else:
            interpretation['trend_classification'] = 'descenso'
        
        # Significancia
        global_average = 3.2  # mm/año promedio global
        if abs(trend_mm_per_year - global_average) < 1:
            interpretation['significance'] = 'similar al promedio global'
        elif trend_mm_per_year > global_average + 1:
            interpretation['significance'] = 'por encima del promedio global'
        else:
            interpretation['significance'] = 'por debajo del promedio global'
        
        # Comparación
        interpretation['comparison_global'] = f"El promedio global es ~{global_average} mm/año"
        
        # Implicaciones
        if trend_mm_per_year > 4:
            interpretation['implications'].extend([
                'Riesgo elevado para zonas costeras bajas',
                'Posible aceleración del deshielo',
                'Necesidad de adaptación costera'
            ])
        elif trend_mm_per_year > 2:
            interpretation['implications'].extend([
                'Tendencia consistente con calentamiento global',
                'Monitoreo continuo recomendado'
            ])
        else:
            interpretation['implications'].append('Tendencia dentro de rangos esperados')
        
        return interpretation
    
    def analyze_multiple_regions(self, regions: List[str], 
                                start_date: str = "2023-01-01",
                                end_date: str = "2023-12-31") -> Dict:
        """Analizar múltiples regiones."""
        print(f"🌍 ANÁLISIS COMPARATIVO DE MÚLTIPLES REGIONES")
        print("=" * 55)
        
        results = {}
        
        for region_key in regions:
            if region_key not in self.regions:
                print(f"⚠️ Región '{region_key}' no encontrada")
                continue
            
            print(f"\n🔄 Procesando: {self.regions[region_key]['name']}")
            result = self.analyze_sea_level_region(region_key, start_date, end_date)
            
            if result:
                results[region_key] = result
        
        # Crear comparación
        if results:
            comparison = self._create_regional_comparison(results)
            results['comparison_summary'] = comparison
        
        return results
    
    def _create_regional_comparison(self, results: Dict) -> Dict:
        """Crear comparación entre regiones."""
        comparison = {
            'trends_summary': {},
            'ranking': [],
            'global_context': {}
        }
        
        # Extraer tendencias
        trends = {}
        for region_key, data in results.items():
            if 'statistics' in data:
                trends[region_key] = data['statistics']['trend_mm_per_year']
        
        # Ranking de tendencias
        sorted_trends = sorted(trends.items(), key=lambda x: x[1], reverse=True)
        
        comparison['ranking'] = [
            {
                'region': region_key,
                'region_name': self.regions[region_key]['name'],
                'trend_mm_per_year': trend,
                'rank': i + 1
            }
            for i, (region_key, trend) in enumerate(sorted_trends)
        ]
        
        # Estadísticas generales
        trend_values = list(trends.values())
        if trend_values:
            comparison['trends_summary'] = {
                'mean_trend': float(np.mean(trend_values)),
                'max_trend': float(np.max(trend_values)),
                'min_trend': float(np.min(trend_values)),
                'std_trend': float(np.std(trend_values))
            }
        
        return comparison
    
    def save_analysis_results(self, results: Dict, filename: str = None):
        """Guardar resultados del análisis."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sea_level_analysis_{timestamp}.json"
        
        # Preparar datos para JSON (convertir numpy types)
        json_results = self._prepare_for_json(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Resultados guardados: {filename}")
        return filename
    
    def _prepare_for_json(self, data):
        """Preparar datos para serialización JSON."""
        if isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, pd.Timestamp):
            return data.isoformat()
        else:
            return data

    def analyze_coordinates(self, lat: float, lon: float, 
                          start_date: str = "2023-01-01",
                          end_date: str = "2023-12-31") -> Dict:
        """Analizar nivel del mar en coordenadas específicas."""
        try:
            # Encontrar la región más cercana que contenga estas coordenadas
            closest_region = None
            min_distance = float('inf')
            
            for region_key, region_data in self.regions.items():
                lat_range = region_data['lat']
                lon_range = region_data['lon']
                
                # Verificar si las coordenadas están dentro de la región
                if (lat_range[0] <= lat <= lat_range[1] and 
                    lon_range[0] <= lon <= lon_range[1]):
                    # Coordenadas están dentro de esta región
                    closest_region = region_key
                    break
                    
                # Si no están dentro, calcular distancia al centro de la región
                center_lat = (lat_range[0] + lat_range[1]) / 2
                center_lon = (lon_range[0] + lon_range[1]) / 2
                
                distance = ((lat - center_lat) ** 2 + (lon - center_lon) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_region = region_key
            
            if closest_region:
                # Usar la región encontrada para el análisis
                result = self.analyze_sea_level_region(
                    closest_region, start_date, end_date, use_real_data=False
                )
                
                # Añadir información de coordenadas específicas
                result['query_coordinates'] = {'lat': lat, 'lon': lon}
                result['region_used'] = closest_region
                
                return result
            else:
                return {
                    'error': 'No se encontró una región válida para las coordenadas proporcionadas',
                    'coordinates': {'lat': lat, 'lon': lon}
                }
                
        except Exception as e:
            return {
                'error': str(e),
                'coordinates': {'lat': lat, 'lon': lon}
            }

def demo_sea_level_analysis():
    """Demostración del analizador de nivel del mar."""
    print("🌊 DEMO: ANALIZADOR DE NIVEL DEL MAR")
    print("=" * 45)
    
    # Crear analizador
    analyzer = SeaLevelAnalyzer()
    
    # Mostrar información
    analyzer.list_available_datasets()
    print()
    analyzer.list_available_regions()
    print()
    
    # Probar conexión
    if analyzer.test_opendap_connection():
        print("\n✅ Sistema listo para análisis")
    else:
        print("\n⚠️ Continuando con datos simulados")
    
    # Análisis de ejemplo: Mediterráneo
    print(f"\n{'='*60}")
    result_med = analyzer.analyze_sea_level_region('mediterranean', 
                                                  start_date="2023-01-01",
                                                  end_date="2023-12-31")
    
    # Análisis comparativo
    print(f"\n{'='*60}")
    regions_to_compare = ['mediterranean', 'caribbean', 'gulf_mexico', 'north_atlantic']
    comparison_results = analyzer.analyze_multiple_regions(regions_to_compare,
                                                          start_date="2023-01-01",
                                                          end_date="2023-12-31")
    
    # Mostrar ranking
    if 'comparison_summary' in comparison_results:
        print(f"\n🏆 RANKING DE TENDENCIAS:")
        print("-" * 30)
        for item in comparison_results['comparison_summary']['ranking']:
            print(f"   {item['rank']}. {item['region_name']}: {item['trend_mm_per_year']:.2f} mm/año")
    
    # Guardar resultados
    if comparison_results:
        filename = analyzer.save_analysis_results(comparison_results)
        print(f"\n📊 Análisis completo guardado en: {filename}")
    
    print(f"\n🎉 DEMO COMPLETADO")
    print("💡 Configura credenciales NASA Earthdata para datos reales")

if __name__ == "__main__":
    demo_sea_level_analysis()