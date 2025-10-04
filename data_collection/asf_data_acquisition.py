#!/usr/bin/env python3
"""
Script para adquisición automática de datos SLC de Sentinel-1 via ASF Search API
y generación de interferogramas via HyP3 On-Demand service.
"""

import requests
import json
import datetime
import time
from typing import List, Dict, Optional
import hyp3_sdk as hyp3

# Configuración de la API
ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"

# Áreas de interés (ejemplos: Cinturón de Fuego del Pacífico, falla de Anatolia)
# Formato: [min_lon, min_lat, max_lon, max_lat]
AREAS_INTERES = {
    "cinturon_fuego_pacifico": [-180, -60, 180, 60],  # Aproximado
    "falla_anatolia": [25, 35, 45, 42],  # Aproximado
    # Agregar más áreas según necesidad
}

class ASFAcquisition:
    def __init__(self, hyp3_username: Optional[str] = None, hyp3_password: Optional[str] = None):
        self.session = requests.Session()
        self.hyp3_client = None
        if hyp3_username and hyp3_password:
            self.hyp3_client = hyp3.HyP3(username=hyp3_username, password=hyp3_password)

    def buscar_scenes_slc(self,
                         area: str,
                         start_date: str,
                         end_date: str,
                         max_results: int = 100) -> List[Dict]:
        """
        Busca escenas SLC de Sentinel-1 en un área y período específicos.

        Args:
            area: Nombre del área de interés (clave en AREAS_INTERES)
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            max_results: Máximo número de resultados

        Returns:
            Lista de escenas encontradas
        """
        if area not in AREAS_INTERES:
            raise ValueError(f"Área '{area}' no definida en AREAS_INTERES")

        bbox = AREAS_INTERES[area]

        params = {
            'platform': 'S1',
            'processingLevel': 'SLC',
            'bbox': ','.join(map(str, bbox)),
            'start': start_date,
            'end': end_date,
            'maxResults': max_results,
            'output': 'json'
        }

        try:
            response = self.session.get(ASF_SEARCH_URL, params=params)
            response.raise_for_status()

            data = response.json()
            # La API puede devolver una lista anidada
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                return data[0]
            else:
                return data if isinstance(data, list) else []

        except requests.RequestException as e:
            print(f"Error en la búsqueda: {e}")
            return []

    def buscar_nuevas_scenes(self,
                           area: str,
                           dias_atras: int = 7) -> List[Dict]:
        """
        Busca escenas nuevas disponibles en los últimos días.

        Args:
            area: Nombre del área de interés
            dias_atras: Número de días hacia atrás para buscar

        Returns:
            Lista de nuevas escenas
        """
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=dias_atras)).strftime('%Y-%m-%d')

        return self.buscar_scenes_slc(area, start_date, end_date)

    def filtrar_por_orbita(self, scenes: List[Dict], orbita: str) -> List[Dict]:
        """
        Filtra escenas por tipo de órbita (ASCENDING/DESCENDING).

        Args:
            scenes: Lista de escenas
            orbita: Tipo de órbita

        Returns:
            Escenas filtradas
        """
        return [scene for scene in scenes if scene.get('flightDirection', '').upper() == orbita.upper()]

    def generar_interferogramas(self, area: str, nuevas_scenes: List[Dict]) -> None:
        """
        Genera interferogramas usando HyP3 para las nuevas escenas.

        Args:
            area: Nombre del área
            nuevas_scenes: Lista de nuevas escenas SLC
        """
        if not self.hyp3_client:
            print("HyP3 client no configurado. Omitiendo generación de interferogramas.")
            return

        # Para cada nueva escena, buscar una escena de referencia anterior
        for nueva_scene in nuevas_scenes:
            referencia_scene = self._encontrar_escena_referencia(area, nueva_scene)
            if referencia_scene:
                self._solicitar_insar(referencia_scene, nueva_scene, area)
            else:
                print(f"No se encontró escena de referencia para {nueva_scene.get('sceneName', 'N/A')}")

    def _encontrar_escena_referencia(self, area: str, nueva_scene: Dict) -> Optional[Dict]:
        """
        Encuentra una escena de referencia anterior para generar interferograma.

        Args:
            area: Nombre del área
            nueva_scene: Nueva escena SLC

        Returns:
            Escena de referencia o None
        """
        # Buscar escenas anteriores en la misma órbita relativa
        relative_orbit = nueva_scene.get('relativeOrbit')
        flight_direction = nueva_scene.get('flightDirection')

        # Buscar escenas de 6-24 días atrás (temporal baseline típico)
        fecha_nueva = datetime.datetime.fromisoformat(nueva_scene.get('startTime', '').replace('Z', '+00:00'))
        fecha_referencia_inicio = fecha_nueva - datetime.timedelta(days=24)
        fecha_referencia_fin = fecha_nueva - datetime.timedelta(days=6)

        escenas_anteriores = self.buscar_scenes_slc(
            area,
            fecha_referencia_inicio.strftime('%Y-%m-%d'),
            fecha_referencia_fin.strftime('%Y-%m-%d'),
            max_results=50
        )

        # Filtrar por órbita relativa y dirección
        candidatos = [
            scene for scene in escenas_anteriores
            if scene.get('relativeOrbit') == relative_orbit and
               scene.get('flightDirection') == flight_direction
        ]

        # Retornar la más reciente si existe
        if candidatos:
            return max(candidatos, key=lambda x: x.get('startTime', ''))

        return None

    def _solicitar_insar(self, referencia: Dict, secundaria: Dict, area: str) -> None:
        """
        Solicita procesamiento InSAR via HyP3.

        Args:
            referencia: Escena de referencia
            secundaria: Escena secundaria
            area: Nombre del área
        """
        try:
            job_name = f"InSAR_{area}_{referencia.get('sceneName', '')[:20]}_{secundaria.get('sceneName', '')[:20]}"

            # Crear job de InSAR
            insar_job = hyp3.InSARJob(
                reference_granule=referencia.get('granuleName', ''),
                secondary_granule=secundaria.get('granuleName', ''),
                name=job_name,
                include_dem=True,
                include_look_vectors=True
            )

            # Enviar job
            submitted_jobs = self.hyp3_client.submit_job(insar_job)
            print(f"Job InSAR enviado: {job_name}")

        except Exception as e:
            print(f"Error al enviar job InSAR: {e}")

def main():
    """Función principal para ejecutar la adquisición automática."""
    # Configurar credenciales HyP3 (deberían venir de variables de entorno o config)
    import os
    username = os.getenv('HYP3_USERNAME')
    password = os.getenv('HYP3_PASSWORD')

    acquirer = ASFAcquisition(hyp3_username=username, hyp3_password=password)

    # Buscar nuevas escenas en todas las áreas definidas
    for area in AREAS_INTERES.keys():
        print(f"Buscando nuevas escenas SLC en {area}...")

        nuevas_scenes = acquirer.buscar_nuevas_scenes(area, dias_atras=7)

        if nuevas_scenes:
            print(f"Encontradas {len(nuevas_scenes)} nuevas escenas en {area}")

            # Generar interferogramas si HyP3 está configurado
            acquirer.generar_interferogramas(area, nuevas_scenes)

            # Mostrar información básica
            for scene in nuevas_scenes[:3]:  # Mostrar primeras 3
                print(f"  - {scene.get('granuleName', 'N/A')}: {scene.get('startTime', 'N/A')}")
        else:
            print(f"No se encontraron nuevas escenas en {area}")

        time.sleep(1)  # Pequeña pausa entre búsquedas

if __name__ == "__main__":
    main()

import requests
import json
import datetime
import time
from typing import List, Dict, Optional

# Configuración de la API
ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"

# Áreas de interés (ejemplos: Cinturón de Fuego del Pacífico, falla de Anatolia)
# Formato: [min_lon, min_lat, max_lon, max_lat]
AREAS_INTERES = {
    "cinturon_fuego_pacifico": [-180, -60, 180, 60],  # Aproximado
    "falla_anatolia": [25, 35, 45, 42],  # Aproximado
    # Agregar más áreas según necesidad
}

class ASFAcquisition:
    def __init__(self):
        self.session = requests.Session()

    def buscar_scenes_slc(self,
                         area: str,
                         start_date: str,
                         end_date: str,
                         max_results: int = 100) -> List[Dict]:
        """
        Busca escenas SLC de Sentinel-1 en un área y período específicos.

        Args:
            area: Nombre del área de interés (clave en AREAS_INTERES)
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            max_results: Máximo número de resultados

        Returns:
            Lista de escenas encontradas
        """
        if area not in AREAS_INTERES:
            raise ValueError(f"Área '{area}' no definida en AREAS_INTERES")

        bbox = AREAS_INTERES[area]

        params = {
            'platform': 'S1',
            'processingLevel': 'SLC',
            'bbox': ','.join(map(str, bbox)),
            'start': start_date,
            'end': end_date,
            'maxResults': max_results,
            'output': 'json'
        }

        try:
            response = self.session.get(ASF_SEARCH_URL, params=params)
            response.raise_for_status()

            data = response.json()
            # La API puede devolver una lista anidada
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                return data[0]
            else:
                return data if isinstance(data, list) else []

        except requests.RequestException as e:
            print(f"Error en la búsqueda: {e}")
            return []

    def buscar_nuevas_scenes(self,
                           area: str,
                           dias_atras: int = 7) -> List[Dict]:
        """
        Busca escenas nuevas disponibles en los últimos días.

        Args:
            area: Nombre del área de interés
            dias_atras: Número de días hacia atrás para buscar

        Returns:
            Lista de nuevas escenas
        """
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=dias_atras)).strftime('%Y-%m-%d')

        return self.buscar_scenes_slc(area, start_date, end_date)

    def filtrar_por_orbita(self, scenes: List[Dict], orbita: str) -> List[Dict]:
        """
        Filtra escenas por tipo de órbita (ASCENDING/DESCENDING).

        Args:
            scenes: Lista de escenas
            orbita: Tipo de órbita

        Returns:
            Escenas filtradas
        """
        return [scene for scene in scenes if scene.get('orbitDirection', '').upper() == orbita.upper()]

def main():
    """Función principal para ejecutar la adquisición automática."""
    acquirer = ASFAcquisition()

    # Buscar nuevas escenas en todas las áreas definidas
    for area in AREAS_INTERES.keys():
        print(f"Buscando nuevas escenas SLC en {area}...")

        nuevas_scenes = acquirer.buscar_nuevas_scenes(area, dias_atras=7)

        if nuevas_scenes:
            print(f"Encontradas {len(nuevas_scenes)} nuevas escenas en {area}")

            # Aquí se podría agregar lógica para procesar las escenas,
            # como descargarlas o enviar a On-Demand service

            # Por ahora, solo imprimir información básica
            for scene in nuevas_scenes[:5]:  # Mostrar primeras 5
                print(f"  - {scene.get('sceneName', 'N/A')}: {scene.get('startTime', 'N/A')}")
        else:
            print(f"No se encontraron nuevas escenas en {area}")

        time.sleep(1)  # Pequeña pausa entre búsquedas

if __name__ == "__main__":
    main()