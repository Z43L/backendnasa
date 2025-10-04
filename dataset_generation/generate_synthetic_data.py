#!/usr/bin/env python3
"""
Generador de datos sintéticos para entrenamiento del modelo.
Crea datasets realistas de deformación tectónica para áreas de falla.
"""

import os
import numpy as np
import pandas as pd
import h5py
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
import random
import multiprocessing as mp
from tqdm import tqdm

class SyntheticDataGenerator:
    """
    Genera datos sintéticos de deformación basados en patrones tectónicos realistas.
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)

        # Parámetros de deformación típicos (en mm/año)
        self.deformation_params = {
            'falla_anatolia': {
                'velocidad_media': 5.0,  # mm/año
                'aceleracion_max': 2.0,  # mm/año² durante precursores
                'ruido_std': 1.0,        # desviación estándar del ruido
                'coherencia_base': 0.8,  # coherencia típica
            },
            'cinturon_fuego_pacifico': {
                'velocidad_media': 8.0,
                'aceleracion_max': 3.0,
                'ruido_std': 1.5,
                'coherencia_base': 0.75,
            }
        }

    def generar_patron_deformacion(self, area: str, num_dias: int = 365*2,
                                 incluir_precursor: bool = False,
                                 fecha_terremoto: datetime = None) -> Tuple[np.ndarray, List[datetime]]:
        """
        Genera un patrón de deformación temporal para un punto.

        Args:
            area: Área tectónica
            num_dias: Número de días de datos
            incluir_precursor: Si incluir un evento precursor
            fecha_terremoto: Fecha del terremoto (si aplica)

        Returns:
            Tupla de (deformacion_acumulada, fechas)
        """
        params = self.deformation_params[area]

        # Generar timestamps
        fecha_inicio = datetime(2022, 1, 1)
        fechas = [fecha_inicio + timedelta(days=i) for i in range(num_dias)]

        # Deformación base (lineal + ruido)
        tiempo_anios = np.array([i/365.25 for i in range(num_dias)])
        deformacion_base = params['velocidad_media'] * tiempo_anios

        # Añadir componente estacional (pequeña variación anual)
        componente_estacional = 0.5 * np.sin(2 * np.pi * tiempo_anios)
        deformacion_base += componente_estacional

        # Añadir ruido
        ruido = np.random.normal(0, params['ruido_std'], num_dias)
        deformacion_total = deformacion_base + np.cumsum(ruido * 0.1)  # ruido acumulado

        # Añadir evento sísmico si se solicita
        if incluir_precursor and fecha_terremoto:
            idx_terremoto = (fecha_terremoto - fecha_inicio).days

            if idx_terremoto >= 0 and idx_terremoto < num_dias:
                # Evento dentro del rango de la secuencia
                # Añadir desplazamiento coseísmico
                deformacion_total[idx_terremoto:] += 50.0  # 50mm de desplazamiento

                # Añadir precursor en los días previos
                ventana_precursor = min(45, idx_terremoto)  # máximo 45 días de precursor
                inicio_precursor = max(0, idx_terremoto - ventana_precursor)

                for i in range(inicio_precursor, idx_terremoto):
                    dias_hasta_terremoto = idx_terremoto - i
                    # Aceleración que aumenta exponencialmente cerca del evento
                    factor_aceleracion = np.exp(-dias_hasta_terremoto / 15)  # decaimiento más rápido
                    deformacion_total[i] += params['aceleracion_max'] * factor_aceleracion

            elif idx_terremoto < 0:
                # Terremoto ocurrió antes de la secuencia (post-terremoto)
                # Añadir relajación post-sísmica
                for i in range(num_dias):
                    dias_desde_terremoto = -idx_terremoto + i
                    # Decaimiento exponencial post-terremoto
                    factor_relajacion = np.exp(-dias_desde_terremoto / 30)
                    deformacion_total[i] += 20.0 * factor_relajacion  # 20mm adicionales que decaen

            # Para precursores (terremoto futuro), la aceleración ya se añade arriba

        return deformacion_total, fechas

    def generar_grid_espacial(self, area: str, grid_size: Tuple[int, int] = (50, 50),
                            incluir_falla: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera un grid espacial con patrones de deformación realistas.

        Args:
            area: Área tectónica
            grid_size: Tamaño del grid (H, W)
            incluir_falla: Si incluir una zona de falla con deformación localizada

        Returns:
            Tupla de (lat_grid, lon_grid, deformacion_grid)
        """
        h, w = grid_size

        # Definir límites aproximados de las áreas
        limites = {
            'falla_anatolia': {'lat_min': 35.0, 'lat_max': 42.0, 'lon_min': 25.0, 'lon_max': 45.0},
            'cinturon_fuego_pacifico': {'lat_min': -60.0, 'lat_max': 60.0, 'lon_min': -180.0, 'lon_max': 180.0}
        }

        limites_area = limites[area]

        # Crear grids de coordenadas
        lat_grid = np.linspace(limites_area['lat_min'], limites_area['lat_max'], h)
        lon_grid = np.linspace(limites_area['lon_min'], limites_area['lon_max'], w)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        # Deformación base (gradiente regional)
        deformacion_base = np.zeros_like(lat_mesh)

        if incluir_falla:
            # Añadir zona de falla localizada
            if area == 'falla_anatolia':
                # Falla Norte de Anatolia (aproximada)
                falla_lat = 40.5
                falla_lon = 35.0
                ancho_falla = 0.5  # grados

                # Distancia a la falla
                dist_lat = lat_mesh - falla_lat
                dist_lon = lon_mesh - falla_lon
                distancia_falla = np.sqrt(dist_lat**2 + dist_lon**2)

                # Deformación máxima en la falla, decae con la distancia
                deformacion_falla = 10.0 * np.exp(-distancia_falla / ancho_falla)
                deformacion_base += deformacion_falla

            elif area == 'cinturon_fuego_pacifico':
                # Zona de subducción (simplificada)
                # Deformación aumenta hacia la costa
                deformacion_costa = np.exp(-(lon_mesh + 100) / 50) * 15.0
                deformacion_base += deformacion_costa

        # Añadir variabilidad espacial
        ruido_espacial = np.random.normal(0, 2.0, (h, w))
        deformacion_grid = deformacion_base + ruido_espacial

        return lat_mesh, lon_mesh, deformacion_grid

    def _generar_secuencia_individual(self, args) -> Tuple[np.ndarray, str, List[datetime]]:
        """
        Genera una secuencia individual (función auxiliar para paralelización).
        """
        i, area, longitud_secuencia, deformacion_espacial = args
        
        # Decidir tipo de secuencia
        rand_val = random.random()
        if rand_val < 0.4:
            # 40% secuencias normales
            incluir_terremoto = False
            etiqueta = 'normal'
            fecha_terremoto = None
        elif rand_val < 0.7:
            # 30% secuencias con precursores
            incluir_terremoto = True
            etiqueta = 'precursor'
            # Terremoto ocurre después de la secuencia (fuera del rango visible)
            dias_hasta_terremoto = random.randint(longitud_secuencia + 1, longitud_secuencia + 30)
            fecha_base = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))
            fecha_terremoto = fecha_base + timedelta(days=dias_hasta_terremoto)
        else:
            # 30% secuencias post-terremoto
            incluir_terremoto = True
            etiqueta = 'post_terremoto'
            # Terremoto ocurrió antes de la secuencia
            dias_desde_terremoto = random.randint(1, 30)
            fecha_base = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))
            fecha_terremoto = fecha_base - timedelta(days=dias_desde_terremoto)

        # Generar patrón temporal base
        deformacion_temporal, fechas = self.generar_patron_deformacion(
            area, num_dias=longitud_secuencia, incluir_precursor=incluir_terremoto,
            fecha_terremoto=fecha_terremoto
        )

        # Crear fotogramas para cada timestamp
        fotogramas = []
        for j, fecha in enumerate(fechas):
            # Deformación espacial + componente temporal
            fotograma = deformacion_espacial + deformacion_temporal[j]

            # Añadir ruido temporal al fotograma
            ruido_temporal = np.random.normal(0, 0.5, fotograma.shape)
            fotograma += ruido_temporal

            fotogramas.append(fotograma)

        secuencia = np.stack(fotogramas, axis=0)  # [T, H, W]
        
        return secuencia, etiqueta, fechas

    def generar_dataset_area(self, area: str, num_secuencias: int = 100,
                           longitud_secuencia: int = 30, incluir_eventos: bool = True) -> Dict:
        """
        Genera un dataset completo para un área.

        Args:
            area: Área tectónica
            num_secuencias: Número de secuencias a generar
            longitud_secuencia: Longitud de cada secuencia (días)
            incluir_eventos: Si incluir terremotos y precursores

        Returns:
            Diccionario con datos del dataset
        """
        print(f"Generando dataset para {area} con {num_secuencias} secuencias...")

        # Generar grid espacial base
        lat_grid, lon_grid, deformacion_espacial = self.generar_grid_espacial(area)

        # Preparar argumentos para paralelización
        args_list = [(i, area, longitud_secuencia, deformacion_espacial.copy()) 
                    for i in range(num_secuencias)]

        # Usar multiprocessing para generar secuencias en paralelo
        num_workers = min(mp.cpu_count(), 8)  # Máximo 8 workers
        print(f"Usando {num_workers} procesos para generación paralela...")

        with mp.Pool(num_workers) as pool:
            try:
                from tqdm import tqdm
                resultados = list(tqdm(
                    pool.imap(self._generar_secuencia_individual, args_list),
                    total=num_secuencias,
                    desc="Generando secuencias"
                ))
            except ImportError:
                # Fallback sin barra de progreso
                resultados = pool.map(self._generar_secuencia_individual, args_list)

        # Separar resultados
        secuencias_clasificacion = []
        etiquetas_clasificacion = []
        fechas_secuencias = []

        for secuencia, etiqueta, fechas in resultados:
            secuencias_clasificacion.append(secuencia)
            etiquetas_clasificacion.append(etiqueta)
            fechas_secuencias.append(fechas)

        # Para regresión: usar las mismas secuencias
        secuencias_regresion = secuencias_clasificacion

        # Crear dataset de clasificación
        dataset_clf = {
            'secuencias': secuencias_clasificacion,
            'etiquetas': etiquetas_clasificacion,
            'fechas': fechas_secuencias,
            'tipo': 'clasificacion'
        }

        # Crear dataset de regresión (auto-regresiva)
        secuencias_reg_entrada = []
        secuencias_reg_objetivo = []

        for seq in secuencias_regresion:
            entrada = seq[:-1]  # Todos menos el último
            objetivo = seq[-1:]  # Solo el último
            secuencias_reg_entrada.append(entrada)
            secuencias_reg_objetivo.append(objetivo)

        dataset_reg = {
            'secuencias_entrada': secuencias_reg_entrada,
            'secuencias_objetivo': secuencias_reg_objetivo,
            'tipo': 'regresion'
        }

        return {
            'clasificacion': dataset_clf,
            'regresion': dataset_reg,
            'metadata': {
                'area': area,
                'num_secuencias': num_secuencias,
                'longitud_secuencia': longitud_secuencia,
                'grid_size': deformacion_espacial.shape,
                'fecha_generacion': datetime.now().isoformat(),
                'parametros': self.deformation_params[area]
            }
        }

    def guardar_dataset(self, dataset: Dict, archivo_salida: str) -> None:
        """
        Guarda un dataset en formato HDF5.

        Args:
            dataset: Diccionario con datos del dataset
            archivo_salida: Ruta del archivo de salida
        """
        os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)

        with h5py.File(archivo_salida, 'w') as f:
            # Guardar metadatos
            for key, value in dataset['metadata'].items():
                if isinstance(value, (str, int, float)):
                    f.attrs[key] = value
                elif isinstance(value, dict):
                    f.attrs[key] = json.dumps(value)
                elif isinstance(value, tuple):
                    f.attrs[key] = json.dumps(list(value))

            # Guardar datasets
            for tipo_modelo, datos_modelo in dataset.items():
                if tipo_modelo == 'metadata':
                    continue

                grupo = f.create_group(tipo_modelo)

                if tipo_modelo == 'clasificacion':
                    # Secuencias
                    if datos_modelo['secuencias']:
                        seq_array = np.stack(datos_modelo['secuencias'], axis=0)
                        grupo.create_dataset('secuencias', data=seq_array, compression='gzip')

                    # Etiquetas (convertir a índices)
                    clases_unicas = list(set(datos_modelo['etiquetas']))
                    clase_a_indice = {clase: i for i, clase in enumerate(clases_unicas)}
                    etiquetas_indices = np.array([clase_a_indice[etiqueta] for etiqueta in datos_modelo['etiquetas']])

                    grupo.create_dataset('etiquetas', data=etiquetas_indices)
                    grupo.attrs['clases'] = json.dumps(clases_unicas)
                    grupo.attrs['clase_a_indice'] = json.dumps(clase_a_indice)

                elif tipo_modelo == 'regresion':
                    # Secuencias de entrada
                    if datos_modelo['secuencias_entrada']:
                        entrada_array = np.stack(datos_modelo['secuencias_entrada'], axis=0)
                        grupo.create_dataset('secuencias_entrada', data=entrada_array, compression='gzip')

                    # Secuencias objetivo
                    if datos_modelo['secuencias_objetivo']:
                        objetivo_array = np.stack(datos_modelo['secuencias_objetivo'], axis=0)
                        grupo.create_dataset('secuencias_objetivo', data=objetivo_array, compression='gzip')

        print(f"Dataset guardado en {archivo_salida}")
        print(f"Shape secuencias: {seq_array.shape if 'seq_array' in locals() else 'N/A'}")

def main():
    """Función principal para generar datasets sintéticos."""
    generador = SyntheticDataGenerator(seed=42)

    # Crear directorio de datasets
    os.makedirs('datasets', exist_ok=True)

    # Generar datasets para cada área
    areas = ['falla_anatolia', 'cinturon_fuego_pacifico']

    for area in areas:
        print(f"\n=== Generando datos para {area} ===")

        # Generar dataset con más datos para mejor precisión
        dataset = generador.generar_dataset_area(
            area=area,
            num_secuencias=5000,  # Reducido a 8000 para mantener < 3GB por archivo
            longitud_secuencia=30,  # 30 días por secuencia
            incluir_eventos=True
        )

        # Guardar datasets
        archivo_clf = f"datasets/{area}_synthetic_clasificacion.h5"
        generador.guardar_dataset(
            {'clasificacion': dataset['clasificacion'], 'metadata': dataset['metadata']},
            archivo_clf
        )

        archivo_reg = f"datasets/{area}_synthetic_regresion.h5"
        generador.guardar_dataset(
            {'regresion': dataset['regresion'], 'metadata': dataset['metadata']},
            archivo_reg
        )

        # Mostrar estadísticas
        etiquetas = dataset['clasificacion']['etiquetas']
        print(f"Distribución de etiquetas en {area}:")
        for etiqueta in set(etiquetas):
            count = etiquetas.count(etiqueta)
            print(f"  {etiqueta}: {count} secuencias ({count/len(etiquetas)*100:.1f}%)")

    print("\n=== Generación de datos completada ===")
    print("Datasets disponibles:")
    for area in areas:
        print(f"  - {area}_synthetic_clasificacion.h5")
        print(f"  - {area}_synthetic_regresion.h5")

if __name__ == "__main__":
    main()