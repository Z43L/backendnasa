#!/usr/bin/env python3
"""
Preparación de datos para el modelo de IA.
Genera secuencias temporales de deformación para entrenamiento y predicción.
"""

import os
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import h5py
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import json

# Configuración de base de datos
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', 5432),
    'database': os.getenv('DB_NAME', 'deformacion_monitor'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

class DataPreparation:
    def __init__(self):
        self.db_connection = None

    def conectar_db(self):
        """Establece conexión con la base de datos."""
        try:
            self.db_connection = psycopg2.connect(**DB_CONFIG)
            print("Conexión a base de datos establecida.")
        except Exception as e:
            print(f"Error conectando a base de datos: {e}")
            raise

    def cerrar_db(self):
        """Cierra la conexión con la base de datos."""
        if self.db_connection:
            self.db_connection.close()

    def extraer_series_temporales(self, area_interes: str, fecha_inicio: datetime, fecha_fin: datetime) -> pd.DataFrame:
        """
        Extrae series temporales de deformación para un área específica.

        Args:
            area_interes: Nombre del área (ej. 'falla_anatolia')
            fecha_inicio: Fecha de inicio
            fecha_fin: Fecha de fin

        Returns:
            DataFrame con series temporales
        """
        query = """
        SELECT
            ds.fecha,
            ds.punto_lat,
            ds.punto_lon,
            ds.deformacion_mm,
            ds.coherencia,
            i.job_name,
            i.baseline_temporal
        FROM deformacion_series ds
        JOIN interferogramas i ON ds.interferograma_id = i.id
        WHERE i.area_interes = %s
        AND ds.fecha BETWEEN %s AND %s
        AND ds.coherencia > 0.3  -- Filtrar por buena coherencia
        ORDER BY ds.fecha, ds.punto_lat, ds.punto_lon
        """

        df = pd.read_sql_query(query, self.db_connection,
                              params=[area_interes, fecha_inicio, fecha_fin])

        return df

    def crear_secuencias(self, df: pd.DataFrame, longitud_secuencia: int = 50,
                        paso_temporal: int = 1) -> List[np.ndarray]:
        """
        Crea secuencias de longitud fija a partir de los datos temporales.

        Args:
            df: DataFrame con datos temporales
            longitud_secuencia: Número de fotogramas por secuencia
            paso_temporal: Paso entre fotogramas consecutivos

        Returns:
            Lista de secuencias (arrays de shape [longitud_secuencia, altura, ancho])
        """
        # Agrupar por fecha
        fechas_unicas = sorted(df['fecha'].unique())

        if len(fechas_unicas) < longitud_secuencia:
            print(f"Insuficientes fechas para crear secuencias. Necesarias: {longitud_secuencia}, disponibles: {len(fechas_unicas)}")
            return []

        # Crear grid regular para los puntos
        lats_unicas = sorted(df['punto_lat'].unique())
        lons_unicas = sorted(df['punto_lon'].unique())

        # Definir límites del grid
        lat_min, lat_max = lats_unicas[0], lats_unicas[-1]
        lon_min, lon_max = lons_unicas[0], lons_unicas[-1]

        # Crear grid de 100x100 puntos
        grid_size = 50
        lat_grid = np.linspace(lat_min, lat_max, grid_size)
        lon_grid = np.linspace(lon_min, lon_max, grid_size)

        secuencias = []

        # Para cada ventana temporal posible
        for i in range(0, len(fechas_unicas) - longitud_secuencia + 1, paso_temporal):
            fechas_secuencia = fechas_unicas[i:i + longitud_secuencia]

            # Crear fotogramas para esta secuencia
            fotogramas = []

            for fecha in fechas_secuencia:
                # Filtrar datos para esta fecha
                df_fecha = df[df['fecha'] == fecha]

                if df_fecha.empty:
                    continue

                # Crear grid interpolado
                from scipy.interpolate import griddata

                points = df_fecha[['punto_lon', 'punto_lat']].values
                values = df_fecha['deformacion_mm'].values

                # Crear meshgrid
                lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

                # Interpolar valores
                grid_deformacion = griddata(points, values, (lon_mesh, lat_mesh),
                                          method='linear', fill_value=0)

                fotogramas.append(grid_deformacion)

            if len(fotogramas) == longitud_secuencia:
                secuencia = np.stack(fotogramas, axis=0)  # Shape: [T, H, W]
                secuencias.append(secuencia)

        return secuencias

    def guardar_dataset(self, secuencias: List[np.ndarray], archivo_salida: str,
                       metadatos: Optional[Dict] = None) -> None:
        """
        Guarda las secuencias en un archivo HDF5.

        Args:
            secuencias: Lista de arrays de secuencias
            archivo_salida: Ruta del archivo de salida
            metadatos: Metadatos adicionales
        """
        os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)

        with h5py.File(archivo_salida, 'w') as f:
            # Crear dataset principal
            if secuencias:
                # Stack all sequences
                dataset_array = np.stack(secuencias, axis=0)  # Shape: [N, T, H, W]
                dataset = f.create_dataset('secuencias', data=dataset_array,
                                         compression='gzip', compression_opts=9)

                # Agregar atributos
                dataset.attrs['num_secuencias'] = len(secuencias)
                dataset.attrs['longitud_secuencia'] = secuencias[0].shape[0]
                dataset.attrs['altura_grid'] = secuencias[0].shape[1]
                dataset.attrs['ancho_grid'] = secuencias[0].shape[2]
                dataset.attrs['unidades'] = 'mm'  # milímetros de deformación

            # Guardar metadatos
            if metadatos:
                for key, value in metadatos.items():
                    if isinstance(value, (str, int, float)):
                        f.attrs[key] = value
                    elif isinstance(value, dict):
                        f.attrs[key] = json.dumps(value)

        print(f"Dataset guardado en {archivo_salida}")
        print(f"Shape del dataset: {dataset_array.shape}")

    def preparar_dataset_entrenamiento(self, area_interes: str, fecha_inicio: datetime,
                                     fecha_fin: datetime, longitud_secuencia: int = 50) -> str:
        """
        Prepara dataset completo de entrenamiento para un área.

        Args:
            area_interes: Nombre del área
            fecha_inicio: Fecha de inicio
            fecha_fin: Fecha de fin
            longitud_secuencia: Longitud de las secuencias

        Returns:
            Ruta del archivo generado
        """
        print(f"Preparando dataset para {area_interes}...")

        # Extraer datos
        df = self.extraer_series_temporales(area_interes, fecha_inicio, fecha_fin)

        if df.empty:
            print(f"No se encontraron datos para {area_interes}")
            return None

        print(f"Datos extraídos: {len(df)} registros de {len(df['fecha'].unique())} fechas")

        # Crear secuencias
        secuencias = self.crear_secuencias(df, longitud_secuencia)

        if not secuencias:
            print("No se pudieron crear secuencias")
            return None

        # Metadatos
        metadatos = {
            'area_interes': area_interes,
            'fecha_inicio': fecha_inicio.isoformat(),
            'fecha_fin': fecha_fin.isoformat(),
            'longitud_secuencia': longitud_secuencia,
            'num_secuencias_generadas': len(secuencias),
            'fecha_creacion': datetime.now().isoformat()
        }

        # Guardar dataset
        archivo_salida = f"datasets/{area_interes}_secuencias_{longitud_secuencia}.h5"
        self.guardar_dataset(secuencias, archivo_salida, metadatos)

        return archivo_salida

    def ejecutar_preparacion(self) -> None:
        """Ejecuta la preparación completa de datasets."""
        try:
            self.conectar_db()

            # Configurar fechas (últimos 2 años)
            fecha_fin = datetime.now()
            fecha_inicio = fecha_fin - timedelta(days=730)

            # Preparar datasets para cada área
            areas = ['falla_anatolia', 'cinturon_fuego_pacifico']

            for area in areas:
                archivo = self.preparar_dataset_entrenamiento(
                    area, fecha_inicio, fecha_fin, longitud_secuencia=30  # Más corto para prueba
                )
                if archivo:
                    print(f"Dataset creado: {archivo}")

        finally:
            self.cerrar_db()

def main():
    """Función principal de preparación de datos."""
    preparador = DataPreparation()
    preparador.ejecutar_preparacion()

if __name__ == "__main__":
    main()