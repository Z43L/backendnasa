#!/usr/bin/env python3
"""
Sistema de Inferencia en Tiempo Real para predicción de deformación.
Monitorea nuevos datos de InSAR y genera predicciones automáticamente.
"""

import os
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import griddata

from model_architecture import load_model_checkpoint, predict_next_frame, classify_sequence_state

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_time_inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración de base de datos
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', 5432),
    'database': os.getenv('DB_NAME', 'deformacion_monitor'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

class RealTimeInference:
    """
    Sistema de inferencia en tiempo real para predicción de deformación.
    """

    def __init__(self, check_interval: int = 300):  # 5 minutos por defecto
        """
        Inicializa el sistema de inferencia en tiempo real.

        Args:
            check_interval: Intervalo en segundos entre verificaciones de nuevos datos
        """
        self.check_interval = check_interval
        self.running = False
        self.models = {}
        self.last_processed_dates = {}  # Para evitar reprocesar datos

        # Conectar a base de datos
        self.db_connection = None
        self._connect_db()

        # Cargar modelos
        self._load_models()

        logger.info("Sistema de inferencia en tiempo real inicializado")

    def _connect_db(self):
        """Establece conexión con la base de datos."""
        try:
            self.db_connection = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            self.db_connection.autocommit = True  # Para notificaciones en tiempo real
            logger.info("Conexión a base de datos establecida")
        except Exception as e:
            logger.error(f"Error conectando a base de datos: {e}")
            raise

    def _load_models(self):
        """Carga los modelos entrenados."""
        model_paths = {
            'falla_anatolia_clasificacion': 'checkpoints_classification_falla_anatolia/best_model.pth',
            'falla_anatolia_regresion': 'checkpoints_regression_falla_anatolia/best_model.pth',
            'cinturon_fuego_pacifico_clasificacion': 'checkpoints_classification_cinturon_fuego_pacifico/best_model.pth',
            'cinturon_fuego_pacifico_regresion': 'checkpoints_regression_cinturon_fuego_pacifico/best_model.pth'
        }

        for model_name, path in model_paths.items():
            if os.path.exists(path):
                try:
                    model = load_model_checkpoint(path)
                    self.models[model_name] = model
                    logger.info(f"Modelo cargado: {model_name}")
                except Exception as e:
                    logger.error(f"Error cargando {model_name}: {e}")
            else:
                logger.warning(f"Modelo no encontrado: {model_name}")

        logger.info(f"Modelos disponibles: {list(self.models.keys())}")

    def _get_new_interferograms(self) -> List[Dict]:
        """
        Obtiene interferogramas nuevos que no han sido procesados aún.

        Returns:
            Lista de interferogramas nuevos
        """
        try:
            cursor = self.db_connection.cursor()

            # Buscar interferogramas procesados en las últimas 24 horas
            # que no tengan predicciones asociadas
            cursor.execute("""
                SELECT i.*,
                       COUNT(p.id) as num_predicciones
                FROM interferogramas i
                LEFT JOIN predicciones p ON i.id = p.interferograma_id
                WHERE i.fecha_procesamiento >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
                GROUP BY i.id
                HAVING COUNT(p.id) = 0  -- Solo interferogramas sin predicciones
                ORDER BY i.fecha_procesamiento DESC
            """)

            interferogramas = cursor.fetchall()
            cursor.close()

            logger.info(f"Encontrados {len(interferogramas)} interferogramas nuevos para procesar")
            return interferogramas

        except Exception as e:
            logger.error(f"Error obteniendo interferogramas nuevos: {e}")
            return []

    def _get_sequence_data(self, area_interes: str, fecha_referencia: datetime,
                          num_pasos: int = 30) -> Optional[np.ndarray]:
        """
        Obtiene secuencia de datos para una predicción.

        Args:
            area_interes: Área de interés
            fecha_referencia: Fecha del interferograma más reciente
            num_pasos: Número de pasos temporales

        Returns:
            Array de secuencia [T, H, W] o None si no hay suficientes datos
        """
        try:
            cursor = self.db_connection.cursor()

            # Obtener datos históricos incluyendo el nuevo interferograma
            cursor.execute("""
                SELECT ds.fecha, ds.punto_lat, ds.punto_lon, ds.deformacion_mm
                FROM deformacion_series ds
                JOIN interferogramas i ON ds.interferograma_id = i.id
                WHERE i.area_interes = %s
                AND ds.fecha <= %s
                AND ds.coherencia > 0.3
                ORDER BY ds.fecha DESC, ds.punto_lat, ds.punto_lon
                LIMIT 100000  -- Límite para evitar sobrecarga
            """, [area_interes, fecha_referencia])

            rows = cursor.fetchall()
            cursor.close()

            if not rows:
                logger.warning(f"No se encontraron datos para {area_interes}")
                return None

            # Convertir a DataFrame
            df = pd.DataFrame(rows)

            # Obtener fechas únicas más recientes
            fechas_unicas = sorted(df['fecha'].unique())[-num_pasos:]

            if len(fechas_unicas) < num_pasos:
                logger.warning(f"Secuencia insuficiente para {area_interes}: {len(fechas_unicas)} de {num_pasos}")
                return None

            # Crear grid regular
            lats_unicas = sorted(df['punto_lat'].unique())
            lons_unicas = sorted(df['punto_lon'].unique())

            lat_min, lat_max = lats_unicas[0], lats_unicas[-1]
            lon_min, lon_max = lons_unicas[0], lons_unicas[-1]

            grid_size = 50
            lat_grid = np.linspace(lat_min, lat_max, grid_size)
            lon_grid = np.linspace(lon_min, lon_max, grid_size)

            # Crear secuencia de fotogramas
            fotogramas = []
            for fecha in fechas_unicas:
                df_fecha = df[df['fecha'] == fecha]
                if df_fecha.empty:
                    continue

                points = df_fecha[['punto_lon', 'punto_lat']].values
                values = df_fecha['deformacion_mm'].values

                lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
                grid_deformacion = griddata(points, values, (lon_mesh, lat_mesh),
                                          method='linear', fill_value=0)
                fotogramas.append(grid_deformacion)

            if len(fotogramas) == num_pasos:
                sequence = np.stack(fotogramas, axis=0)  # [T, H, W]
                return sequence
            else:
                logger.warning(f"No se pudieron crear suficientes fotogramas: {len(fotogramas)} de {num_pasos}")
                return None

        except Exception as e:
            logger.error(f"Error obteniendo secuencia de datos: {e}")
            return None

    def _generate_predictions(self, interferograma: Dict) -> List[Dict]:
        """
        Genera predicciones para un interferograma.

        Args:
            interferograma: Datos del interferograma

        Returns:
            Lista de predicciones generadas
        """
        area_interes = interferograma['area_interes']
        fecha_procesamiento = interferograma['fecha_procesamiento']

        logger.info(f"Generando predicciones para {area_interes} - {fecha_procesamiento}")

        predictions = []

        # Para cada tipo de modelo disponible
        for model_type in ['clasificacion', 'regresion']:
            model_key = f"{area_interes}_{model_type}"

            if model_key not in self.models:
                logger.warning(f"Modelo no disponible: {model_key}")
                continue

            model = self.models[model_key]

            # Obtener secuencia de datos
            sequence = self._get_sequence_data(area_interes, fecha_procesamiento)

            if sequence is None:
                logger.warning(f"No se pudo obtener secuencia para {model_key}")
                continue

            try:
                # Convertir a tensor
                sequence_tensor = torch.from_numpy(sequence).float()

                # Generar predicción
                if model_type == 'regresion':
                    prediction = predict_next_frame(model, sequence_tensor)
                    prediction_data = {
                        'tipo': 'regresion',
                        'prediccion': prediction.numpy().tolist(),
                        'descripcion': 'Fotograma de deformación predicho'
                    }
                elif model_type == 'clasificacion':
                    clase, probabilidades = classify_sequence_state(model, sequence_tensor)
                    clases_nombres = ['normal', 'precursor', 'post_terremoto']
                    prediction_data = {
                        'tipo': 'clasificacion',
                        'clase_predicha': int(clase),
                        'clase_nombre': clases_nombres[clase],
                        'probabilidades': probabilidades.numpy().tolist(),
                        'descripcion': f'Estado: {clases_nombres[clase]}'
                    }

                # Crear registro de predicción
                prediction_record = {
                    'interferograma_id': interferograma['id'],
                    'tipo_modelo': model_type,
                    'prediccion': json.dumps(prediction_data),
                    'fecha_prediccion': datetime.now(),
                    'confidence_score': float(max(probabilidades.numpy())) if model_type == 'clasificacion' else None,
                    'alerta_requerida': self._evaluar_alerta(prediction_data)
                }

                predictions.append(prediction_record)
                logger.info(f"Predicción generada: {model_key} - {prediction_data.get('clase_nombre', 'regresión')}")

            except Exception as e:
                logger.error(f"Error generando predicción {model_key}: {e}")
                continue

        return predictions

    def _evaluar_alerta(self, prediction_data: Dict) -> bool:
        """
        Evalúa si una predicción requiere alerta.

        Args:
            prediction_data: Datos de la predicción

        Returns:
            True si requiere alerta
        """
        if prediction_data['tipo'] == 'clasificacion':
            # Alertar si se detecta estado precursor
            return prediction_data.get('clase_nombre') == 'precursor'
        elif prediction_data['tipo'] == 'regresion':
            # Alertar si hay deformación significativa (> 10mm)
            prediccion_array = np.array(prediction_data['prediccion'])
            return np.max(np.abs(prediccion_array)) > 10.0

        return False

    def _save_predictions(self, predictions: List[Dict]):
        """
        Guarda las predicciones en la base de datos.

        Args:
            predictions: Lista de predicciones a guardar
        """
        if not predictions:
            return

        try:
            cursor = self.db_connection.cursor()

            # Insertar predicciones
            insert_query = """
                INSERT INTO predicciones
                (interferograma_id, tipo_modelo, prediccion, fecha_prediccion,
                 confidence_score, alerta_requerida)
                VALUES (%s, %s, %s, %s, %s, %s)
            """

            for pred in predictions:
                cursor.execute(insert_query, (
                    pred['interferograma_id'],
                    pred['tipo_modelo'],
                    pred['prediccion'],
                    pred['fecha_prediccion'],
                    pred['confidence_score'],
                    pred['alerta_requerida']
                ))

            self.db_connection.commit()
            cursor.close()

            logger.info(f"Guardadas {len(predictions)} predicciones en BD")

            # Verificar si hay alertas
            alertas = [p for p in predictions if p['alerta_requerida']]
            if alertas:
                self._enviar_alertas(alertas)

        except Exception as e:
            logger.error(f"Error guardando predicciones: {e}")
            self.db_connection.rollback()

    def _enviar_alertas(self, alertas: List[Dict]):
        """
        Envía alertas para predicciones críticas.

        Args:
            alertas: Lista de predicciones que requieren alerta
        """
        logger.warning(f"🚨 ALERTA: {len(alertas)} predicciones críticas detectadas!")

        for alerta in alertas:
            logger.warning(f"  - Área: {alerta.get('area_interes', 'desconocida')}")
            logger.warning(f"  - Tipo: {alerta['tipo_modelo']}")
            logger.warning(f"  - Detalles: {alerta['prediccion'][:200]}...")

        # En un sistema real, aquí se enviarían:
        # - Emails a expertos
        # - Notificaciones push
        # - Integración con sistemas de alerta sísmica
        # - Logs en sistemas de monitoreo

    async def _run_inference_cycle(self):
        """Ejecuta un ciclo completo de inferencia."""
        try:
            logger.info("Iniciando ciclo de inferencia...")

            # Obtener nuevos interferogramas
            nuevos_interferogramas = self._get_new_interferograms()

            total_predictions = 0

            for interferograma in nuevos_interferogramas:
                # Generar predicciones
                predictions = self._generate_predictions(interferograma)

                # Guardar predicciones
                if predictions:
                    self._save_predictions(predictions)
                    total_predictions += len(predictions)

            if total_predictions > 0:
                logger.info(f"Ciclo completado: {total_predictions} predicciones generadas")
            else:
                logger.info("Ciclo completado: No hay nuevos datos para procesar")

        except Exception as e:
            logger.error(f"Error en ciclo de inferencia: {e}")

    async def start_monitoring(self):
        """Inicia el monitoreo continuo de nuevos datos."""
        self.running = True
        logger.info(f"Iniciando monitoreo continuo (intervalo: {self.check_interval}s)")

        while self.running:
            try:
                await self._run_inference_cycle()
                await asyncio.sleep(self.check_interval)
            except KeyboardInterrupt:
                logger.info("Monitoreo interrumpido por usuario")
                break
            except Exception as e:
                logger.error(f"Error en monitoreo: {e}")
                await asyncio.sleep(self.check_interval)

        logger.info("Monitoreo detenido")

    def stop_monitoring(self):
        """Detiene el monitoreo."""
        self.running = False
        logger.info("Solicitud de detener monitoreo recibida")

    def run_single_cycle(self):
        """Ejecuta un solo ciclo de inferencia (para pruebas)."""
        logger.info("Ejecutando ciclo único de inferencia...")
        asyncio.run(self._run_inference_cycle())
        logger.info("Ciclo único completado")

    def __del__(self):
        """Limpieza al destruir el objeto."""
        if self.db_connection:
            self.db_connection.close()
            logger.info("Conexión a base de datos cerrada")

def main():
    """Función principal."""
    import argparse

    parser = argparse.ArgumentParser(description='Sistema de inferencia en tiempo real')
    parser.add_argument('--mode', choices=['continuous', 'single'],
                       default='single', help='Modo de ejecución')
    parser.add_argument('--interval', type=int, default=300,
                       help='Intervalo en segundos para monitoreo continuo')

    args = parser.parse_args()

    # Crear sistema de inferencia
    inference_system = RealTimeInference(check_interval=args.interval)

    if args.mode == 'continuous':
        # Ejecutar monitoreo continuo
        try:
            asyncio.run(inference_system.start_monitoring())
        except KeyboardInterrupt:
            inference_system.stop_monitoring()
    else:
        # Ejecutar un solo ciclo
        inference_system.run_single_cycle()

if __name__ == "__main__":
    main()