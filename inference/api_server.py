#!/usr/bin/env python3
"""
API Backend para el sistema de monitoreo de deformación del terreno.
Proporciona endpoints para consultar datos, hacer predicciones y gestionar el sistema.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
import torch
import h5py

from model_architecture import load_model_checkpoint, predict_next_frame, classify_sequence_state

# Configuración de base de datos
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', 5432),
    'database': os.getenv('DB_NAME', 'deformacion_monitor'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

# Modelos globales
models = {}
db_connection_pool = None

class PredictionRequest(BaseModel):
    """Modelo para solicitud de predicción."""
    area_interes: str = Field(..., description="Área de interés (ej. 'falla_anatolia')")
    fecha_inicio: Optional[datetime] = Field(None, description="Fecha de inicio de la secuencia")
    num_pasos: int = Field(30, description="Número de pasos temporales para la predicción")
    tipo_prediccion: str = Field("clasificacion", description="Tipo: 'clasificacion' o 'regresion'")

class AreaInteres(BaseModel):
    """Modelo para área de interés."""
    nombre: str
    descripcion: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    tipo_falla: str

class EstadoSistema(BaseModel):
    """Modelo para estado del sistema."""
    estado: str
    ultimo_procesamiento: Optional[datetime]
    areas_activas: List[str]
    modelos_disponibles: List[str]
    estadisticas: Dict[str, Any]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación."""
    # Inicializar al startup
    global db_connection_pool, models

    print("Iniciando API backend...")

    # Inicializar pool de conexiones a BD
    try:
        # En producción usaríamos un pool real como asyncpg
        db_connection_pool = "initialized"
        print("✓ Conexión a base de datos inicializada")
    except Exception as e:
        print(f"✗ Error conectando a BD: {e}")

    # Cargar modelos entrenados
    try:
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
                    models[model_name] = model
                    print(f"✓ Modelo cargado: {model_name}")
                except Exception as e:
                    print(f"✗ Error cargando {model_name}: {e}")
            else:
                print(f"⚠ Modelo no encontrado: {model_name}")

        print(f"Modelos disponibles: {list(models.keys())}")

    except Exception as e:
        print(f"Error cargando modelos: {e}")

    yield

    # Cleanup al shutdown
    print("Cerrando API backend...")
    if db_connection_pool:
        # Cerrar conexiones
        pass

# Crear aplicación FastAPI
app = FastAPI(
    title="API de Monitoreo de Deformación del Terreno",
    description="Sistema de IA para predicción de deformaciones terrestres usando InSAR",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    """Obtiene una conexión a la base de datos."""
    try:
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de base de datos: {str(e)}")

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Endpoint raíz con información del API."""
    return {
        "mensaje": "API de Monitoreo de Deformación del Terreno",
        "version": "1.0.0",
        "documentacion": "/docs",
        "estado": "/estado"
    }

@app.get("/estado", response_model=EstadoSistema)
async def obtener_estado_sistema():
    """Obtiene el estado actual del sistema."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Obtener último procesamiento
        cursor.execute("""
            SELECT MAX(fecha_procesamiento) as ultimo_procesamiento
            FROM interferogramas
        """)
        ultimo_proc = cursor.fetchone()['ultimo_procesamiento']

        # Obtener áreas activas
        cursor.execute("""
            SELECT DISTINCT area_interes
            FROM interferogramas
            WHERE fecha_procesamiento >= CURRENT_DATE - INTERVAL '30 days'
        """)
        areas = [row['area_interes'] for row in cursor.fetchall()]

        # Estadísticas
        cursor.execute("""
            SELECT
                COUNT(*) as total_interferogramas,
                COUNT(DISTINCT area_interes) as areas_cubiertas,
                AVG(coherencia) as coherencia_promedio
            FROM deformacion_series
        """)
        stats = cursor.fetchone()

        conn.close()

        return EstadoSistema(
            estado="operativo",
            ultimo_procesamiento=ultimo_proc,
            areas_activas=areas,
            modelos_disponibles=list(models.keys()),
            estadisticas=dict(stats)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estado: {str(e)}")

@app.get("/areas")
async def listar_areas_interes():
    """Lista todas las áreas de interés disponibles."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT area_interes,
                   MIN(punto_lat) as lat_min, MAX(punto_lat) as lat_max,
                   MIN(punto_lon) as lon_min, MAX(punto_lon) as lon_max,
                   COUNT(*) as num_puntos
            FROM deformacion_series
            GROUP BY area_interes
        """)

        areas = []
        for row in cursor.fetchall():
            areas.append({
                "nombre": row['area_interes'],
                "lat_min": float(row['lat_min']),
                "lat_max": float(row['lat_max']),
                "lon_min": float(row['lon_min']),
                "lon_max": float(row['lon_max']),
                "num_puntos": row['num_puntos']
            })

        conn.close()
        return {"areas": areas}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando áreas: {str(e)}")

@app.get("/datos/historicos/{area_interes}")
async def obtener_datos_historicos(
    area_interes: str,
    fecha_inicio: Optional[datetime] = Query(None),
    fecha_fin: Optional[datetime] = Query(None),
    limite: int = Query(1000, description="Número máximo de registros")
):
    """Obtiene datos históricos de deformación para un área."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Construir query
        query = """
            SELECT
                ds.fecha,
                ds.punto_lat,
                ds.punto_lon,
                ds.deformacion_mm,
                ds.coherencia,
                i.baseline_temporal
            FROM deformacion_series ds
            JOIN interferogramas i ON ds.interferograma_id = i.id
            WHERE i.area_interes = %s
        """
        params = [area_interes]

        if fecha_inicio:
            query += " AND ds.fecha >= %s"
            params.append(fecha_inicio)

        if fecha_fin:
            query += " AND ds.fecha <= %s"
            params.append(fecha_fin)

        query += " ORDER BY ds.fecha DESC LIMIT %s"
        params.append(limite)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convertir a formato JSON serializable
        datos = []
        for row in rows:
            datos.append({
                "fecha": row['fecha'].isoformat() if row['fecha'] else None,
                "lat": float(row['punto_lat']),
                "lon": float(row['punto_lon']),
                "deformacion_mm": float(row['deformacion_mm']),
                "coherencia": float(row['coherencia']),
                "baseline_dias": row['baseline_temporal']
            })

        conn.close()
        return {"area": area_interes, "datos": datos, "total": len(datos)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo datos históricos: {str(e)}")

@app.post("/prediccion")
async def hacer_prediccion(request: PredictionRequest):
    """Realiza una predicción usando el modelo entrenado."""
    try:
        # Validar que el modelo existe
        model_key = f"{request.area_interes}_{request.tipo_prediccion}"
        if model_key not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Modelo no encontrado: {model_key}. Modelos disponibles: {list(models.keys())}"
            )

        model = models[model_key]

        # Obtener datos recientes para crear secuencia
        conn = get_db_connection()
        cursor = conn.cursor()

        # Determinar fecha de inicio
        fecha_inicio = request.fecha_inicio or (datetime.now() - timedelta(days=request.num_pasos * 6))

        # Obtener secuencia de datos
        cursor.execute("""
            SELECT ds.fecha, ds.punto_lat, ds.punto_lon, ds.deformacion_mm
            FROM deformacion_series ds
            JOIN interferogramas i ON ds.interferograma_id = i.id
            WHERE i.area_interes = %s
            AND ds.fecha >= %s
            AND ds.coherencia > 0.3
            ORDER BY ds.fecha, ds.punto_lat, ds.punto_lon
        """, [request.area_interes, fecha_inicio])

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron datos suficientes para {request.area_interes} desde {fecha_inicio}"
            )

        # Convertir a secuencia temporal
        df = pd.DataFrame(rows)
        fechas_unicas = sorted(df['fecha'].unique())

        if len(fechas_unicas) < request.num_pasos:
            raise HTTPException(
                status_code=400,
                detail=f"Secuencia insuficiente: {len(fechas_unicas)} fechas disponibles, {request.num_pasos} requeridas"
            )

        # Crear grid y secuencia (lógica similar a data_preparation.py)
        lats_unicas = sorted(df['punto_lat'].unique())
        lons_unicas = sorted(df['punto_lon'].unique())

        lat_min, lat_max = lats_unicas[0], lats_unicas[-1]
        lon_min, lon_max = lons_unicas[0], lons_unicas[-1]

        grid_size = 50
        lat_grid = np.linspace(lat_min, lat_max, grid_size)
        lon_grid = np.linspace(lon_min, lon_max, grid_size)

        # Crear secuencia de fotogramas
        fotogramas = []
        for fecha in fechas_unicas[-request.num_pasos:]:
            df_fecha = df[df['fecha'] == fecha]
            if df_fecha.empty:
                continue

            from scipy.interpolate import griddata
            points = df_fecha[['punto_lon', 'punto_lat']].values
            values = df_fecha['deformacion_mm'].values

            lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
            grid_deformacion = griddata(points, values, (lon_mesh, lat_mesh),
                                      method='linear', fill_value=0)
            fotogramas.append(grid_deformacion)

        if len(fotogramas) != request.num_pasos:
            raise HTTPException(
                status_code=400,
                detail=f"No se pudieron crear suficientes fotogramas: {len(fotogramas)} de {request.num_pasos}"
            )

        # Crear tensor de entrada
        sequence = torch.from_numpy(np.stack(fotogramas, axis=0)).float().unsqueeze(0)  # [1, T, H, W]

        # Hacer predicción
        if request.tipo_prediccion == 'regresion':
            prediction = predict_next_frame(model, sequence.squeeze(0))
            resultado = {
                "tipo": "regresion",
                "prediccion": prediction.numpy().tolist(),
                "descripcion": "Fotograma de deformación predicho para el siguiente período"
            }
        elif request.tipo_prediccion == 'clasificacion':
            clase, probabilidades = classify_sequence_state(model, sequence.squeeze(0))
            clases_nombres = ['normal', 'precursor', 'post_terremoto']
            
            # Preparar resultado base
            resultado = {
                "tipo": "clasificacion",
                "clase_predicha": clase,
                "clase_nombre": clases_nombres[clase],
                "probabilidades": probabilidades.numpy().tolist(),
                "descripcion": f"Estado de la falla: {clases_nombres[clase]}"
            }
            
            # Si detecta indicios de sismo, incluir coordenadas
            if clase > 0:  # precursor o post_terremoto
                # Calcular coordenadas del centro del área
                lat_centro = (lat_min + lat_max) / 2
                lon_centro = (lon_min + lon_max) / 2
                
                # Encontrar punto con máxima deformación en el último fotograma
                ultimo_fotograma = fotogramas[-1]
                max_deformacion_idx = np.unravel_index(np.argmax(np.abs(ultimo_fotograma)), ultimo_fotograma.shape)
                lat_max_def = lat_grid[max_deformacion_idx[0]]
                lon_max_def = lon_grid[max_deformacion_idx[1]]
                
                resultado["coordenadas_alerta"] = {
                    "centro_area": {
                        "latitud": float(lat_centro),
                        "longitud": float(lon_centro)
                    },
                    "maxima_deformacion": {
                        "latitud": float(lat_max_def),
                        "longitud": float(lon_max_def),
                        "deformacion_mm": float(ultimo_fotograma[max_deformacion_idx])
                    },
                    "area_analizada": {
                        "lat_min": float(lat_min),
                        "lat_max": float(lat_max),
                        "lon_min": float(lon_min),
                        "lon_max": float(lon_max)
                    }
                }
                resultado["alerta_sismo"] = True
                resultado["descripcion"] += ". ¡ALERTA! Se detectan indicios de actividad sísmica."
            else:
                resultado["alerta_sismo"] = False

        return {
            "area_interes": request.area_interes,
            "fecha_prediccion": datetime.now().isoformat(),
            "secuencia_usada": {
                "fecha_inicio": fechas_unicas[-request.num_pasos].isoformat(),
                "fecha_fin": fechas_unicas[-1].isoformat(),
                "num_fotogramas": request.num_pasos
            },
            "resultado": resultado
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/terremotos/{area_interes}")
async def obtener_terremotos(area_interes: str, dias: int = 365):
    """Obtiene datos de terremotos históricos para un área."""
    try:
        # En un sistema real, esto vendría de una API externa o BD dedicada
        # Por ahora, devolver datos simulados basados en el área

        terremotos_simulados = {
            'falla_anatolia': [
                {
                    'fecha': '2020-01-24T00:00:00',
                    'magnitud': 6.8,
                    'lat': 38.4,
                    'lon': 39.1,
                    'profundidad': 10,
                    'lugar': 'Elazığ, Turquía'
                },
                {
                    'fecha': '2023-02-06T00:00:00',
                    'magnitud': 7.8,
                    'lat': 37.2,
                    'lon': 37.0,
                    'profundidad': 7,
                    'lugar': 'Kahramanmaraş, Turquía'
                }
            ],
            'cinturon_fuego_pacifico': [
                {
                    'fecha': '2020-05-15T00:00:00',
                    'magnitud': 7.5,
                    'lat': -5.8,
                    'lon': -75.3,
                    'profundidad': 110,
                    'lugar': 'Loreto, Perú'
                },
                {
                    'fecha': '2023-06-22T00:00:00',
                    'magnitud': 7.4,
                    'lat': -29.7,
                    'lon': -177.9,
                    'profundidad': 40,
                    'lugar': 'Kermadec Islands'
                }
            ]
        }

        if area_interes not in terremotos_simulados:
            return {"terremotos": [], "mensaje": f"No hay datos de terremotos para {area_interes}"}

        # Filtrar por fecha reciente
        fecha_limite = datetime.now() - timedelta(days=dias)
        terremotos_filtrados = []

        for terremoto in terremotos_simulados[area_interes]:
            fecha_terremoto = datetime.fromisoformat(terremoto['fecha'])
            if fecha_terremoto >= fecha_limite:
                terremotos_filtrados.append(terremoto)

        return {
            "area": area_interes,
            "periodo_dias": dias,
            "terremotos": terremotos_filtrados,
            "total": len(terremotos_filtrados)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo terremotos: {str(e)}")

@app.post("/procesar/nuevos-datos")
async def procesar_nuevos_datos(background_tasks: BackgroundTasks):
    """Endpoint para procesar nuevos datos de InSAR (simulado)."""
    # En un sistema real, esto activaría el pipeline de ingesta
    background_tasks.add_task(procesar_datos_asf_background)

    return {
        "mensaje": "Procesamiento de nuevos datos iniciado",
        "timestamp": datetime.now().isoformat()
    }

async def procesar_datos_asf_background():
    """Función en background para procesar nuevos datos de ASF."""
    # Simular procesamiento
    await asyncio.sleep(2)
    print("Procesamiento de nuevos datos completado")

# Manejador de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Error interno del servidor: {str(exc)}"}
    )

if __name__ == "__main__":
    # Ejecutar servidor
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )