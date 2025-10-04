#!/usr/bin/env python3
"""
Backend unificado para el sistema de IA de monitoreo sísmico.
Punto de entrada principal que coordina todos los componentes.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Crear directorio de logs si no existe
Path('logs').mkdir(exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backend.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Añadir directorios al path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

def setup_directories():
    """Crear directorios necesarios si no existen."""
    dirs = ['data', 'models', 'logs', 'datasets', 'checkpoints', 'temp']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    logger.info("Directorios verificados/creados")

def run_data_collection(args):
    """Ejecutar recolección de datos."""
    logger.info("Iniciando recolección de datos...")
    try:
        try:
            from data_collection.asf_data_acquisition import main as asf_main
        except ImportError:
            from .data_collection.asf_data_acquisition import main as asf_main
        # Aquí se pueden agregar más fuentes de datos
        asf_main()
    except Exception as e:
        logger.error(f"Error en recolección de datos: {e}")
        sys.exit(1)

def run_dataset_generation(args):
    """Ejecutar generación de datasets."""
    logger.info("Iniciando generación de datasets...")
    try:
        try:
            from dataset_generation.generate_synthetic_data import main as gen_main
        except ImportError:
            from .dataset_generation.generate_synthetic_data import main as gen_main
        gen_main()
    except Exception as e:
        logger.error(f"Error en generación de datasets: {e}")
        sys.exit(1)

def run_model_training(args):
    """Ejecutar entrenamiento de modelos."""
    logger.info("Iniciando entrenamiento de modelos...")
    try:
                # Importar módulos de entrenamiento
        try:
            from model_training.train_model import train_classification_model, train_regression_model
        except ImportError:
            # Si no se encuentra, intentar importación desde la raíz
            import sys
            sys.path.insert(0, str(BASE_DIR.parent))
            from train_model import train_classification_model, train_regression_model

        # Determinar la ruta base del proyecto
        current_dir = Path.cwd()
        
        # Buscar el directorio datasets en múltiples ubicaciones posibles
        possible_dataset_dirs = [
            current_dir / "datasets",  # Desde la raíz del proyecto
            current_dir / "backend" / "datasets",  # Desde backend/
            Path(__file__).parent.parent / "datasets",  # Relativo al archivo main.py
        ]
        
        dataset_dir = None
        for possible_dir in possible_dataset_dirs:
            if possible_dir.exists():
                dataset_dir = possible_dir
                break
        
        if dataset_dir is None:
            raise FileNotFoundError("No se encontró el directorio 'datasets' en ninguna ubicación esperada")
        
        # Configurar dispositivo
        device = 'cpu' if getattr(args, 'use_cpu', False) else 'auto'
        
        # Configurar archivo de datos basado en el área y tarea
        if args.task == 'classification':
            h5_file = dataset_dir / f"{args.area}_synthetic_clasificacion.h5"
            save_dir = current_dir / f"checkpoints_{args.task}_{args.area}"
            # Intentar con device, si falla intentar sin device
            try:
                trainer = train_classification_model(
                    h5_file=str(h5_file),
                    save_dir=str(save_dir),
                    num_epochs=args.epochs,
                    batch_size=args.batch_size,
                    chunk_size=args.chunk_size,
                    device=device
                )
            except TypeError as e:
                if 'device' in str(e):
                    # La función no soporta device, intentar sin él
                    trainer = train_classification_model(
                        h5_file=str(h5_file),
                        save_dir=str(save_dir),
                        num_epochs=args.epochs,
                        batch_size=args.batch_size,
                        chunk_size=args.chunk_size
                    )
                else:
                    raise
        elif args.task == 'regression':
            h5_file = dataset_dir / f"{args.area}_synthetic_regresion.h5"
            save_dir = current_dir / f"checkpoints_{args.task}_{args.area}"
            # Intentar con device, si falla intentar sin device
            try:
                trainer = train_regression_model(
                    h5_file=str(h5_file),
                    save_dir=str(save_dir),
                    num_epochs=args.epochs,
                    batch_size=args.batch_size,
                    chunk_size=args.chunk_size,
                    device=device
                )
            except TypeError as e:
                if 'device' in str(e):
                    # La función no soporta device, intentar sin él
                    trainer = train_regression_model(
                        h5_file=str(h5_file),
                        save_dir=str(save_dir),
                        num_epochs=args.epochs,
                        batch_size=args.batch_size,
                        chunk_size=args.chunk_size
                    )
                else:
                    raise

        logger.info(f"Entrenamiento completado para {args.task} en área {args.area}")

    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise SystemExit(1)

def run_weather_prediction(args):
    """Ejecutar servidor de predicción meteorológica."""
    logger.info("Iniciando servidor de predicción meteorológica...")
    try:
        from inference.weather_api_server import app

        host = args.host or "0.0.0.0"
        port = args.weather_port or 8003

        logger.info(f"Servidor meteorológico iniciándose en {host}:{port}")
        app.run(host=host, port=port, debug=True, threaded=True)
    except Exception as e:
        logger.error(f"Error en servidor meteorológico: {e}")
        sys.exit(1)

def run_air_quality_prediction(args):
    """Ejecutar servidor de predicción de calidad del aire."""
    logger.info("Iniciando servidor de calidad del aire...")
    try:
        from inference.dsa_inference_api import app

        host = args.host or "0.0.0.0"
        port = args.air_quality_port or 8004

        logger.info(f"Servidor de calidad del aire iniciándose en {host}:{port}")
        app.run(host=host, port=port, debug=True, threaded=True)
    except Exception as e:
        logger.error(f"Error en servidor de calidad del aire: {e}")
        sys.exit(1)

def run_integrated_analysis(args):
    """Ejecutar análisis integrado (meteorología + nivel del mar + sismos)."""
    logger.info("Iniciando análisis integrado...")
    try:
        from inference.integrated_api_server import app

        host = args.host or "0.0.0.0"
        port = args.integrated_port or 8005

        logger.info(f"Servidor integrado iniciándose en {host}:{port}")
        app.run(host=host, port=port, debug=True, threaded=True)
    except Exception as e:
        logger.error(f"Error en servidor integrado: {e}")
        sys.exit(1)

def run_seismic_inference(args):
    """Ejecutar servidor de inferencia sísmica."""
    logger.info("Iniciando servidor de inferencia sísmica...")
    try:
        import uvicorn
        from inference.api_server import app

        host = args.host or "0.0.0.0"
        port = args.port or 8000

        logger.info(f"Servidor sísmico iniciándose en {host}:{port}")
        uvicorn.run(app, host=host, port=port, reload=True)
    except Exception as e:
        logger.error(f"Error en servidor sísmico: {e}")
        sys.exit(1)

def run_pipeline(args):
    """Ejecutar pipeline completo."""
    logger.info("Ejecutando pipeline completo...")

    # 1. Recolección de datos
    if args.include_data_collection:
        run_data_collection(args)

    # 2. Generación de datasets
    if args.include_dataset_generation:
        run_dataset_generation(args)

    # 3. Entrenamiento de modelos
    if args.include_training:
        run_model_training(args)

    # 4. Iniciar inferencia
    if args.include_inference:
        run_seismic_inference(args)

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Backend unificado de IA para monitoreo sísmico')
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')

    # Comando para recolección de datos
    data_parser = subparsers.add_parser('collect', help='Recolección de datos')
    data_parser.set_defaults(func=run_data_collection)

    # Comando para generación de datasets
    dataset_parser = subparsers.add_parser('generate', help='Generación de datasets')
    dataset_parser.set_defaults(func=run_dataset_generation)

    # Comando para entrenamiento
    train_parser = subparsers.add_parser('train', help='Entrenamiento de modelos')
    train_parser.add_argument('--task', choices=['classification', 'regression'], default='classification')
    train_parser.add_argument('--area', default='falla_anatolia')
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--batch-size', type=int, default=2)  # Reducido de 8 a 2 para evitar OOM
    train_parser.add_argument('--chunk-size', type=int, default=1000)
    train_parser.add_argument('--use-cpu', action='store_true', help='Forzar uso de CPU en lugar de GPU')
    train_parser.set_defaults(func=run_model_training)

    # Comando para inferencia sísmica
    seismic_parser = subparsers.add_parser('seismic', help='Servidor de inferencia sísmica')
    seismic_parser.add_argument('--host', default='0.0.0.0')
    seismic_parser.add_argument('--port', type=int, default=8000)
    seismic_parser.set_defaults(func=run_seismic_inference)

    # Comando para predicción meteorológica
    weather_parser = subparsers.add_parser('weather', help='Servidor de predicción meteorológica')
    weather_parser.add_argument('--host', default='0.0.0.0')
    weather_parser.add_argument('--weather-port', type=int, default=8003)
    weather_parser.set_defaults(func=run_weather_prediction)

    # Comando para calidad del aire
    air_quality_parser = subparsers.add_parser('air-quality', help='Servidor de calidad del aire')
    air_quality_parser.add_argument('--host', default='0.0.0.0')
    air_quality_parser.add_argument('--air-quality-port', type=int, default=8004)
    air_quality_parser.set_defaults(func=run_air_quality_prediction)

    # Comando para análisis integrado
    integrated_parser = subparsers.add_parser('integrated', help='Análisis integrado completo')
    integrated_parser.add_argument('--host', default='0.0.0.0')
    integrated_parser.add_argument('--integrated-port', type=int, default=8005)
    integrated_parser.set_defaults(func=run_integrated_analysis)

    # Comando para pipeline completo
    pipeline_parser = subparsers.add_parser('pipeline', help='Pipeline completo')
    pipeline_parser.add_argument('--include-data-collection', action='store_true')
    pipeline_parser.add_argument('--include-dataset-generation', action='store_true')
    pipeline_parser.add_argument('--include-training', action='store_true')
    pipeline_parser.add_argument('--include-inference', action='store_true')
    pipeline_parser.add_argument('--host', default='0.0.0.0')
    pipeline_parser.add_argument('--port', type=int, default=8000)
    pipeline_parser.set_defaults(func=run_pipeline)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Configurar directorios
    setup_directories()

    # Ejecutar comando
    try:
        if hasattr(args, 'func'):
            args.func(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        logger.info("Interrupción por usuario")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()