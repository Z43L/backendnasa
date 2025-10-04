# Backend de IA para Monitoreo Sísmico

Sistema completo de inteligencia artificial para el monitoreo y predicción de actividad sísmica utilizando datos InSAR de la NASA.

## 📁 Estructura del Proyecto

```
backend/
├── data_collection/          # Scripts de recolección de datos
│   ├── asf_data_acquisition.py
│   └── ...
├── dataset_generation/       # Generación de datasets sintéticos
│   ├── generate_synthetic_data.py
│   └── data_preparation.py
├── model_training/          # Entrenamiento de modelos
│   ├── train_model.py
│   └── model_architecture.py
├── inference/               # APIs de inferencia
│   ├── api_server.py
│   └── real_time_inference.py
├── docker/                  # Configuración Docker
│   ├── Dockerfile
│   └── docker-compose.yml
├── scripts/                 # Scripts de automatización
│   ├── train_models.sh      # Bash (Linux/macOS)
│   └── train_models.ps1     # PowerShell (Windows)
├── config/                  # Configuración
│   └── config.ini
├── main.py                  # Punto de entrada principal
├── requirements.txt         # Dependencias Python
└── README.md               # Esta documentación
```

## 🚀 Inicio Rápido

### Opción 1: Usando Scripts de Automatización

#### Linux/macOS (Bash)
```bash
# Configurar entorno
./scripts/train_models.sh setup

# Ejecutar pipeline completo
./scripts/train_models.sh pipeline

# O entrenar modelo específico
./scripts/train_models.sh train --task classification --epochs 100
```

#### Windows (PowerShell)
```powershell
# Configurar entorno
.\scripts\train_models.ps1 -Command setup

# Ejecutar pipeline completo
.\scripts\train_models.ps1 -Command pipeline

# O entrenar modelo específico
.\scripts\train_models.ps1 -Command train -Task classification -Epochs 100
```

### Opción 2: Usando Docker

```bash
# Construir e iniciar servicios
cd docker
docker-compose up --build -d

# Ver logs
docker-compose logs -f backend
```

### Opción 3: Instalación Manual

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar comandos individuales
python main.py generate   # Generar datasets
python main.py train --task classification  # Entrenar modelo
python main.py serve      # Iniciar API
```

## 📊 Componentes del Sistema

### 1. Recolección de Datos (`data_collection/`)
- **ASF Data Acquisition**: Descarga automática de datos InSAR de Alaska Satellite Facility
- **NASA IMERG**: Datos de precipitación para correlación climática
- **GNSS Data**: Integración con datos GPS para validación

### 2. Generación de Datasets (`dataset_generation/`)
- **Datos Sintéticos**: Generación de patrones tectónicos realistas
- **Aumento de Datos**: Técnicas de data augmentation para datasets limitados
- **Preprocesamiento**: Normalización y preparación de datos

### 3. Entrenamiento de Modelos (`model_training/`)
- **Spatial-Temporal Transformer**: Arquitectura principal para predicción
- **Modelos de Clasificación**: Detección de precursores sísmicos
- **Modelos de Regresión**: Predicción de deformación futura
- **Entrenamiento Optimizado**: Carga segmentada para datasets grandes

### 4. Inferencia (`inference/`)
- **API REST**: Endpoints para predicciones en tiempo real
- **Inferencia por Lotes**: Procesamiento eficiente de múltiples consultas
- **Monitoreo Continuo**: Sistema de alertas automáticas

## 🔧 Configuración

El sistema se configura mediante `config/config.ini`. Variables importantes:

```ini
[models]
d_model = 256              # Dimensión del modelo
num_heads = 8             # Número de cabezas de atención
default_epochs = 50       # Épocas de entrenamiento por defecto

[data]
max_dataset_size_gb = 3.0  # Tamaño máximo de datasets
chunk_size = 1000         # Tamaño de chunks para carga segmentada

[api]
host = 0.0.0.0
port = 8000
```

## 📈 API Endpoints

### Predicción Principal
```http
POST /prediccion
Content-Type: application/json

{
  "area_interes": "falla_anatolia",
  "tipo_prediccion": "clasificacion",
  "num_pasos": 30
}
```

### Respuesta con Coordenadas de Alerta
```json
{
  "area_interes": "falla_anatolia",
  "resultado": {
    "tipo": "clasificacion",
    "clase_predicha": 1,
    "clase_nombre": "precursor",
    "alerta_sismo": true,
    "coordenadas_alerta": {
      "centro_area": {"latitud": 39.5, "longitud": 35.2},
      "maxima_deformacion": {"latitud": 39.7, "longitud": 35.1, "deformacion_mm": 15.3}
    }
  }
}
```

## 🐳 Despliegue con Docker

### Servicios Incluidos
- **Backend API**: Servidor FastAPI principal
- **Base de Datos**: PostgreSQL con PostGIS
- **Cache**: Redis para optimización
- **Monitoreo**: Prometheus para métricas

### Comandos Útiles
```bash
# Ver estado de servicios
docker-compose ps

# Ver logs
docker-compose logs backend

# Ejecutar comandos en el contenedor
docker-compose exec backend bash

# Escalar servicios
docker-compose up -d --scale backend=3
```

## 📊 Monitoreo y Métricas

### Métricas Disponibles
- Rendimiento de modelos (accuracy, F1-score)
- Latencia de inferencia
- Uso de recursos (CPU, memoria, GPU)
- Estado de la base de datos

### Dashboard
Accede a Prometheus en `http://localhost:9090` para visualizar métricas.

## 🔒 Seguridad

### Variables de Entorno Requeridas
```bash
# Base de datos
DB_PASSWORD=your_secure_password

# NASA ASF
ASF_USERNAME=your_asf_username
ASF_PASSWORD=your_asf_password

# GPU (opcional)
CUDA_VISIBLE_DEVICES=0
```

### Mejores Prácticas
- Usa HTTPS en producción
- Implementa rate limiting
- Configura firewall apropiadamente
- Monitorea logs regularmente

## 🚀 Optimizaciones Implementadas

### 1. Carga Segmentada
- Datasets se cargan en chunks para evitar saturar memoria
- Cache LRU para chunks frecuentemente accedidos
- Optimización automática del tamaño de chunks

### 2. Entrenamiento Eficiente
- DataLoader optimizado con prefetching
- Mixed precision training cuando está disponible
- Early stopping y checkpointing automático

### 3. Inferencia Optimizada
- Batch processing para múltiples predicciones
- Cache de modelos en GPU
- Async processing para alta concurrencia

## 📝 Desarrollo

### Añadir Nueva Área de Estudio
1. Actualizar `config/config.ini` con coordenadas
2. Añadir parámetros tectónicos en `generate_synthetic_data.py`
3. Ejecutar pipeline de generación de datos

### Extender la API
1. Añadir endpoints en `inference/api_server.py`
2. Actualizar documentación OpenAPI
3. Añadir tests correspondientes

### Personalizar Modelos
1. Modificar `model_architecture.py`
2. Ajustar hiperparámetros en `config/config.ini`
3. Re-entrenar con scripts de automatización

## 🐛 Solución de Problemas

### Problemas Comunes

**Error de Memoria en Entrenamiento:**
```bash
# Reducir batch size
./scripts/train_models.sh train --batch-size 4

# O usar chunk_size más pequeño
export CHUNK_SIZE=500
```

**Error de Conexión a Base de Datos:**
```bash
# Verificar variables de entorno
echo $DB_PASSWORD

# Reiniciar servicios Docker
docker-compose restart database
```

**Modelo no Converge:**
```bash
# Aumentar épocas
./scripts/train_models.sh train --epochs 200

# O ajustar learning rate en config
```

### Logs y Debugging
```bash
# Ver logs del backend
tail -f logs/backend.log

# Ver logs de Docker
docker-compose logs -f backend

# Debug mode
export LOG_LEVEL=DEBUG
./scripts/train_models.sh serve
```

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo LICENSE para más detalles.

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Soporte

Para soporte técnico:
- 📧 Email: support@seismic-ai.com
- 📖 Documentación: [Wiki del Proyecto](https://github.com/your-org/seismic-ai/wiki)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/seismic-ai/issues)

---

**Desarrollado con ❤️ para la comunidad científica y de respuesta a desastres**