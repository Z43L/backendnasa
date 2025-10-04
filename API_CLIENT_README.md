# Seismic AI API Client Demo

Este script demuestra cómo interactuar con el servidor API completo de Seismic AI, que integra todas las capacidades de IA sísmica desarrolladas en el proyecto.

## 🚀 Inicio Rápido

### 1. Iniciar el Servidor API
```bash
# Asegúrate de que el servidor esté ejecutándose
python simple_api_server.py
```

### 2. Ejecutar la Demo del Cliente
```bash
python api_client_demo.py
```

## 📋 Funcionalidades Demostradas

### ✅ Verificación de Salud del Sistema
- Estado general del sistema
- Número de alertas activas
- Modelos cargados

### 🌦️ Predicción Meteorológica
- Predicciones de precipitación para cualquier coordenada
- Pronósticos a corto plazo (hasta 24 horas)
- Basado en datos satelitales NASA IMERG

### 🌬️ Predicción de Calidad del Aire
- Índice de calidad del aire (AQI)
- Niveles de contaminación
- Monitoreo ambiental

### 🏔️ Evaluación de Riesgo Sísmico
- Análisis de riesgo sísmico por coordenadas
- Puntajes de riesgo cuantitativos
- Evaluación de zonas de falla

### 🔬 Análisis Integrado
- Combinación de datos meteorológicos, sísmicos y ambientales
- Análisis completo de riesgos ambientales
- Evaluación de condiciones costeras

### 🚨 Sistema de Alertas
- Creación de alertas en tiempo real
- Diferentes niveles de severidad (bajo, medio, alto, crítico)
- Alertas por tipo (sísmico, meteorológico, integrado)

### 📊 Escenario de Monitoreo Completo
- Monitoreo simultáneo de múltiples ubicaciones
- Detección automática de condiciones críticas
- Generación automática de alertas

## 🛠️ Uso Programático

```python
from api_client_demo import SeismicAIClient

# Crear cliente
client = SeismicAIClient("http://127.0.0.1:8000")

# Verificar salud del sistema
health = client.get_health()
print(f"Estado: {health.get('status')}")

# Predecir clima
weather = client.predict_weather(35.0, -118.0, 24)
print(f"Precipitación esperada: {weather.get('prediction', {}).get('precipitation', 0)} mm/h")

# Evaluar riesgo sísmico
risk = client.check_seismic_risk(35.0, -118.0)
print(f"Nivel de riesgo: {risk.get('risk_assessment', {}).get('risk_level')}")

# Crear alerta
alert = client.create_alert(
    alert_type="seismic",
    level="high",
    title="Actividad Sísmica Detectada",
    message="Se detectó actividad sísmica significativa en la zona",
    lat=35.0,
    lon=-118.0
)
```

## 🌐 Endpoints de la API

### Predicciones
- `POST /predict/weather` - Predicción meteorológica
- `POST /predict/air-quality` - Calidad del aire
- `POST /predict/seismic` - Actividad sísmica

### Monitoreo
- `POST /monitor/check/seismic` - Verificación de riesgo sísmico
- `POST /monitor/alert` - Crear alerta
- `GET /monitor/alerts/active` - Alertas activas

### Análisis Integrado
- `POST /analyze/integrated` - Análisis completo

### Sistema
- `GET /health` - Estado del sistema
- `GET /docs` - Documentación interactiva (FastAPI)

## 📊 Datos de Ejemplo

El script incluye coordenadas de prueba para:
- **Costa de California**: `35.0, -118.0` (zona de fallas)
- **Mediterráneo**: `40.0, 4.0` (zona costera)
- **Anillo de Fuego del Pacífico**: `-20.0, -175.0` (alta actividad sísmica)

## 🔧 Requisitos

- Python 3.8+
- Servidor API ejecutándose
- Conexión a internet (para algunas funcionalidades)
- Bibliotecas: `requests`, `numpy`, `matplotlib`

## 📈 Salida Esperada

```
🌟 Seismic AI API Client Demo
========================================

1. System Health Check
✅ System Status: healthy
📊 Active Alerts: 0
🤖 Models Loaded: 3

2. Weather Prediction
✅ Weather prediction successful
   Coordinates: 35.0, -118.0
   Hours ahead: 24

3. Air Quality Prediction
✅ Air Quality: AQI 45 - Good

4. Seismic Risk Assessment
✅ Seismic Risk: MEDIUM
   Risk Score: 0.723

5. Integrated Analysis
✅ Integrated analysis completed
   seismic: Available
   weather: Available
   air_quality: Available

6. Alert Creation
✅ Alert created: alert_123456

7. Active Alerts Check
📊 Active alerts: 1

🧪 Seismic Prediction Demo
------------------------------
📊 Mock seismic data shape: (1, 30, 50, 50)
✅ Seismic prediction successful!

📡 Complete Monitoring Scenario
-----------------------------------
🔍 Monitoring locations:
   📍 California Coast: 36.7783, -119.4179
   📍 Mediterranean: 40.0, 4.0
   📍 Pacific Ring: -20.0, -175.0

🏠 Analyzing California Coast...
✅ California Coast conditions normal

🎉 Demo completed successfully!
📚 Check the API documentation at http://127.0.0.1:8000/docs
```

## 🚨 Manejo de Errores

El cliente incluye manejo robusto de errores:
- Timeouts de red
- Respuestas de error del servidor
- Datos faltantes
- Excepciones durante la ejecución

## 🔄 Próximos Pasos

1. **Documentación Completa**: Visita `http://127.0.0.1:8000/docs`
2. **Pruebas Avanzadas**: Ejecuta `python test_complete_api.py`
3. **Desarrollo Frontend**: Integra con aplicaciones web
4. **Despliegue**: Configura para producción

## 📞 Soporte

Para problemas o preguntas:
1. Verifica que el servidor esté ejecutándose
2. Revisa los logs del servidor
3. Consulta la documentación de la API
4. Ejecuta las pruebas unitarias

---

**Proyecto NASA Hackathon** - Sistema de IA Sísmica Completo