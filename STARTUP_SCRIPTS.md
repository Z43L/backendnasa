# Backend Startup Scripts

Scripts para iniciar y gestionar el backend completo del sistema Seismic AI.

## 📋 Scripts Disponibles

### 1. `start_backend.py` - Inicio Completo del Backend
**Descripción**: Script principal que inicializa todos los componentes del sistema.

**Características**:
- ✅ Inicialización completa de todos los agentes IA
- ✅ Carga de modelos de machine learning
- ✅ Configuración del sistema de alertas
- ✅ Inicio del servidor FastAPI
- ✅ Manejo de señales de parada
- ✅ Logging configurable

**Uso básico**:
```bash
python start_backend.py
```

**Opciones avanzadas**:
```bash
# Desarrollo con auto-reload
python start_backend.py --reload --log-level debug

# Puerto personalizado
python start_backend.py --port 8080 --host 0.0.0.0

# Inicio rápido (sin inicialización completa)
python start_backend.py --skip-init
```

**Variables de entorno**:
- `SEISMIC_AI_HOST`: Host del servidor (default: 127.0.0.1)
- `SEISMIC_AI_PORT`: Puerto del servidor (default: 8000)
- `SEISMIC_AI_RELOAD`: Auto-reload (default: false)
- `SEISMIC_AI_LOG_LEVEL`: Nivel de logging (default: info)

### 2. `dev_server.py` - Servidor de Desarrollo
**Descripción**: Script simplificado optimizado para desarrollo.

**Características**:
- 🔄 Auto-reload habilitado por defecto
- 📝 Logging directo a consola
- ⚡ Inicio rápido
- 🛑 Interrupción fácil con Ctrl+C

**Uso**:
```bash
python dev_server.py
```

### 3. `check_backend.py` - Verificación de Estado
**Descripción**: Script para verificar el estado de todos los componentes.

**Características**:
- 🌐 Verificación de conectividad del servidor
- 📦 Chequeo de dependencias instaladas
- 🤖 Estado de modelos cargados
- 🎯 Estado de agentes IA
- 📊 Reporte detallado o JSON

**Uso**:
```bash
# Verificación completa
python check_backend.py

# Salida JSON para integración con otros scripts
python check_backend.py --json

# Modo silencioso para scripts automatizados
python check_backend.py --quiet
```

## 🚀 Guía de Inicio Rápido

### Para Producción
```bash
cd backend
python start_backend.py --log-level info
```

### Para Desarrollo
```bash
cd backend
python dev_server.py
```

### Verificación de Estado
```bash
cd backend
python check_backend.py
```

## 🔧 Configuración

### Archivo de Variables de Entorno
Crea un archivo `.env` en el directorio `backend/`:

```bash
cp .env.example .env
# Edita las variables según tu entorno
```

### Dependencias
Asegúrate de tener instaladas las dependencias:

```bash
pip install -r requirements.txt
```

## 📊 Estados y Códigos de Salida

### `start_backend.py`
- **0**: Inicio exitoso
- **1**: Error de dependencias o inicialización

### `check_backend.py`
- **0**: Todos los componentes operativos
- **1**: Servidor no responde
- **2**: Dependencias faltantes

## 🐛 Solución de Problemas

### Problema: "Connection refused"
```
❌ Backend status: not_running
```
**Solución**: El servidor no está ejecutándose. Inicia con `python start_backend.py`

### Problema: "Missing dependencies"
```
❌ Missing required modules: torch, fastapi
```
**Solución**: Instala dependencias faltantes con `pip install -r requirements.txt`

### Problema: "Model loading failed"
```
⚠️ Model loader initialization failed
```
**Solución**: Verifica que los archivos de modelo estén en `backend/models/`

### Problema: Puerto ocupado
```
ERROR: [Errno 48] Address already in use
```
**Solución**: Cambia el puerto con `--port 8080` o libera el puerto 8000

## 📝 Logs y Monitoreo

### Ubicación de Logs
- **Inicio del backend**: `backend/logs/backend_startup.log`
- **Servidor FastAPI**: Consola (o archivo si se configura)
- **Modelos y agentes**: Logs del sistema

### Monitoreo en Tiempo Real
```bash
# Ver estado cada 30 segundos
while true; do python check_backend.py --quiet; sleep 30; done
```

## 🔄 Ciclo de Desarrollo

1. **Desarrollo**: `python dev_server.py`
2. **Verificación**: `python check_backend.py`
3. **Testing**: Ejecuta tests automatizados
4. **Producción**: `python start_backend.py --log-level warning`

## 📞 Soporte

Para diagnóstico avanzado:
```bash
# Información completa del sistema
python check_backend.py --json | jq .

# Logs detallados
tail -f backend/logs/backend_startup.log
```