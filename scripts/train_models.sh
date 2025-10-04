#!/bin/bash
# Script de automatización para el backend de IA de monitoreo sísmico
# Compatible con Linux/macOS

set -e  # Salir en caso de error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuración
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$BACKEND_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"

# Función de logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2 | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}" | tee -a "$LOG_FILE"
}

# Función para verificar dependencias
check_dependencies() {
    log "Verificando dependencias..."

    # Verificar Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 no está instalado"
        exit 1
    fi

    # Verificar Docker
    if ! command -v docker &> /dev/null; then
        warning "Docker no está instalado - algunas funciones estarán limitadas"
    fi

    # Verificar Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        warning "Docker Compose no está disponible"
    fi

    log "Dependencias verificadas"
}

# Función para configurar entorno
setup_environment() {
    log "Configurando entorno..."

    cd "$BACKEND_DIR"

    # Crear directorios necesarios
    mkdir -p data models logs datasets checkpoints temp

    # Verificar/crear entorno virtual
    if [ ! -d "venv" ]; then
        log "Creando entorno virtual..."
        python3 -m venv venv
    fi

    # Activar entorno virtual
    source venv/bin/activate

    # Instalar dependencias
    if [ -f "requirements.txt" ]; then
        log "Instalando dependencias Python..."
        pip install -r requirements.txt
    fi

    log "Entorno configurado"
}

# Función para ejecutar pipeline completo
run_full_pipeline() {
    log "Ejecutando pipeline completo de IA sísmica..."

    cd "$BACKEND_DIR"

    # Activar entorno virtual
    source venv/bin/activate

    # 1. Generar datasets
    log "Paso 1: Generando datasets sintéticos..."
    python main.py generate

    # 2. Entrenar modelo de clasificación
    log "Paso 2: Entrenando modelo de clasificación..."
    python main.py train --task classification --area falla_anatolia --epochs 100 --batch-size 8

    # 3. Entrenar modelo de regresión
    log "Paso 3: Entrenando modelo de regresión..."
    python main.py train --task regression --area falla_anatolia --epochs 100 --batch-size 8

    # 4. Iniciar servidor de inferencia
    log "Paso 4: Iniciando servidor de inferencia..."
    nohup python main.py serve --host 0.0.0.0 --port 8000 > logs/inference.log 2>&1 &
    echo $! > inference.pid

    log "Pipeline completado. Servidor ejecutándose en background (PID: $(cat inference.pid))"
}

# Función para entrenamiento específico
train_model() {
    local task=${1:-classification}
    local area=${2:-falla_anatolia}
    local epochs=${3:-50}
    local batch_size=${4:-8}

    log "Entrenando modelo $task para área $area..."

    cd "$BACKEND_DIR"
    source venv/bin/activate

    python main.py train \
        --task "$task" \
        --area "$area" \
        --epochs "$epochs" \
        --batch-size "$batch_size"
}

# Función para iniciar servicios con Docker
start_docker_services() {
    log "Iniciando servicios con Docker..."

    cd "$BACKEND_DIR/docker"

    if command -v docker-compose &> /dev/null; then
        docker-compose up -d
    elif docker compose version &> /dev/null; then
        docker compose up -d
    else
        error "Docker Compose no disponible"
        exit 1
    fi

    log "Servicios Docker iniciados"
}

# Función para detener servicios
stop_services() {
    log "Deteniendo servicios..."

    # Detener proceso de inferencia local
    if [ -f "$BACKEND_DIR/inference.pid" ]; then
        kill "$(cat "$BACKEND_DIR/inference.pid")" 2>/dev/null || true
        rm -f "$BACKEND_DIR/inference.pid"
        log "Servidor de inferencia detenido"
    fi

    # Detener servicios Docker
    cd "$BACKEND_DIR/docker"
    if command -v docker-compose &> /dev/null; then
        docker-compose down
    elif docker compose version &> /dev/null; then
        docker compose down
    fi
}

# Función para limpiar datos temporales
cleanup() {
    log "Limpiando datos temporales..."

    cd "$BACKEND_DIR"

    # Limpiar logs antiguos (mantener últimos 7 días)
    find logs -name "*.log" -mtime +7 -delete 2>/dev/null || true

    # Limpiar checkpoints antiguos (mantener últimos 5)
    if [ -d "checkpoints" ]; then
        ls -t checkpoints/checkpoint_*.pth 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null || true
    fi

    # Limpiar archivos temporales
    rm -rf temp/*

    log "Limpieza completada"
}

# Función para mostrar estado del sistema
show_status() {
    log "Estado del sistema de IA sísmica:"

    # Verificar procesos
    if [ -f "$BACKEND_DIR/inference.pid" ] && kill -0 "$(cat "$BACKEND_DIR/inference.pid")" 2>/dev/null; then
        info "✓ Servidor de inferencia ejecutándose (PID: $(cat "$BACKEND_DIR/inference.pid"))"
    else
        warning "✗ Servidor de inferencia no ejecutándose"
    fi

    # Verificar modelos
    if [ -d "$BACKEND_DIR/models" ] && [ "$(ls -A "$BACKEND_DIR/models" 2>/dev/null)" ]; then
        info "✓ Modelos encontrados: $(ls "$BACKEND_DIR/models"/*.pth 2>/dev/null | wc -l) archivos"
    else
        warning "✗ No se encontraron modelos entrenados"
    fi

    # Verificar datasets
    if [ -d "$BACKEND_DIR/datasets" ] && [ "$(ls -A "$BACKEND_DIR/datasets" 2>/dev/null)" ]; then
        info "✓ Datasets encontrados: $(ls "$BACKEND_DIR/datasets"/*.h5 2>/dev/null | wc -l) archivos"
    else
        warning "✗ No se encontraron datasets"
    fi

    # Verificar servicios Docker
    if command -v docker &> /dev/null; then
        running_containers=$(docker ps --filter "label=com.seismic.backend" --format "{{.Names}}" | wc -l)
        if [ "$running_containers" -gt 0 ]; then
            info "✓ Servicios Docker ejecutándose: $running_containers contenedores"
        else
            info "ℹ No hay servicios Docker ejecutándose"
        fi
    fi
}

# Función para mostrar ayuda
show_help() {
    cat << EOF
Script de automatización para el backend de IA de monitoreo sísmico

Uso: $0 [comando] [opciones]

Comandos disponibles:
    setup           Configurar entorno y dependencias
    pipeline        Ejecutar pipeline completo (generar datasets + entrenar + servir)
    train           Entrenar modelo específico
        --task      Tipo de tarea (classification|regression) [default: classification]
        --area      Área geográfica [default: falla_anatolia]
        --epochs    Número de épocas [default: 50]
        --batch     Tamaño del batch [default: 8]
    serve           Iniciar servidor de inferencia
        --host      Host para el servidor [default: 0.0.0.0]
        --port      Puerto del servidor [default: 8000]
    docker-up       Iniciar servicios con Docker
    docker-down     Detener servicios Docker
    status          Mostrar estado del sistema
    cleanup         Limpiar archivos temporales
    help            Mostrar esta ayuda

Ejemplos:
    $0 setup
    $0 pipeline
    $0 train --task classification --epochs 100
    $0 serve --port 8080
    $0 docker-up

Variables de entorno:
    CUDA_VISIBLE_DEVICES    Dispositivos CUDA a usar
    DB_PASSWORD            Contraseña de la base de datos

Logs disponibles en: $BACKEND_DIR/logs/
EOF
}

# Parsear argumentos
case "${1:-help}" in
    setup)
        check_dependencies
        setup_environment
        ;;
    pipeline)
        check_dependencies
        setup_environment
        run_full_pipeline
        ;;
    train)
        shift
        while [[ $# -gt 0 ]]; do
            case $1 in
                --task) TASK="$2"; shift 2 ;;
                --area) AREA="$2"; shift 2 ;;
                --epochs) EPOCHS="$2"; shift 2 ;;
                --batch) BATCH_SIZE="$2"; shift 2 ;;
                *) error "Opción desconocida: $1"; exit 1 ;;
            esac
        done
        check_dependencies
        setup_environment
        train_model "${TASK:-classification}" "${AREA:-falla_anatolia}" "${EPOCHS:-50}" "${BATCH_SIZE:-8}"
        ;;
    serve)
        shift
        HOST="0.0.0.0"
        PORT="8000"
        while [[ $# -gt 0 ]]; do
            case $1 in
                --host) HOST="$2"; shift 2 ;;
                --port) PORT="$2"; shift 2 ;;
                *) error "Opción desconocida: $1"; exit 1 ;;
            esac
        done
        check_dependencies
        setup_environment
        cd "$BACKEND_DIR"
        source venv/bin/activate
        python main.py serve --host "$HOST" --port "$PORT"
        ;;
    docker-up)
        start_docker_services
        ;;
    docker-down)
        stop_services
        ;;
    status)
        show_status
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        error "Comando desconocido: $1"
        echo
        show_help
        exit 1
        ;;
esac