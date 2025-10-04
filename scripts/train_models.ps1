# Script de automatización para el backend de IA de monitoreo sísmico
# Compatible con Windows PowerShell

param(
    [Parameter(Mandatory=$false)]
    [string]$Command = "help",

    [Parameter(Mandatory=$false)]
    [string]$Task = "classification",

    [Parameter(Mandatory=$false)]
    [string]$Area = "falla_anatolia",

    [Parameter(Mandatory=$false)]
    [int]$Epochs = 50,

    [Parameter(Mandatory=$false)]
    [int]$BatchSize = 8,

    [Parameter(Mandatory=$false)]
    [string]$HostName = "0.0.0.0",

    [Parameter(Mandatory=$false)]
    [int]$Port = 8000
)

# Configuración
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Split-Path -Parent $ScriptDir
$LogFile = Join-Path $BackendDir "logs\training_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Función de logging
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $Color = switch ($Level) {
        "ERROR" { "Red" }
        "WARNING" { "Yellow" }
        "INFO" { "Blue" }
        "SUCCESS" { "Green" }
        default { "White" }
    }
    $LogMessage = "[$Timestamp] [$Level] $Message"
    Write-Host $LogMessage -ForegroundColor $Color
    Add-Content -Path $LogFile -Value $LogMessage
}

function Write-Success { param([string]$Message) Write-Log $Message "SUCCESS" }
function Write-Error { param([string]$Message) Write-Log $Message "ERROR" }
function Write-Warning { param([string]$Message) Write-Log $Message "WARNING" }
function Write-Info { param([string]$Message) Write-Log $Message "INFO" }

# Función para verificar dependencias
function Test-Dependencies {
    Write-Info "Verificando dependencias..."

    # Verificar Python
    try {
        $pythonVersion = python --version 2>$null
        if ($LASTEXITCODE -ne 0) { throw "Python no encontrado" }
        Write-Success "Python encontrado: $pythonVersion"
    } catch {
        Write-Error "Python 3 no está instalado o no está en PATH"
        exit 1
    }

    # Verificar Docker
    try {
        $dockerVersion = docker --version 2>$null
        Write-Success "Docker encontrado: $dockerVersion"
    } catch {
        Write-Warning "Docker no está instalado - algunas funciones estarán limitadas"
    }

    # Verificar Docker Compose
    try {
        docker-compose --version >$null 2>&1
        Write-Success "Docker Compose encontrado"
    } catch {
        try {
            docker compose version >$null 2>&1
            Write-Success "Docker Compose V2 encontrado"
        } catch {
            Write-Warning "Docker Compose no está disponible"
        }
    }

    Write-Success "Dependencias verificadas"
}

# Función para configurar entorno
function Initialize-Environment {
    Write-Info "Configurando entorno..."

    Set-Location $BackendDir

    # Crear directorios necesarios
    $dirs = @("data", "models", "logs", "datasets", "checkpoints", "temp")
    foreach ($dir in $dirs) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }

    # Verificar/crear entorno virtual
    if (!(Test-Path "venv")) {
        Write-Info "Creando entorno virtual..."
        python -m venv venv
    }

    # Activar entorno virtual y instalar dependencias
    & ".\venv\Scripts\Activate.ps1"

    if (Test-Path "requirements.txt") {
        Write-Info "Instalando dependencias Python..."
        pip install -r requirements.txt
    }

    Write-Success "Entorno configurado"
}

# Función para ejecutar pipeline completo
function Invoke-FullPipeline {
    Write-Info "Ejecutando pipeline completo de IA sísmica..."

    Set-Location $BackendDir

    # Activar entorno virtual
    & ".\venv\Scripts\Activate.ps1"

    try {
        # 1. Generar datasets
        Write-Info "Paso 1: Generando datasets sintéticos..."
        python main.py generate

        # 2. Entrenar modelo de clasificación
        Write-Info "Paso 2: Entrenando modelo de clasificación..."
        python main.py train --task classification --area falla_anatolia --epochs 100 --batch-size 8

        # 3. Entrenar modelo de regresión
        Write-Info "Paso 3: Entrenando modelo de regresión..."
        python main.py train --task regression --area falla_anatolia --epochs 100 --batch-size 8

        # 4. Iniciar servidor de inferencia
        Write-Info "Paso 4: Iniciando servidor de inferencia..."
        $inferenceJob = Start-Job -ScriptBlock {
            Set-Location $using:BackendDir
            & ".\venv\Scripts\Activate.ps1"
            python main.py serve --host 0.0.0.0 --port 8000
        }

        # Guardar ID del job
        $inferenceJob.Id | Out-File -FilePath (Join-Path $BackendDir "inference.pid") -Encoding ASCII

        Write-Success "Pipeline completado. Servidor ejecutándose en background (Job ID: $($inferenceJob.Id))"

    } catch {
        Write-Error "Error en pipeline: $($_.Exception.Message)"
        throw
    }
}

# Función para entrenamiento específico
function Start-ModelTraining {
    param(
        [string]$TaskType = "classification",
        [string]$AreaName = "falla_anatolia",
        [int]$NumEpochs = 50,
        [int]$BatchSizeValue = 8
    )

    Write-Info "Entrenando modelo $TaskType para área $AreaName..."

    Set-Location $BackendDir
    & ".\venv\Scripts\Activate.ps1"

    try {
        python main.py train `
            --task $TaskType `
            --area $AreaName `
            --epochs $NumEpochs `
            --batch-size $BatchSizeValue
    } catch {
        Write-Error "Error en entrenamiento: $($_.Exception.Message)"
        throw
    }
}

# Función para iniciar servicios con Docker
function Start-DockerServices {
    Write-Info "Iniciando servicios con Docker..."

    Set-Location (Join-Path $BackendDir "docker")

    try {
        # Intentar Docker Compose V1
        docker-compose up -d 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Servicios Docker iniciados (Docker Compose V1)"
            return
        }

        # Intentar Docker Compose V2
        docker compose up -d 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Servicios Docker iniciados (Docker Compose V2)"
            return
        }

        throw "No se pudo iniciar Docker Compose"

    } catch {
        Write-Error "Docker Compose no disponible o error al iniciar servicios"
        throw
    }
}

# Función para detener servicios
function Stop-Services {
    Write-Info "Deteniendo servicios..."

    # Detener job de inferencia local
    $pidFile = Join-Path $BackendDir "inference.pid"
    if (Test-Path $pidFile) {
        try {
            $jobId = Get-Content $pidFile -Raw
            Stop-Job -Id $jobId -ErrorAction SilentlyContinue
            Remove-Job -Id $jobId -ErrorAction SilentlyContinue
            Remove-Item $pidFile -Force
            Write-Success "Servidor de inferencia detenido"
        } catch {
            Write-Warning "No se pudo detener el servidor de inferencia"
        }
    }

    # Detener servicios Docker
    Set-Location (Join-Path $BackendDir "docker")
    try {
        docker-compose down 2>$null
        if ($LASTEXITCODE -ne 0) {
            docker compose down 2>$null
        }
    } catch {
        # Ignorar errores
    }
}

# Función para limpiar datos temporales
function Clear-TempData {
    Write-Info "Limpiando datos temporales..."

    Set-Location $BackendDir

    # Limpiar logs antiguos (mantener últimos 7 días)
    Get-ChildItem -Path "logs\*.log" -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
        Remove-Item -Force

    # Limpiar checkpoints antiguos (mantener últimos 5)
    if (Test-Path "checkpoints") {
        Get-ChildItem -Path "checkpoints\checkpoint_*.pth" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -Skip 5 |
            Remove-Item -Force
    }

    # Limpiar archivos temporales
    if (Test-Path "temp") {
        Remove-Item -Path "temp\*" -Recurse -Force -ErrorAction SilentlyContinue
    }

    Write-Success "Limpieza completada"
}

# Función para mostrar estado del sistema
function Show-SystemStatus {
    Write-Info "Estado del sistema de IA sísmica:"

    # Verificar procesos
    $pidFile = Join-Path $BackendDir "inference.pid"
    if (Test-Path $pidFile) {
        try {
            $jobId = Get-Content $pidFile -Raw
            $job = Get-Job -Id $jobId -ErrorAction SilentlyContinue
            if ($job -and $job.State -eq "Running") {
                Write-Success "✓ Servidor de inferencia ejecutándose (Job ID: $jobId)"
            } else {
                Write-Error "✗ Servidor de inferencia no ejecutándose"
            }
        } catch {
            Write-Error "✗ Error al verificar servidor de inferencia"
        }
    } else {
        Write-Error "✗ Servidor de inferencia no ejecutándose"
    }

    # Verificar modelos
    $modelsDir = Join-Path $BackendDir "models"
    if (Test-Path $modelsDir) {
        $modelFiles = Get-ChildItem -Path "$modelsDir\*.pth" -ErrorAction SilentlyContinue
        if ($modelFiles) {
            Write-Success "✓ Modelos encontrados: $($modelFiles.Count) archivos"
        } else {
            Write-Error "✗ No se encontraron modelos entrenados"
        }
    } else {
        Write-Error "✗ Directorio de modelos no existe"
    }

    # Verificar datasets
    $datasetsDir = Join-Path $BackendDir "datasets"
    if (Test-Path $datasetsDir) {
        $datasetFiles = Get-ChildItem -Path "$datasetsDir\*.h5" -ErrorAction SilentlyContinue
        if ($datasetFiles) {
            Write-Success "✓ Datasets encontrados: $($datasetFiles.Count) archivos"
        } else {
            Write-Error "✗ No se encontraron datasets"
        }
    } else {
        Write-Error "✗ Directorio de datasets no existe"
    }

    # Verificar servicios Docker
    try {
        $runningContainers = docker ps --filter "label=com.seismic.backend" --format "{{.Names}}" 2>$null
        if ($LASTEXITCODE -eq 0 -and $runningContainers) {
            $containerCount = ($runningContainers | Measure-Object -Line).Lines
            Write-Success "✓ Servicios Docker ejecutándose: $containerCount contenedores"
        } else {
            Write-Info "ℹ No hay servicios Docker ejecutándose"
        }
    } catch {
        Write-Info "ℹ Docker no disponible para verificación"
    }
}

# Función para mostrar ayuda
function Show-Help {
    @"
Script de automatización para el backend de IA de monitoreo sísmico

Uso: .\train_models.ps1 [-Command] <comando> [opciones]

Comandos disponibles:
    setup           Configurar entorno y dependencias
    pipeline        Ejecutar pipeline completo (generar datasets + entrenar + servir)
    train           Entrenar modelo específico
        -Task       Tipo de tarea (classification|regression) [default: classification]
        -Area       Área geográfica [default: falla_anatolia]
        -Epochs     Número de épocas [default: 50]
        -BatchSize  Tamaño del batch [default: 8]
    serve           Iniciar servidor de inferencia
        -HostName   Host para el servidor [default: 0.0.0.0]
        -Port       Puerto del servidor [default: 8000]
    docker-up       Iniciar servicios con Docker
    docker-down     Detener servicios Docker
    status          Mostrar estado del sistema
    cleanup         Limpiar archivos temporales
    help            Mostrar esta ayuda

Ejemplos:
    .\train_models.ps1 -Command setup
    .\train_models.ps1 -Command pipeline
    .\train_models.ps1 -Command train -Task classification -Epochs 100
    .\train_models.ps1 -Command serve -Port 8080
    .\train_models.ps1 -Command docker-up

Variables de entorno:
    CUDA_VISIBLE_DEVICES    Dispositivos CUDA a usar
    DB_PASSWORD            Contraseña de la base de datos

Logs disponibles en: $BackendDir\logs\
"@
}

# Ejecutar comando
switch ($Command) {
    "setup" {
        Test-Dependencies
        Initialize-Environment
    }
    "pipeline" {
        Test-Dependencies
        Initialize-Environment
        Invoke-FullPipeline
    }
    "train" {
        Test-Dependencies
        Initialize-Environment
        Start-ModelTraining -TaskType $Task -AreaName $Area -NumEpochs $Epochs -BatchSizeValue $BatchSize
    }
    "serve" {
        Test-Dependencies
        Initialize-Environment
        Set-Location $BackendDir
        & ".\venv\Scripts\Activate.ps1"
        python main.py serve --host $HostName --port $Port
    }
    "docker-up" {
        Start-DockerServices
    }
    "docker-down" {
        Stop-Services
    }
    "status" {
        Show-SystemStatus
    }
    "cleanup" {
        Clear-TempData
    }
    "help" {
        Show-Help
    }
    default {
        Write-Error "Comando desconocido: $Command"
        Write-Host ""
        Show-Help
        exit 1
    }
}