# Dockerfile para el backend de IA de monitoreo sísmico
FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libhdf5-dev \
    libnetcdf-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements primero para aprovechar cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Crear directorios necesarios
RUN mkdir -p /app/data /app/models /app/logs /app/datasets

# Exponer puertos
EXPOSE 8000 8001 8002

# Variables de entorno
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=""

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1

# Comando por defecto
CMD ["python", "main.py"]