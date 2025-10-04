-- Inicialización de la base de datos para el sistema de monitoreo sísmico

-- Crear usuario si no existe
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'seismic_user') THEN
      CREATE USER seismic_user WITH PASSWORD 'seismic2024';
   END IF;
END
$$;

-- Otorgar permisos
GRANT ALL PRIVILEGES ON DATABASE deformacion_monitor TO seismic_user;
GRANT ALL ON SCHEMA public TO seismic_user;

-- Crear extensiones necesarias
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Crear tabla de interferogramas
CREATE TABLE IF NOT EXISTS interferogramas (
    id SERIAL PRIMARY KEY,
    nombre_archivo VARCHAR(255) NOT NULL,
    fecha_adquisicion TIMESTAMP NOT NULL,
    area_interes VARCHAR(100) NOT NULL,
    orbita INTEGER,
    frame INTEGER,
    modo VARCHAR(50),
    procesado_en TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    UNIQUE(nombre_archivo)
);

-- Crear tabla de series de deformación
CREATE TABLE IF NOT EXISTS deformacion_series (
    id SERIAL PRIMARY KEY,
    interferograma_id INTEGER REFERENCES interferogramas(id),
    fecha TIMESTAMP NOT NULL,
    punto_lat DOUBLE PRECISION NOT NULL,
    punto_lon DOUBLE PRECISION NOT NULL,
    deformacion_mm DOUBLE PRECISION NOT NULL,
    coherencia DOUBLE PRECISION DEFAULT 1.0,
    fase_radianes DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Índices para optimización
    CONSTRAINT deformacion_series_geom UNIQUE (punto_lat, punto_lon, fecha)
);

-- Crear índices espaciales
CREATE INDEX IF NOT EXISTS idx_deformacion_series_geom
    ON deformacion_series USING GIST (ST_Point(punto_lon, punto_lat));

CREATE INDEX IF NOT EXISTS idx_deformacion_series_fecha
    ON deformacion_series (fecha);

CREATE INDEX IF NOT EXISTS idx_deformacion_series_area
    ON deformacion_series (interferograma_id);

-- Crear tabla de modelos entrenados
CREATE TABLE IF NOT EXISTS modelos_entrenados (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    tipo VARCHAR(50) NOT NULL, -- 'clasificacion' o 'regresion'
    area_interes VARCHAR(100) NOT NULL,
    ruta_modelo VARCHAR(500) NOT NULL,
    metadatos JSONB,
    accuracy DOUBLE PRECISION,
    precision_score DOUBLE PRECISION,
    recall_score DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    fecha_entrenamiento TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version VARCHAR(50)
);

-- Crear tabla de predicciones
CREATE TABLE IF NOT EXISTS predicciones (
    id SERIAL PRIMARY KEY,
    modelo_id INTEGER REFERENCES modelos_entrenados(id),
    area_interes VARCHAR(100) NOT NULL,
    fecha_prediccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tipo_prediccion VARCHAR(50) NOT NULL,
    resultado JSONB NOT NULL,
    confianza DOUBLE PRECISION,
    alerta_generada BOOLEAN DEFAULT FALSE
);

-- Crear tabla de alertas sísmicas
CREATE TABLE IF NOT EXISTS alertas_sismicas (
    id SERIAL PRIMARY KEY,
    prediccion_id INTEGER REFERENCES predicciones(id),
    nivel_alerta VARCHAR(20) NOT NULL, -- 'bajo', 'medio', 'alto', 'critico'
    descripcion TEXT,
    coordenadas JSONB,
    area_afectada JSONB,
    fecha_alerta TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    atendida BOOLEAN DEFAULT FALSE,
    fecha_atencion TIMESTAMP
);

-- Convertir tablas a hypertables de TimescaleDB
SELECT create_hypertable('deformacion_series', 'fecha', if_not_exists => TRUE);
SELECT create_hypertable('predicciones', 'fecha_prediccion', if_not_exists => TRUE);
SELECT create_hypertable('alertas_sismicas', 'fecha_alerta', if_not_exists => TRUE);

-- Crear índices en hypertables
CREATE INDEX IF NOT EXISTS idx_deformacion_series_fecha_hypertable
    ON deformacion_series (fecha DESC);

CREATE INDEX IF NOT EXISTS idx_predicciones_fecha_hypertable
    ON predicciones (fecha_prediccion DESC);

-- Crear vistas útiles
CREATE OR REPLACE VIEW vista_deformacion_reciente AS
SELECT
    ds.*,
    i.area_interes,
    i.fecha_adquisicion as fecha_interferograma
FROM deformacion_series ds
JOIN interferogramas i ON ds.interferograma_id = i.id
WHERE ds.fecha >= NOW() - INTERVAL '30 days'
ORDER BY ds.fecha DESC;

-- Vista de estadísticas de modelos
CREATE OR REPLACE VIEW vista_estadisticas_modelos AS
SELECT
    tipo,
    area_interes,
    COUNT(*) as num_modelos,
    AVG(accuracy) as accuracy_promedio,
    MAX(fecha_entrenamiento) as ultimo_entrenamiento,
    AVG(EXTRACT(EPOCH FROM (NOW() - fecha_entrenamiento))/86400) as dias_desde_entrenamiento
FROM modelos_entrenados
GROUP BY tipo, area_interes;

-- Vista de alertas activas
CREATE OR REPLACE VIEW vista_alertas_activas AS
SELECT
    a.*,
    p.area_interes,
    p.fecha_prediccion,
    p.resultado
FROM alertas_sismicas a
JOIN predicciones p ON a.prediccion_id = p.id
WHERE a.atendida = FALSE
ORDER BY a.fecha_alerta DESC;

-- Insertar datos de ejemplo
INSERT INTO interferogramas (nombre_archivo, fecha_adquisicion, area_interes, orbita, frame, modo)
VALUES
    ('S1A_IW_SLC_20231001_FallaAnatolia', '2023-10-01 00:00:00', 'falla_anatolia', 15, 45, 'IW'),
    ('S1A_IW_SLC_20231008_FallaAnatolia', '2023-10-08 00:00:00', 'falla_anatolia', 15, 45, 'IW')
ON CONFLICT (nombre_archivo) DO NOTHING;

-- Otorgar permisos finales
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO seismic_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO seismic_user;

-- Crear función para limpieza automática de datos antiguos
CREATE OR REPLACE FUNCTION limpiar_datos_antiguos()
RETURNS void AS $$
BEGIN
    -- Eliminar datos de deformación mayores a 1 año
    DELETE FROM deformacion_series
    WHERE fecha < NOW() - INTERVAL '1 year';

    -- Eliminar predicciones mayores a 6 meses
    DELETE FROM predicciones
    WHERE fecha_prediccion < NOW() - INTERVAL '6 months';

    -- Marcar alertas antiguas como atendidas
    UPDATE alertas_sismicas
    SET atendida = TRUE, fecha_atencion = NOW()
    WHERE fecha_alerta < NOW() - INTERVAL '30 days'
    AND atendida = FALSE;
END;
$$ LANGUAGE plpgsql;

-- Crear job programado para limpieza (se ejecuta semanalmente)
-- Nota: Requiere pg_cron extension si está disponible
-- SELECT cron.schedule('limpieza-semanal', '0 2 * * 0', 'SELECT limpiar_datos_antiguos();');

COMMENT ON DATABASE deformacion_monitor IS 'Base de datos para el sistema de IA de monitoreo sísmico';
COMMENT ON TABLE deformacion_series IS 'Series temporales de deformación del terreno';
COMMENT ON TABLE interferogramas IS 'Metadatos de interferogramas procesados';
COMMENT ON TABLE modelos_entrenados IS 'Modelos de IA entrenados y sus métricas';
COMMENT ON TABLE predicciones IS 'Historial de predicciones realizadas';
COMMENT ON TABLE alertas_sismicas IS 'Alertas sísmicas generadas por el sistema';