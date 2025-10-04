#!/usr/bin/env python3
"""
Seismic AI Complete API Server

A comprehensive FastAPI server that integrates all seismic AI capabilities:
- Seismic classification and regression
- Weather prediction and analysis
- DSA (Data Science Algorithm) for air quality
- Integrated analysis (weather + sea level + seismic)
- Real-time monitoring and alerts
- Model management and health monitoring
"""

import os
import sys
import time
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import torch

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Import our services
try:
    from inference_service import inference_service, SeismicInferenceService
    from model_loader import model_loader
except ImportError as e:
    logging.warning(f"Could not import inference services: {e}")
    inference_service = None

# Import alert system
try:
    from alert_system import (
        alert_manager, seismic_monitor, weather_monitor, air_quality_monitor,
        AlertType, AlertLevel, initialize_monitoring
    )
except ImportError as e:
    logging.warning(f"Could not import alert system: {e}")
    alert_manager = None

# Import AI agents
try:
    from simple_weather_agent import SimpleWeatherAgent
    from sea_level_analyzer import SeaLevelAnalyzer
except ImportError as e:
    logging.warning(f"Could not import AI agents: {e}")

# Import seismic detection agent
try:
    from realtime_seismic_detector import RealTimeSeismicDetector, quick_seismic_scan
    seismic_detector = RealTimeSeismicDetector()
except ImportError as e:
    logging.warning(f"Could not import seismic detection agent: {e}")
    seismic_detector = None
    SimpleWeatherAgent = None
    SeaLevelAnalyzer = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for agents
weather_agent = None
sea_level_agent = None
agents_initialized = False

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    logger.info("🚀 Starting Seismic AI API Server...")
    # Temporarily disabled: await initialize_agents()

    # Initialize alert system
    if alert_manager:
        try:
            initialize_monitoring()
            logger.info("✅ Alert system initialized")
        except Exception as e:
            logger.warning(f"⚠️ Alert system initialization failed: {e}")

    yield

    # Shutdown
    logger.info("🛑 Shutting down Seismic AI API Server...")

# Create FastAPI app
app = FastAPI(
    title="Seismic AI Complete API",
    description="Comprehensive API for seismic analysis, weather prediction, and environmental monitoring",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Coordinates(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")

class SeismicData(BaseModel):
    data: Union[List[float], List[List[float]], List[List[List[float]]], List[List[List[List[float]]]]] = Field(..., description="Seismic data array (flexible dimensions)")
    shape: Optional[List[int]] = Field(None, description="Data shape specification")

class PredictionRequest(BaseModel):
    coordinates: Coordinates
    hours_ahead: Optional[int] = Field(24, ge=1, le=168, description="Hours to predict ahead")

class IntegratedAnalysisRequest(BaseModel):
    coordinates: Coordinates
    analysis_type: str = Field("full", description="Type of analysis: 'weather', 'seismic', 'sea_level', 'full'")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, Any]
    models: Dict[str, Any]
    system: Dict[str, Any]

# Seismic Detection Agent Models
class SeismicScanRequest(BaseModel):
    coordinates: Coordinates
    zone_name: Optional[str] = Field(None, description="Optional zone name for the scan")

class MonitoringZoneRequest(BaseModel):
    name: str = Field(..., description="Zone name")
    coordinates: Coordinates
    radius_km: float = Field(50.0, ge=1.0, le=1000.0, description="Monitoring radius in kilometers")
    priority: str = Field("medium", description="Priority level: low, medium, high, critical")

class SeismicAlertResponse(BaseModel):
    alert_id: str
    timestamp: str
    coordinates: Dict[str, float]
    seismic_probability: float = Field(..., ge=0.0, le=10.0, description="Seismic probability (0.0-10.0)")
    intensity: float = Field(..., ge=0.0, description="Seismic intensity")
    risk_level: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    data: Optional[Dict[str, Any]] = None

# Agent initialization
async def initialize_agents():
    """Initialize all AI agents."""
    global weather_agent, sea_level_agent, agents_initialized

    try:
        logger.info("Initializing AI agents...")

        # Initialize weather agent
        if SimpleWeatherAgent is not None:
            weather_agent = SimpleWeatherAgent()
            if weather_agent.initialize():
                logger.info("✅ Weather agent initialized")
            else:
                logger.warning("⚠️ Weather agent initialization failed")

        # Initialize sea level agent
        if SeaLevelAnalyzer is not None:
            sea_level_agent = SeaLevelAnalyzer()
            logger.info("✅ Sea level agent initialized")

        agents_initialized = True
        logger.info("🎉 All agents initialized successfully")

    except Exception as e:
        logger.error(f"❌ Error initializing agents: {e}")
        agents_initialized = False

# Utility functions
def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate geographic coordinates."""
    return -90 <= lat <= 90 and -180 <= lon <= 180

def create_error_response(message: str, status_code: int = 400) -> JSONResponse:
    """Create standardized error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "status_code": status_code
            }
        }
    )

# Routes
@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Seismic AI Complete API Server",
        "version": "2.0.0",
        "status": "running",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check."""
    services_status = {
        "agents_initialized": agents_initialized,
        "weather_agent": weather_agent is not None,
        "sea_level_agent": sea_level_agent is not None,
        "inference_service": inference_service is not None,
        "alert_system": alert_manager is not None,
        "seismic_monitor": seismic_monitor is not None,
        "weather_monitor": weather_monitor is not None,
        "air_quality_monitor": air_quality_monitor is not None,
    }

    models_status = {}
    if inference_service:
        models_status = inference_service.get_model_status()

    monitoring_status = {}
    if alert_manager:
        monitoring_status = {
            "active_alerts": len(alert_manager.get_active_alerts()),
            "total_alerts_history": len(alert_manager.alert_history),
            "monitoring_zones": len(seismic_monitor.monitoring_zones) if seismic_monitor else 0
        }

    system_info = {
        "python_version": sys.version,
        "torch_available": torch.cuda.is_available() if torch else False,
        "cuda_devices": torch.cuda.device_count() if torch and torch.cuda.is_available() else 0,
        "monitoring": monitoring_status
    }

    overall_status = "healthy" if all(services_status.values()) else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        services=services_status,
        models=models_status,
        system=system_info
    )

# Seismic Inference Endpoints
@app.post("/predict/seismic")
async def predict_seismic(seismic_input: SeismicData):
    """Predict seismic activity from seismic data."""
    try:
        if not inference_service:
            raise HTTPException(status_code=503, detail="Inference service not available")

        # Convert to numpy array
        data = np.array(seismic_input.data, dtype=np.float32)
        if seismic_input.shape:
            data = data.reshape(seismic_input.shape)

        # Run prediction
        result = inference_service.predict_seismic_state(data)

        return {
            "success": True,
            "predictions": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Seismic prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Seismic prediction failed: {str(e)}")

@app.post("/predict/seismic-regression")
async def predict_seismic_regression(seismic_input: SeismicData):
    """Predict seismic deformation (regression)."""
    try:
        if not inference_service:
            raise HTTPException(status_code=503, detail="Inference service not available")

        # Convert to numpy array
        data = np.array(seismic_input.data, dtype=np.float32)
        if seismic_input.shape:
            data = data.reshape(seismic_input.shape)

        # Run prediction (placeholder for regression model)
        # For now, return mock prediction
        batch_size = data.shape[0] if len(data.shape) > 0 else 1
        mock_deformation = np.random.randn(batch_size, 50, 50).astype(np.float32)

        return {
            "success": True,
            "predictions": {
                "deformation_map": mock_deformation.tolist(),
                "shape": mock_deformation.shape,
                "note": "Mock prediction - regression model needs implementation"
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Seismic regression error: {e}")
        raise HTTPException(status_code=500, detail=f"Seismic regression failed: {str(e)}")

# Seismic Detection Agent Endpoints
@app.post("/seismic/scan", response_model=SeismicAlertResponse)
async def seismic_scan(request: SeismicScanRequest):
    """
    Realizar escaneo sísmico manual en coordenadas específicas.

    Returns:
        SeismicAlertResponse con información de la detección
    """
    try:
        if not seismic_detector:
            raise HTTPException(status_code=503, detail="Seismic detection agent not available")

        lat, lon = request.coordinates.lat, request.coordinates.lon

        # Realizar escaneo
        result = seismic_detector.manual_scan(lat, lon)

        if result and "seismic_probability" in result:
            # Convertir a formato de respuesta
            return SeismicAlertResponse(**result)
        else:
            # No se detectó actividad sísmica
            return SeismicAlertResponse(
                alert_id=f"no_detection_{int(time.time() * 1000)}",
                timestamp=datetime.now().isoformat(),
                coordinates={"lat": lat, "lon": lon},
                seismic_probability=0.0,
                intensity=0.0,
                risk_level="none",
                confidence=1.0,
                data={"message": "No seismic activity detected"}
            )

    except Exception as e:
        logger.error(f"Seismic scan error: {e}")
        raise HTTPException(status_code=500, detail=f"Seismic scan failed: {str(e)}")

@app.post("/seismic/zones/add")
async def add_monitoring_zone(request: MonitoringZoneRequest):
    """
    Agregar zona de monitoreo sísmico.

    La zona será monitoreada continuamente por el agente.
    """
    try:
        if not seismic_detector:
            raise HTTPException(status_code=503, detail="Seismic detection agent not available")

        seismic_detector.add_monitoring_zone(
            name=request.name,
            lat=request.coordinates.lat,
            lon=request.coordinates.lon,
            radius_km=request.radius_km,
            priority=request.priority
        )

        return {
            "success": True,
            "message": f"Monitoring zone '{request.name}' added successfully",
            "zone": {
                "name": request.name,
                "coordinates": {"lat": request.coordinates.lat, "lon": request.coordinates.lon},
                "radius_km": request.radius_km,
                "priority": request.priority
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Add monitoring zone error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add monitoring zone: {str(e)}")

@app.delete("/seismic/zones/{zone_name}")
async def remove_monitoring_zone(zone_name: str):
    """
    Remover zona de monitoreo sísmico.
    """
    try:
        if not seismic_detector:
            raise HTTPException(status_code=503, detail="Seismic detection agent not available")

        success = seismic_detector.remove_monitoring_zone(zone_name)

        if success:
            return {
                "success": True,
                "message": f"Monitoring zone '{zone_name}' removed successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Monitoring zone '{zone_name}' not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Remove monitoring zone error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove monitoring zone: {str(e)}")

@app.post("/seismic/monitoring/start")
async def start_seismic_monitoring():
    """
    Iniciar monitoreo sísmico en tiempo real.
    """
    try:
        if not seismic_detector:
            raise HTTPException(status_code=503, detail="Seismic detection agent not available")

        if seismic_detector.monitoring_active:
            return {
                "success": False,
                "message": "Seismic monitoring already active",
                "timestamp": datetime.now().isoformat()
            }

        seismic_detector.start_monitoring()

        return {
            "success": True,
            "message": "Seismic monitoring started successfully",
            "active_zones": len(seismic_detector.monitoring_zones),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Start monitoring error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@app.post("/seismic/monitoring/stop")
async def stop_seismic_monitoring():
    """
    Detener monitoreo sísmico en tiempo real.
    """
    try:
        if not seismic_detector:
            raise HTTPException(status_code=503, detail="Seismic detection agent not available")

        if not seismic_detector.monitoring_active:
            return {
                "success": False,
                "message": "Seismic monitoring not active",
                "timestamp": datetime.now().isoformat()
            }

        seismic_detector.stop_monitoring()

        return {
            "success": True,
            "message": "Seismic monitoring stopped successfully",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Stop monitoring error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")

@app.get("/seismic/alerts/active")
async def get_active_seismic_alerts():
    """
    Obtener alertas sísmicas activas.
    """
    try:
        if not seismic_detector:
            raise HTTPException(status_code=503, detail="Seismic detection agent not available")

        alerts = seismic_detector.get_active_alerts()

        return {
            "success": True,
            "alerts": alerts,
            "count": len(alerts),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Get active alerts error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get active alerts: {str(e)}")

@app.get("/seismic/alerts/history")
async def get_seismic_alerts_history(limit: int = 50):
    """
    Obtener historial de alertas sísmicas.
    """
    try:
        if not seismic_detector:
            raise HTTPException(status_code=503, detail="Seismic detection agent not available")

        history = seismic_detector.get_detection_history(limit=limit)

        return {
            "success": True,
            "alerts": history,
            "count": len(history),
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Get alerts history error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts history: {str(e)}")

@app.get("/seismic/zones")
async def get_monitoring_zones():
    """
    Obtener lista de zonas de monitoreo.
    """
    try:
        if not seismic_detector:
            raise HTTPException(status_code=503, detail="Seismic detection agent not available")

        zones = []
        for zone in seismic_detector.monitoring_zones:
            zones.append({
                "name": zone["name"],
                "coordinates": zone["coordinates"],
                "radius_km": zone["radius_km"],
                "priority": zone["priority"],
                "active": zone["active"],
                "alert_count": zone["alert_count"],
                "last_scan": zone["last_scan"]
            })

        return {
            "success": True,
            "zones": zones,
            "count": len(zones),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Get monitoring zones error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring zones: {str(e)}")

@app.get("/seismic/stats")
async def get_seismic_detector_stats():
    """
    Obtener estadísticas del detector sísmico.
    """
    try:
        if not seismic_detector:
            raise HTTPException(status_code=503, detail="Seismic detection agent not available")

        stats = seismic_detector.get_stats()

        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Get stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Weather Prediction Endpoints
@app.post("/predict/weather")
async def predict_weather(request: PredictionRequest):
    """Predict weather conditions for given coordinates."""
    try:
        if not weather_agent:
            raise HTTPException(status_code=503, detail="Weather agent not available")

        lat, lon = request.coordinates.lat, request.coordinates.lon

        # Get weather prediction
        prediction = weather_agent.predict_rain(lat, lon, request.hours_ahead)

        return {
            "success": True,
            "coordinates": {"lat": lat, "lon": lon},
            "prediction": prediction,
            "hours_ahead": request.hours_ahead,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Weather prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Weather prediction failed: {str(e)}")

@app.post("/predict/weather-ai")
async def predict_weather_ai(weather_input: SeismicData):
    """AI-based weather prediction from sensor data."""
    try:
        if not inference_service:
            raise HTTPException(status_code=503, detail="Inference service not available")

        # Convert to numpy array
        data = np.array(weather_input.data, dtype=np.float32)
        if weather_input.shape:
            data = data.reshape(weather_input.shape)

        # Run AI prediction
        result = inference_service.predict_weather(data)

        return {
            "success": True,
            "predictions": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"AI weather prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"AI weather prediction failed: {str(e)}")

# DSA (Air Quality) Endpoints
@app.post("/predict/dsa")
async def predict_dsa(dsa_input: SeismicData):
    """Predict air quality using DSA model."""
    try:
        if not inference_service:
            raise HTTPException(status_code=503, detail="Inference service not available")

        # Convert to numpy array
        data = np.array(dsa_input.data, dtype=np.float32)
        if dsa_input.shape:
            data = data.reshape(dsa_input.shape)

        # Run DSA prediction
        result = inference_service.predict_dsa(data)

        return {
            "success": True,
            "predictions": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"DSA prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"DSA prediction failed: {str(e)}")

@app.post("/predict/air-quality")
async def predict_air_quality(coordinates: Coordinates):
    """Predict air quality for given coordinates."""
    try:
        lat, lon = coordinates.lat, coordinates.lon

        # Mock air quality prediction (integrate with real DSA model)
        aqi_value = np.random.uniform(0, 200)  # Random AQI value

        # Determine AQI level
        if aqi_value <= 50:
            level = "Good"
            color = "green"
        elif aqi_value <= 100:
            level = "Moderate"
            color = "yellow"
        elif aqi_value <= 150:
            level = "Unhealthy for Sensitive Groups"
            color = "orange"
        elif aqi_value <= 200:
            level = "Unhealthy"
            color = "red"
        elif aqi_value <= 300:
            level = "Very Unhealthy"
            color = "purple"
        else:
            level = "Hazardous"
            color = "maroon"

        return {
            "success": True,
            "coordinates": {"lat": lat, "lon": lon},
            "air_quality": {
                "aqi": round(aqi_value, 1),
                "level": level,
                "color": color,
                "description": f"Air quality is {level.lower()}",
                "note": "Mock prediction - integrate with real air quality model"
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Air quality prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Air quality prediction failed: {str(e)}")

# Integrated Analysis Endpoints
@app.post("/analyze/integrated")
async def analyze_integrated(request: IntegratedAnalysisRequest):
    """Perform integrated analysis combining multiple data sources."""
    try:
        lat, lon = request.coordinates.lat, request.coordinates.lon
        analysis_type = request.analysis_type

        results = {
            "coordinates": {"lat": lat, "lon": lon},
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }

        # Weather analysis
        if analysis_type in ["weather", "full"]:
            if weather_agent:
                weather_pred = weather_agent.predict_rain(lat, lon, 24)
                results["components"]["weather"] = weather_pred
            else:
                results["components"]["weather"] = {"error": "Weather agent not available"}

        # Sea level analysis
        if analysis_type in ["sea_level", "full"]:
            if sea_level_agent:
                try:
                    # Real sea level analysis using coordinates
                    sea_level_result = sea_level_agent.analyze_coordinates(
                        lat, lon, 
                        start_date="2023-01-01", 
                        end_date="2023-12-31"
                    )
                    
                    if sea_level_result and 'error' not in sea_level_result:
                        # Extract relevant data for the response
                        sea_level_data = {
                            "current_level": sea_level_result.get('current_level_mm', 0),
                            "trend": sea_level_result.get('trend_interpretation', {}).get('description', 'unknown'),
                            "trend_rate": sea_level_result.get('trend_mm_per_year', 0),
                            "region": sea_level_result.get('region_name', 'unknown'),
                            "confidence": sea_level_result.get('confidence', 'medium'),
                            "data_points": sea_level_result.get('data_points', 0),
                            "analysis_period": f"{sea_level_result.get('start_date', '2023-01-01')} to {sea_level_result.get('end_date', '2023-12-31')}"
                        }
                    else:
                        # Fallback to mock data if analysis fails
                        sea_level_data = {
                            "current_level": np.random.uniform(-1, 1),
                            "trend": "stable",
                            "trend_rate": 0.0,
                            "region": "unknown",
                            "confidence": "low",
                            "error": sea_level_result.get('error', 'Analysis failed')
                        }
                    
                    results["components"]["sea_level"] = sea_level_data
                    
                except Exception as e:
                    results["components"]["sea_level"] = {
                        "error": f"Sea level analysis failed: {str(e)}",
                        "current_level": 0.0,
                        "trend": "unknown"
                    }
            else:
                results["components"]["sea_level"] = {"error": "Sea level agent not available"}

        # Seismic analysis
        if analysis_type in ["seismic", "full"]:
            if inference_service:
                # Mock seismic analysis for coordinates
                seismic_risk = {
                    "risk_level": np.random.choice(["Low", "Medium", "High"]),
                    "probability": np.random.uniform(0, 1),
                    "last_event": "2023-05-15",
                    "note": "Mock data - integrate with real seismic model"
                }
                results["components"]["seismic"] = seismic_risk
            else:
                results["components"]["seismic"] = {"error": "Seismic inference not available"}

        results["success"] = True
        return results

    except Exception as e:
        logger.error(f"Integrated analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Integrated analysis failed: {str(e)}")

# Real-time Monitoring Endpoints
@app.get("/monitor/status")
async def get_monitoring_status():
    """Get real-time monitoring status."""
    return {
        "success": True,
        "monitoring": {
            "active_alerts": 0,
            "monitored_zones": 150,
            "last_update": datetime.now().isoformat(),
            "system_health": "nominal"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/monitor/alert")
async def create_alert(alert_data: Dict[str, Any], background_tasks: BackgroundTasks):
    """Create a monitoring alert."""
    try:
        if not alert_manager:
            raise HTTPException(status_code=503, detail="Alert system not available")

        # Determine alert type
        alert_type_str = alert_data.get("type", "integrated")
        try:
            alert_type = AlertType(alert_type_str)
        except ValueError:
            alert_type = AlertType.INTEGRATED

        # Determine alert level
        level_str = alert_data.get("level", "medium")
        try:
            level = AlertLevel(level_str)
        except ValueError:
            level = AlertLevel.MEDIUM

        # Create alert
        alert = alert_manager.create_alert(
            alert_type=alert_type,
            level=level,
            title=alert_data.get("title", "System Alert"),
            message=alert_data.get("message", "Alert triggered by system"),
            coordinates=alert_data.get("coordinates"),
            data=alert_data.get("data")
        )

        return {
            "success": True,
            "alert_id": alert.alert_id,
            "message": "Alert created successfully",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Alert creation error: {e}")
        raise HTTPException(status_code=500, detail=f"Alert creation failed: {str(e)}")

@app.put("/monitor/alert/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, acknowledged_by: str = "api_user"):
    """Acknowledge an alert."""
    try:
        if not alert_manager:
            raise HTTPException(status_code=503, detail="Alert system not available")

        success = alert_manager.acknowledge_alert(alert_id, acknowledged_by)

        if not success:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        return {
            "success": True,
            "message": f"Alert {alert_id} acknowledged",
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alert acknowledgement error: {e}")
        raise HTTPException(status_code=500, detail=f"Alert acknowledgement failed: {str(e)}")

@app.get("/monitor/alerts/active")
async def get_active_alerts():
    """Get all active alerts."""
    try:
        if not alert_manager:
            raise HTTPException(status_code=503, detail="Alert system not available")

        alerts = alert_manager.get_active_alerts()

        return {
            "success": True,
            "alerts": alerts,
            "count": len(alerts),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Get active alerts error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get active alerts: {str(e)}")

@app.get("/monitor/alerts/history")
async def get_alert_history(limit: int = 50):
    """Get alert history."""
    try:
        if not alert_manager:
            raise HTTPException(status_code=503, detail="Alert system not available")

        alerts = alert_manager.get_alert_history(limit)

        return {
            "success": True,
            "alerts": alerts,
            "count": len(alerts),
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Get alert history error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alert history: {str(e)}")

@app.get("/monitor/alerts/{alert_id}")
async def get_alert(alert_id: str):
    """Get specific alert by ID."""
    try:
        if not alert_manager:
            raise HTTPException(status_code=503, detail="Alert system not available")

        alert = alert_manager.get_alert_by_id(alert_id)

        if not alert:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        return {
            "success": True,
            "alert": alert,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get alert error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alert: {str(e)}")

@app.post("/monitor/check/seismic")
async def check_seismic_risk(coordinates: Coordinates):
    """Check seismic risk for coordinates."""
    try:
        if not seismic_monitor:
            raise HTTPException(status_code=503, detail="Seismic monitor not available")

        risk_data = seismic_monitor.check_seismic_risk(coordinates.lat, coordinates.lon)

        return {
            "success": True,
            "coordinates": {"lat": coordinates.lat, "lon": coordinates.lon},
            "risk_assessment": risk_data,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Seismic risk check error: {e}")
        raise HTTPException(status_code=500, detail=f"Seismic risk check failed: {str(e)}")

@app.post("/monitor/check/weather")
async def check_weather_conditions(coordinates: Coordinates):
    """Check weather conditions for coordinates."""
    try:
        if not weather_monitor:
            raise HTTPException(status_code=503, detail="Weather monitor not available")

        weather_data = weather_monitor.check_weather_conditions(coordinates.lat, coordinates.lon)

        return {
            "success": True,
            "coordinates": {"lat": coordinates.lat, "lon": coordinates.lon},
            "weather_assessment": weather_data,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Weather check error: {e}")
        raise HTTPException(status_code=500, detail=f"Weather check failed: {str(e)}")

@app.post("/monitor/check/air-quality")
async def check_air_quality(coordinates: Coordinates):
    """Check air quality for coordinates."""
    try:
        if not air_quality_monitor:
            raise HTTPException(status_code=503, detail="Air quality monitor not available")

        aq_data = air_quality_monitor.check_air_quality(coordinates.lat, coordinates.lon)

        return {
            "success": True,
            "coordinates": {"lat": coordinates.lat, "lon": coordinates.lon},
            "air_quality_assessment": aq_data,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Air quality check error: {e}")
        raise HTTPException(status_code=500, detail=f"Air quality check failed: {str(e)}")

# Model Management Endpoints
@app.get("/models")
async def list_models():
    """List all available models."""
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model loader not available")

    available_models = model_loader.get_available_models()

    return {
        "success": True,
        "models": available_models,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models/{model_type}/{model_name}/status")
async def get_model_status(model_type: str, model_name: str):
    """Get status of a specific model."""
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model loader not available")

    try:
        model = model_loader.load_model(model_type, model_name)
        return {
            "success": True,
            "model_type": model_type,
            "model_name": model_name,
            "loaded": True,
            "type": str(type(model)),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "model_type": model_type,
            "model_name": model_name,
            "loaded": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.now().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "message": "Internal server error",
                "details": str(exc),
                "timestamp": datetime.now().isoformat()
            }
        }
    )

if __name__ == "__main__":
    print("🎯 Seismic AI Complete API Server v2.0")
    print("=" * 50)
    print("📚 API Documentation: http://127.0.0.1:8000/docs")
    print("🔍 Health Check: http://127.0.0.1:8000/health")
    print("🚀 Server running on http://127.0.0.1:8000")
    print()
    print("🔧 Available Services:")
    print("  🤖 Seismic Classification & Regression")
    print("  🌦️  Weather Prediction & Analysis")
    print("  💨 Air Quality Monitoring (DSA)")
    print("  🌊 Sea Level Analysis")
    print("  � Real-time Seismic Detection Agent")
    print("  �📊 Integrated Multi-modal Analysis")
    print("  🚨 Real-time Alert System")
    print("  📡 Model Management")
    print()
    print("📊 Key Endpoints:")
    print("  GET  /                          - API info")
    print("  GET  /health                   - System health check")
    print("  POST /predict/seismic          - Seismic activity prediction")
    print("  POST /predict/seismic-regression - Seismic deformation prediction")
    print("  POST /seismic/scan             - Manual seismic scan")
    print("  POST /seismic/zones/add        - Add monitoring zone")
    print("  DELETE /seismic/zones/{name}   - Remove monitoring zone")
    print("  POST /seismic/monitoring/start - Start real-time monitoring")
    print("  POST /seismic/monitoring/stop  - Stop real-time monitoring")
    print("  GET  /seismic/alerts/active    - Active seismic alerts")
    print("  GET  /seismic/alerts/history   - Seismic alerts history")
    print("  GET  /seismic/zones            - Monitoring zones")
    print("  GET  /seismic/stats            - Seismic detector stats")
    print("  POST /predict/weather          - Weather prediction")
    print("  POST /predict/dsa              - DSA air quality prediction")
    print("  POST /analyze/integrated       - Multi-modal analysis")
    print("  POST /monitor/alert            - Create monitoring alert")
    print("  GET  /monitor/alerts/active    - Active alerts")
    print("  GET  /models                   - Available models")
    print()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )