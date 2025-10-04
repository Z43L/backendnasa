# Seismic AI Complete API Server

A comprehensive FastAPI server that integrates all seismic AI capabilities for real-time environmental monitoring and predictive analytics.

## 🚀 Features

### 🤖 AI Models & Predictions
- **Seismic Classification**: Predict precursor/normal/post-earthquake states
- **Seismic Regression**: Forecast ground deformation and displacement
- **Weather Prediction**: AI-powered weather forecasting with traditional methods
- **DSA (Data Science Algorithm)**: Air quality monitoring and prediction

### 🌊 Environmental Monitoring
- **Sea Level Analysis**: Coastal monitoring and flood prediction
- **Multi-modal Integration**: Combined analysis of seismic, weather, and sea level data
- **Real-time Risk Assessment**: Dynamic risk evaluation for any coordinates

### 🚨 Alert System
- **Real-time Alerts**: Automated alert generation for critical conditions
- **Multi-level Alerting**: Low, medium, high, and critical alert levels
- **Alert Management**: Create, acknowledge, and track alerts
- **Monitoring Zones**: Configurable geographic monitoring areas

### 📊 System Management
- **Health Monitoring**: Comprehensive system health checks
- **Model Management**: Load, unload, and monitor AI models
- **Performance Metrics**: System performance and resource usage
- **API Documentation**: Auto-generated Swagger/OpenAPI documentation

## 🛠️ Installation

```bash
# Install dependencies
pip install fastapi uvicorn pydantic requests numpy

# For GPU support (optional)
pip install torch torchvision torchaudio

# For additional features
pip install safetensors flask flask-cors
```

## 🚀 Quick Start

### Start the Server

```bash
python simple_api_server.py
```

The server will start on `http://127.0.0.1:8000`

### API Documentation

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc
- **Health Check**: http://127.0.0.1:8000/health

## 📚 API Endpoints

### Core Prediction Endpoints

#### Seismic Analysis
```http
POST /predict/seismic
POST /predict/seismic-regression
```

#### Weather & Climate
```http
POST /predict/weather
POST /predict/weather-ai
```

#### Air Quality
```http
POST /predict/dsa
POST /predict/air-quality
```

#### Integrated Analysis
```http
POST /analyze/integrated
```

### Monitoring & Alerts

#### Alert Management
```http
POST /monitor/alert
PUT  /monitor/alert/{alert_id}/acknowledge
GET  /monitor/alerts/active
GET  /monitor/alerts/history
GET  /monitor/alerts/{alert_id}
```

#### Risk Assessment
```http
POST /monitor/check/seismic
POST /monitor/check/weather
POST /monitor/check/air-quality
GET  /monitor/status
```

### System Management

#### Health & Status
```http
GET  /health
GET  /
```

#### Model Management
```http
GET  /models
GET  /models/{model_type}/{model_name}/status
```

## 💡 Usage Examples

### Seismic Prediction

```python
import requests

# Seismic classification
seismic_data = {
    "data": [[[0.1, 0.2, 0.3] * 16] * 30] * 1,  # 1 sample, 30 timesteps, 50x50 grid
    "shape": [1, 30, 50]
}

response = requests.post("http://127.0.0.1:8000/predict/seismic", json=seismic_data)
print(response.json())
```

### Weather Prediction

```python
# Weather prediction for coordinates
weather_request = {
    "coordinates": {"lat": 35.0, "lon": -118.0},
    "hours_ahead": 24
}

response = requests.post("http://127.0.0.1:8000/predict/weather", json=weather_request)
print(response.json())
```

### Create Alert

```python
alert_data = {
    "type": "seismic",
    "level": "high",
    "title": "High Seismic Activity Detected",
    "message": "Seismic sensors detected unusual activity",
    "coordinates": {"lat": 35.0, "lon": -118.0},
    "data": {"magnitude": 4.2, "confidence": 0.85}
}

response = requests.post("http://127.0.0.1:8000/monitor/alert", json=alert_data)
print(response.json())
```

### Integrated Analysis

```python
analysis_request = {
    "coordinates": {"lat": 35.0, "lon": -118.0},
    "analysis_type": "full"  # Options: "weather", "seismic", "sea_level", "full"
}

response = requests.post("http://127.0.0.1:8000/analyze/integrated", json=analysis_request)
print(response.json())
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_complete_api.py
```

This will test all endpoints and provide a detailed report of the system's functionality.

## 🏗️ Architecture

### Core Components

1. **FastAPI Application**: Main web framework with automatic API documentation
2. **Inference Service**: Handles AI model loading and prediction
3. **Alert System**: Real-time monitoring and alert management
4. **Model Loader**: Unified interface for different model formats
5. **Monitoring Agents**: Specialized monitors for different environmental factors

### Data Flow

```
Client Request → FastAPI → Service Layer → AI Models → Response
                      ↓
                Alert System → Notifications
                      ↓
               Monitoring → Risk Assessment
```

## 🔧 Configuration

### Environment Variables

```bash
# Server configuration
HOST=127.0.0.1
PORT=8000

# Model paths
MODEL_DIR=backend/models/

# Alert thresholds
SEISMIC_HIGH_THRESHOLD=0.8
WEATHER_ALERT_THRESHOLD=10.0
```

### Model Configuration

Models are automatically loaded from the `backend/models/` directory with the following structure:

```
backend/models/
├── seismic_classification/
│   └── cl_falla_anatolia.pth
├── weather/
│   ├── large_weather_model_20251003_192729.pkl
│   └── large_weather_model_20251003_192729.safetensors
└── dsa/
    └── dsa_model.pt
```

## 📊 Monitoring & Health

### Health Check Response

```json
{
  "status": "healthy",
  "timestamp": "2025-10-04T12:00:00",
  "services": {
    "inference_service": true,
    "alert_system": true,
    "weather_agent": true,
    "seismic_monitor": true
  },
  "models": {
    "seismic_classification": true,
    "weather": true,
    "dsa": true
  },
  "system": {
    "monitoring": {
      "active_alerts": 2,
      "monitoring_zones": 150
    }
  }
}
```

## 🚨 Alert Types

- **Seismic**: Earthquake detection and risk assessment
- **Weather**: Severe weather warnings and precipitation alerts
- **Air Quality**: Poor air quality and pollution alerts
- **Sea Level**: Coastal flooding and sea level change alerts
- **Integrated**: Multi-factor environmental alerts

## 🔒 Security

- CORS enabled for cross-origin requests
- Input validation with Pydantic models
- Error handling with detailed logging
- Rate limiting (configurable)
- Authentication support (extensible)

## 📈 Performance

- Asynchronous request handling
- GPU acceleration support (when available)
- Model caching and lazy loading
- Background task processing for alerts
- Connection pooling for external APIs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the API documentation at `/docs`
- Review the test suite in `test_complete_api.py`
- Check system logs for detailed error information

---

**Built with ❤️ for environmental monitoring and disaster prevention**