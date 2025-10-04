"""
Seismic AI Inference Service

This module provides inference capabilities for all trained models:
- Seismic classification (precursor/normal/post-earthquake states)
- Seismic regression (deformation prediction)
- Weather prediction
- DSA (Data Science Algorithm) models
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from model_loader import ModelLoader, load_seismic_classification_model, load_seismic_regression_model, load_weather_model, load_dsa_model
import logging

logger = logging.getLogger(__name__)


class SeismicInferenceService:
    """Service for running inference on seismic AI models."""

    def __init__(self, device: str = "auto"):
        """
        Initialize the inference service.

        Args:
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.device = self._setup_device(device)
        self.model_loader = ModelLoader()
        self.models = {}

        # Load models on initialization
        self._load_models()

    def _setup_device(self, device: str) -> str:
        """Setup the compute device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_models(self):
        """Load all available models."""
        try:
            # Load seismic regression model
            try:
                self.models['seismic_regression'] = load_seismic_regression_model(self.device)
                logger.info("Seismic regression model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load seismic regression model: {e}")

            # Load seismic classification model
            try:
                self.models['seismic_classification'] = load_seismic_classification_model(self.device)
                logger.info("Seismic classification model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load seismic classification model: {e}")

            # Load weather model
            try:
                self.models['weather'] = load_weather_model(self.device)
                logger.info("Weather model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load weather model: {e}")

            # Load DSA model
            try:
                self.models['dsa'] = load_dsa_model(self.device)
                logger.info("DSA model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load DSA model: {e}")

        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def predict_seismic_state(self, seismic_data: np.ndarray) -> Dict[str, Any]:
        """
        Predict seismic state from seismic data.

        Args:
            seismic_data: Seismic data array with shape (batch_size, channels, height, width)

        Returns:
            Dictionary with predictions and probabilities
        """
        if 'seismic_classification' not in self.models:
            raise RuntimeError("Seismic classification model not loaded")

        try:
            # Convert to tensor
            if isinstance(seismic_data, np.ndarray):
                seismic_data = torch.from_numpy(seismic_data).float()

            seismic_data = seismic_data.to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.models['seismic_classification'](seismic_data)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

            # Convert to numpy for response
            predictions_np = predictions.cpu().numpy()
            probabilities_np = probabilities.cpu().numpy()

            # Map predictions to class names
            class_names = ['precursor', 'normal', 'post-earthquake']
            predicted_classes = [class_names[pred] for pred in predictions_np]

            return {
                'predictions': predicted_classes,
                'probabilities': probabilities_np.tolist(),
                'class_names': class_names
            }

        except Exception as e:
            logger.error(f"Error in seismic state prediction: {e}")
            raise RuntimeError(f"Seismic prediction failed: {e}")

    def predict_seismic_deformation(self, seismic_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Predict seismic deformation from a sequence of seismic data.

        Args:
            seismic_sequence: Seismic data sequence with shape (seq_length, height, width) or (batch_size, seq_length, height, width)

        Returns:
            Dictionary with deformation predictions
        """
        if 'seismic_regression' not in self.models:
            raise RuntimeError("Seismic regression model not loaded")

        try:
            # Convert to tensor if needed
            if isinstance(seismic_sequence, np.ndarray):
                seismic_sequence = torch.from_numpy(seismic_sequence).float()

            # Ensure we have the right shape - add batch dimension if needed
            if len(seismic_sequence.shape) == 3:
                # Shape is (seq_length, height, width) - add batch dimension
                seismic_sequence = seismic_sequence.unsqueeze(0)

            seismic_sequence = seismic_sequence.to(self.device)

            # Run inference
            with torch.no_grad():
                deformation_prediction = self.models['seismic_regression'](seismic_sequence)

            # Convert to numpy for response
            prediction_np = deformation_prediction.cpu().numpy()

            # Get input dimensions for context
            batch_size, seq_length, height, width = seismic_sequence.shape

            return {
                'deformation_prediction': prediction_np.tolist(),
                'shape': prediction_np.shape,
                'input_shape': [batch_size, seq_length, height, width],
                'description': 'Predicted deformation map for the next time step',
                'units': 'meters (relative displacement)'
            }

        except Exception as e:
            logger.error(f"Error in seismic deformation prediction: {e}")
            raise RuntimeError(f"Seismic deformation prediction failed: {e}")

    def predict_weather(self, weather_features: np.ndarray) -> Dict[str, Any]:
        """
        Predict weather conditions from features.

        Args:
            weather_features: Weather feature array

        Returns:
            Dictionary with weather predictions
        """
        if 'weather' not in self.models:
            raise RuntimeError("Weather model not loaded")

        try:
            # Weather models appear to be sklearn-like models stored as dictionaries
            model_dict = self.models['weather']

            # For now, return a mock prediction since we don't know the exact format
            # In a real implementation, you'd load the sklearn model and use it
            if isinstance(weather_features, np.ndarray):
                batch_size = weather_features.shape[0]
            else:
                batch_size = len(weather_features)

            # Mock prediction - replace with actual model inference
            mock_predictions = np.random.randn(batch_size, 5).astype(np.float32)  # Assuming 5 output features

            return {
                'predictions': mock_predictions.tolist(),
                'shape': mock_predictions.shape,
                'note': 'Mock prediction - weather model needs proper loading implementation'
            }

        except Exception as e:
            logger.error(f"Error in weather prediction: {e}")
            raise RuntimeError(f"Weather prediction failed: {e}")

    def predict_dsa(self, dsa_input: np.ndarray) -> Dict[str, Any]:
        """
        Run DSA model prediction.

        Args:
            dsa_input: DSA input data

        Returns:
            Dictionary with DSA predictions
        """
        if 'dsa' not in self.models:
            raise RuntimeError("DSA model not loaded")

        try:
            # DSA models appear to be stored as dictionaries
            model_dict = self.models['dsa']

            # For now, return a mock prediction since we don't know the exact format
            if isinstance(dsa_input, np.ndarray):
                batch_size = dsa_input.shape[0]
            else:
                batch_size = len(dsa_input)

            # Mock prediction - replace with actual model inference
            mock_predictions = np.random.randn(batch_size, 3).astype(np.float32)  # Assuming 3 output classes

            return {
                'predictions': mock_predictions.tolist(),
                'shape': mock_predictions.shape,
                'note': 'Mock prediction - DSA model needs proper loading implementation'
            }

        except Exception as e:
            logger.error(f"Error in DSA prediction: {e}")
            raise RuntimeError(f"DSA prediction failed: {e}")

    def get_model_status(self) -> Dict[str, bool]:
        """Get the loading status of all models."""
        return {
            model_type: (model_type in self.models)
            for model_type in ['seismic_classification', 'seismic_regression', 'weather', 'dsa']
        }

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the current device."""
        return {
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_cuda_device': torch.cuda.current_device() if torch.cuda.is_available() else None
        }


class SeismicRegressionService:
    """Service for seismic regression predictions (deformation forecasting)."""

    def __init__(self, device: str = "auto"):
        """
        Initialize the regression service.

        Args:
            device: Device to run inference on
        """
        self.device = self._setup_device(device)
        self.model = None

        # Load regression model (when available)
        self._load_model()

    def _setup_device(self, device: str) -> str:
        """Setup the compute device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self):
        """Load the seismic regression model."""
        try:
            # For now, we don't have a regression model
            # This will be implemented when regression training is complete
            logger.info("Seismic regression model not yet available")
        except Exception as e:
            logger.warning(f"Failed to load seismic regression model: {e}")

    def predict_deformation(self, seismic_sequence: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Predict deformation from seismic sequence.

        Args:
            seismic_sequence: Seismic sequence data

        Returns:
            Dictionary with deformation predictions or None if model not available
        """
        if self.model is None:
            logger.warning("Seismic regression model not available")
            return None

        try:
            # Convert to tensor
            if isinstance(seismic_sequence, np.ndarray):
                seismic_sequence = torch.from_numpy(seismic_sequence).float()

            seismic_sequence = seismic_sequence.to(self.device)

            # Run inference
            with torch.no_grad():
                predictions = self.model(seismic_sequence)

            # Convert to numpy for response
            predictions_np = predictions.cpu().numpy()

            return {
                'deformation_predictions': predictions_np.tolist(),
                'shape': predictions_np.shape
            }

        except Exception as e:
            logger.error(f"Error in deformation prediction: {e}")
            raise RuntimeError(f"Deformation prediction failed: {e}")


# Global service instances
inference_service = SeismicInferenceService()
regression_service = SeismicRegressionService()


def get_seismic_prediction(seismic_data: np.ndarray) -> Dict[str, Any]:
    """Convenience function for seismic state prediction."""
    return inference_service.predict_seismic_state(seismic_data)


def get_weather_prediction(weather_features: np.ndarray) -> Dict[str, Any]:
    """Convenience function for weather prediction."""
    return inference_service.predict_weather(weather_features)


def get_dsa_prediction(dsa_input: np.ndarray) -> Dict[str, Any]:
    """Convenience function for DSA prediction."""
    return inference_service.predict_dsa(dsa_input)


def get_deformation_prediction(seismic_sequence: np.ndarray) -> Optional[Dict[str, Any]]:
    """Convenience function for deformation prediction."""
    return regression_service.predict_deformation(seismic_sequence)


if __name__ == "__main__":
    # Test the inference service
    print("Testing Seismic AI Inference Service...")

    # Check model status
    status = inference_service.get_model_status()
    print(f"Model status: {status}")

    # Check device info
    device_info = inference_service.get_device_info()
    print(f"Device info: {device_info}")

    print("Inference service initialized successfully!")