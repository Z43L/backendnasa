"""
Unified Model Loader for Seismic AI Backend

This module provides a unified interface for loading different types of trained models
including PyTorch models, SafeTensors, and Pickle models.
"""

import os
import torch
import pickle
from safetensors.torch import load_file as safetensors_load
from typing import Dict, Any, Optional, Union
from pathlib import Path


class ModelLoader:
    """Unified model loader supporting multiple formats and model types."""

    def __init__(self, models_dir: str = None):
        """
        Initialize the model loader.

        Args:
            models_dir: Base directory containing model subdirectories.
                       If None, will search for it relative to this file.
        """
        if models_dir is None:
            # Try to find models directory relative to this file
            backend_dir = Path(__file__).parent
            possible_dirs = [
                backend_dir / "models",  # backend/models/
                backend_dir.parent / "models",  # project_root/models/
            ]

            for possible_dir in possible_dirs:
                if possible_dir.exists():
                    self.models_dir = possible_dir
                    break
            else:
                # Create default path
                self.models_dir = backend_dir / "models"
        else:
            self.models_dir = Path(models_dir)

        self.loaded_models = {}

    def load_pytorch_model(self, model_path: Union[str, Path], device: str = "cpu") -> torch.nn.Module:
        """Load a PyTorch model (.pth, .pt)."""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load the model
            model = torch.load(model_path, map_location=device, weights_only=False)

            # If it's a state_dict, we need the architecture
            if isinstance(model, dict):
                # For now, return the state_dict - caller needs to know the architecture
                return model

            model.to(device)
            model.eval()
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model {model_path}: {e}")

    def load_safetensors_model(self, model_path: Union[str, Path], device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Load a SafeTensors model."""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            state_dict = safetensors_load(model_path)
            # Move tensors to specified device
            state_dict = {k: v.to(device) for k, v in state_dict.items()}
            return state_dict

        except Exception as e:
            raise RuntimeError(f"Failed to load SafeTensors model {model_path}: {e}")

    def load_pickle_model(self, model_path: Union[str, Path]) -> Any:
        """Load a Pickle model."""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load Pickle model {model_path}: {e}")

    def load_model(self, model_type: str, model_name: str, device: str = "cpu") -> Any:
        """
        Load a model by type and name.

        Args:
            model_type: Type of model ('seismic_classification', 'seismic_regression', 'weather', 'dsa')
            model_name: Name of the model file (without extension)
            device: Device to load the model on ('cpu' or 'cuda')

        Returns:
            Loaded model object
        """
        model_dir = self.models_dir / model_type
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Try different extensions in order of preference
        extensions = ['.safetensors', '.pth', '.pt', '.pkl']

        for ext in extensions:
            model_path = model_dir / f"{model_name}{ext}"
            if model_path.exists():
                if ext == '.safetensors':
                    state_dict = self.load_safetensors_model(model_path, device)
                    return self._create_model_from_state_dict(model_type, state_dict, device)
                elif ext in ['.pth', '.pt']:
                    model_or_state = self.load_pytorch_model(model_path, device)
                    if isinstance(model_or_state, dict):
                        return self._create_model_from_state_dict(model_type, model_or_state, device)
                    else:
                        return model_or_state
                elif ext == '.pkl':
                    return self.load_pickle_model(model_path)
                break

        # If no model found, list available models
        available = list(model_dir.glob("*"))
        available_names = [f.stem for f in available]
        raise FileNotFoundError(f"Model '{model_name}' not found in {model_type}. Available: {available_names}")

    def _create_model_from_state_dict(self, model_type: str, state_dict: Dict[str, Any], device: str = "cpu") -> Any:
        """
        Create a model instance and load state_dict based on model type.

        Args:
            model_type: Type of model
            state_dict: Model state dictionary or checkpoint
            device: Device to load the model on

        Returns:
            Model with loaded weights
        """
        # Check if this is a full checkpoint or just a state_dict
        if 'model_state_dict' in state_dict:
            # This is a full checkpoint
            model_params = state_dict.get('model_params', {})
            actual_state_dict = state_dict['model_state_dict']
        else:
            # This is just a state_dict
            actual_state_dict = state_dict
            model_params = {}

        if model_type == 'seismic_classification':
            # Import the architecture
            try:
                from model_architecture import create_model
                # Use saved params but ensure they match the actual state_dict
                params = {
                    'task_type': 'classification',
                    'seq_length': 30,
                    'grid_size': (50, 50),
                    'num_classes': 3
                }
                params.update(model_params)
                # Fix parameters to match the actual checkpoint
                params['d_model'] = 128  # Override with correct value
                params['num_encoder_layers'] = 3  # State dict has layers 0-2
                params['dim_feedforward'] = 512  # Override with correct value
                model = create_model(**params)
            except ImportError:
                raise RuntimeError("Could not import model architecture for seismic classification")

        elif model_type == 'seismic_regression':
            try:
                from model_architecture import create_model
                # For regression, use the saved params but ensure we have the right defaults
                params = {
                    'task_type': 'regression',
                    'seq_length': 30,
                    'grid_size': (50, 50)
                }
                params.update(model_params)
                # Ensure num_encoder_layers matches the state_dict (which has 6 layers: 0-5)
                if 'num_encoder_layers' not in params or params['num_encoder_layers'] != 6:
                    params['num_encoder_layers'] = 6
                model = create_model(**params)
            except ImportError:
                raise RuntimeError("Could not import model architecture for seismic regression")

        elif model_type == 'weather':
            # For weather models, they appear to be sklearn-like models stored as dict
            # Return as-is for now
            return state_dict

        elif model_type == 'dsa':
            # For DSA models, they appear to be stored as dict
            return state_dict

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load the state dict
        model.load_state_dict(actual_state_dict)
        model.to(device)
        model.eval()
        return model

    def get_available_models(self) -> Dict[str, list]:
        """Get a dictionary of available models by type."""
        available = {}
        if not self.models_dir.exists():
            return available

        for model_type_dir in self.models_dir.iterdir():
            if model_type_dir.is_dir():
                model_files = list(model_type_dir.glob("*"))
                model_names = [f.stem for f in model_files if f.suffix in ['.pth', '.pt', '.pkl', '.safetensors']]
                if model_names:
                    available[model_type_dir.name] = model_names

        return available

    def preload_models(self, device: str = "cpu") -> None:
        """Preload all available models into memory."""
        available = self.get_available_models()
        for model_type, model_names in available.items():
            for model_name in model_names:
                try:
                    model = self.load_model(model_type, model_name, device)
                    self.loaded_models[f"{model_type}/{model_name}"] = model
                    print(f"Preloaded model: {model_type}/{model_name}")
                except Exception as e:
                    print(f"Failed to preload {model_type}/{model_name}: {e}")

    def get_model(self, model_type: str, model_name: str) -> Any:
        """Get a preloaded model."""
        key = f"{model_type}/{model_name}"
        if key not in self.loaded_models:
            raise KeyError(f"Model not preloaded: {key}")
        return self.loaded_models[key]


# Global model loader instance
model_loader = ModelLoader()


def load_seismic_classification_model(device: str = "cpu"):
    """Load the seismic classification model."""
    return model_loader.load_model("seismic_classification", "cl_falla_anatolia", device)


def load_seismic_regression_model(device: str = "cpu"):
    """Load the seismic regression model."""
    return model_loader.load_model("seismic_regression", "re_galla_anatolia", device)


def load_weather_model(device: str = "cpu"):
    """Load the weather prediction model."""
    return model_loader.load_model("weather", "large_weather_model_20251003_192729", device)


def load_dsa_model(device: str = "cpu"):
    """Load the DSA model."""
    return model_loader.load_model("dsa", "dsa_model", device)


def load_test_model(device: str = "cpu"):
    """Load the test model."""
    return model_loader.load_model("test_model", "test_model", device)


if __name__ == "__main__":
    # Test the model loader
    print("Available models:")
    available = model_loader.get_available_models()
    for model_type, models in available.items():
        print(f"  {model_type}: {models}")

    print("\nTesting model loading...")
    try:
        # Test loading seismic classification model
        seismic_model = load_seismic_classification_model()
        print(f"Seismic classification model loaded: {type(seismic_model)}")

        # Test loading weather model
        weather_model = load_weather_model()
        print(f"Weather model loaded: {type(weather_model)}")

        # Test loading DSA model
        dsa_model = load_dsa_model()
        print(f"DSA model loaded: {type(dsa_model)}")

    except Exception as e:
        print(f"Error during testing: {e}")