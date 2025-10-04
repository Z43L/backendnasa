#!/usr/bin/env python3
"""
Test script for seismic regression inference using the re_galla_anatolia.pth model.
This script tests the complete inference pipeline for seismic deformation prediction.
"""

import numpy as np
import torch
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.dirname(__file__))

from inference_service import SeismicInferenceService

def create_sample_seismic_data(seq_length: int = 30, height: int = 50, width: int = 50) -> np.ndarray:
    """
    Create sample seismic data for testing.

    Args:
        seq_length: Number of time steps in the sequence
        height: Height of the deformation grid
        width: Width of the deformation grid

    Returns:
        Sample seismic sequence with shape (seq_length, height, width)
    """
    # Create synthetic deformation data that simulates tectonic movement
    # Start with some base deformation pattern
    base_pattern = np.zeros((height, width))

    # Add some tectonic features (fault lines, uplift zones)
    # Simulate a fault line diagonally across the grid
    for i in range(height):
        for j in range(width):
            # Distance from diagonal fault line
            fault_distance = abs(i - j) / max(height, width)
            # Simulate deformation that increases near the fault
            deformation = 0.1 * np.exp(-fault_distance * 10) * np.sin(i * 0.1) * np.cos(j * 0.1)
            base_pattern[i, j] = deformation

    # Create temporal sequence with slight variations
    sequence = []
    for t in range(seq_length):
        # Add temporal variation
        time_factor = 1.0 + 0.1 * np.sin(t * 0.2)
        frame = base_pattern * time_factor

        # Add some noise to simulate real data
        noise = np.random.normal(0, 0.01, (height, width))
        frame += noise

        sequence.append(frame)

    return np.array(sequence)

def test_seismic_regression_inference():
    """Test the seismic regression inference pipeline."""
    print("Testing Seismic Regression Inference")
    print("=" * 50)

    try:
        # Initialize the inference service
        print("Initializing inference service...")
        service = SeismicInferenceService(device="cpu")

        # Check model status
        print("Checking model status...")
        status = service.get_model_status()
        print(f"Model status: {status}")

        if not status.get('seismic_regression', False):
            print("❌ Seismic regression model not loaded!")
            return False

        print("✅ Seismic regression model loaded successfully")

        # Create sample data
        print("\nCreating sample seismic data...")
        sample_data = create_sample_seismic_data(seq_length=30, height=50, width=50)
        print(f"Sample data shape: {sample_data.shape}")
        print(".3f")

        # Run inference
        print("\nRunning seismic deformation prediction...")
        result = service.predict_seismic_deformation(sample_data)

        print("✅ Inference completed successfully!")
        print(f"Prediction shape: {result['shape']}")
        print(f"Input shape: {result['input_shape']}")
        print(f"Description: {result['description']}")
        print(f"Units: {result['units']}")

        # Basic validation
        expected_shape = (1, 50, 50)  # batch_size=1, height=50, width=50
        if result['shape'] != expected_shape:
            print(f"❌ Unexpected prediction shape. Expected {expected_shape}, got {result['shape']}")
            return False

        # Check that prediction values are reasonable (not all zeros, not extreme values)
        prediction_array = np.array(result['deformation_prediction'])
        if np.allclose(prediction_array, 0):
            print("⚠️  Warning: All predictions are zero")
        elif np.any(np.abs(prediction_array) > 10):
            print("⚠️  Warning: Some predictions have extreme values")

        print(".3f")
        print(".3f")

        print("\n✅ All tests passed! Seismic regression inference is working correctly.")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading_directly():
    """Test loading the model directly to ensure it works."""
    print("\nTesting Direct Model Loading")
    print("=" * 30)

    try:
        from model_loader import load_seismic_regression_model

        print("Loading seismic regression model directly...")
        model = load_seismic_regression_model(device="cpu")

        if model is None:
            print("❌ Model loading returned None")
            return False

        print(f"✅ Model loaded successfully: {type(model)}")
        print(f"Model task type: {getattr(model, 'task_type', 'unknown')}")

        # Test forward pass with sample data
        print("Testing forward pass...")
        sample_input = torch.randn(1, 30, 50, 50)  # batch_size=1, seq_length=30, height=50, width=50

        with torch.no_grad():
            output = model(sample_input)

        print(f"✅ Forward pass successful. Output shape: {output.shape}")
        expected_shape = (1, 50, 50)
        if output.shape != expected_shape:
            print(f"❌ Unexpected output shape. Expected {expected_shape}, got {output.shape}")
            return False

        return True

    except Exception as e:
        print(f"❌ Direct model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Seismic Regression Inference Test Suite")
    print("=" * 50)

    # Test 1: Direct model loading
    test1_passed = test_model_loading_directly()

    # Test 2: Full inference pipeline
    test2_passed = test_seismic_regression_inference()

    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print(f"Direct Model Loading: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Full Inference Pipeline: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! Seismic regression inference is fully implemented.")
        sys.exit(0)
    else:
        print("\n💥 SOME TESTS FAILED. Please check the implementation.")
        sys.exit(1)