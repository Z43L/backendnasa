#!/usr/bin/env python3
"""
Demo script for seismic regression inference using the re_galla_anatolia.pth model.

This script demonstrates how to use the complete seismic regression inference pipeline
to predict deformation from seismic sequences.
"""

import numpy as np
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.dirname(__file__))

from inference_service import SeismicInferenceService

def main():
    print("Seismic Regression Inference Demo")
    print("=" * 40)

    # Initialize the service
    print("Loading models...")
    service = SeismicInferenceService()

    # Check status
    status = service.get_model_status()
    print(f"Model status: {status}")

    if not status['seismic_regression']:
        print("❌ Seismic regression model not available")
        return

    # Create sample seismic data (30 time steps, 50x50 grid)
    print("\nGenerating sample seismic sequence...")
    np.random.seed(42)  # For reproducible results

    # Create a sequence with some realistic deformation patterns
    seq_length, height, width = 30, 50, 50
    sequence = []

    for t in range(seq_length):
        # Base deformation with fault-like features
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        deformation = 0.05 * np.sin(0.1 * x) * np.cos(0.1 * y)
        deformation += 0.02 * np.sin(0.05 * (x + y))  # Diagonal fault
        deformation += 0.01 * np.random.randn(height, width)  # Noise

        # Add temporal evolution
        deformation *= (1 + 0.1 * np.sin(0.2 * t))

        sequence.append(deformation)

    seismic_sequence = np.array(sequence)
    print(f"Created seismic sequence: {seismic_sequence.shape}")

    # Run inference
    print("\nRunning seismic deformation prediction...")
    result = service.predict_seismic_deformation(seismic_sequence)

    # Display results
    print("✅ Prediction completed!")
    print(f"Output shape: {result['shape']}")
    print(f"Description: {result['description']}")
    print(f"Units: {result['units']}")

    # Show some statistics
    predictions = np.array(result['deformation_prediction'])[0]  # Remove batch dimension
    print("\nPrediction Statistics:")
    print(f"  Mean deformation: {predictions.mean():.6f}")
    print(f"  Std deformation: {predictions.std():.6f}")
    print(f"  Max deformation: {predictions.max():.6f}")
    print(f"  Min deformation: {predictions.min():.6f}")

    # Show a small sample of the prediction grid
    print("\nSample prediction values (top-left 5x5 corner):")
    print(predictions[:5, :5])

    print("\n🎉 Demo completed successfully!")
    print("\nThe seismic regression model is now fully operational for predicting")
    print("terrain deformation from seismic sequences using the re_galla_anatolia.pth model.")

if __name__ == "__main__":
    main()