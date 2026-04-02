#!/bin/bash

# Run the pipeline for reverse Risley prism problem

echo "=================================================="
echo "REVERSE RISLEY PRISM TRAINING PIPELINE"
echo "=================================================="
echo ""
echo "This will:"
echo "1. Generate 50,000 training samples using REAL PHYSICS"
echo "2. Train neural network for 300 epochs"
echo "3. Evaluate performance on test set"
echo ""
echo "Using verified forward model (model.py) for data generation"
echo "Expected time: 6-12 hours (real physics takes time)"
echo "=================================================="
echo ""

# Run pipeline
python3 pipeline.py --samples 50000 --epochs 300

echo ""
echo "Pipeline complete. Check results_*.json for metrics."