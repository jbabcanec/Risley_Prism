# Reverse Risley Prism Solver

Clean implementation using REAL PHYSICS from the verified forward model.

## Core Files

- `core.py` - Data structures and interface to real physics model
- `neural_network.py` - Neural network implementation
- `genetic_algorithm.py` - Optional GA refinement
- `pipeline.py` - Complete training pipeline using REAL PHYSICS
- `run.sh` - Run everything

## To Run

```bash
./run.sh
```

This will:
1. Generate 50,000 training samples using REAL PHYSICS (model.py)
2. Train neural network for 300 epochs
3. Evaluate on test set
4. Save results

Expected time: 6-12 hours (real physics is slower but accurate)

## Results

After running, you'll get:
- `model_*` - Trained neural network
- `dataset_*.pkl` - Training data from real physics
- `results_*.json` - Performance metrics

Expected performance with real physics:
- Wedge classification: 60-70% accuracy
- Speed error: ~0.8 Hz MAE
- Angle error: ~4-5° MAE

## The Physics

The pipeline uses `ForwardModel.simulate()` which calls the verified `model.main()` to generate training data. This is REAL physics with:
- Iterative refraction through multiple surfaces
- Snell's law in vector form
- Proper rotation matrices
- Actual beam propagation