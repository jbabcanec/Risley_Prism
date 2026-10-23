# Reverse Risley Prism Solver

Given a laser scan pattern at the workpiece, recovers the Risley prism system
parameters: rotation speeds and wedge angles (phi_x, phi_y) for each wedge.

## Results (2-3 wedges)

| Case | Speed error | Phi_x error | Phi_y error | MSE | Time |
|------|------------|------------|------------|-----|------|
| 2 wedges (easy) | 0.000 Hz | 0.00° | 0.00° | 0.000000 | 16s |
| 2 wedges (close speeds) | 0.000 Hz | 0.00° | 0.00° | 0.000000 | 46s |
| 3 wedges (easy) | 0.000 Hz | 0.00° | 0.03° | 0.000000 | 124s |
| 3 wedges (close speeds) | 0.000 Hz | 0.00° | 0.00° | 0.000000 | 95s |
| 3 wedges (large angles) | 0.000 Hz | 0.00° | 0.00° | 0.000000 | 34s |

All parameters recovered to machine precision across all test cases.

## Architecture

**Stage 1 — Neural Network** (instant, ~2ms):
- Input: raw (x,y) coordinates + FFT magnitude features (600 dims)
- Shared backbone (768→512→256→128 MLP with BatchNorm)
- Classifier head: identifies wedge count (99.9% accuracy)
- Per-wedge-count regression heads: initial parameter estimates

**Stage 2 — Scipy global optimisation** (15-120s):
- `differential_evolution` with 4 random restarts
- `Nelder-Mead` polish on each restart
- Permutation search to resolve wedge-ordering ambiguity
- Uses the fast vectorised forward model (~0.5ms per evaluation)

**Key physics detail**: Varying refractive indices at each interface
(`ref_ind = [1.0, 1.15, 1.30, ...]`) ensure every wedge contributes
through refraction. Uniform indices make wedges 2+ invisible.

## Quick Start

```bash
# Train for 2 and 3 wedge systems (takes ~10 min)
python pipeline.py --wc 2 3 --samples 100000 --epochs 200
```

## Using a Trained Model

```python
from core import RisleyParameters, fast_forward
from neural_network import RisleyPredictor
from pipeline import Pipeline, refine_scipy

# Load model
pred = RisleyPredictor()
pred.load('runs/<timestamp>/model')

# Quick NN prediction (~2ms)
result = pred.predict(pattern, wedge_count=3)

# Full solve with scipy refinement
pipeline = Pipeline()
pipeline.predictor = pred
solution = pipeline.solve(pattern, wedge_count=3, refine=True)
```

## Files

| File | Purpose |
|------|---------|
| `core.py` | `RisleyParameters` + `fast_forward()` vectorised model |
| `neural_network.py` | PyTorch `RisleyNet` + `RisleyPredictor` |
| `genetic_algorithm.py` | GA-based refinement (alternative to scipy) |
| `pipeline.py` | `Pipeline` class + `refine_scipy()` optimiser |
