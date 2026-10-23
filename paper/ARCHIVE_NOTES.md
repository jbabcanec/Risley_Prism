# Archive Notes — Supplementary Analysis Not in Paper

These results were computed during the bulletproofing process and are
preserved here for future use. Raw data is in the JSON files.

## 1. Real NN Ablation (analysis_results_v2.json → real_nn_ablation)

Trained actual PyTorch NN on 100k samples (50k per prism count), 100 epochs.
- Classification: 98.8% accuracy
- Regression MAE: ~1 Hz speed, ~5 deg angles (per-prism)
- NN+DE vs DE-only: identical MSE (1e-24), identical wall time (~100s)
- **Result: NN warm start provides NO speedup. Value is classification only.**

This was in the original 13-page paper but compressed to 1 paragraph in the
8-page version. Full NN architecture details are in neural_network.py.

## 2. NN Regression MAE by Prism (was Table tab:nn, now cut)

| P | Prism | |dN| (Hz) | |d_ax| (deg) | |d_ay| (deg) |
|---|-------|----------|-------------|-------------|
| 2 | 1     | 1.065    | 5.03        | 5.46        |
| 2 | 2     | 1.046    | 4.92        | 5.33        |
| 3 | 1     | 1.200    | 5.77        | 6.68        |
| 3 | 2     | 1.226    | 5.69        | 6.55        |
| 3 | 3     | 1.216    | 5.78        | 6.40        |

Key observation: per-prism error is approximately uniform (physical
2-interface model distributes sensitivity evenly, unlike single-interface
model where prism 1 dominated).

## 3. Hessian Error-Bound Validation (analysis_results.json → error_pred)

| Param | H_jj      | Predicted bound | Actual error | Ratio |
|-------|-----------|----------------|-------------|-------|
| N1    | 2.24e+05  | 2.11e-13       | 6.66e-16    | 0.003 |
| N2    | 1.01e+05  | 3.15e-13       | 1.78e-15    | 0.006 |
| N3    | 4.03e+04  | 4.98e-13       | 1.95e-14    | 0.039 |
| ax1   | 1.32e+00  | 8.70e-11       | 1.41e-12    | 0.016 |
| ax2   | 1.31e+00  | 8.75e-11       | 7.11e-13    | 0.008 |
| ax3   | 1.26e+00  | 8.90e-11       | 2.35e-12    | 0.026 |
| ay1   | 5.23e-02  | 4.37e-10       | 3.40e-12    | 0.008 |
| ay2   | 2.35e-02  | 6.52e-10       | 3.17e-12    | 0.005 |
| ay3   | 9.41e-03  | 1.03e-09       | 2.85e-11    | 0.028 |

Condition number kappa = 2.38e+07
Actual errors are 0.3-4% of Hessian-predicted bounds.

## 4. Extended P=4 Results (analysis_results_v2.json → p4_extended)

| Case           | MSE      | |dN| max   | |dax| max  | |day| max  | Time  |
|----------------|----------|-----------|-----------|-----------|-------|
| 4P easy        | 2.12e-23 | 1.29e-14  | 2.37e-12  | 2.78e-11  | 882s  |
| 4P close speeds| 2.57e-24 | 1.07e-14  | 1.13e-12  | 2.38e-11  | 890s  |
| 4P small angles| 6.41e-24 | 3.73e-14  | 7.00e-13  | 8.06e-11  | 943s  |
| 4P large angles| 4.12e-23 | 1.20e-14  | 2.87e-12  | 2.66e-11  | 865s  |

## 5. 14-D Full System Recovery (speeds + angles + glass + distances)

Recovered all 14 parameters for 3-prism system to floating-point precision:
- N = [1.500, -1.000, 2.000], ax = [12.000, -8.000, 5.000]
- ay = [3.000, 10.000, -6.000], n_g = [1.5000, 1.5500, 1.6000]
- D_wp = 120.000, gap = 8.000
- MSE = 4.4e-18, 2.1M evaluations, 48 min
- Required 7 restarts before one landed in global basin

## 6. Convergence Figure Data (was Fig 8, reconstruction.pdf)

DE seeded with x0=[0.5,-0.3,0.2, 8.0,-3.0,2.0, 2.0,-1.0,-0.5]
Snapshots captured at 5k, 12k, 18k evaluations.
Final NM polish: MSE=5.6e-03 (limited by DE not fully converging in this run).
Full pipeline with 4+ restarts reaches MSE~1e-24.

## 7. Model Mismatch Full Results (analysis_results_v2.json → model_mismatch)

| Perturbation         | Residual MSE | |dN| (Hz) | |dax| (deg) | |day| (deg) |
|---------------------|-------------|----------|------------|------------|
| Exact geometry      | 1.02e-12    | 8.01e-09 | 3.48e-07   | 1.18e-05   |
| thickness +3%       | 6.64e-04    | 1.20e-05 | 1.40e-02   | 2.15e-02   |
| thickness +17%      | 1.52e-02    | >1       | >10        | >10        |
| gap +8%             | 1.65e-02    | >1       | >10        | >10        |
| gap +33%            | 2.80e-01    | >1       | >10        | >10        |
| workpiece +5%       | 4.44e-01    | 2.90e-04 | 5.35e-01   | 5.13e-01   |
| workpiece +20%      | 6.95e+00    | >1       | >10        | >10        |
| beam angle +10%     | 2.61e+00    | >1       | >10        | >10        |
| all perturbed       | 1.75e+00    | >1       | >10        | >10        |

Interesting: workpiece distance error degrades angles but NOT speeds (until
20% error). Thickness/gap errors break everything at ~15%.

## 8. Glass-Based Identifiability (DISPROVEN)

Original hypothesis: distinct glass types needed for identifiability.
Numerical test (5000 random perturbations):
- Uniform glass: 511/5000 have MSE<1.0 (10.3%)
- Distinct glass: 446/5000 have MSE<1.0 (8.9%)
Difference is ~10%, NOT orders of magnitude. Glass type is irrelevant;
speed ratio is the real identifiability driver.

## 9. First OOD Experiment (simple patterns)

| Pattern              | MSE     |
|---------------------|---------|
| Real Risley         | 1.9e-03 |  (solver settings too weak)
| Circle              | 1.7e-01 |
| Square              | 3.6e-01 |
| Random walk         | 4.6e+00 |
| Spiral              | 3.4e-01 |
| Figure-8            | 1.9e+01 |
| Constant point      | 2.7e-15 |  (achievable: zero angles)
| Sawtooth            | 5.8e+00 |

## Files

- `analysis_results.json` — v1 analysis (noise single-trial, proxy ablation, Yang, extended battery, P=4, Hessian)
- `analysis_results_v2.json` — v2 analysis (model mismatch, P=4 extended, real NN ablation)
- `analysis_results_v2_noise.json` — 30-trial noise study cache
- `run_analysis.py` — v1 analysis script
- `run_analysis_v2.py` — v2 analysis script
- `generate_figures.py` — all figure generation (including cut figures)
