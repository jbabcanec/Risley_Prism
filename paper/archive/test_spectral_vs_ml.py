#!/usr/bin/env python3
"""
Diagnostic: compare spectral init vs ML init for 9-D recovery.
For each of 30 random cases:
  1. Extract FFT at TRUE speed frequencies → spectral αₓ, αᵧ
  2. Compare spectral vs ML init errors
  3. Run TRF from each → compare success rates
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from scipy.optimize import least_squares
from ml_staged_solver import (
    AngleNet, RemainNet, ANG_MODEL, REM_MODEL,
    DEVICE, N_PAR, NAMES, LO, HI, RG, canon,
    extract_speeds_and_peaks, _build_peak_feats_single, P,
    T_PTS, T_OBS, SRC, THK, FREQS,
)
from solve_preconditioned import vec2pat, ml_init

# Load ML models
print("Loading models...", flush=True)
ang = AngleNet(); rem = RemainNet()
ang.load_state_dict(torch.load(ANG_MODEL, map_location=DEVICE, weights_only=True))
rem.load_state_dict(torch.load(REM_MODEL, map_location=DEVICE, weights_only=True))
ang.to(DEVICE); rem.to(DEVICE)

# Generate 30 random cases (same seed as test_dimensions.py)
rng = np.random.default_rng(2026)
cases = []
for i in range(30):
    v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15:
            v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))


def spectral_init_at_speeds(speeds, pattern):
    """
    Extract angles from FFT at the given speed frequencies.

    Paraxial model: p_x(t) ≈ Σ_i d_eff_i * (n_i-1) * tan(αₓ,ᵢ) * cos(2π|Nᵢ|t + φᵢ)
    where φᵢ depends on αᵧ,ᵢ and the sign of Nᵢ.

    Returns 18-D parameter vector.
    """
    fx = np.fft.rfft(pattern[:, 0])
    fy = np.fft.rfft(pattern[:, 1])
    freqs = np.fft.rfftfreq(T_PTS, d=T_OBS / T_PTS)

    v = np.zeros(N_PAR, dtype=np.float64)
    v[:3] = speeds

    nom_ng = 1.5
    nom_dw = 100.0

    for i in range(P):
        # Find FFT bin closest to |speed_i|
        target_f = abs(speeds[i])
        bin_idx = np.argmin(np.abs(freqs - target_f))
        if bin_idx == 0:
            bin_idx = 1

        # Complex FFT coefficients
        cx = fx[bin_idx]
        cy = fy[bin_idx]

        # Amplitude (scale by 2/T for real amplitude)
        amp_x = abs(cx) * 2 / T_PTS
        amp_y = abs(cy) * 2 / T_PTS

        # αᵧ from phase: in paraxial limit, the x-component phase at Nᵢ ≈ αᵧ,ᵢ
        # For negative speed: cos(-2π|N|t + αᵧ) = cos(2π|N|t - αᵧ), so phase = -αᵧ
        phase = np.angle(cx)
        if speeds[i] < 0:
            alpha_y = -np.degrees(phase)
        else:
            alpha_y = np.degrees(phase)

        # αₓ from amplitude: A ≈ d_eff * (n_g - 1) * tan(αₓ)
        # d_eff for prism i: remaining path to workpiece
        # With 3 prisms at positions ~SRC, SRC+THK+gap, SRC+2*(THK+gap)
        # d_eff_i ≈ (P-1-i)*(THK+gap) + d_W
        d_eff = (P - 1 - i) * (THK + 8.0) + nom_dw  # gap nominal = 8
        alpha_x_rad = np.arctan(amp_x / (d_eff * (nom_ng - 1.0) + 1e-10))
        alpha_x = np.degrees(alpha_x_rad)
        alpha_x = max(alpha_x, 0.5)  # floor

        v[3 + i] = alpha_x
        v[6 + i] = alpha_y

    # Nominal glass, geometry, beam
    v[9:12] = nom_ng
    v[12] = nom_dw
    v[13] = 8.0
    v[14:18] = 0.0

    return np.clip(v, LO, HI)


print(f"\n{'='*80}")
print("SPECTRAL vs ML INIT DIAGNOSTIC — 9-D (speeds+angles, geometry fixed)")
print(f"{'='*80}\n")

ml_perfect = 0
spec_perfect = 0
both_perfect = 0
either_perfect = 0

results = []

for ci, tc in enumerate(cases):
    pat = vec2pat(tc)
    target = pat.reshape(-1)
    pf, pi_info = extract_speeds_and_peaks(pat)
    pk = _build_peak_feats_single(pat, pf, pi_info)

    true_speeds = tc[:3]
    true_ax = tc[3:6]
    true_ay = tc[6:9]
    true_9d = tc[:9]

    # Fixed: glass/geometry/beam at true values
    fixed = tc[9:].copy()
    lo9 = LO[:9].astype(np.float64)
    hi9 = HI[:9].astype(np.float64)

    def make_res_9d(fix):
        def residual(x9):
            theta = np.concatenate([x9, fix])
            return vec2pat(theta).reshape(-1) - target
        return residual

    res_fn = make_res_9d(fixed)

    # --- ML init: try 8 sign combos ---
    ml_best_mse = 1e30
    ml_best_x = None
    ml_best_init_err = 999

    for bits in range(8):
        signs = np.array([(1.0 if (bits >> j) & 1 == 0 else -1.0) for j in range(P)], np.float64)
        speeds = signs * np.sort(pf)[::-1].astype(np.float64)
        ml = ml_init(ang, rem, speeds.astype(np.float32), pk, pat)

        x0 = ml[:9]
        init_err = float(np.max(np.abs(x0 - true_9d)))
        if init_err < ml_best_init_err:
            ml_best_init_err = init_err

        try:
            res = least_squares(res_fn, x0, jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            mse = float(np.mean(res.fun ** 2))
            if mse < ml_best_mse:
                ml_best_mse = mse
                ml_best_x = res.x.copy()
        except Exception:
            pass

    ml_err = float(np.max(np.abs(ml_best_x - true_9d))) if ml_best_x is not None else 999
    ml_ok = ml_err < 1e-3

    # --- Spectral init: try 8 sign combos ---
    spec_best_mse = 1e30
    spec_best_x = None
    spec_best_init_err = 999

    for bits in range(8):
        signs = np.array([(1.0 if (bits >> j) & 1 == 0 else -1.0) for j in range(P)], np.float64)
        speeds = signs * np.sort(pf)[::-1].astype(np.float64)
        sp = spectral_init_at_speeds(speeds, pat)

        x0 = sp[:9]
        init_err = float(np.max(np.abs(x0 - true_9d)))
        if init_err < spec_best_init_err:
            spec_best_init_err = init_err

        try:
            res = least_squares(res_fn, x0, jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            mse = float(np.mean(res.fun ** 2))
            if mse < spec_best_mse:
                spec_best_mse = mse
                spec_best_x = res.x.copy()
        except Exception:
            pass

    spec_err = float(np.max(np.abs(spec_best_x - true_9d))) if spec_best_x is not None else 999
    spec_ok = spec_err < 1e-3

    if ml_ok: ml_perfect += 1
    if spec_ok: spec_perfect += 1
    if ml_ok and spec_ok: both_perfect += 1
    if ml_ok or spec_ok: either_perfect += 1

    tag_ml = "P" if ml_ok else "F"
    tag_sp = "P" if spec_ok else "F"

    # Show per-case detail for interesting cases
    if tag_ml != tag_sp:
        print(f"  Case {ci+1:2d}: ML={tag_ml} (init_err={ml_best_init_err:.2f}, "
              f"final_err={ml_err:.2e})  "
              f"SPEC={tag_sp} (init_err={spec_best_init_err:.2f}, "
              f"final_err={spec_err:.2e})  ***DIFFER***", flush=True)
    else:
        sys.stdout.write(".")
        sys.stdout.flush()

    results.append({
        'case': ci,
        'ml_ok': ml_ok, 'spec_ok': spec_ok,
        'ml_init_err': ml_best_init_err, 'spec_init_err': spec_best_init_err,
        'ml_final_err': ml_err, 'spec_final_err': spec_err,
        'ml_mse': ml_best_mse, 'spec_mse': spec_best_mse,
        'true_speeds': true_speeds.tolist(),
    })

print(flush=True)
print(f"\n{'='*80}")
print(f"  ML init:       {ml_perfect}/30 PERFECT")
print(f"  Spectral init: {spec_perfect}/30 PERFECT")
print(f"  Both:          {both_perfect}/30")
print(f"  Either (union):{either_perfect}/30")
print(f"{'='*80}")

# Save
with open(os.path.join(os.path.dirname(__file__), 'spectral_vs_ml_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("Saved to spectral_vs_ml_results.json")
