#!/usr/bin/env python3
"""
Combined init: ML αₓ + spectral αᵧ + 180° flip resolution.

Key insight from harmonic decomposition:
  - αᵧ from phase is correct to <5° for 50% of prisms
  - The other 50% are off by exactly 180° (paraxial degeneracy)
  - αₓ from amplitude is terrible (median 4.55° error)
  - ML gives better αₓ

Strategy:
  For each (speed_signs, freq_triple):
    1. ML init → get αₓ_ml, αᵧ_ml
    2. Harmonic decomp → get αᵧ_spec (up to ±180°)
    3. Combined: (αₓ_ml, αᵧ_spec)
    4. Flipped:  (-αₓ_ml, αᵧ_spec + 180°)
    5. TRF from best of {ml, combined, flipped}

Test: 9-D recovery on 30 cases.
"""
import sys, os, time, json
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from ml_staged_solver import (
    AngleNet, RemainNet, ANG_MODEL, REM_MODEL,
    DEVICE, N_PAR, LO, HI, RG, canon,
    extract_speeds_and_peaks, _build_peak_feats_single, P,
    T_PTS, T_OBS, SRC, THK,
)
from solve_preconditioned import vec2pat, ml_init


def harmonic_decompose(pattern, freqs):
    """Decompose pattern at exact frequencies using least-squares."""
    T = len(pattern)
    t = np.arange(T) * (T_OBS / T)
    n = len(freqs)
    X = np.zeros((T, 2*n + 1))
    for i, f in enumerate(freqs):
        X[:, 2*i] = np.cos(2*np.pi*f*t)
        X[:, 2*i+1] = np.sin(2*np.pi*f*t)
    X[:, -1] = 1.0

    bx, _, _, _ = np.linalg.lstsq(X, pattern[:, 0], rcond=None)
    by, _, _, _ = np.linalg.lstsq(X, pattern[:, 1], rcond=None)

    results = []
    for i in range(n):
        phase_x = np.arctan2(-bx[2*i+1], bx[2*i])
        phase_y = np.arctan2(-by[2*i+1], by[2*i])
        results.append({'phase_x': phase_x, 'phase_y': phase_y})
    return results


# Load ML models
print("Loading models...", flush=True)
ang = AngleNet(); rem = RemainNet()
ang.load_state_dict(torch.load(ANG_MODEL, map_location=DEVICE, weights_only=True))
rem.load_state_dict(torch.load(REM_MODEL, map_location=DEVICE, weights_only=True))
ang.to(DEVICE); rem.to(DEVICE)

# Generate 30 random cases
rng = np.random.default_rng(2026)
cases = []
for i in range(30):
    v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))

print(f"\n{'='*80}")
print("COMBINED INIT (ML αₓ + Spectral αᵧ + 180° flips) — 9-D TRF")
print(f"{'='*80}\n")

ml_only_perfect = 0
combined_perfect = 0
either_perfect = 0

for ci, tc in enumerate(cases):
    pat = vec2pat(tc)
    target = pat.reshape(-1)
    pf, pi_info = extract_speeds_and_peaks(pat)
    pk = _build_peak_feats_single(pat, pf, pi_info)

    fixed = tc[9:].copy()
    lo9 = LO[:9].astype(np.float64)
    hi9 = HI[:9].astype(np.float64)

    def make_res(fix):
        def residual(x9):
            return vec2pat(np.concatenate([x9, fix])).reshape(-1) - target
        return residual
    res_fn = make_res(fixed)

    ml_best_mse = 1e30
    ml_best_x = None
    comb_best_mse = 1e30
    comb_best_x = None

    for bits in range(8):
        signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
        speeds = signs * np.sort(pf)[::-1].astype(np.float64)

        # (A) ML init
        ml = ml_init(ang, rem, speeds.astype(np.float32), pk, pat)
        try:
            res = least_squares(res_fn, ml[:9], jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            mse = float(np.mean(res.fun**2))
            if mse < ml_best_mse: ml_best_mse = mse; ml_best_x = res.x.copy()
            if mse < comb_best_mse: comb_best_mse = mse; comb_best_x = res.x.copy()
        except Exception: pass

        # (B) Harmonic decomposition → αᵧ
        decomp = harmonic_decompose(pat, np.abs(speeds))

        for flip_bits in range(8):
            # For each prism, optionally flip (αₓ → -αₓ, αᵧ → αᵧ+180°)
            x0 = ml[:9].copy()
            for j in range(P):
                phase = decomp[j]['phase_x']
                # Base αᵧ from spectral
                if speeds[j] < 0:
                    ay_spec = -np.degrees(phase)
                else:
                    ay_spec = np.degrees(phase)

                if (flip_bits >> j) & 1:
                    # Flipped: αᵧ + 180°, αₓ negated
                    x0[6+j] = ay_spec + 180.0
                    if x0[6+j] > 180: x0[6+j] -= 360
                    x0[3+j] = -abs(ml[3+j])
                else:
                    # Unflipped: use spectral αᵧ, ML αₓ
                    x0[6+j] = ay_spec
                    x0[3+j] = abs(ml[3+j])

            x0 = np.clip(x0, lo9, hi9)

            try:
                res = least_squares(res_fn, x0, jac='2-point',
                    bounds=(lo9, hi9), method='trf',
                    ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
                mse = float(np.mean(res.fun**2))
                if mse < comb_best_mse: comb_best_mse = mse; comb_best_x = res.x.copy()
            except Exception: pass

    ml_err = float(np.max(np.abs(ml_best_x - tc[:9]))) if ml_best_x is not None else 999
    comb_err = float(np.max(np.abs(comb_best_x - tc[:9]))) if comb_best_x is not None else 999
    ml_ok = ml_err < 1e-3
    comb_ok = comb_err < 1e-3

    if ml_ok: ml_only_perfect += 1
    if comb_ok: combined_perfect += 1
    if ml_ok or comb_ok: either_perfect += 1

    tag = ""
    if comb_ok and not ml_ok: tag = " ***COMBINED SAVED***"
    elif ml_ok and not comb_ok: tag = " ***ML ONLY***"

    print(f"  {ci+1:2d}: ML={'P' if ml_ok else 'F'}(e={ml_err:.1e}) "
          f"COMB={'P' if comb_ok else 'F'}(e={comb_err:.1e}){tag}", flush=True)

print(f"\n{'='*80}")
print(f"  ML only:       {ml_only_perfect}/30 PERFECT")
print(f"  Combined:      {combined_perfect}/30 PERFECT")
print(f"  Either (union):{either_perfect}/30")
print(f"{'='*80}")
