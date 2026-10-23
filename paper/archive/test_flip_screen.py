#!/usr/bin/env python3
"""
Fast test: ML init + 180° flip screening.

For each speed sign combo:
  1. ML init → compute residual
  2. Generate 7 flipped versions (αₓ→-αₓ, αᵧ→αᵧ+180° for each prism subset)
  3. Screen all 8 with single forward eval → pick lowest MSE
  4. TRF from the winner

Same 8 TRF runs as ML-only, but each starts from best of {ML, flips}.
"""
import sys, os, time
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
    T_PTS, T_OBS,
)
from solve_preconditioned import vec2pat, ml_init


def harmonic_phases(pattern, freqs):
    """Extract phases at exact frequencies using harmonic LS."""
    T = len(pattern)
    t = np.arange(T) * (T_OBS / T)
    n = len(freqs)
    X = np.zeros((T, 2*n + 1))
    for i, f in enumerate(freqs):
        X[:, 2*i] = np.cos(2*np.pi*f*t)
        X[:, 2*i+1] = np.sin(2*np.pi*f*t)
    X[:, -1] = 1.0
    bx, _, _, _ = np.linalg.lstsq(X, pattern[:, 0], rcond=None)
    phases = []
    for i in range(n):
        phases.append(np.arctan2(-bx[2*i+1], bx[2*i]))
    return phases


def flip_init(ml_vec, spectral_phases, speeds, flip_bits):
    """
    Create flipped version of ML init.
    For each prism where flip_bits has a 1:
      αₓ → -αₓ, αᵧ → spectral_αᵧ + 180°
    For others: αᵧ → spectral_αᵧ (unflipped), keep ML αₓ
    """
    v = ml_vec.copy()
    for j in range(P):
        phase = spectral_phases[j]
        if speeds[j] < 0:
            ay_spec = -np.degrees(phase)
        else:
            ay_spec = np.degrees(phase)

        if (flip_bits >> j) & 1:
            v[6+j] = ay_spec + 180.0
            if v[6+j] > 180: v[6+j] -= 360
            v[3+j] = -abs(ml_vec[3+j])
        else:
            v[6+j] = ay_spec
            v[3+j] = abs(ml_vec[3+j])  # keep ML magnitude, ensure positive

    return np.clip(v, LO[:len(v)], HI[:len(v)])


# Load models
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
print("ML-ONLY vs ML+FLIP-SCREEN — 9-D TRF (30 cases)")
print(f"{'='*80}\n")

ml_perfect = 0
flip_perfect = 0

t_start = time.time()

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

    ml_best_mse = 1e30; ml_best_x = None
    flip_best_mse = 1e30; flip_best_x = None

    for bits in range(8):
        signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
        speeds = signs * np.sort(pf)[::-1].astype(np.float64)

        # ML init
        ml = ml_init(ang, rem, speeds.astype(np.float32), pk, pat)

        # --- ML-only path ---
        try:
            res = least_squares(res_fn, ml[:9], jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            mse = float(np.mean(res.fun**2))
            if mse < ml_best_mse: ml_best_mse = mse; ml_best_x = res.x.copy()
        except Exception: pass

        # --- Flip-screen path ---
        # Get spectral phases
        phases = harmonic_phases(pat, np.abs(speeds))

        # Screen: evaluate all 8 flip options, pick best
        best_screen_mse = 1e30
        best_screen_init = ml[:9].copy()

        for fb in range(8):
            candidate = flip_init(ml, phases, speeds, fb)
            try:
                r = vec2pat(np.concatenate([candidate[:9], fixed])).reshape(-1) - target
                screen_mse = float(np.mean(r**2))
                if screen_mse < best_screen_mse:
                    best_screen_mse = screen_mse
                    best_screen_init = candidate[:9].copy()
            except Exception: pass

        # Also include the raw ML init in screening
        try:
            r = vec2pat(np.concatenate([ml[:9], fixed])).reshape(-1) - target
            ml_screen_mse = float(np.mean(r**2))
            if ml_screen_mse < best_screen_mse:
                best_screen_init = ml[:9].copy()
        except Exception: pass

        # TRF from best screen result
        try:
            res = least_squares(res_fn, best_screen_init, jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            mse = float(np.mean(res.fun**2))
            if mse < flip_best_mse: flip_best_mse = mse; flip_best_x = res.x.copy()
        except Exception: pass

    ml_err = float(np.max(np.abs(ml_best_x - tc[:9]))) if ml_best_x is not None else 999
    flip_err = float(np.max(np.abs(flip_best_x - tc[:9]))) if flip_best_x is not None else 999
    ml_ok = ml_err < 1e-3
    flip_ok = flip_err < 1e-3

    if ml_ok: ml_perfect += 1
    if flip_ok: flip_perfect += 1

    tag = ""
    if flip_ok and not ml_ok: tag = " ***FLIP SAVED***"
    elif ml_ok and not flip_ok: tag = " ***FLIP HURT***"

    sys.stdout.write(f"  {ci+1:2d}: ML={'P' if ml_ok else 'F'} FLIP={'P' if flip_ok else 'F'}{tag}\n")
    sys.stdout.flush()

elapsed = time.time() - t_start
print(f"\n{'='*80}")
print(f"  ML only:       {ml_perfect}/30 PERFECT")
print(f"  ML+flip:       {flip_perfect}/30 PERFECT")
print(f"  Time: {elapsed:.0f}s ({elapsed/30:.1f}s/case)")
print(f"{'='*80}")
