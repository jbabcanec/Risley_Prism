#!/usr/bin/env python3
"""
ML + 180° flip augmentation — v2.

For each speed sign combo:
  1. TRF from ML init (as baseline)
  2. Screen 8 flip combos with forward eval → pick best
  3. TRF from best flip
  4. Keep the better of ML-TRF and Flip-TRF

Total: 16 TRF runs per case (same budget as before + 8 extra).
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
    T = len(pattern)
    t = np.arange(T) * (T_OBS / T)
    n = len(freqs)
    X = np.zeros((T, 2*n + 1))
    for i, f in enumerate(freqs):
        X[:, 2*i] = np.cos(2*np.pi*f*t)
        X[:, 2*i+1] = np.sin(2*np.pi*f*t)
    X[:, -1] = 1.0
    bx, _, _, _ = np.linalg.lstsq(X, pattern[:, 0], rcond=None)
    return [np.arctan2(-bx[2*i+1], bx[2*i]) for i in range(n)]


def make_flipped_inits(ml_vec, phases, speeds):
    """Generate 8 flip variants: for each prism subset, flip (αₓ→-αₓ, αᵧ→spec+180°)."""
    inits = []
    for fb in range(8):
        v = ml_vec.copy()
        for j in range(P):
            ph = phases[j]
            ay_spec = -np.degrees(ph) if speeds[j] < 0 else np.degrees(ph)
            if (fb >> j) & 1:
                v[6+j] = ay_spec + 180.0
                if v[6+j] > 180: v[6+j] -= 360
                v[3+j] = -abs(ml_vec[3+j])
            else:
                v[6+j] = ay_spec
                v[3+j] = abs(ml_vec[3+j])
        inits.append(np.clip(v, LO[:len(v)], HI[:len(v)]))
    return inits


print("Loading models...", flush=True)
ang = AngleNet(); rem = RemainNet()
ang.load_state_dict(torch.load(ANG_MODEL, map_location=DEVICE, weights_only=True))
rem.load_state_dict(torch.load(REM_MODEL, map_location=DEVICE, weights_only=True))
ang.to(DEVICE); rem.to(DEVICE)

rng = np.random.default_rng(2026)
cases = []
for i in range(30):
    v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))

print(f"\n{'='*80}")
print("ML vs ML+FLIPS (augmented, not replaced) — 9-D TRF")
print(f"{'='*80}\n")

ml_perfect = 0
aug_perfect = 0
t_total = time.time()

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
    aug_best_mse = 1e30; aug_best_x = None

    for bits in range(8):
        signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
        speeds = signs * np.sort(pf)[::-1].astype(np.float64)

        # ML init → TRF
        ml = ml_init(ang, rem, speeds.astype(np.float32), pk, pat)
        try:
            res = least_squares(res_fn, ml[:9], jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            mse = float(np.mean(res.fun**2))
            if mse < ml_best_mse: ml_best_mse = mse; ml_best_x = res.x.copy()
            if mse < aug_best_mse: aug_best_mse = mse; aug_best_x = res.x.copy()
        except Exception: pass

        # Flip variants: screen → TRF best
        phases = harmonic_phases(pat, np.abs(speeds))
        flips = make_flipped_inits(ml, phases, speeds)

        # Screen all 8 with forward eval
        best_screen_mse = 1e30
        best_flip_init = None
        for fv in flips:
            try:
                r = vec2pat(np.concatenate([fv[:9], fixed])).reshape(-1) - target
                sm = float(np.mean(r**2))
                if sm < best_screen_mse:
                    best_screen_mse = sm
                    best_flip_init = fv[:9].copy()
            except Exception: pass

        # TRF from best flip
        if best_flip_init is not None:
            try:
                res = least_squares(res_fn, best_flip_init, jac='2-point',
                    bounds=(lo9, hi9), method='trf',
                    ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
                mse = float(np.mean(res.fun**2))
                if mse < aug_best_mse: aug_best_mse = mse; aug_best_x = res.x.copy()
            except Exception: pass

    ml_err = float(np.max(np.abs(ml_best_x - tc[:9]))) if ml_best_x is not None else 999
    aug_err = float(np.max(np.abs(aug_best_x - tc[:9]))) if aug_best_x is not None else 999
    ml_ok = ml_err < 1e-3
    aug_ok = aug_err < 1e-3

    if ml_ok: ml_perfect += 1
    if aug_ok: aug_perfect += 1

    tag = ""
    if aug_ok and not ml_ok: tag = " ***FLIP SAVED***"
    print(f"  {ci+1:2d}: ML={'P' if ml_ok else 'F'} AUG={'P' if aug_ok else 'F'}{tag}", flush=True)

elapsed = time.time() - t_total
print(f"\n{'='*80}")
print(f"  ML only:     {ml_perfect}/30 PERFECT")
print(f"  ML+flips:    {aug_perfect}/30 PERFECT")
print(f"  Time: {elapsed:.0f}s ({elapsed/30:.1f}s/case)")
print(f"{'='*80}")
