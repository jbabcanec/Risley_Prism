#!/usr/bin/env python3
"""
Multi-scale (graduated non-convexity) for 9-D recovery.

Smooth both pattern and forward model output with decreasing kernel width.
Smoothed pattern has fewer wiggles → wider basins → easier initialization.

Scales: σ = [8, 4, 2, 0] (Gaussian kernel in sample units)
At each scale, TRF from previous solution.
"""
import sys, os, time
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
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


def smooth_pattern(pat, sigma):
    """Apply Gaussian smoothing along time axis."""
    if sigma <= 0:
        return pat
    return np.column_stack([
        gaussian_filter1d(pat[:, 0], sigma),
        gaussian_filter1d(pat[:, 1], sigma),
    ])


def multiscale_solve(init_9d, target, fixed, sigmas=[8, 4, 2, 0]):
    """Multi-scale TRF: smooth → refine → sharp."""
    lo9 = LO[:9].astype(np.float64)
    hi9 = HI[:9].astype(np.float64)
    current = init_9d.copy()

    for sigma in sigmas:
        target_s = smooth_pattern(target, sigma).reshape(-1)

        def residual(x9, _sigma=sigma):
            pat = vec2pat(np.concatenate([x9, fixed]))
            pat_s = smooth_pattern(pat, _sigma).reshape(-1)
            return pat_s - target_s

        try:
            nfev = 3000 if sigma == 0 else 1000
            res = least_squares(
                residual, current, jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=nfev)
            current = res.x.copy()
        except Exception:
            pass

    # Final MSE at original scale
    r = vec2pat(np.concatenate([current, fixed])).reshape(-1) - target.reshape(-1)
    return current, float(np.mean(r**2))


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
print("ML vs ML+MULTISCALE — 9-D (30 cases)")
print(f"{'='*80}\n")

ml_p = 0; ms_p = 0
t0 = time.time()

for ci, tc in enumerate(cases):
    pat = vec2pat(tc)
    tf = pat.reshape(-1)
    pf, pi_info = extract_speeds_and_peaks(pat)
    pk = _build_peak_feats_single(pat, pf, pi_info)
    fixed = tc[9:].copy()
    lo9 = LO[:9].astype(np.float64); hi9 = HI[:9].astype(np.float64)

    def make_res(fix):
        def r(x9): return vec2pat(np.concatenate([x9, fix])).reshape(-1) - tf
        return r
    res_fn = make_res(fixed)

    ml_b = 1e30; ml_x = None
    ms_b = 1e30; ms_x = None

    for bits in range(8):
        signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
        speeds = signs * np.sort(pf)[::-1].astype(np.float64)
        ml = ml_init(ang, rem, speeds.astype(np.float32), pk, pat)

        # (A) ML → direct TRF
        try:
            res = least_squares(res_fn, ml[:9], jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            mse = float(np.mean(res.fun**2))
            if mse < ml_b: ml_b = mse; ml_x = res.x.copy()
        except Exception: pass

        # (B) ML → multiscale TRF
        x, mse = multiscale_solve(ml[:9], pat, fixed, sigmas=[10, 5, 2, 0])
        if mse < ms_b: ms_b = mse; ms_x = x.copy()

    ml_err = float(np.max(np.abs(ml_x - tc[:9]))) if ml_x is not None else 999
    ms_err = float(np.max(np.abs(ms_x - tc[:9]))) if ms_x is not None else 999
    ml_ok = ml_err < 1e-3; ms_ok = ms_err < 1e-3

    if ml_ok: ml_p += 1
    if ms_ok: ms_p += 1

    tag = ""
    if ms_ok and not ml_ok: tag = " ***MS SAVED***"

    print(f"  {ci+1:2d}: ML={'P' if ml_ok else 'F'} MS={'P' if ms_ok else 'F'}{tag}  "
          f"[{time.time()-t0:.0f}s]", flush=True)

print(f"\n{'='*80}")
print(f"  ML only:      {ml_p}/30")
print(f"  ML+Multiscale:{ms_p}/30")
print(f"  Time: {time.time()-t0:.0f}s ({(time.time()-t0)/30:.1f}s/case)")
print(f"{'='*80}")
