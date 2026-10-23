#!/usr/bin/env python3
"""
Diagnose what distinguishes success from failure in the 9-D inverse problem.

For each of 30 cases, compute:
  1. ML init quality (MSE of ML prediction vs true params)
  2. Speed separation (min |Nᵢ - Nⱼ|)
  3. Angle magnitudes
  4. Whether correct speeds are in top-3 FFT peaks
  5. ML angle errors per parameter
  6. Basin width estimate (from Jacobian at true params)
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
    T_PTS, T_OBS, FREQS,
)
from solve_preconditioned import vec2pat, ml_init

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
print("FAILURE DIAGNOSIS — 9-D (30 cases)")
print(f"{'='*80}\n")

results = []
for ci, tc in enumerate(cases):
    pat = vec2pat(tc)
    tf = pat.reshape(-1)
    pf, pi_info = extract_speeds_and_peaks(pat)
    pk = _build_peak_feats_single(pat, pf, pi_info)
    fixed = tc[9:].copy()
    lo9 = LO[:9].astype(np.float64); hi9 = HI[:9].astype(np.float64)

    true_speeds = tc[:3]
    true_ax = tc[3:6]
    true_ay = tc[6:9]

    # Check if true speeds match FFT top-3
    fft_freqs = np.sort(pf)[::-1]
    true_freq_mags = np.sort(np.abs(true_speeds))[::-1]
    speed_match = all(
        min(abs(fft_freqs - tf_mag)) < 0.1
        for tf_mag in true_freq_mags
    )

    # Speed separation
    speed_sep = min(
        abs(abs(true_speeds[i]) - abs(true_speeds[j]))
        for i in range(P) for j in range(i+1, P)
    )

    # Max angle magnitude
    max_ax = float(np.max(np.abs(true_ax)))

    # Best ML init quality (across 8 sign combos)
    def make_res(fix):
        def r(x9): return vec2pat(np.concatenate([x9, fix])).reshape(-1) - tf
        return r
    res_fn = make_res(fixed)

    best_mse = 1e30; best_x = None; best_ml = None
    best_ml_ax_err = 999; best_ml_ay_err = 999

    for bits in range(8):
        signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
        speeds = signs * np.sort(pf)[::-1].astype(np.float64)
        ml = ml_init(ang, rem, speeds.astype(np.float32), pk, pat)

        # ML init angle errors (for this sign combo)
        ax_err = float(np.max(np.abs(ml[3:6] - true_ax)))
        ay_err = float(np.max(np.abs(ml[6:9] - true_ay)))

        try:
            res = least_squares(res_fn, ml[:9], jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            mse = float(np.mean(res.fun**2))
            if mse < best_mse:
                best_mse = mse; best_x = res.x.copy(); best_ml = ml.copy()
                best_ml_ax_err = ax_err; best_ml_ay_err = ay_err
        except Exception: pass

    final_err = float(np.max(np.abs(best_x - tc[:9]))) if best_x is not None else 999
    solved = final_err < 1e-3

    r = {
        'case': ci+1, 'solved': solved,
        'speed_match': speed_match, 'speed_sep': round(speed_sep, 3),
        'max_ax': round(max_ax, 1),
        'ml_ax_err': round(best_ml_ax_err, 2),
        'ml_ay_err': round(best_ml_ay_err, 2),
        'final_err': final_err, 'final_mse': best_mse,
    }
    results.append(r)

    tag = 'P' if solved else 'F'
    print(f"  {ci+1:2d} [{tag}]: spd_match={speed_match} sep={speed_sep:.3f} "
          f"|αₓ|_max={max_ax:.1f}° ML_αₓ_err={best_ml_ax_err:.1f}° "
          f"ML_αᵧ_err={best_ml_ay_err:.1f}°", flush=True)

# Summary statistics
solved_cases = [r for r in results if r['solved']]
failed_cases = [r for r in results if not r['solved']]

print(f"\n{'='*80}")
print(f"  SOLVED: {len(solved_cases)}/30")
print(f"\n  --- Solved cases ---")
print(f"  Speed match: {sum(1 for r in solved_cases if r['speed_match'])}/{len(solved_cases)}")
print(f"  Speed sep:   median={np.median([r['speed_sep'] for r in solved_cases]):.3f}")
print(f"  ML αₓ err:   median={np.median([r['ml_ax_err'] for r in solved_cases]):.1f}°")
print(f"  ML αᵧ err:   median={np.median([r['ml_ay_err'] for r in solved_cases]):.1f}°")
print(f"  |αₓ| max:    median={np.median([r['max_ax'] for r in solved_cases]):.1f}°")

print(f"\n  --- Failed cases ---")
print(f"  Speed match: {sum(1 for r in failed_cases if r['speed_match'])}/{len(failed_cases)}")
print(f"  Speed sep:   median={np.median([r['speed_sep'] for r in failed_cases]):.3f}")
print(f"  ML αₓ err:   median={np.median([r['ml_ax_err'] for r in failed_cases]):.1f}°")
print(f"  ML αᵧ err:   median={np.median([r['ml_ay_err'] for r in failed_cases]):.1f}°")
print(f"  |αₓ| max:    median={np.median([r['max_ax'] for r in failed_cases]):.1f}°")
print(f"{'='*80}")
