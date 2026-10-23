#!/usr/bin/env python3
"""
Test the FFT + ML + TRF pipeline at 9-D, 12-D, and 14-D.
Fix the remaining parameters at their true values.
30 random cases each.
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
        if abs(v[j]) < 0.15:
            v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))

# 9-D:  speeds(3) + angles(6)
# 12-D: speeds(3) + angles(6) + glass(3)
# 14-D: speeds(3) + angles(6) + glass(3) + geometry(2)
dims = {
    9:  list(range(9)),
    12: list(range(12)),
    14: list(range(14)),
}

for ndim, free_idx in dims.items():
    fixed_idx = [j for j in range(N_PAR) if j not in free_idx]
    lo_free = np.array([LO[j] for j in free_idx])
    hi_free = np.array([HI[j] for j in free_idx])

    perfect = 0
    times = []
    max_errs = []

    for i in range(30):
        tc = cases[i]
        pat = vec2pat(tc)
        target = pat.reshape(-1)
        pf, pi_info = extract_speeds_and_peaks(pat)
        pk = _build_peak_feats_single(pat, pf, pi_info)

        fixed_vals = np.array([tc[j] for j in fixed_idx])

        def make_residual(fv):
            def residual(x_free):
                theta = np.zeros(N_PAR)
                for k, idx in enumerate(free_idx):
                    theta[idx] = x_free[k]
                for k, idx in enumerate(fixed_idx):
                    theta[idx] = fv[k]
                return vec2pat(theta).reshape(-1) - target
            return residual

        res_fn = make_residual(fixed_vals)

        best_mse = 1e30
        best_x = None
        t0 = time.time()

        for bits in range(8):
            signs = np.array([(1.0 if (bits >> j) & 1 == 0 else -1.0)
                              for j in range(P)], np.float64)
            speeds = signs * np.sort(pf)[::-1].astype(np.float64)
            ml = ml_init(ang, rem, speeds.astype(np.float32), pk, pat)
            x0 = np.array([ml[j] for j in free_idx])

            try:
                res = least_squares(
                    res_fn, x0, jac='2-point',
                    bounds=(lo_free, hi_free), method='trf',
                    ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=1500)
                mse = float(np.mean(res.fun ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_x = res.x.copy()
            except Exception:
                pass

        dt = time.time() - t0

        if best_x is not None:
            true_free = np.array([tc[j] for j in free_idx])
            err = float(np.max(np.abs(best_x - true_free)))
        else:
            err = 999.0

        tag = "P" if err < 1e-3 else "F"
        if tag == "P":
            perfect += 1
        times.append(dt)
        max_errs.append(err)

        sys.stdout.write(".")
        sys.stdout.flush()

    print(flush=True)
    rate = perfect / 30 * 100
    med_time = np.median(times)
    med_err = np.median([e for e in max_errs if e < 100])
    print(f"{ndim:2d}-D: {perfect}/30 PERFECT ({rate:.0f}%)  "
          f"median_time={med_time:.1f}s  median_err={med_err:.2e}", flush=True)
