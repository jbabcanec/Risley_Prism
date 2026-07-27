#!/usr/bin/env python3
"""Test success rate at 9-D and 12-D subproblems."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from scipy.optimize import least_squares
from ml_staged_solver import (AngleNet, RemainNet, ANG_MODEL, REM_MODEL,
    DEVICE, N_PAR, LO, HI, RG, canon,
    extract_speeds_and_peaks, _build_peak_feats_single, P)
from solve_preconditioned import vec2pat, ml_init

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

for ndim in [9, 12, 15, 18]:
    perfect = 0
    for i in range(30):
        tc = cases[i]
        pat = vec2pat(tc)
        target = pat.reshape(-1)
        pf, pi = extract_speeds_and_peaks(pat)
        pk = _build_peak_feats_single(pat, pf, pi)

        best_mse = 1e30
        best_x = None
        for bits in range(8):
            signs = np.array([(1.0 if (bits >> j) & 1 == 0 else -1.0)
                              for j in range(P)], np.float64)
            speeds = signs * np.sort(pf)[::-1].astype(np.float64)
            ml = ml_init(ang, rem, speeds.astype(np.float32), pk, pat)

            x0 = ml[:ndim].copy()
            fixed = tc[ndim:].copy()

            def make_res(fix):
                def residual(x):
                    theta = np.concatenate([x, fix])
                    return vec2pat(theta).reshape(-1) - target
                return residual

            try:
                res = least_squares(
                    make_res(fixed), x0, jac='3-point',
                    bounds=(LO[:ndim], HI[:ndim]), method='trf',
                    ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=3000)
                mse = float(np.mean(res.fun ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_x = res.x.copy()
            except Exception:
                pass

        if best_x is not None:
            err = float(np.max(np.abs(best_x - tc[:ndim])))
        else:
            err = 999
        if err < 1e-3:
            perfect += 1

    print(f'{ndim:2d}-D: {perfect}/30 PERFECT', flush=True)
