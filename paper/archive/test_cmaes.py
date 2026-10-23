#!/usr/bin/env python3
"""
CMA-ES for 9-D Risley inverse problem.

CMA-ES is the gold standard for non-convex optimization in moderate dimensions.
It adapts a full covariance matrix to the local landscape, handling:
  - Ill-conditioning (κ up to 10^14)
  - Non-separable objectives
  - Multiple local minima (with restarts)

Strategy: for each of 8 speed sign combos, run CMA-ES seeded at ML init
with initial σ₀ adapted per-parameter. Then TRF polish the best.
"""
import sys, os, time
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

import cma
import torch
from ml_staged_solver import (
    AngleNet, RemainNet, ANG_MODEL, REM_MODEL,
    DEVICE, N_PAR, LO, HI, RG, canon,
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
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))

print(f"\n{'='*80}")
print("CMA-ES + TRF — 9-D (speeds + angles, geometry fixed)")
print(f"{'='*80}\n")

ml_perfect = 0
cma_perfect = 0
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

    def obj_9d(x9):
        try:
            theta = np.concatenate([np.array(x9), fixed])
            return float(np.mean((vec2pat(theta).reshape(-1) - target)**2))
        except Exception:
            return 1e6

    ml_best_mse = 1e30; ml_best_x = None
    cma_best_mse = 1e30; cma_best_x = None

    for bits in range(8):
        signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
        speeds = signs * np.sort(pf)[::-1].astype(np.float64)
        ml = ml_init(ang, rem, speeds.astype(np.float32), pk, pat)

        # ML-only TRF
        try:
            res = least_squares(res_fn, ml[:9], jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            mse = float(np.mean(res.fun**2))
            if mse < ml_best_mse: ml_best_mse = mse; ml_best_x = res.x.copy()
        except Exception: pass

        # CMA-ES from ML init
        # Initial sigma: moderate — we know ML gets speeds right but angles wrong
        sigma0 = 3.0  # ~3° search radius for angles
        opts = cma.CMAOptions()
        opts['bounds'] = [lo9.tolist(), hi9.tolist()]
        opts['maxfevals'] = 5000
        opts['verbose'] = -9  # silent
        opts['tolfun'] = 1e-15
        opts['tolx'] = 1e-10
        opts['CMA_stds'] = [0.3, 0.3, 0.3,  # speeds: small search
                            5.0, 5.0, 5.0,  # αₓ: moderate
                            5.0, 5.0, 5.0]  # αᵧ: moderate

        try:
            es = cma.CMAEvolutionStrategy(ml[:9].tolist(), sigma0, opts)
            es.optimize(obj_9d)
            result = es.result
            cma_x = np.array(result.xbest)
            cma_mse = result.fbest

            # TRF polish from CMA-ES solution
            try:
                res = least_squares(res_fn, cma_x, jac='2-point',
                    bounds=(lo9, hi9), method='trf',
                    ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=3000)
                mse = float(np.mean(res.fun**2))
                if mse < cma_best_mse: cma_best_mse = mse; cma_best_x = res.x.copy()
            except Exception:
                if cma_mse < cma_best_mse: cma_best_mse = cma_mse; cma_best_x = cma_x.copy()
        except Exception as e:
            pass

    ml_err = float(np.max(np.abs(ml_best_x - tc[:9]))) if ml_best_x is not None else 999
    cma_err = float(np.max(np.abs(cma_best_x - tc[:9]))) if cma_best_x is not None else 999
    ml_ok = ml_err < 1e-3
    cma_ok = cma_err < 1e-3

    if ml_ok: ml_perfect += 1
    if cma_ok: cma_perfect += 1

    tag = ""
    if cma_ok and not ml_ok: tag = " ***CMA SAVED***"
    elif ml_ok and not cma_ok: tag = " ***CMA MISSED***"

    dt = time.time() - t_total
    print(f"  {ci+1:2d}: ML={'P' if ml_ok else 'F'} CMA={'P' if cma_ok else 'F'}{tag}  "
          f"[{dt:.0f}s elapsed]", flush=True)

elapsed = time.time() - t_total
print(f"\n{'='*80}")
print(f"  ML only:   {ml_perfect}/30 PERFECT")
print(f"  CMA-ES:    {cma_perfect}/30 PERFECT")
print(f"  Time: {elapsed:.0f}s ({elapsed/30:.1f}s/case)")
print(f"{'='*80}")
