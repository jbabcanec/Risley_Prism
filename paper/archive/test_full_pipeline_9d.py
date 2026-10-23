#!/usr/bin/env python3
"""
9-D test using the FULL solve() pipeline from solve_preconditioned.py.

The previous 9-D tests used a stripped-down version (8 sign combos, single triple).
This test calls the actual solver and just evaluates the 9-D subproblem accuracy.

If the full solver (multi-triple, Adam screening, basin-hopping) succeeds at 18-D,
it should succeed at 9-D too (since we fix the remaining 9 params at true values,
making the problem strictly easier).
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
)
from solve_preconditioned import vec2pat, ml_init, solve

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
print("FULL 18-D SOLVER → check 9-D accuracy (30 cases)")
print(f"{'='*80}\n")

perfect_9d = 0
perfect_18d = 0
t0 = time.time()

for ci, tc in enumerate(cases):
    pat = vec2pat(tc)

    t_case = time.time()
    solved, mse = solve(ang, rem, pat, verbose=False)
    dt = time.time() - t_case

    err_9d = float(np.max(np.abs(solved[:9] - tc[:9])))
    err_18d = float(np.max(np.abs(solved - tc)))
    ok_9d = err_9d < 1e-3
    ok_18d = err_18d < 1e-3

    if ok_9d: perfect_9d += 1
    if ok_18d: perfect_18d += 1

    print(f"  {ci+1:2d}: 9D={'P' if ok_9d else 'F'}(e={err_9d:.1e}) "
          f"18D={'P' if ok_18d else 'F'}(e={err_18d:.1e}) "
          f"MSE={mse:.1e} {dt:.0f}s", flush=True)

print(f"\n{'='*80}")
print(f"  9-D perfect:  {perfect_9d}/30")
print(f"  18-D perfect: {perfect_18d}/30")
print(f"  Time: {time.time()-t0:.0f}s ({(time.time()-t0)/30:.0f}s/case)")
print(f"{'='*80}")
