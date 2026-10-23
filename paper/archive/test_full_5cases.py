#!/usr/bin/env python3
"""Quick 5-case test with full solver, verbose to see progress."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from ml_staged_solver import (
    AngleNet, RemainNet, ANG_MODEL, REM_MODEL,
    DEVICE, N_PAR, LO, HI, RG, canon,
)
from solve_preconditioned import vec2pat, solve

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

# Pick 5 cases: 2 that single-triple solves + 3 that fail (with wrong FFT)
test_idx = [0, 1, 3, 5, 7]  # cases 1,2 (P), 4,6,8 (F, spd_match=False)

for ci in test_idx:
    tc = cases[ci]
    pat = vec2pat(tc)
    print(f"\n=== Case {ci+1} ===", flush=True)
    t0 = time.time()
    solved, mse = solve(ang, rem, pat, verbose=True)
    dt = time.time() - t0

    err_9d = float(np.max(np.abs(solved[:9] - tc[:9])))
    err_18d = float(np.max(np.abs(solved - tc)))
    print(f"  >> 9D_err={err_9d:.1e}  18D_err={err_18d:.1e}  "
          f"MSE={mse:.1e}  {dt:.0f}s  "
          f"{'PERFECT' if err_18d < 1e-3 else 'FAIL'}", flush=True)
