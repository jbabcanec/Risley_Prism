#!/usr/bin/env python3
"""
Given TRUE speeds (and glass/geo/beam fixed to truth), how many of the 30 cases
does TRF recover from (a) a trivial zero-angle init, vs (b) the joint alpha_x grid?

This isolates the ANGLE sub-problem from the SPEED sub-problem. If (a) already
solves most cases, then speeds -- not the angle basin -- are the real bottleneck.
"""
import sys, os, time
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from ml_staged_solver import N_PAR, LO, HI, RG, canon, P, DiffFwd, DEVICE
from solve_preconditioned import vec2pat

_LO9 = LO[:9].astype(np.float64); _HI9 = HI[:9].astype(np.float64)
_axg = np.linspace(float(LO[3]), float(HI[3]), 9)
AX_GRID = np.stack(np.meshgrid(_axg, _axg, _axg, indexing='ij'), axis=-1).reshape(-1, 3)
_FWD = DiffFwd().to(DEVICE)


def trf(init9, fixed9, tf):
    def res(x9):
        return vec2pat(np.concatenate([x9, fixed9])).reshape(-1) - tf
    r = least_squares(res, init9, jac='2-point', bounds=(_LO9, _HI9), method='trf',
                      ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=4000)
    return r.x, float(np.mean(r.fun ** 2))


def grid_angles(sp, fixed9, pat, tf, topk=16):
    """Joint alpha_x grid (alpha_y=0) with TRUE speeds -> screen -> TRF."""
    target_t = torch.tensor(pat, dtype=torch.float32, device=DEVICE)
    n = AX_GRID.shape[0]
    cand = np.zeros((n, N_PAR))
    cand[:, :3] = sp; cand[:, 3:6] = AX_GRID; cand[:, 9:] = fixed9
    with torch.no_grad():
        p = _FWD(torch.tensor(cand, dtype=torch.float32, device=DEVICE),
                 high_precision=False)
        mses = ((p - target_t) ** 2).mean(dim=(1, 2)).cpu().numpy()
    best = (1e30, None)
    for idx in np.argsort(mses)[:topk]:
        x, m = trf(cand[idx, :9].copy(), fixed9, tf)
        if m < best[0]:
            best = (m, x)
        if m < 1e-18:
            break
    return best[1], best[0]


rng = np.random.default_rng(2026)
cases = []
for _ in range(30):
    v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))

z_ok = g_ok = either = 0
t0 = time.time()
print(f"{'case':>4} {'min_cyc':>7} {'sep':>6}   zero-init   grid-init")
for ci, tc in enumerate(cases):
    pat = vec2pat(tc); tf = pat.reshape(-1); fixed9 = tc[9:].copy()
    sp = tc[:3].copy()
    sep = min(abs(abs(sp[i]) - abs(sp[j])) for i in range(3) for j in range(i+1, 3))
    cyc = float(min(abs(sp)) * 10.0)

    xz, mz = trf(np.concatenate([sp, np.zeros(6)]), fixed9, tf)
    ez = float(np.max(np.abs(xz - tc[:9])))
    xg, mg = grid_angles(sp, fixed9, pat, tf)
    eg = float(np.max(np.abs(xg - tc[:9])))
    zo, go = ez < 1e-3, eg < 1e-3
    z_ok += zo; g_ok += go; either += (zo or go)
    print(f"{ci+1:>4} {cyc:>7.1f} {sep:>6.3f}   "
          f"{'P' if zo else 'F'}(e={ez:.0e})   {'P' if go else 'F'}(e={eg:.0e})"
          f"{'   <<< grid only' if go and not zo else ''}", flush=True)

print(f"\n  TRUE speeds, zero-angle init : {z_ok}/30")
print(f"  TRUE speeds, alpha_x grid    : {g_ok}/30")
print(f"  Either                       : {either}/30   ({time.time()-t0:.0f}s)")
