#!/usr/bin/env python3
"""Debug the v3 beam-search lattice VarPro on specific battery cases."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

from ml_staged_solver import N_PAR, LO, HI, RG, canon, vec2pat, T_PTS, T_OBS
from spectral_speeds import (matrix_pencil, lattice_fit, model_eval, kset,
                             line_candidates, fundamentals, deglitch_mask,
                             interp_masked)

DT = T_OBS / T_PTS

rng = np.random.default_rng(2026)
cases = []
for _ in range(30):
    v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))

for CASE in (21, 0):        # case 22 and case 1 (0-based)
    tc = cases[CASE]
    pat = vec2pat(tc)
    sp = tc[:3]
    print(f"\n################ CASE {CASE+1}  true speeds {sp}")

    t = np.arange(len(pat)) * DT
    mask = deglitch_mask(pat)
    z = pat[:, 0] + 1j * pat[:, 1]
    zc = interp_masked(z, t, mask)
    zc0 = zc - zc[mask].mean()

    # 1. sanity: fit at TRUE generators
    g, K, c, res = lattice_fit(z, t, mask, list(sp))
    print(f"fit from TRUE gens:  g={np.round(g,4)}  res={res:.2e}")
    fnd = fundamentals(K, g, c)
    if fnd is not None:
        print(f"   fundamentals: {np.round(fnd[0],4)}  rows={[r.astype(int) for r in fnd[1]]}")

    # 2. beam trace
    branches = [([], None, None, 1.0)]
    for stage in range(3):
        new = []
        for gens, Kb, cb, _res in branches:
            if Kb is None:
                resid_clean = zc0
                cur_lines = None
            else:
                resid_clean = zc - model_eval(t, np.array(gens), Kb, cb)
                cur_lines = Kb @ np.array(gens)
            cands = line_candidates(resid_clean, DT, cur_lines)
            print(f"stage {stage}: branch gens={np.round(np.array(gens),3)} "
                  f"-> candidates {np.round(np.array(cands),3)}")
            for fnew in cands:
                g2, K2, c2, res2 = lattice_fit(z, t, mask, list(gens) + [fnew])
                print(f"      seed {fnew:+.3f} -> gens={np.round(g2,3)} res={res2:.2e}")
                new.append((list(g2), K2, c2, res2))
        new.sort(key=lambda b: b[3])
        seen, kept = set(), []
        for b in new:
            key = tuple(sorted(np.round(b[0], 3)))
            if key in seen: continue
            seen.add(key); kept.append(b)
            if len(kept) == 2: break
        branches = kept
        print(f"   kept: {[ (np.round(np.array(b[0]),3).tolist(), f'{b[3]:.1e}') for b in branches]}")
