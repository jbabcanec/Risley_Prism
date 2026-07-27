#!/usr/bin/env python3
"""Trace every pipeline decision for cases 4, 23, 27 (0-based 3, 22, 26)."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

from ml_staged_solver import N_PAR, LO, HI, RG, canon, vec2pat, T_PTS, T_OBS
import spectral_speeds as ss

DT = T_OBS / T_PTS

rng = np.random.default_rng(2026)
cases = []
for _ in range(30):
    v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))

for CASE in (3, 22, 26):
    tc = cases[CASE]
    pat = vec2pat(tc)
    sp = tc[:3]
    print(f"\n################ CASE {CASE+1}  true {np.round(sp, 4)}")

    t = np.arange(len(pat)) * DT
    mask = ss.deglitch_mask(pat)
    z = pat[:, 0] + 1j * pat[:, 1]
    zc = ss.interp_masked(z, t, mask)
    fs = 1.0 / DT

    lines, amps, res_c = ss.clean_lines(z, zc, t, mask, DT)
    print(f"lines {np.round(lines, 4)}")
    print(f"amps  {np.round(np.abs(amps), 3)}")

    cands = ss.select_bases(lines, amps, fs)
    cov_max = max(cv for cv, _ in cands)
    print(f"cov_max={cov_max:.4f}")
    fits = []
    for cv, gseed in cands:
        g, K, c, res = ss.lattice_fit(z, t, mask, gseed, B=3)
        score = res * (1.0 + 2.0 * ss.parsimony(K, c))
        moved = float(np.max(np.abs(g - gseed)))
        gated = (cv >= cov_max - 0.05) and moved < 0.06
        fits.append((not gated, score, cv, list(g), K, c, res))
        print(f"  cand {np.round(gseed,3)} cov={cv:.4f} -> g={np.round(g,4)} "
              f"res={res:.2e} moved={moved:.3f} score={score:.3e} "
              f"{'GATED-IN' if gated else 'out'}")
    fits.sort(key=lambda f: (f[0], f[1]))
    _, _, cov_best, gens, K, c, res = fits[0]
    print(f"chosen: {np.round(np.array(gens),4)}  cov_best={cov_best:.4f} res={res:.2e}")

    # what fundamentals extracts from the chosen fit
    fnd = ss.fundamentals(K, np.array(gens), c)
    if fnd is not None:
        print(f"fundamentals: {np.round(fnd[0],4)}  rows={[r.astype(int) for r in fnd[1]]}"
              f"  amps={np.round(np.abs(fnd[2]),2)}")
    else:
        print("fundamentals: None")

    N, info = ss.extract_speeds(pat, DT)
    print(f"pipeline: {np.round(N,4) if N is not None else None}  "
          f"resid={info.get('resid'):.2e}")
