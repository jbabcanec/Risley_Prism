#!/usr/bin/env python3
"""Debug v4 selection/polish on cases 8, 15, 17 (0-based 7, 14, 16)."""
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

for CASE in (22, 11, 12):
    tc = cases[CASE]
    pat = vec2pat(tc)
    sp = tc[:3]
    print(f"\n################ CASE {CASE+1}  true {np.round(sp,4)}")

    t = np.arange(len(pat)) * DT
    mask = ss.deglitch_mask(pat)
    z = pat[:, 0] + 1j * pat[:, 1]
    zc = ss.interp_masked(z, t, mask)
    fs = 1.0 / DT

    lines, amps, res_c = ss.clean_lines(z, zc, t, mask, DT)
    print(f"lines: {np.round(lines,4)}")
    print(f"amps : {np.round(np.abs(amps),3)}   res_clean={res_c:.2e}")

    w = np.abs(amps) / (np.abs(amps).sum() + 1e-30)
    from itertools import combinations
    idx = [i for i in range(len(lines)) if abs(lines[i]) <= ss.SPEED_MAX]
    scored = []
    for sub in combinations(idx, 3):
        g = lines[list(sub)]
        scored.append((ss.coverage_score(lines, w, g, fs), sub))
    scored.sort(key=lambda s: -s[0])
    print("top-5 coverage bases:")
    for sc, sub in scored[:5]:
        print(f"   {np.round(lines[list(sub)],4)}  cov={sc:.4f}")

    # fit top-5 and score
    for sc, sub in scored[:5]:
        g, K, c, res = ss.lattice_fit(z, t, mask, lines[list(sub)], B=3)
        p = ss.parsimony(K, c)
        print(f"   seed {np.round(lines[list(sub)],3)} -> g={np.round(g,4)} "
              f"res={res:.2e} pars={p:.3f} score={res*(1+2*p):.2e}")

    # true-triple reference
    g, K, c, res = ss.lattice_fit(z, t, mask, sp, B=3)
    print(f"   TRUE seed -> g={np.round(g,4)} res={res:.2e} "
          f"pars={ss.parsimony(K,c):.3f}")

    N, info = ss.extract_speeds(pat, DT)
    print(f"pipeline result: {np.round(N,4) if N is not None else None}  "
          f"resid={info.get('resid'):.2e}")
