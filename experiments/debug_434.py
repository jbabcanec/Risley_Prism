#!/usr/bin/env python3
"""Full forensics on ensemble case 434 -- the single failure in 617 with no
visible pathology. Instruments every spectral decision and probes the TRF
basin directly. Run from repo root: python experiments/debug_434.py"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from risley_lattice import (DT, FS, N_PAR, LO, HI, vec2pat, battery_cases,
                            extract_speeds, lattice_fit, matrix_pencil)
from risley_lattice.spectral import (deglitch_mask, interp_masked, clean_lines,
                                     select_bases)
from risley_lattice.angles import angle_init, proj_phases, nominal_rest
from risley_lattice.solve import trf, solve18

CASE = 434
tc = battery_cases(n=1000, seed=7777)[CASE]
sp = tc[:3]
print(f"true N  = {np.round(sp, 4)}")
print(f"true ax = {np.round(tc[3:6], 2)}   ay = {np.round(tc[6:9], 2)}")
print(f"true ng = {np.round(tc[9:12], 3)}  dW/gap = {np.round(tc[12:14], 2)}"
      f"  beam = {np.round(tc[14:], 2)}")

pat = vec2pat(tc)
t = np.arange(len(pat)) * DT
mask = deglitch_mask(pat)
z = pat[:, 0] + 1j * pat[:, 1]
zc = interp_masked(z, t, mask)

# 1. raw pencil on the full signal
f, c = matrix_pencil(zc - zc.mean(), DT)
o = np.argsort(-np.abs(c))[:14]
print("\nraw pencil lines (top 14):")
for j in o:
    near = min(abs(f[j] - s) for s in sp)
    print(f"   f={f[j]:+8.4f}  |c|={abs(c[j]):8.3f}"
          f"{'   <== near-true' if near < 8e-3 else ''}")

# 2. CLEAN growth
lines, amps, res_c = clean_lines(z, zc, t, mask, DT)
print(f"\nCLEAN lines: {np.round(lines, 4)}")
print(f"CLEAN amps : {np.round(np.abs(amps), 2)}   res={res_c:.1e}")
for s in sp:
    d = np.min(np.abs(lines - s)) if len(lines) else 9e9
    print(f"   true {s:+.4f}: nearest line off by {d*1000:.1f} mHz")

# 3. candidate bases + the true triple's standing
cands = select_bases(lines, amps, FS)
print("\ncandidate bases (coverage):")
for cv, g in cands:
    print(f"   cov={cv:.4f}  {np.round(g, 4)}")
g_t, K_t, c_t, res_t = lattice_fit(z, t, mask, list(sp), B=3)
print(f"\nlattice fit seeded AT TRUTH: g={np.round(g_t, 4)} res={res_t:.2e}"
      f"  drift={np.max(np.abs(g_t - sp))*1000:.1f} mHz")

# 4. what the pipeline returns + sign-test amplitudes
N, info = extract_speeds(pat, DT)
print(f"\npipeline est: {np.round(N, 4) if N is not None else None}"
      f"  resid={info.get('resid'):.1e}")
if N is not None:
    g5, K5, c5, r5 = lattice_fit(z, t, mask, list(N), B=3, sharp=True)
    for i in range(3):
        rowp = np.zeros(3); rowp[i] = 1
        jp = int(np.argmin(np.abs(K5 - rowp).sum(1)))
        jm = int(np.argmin(np.abs(K5 + rowp).sum(1)))
        print(f"   gen {N[i]:+.4f}: |c(+e)|={abs(c5[jp]):.3f} "
              f"|c(-e)|={abs(c5[jm]):.3f}"
              f"{'  SIGN SUSPECT' if abs(c5[jm]) > abs(c5[jp]) else ''}")

# 5. TRF basin probes (full 18-D)
target = pat.reshape(-1).astype(np.float64)
rmask = np.repeat(mask, 2)
rest0 = nominal_rest(pat)


def probe(N_est, tag):
    ph, am = proj_phases(pat, N_est)
    ax, ay, amb = angle_init(N_est, ph, am, rest0)
    x0 = np.concatenate([N_est, ax, ay, rest0])
    bx, bm = trf(x0, target, rmask, LO.copy(), HI.copy(), max_nfev=3000)
    err = np.max(np.abs(bx - tc))
    print(f"   {tag:<34} mse={bm:8.1e}  err={err:8.1e}")


print("\nTRF basin probes:")
probe(sp.copy(), "truth speeds")
if N is not None:
    probe(N.copy(), "pipeline est (as returned)")
    N2 = N.copy(); N2[np.argmin(np.abs(N2))] *= -1
    probe(N2, "pipeline est, weakest sign flipped")
