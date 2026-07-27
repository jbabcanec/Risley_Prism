#!/usr/bin/env python3
"""Line-level autopsy of the N=617 ensemble's algorithmic-residual cases:
the 11 where certificates deemed recovery feasible but the solver failed.
For each: ground-truth structure (relation gaps, fundamental amplitudes,
TIR), then what the spectral stage actually extracted at T=10 and T=80,
with truth-in-lines flags to localize the failing stage.

Run from repo root: python experiments/autopsy_residual.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from itertools import product as iproduct
import numpy as np
from risley_lattice import (DT, FS, vec2pat, battery_cases, extract_speeds)
from risley_lattice.spectral import deglitch_mask

NOSPEED = [81, 130, 223, 317]
LADDER = [102, 161, 268, 313, 408, 434, 529]
cases = battery_cases(n=1000, seed=7777)


def truth_structure(tc):
    sp = tc[:3]
    K = np.array([k for k in iproduct(range(-4, 5), repeat=3)
                  if 0 < sum(abs(x) for x in k) <= 4], float)
    gmin, kmin, jmin = 1e9, None, None
    for j in range(3):
        ej = np.zeros(3); ej[j] = 1
        for k in K:
            if np.array_equal(k, ej) or np.array_equal(k, -ej):
                continue
            gap = abs(k @ sp - sp[j])
            if gap < gmin:
                gmin, kmin, jmin = gap, k.astype(int), j
    pat = vec2pat(tc)
    z = pat[:, 0] + 1j * pat[:, 1]
    z = z - z.mean()
    tg = np.arange(len(z)) * DT
    amps = [abs(np.vdot(np.exp(2j * np.pi * f * tg), z)) / len(z)
            for f in sp]
    mask = deglitch_mask(pat)
    step = float(np.max(np.abs(np.diff(pat, axis=0))))
    return gmin, kmin, jmin, amps, int((~mask).sum()), step


def try_extract(tc, T):
    pat = vec2pat(tc, int(round(T * FS)), T)
    N, info = extract_speeds(pat, DT)
    sp = tc[:3]
    if N is None:
        lines = info.get('lines', np.array([]))
        hit = [bool(len(lines)) and bool(np.min(np.abs(lines - s)) < 5e-3)
               for s in sp]
        return (f'FAIL({info.get("fail")}) lines='
                f'[{" ".join(f"{x:+.3f}" for x in lines)}] '
                f'truth-in-lines={hit} resid={info.get("res_clean", -1):.0e}')
    err = np.max(np.abs(N - sp))
    lines = info.get('lines', np.array([]))
    hit = [bool(len(lines)) and bool(np.min(np.abs(lines - s)) < 5e-3)
           for s in sp]
    return (f'est=[{" ".join(f"{x:+.3f}" for x in N)}] err={err:.1e} '
            f'truth-in-lines={hit}')


for label, group in (('NO-SPEEDS', NOSPEED), ('LADDER-MISS', LADDER)):
    print(f'\n{"="*94}\n{label}\n{"="*94}')
    for ci in group:
        tc = cases[ci]
        sp = tc[:3]
        gmin, kmin, jmin, amps, nmask, step = truth_structure(tc)
        print(f"\ncase {ci}: N=[{' '.join(f'{x:+.3f}' for x in sp)}]  "
              f"ax=[{' '.join(f'{x:+.1f}' for x in tc[3:6])}]")
        print(f"   truth: min relation gap {gmin*1000:.2f} mHz "
              f"(N{jmin+1} ~ {tuple(kmin)});  fund amps "
              f"[{' '.join(f'{a:.2f}' for a in amps)}];  "
              f"TIR-masked {nmask}; maxstep {step:.0f}")
        for T in (10.0, 80.0):
            print(f"   T={T:>2.0f}: {try_extract(tc, T)}", flush=True)
