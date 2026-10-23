#!/usr/bin/env python3
"""9-D battery (glass/geometry/beam known). Success = max|d| < 1e-3.
Run from repo root: python experiments/solve9_battery.py"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from risley_lattice import vec2pat, battery_cases, case_stats, solve9

if __name__ == '__main__':
    cases = battery_cases()
    print(f"\n{'='*96}")
    print("SOLVE9 BATTERY: spectral speeds + phase/amp angles + TRF "
          "(no grids, no sign combos)")
    print(f"{'='*96}")
    n_ok = 0
    t0 = time.time()
    for ci, tc in enumerate(cases):
        pat = vec2pat(tc)
        cyc, sep = case_stats(tc[:3])
        tg = time.time()
        x9, mse, how = solve9(pat, tc[9:].copy())
        tg = time.time() - tg
        err = float(np.max(np.abs(x9 - tc[:9]))) if x9 is not None else 999.0
        ok = err < 1e-3
        n_ok += ok
        print(f"{ci+1:>3} cyc {cyc:>5.1f} sep {sep:>6.3f}  "
              f"{'PERFECT' if ok else 'fail':<8} err={err:8.1e} "
              f"mse={mse:8.1e} ({how}) {tg:>4.0f}s", flush=True)
    dt_all = time.time() - t0
    print(f"\n  SPECTRAL 9-D: {n_ok}/30   "
          f"[reference: alpha_x-grid solver 16/30, ML init 9/30]")
    print(f"  {dt_all:.0f}s total")
