#!/usr/bin/env python3
"""Full 18-D battery, nothing assumed known. Success = max|d| < 1e-3 over
all 18 parameters. Run from repo root: python experiments/solve18_battery.py"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from risley_lattice import vec2pat, battery_cases, case_stats, solve18

if __name__ == '__main__':
    cases = battery_cases()
    print(f"\n{'='*96}")
    print("SOLVE18 BATTERY: ALL 18 parameters from one pattern. "
          "NOTHING assumed known.")
    print(f"{'='*96}")
    n_ok = 0
    t0 = time.time()
    for ci, tc in enumerate(cases):
        pat = vec2pat(tc)
        cyc, sep = case_stats(tc[:3])
        tg = time.time()
        x18, mse, how, info = solve18(pat)
        tg = time.time() - tg
        err = float(np.max(np.abs(x18 - tc))) if x18 is not None else 999.0
        ok = err < 1e-3
        n_ok += ok
        print(f"{ci+1:>3} cyc {cyc:>5.1f} sep {sep:>6.3f}  "
              f"{'PERFECT' if ok else 'fail':<8} err={err:8.1e} "
              f"mse={mse:8.1e} ({how}) {tg:>4.0f}s", flush=True)
    dt_all = time.time() - t0
    print(f"\n  FULL 18-D: {n_ok}/30 PERFECT")
    print(f"  {dt_all:.0f}s total ({dt_all/30:.0f}s/case)")
