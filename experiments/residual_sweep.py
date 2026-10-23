#!/usr/bin/env python3
"""Re-run the 11 ensemble residual cases through the adaptive protocol
after the overload/sign-test fixes. Run: python experiments/residual_sweep.py"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from risley_lattice import battery_cases, vec2pat, solve18, FS

RESIDUAL = [81, 130, 223, 317, 102, 161, 268, 313, 408, 434, 529]
cases = battery_cases(n=1000, seed=7777)

if __name__ == '__main__':
    n_fixed = 0
    for ci in RESIDUAL:
        tc = cases[ci]
        t0 = time.time()
        out = 'unsolved<=80s'
        for T in (10.0, 20.0, 40.0, 80.0):
            pat = vec2pat(tc, int(round(T * FS)), T)
            x18, mse, how, info = solve18(pat)
            err = np.max(np.abs(x18 - tc)) if x18 is not None else 999.0
            if err < 1e-3:
                out = f'SOLVED T={T:.0f} err={err:.1e} ({how})' + \
                    (' [overload-retry]' if info.get('overload_retry')
                     else '')
                n_fixed += 1
                break
        print(f'case {ci:>3}: {out}  [{time.time()-t0:.0f}s]', flush=True)
    print(f'\nrescued: {n_fixed}/11')
