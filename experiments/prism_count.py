#!/usr/bin/env python3
"""Prism-count generality: the spectral stage is P-agnostic. Signed-speed
extraction batteries at P = 2 and P = 4 (20 cases each, seed 2026).
Run from repo root: python experiments/prism_count.py"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from risley_lattice import (DT, FS, pat_P, battery_cases_P, case_stats,
                            extract_speeds)

if __name__ == '__main__':
    print(f"\n{'='*92}")
    print("PRISM-COUNT GENERALITY: signed-speed extraction at P = 2, 4")
    print(f"{'='*92}")
    for P, T in ((2, 10.0), (4, 10.0), (4, 40.0)):
        cases = battery_cases_P(P, n=20)
        n_ok = 0
        errs = []
        t0 = time.time()
        for ci, (sp, ax, ay, ng, geom9) in enumerate(cases):
            pat = pat_P(sp, ax, ay, ng, geom9,
                        n_points=int(round(T * FS)), time_limit=T)
            cyc, sep = case_stats(sp)
            N, info = extract_speeds(pat, DT, n_gen=P)
            err = float(np.max(np.abs(N - sp))) if N is not None else 999.0
            ok = err < 0.02
            n_ok += ok
            if ok:
                errs.append(err)
            n_s = " ".join(f"{v:+.3f}" for v in N) if N is not None \
                else "rank<n"
            print(f"  P={P} T={T:.0f} {ci+1:>3} cyc {cyc:>5.1f} "
                  f"sep {sep:>6.3f}  est [{n_s}]  err {err:8.1e}  "
                  f"{'OK' if ok else '--'}", flush=True)
        dt_all = time.time() - t0
        med = np.median(errs) if errs else np.nan
        print(f"  ---- P={P} T={T:.0f}: {n_ok}/20 exact signed  "
              f"(median err {med:.1e})  {dt_all:.0f}s\n", flush=True)
    print(f"{'='*92}")
