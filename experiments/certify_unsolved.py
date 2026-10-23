#!/usr/bin/env python3
"""Certify the fresh ensemble's unsolved cases in one pass with the final
certifier; the paper's zero-silent-failures claim cites this output.
Run from repo root: python experiments/certify_unsolved.py"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from risley_lattice import (DT, vec2pat, battery_cases, extract_speeds,
                            spectral_certificate)

cases = battery_cases(n=1000, seed=7777)
recs = []
for p in sorted(glob.glob('experiments/results/adaptive_*.jsonl')):
    for line in open(p):
        try:
            recs.append(json.loads(line))
        except Exception:
            pass
uns = sorted(r['case'] for r in recs if r.get('T_solved') is None)
print(f'unsolved cases ({len(uns)}): {uns}\n')

for ci in uns:
    tc = cases[ci]
    pat = vec2pat(tc)
    N, info = extract_speeds(pat, DT)
    N_est = info.get('N') if N is not None else info.get('gens_partial')
    if N_est is None or (hasattr(N_est, '__len__') and len(N_est) == 0):
        print(f'case {ci}: NO GENERATORS (front-end failure)')
        continue
    reasons, sg = spectral_certificate(pat, N_est, info)
    print(f"case {ci} (true ax = {np.round(tc[3:6], 1)}):")
    for r in reasons[:4]:
        print(f'   -> {r}')
    print(flush=True)
