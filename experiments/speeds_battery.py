#!/usr/bin/env python3
"""Signed-speed extraction battery (30 cases, seed 2026) vs the FFT top-3
baseline. Run from repo root: python experiments/speeds_battery.py"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from risley_lattice import (DT, T_PTS, vec2pat, battery_cases, case_stats,
                            extract_speeds)

FREQS = np.fft.rfftfreq(T_PTS, d=DT)


def fft_top3(pattern):
    fx = np.fft.rfft(pattern[:, 0]); fy = np.fft.rfft(pattern[:, 1])
    pw = np.abs(fx) + np.abs(fy); pw[0] = 0.0
    out = []
    for _ in range(3):
        i = int(np.argmax(pw))
        out.append(float(FREQS[i]))
        pw[max(1, i - 2):i + 3] = 0.0
    return np.sort(out)[::-1]


if __name__ == '__main__':
    cases = battery_cases()
    print(f"\n{'='*104}")
    print("SPEED EXTRACTION BATTERY: lattice VarPro (signed) vs FFT top-3 "
          "(magnitudes only)")
    print(f"{'='*104}")
    n_lat = n_fft = n_top3 = 0
    errs = []
    t0 = time.time()
    for ci, tc in enumerate(cases):
        pat = vec2pat(tc)
        sp = tc[:3]
        cyc, sep = case_stats(sp)
        fft_ok = np.max(np.abs(fft_top3(pat) -
                               np.sort(np.abs(sp))[::-1])) < 0.05
        N, info = extract_speeds(pat, DT)
        err = float(np.max(np.abs(N - sp))) if N is not None else 999.0
        ok = err < 0.02
        ok3 = ok or any(len(a) == 3 and np.max(np.abs(a - sp)) < 0.02
                        for a in info.get('alts', [])[:2])
        n_lat += ok; n_fft += fft_ok; n_top3 += ok3
        if ok:
            errs.append(err)
        n_s = " ".join(f"{v:+.3f}" for v in N) if N is not None else "rank<3"
        print(f"{ci+1:>3} cyc {cyc:>5.1f} sep {sep:>6.3f}  est [{n_s}]  "
              f"err {err:8.1e}  {'FFTok' if fft_ok else 'FFT--'} "
              f"{'OK' if ok else '--'}", flush=True)
    dt_all = time.time() - t0
    print(f"\n  lattice (signed): {n_lat}/30  (top-3 bases: {n_top3}/30)"
          f"    FFT top-3 magnitudes: {n_fft}/30")
    if errs:
        print(f"  err among successes: median {np.median(errs):.1e} "
              f"max {np.max(errs):.1e}")
    print(f"  {dt_all:.1f}s total")
