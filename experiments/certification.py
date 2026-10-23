#!/usr/bin/env python3
"""Certification battery: every case -> tight per-parameter bounds (success,
validated against ground truth) or a quantitative impossibility statement
(failure). Run from repo root: python experiments/certification.py"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from risley_lattice import (DT, T_OBS, vec2pat, battery_cases, case_stats,
                            solve18, certify_success, spectral_certificate,
                            extract_speeds, MSE_OK)

if __name__ == '__main__':
    cases = battery_cases()
    print(f"\n{'='*100}")
    print("CERTIFICATION BATTERY: tight bounds (success) or quantitative "
          "impossibility (failure)")
    print(f"{'='*100}")
    n_ok = n_cov = 0
    tight = []
    t0 = time.time()
    for ci, tc in enumerate(cases):
        pat = vec2pat(tc)
        x18, mse, how, info = solve18(pat)
        if x18 is not None and mse < MSE_OK:
            smask = info.get('mask', np.ones(len(pat), bool))
            rmask = np.repeat(smask, 2)
            bounds = certify_success(x18, pat, rmask)
            errs = np.abs(x18 - tc)
            covered = bool(np.all(errs <= np.maximum(bounds, 1e-14)))
            ratio = np.median(bounds / np.maximum(errs, 1e-16))
            n_ok += 1; n_cov += covered
            tight.append(ratio)
            print(f"{ci+1:>3} CERT-OK   max-bound {bounds.max():8.1e}  "
                  f"max-err {errs.max():8.1e}  "
                  f"covered={'Y' if covered else 'N'}  "
                  f"med bound/err {ratio:7.1f}x  ({how})", flush=True)
        else:
            N_est = info.get('N') if info else None
            if N_est is None and info:
                N_est = info.get('gens_partial')
            if N_est is None:
                Ns, inf2 = extract_speeds(pat, DT)
                info = inf2 if inf2 else (info or {})
                N_est = Ns if Ns is not None else \
                    info.get('gens_partial', np.array([1.0, 2.0, 3.0]))
            reasons, sig_g = spectral_certificate(pat, N_est, info)
            cyc, sep = case_stats(tc[:3])
            print(f"{ci+1:>3} CERT-FAIL (true sep {sep:.3f} Hz, "
                  f"min cyc {cyc:.1f})", flush=True)
            for rr in reasons:
                print(f"      -> {rr}")
    dt_all = time.time() - t0
    print(f"\n  CERT-OK: {n_ok}/30   bound coverage: {n_cov}/{n_ok}"
          + (f"   median tightness {np.median(tight):.0f}x" if tight else ""))
    print(f"  {dt_all:.0f}s total")
