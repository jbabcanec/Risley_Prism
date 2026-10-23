#!/usr/bin/env python3
"""Noise battery: the 18-D pipeline + certificate coverage under additive
white Gaussian noise. The certificate sigma is estimated from the residual,
so bounds inflate with the noise floor and must keep covering the truth.
Run from repo root: python experiments/noise.py"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from risley_lattice import vec2pat, battery_cases, solve18, certify_success

CASES = battery_cases()
CLEAN = [0, 1, 2, 4, 5, 6, 7, 8, 11, 12]     # solvable at zero noise

if __name__ == '__main__':
    nz = np.random.default_rng(77)
    print(f"\n{'='*88}")
    print("NOISE BATTERY: 18-D pipeline + certificate coverage vs SNR")
    print(f"{'='*88}")
    print(f"{'SNR dB':>7} {'recovered':>10} {'cert covered':>13} "
          f"{'med max-bound':>14} {'med max-err':>12}")
    for snr_db in (np.inf, 60.0, 50.0, 40.0, 30.0):
        n_rec = n_cov = 0
        bmax, emax = [], []
        for ci in CLEAN:
            tc = CASES[ci]
            pat = vec2pat(tc)
            sig = 0.0
            if np.isfinite(snr_db):
                prms = np.std(pat - pat.mean(0))
                sig = prms / (10 ** (snr_db / 20.0))
                pat = pat + nz.normal(0.0, sig, pat.shape)
            x18, mse, how, info = solve18(pat)
            if x18 is None:
                continue
            errs = np.abs(x18 - tc)
            if mse < max(1e-12, 3.0 * sig ** 2):
                smask = info.get('mask', np.ones(len(pat), bool))
                rmask = np.repeat(smask, 2)
                bounds = certify_success(x18, pat, rmask)
                n_rec += 1
                n_cov += bool(np.all(errs <= np.maximum(bounds, 1e-14)))
                bmax.append(bounds.max())
                emax.append(errs.max())
        b = np.median(bmax) if bmax else np.nan
        e = np.median(emax) if emax else np.nan
        lbl = 'inf' if not np.isfinite(snr_db) else f'{snr_db:.0f}'
        print(f"{lbl:>7} {n_rec:>7}/10 {n_cov:>10}/{n_rec:<2} "
              f"{b:>14.2e} {e:>12.2e}", flush=True)
    print(f"{'='*88}")
