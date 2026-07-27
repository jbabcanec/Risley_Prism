#!/usr/bin/env python3
"""
test_noise_spectral.py -- The spectral 18-D pipeline + certificates under
measurement noise. The certificate story must survive noise: sigma is
estimated from the residual, so the bounds inflate with the noise floor and
must KEEP COVERING the true error (that is the bulletproof claim).

For each SNR: 10 battery cases, additive white Gaussian noise on the pattern.
Reports: recovery rate (err < tol_snr), certificate coverage (true error
within the 3-sigma bound), median bound (how tolerance degrades with SNR).

Run: python paper/test_noise_spectral.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

from ml_staged_solver import N_PAR, LO, HI, RG, canon, vec2pat
from solve18_spectral import solve18
from certify import certify_success

rng = np.random.default_rng(2026)
CASES = []
for _ in range(30):
    v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    CASES.append(canon(v))

# use the first 10 cases that are clean at zero noise (known-solvable)
CLEAN = [0, 1, 2, 4, 5, 6, 7, 8, 11, 12]     # battery indices (0-based)

if __name__ == '__main__':
    nz = np.random.default_rng(77)
    print(f"\n{'='*88}")
    print("NOISE BATTERY: spectral 18-D pipeline + certificate coverage vs SNR")
    print(f"{'='*88}")
    print(f"{'SNR dB':>7} {'recovered':>10} {'cert covered':>13} "
          f"{'med max-bound':>14} {'med max-err':>12}")
    for snr_db in (np.inf, 60.0, 50.0, 40.0, 30.0):
        n_rec = n_cov = 0
        bmax, emax = [], []
        for ci in CLEAN:
            tc = CASES[ci]
            pat = vec2pat(tc)
            if np.isfinite(snr_db):
                prms = np.std(pat - pat.mean(0))
                sig = prms / (10 ** (snr_db / 20.0))
                pat = pat + nz.normal(0.0, sig, pat.shape)
            x18, mse, how, info = solve18(pat)
            if x18 is None:
                continue
            errs = np.abs(x18 - tc)
            # noise floor: success once the pattern is explained to the floor
            noise_mse = 0.0 if not np.isfinite(snr_db) else sig ** 2
            if mse < max(1e-12, 3.0 * noise_mse):
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
