#!/usr/bin/env python3
"""Large-N adaptive-observation battery.

Protocol per case: solve the full 18-D problem at T = 10 s; on failure,
follow the certificate's prescription up the ladder T = 20, 40, 80 s.
Records, per case: the T at which recovery succeeded (or exhaustion), the
error/MSE, the ladder rung, timing, and the T=10 certificate reasons for
initially-failing cases. Writes JSONL so shards can run in parallel.

Run (shard):  python experiments/adaptive_battery.py --start 0 --count 100
Aggregate:    python experiments/adaptive_battery.py --report
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys, time, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from risley_lattice import (FS, DT, vec2pat, battery_cases, case_stats,
                            solve18, extract_speeds, spectral_certificate)

SEED = 7777
N_TOTAL = 1000
T_LADDER = (10.0, 20.0, 40.0, 80.0)
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def run_shard(start, count):
    os.makedirs(OUTDIR, exist_ok=True)
    cases = battery_cases(n=N_TOTAL, seed=SEED)
    path = os.path.join(OUTDIR, f'adaptive_{start:04d}_{count}.jsonl')
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['case'])
                except Exception:
                    pass
    with open(path, 'a') as f:
        for ci in range(start, min(start + count, N_TOTAL)):
            if ci in done:
                continue
            tc = cases[ci]
            cyc, sep = case_stats(tc[:3])
            rec = {'case': ci, 'cyc': cyc, 'sep': sep}
            t0 = time.time()
            for T in T_LADDER:
                pat = vec2pat(tc, int(round(T * FS)), T)
                x18, mse, how, info = solve18(pat)
                err = float(np.max(np.abs(x18 - tc))) if x18 is not None \
                    else 999.0
                if T == T_LADDER[0]:
                    rec['err10'] = err
                    rec['how10'] = how
                    if err >= 1e-3:
                        try:
                            N_est = info.get('N')
                            if N_est is None:
                                N_est = info.get('gens_partial')
                            if N_est is None:
                                N_est, info = extract_speeds(pat, DT)
                                if N_est is None and info:
                                    N_est = info.get('gens_partial')
                            reasons, _sg = spectral_certificate(
                                pat, N_est, info) if N_est is not None \
                                else (['no-speeds'], None)
                        except Exception as ex:
                            reasons = [f'cert-error: {ex}']
                        rec['cert10'] = reasons
                if err < 1e-3:
                    rec.update(T_solved=T, err=err, mse=mse, how=how)
                    break
            else:
                rec.update(T_solved=None, err=err, mse=mse, how=how)
            rec['secs'] = round(time.time() - t0, 1)
            f.write(json.dumps(rec) + '\n')
            f.flush()
            print(f"case {ci}: T={rec.get('T_solved')} err={rec.get('err'):.1e}"
                  f" [{rec['secs']}s]", flush=True)


def report():
    recs = []
    for p in sorted(glob.glob(os.path.join(OUTDIR, 'adaptive_*.jsonl'))):
        with open(p) as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    n = len(recs)
    if not n:
        print('no records yet')
        return
    by_T = {}
    for r in recs:
        by_T[r.get('T_solved')] = by_T.get(r.get('T_solved'), 0) + 1
    n10 = by_T.get(10.0, 0)
    solved = sum(v for k, v in by_T.items() if k is not None)
    print(f"cases: {n}")
    print(f"solved at T=10 s:          {n10}  ({100*n10/n:.1f}%)")
    for T in (20.0, 40.0, 80.0):
        if by_T.get(T):
            print(f"solved at T={T:<4.0f}         +{by_T[T]}")
    print(f"solved with adaptive T:    {solved}  ({100*solved/n:.2f}%)")
    print(f"unsolved at T<=80 s:       {n - solved}")
    for r in recs:
        if r.get('T_solved') is None:
            head = (r.get('cert10') or ['?'])[0]
            print(f"   case {r['case']}: cyc {r['cyc']:.1f} sep {r['sep']:.3f}"
                  f" err {r.get('err', 0):.1e} | {head}")
    errs = [r['err'] for r in recs if r.get('T_solved') is not None]
    secs = [r['secs'] for r in recs]
    print(f"err among solved: median {np.median(errs):.1e} "
          f"max {np.max(errs):.1e}")
    print(f"time/case: median {np.median(secs):.1f}s  "
          f"p90 {np.percentile(secs, 90):.0f}s")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=N_TOTAL)
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.report:
        report()
    else:
        run_shard(a.start, a.count)
