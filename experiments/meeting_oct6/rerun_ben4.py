#!/usr/bin/env python3
"""Re-run the 4 unsolved Ben-regime configurations (ben cases 16, 62, 87, 90)
with the sign-sweep rung ON and no ladder cut-off.

  python3 rerun_ben4.py --case 16 --T 10      # one trial
  python3 rerun_ben4.py --all                 # whole ladder for all four
                                              # (no time limit; run in Terminal)
  python3 rerun_ben4.py --report
Results append to out/rerun_ben4.jsonl.
"""
import os
os.environ['RISLEY_SIGN_SWEEP'] = '1'
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
_argv = sys.argv; sys.argv = sys.argv[:1]
import local_campaign as L
sys.argv = _argv
from sweep import make_pattern
from risley_lattice import solve18
from risley_lattice.model import NAMES

CASES = [16, 62, 87, 90]
LADDER = [10., 20., 40., 80., 160., 320.]
OUTF = os.path.join(HERE, 'out', 'rerun_ben4.jsonl')


def trial(i, T):
    tc = L.gen('ben', i)
    t0 = time.time()
    x, mse, how, info = solve18(make_pattern(tc, T))
    err = np.abs(x - tc) if x is not None else None
    row = dict(case=i, T=T, sign_sweep=True, mse=float(mse), how=how,
               err=float(err.max()) if err is not None else None,
               worst_param=NAMES[int(err.argmax())] if err is not None else None,
               secs=round(time.time() - t0, 1),
               truth_ax=np.round(tc[3:6], 3).tolist(),
               truth_N=np.round(tc[:3], 5).tolist())
    with open(OUTF, 'a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row))
    return row


def report():
    rows = [json.loads(l) for l in open(OUTF)] if os.path.exists(OUTF) else []
    for i in CASES:
        rs = sorted([r for r in rows if r['case'] == i], key=lambda r: r['T'])
        ok = [r for r in rs if r['err'] is not None and r['err'] < 1e-3]
        print(i, 'SOLVED at T=%g' % ok[0]['T'] if ok else 'not solved',
              [(r['T'], r['how'], '%.1e' % r['mse'], None if r['err'] is None else '%.1e' % r['err'],
                r['worst_param'], r['secs']) for r in rs])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', type=int); ap.add_argument('--T', type=float)
    ap.add_argument('--all', action='store_true'); ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.report:
        report()
    elif a.all:
        done = {}
        if os.path.exists(OUTF):
            for l in open(OUTF):
                r = json.loads(l); done[(r['case'], r['T'])] = r
        for i in CASES:
            for T in LADDER:
                r = done.get((i, T)) or trial(i, T)
                if r['err'] is not None and r['err'] < 1e-3:
                    break
        report()
    else:
        trial(a.case, a.T)
