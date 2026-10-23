#!/usr/bin/env python3
"""Regression check for the sign-sweep rung added to solve18.
(1) seed-2026 30-case battery: with the rung ON, every case that solved
    with it OFF must give the identical answer (the rung only runs after
    every other rung has failed).
(2) the case where the prototype accepted a ~3e-3 answer (G1 214731, T=80)
    must now be solved correctly by the full solve18.
  python3 regress_solve18.py part --k 0..3 ; python3 regress_solve18.py report
"""
import os, sys, json, subprocess, argparse
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(os.path.expanduser('~/work'), 'regress')


def run_one(job):
    import numpy as np
    from risley_lattice import vec2pat, battery_cases, case_at, FS, solve18
    import risley_lattice.solve as S
    kind, i, T, flag = job
    S.SIGN_SWEEP = flag
    tc = battery_cases()[i] if kind == 'battery' else case_at(i, 424242)
    pat = vec2pat(tc, int(round(T * FS)), T)
    x, mse, how, info = solve18(pat)
    err = float(np.max(np.abs(x - tc))) if x is not None else 1e9
    return dict(kind=kind, i=i, T=T, sweep=flag, mse=float(mse), err=err,
                how=how, x=None if x is None else x.tolist())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd')
    ap.add_argument('--k', type=int, default=0); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = [('battery', i, 10.0, f) for i in range(30) for f in (False, True)]
    jobs += [('g1', 214731, 80.0, True)]
    if a.cmd == 'part':
        done = set()
        fp = os.path.join(OUT, f'r{a.k}.jsonl')
        if os.path.exists(fp):
            done = {(r['kind'], r['i'], r['sweep']) for r in map(json.loads, open(fp))}
        with open(fp, 'a') as f:
            for n, j in enumerate(jobs):
                if n % 4 == a.k and (j[0], j[1], j[3]) not in done:
                    f.write(json.dumps(run_one(j)) + '\n'); f.flush()
    else:
        import glob, numpy as np
        rows = [json.loads(l) for fp in glob.glob(os.path.join(OUT, 'r*.jsonl')) for l in open(fp)]
        bat = {(r['i'], r['sweep']): r for r in rows if r['kind'] == 'battery'}
        same = changed = newly = 0
        for i in range(30):
            if (i, False) not in bat or (i, True) not in bat:
                continue
            off, on = bat[(i, False)], bat[(i, True)]
            if off['mse'] < 1e-12:
                same += off['x'] == on['x']
                changed += off['x'] != on['x']
            elif on['mse'] < 1e-12:
                newly += 1
        solved_off = sum(1 for i in range(30) if (i, False) in bat and bat[(i, False)]['err'] < 1e-3)
        solved_on = sum(1 for i in range(30) if (i, True) in bat and bat[(i, True)]['err'] < 1e-3)
        g1 = [r for r in rows if r['kind'] == 'g1']
        out = dict(battery_pairs=sum(1 for i in range(30) if (i, True) in bat and (i, False) in bat),
                   identical_on_previously_solved=same, changed_on_previously_solved=changed,
                   newly_accepted=newly, solved_off=solved_off, solved_on=solved_on,
                   g1_214731=[{k: r[k] for k in ('err', 'mse', 'how')} for r in g1])
        json.dump(out, open(os.path.join(HERE, 'out', 'regress_solve18.json'), 'w'), indent=1)
        print(json.dumps(out, indent=1))
