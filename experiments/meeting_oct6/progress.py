import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv = ['x']
import local_campaign as L
N = dict(ben=160, ben066=60, box10=150, calib=150)
for e, n in N.items():
    s = L.summarize(e, n)
    k = sum(c['T_solved'] is not None for c in s)
    print(f'{e:7s} finished {len(s):4d}/{n}  solved {k}')
import sign_sweep_local as S
ss = os.path.expanduser('~/work/ss')
jobs = json.load(open(os.path.join(ss, 'ss_jobs.json')))['jobs']
u = S.load_units(ss)
print('signsweep finished', sum(S.done(u.get((j['case'], j['T']), [])) for j in jobs), '/', len(jobs))

import nonunique; nonunique.progress()
