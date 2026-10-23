#!/usr/bin/env python3
"""Summarize the sign-sweep rung prototype and draw figure B7."""
import os, sys, json, glob, pickle
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sign_sweep_units as U

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
rows, jobs = U.summarize()
amb = pickle.load(open('/tmp/angle_amb.pkl', 'rb'))

# population sizes behind each sample
pure = lambda r: r['spd_ok'] and 'sign-flip' in r['cls'] and 'angle-off' not in r['cls']
other = lambda r: r['spd_ok'] and 'angle-off' in r['cls']
pop = {'G1': sum(map(pure, amb['last'])), 'G2': sum(map(pure, amb['rescued'])),
       'G3': sum(map(other, amb['last']))}

print('=== rescue rate by group ===')
summ = {}
for g in ('G1', 'G2', 'G3'):
    R = [r for r in rows if r['group'] == g]
    n = len(R); res = sum(r['rescued'] for r in R)
    cor = sum(r['rescued'] and r['correct'] for r in R)
    wrong = sum(r['rescued'] and not r['correct'] for r in R)
    ntrf = [r['n_trf'] for r in R if r['rescued']]
    secs = [r['secs'] for r in R if r['rescued']]
    secs_f = [r['secs'] for r in R if not r['rescued']]
    summ[g] = dict(n=n, pop=pop[g], res=res, cor=cor, wrong=wrong,
                   ntrf=np.median(ntrf) if ntrf else np.nan,
                   secs=np.median(secs) if secs else np.nan,
                   secs_f=np.median(secs_f) if secs_f else np.nan)
    print(g, summ[g])


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


for g in summ:
    lo, hi = wilson(summ[g]['res'], summ[g]['n'])
    summ[g]['ci'] = (lo, hi)
    print(g, f"rate {summ[g]['res']/summ[g]['n']:.3f}  95% CI [{lo:.3f},{hi:.3f}]")

print('\n=== by group x T ===')
byT = defaultdict(lambda: [0, 0])
for r in rows:
    byT[(r['group'], r['T'])][1] += 1
    byT[(r['group'], r['T'])][0] += r['rescued']
for k in sorted(byT):
    print(k, byT[k])

print('\n=== winning candidate ===')
print(Counter(r['tag'] for r in rows if r['rescued']).most_common())
# was the winning flip the weakest prism?  tag keep:<idx> where idx is prism id
weak_first = 0; nwin = 0
for r in rows:
    if r['rescued'] and r['tag'].startswith('keep:') and len(r['tag']) == 6:
        nwin += 1
        if int(r['tag'][-1]) == int(np.argmin(r['axt'])):
            weak_first += 1
print(f'single-prism wins on the truly weakest prism: {weak_first}/{nwin}')

# rescue vs true min |ax|
axm = np.array([min(r['axt']) for r in rows]); ok = np.array([r['rescued'] for r in rows])
edges = [0, 0.5, 1, 2, 5, 18]
ax_rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (axm >= lo) & (axm < hi)
    ax_rows.append((lo, hi, int(m.sum()), int(ok[m].sum())))
    print(f'min|ax| [{lo},{hi}): n={m.sum()}  rescued={ok[m].sum()}')

# ---- G2: escalation time the rung would have saved
g2 = {(r['atlas'], r['case'], r['T']): r for r in rows if r['group'] == 'G2' and r['rescued']}
need = defaultdict(set)
for (a, c, T) in g2:
    need[a].add(c)
saved, cost = [], []
for a, cases in need.items():
    for fp in sorted(glob.glob(f'{RES}/{a}/*.jsonl')):
        for line in open(fp):
            rec = json.loads(line)
            if rec['case'] not in cases:
                continue
            for (aa, cc, T), r in g2.items():
                if aa == a and cc == rec['case']:
                    later = sum(t['secs'] for t in rec['trail'] if t['T'] > T)
                    saved.append(later); cost.append(r['secs'])
saved = np.array(saved); cost = np.array(cost)
print(f'\nG2 rescued: {len(saved)}  median later-trial time avoided {np.median(saved):.1f} s, '
      f'median rung cost {np.median(cost):.1f} s, net saving median {np.median(saved-cost):.1f} s')

out = dict(summ=summ, byT={f'{k[0]}@{int(k[1])}': v for k, v in byT.items()},
           tags=Counter(r['tag'] for r in rows if r['rescued']).most_common(),
           weak_first=(weak_first, nwin), ax_rows=ax_rows,
           g2_saved=float(np.median(saved)) if len(saved) else None,
           g2_cost=float(np.median(cost)) if len(cost) else None,
           g2_net=float(np.median(saved - cost)) if len(saved) else None,
           g2_total_saved=float(saved.sum()), g2_total_cost=float(cost.sum()),
           rate_rows=amb['rate_rows'], flip_win=amb['flip_win'])
json.dump(out, open('/tmp/b7_summary.json', 'w'), indent=1, default=float)

# ---- figure
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
fig.patch.set_facecolor('#F8F9FA')
ax = axs[0]
rr = amb['rate_rows']
lab = [f'{lo:g}–{hi:g}' for lo, hi, n, f in rr]
ax.bar(lab, [100 * f / n for lo, hi, n, f in rr], color='#2E5A8E')
for i, (lo, hi, n, f) in enumerate(rr):
    ax.text(i, 100 * f / n + 0.2, f'{100*f/n:.1f}%', ha='center', fontsize=8)
ax.set_xlabel('true smallest wedge tilt |ax| (deg)'); ax.set_ylabel('% unsolved at T = 80 s')
ax.set_title('B7a: failure rate vs smallest tilt', fontweight='bold')
ax.grid(axis='y', alpha=.3)

ax = axs[1]
names = {'G1': 'G1 unsolved,\npure sign flip', 'G2': 'G2 early attempt,\npure sign flip',
         'G3': 'G3 unsolved,\nother angle error'}
xs = np.arange(3); vals = []; errs = [[], []]
for g in ('G1', 'G2', 'G3'):
    s = summ[g]; p = s['res'] / s['n']; lo, hi = s['ci']
    vals.append(100 * p); errs[0].append(100 * (p - lo)); errs[1].append(100 * (hi - p))
ax.bar(xs, vals, yerr=errs, capsize=5, color=['#C73E1D', '#2E86AB', '#888888'])
for i, g in enumerate(('G1', 'G2', 'G3')):
    ax.text(i, vals[i] + errs[1][i] + 1.5, f"{summ[g]['res']}/{summ[g]['n']}", ha='center', fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels([names[g] for g in ('G1', 'G2', 'G3')], fontsize=8)
ax.set_ylabel('% rescued by sign-sweep rung'); ax.set_ylim(0, 100)
ax.set_title('B7b: sign-sweep rescue rate (95% CI)', fontweight='bold')
ax.grid(axis='y', alpha=.3)

ax = axs[2]
lab = [f'{lo:g}–{hi:g}' for lo, hi, n, k in ax_rows]
ax.bar(lab, [100 * k / max(n, 1) for lo, hi, n, k in ax_rows], color='#44BBA4')
for i, (lo, hi, n, k) in enumerate(ax_rows):
    ax.text(i, 100 * k / max(n, 1) + 1.5, f'{k}/{n}', ha='center', fontsize=8)
ax.set_xlabel('true smallest wedge tilt |ax| (deg)'); ax.set_ylabel('% rescued (all groups)')
ax.set_ylim(0, 100); ax.set_title('B7c: rescue rate vs smallest tilt', fontweight='bold')
ax.grid(axis='y', alpha=.3)
plt.tight_layout()
plt.savefig('/tmp/risley_B7_angle_ambiguity.png', dpi=150, bbox_inches='tight')
print('figure saved')
