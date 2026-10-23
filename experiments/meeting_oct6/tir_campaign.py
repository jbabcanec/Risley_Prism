#!/usr/bin/env python3
"""What the TIR condition explains in the 669,574-case campaign, and the
recovery rate of restricted boxes (sub-populations of a uniform box are
themselves uniform samples of the smaller box, so no re-run is needed)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vizstyle import setup, C, SEQ, INK2, wilson, plt
from tir import critical_wedge

camp = os.environ.get('CAMP', os.path.expanduser('~/work/camp/campaign.npz'))
m = np.load(camp)
OUT, FIG = os.path.join(HERE, 'out'), os.path.join(HERE, 'figures')
CERTS = ['relation', 'weak-prism', 'glitch-floor', 'close-pair',
         'rank-deficient', 'basin', 'no-generators']
solved = np.isfinite(m['T_solved'])
s10 = m['T_solved'] == 10
marg = m['tir_margin']
tr = m['truth']
axmax = np.abs(tr[:, 3:6]).max(1)
gl = (m['cert_bits'] >> CERTS.index('glitch-floor')) & 1 == 1
res = {}


def row(name, sel):
    n = int(sel.sum())
    k, k10, g = int(solved[sel].sum()), int(s10[sel].sum()), int(gl[sel].sum())
    p, lo, hi = wilson(k, n)
    p10 = k10 / n
    fails = n - k
    res[name] = dict(n=n, solved=k, rate=float(p), ci=[float(lo), float(hi)],
                     rate_T10=float(p10), unsolved=fails,
                     glitch_share_of_failures=float(g / max(fails, 1)))
    print(f'{name:34s} n={n:7d}  adaptive {100*p:6.2f}% [{100*lo:.2f},'
          f'{100*hi:.2f}]  T=10 {100*p10:6.2f}%  unsolved {fails:6d}  '
          f'glitch-floor in {100*g/max(fails,1):5.1f}% of failures')


print('fraction of the +/-18 deg box with worst-case TIR (margin<0): '
      f'{np.mean(marg < 0)*100:.2f}%')
res['frac_tir_box18'] = float(np.mean(marg < 0))
row('full box (|ax|<=18)', np.ones(len(marg), bool))
row('TIR-prone (margin<0)', marg < 0)
row('TIR-free (margin>0)', marg > 0)
row('TIR-free with 1 deg safety', marg > 1)
for A in (14, 12, 10, 9.296, 8):
    sel = axmax <= A
    row(f'|ax|<={A} box', sel)
    res[f'frac_tir_box{A}'] = float(np.mean(marg[sel] < 0))
    print(f'    TIR-prone fraction inside |ax|<={A}: {100*np.mean(marg[sel]<0):.2f}%')
row('|ax|<=10 and TIR-free', (axmax <= 10) & (marg > 0))

# where do the failures go when TIR is removed?
for name, sel in [('all', np.ones(len(marg), bool)), ('TIR-free', marg > 0)]:
    f = sel & ~solved
    sh = {c: float(np.mean(((m['cert_bits'][f] >> i) & 1) == 1))
          for i, c in enumerate(CERTS)}
    res[f'cert_share_{name}'] = sh
    print(name, 'failure cert shares:',
          {k: f'{100*v:.1f}%' for k, v in sh.items()})

# success vs margin
bins = np.array([-60, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 60])
idx = np.digitize(marg, bins) - 1
rb = []
for b in range(len(bins) - 1):
    sel = idx == b
    n, k = int(sel.sum()), int(solved[sel].sum())
    rb.append((float(bins[b]), float(bins[b + 1]), n, k))
res['rate_by_margin'] = rb
json.dump(res, open(os.path.join(OUT, 'tir_campaign.json'), 'w'), indent=1)

setup()
# Figure T1: the closed-form bound (single prism)
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3))
th = np.linspace(0, 40, 300)
for j, n in enumerate([1.3, 1.5, 1.65, 1.8]):
    ax[0].plot(th, critical_wedge(th, n), color=C[j], label=f'n = {n}')
ax[0].axhline(18, color=INK2, lw=1, ls='--')
ax[0].text(0.5, 18.6, 'current search box 18°', ha='left', color=INK2, fontsize=8)
ax[0].axhline(10, color=INK2, lw=1, ls=':')
ax[0].text(0.5, 10.6, 'soft goal 10°', ha='left', color=INK2, fontsize=8)
ax[0].set_xlabel('incident angle on the prism (deg, air)')
ax[0].set_ylabel('largest wedge angle without TIR (deg)')
ax[0].set_title('Single prism: TIR-free wedge limit', fontsize=9)
ax[0].legend(fontsize=8, loc='upper right')
# 3 identical prisms, beam on axis: max common wedge vs n
from tir import tir_margins
ns = np.linspace(1.3, 1.8, 200)
aa = np.linspace(0, 45, 901)
for j, (P, b) in enumerate([(1, 0), (2, 0), (3, 0), (3, 25)]):
    lim = []
    for n in ns:
        mm = tir_margins(np.tile(aa[:, None], (1, P)), np.full((len(aa), P), n),
                         np.full(len(aa), b)).min(-1)
        lim.append(aa[mm > 0].max() if (mm > 0).any() else 0)
    ax[1].plot(ns, lim, color=C[j], label=f'{P} prism{"s" if P>1 else ""}, beam {b}°')
ax[1].axhline(18, color=INK2, lw=1, ls='--'); ax[1].axhline(10, color=INK2, lw=1, ls=':')
ax[1].set_xlabel('glass index n'); ax[1].set_ylabel('largest common wedge (deg)')
ax[1].set_title('Worst phase alignment, identical prisms', fontsize=9)
ax[1].legend(fontsize=8)
fig.savefig(os.path.join(FIG, 'T1_tir_bound.png'))

# Figure T2: recovery vs margin
fig, ax = plt.subplots(figsize=(5.2, 3.2))
mid = np.arange(len(rb))
n = np.array([r[2] for r in rb]); k = np.array([r[3] for r in rb])
p, lo, hi = wilson(k, n)
cols = [C[1] if r[1] <= 0 else C[0] for r in rb]
ax.bar(mid, 100 * (1 - p), color=cols, width=0.8)
ax.errorbar(mid, 100 * (1 - p), yerr=[100 * (hi - p), 100 * (p - lo)],
            fmt='none', ecolor=INK2, lw=1)
ax.set_xticks(mid, [f'{r[0]:g}..{r[1]:g}' for r in rb], rotation=45, fontsize=7)
ax.set_xlabel('worst-case TIR margin (deg); orange = TIR reachable')
ax.set_ylabel('unsolved at T = 80 s (%)')
ax.set_title('Failure rate vs TIR margin (669,574 cases)', fontsize=9)
fig.savefig(os.path.join(FIG, 'T2_failure_vs_margin.png'))
print('figures written')
