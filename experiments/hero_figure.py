#!/usr/bin/env python3
"""The thesis figure: a complicated observed pattern, the line spectrum the
method reads (lattice-labeled), and the reproduction from the recovered
18 parameters overlaid on the observation -- with the measured solve time
and errors. Also regenerates the Jacobian SVD spectrum (fresh, from the
package, at the recovered solution).

Run from repo root: python experiments/hero_figure.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from risley_lattice import (DT, N_PAR, vec2pat, battery_cases, solve18,
                            extract_speeds, lattice_fit)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'paper', 'figures')
BLUE = '#3b6fb6'
ORANGE = '#d1495b'
INK = '#333333'
MUT = '#8a8a8a'

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.edgecolor': MUT, 'axes.linewidth': 0.6,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
})

# ---------------------------------------------------------------- solve
CASE = 15                       # battery case 16: 30 cycles, dense, aliased
tc = battery_cases()[CASE]
pat = vec2pat(tc)
t0 = time.time()
x18, mse, how, info = solve18(pat)
secs = time.time() - t0
err = float(np.max(np.abs(x18 - tc)))
rep = vec2pat(x18)
print(f'case {CASE+1}: err {err:.1e} mse {mse:.1e} ({how}) in {secs:.1f}s')

# fitted lattice model for the spectrum panel
t = np.arange(len(pat)) * DT
z = pat[:, 0] + 1j * pat[:, 1]
mask = info.get('mask', np.ones(len(pat), bool))
g, K, c, res = lattice_fit(z, t, mask, list(x18[:3]), B=3)

# dense evaluations: the model is continuous-time; draw the true smooth
# trajectory, with the 200 observed samples as dots
dense_true = vec2pat(tc, 4000, 10.0)
dense_rep = vec2pat(x18, 4000, 10.0)


def sci(v):
    m, e = f'{v:.0e}'.split('e')
    return f'{m}\\times10^{{{int(e)}}}'


fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

# (a) observed pattern
ax = axes[0]
ax.plot(dense_true[:, 0], dense_true[:, 1], '-', color=INK, lw=0.45,
        alpha=0.55, zorder=2)
ax.scatter(pat[:, 0], pat[:, 1], c=np.arange(len(pat)), cmap='viridis',
           s=4.5, zorder=3)
ax.set_title('(a) observed: 200 samples, 10 s', loc='left')
ax.set_xlabel('$x$ (workpiece units)')
ax.set_ylabel('$y$')
ax.set_aspect('equal', adjustable='datalim')

# (b) line spectrum with lattice labels (DC excluded)
ax = axes[1]
amp = np.abs(c)
freqs = K @ g
nz = np.abs(K).sum(1) > 0
ref = amp[nz].max()
shown = [j for j in np.argsort(-amp) if nz[j] and amp[j] > 1e-4 * ref][:30]
ax.vlines(freqs[shown], 1e-4, amp[shown] / ref, color=BLUE, lw=1.1)
lab = [((1, 0, 0), '$N_1$', 8), ((0, 1, 0), '$N_2$', 0),
       ((0, 0, 1), '$N_3$', -10)]
for k, txt, dx in lab:
    j = int(np.argmin(np.abs(K - np.array(k, float)).sum(1)))
    ax.annotate(txt, (freqs[j], amp[j] / ref), textcoords='offset points',
                xytext=(dx, 4), ha='center', fontsize=7.5, color=ORANGE)
ax.set_yscale('log')
ax.set_ylim(1e-4, 4.0)
ax.set_xlabel('signed frequency (Hz)')
ax.set_ylabel('$|c_{\\bf k}|$ / max')
ax.set_title('(b) its line spectrum: ${\\bf k}\\cdot{\\bf N}$ lattice',
             loc='left')

# (c) reproduction overlay
ax = axes[2]
ax.plot(dense_true[:, 0], dense_true[:, 1], '-', color=INK, lw=2.0,
        alpha=0.28, label='observed', solid_capstyle='round')
ax.plot(dense_rep[:, 0], dense_rep[:, 1], '-', color=ORANGE, lw=0.55,
        label='reproduced from $\\hat{\\bm{\\theta}}$'
        .replace('\\bm', ''))
ax.set_title(f'(c) all 18 parameters, {secs:.1f} s', loc='left')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_aspect('equal', adjustable='datalim')
# no legend inside the panel: observed vs reproduced is stated in the caption
ax.text(0.03, 0.03,
        f'max param err ${sci(err)}$\npattern MSE ${sci(mse)}$',
        transform=ax.transAxes, fontsize=7, color=INK, va='bottom',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.85,
                  boxstyle='round,pad=0.25'))

fig.tight_layout(w_pad=1.6)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'hero.{ext}'), dpi=300,
                bbox_inches='tight')
print('wrote figures/hero.pdf/.png')

# ---------------------------------------------------------------- SVD figure
target = pat.reshape(-1)


def resid(x):
    return vec2pat(x).reshape(-1) - target


J = np.empty((len(target), N_PAR))
for i in range(N_PAR):
    h = max(1e-7, 1e-7 * abs(x18[i]))
    xp = x18.copy(); xp[i] += h
    xm = x18.copy(); xm[i] -= h
    J[:, i] = (resid(xp) - resid(xm)) / (2 * h)
U, S, Vt = np.linalg.svd(J, full_matrices=False)
groups = [('speeds', range(0, 3), '#3b6fb6'),
          ('wedge $\\alpha_x$', range(3, 6), '#66a182'),
          ('wedge $\\alpha_y$', range(6, 9), '#d1495b'),
          ('glass', range(9, 12), '#edae49'),
          ('geometry', range(12, 14), '#9067c6'),
          ('beam', range(14, 18), '#8a8a8a')]
dom = []
for k in range(N_PAR):
    v = np.abs(Vt[k])
    gi = max(range(6), key=lambda q: v[list(groups[q][1])].sum())
    dom.append(gi)

fig2, ax = plt.subplots(figsize=(3.4, 2.5))
for q, (name, rng, colr) in enumerate(groups):
    idx = [k for k in range(N_PAR) if dom[k] == q]
    if idx:
        ax.scatter([k + 1 for k in idx], S[idx], s=18, color=colr,
                   label=name, zorder=3, edgecolor='white', linewidth=0.4)
ax.vlines(range(1, N_PAR + 1), 1e-3, S, color='#cccccc', lw=0.8, zorder=1)
ax.set_yscale('log')
ax.set_xlabel('singular value index $k$')
ax.set_ylabel('$\\sigma_k$')
ax.set_xticks([1, 3, 6, 9, 12, 15, 18])
ax.set_title(f'Jacobian spectrum at the solution: full rank 18, '
             f'$\\kappa = {sci(S[0] / S[-1])}$', loc='left', fontsize=7.5)
ax.legend(frameon=False, fontsize=6.5, ncol=2, loc='upper right',
          borderaxespad=0.1, handletextpad=0.1, columnspacing=0.8)
fig2.tight_layout()
for ext in ('pdf', 'png'):
    fig2.savefig(os.path.join(OUT, f'svd_fresh.{ext}'), dpi=300,
                 bbox_inches='tight')
print(f'wrote figures/svd_fresh.pdf/.png  (kappa {S[0]/S[-1]:.2e})')
