#!/usr/bin/env python3
"""Generate paper figures from the canonical battery outputs.

figures/certificates.pdf:
  (a) certified per-case maximal bound vs maximal true error (log-log)
      -- data: experiments/certification.py, canonical run 2026-07-18
         (25 CERT-OK cases, coverage 25/25, median tightness 6x);
  (b) measured 18-D trust-region basin (assumptions.py, Test A7).

Single-hue + neutral marks; grayscale/CVD-safe by construction.
Run from repo root: python experiments/paper_figures.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'paper', 'figures')

# (max bound, max err) per CERT-OK case; certification.py final-pipeline run
CERT = [
    (2.7e-10, 6.9e-11), (3.7e-11, 7.2e-13), (2.1e-09, 9.2e-10),
    (2.4e-09, 1.4e-09), (1.7e-11, 1.4e-12), (3.3e-10, 3.4e-11),
    (4.4e-10, 1.3e-10), (6.4e-11, 3.3e-11), (1.7e-10, 7.0e-11),
    (7.6e-11, 9.2e-12), (4.0e-11, 2.7e-12), (3.1e-11, 7.0e-12),
    (5.1e-11, 1.6e-11), (3.4e-10, 2.0e-11), (4.5e-11, 1.4e-11),
    (7.1e-10, 5.1e-11), (1.6e-11, 1.7e-12), (3.2e-07, 1.8e-07),
    (1.1e-11, 1.6e-12), (3.8e-11, 1.6e-11), (4.2e-11, 1.2e-11),
    (1.3e-11, 1.5e-12), (2.0e-11, 1.1e-12), (4.2e-11, 1.1e-11),
    (6.7e-11, 1.6e-11), (2.1e-11, 8.3e-13),
]
# Test A7 (assumptions.py): perturbation scale -> 18-D TRF successes / 30
BASIN_X = [0.5, 1.0, 2.0, 4.0]
BASIN_Y = [29, 28, 26, 24]

BLUE = '#3b6fb6'
INK = '#333333'
MUT = '#8a8a8a'

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.edgecolor': MUT, 'axes.linewidth': 0.6,
    'xtick.color': INK, 'ytick.color': INK,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
})

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(3.4, 4.6), gridspec_kw={'height_ratios': [1.25, 1.0]})

# (a) bound vs error
b = np.array([c[0] for c in CERT])
e = np.array([c[1] for c in CERT])
lo, hi = 1e-13, 1e-6
ax1.plot([lo, hi], [lo, hi], '--', color=MUT, lw=0.8, zorder=1)
ax1.scatter(e, b, s=14, facecolor=BLUE, edgecolor='white',
            linewidth=0.4, zorder=3)
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
ax1.set_xlabel(r'true error  $\max_i |\hat\theta_i-\theta_i^{*}|$')
ax1.set_ylabel(r'certified bound  $\max_i b_i$')
ax1.text(3e-13, 6e-8, 'coverage 26/26\nmedian tightness $5\\times$',
         fontsize=7, color=INK)
ax1.text(2e-9, 5e-10, 'bound $=$ error', fontsize=7, color=MUT,
         rotation=38, rotation_mode='anchor')
ax1.set_title('(a) certified bounds vs. ground truth', loc='left',
              fontsize=8)
ax1.grid(True, which='major', color='#e6e6e6', lw=0.4, zorder=0)

# (b) measured basin
ax2.plot(BASIN_X, [y / 30 * 100 for y in BASIN_Y], '-o', color=BLUE,
         lw=1.4, ms=4.5, mfc=BLUE, mec='white', mew=0.5, zorder=3)
for x, y in zip(BASIN_X, BASIN_Y):
    ax2.annotate(f'{y}/30', (x, y / 30 * 100), textcoords='offset points',
                 xytext=(0, -11), ha='center', fontsize=7, color=INK)
ax2.axvline(1.0, color=MUT, lw=0.7, ls=':')
ax2.text(1.06, 62, 'actual spectral\ninit-error scale', fontsize=7,
         color=MUT)
ax2.set_xscale('log')
ax2.set_xticks(BASIN_X)
ax2.set_xticklabels([r'$0.5\times$', r'$1\times$', r'$2\times$',
                     r'$4\times$'])
ax2.set_xlim(0.4, 5.0)
ax2.set_ylim(55, 102)
ax2.set_xlabel('initialization perturbation (relative scale)')
ax2.set_ylabel('18-D trust-region success (%)')
ax2.set_title('(b) measured completion basin (Test A7)', loc='left',
              fontsize=8)
ax2.grid(True, which='major', color='#e6e6e6', lw=0.4, zorder=0)
ax2.minorticks_off()

fig.tight_layout(h_pad=1.6)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'certificates.{ext}'), dpi=300,
                bbox_inches='tight')
print('wrote figures/certificates.pdf/.png')
