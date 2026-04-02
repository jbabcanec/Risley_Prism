#!/usr/bin/env python3
"""Generate all figures for the paper."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

from core import PrismParameters, fast_forward

rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'text.usetex': False,
})

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Fig 1: Risley prism schematic — beam path through wedges
# ═══════════════════════════════════════════════════════════════════════
def fig_schematic():
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(-0.8, 13.5)
    ax.set_ylim(-2.6, 2.8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Laser source
    ax.add_patch(plt.Rectangle((-0.6, -0.2), 0.5, 0.4, fc='black', ec='black'))
    ax.text(-0.35, -0.65, 'Laser', fontsize=8, ha='center')

    # Physical wedge prisms — exaggerated taper for visibility
    prism_data = [
        (2.5, 15, '$W_1$', '$n_{g,1}$'),
        (5.8, -13, '$W_2$', '$n_{g,2}$'),
        (9.1, 10, '$W_3$', '$n_{g,3}$'),
    ]

    beam_pts = [(0.0, 0.0)]
    colors = ['#b3d4fc', '#b3e6cc', '#fce4b3']

    for idx, (xc, wa, plabel, nlabel) in enumerate(prism_data):
        h = 1.4
        thick = 1.1
        taper = thick * 0.5 * np.tan(np.radians(abs(wa)))
        sign = 1 if wa > 0 else -1

        verts = [
            (xc - thick/2, -h),
            (xc + thick/2, -h + sign*taper),
            (xc + thick/2,  h + sign*taper),
            (xc - thick/2,  h),
        ]
        prism = Polygon(verts, closed=True, facecolor=colors[idx],
                        edgecolor='black', lw=1.0, alpha=0.85, zorder=2)
        ax.add_patch(prism)

        # Face labels outside the prism
        ax.text(xc - thick/2 - 0.15, -h - 0.25, 'E', fontsize=7, ha='center',
                va='top', color='#333', weight='bold')
        ax.text(xc + thick/2 + 0.15, -h + sign*taper - 0.25, 'X', fontsize=7,
                ha='center', va='top', color='#333', weight='bold')

        # Glass label inside
        ax.text(xc, 0.0, nlabel, fontsize=8, ha='center', va='center', color='#333')

        # Prism label below
        ax.text(xc, -h - 0.55, plabel, fontsize=9, ha='center', va='top', weight='bold')

        # Rotation arrow
        arc = mpatches.Arc((xc, h + 0.35), 0.7, 0.5, angle=0, theta1=20, theta2=340,
                           color='#333', lw=0.8, ls='--')
        ax.add_patch(arc)
        ax.annotate('', xy=(xc+0.27, h+0.52), xytext=(xc+0.30, h+0.58),
                     arrowprops=dict(arrowstyle='->', lw=0.7, color='#333'))
        ax.text(xc, h + 0.75, f'$N_{idx+1}$', fontsize=7, ha='center', color='#333')

        # Beam path
        entry_x = xc - thick/2
        exit_x = xc + thick/2
        beam_pts.append((entry_x, beam_pts[-1][1] + 0.05 * (idx+1) * (-1)**idx))
        beam_pts.append((exit_x, beam_pts[-1][1] + sign * 0.18))

    beam_pts.append((12.2, beam_pts[-1][1] + 0.4))

    bx = [p[0] for p in beam_pts]
    by = [p[1] for p in beam_pts]
    ax.plot(bx, by, 'r-', lw=1.8, zorder=5)
    ax.plot(bx[-1], by[-1], 'r*', markersize=12, zorder=6)

    # Workpiece
    ax.plot([12.0, 12.4], [-1.5, 2.0], 'k-', lw=3)
    ax.text(12.2, -1.85, 'Workpiece', fontsize=8, ha='center')

    # Air labels
    ax.text(1.25, 1.8, '$n\!=\!1$', fontsize=6, ha='center', color='gray')
    ax.text(4.15, 1.8, '$n\!=\!1$', fontsize=6, ha='center', color='gray')
    ax.text(7.45, 1.8, '$n\!=\!1$', fontsize=6, ha='center', color='gray')

    # Bottom annotation — single line, well below prism labels
    ax.text(6.0, -2.45, 'E = entry face     X = exit face     3 prisms = 6 interfaces',
            fontsize=7, ha='center', style='italic', color='#555')

    fig.savefig(os.path.join(OUT, 'schematic.pdf'))
    fig.savefig(os.path.join(OUT, 'schematic.png'))
    plt.close()
    print("  schematic.pdf")


# ═══════════════════════════════════════════════════════════════════════
# Fig 2: Example scan patterns for 1, 2, 3 wedges
# ═══════════════════════════════════════════════════════════════════════
def fig_patterns():
    cases = [
        ("1 wedge\n$N=[1.5]$",
         PrismParameters(1, [1.5], [12.0], [5.0])),
        ("2 wedges\n$N=[1.5, -1.0]$",
         PrismParameters(2, [1.5, -1.0], [12.0, -8.0], [5.0, 8.0])),
        ("3 wedges\n$N=[1.5, -1.0, 2.0]$",
         PrismParameters(3, [1.5, -1.0, 2.0], [12.0, -8.0, 5.0], [3.0, 10.0, -6.0])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for ax, (title, params) in zip(axes, cases):
        pat = fast_forward(params, 500, 10.0)
        t = np.linspace(0, 10, 500)
        sc = ax.scatter(pat[:, 0], pat[:, 1], c=t, cmap='viridis', s=2, rasterized=True)
        ax.plot(pat[:, 0], pat[:, 1], 'k-', lw=0.3, alpha=0.3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('$x$ (units)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel('$y$ (units)')
    fig.colorbar(sc, ax=axes, label='Time (s)', shrink=0.85, pad=0.02)
    fig.savefig(os.path.join(OUT, 'patterns.pdf'))
    fig.savefig(os.path.join(OUT, 'patterns.png'))
    plt.close()
    print("  patterns.pdf")


# ═══════════════════════════════════════════════════════════════════════
# Fig 3: Sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════
def fig_sensitivity():
    base = PrismParameters(3, [1.5, -1.0, 2.0], [12.0, -8.0, 5.0], [3.0, 10.0, -6.0])
    pat0 = fast_forward(base, 200, 10.0)

    deltas_speed = np.linspace(-1.0, 1.0, 41)
    deltas_angle = np.linspace(-5.0, 5.0, 41)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for w in range(3):
        mses = []
        for d in deltas_speed:
            p = PrismParameters(3, list(base.rotation_speeds), list(base.wedge_angles_x), list(base.wedge_angles_y))
            p.rotation_speeds[w] += d
            mses.append(np.mean((fast_forward(p, 200, 10.0) - pat0)**2))
        axes[0].plot(deltas_speed, mses, color=colors[w], label=f'Wedge {w+1}')
    axes[0].set_xlabel('$\\Delta N$ (Hz)'); axes[0].set_ylabel('MSE'); axes[0].set_title('Rotation speed')
    axes[0].legend(fontsize=8)

    for w in range(3):
        mses = []
        for d in deltas_angle:
            p = PrismParameters(3, list(base.rotation_speeds), list(base.wedge_angles_x), list(base.wedge_angles_y))
            p.wedge_angles_x[w] += d
            mses.append(np.mean((fast_forward(p, 200, 10.0) - pat0)**2))
        axes[1].plot(deltas_angle, mses, color=colors[w])
    axes[1].set_xlabel('$\\Delta \\varphi_x$ (deg)'); axes[1].set_title('Wedge angle $\\varphi_x$')

    for w in range(3):
        mses = []
        for d in deltas_angle:
            p = PrismParameters(3, list(base.rotation_speeds), list(base.wedge_angles_x), list(base.wedge_angles_y))
            p.wedge_angles_y[w] += d
            mses.append(np.mean((fast_forward(p, 200, 10.0) - pat0)**2))
        axes[2].plot(deltas_angle, mses, color=colors[w])
    axes[2].set_xlabel('$\\Delta \\varphi_y$ (deg)'); axes[2].set_title('Wedge angle $\\varphi_y$')

    for ax in axes: ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'sensitivity.pdf'))
    fig.savefig(os.path.join(OUT, 'sensitivity.png'))
    plt.close()
    print("  sensitivity.pdf")


# ═══════════════════════════════════════════════════════════════════════
# Fig 4: Ref-ind impact — uniform vs varying
# ═══════════════════════════════════════════════════════════════════════
def fig_refind_impact():
    """2D heatmap: perturb both prisms' alpha_x simultaneously.
    Same speed → elongated valley (degenerate, prisms interchangeable).
    Different speeds → tight single minimum (well-posed)."""
    N = 71
    d1 = np.linspace(-8, 8, N)
    d2 = np.linspace(-8, 8, N)
    D1, D2 = np.meshgrid(d1, d2)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2),
                             gridspec_kw={'wspace': 0.35, 'right': 0.88})

    cases = [
        ("Same speed: $N = [1.0,\\, 1.0]$ Hz", [1.0, 1.0]),
        ("Different speeds: $N = [1.5,\\, -1.0]$ Hz", [1.5, -1.0]),
    ]

    for ax, (label, speeds) in zip(axes, cases):
        base = PrismParameters(2, speeds, [10.0, 8.0], [5.0, 6.0])
        pat0 = fast_forward(base, 200, 10.0)
        MSE = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                p = PrismParameters(2, speeds,
                                    [10.0 + d1[j], 8.0 + d2[i]],
                                    [5.0, 6.0])
                MSE[i, j] = np.mean((fast_forward(p, 200, 10.0) - pat0)**2)

        im = ax.pcolormesh(D1, D2, np.log10(MSE + 1e-14), cmap='viridis',
                           vmin=-2, vmax=2, shading='auto', rasterized=True)
        ax.contour(D1, D2, np.log10(MSE + 1e-14), levels=[-1, 0, 1],
                   colors='white', linewidths=0.7, linestyles='--')
        ax.plot(0, 0, 'r+', markersize=10, mew=2, zorder=5)
        ax.set_xlabel('$\\Delta\\alpha_{x,1}$ (deg)')
        ax.set_title(label, fontsize=9)
        ax.set_aspect('equal')

    axes[0].set_ylabel('$\\Delta\\alpha_{x,2}$ (deg)')
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='$\\log_{10}$(MSE)')
    fig.savefig(os.path.join(OUT, 'refind_impact.pdf'))
    fig.savefig(os.path.join(OUT, 'refind_impact.png'))
    plt.close()
    print("  refind_impact.pdf")


# ═══════════════════════════════════════════════════════════════════════
# Fig 5: Reconstruction — true vs NN vs optimised
# ═══════════════════════════════════════════════════════════════════════
def fig_reconstruction():
    from scipy.optimize import differential_evolution, minimize

    true_p = PrismParameters(3, [1.5, -1.0, 2.0], [12.0, -8.0, 5.0], [3.0, 10.0, -6.0])
    nn_p = PrismParameters(3, [0.5, -0.3, 0.2], [8.0, -3.0, 2.0], [2.0, -1.0, -0.5])

    pat_true = fast_forward(true_p, 300, 10.0)
    pat_nn   = fast_forward(nn_p, 300, 10.0)
    target   = fast_forward(true_p, 200, 10.0)  # DE uses 200 pts

    # Run DE and capture snapshots at specific eval counts
    eval_count = [0]
    best_so_far = [None, float('inf')]
    snapshots = {}  # eval_count -> best params
    snap_at = {5000, 12000, 18000}

    def objective(x):
        try:
            p = PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist())
            mse = float(np.mean((fast_forward(p, 200, 10.0) - target)**2))
        except:
            mse = 1e6
        eval_count[0] += 1
        if mse < best_so_far[1]:
            best_so_far[0] = x.copy()
            best_so_far[1] = mse
        if eval_count[0] in snap_at:
            snapshots[eval_count[0]] = (best_so_far[0].copy(), best_so_far[1])
        return mse

    bounds = [(-3.5, 3.5)]*3 + [(-18, 18)]*3 + [(-18, 18)]*3
    x0 = np.array([0.5, -0.3, 0.2, 8.0, -3.0, 2.0, 2.0, -1.0, -0.5])
    differential_evolution(objective, bounds, seed=42, maxiter=400, popsize=25,
                           tol=1e-14, polish=False, x0=x0, disp=False)

    # NM polish from best DE result
    def obj_nm(x):
        try:
            p = PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist())
            return float(np.mean((fast_forward(p, 200, 10.0) - target)**2))
        except:
            return 1e6
    res = minimize(obj_nm, best_so_far[0], method='Nelder-Mead',
                   options={'maxiter': 15000, 'fatol': 1e-12, 'xatol': 1e-12})
    final_x = res.x

    # Build panels: truth, NN, early DE, mid DE, late DE, final
    panel_data = [
        ('(a) Ground truth', true_p, None),
        ('(b) NN estimate', nn_p, None),
    ]
    # Add DE snapshots
    labels = ['(c)', '(d)', '(e)']
    for i, ev in enumerate(sorted(snapshots.keys())):
        x, mse = snapshots[ev]
        p = PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist())
        panel_data.append((f'{labels[i]} DE @ {ev//1000}k evals', p, mse))

    # Final polished
    fp = PrismParameters(3, final_x[:3].tolist(), final_x[3:6].tolist(), final_x[6:9].tolist())
    panel_data.append(('(f) After NM polish', fp, res.fun))

    t = np.linspace(0, 10, 300)
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0))
    axes = axes.flatten()

    # Get axis limits from ground truth
    xmin, xmax = pat_true[:, 0].min(), pat_true[:, 0].max()
    ymin, ymax = pat_true[:, 1].min(), pat_true[:, 1].max()
    pad = 0.1 * max(xmax - xmin, ymax - ymin)

    for ax, (title, params, mse) in zip(axes, panel_data):
        pat = fast_forward(params, 300, 10.0)
        ax.plot(pat[:, 0], pat[:, 1], 'k-', lw=0.3, alpha=0.3)
        ax.scatter(pat[:, 0], pat[:, 1], c=t, cmap='viridis', s=3, zorder=3, rasterized=True)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        ax.set_xlabel('$x$')
        if mse is not None:
            ax.text(0.04, 0.96, f'MSE = {mse:.1e}', transform=ax.transAxes,
                    va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    axes[0].set_ylabel('$y$')
    axes[3].set_ylabel('$y$')
    # MSE for NN panel
    mse_nn = np.mean((pat_nn - pat_true)**2)
    axes[1].text(0.04, 0.96, f'MSE = {mse_nn:.1e}', transform=axes[1].transAxes,
                 va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'reconstruction.pdf'))
    fig.savefig(os.path.join(OUT, 'reconstruction.png'))
    plt.close()
    print("  reconstruction.pdf")


# ═══════════════════════════════════════════════════════════════════════
# Fig 6: NN architecture diagram
# ═══════════════════════════════════════════════════════════════════════
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-2.5, 3.5)
    ax.axis('off')

    def box(x, y, w, h, label, color, fontsize=7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                     facecolor=color, edgecolor='black', lw=0.8))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=fontsize)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle='->', lw=1.0, color='black'))

    # Input
    box(-0.3, 0.5, 1.4, 2.0, 'Input\n$(x,y)_{1:200}$\n+\n$|\\mathrm{FFT}|$\n\n600 dims', '#e3f2fd', 8)

    # Backbone layers
    layers = [
        (1.8, 'Linear\n768\nBN+ReLU\nDrop 0.15'),
        (3.3, 'Linear\n512\nBN+ReLU\nDrop 0.15'),
        (4.8, 'Linear\n256\nBN+ReLU\nDrop 0.10'),
        (6.3, 'Linear\n128\nBN+ReLU'),
    ]
    for x, label in layers:
        box(x, 0.5, 1.2, 2.0, label, '#fff9c4', 7)

    # Arrows between backbone
    arrow(1.1, 1.5, 1.8, 1.5)
    for i in range(len(layers)-1):
        arrow(layers[i][0]+1.2, 1.5, layers[i+1][0], 1.5)

    # Classifier head
    box(8.0, 2.0, 1.4, 1.2, 'Classifier\n128→64→$C$\n\nCross-entropy', '#c8e6c9', 7)
    arrow(7.5, 2.0, 8.0, 2.6)

    # Regression heads
    box(8.0, -0.2, 1.4, 1.6, 'Regressors\n(per $W$)\n128→256\n→128→$3W$\nMSE', '#ffccbc', 7)
    arrow(7.5, 1.0, 8.0, 0.6)

    # Output
    box(9.8, 2.2, 1.0, 0.8, '$\\hat{W}$\nwedge\ncount', '#e8eaf6', 7)
    arrow(9.4, 2.6, 9.8, 2.6)

    box(9.8, -0.1, 1.0, 1.4, '$\\hat{N}_i$\n$\\hat{\\varphi}_{x,i}$\n$\\hat{\\varphi}_{y,i}$', '#e8eaf6', 8)
    arrow(9.4, 0.6, 9.8, 0.6)

    # Brace for backbone
    ax.text(4.2, 2.8, 'Shared backbone', fontsize=9, ha='center', style='italic')
    ax.plot([1.8, 7.5], [2.65, 2.65], 'k-', lw=0.5)

    fig.savefig(os.path.join(OUT, 'architecture.pdf'))
    fig.savefig(os.path.join(OUT, 'architecture.png'))
    plt.close()
    print("  architecture.pdf")


# ═══════════════════════════════════════════════════════════════════════
# Fig 7: Pipeline / system diagram
# ═══════════════════════════════════════════════════════════════════════
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.set_xlim(-0.3, 15.0)
    ax.set_ylim(-0.5, 2.6)
    ax.axis('off')

    boxes = [
        (0.0,  2.0, 'Observed\npattern\n$(x,y)_{1:T}$', '#e3f2fd'),
        (2.5,  2.0, 'Feature\nextraction\nraw + FFT\n(600-D)', '#fff3e0'),
        (5.0,  2.0, 'Neural\nnetwork\n(PyTorch)\nclassify + regress', '#c8e6c9'),
        (7.5,  2.0, 'Differential\nevolution\n4 restarts\n+ NM polish', '#ffccbc'),
        (10.3, 2.0, 'Permutation\nsearch\n$P!$ orderings\n+ NM polish', '#e8eaf6'),
        (12.8, 1.5, 'Recovered\nparameters\n$\\hat{\\theta}$', '#e3f2fd'),
    ]

    for x, w, txt, color in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.3), w, 1.9, boxstyle="round,pad=0.12",
                     facecolor=color, edgecolor='black', lw=0.8))
        ax.text(x + w/2, 1.25, txt, ha='center', va='center', fontsize=7.5)

    for i in range(len(boxes)-1):
        x1 = boxes[i][0] + boxes[i][1]
        x2 = boxes[i+1][0]
        ax.annotate('', xy=(x2, 1.25), xytext=(x1, 1.25),
                     arrowprops=dict(arrowstyle='->', lw=1.2, color='#333'))

    # Stage brackets
    ax.annotate('', xy=(0.0, 0.1), xytext=(7.0, 0.1),
                arrowprops=dict(arrowstyle='-', lw=0.5, color='#888'))
    ax.text(3.5, -0.15, 'Stage 1: $\\sim$2 ms', fontsize=9, ha='center',
            style='italic', color='#555')

    ax.annotate('', xy=(7.5, 0.1), xytext=(14.3, 0.1),
                arrowprops=dict(arrowstyle='-', lw=0.5, color='#888'))
    ax.text(10.9, -0.15, 'Stage 2: 25--220 s', fontsize=9, ha='center',
            style='italic', color='#555')

    fig.savefig(os.path.join(OUT, 'pipeline.pdf'))
    fig.savefig(os.path.join(OUT, 'pipeline.png'))
    plt.close()
    print("  pipeline.pdf")


# ═══════════════════════════════════════════════════════════════════════
# Fig 8: Convergence — optimiser MSE vs function evaluations
# ═══════════════════════════════════════════════════════════════════════
def fig_convergence():
    """Simulate convergence by running the optimiser and tracking progress."""
    from scipy.optimize import differential_evolution

    true_p = PrismParameters(3, [1.5, -1.0, 2.0], [12.0, -8.0, 5.0], [3.0, 10.0, -6.0])
    target = fast_forward(true_p, 200, 10.0)

    history = []
    best_so_far = [float('inf')]

    def objective(x):
        try:
            p = PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist())
            mse = float(np.mean((fast_forward(p, 200, 10.0) - target)**2))
        except:
            mse = 1e6
        if mse < best_so_far[0]:
            best_so_far[0] = mse
        history.append(best_so_far[0])
        return mse

    bounds = [(-3.5, 3.5)]*3 + [(-18, 18)]*3 + [(-18, 18)]*3
    # Seed with a rough NN-like estimate
    x0 = np.array([1.4, -0.1, 0.1, 12.0, -2.0, 1.0, 3.5, 0.0, -0.5])

    differential_evolution(objective, bounds, seed=42, maxiter=200, popsize=25,
                           tol=1e-9, polish=False, x0=x0, disp=False)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.semilogy(range(len(history)), history, 'b-', lw=0.8)
    ax.set_xlabel('Function evaluations')
    ax.set_ylabel('Best MSE')
    ax.set_title('Optimiser convergence (3 wedges)')
    ax.grid(True, alpha=0.2, which='both')
    ax.axhline(1e-3, color='r', ls='--', lw=0.8, label='$10^{-3}$ threshold')
    ax.axhline(1e-6, color='g', ls='--', lw=0.8, label='$10^{-6}$ threshold')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'convergence.pdf'))
    fig.savefig(os.path.join(OUT, 'convergence.png'))
    plt.close()
    print("  convergence.pdf")


if __name__ == '__main__':
    print("Generating paper figures...")
    fig_schematic()
    fig_patterns()
    fig_sensitivity()
    fig_refind_impact()
    fig_reconstruction()
    fig_architecture()
    fig_pipeline()
    fig_convergence()
    print("Done — all figures in figures/")
