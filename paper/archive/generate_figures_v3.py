#!/usr/bin/env python3
"""Generate all new figures for the revised paper. Clean, publication-quality."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'text.usetex': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

import torch
from ml_staged_solver import (
    DiffFwd, DEVICE, P, T_PTS, T_OBS, SRC, THK,
    N_PAR, NAMES, LO, HI, RG, canon,
)
from core import PrismParameters, SystemGeometry, fast_forward


def vec2pat(v):
    geo = SystemGeometry(
        source_distance=SRC, prism_thickness=THK,
        workpiece_distance=float(v[12]), inter_prism_gap=float(v[13]),
        beam_angle_x=float(v[14]), beam_angle_y=float(v[15]),
        beam_pos_x=float(v[16]), beam_pos_y=float(v[17]))
    pr = PrismParameters(P, v[:3].tolist(), v[3:6].tolist(), v[6:9].tolist(),
                         glass_indices=v[9:12].tolist(), geometry=geo)
    return fast_forward(pr, T_PTS, T_OBS)


def save(fig, name):
    for ext in ('.pdf', '.png'):
        fig.savefig(os.path.join(OUT, name + ext))
    plt.close(fig)
    print(f'    -> {name}.pdf', flush=True)


# ================================================================
#  Figure: Jacobian Singular Value Spectrum
# ================================================================

def fig_svd_spectrum():
    print('  SVD spectrum...', flush=True)

    tv = np.array([1.5,-1.,2., 12.,-8.,5., 3.,10.,-6.,
                   1.5,1.55,1.6, 100.,6., 10.,5., 0.,0.], np.float64)
    theta = canon(tv)

    fwd = DiffFwd().to(DEVICE)
    theta_t = torch.tensor(theta, dtype=torch.float64, device=DEVICE).requires_grad_(True)
    def fwd_flat(t):
        return fwd(t.unsqueeze(0), high_precision=True).squeeze(0).reshape(-1)
    J = torch.autograd.functional.jacobian(fwd_flat, theta_t).detach().cpu().numpy()
    _, sigma, Vt = np.linalg.svd(J, full_matrices=False)

    # Map each SV to dominant parameter type
    param_types = (['speed']*3 + ['angle_x']*3 + ['angle_y']*3 +
                   ['glass']*3 + ['geom']*2 + ['beam_ang']*2 + ['beam_pos']*2)

    cmap = {
        'speed':    '#1f77b4',
        'glass':    '#2ca02c',
        'beam_ang': '#ff7f0e',
        'angle_x':  '#d62728',
        'angle_y':  '#e377c2',
        'beam_pos': '#aec7e8',
        'geom':     '#9467bd',
    }
    lmap = {
        'speed':    'Speeds $N$',
        'glass':    'Glass $n_g$',
        'beam_ang': 'Beam angles',
        'angle_x':  'Wedge $\\alpha_x$',
        'angle_y':  'Wedge $\\alpha_y$',
        'beam_pos': 'Beam position',
        'geom':     'Geometry',
    }

    colors = []
    for i in range(18):
        dom = param_types[np.argmax(np.abs(Vt[i]))]
        colors.append(cmap[dom])

    fig, ax = plt.subplots(figsize=(3.4, 2.6))  # single-column width

    x = np.arange(1, 19)
    ax.bar(x, sigma, color=colors, edgecolor='black', linewidth=0.4, width=0.75)
    ax.set_yscale('log')
    ax.set_xlabel('Singular value index $k$')
    ax.set_ylabel('$\\sigma_k$')
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=6.5)
    ax.set_xlim(0.2, 18.8)
    ax.set_ylim(5e-3, 2e4)

    # Clean annotations — outside the plot area, no overlap
    kappa = sigma[0] / sigma[-1]
    ax.set_title(f'$\\kappa = \\sigma_1/\\sigma_{{18}} = {kappa:.1e}$',
                 fontsize=9, pad=6)

    # Legend — below the plot, horizontal
    seen = []
    handles = []
    for i in range(18):
        dom = param_types[np.argmax(np.abs(Vt[i]))]
        if dom not in seen:
            seen.append(dom)
            handles.append(Patch(facecolor=cmap[dom], edgecolor='black',
                                 linewidth=0.4, label=lmap[dom]))
    ax.legend(handles=handles, loc='upper center',
              bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=6.5,
              frameon=False, columnspacing=0.8, handletextpad=0.3)

    fig.subplots_adjust(bottom=0.28)
    save(fig, 'svd_spectrum')


# ================================================================
#  Figure: Pipeline Convergence (two panels)
# ================================================================

def fig_pipeline_convergence():
    print('  Pipeline convergence...', flush=True)

    from solve_preconditioned import solve
    from ml_staged_solver import AngleNet, RemainNet, ANG_MODEL, REM_MODEL

    ang_net = AngleNet()
    rem_net = RemainNet()
    ang_net.load_state_dict(torch.load(ANG_MODEL, map_location=DEVICE, weights_only=True))
    rem_net.load_state_dict(torch.load(REM_MODEL, map_location=DEVICE, weights_only=True))
    ang_net.to(DEVICE); rem_net.to(DEVICE)

    tv = np.array([1.5,-1.,2., 12.,-8.,5., 3.,10.,-6.,
                   1.5,1.55,1.6, 100.,6., 10.,5., 0.,0.], np.float64)
    tc = canon(tv)
    pattern = vec2pat(tc)

    solved, mse, details = solve(ang_net, rem_net, pattern,
                                  verbose=False, return_details=True)
    errs = np.abs(solved - tc)

    # ---- Panel (a): MSE per stage ----
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

    ax = axes[0]
    stage_names = ['ML\ninit', 'Sign\nselect', 'Adam', 'Trust\nregion']
    mses = [details['mse_after_ml'], details['mse_after_signs'],
            details['mse_after_adam'], details['mse_final']]
    stage_colors = ['#fee08b', '#fdae61', '#f46d43', '#a50026']

    bars = ax.bar(range(4), mses, color=stage_colors, edgecolor='black',
                  linewidth=0.5, width=0.65)
    ax.set_yscale('log')
    ax.set_xticks(range(4))
    ax.set_xticklabels(stage_names, fontsize=7.5)
    ax.set_ylabel('Pattern MSE')
    ax.set_ylim(1e-28, 1e4)
    ax.set_title('(a) MSE after each stage', fontsize=9)

    for i, (b, m) in enumerate(zip(bars, mses)):
        if m > 1e-25:
            ax.text(i, m * 3, f'{m:.0e}', ha='center', va='bottom', fontsize=7)
        else:
            ax.text(i, 1e-25, f'{m:.0e}', ha='center', va='bottom', fontsize=7)

    # ---- Panel (b): Per-parameter errors ----
    ax = axes[1]
    groups = [
        ('Speeds',    [0,1,2],       '#1f77b4'),
        ('$\\alpha_x$', [3,4,5],     '#d62728'),
        ('$\\alpha_y$', [6,7,8],     '#e377c2'),
        ('Glass',     [9,10,11],     '#2ca02c'),
        ('Geom.',     [12,13],       '#9467bd'),
        ('Beam',      [14,15,16,17], '#ff7f0e'),
    ]

    pos = 0
    tick_pos = []
    tick_lab = []
    for gname, indices, gc in groups:
        gstart = pos
        for idx in indices:
            ax.bar(pos, errs[idx], color=gc, edgecolor='black',
                   linewidth=0.3, width=0.7)
            pos += 1
        tick_pos.append((gstart + pos - 1) / 2)
        tick_lab.append(gname)
        pos += 0.8

    ax.set_yscale('log')
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=7.5)
    ax.set_ylabel('Absolute error')
    ax.set_title('(b) Per-parameter error', fontsize=9)
    ax.set_ylim(1e-16, 1e-9)
    ax.set_xlim(-0.5, pos - 0.5)

    plt.tight_layout(w_pad=2.5)
    save(fig, 'pipeline_convergence')


# ================================================================
#  Figure: OOD — piecewise-linear input vs Risley best fit
# ================================================================

def fig_ood():
    """Simplified 3-panel OOD figure: input, best fit, overlay."""
    print('  OOD convergence (new solver)...', flush=True)

    from solve_preconditioned import solve
    from ml_staged_solver import AngleNet, RemainNet, ANG_MODEL, REM_MODEL
    from scipy.optimize import least_squares

    ang_net = AngleNet()
    rem_net = RemainNet()
    ang_net.load_state_dict(torch.load(ANG_MODEL, map_location=DEVICE, weights_only=True))
    rem_net.load_state_dict(torch.load(REM_MODEL, map_location=DEVICE, weights_only=True))
    ang_net.to(DEVICE); rem_net.to(DEVICE)

    # Generate a real Risley pattern, then linearize it (piecewise-linear)
    tv = np.array([1.5,-1.,2., 12.,-8.,5., 3.,10.,-6.,
                   1.5,1.55,1.6, 100.,6., 10.,5., 0.,0.], np.float64)
    tc = canon(tv)
    real_pat = vec2pat(tc)  # (200, 2)

    # Create piecewise-linear version: connect every 20th point
    n_pts = 200
    step = 20
    knots = list(range(0, n_pts, step)) + [n_pts - 1]
    pw_pat = np.zeros_like(real_pat)
    for i in range(len(knots) - 1):
        i0, i1 = knots[i], knots[i+1]
        for j in range(i0, i1 + 1):
            t = (j - i0) / (i1 - i0)
            pw_pat[j] = (1 - t) * real_pat[i0] + t * real_pat[i1]

    # Solve for the nearest Risley pattern
    solved, mse, details = solve(ang_net, rem_net, pw_pat,
                                  verbose=False, return_details=True)
    fit_pat = vec2pat(solved)

    residual_mse = float(np.mean((fit_pat - pw_pat)**2))

    # ---- 3-panel figure ----
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4))

    t_color = np.linspace(0, 1, n_pts)

    # (a) Input
    ax = axes[0]
    ax.scatter(pw_pat[:, 0], pw_pat[:, 1], c=t_color, cmap='viridis',
               s=4, linewidths=0, zorder=2)
    ax.plot(pw_pat[:, 0], pw_pat[:, 1], 'k-', linewidth=0.3, alpha=0.3, zorder=1)
    ax.set_title('(a) Input (piecewise-linear)', fontsize=9)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_aspect('equal')

    # (b) Best Risley fit
    ax = axes[1]
    ax.scatter(fit_pat[:, 0], fit_pat[:, 1], c=t_color, cmap='viridis',
               s=4, linewidths=0, zorder=2)
    ax.plot(fit_pat[:, 0], fit_pat[:, 1], 'k-', linewidth=0.3, alpha=0.3, zorder=1)
    ax.set_title(f'(b) Best Risley fit', fontsize=9)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_aspect('equal')

    # (c) Overlay
    ax = axes[2]
    ax.plot(pw_pat[:, 0], pw_pat[:, 1], 'k-', linewidth=0.8, alpha=0.6,
            label='Input', zorder=1)
    ax.plot(fit_pat[:, 0], fit_pat[:, 1], 'r-', linewidth=0.8, alpha=0.8,
            label='Risley fit', zorder=2)
    ax.set_title(f'(c) Overlay (MSE = {residual_mse:.1f})', fontsize=9)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='lower right')

    # Match axis limits across all panels
    all_x = np.concatenate([pw_pat[:, 0], fit_pat[:, 0]])
    all_y = np.concatenate([pw_pat[:, 1], fit_pat[:, 1]])
    pad = 3
    for ax in axes:
        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
        ax.tick_params(labelsize=7)

    plt.tight_layout(w_pad=1.5)
    save(fig, 'ood_convergence')
    print(f'    Residual MSE = {residual_mse:.2f}', flush=True)


# ================================================================
#  Main
# ================================================================

if __name__ == '__main__':
    print('Generating figures...', flush=True)
    fig_svd_spectrum()
    fig_pipeline_convergence()
    fig_ood()
    print('All done.', flush=True)
