#!/usr/bin/env python3
"""
solution_manifold.py — Characterize the solution set of the Risley inverse problem.

Given a target pattern p*, the solution set S = {θ : F(θ) ≈ p*} is NOT a single
point for the 18-D problem.  This script:

  1. Computes the Jacobian DF at the true solution
  2. SVD → singular values reveal identifiable vs degenerate directions
  3. Traces the solution manifold by following null-space directions
  4. Shows which parameters are identifiable and which trade off

Usage:  python paper/solution_manifold.py
"""

import sys, os, time, io
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'reverse_problem_v2'))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from core import PrismParameters, SystemGeometry, fast_forward

import torch
import torch.nn.functional as F

# Import DiffFwd from the staged solver
sys.path.insert(0, HERE)
from ml_staged_solver import DiffFwd, DEVICE, P, T_PTS, T_OBS, SRC, THK

NAMES = ['N1','N2','N3','ax1','ax2','ax3','ay1','ay2','ay3',
         'ng1','ng2','ng3','d_W','gap','bm_ax','bm_ay','bm_px','bm_py']
N_PAR = 18


def vec2pat(v):
    geo = SystemGeometry(
        source_distance=SRC, prism_thickness=THK,
        workpiece_distance=float(v[12]), inter_prism_gap=float(v[13]),
        beam_angle_x=float(v[14]), beam_angle_y=float(v[15]),
        beam_pos_x=float(v[16]), beam_pos_y=float(v[17]))
    pr = PrismParameters(P, v[:3].tolist(), v[3:6].tolist(), v[6:9].tolist(),
                         glass_indices=v[9:12].tolist(), geometry=geo)
    return fast_forward(pr, T_PTS, T_OBS)


# ================================================================
#  1. Jacobian Computation
# ================================================================

def compute_jacobian(theta, fwd):
    """
    Compute DF(θ): the Jacobian of the forward map at θ.
    Returns (400, 18) matrix: d(pattern_flat) / d(params).
    """
    theta_t = torch.tensor(theta, dtype=torch.float64, device=DEVICE).requires_grad_(True)

    def fwd_flat(t):
        return fwd(t.unsqueeze(0), high_precision=True).squeeze(0).reshape(-1)

    J = torch.autograd.functional.jacobian(fwd_flat, theta_t)
    return J.detach().cpu().numpy()  # (400, 18)


# ================================================================
#  2. SVD Analysis
# ================================================================

def analyze_jacobian(J):
    """SVD of the Jacobian → identifiability analysis."""
    U, sigma, Vt = np.linalg.svd(J, full_matrices=False)
    # sigma: (18,) singular values, sorted descending
    # Vt: (18, 18) right singular vectors (rows)
    return sigma, Vt


# ================================================================
#  3. Manifold Tracing
# ================================================================

def trace_manifold(theta_true, direction, fwd, n_points=50, step_size=0.01):
    """
    Starting from theta_true, step along `direction` in parameter space.
    At each step, record the pattern MSE (should stay near 0 for null-space dirs).
    Returns array of (param_vector, pattern_mse) along the path.
    """
    target = vec2pat(theta_true)
    target_t = torch.tensor(target, dtype=torch.float64, device=DEVICE)

    path = []
    for i in range(-n_points, n_points + 1):
        theta_i = theta_true + i * step_size * direction
        theta_t = torch.tensor(theta_i, dtype=torch.float64, device=DEVICE)
        with torch.no_grad():
            pred = fwd(theta_t.unsqueeze(0), high_precision=True).squeeze(0)
            mse = F.mse_loss(pred, target_t).item()
        path.append((i * step_size, theta_i.copy(), mse))

    return path


# ================================================================
#  Main
# ================================================================

if __name__ == '__main__':
    fwd = DiffFwd().to(DEVICE)

    # True parameters (paper test case)
    theta_true = np.array([
        2.0, 1.5, -1.0,           # speeds (canonical order)
        5.0, 12.0, -8.0,          # ax
        -6.0, 3.0, 10.0,          # ay
        1.60, 1.50, 1.55,         # glass (canonical)
        100.0, 6.0,               # d_W, gap
        10.0, 5.0,                # beam angles
        0.0, 0.0,                 # beam positions
    ], dtype=np.float64)

    print('=' * 70)
    print(' SOLUTION MANIFOLD ANALYSIS')
    print('=' * 70)

    # ---- Step 1: Jacobian ----
    print('\n--- Jacobian at true solution ---')
    t0 = time.time()
    J = compute_jacobian(theta_true, fwd)
    print(f'  Computed in {time.time()-t0:.1f}s,  shape: {J.shape}')

    # ---- Step 2: SVD ----
    print('\n--- Singular Value Decomposition ---')
    sigma, Vt = analyze_jacobian(J)

    print(f'\n  Singular values (18):')
    for i, s in enumerate(sigma):
        # Identify which parameters contribute most to this SV
        top_params = np.argsort(-np.abs(Vt[i]))[:3]
        param_str = ', '.join(f'{NAMES[j]}({Vt[i,j]:+.2f})' for j in top_params)
        status = 'WELL-COND' if s > 1.0 else 'WEAK' if s > 0.01 else 'DEGENERATE'
        print(f'    sigma_{i+1:2d} = {s:12.4e}  [{status:10s}]  '
              f'direction: {param_str}')

    # Effective rank
    threshold = sigma[0] * 1e-6
    rank = np.sum(sigma > threshold)
    null_dim = N_PAR - rank
    cond = sigma[0] / sigma[-1] if sigma[-1] > 0 else float('inf')

    print(f'\n  Condition number: {cond:.2e}')
    print(f'  Effective rank (threshold={threshold:.2e}): {rank}')
    print(f'  Null space dimension: {null_dim}')
    print(f'  → The solution manifold is (approximately) {null_dim}-dimensional')

    # ---- Step 3: Identifiability map ----
    print('\n--- Per-Parameter Sensitivity ---')
    # For each parameter, its sensitivity = norm of the corresponding column of J
    col_norms = np.linalg.norm(J, axis=0)  # (18,)
    order = np.argsort(-col_norms)

    print(f'  {"Param":8s} {"||dF/dp||":>12s} {"Relative":>10s} {"Status":>12s}')
    max_norm = col_norms.max()
    for i in order:
        rel = col_norms[i] / max_norm
        status = ('IDENTIFIABLE' if rel > 0.01
                  else 'WEAK' if rel > 1e-4
                  else 'DEGENERATE')
        print(f'  {NAMES[i]:8s} {col_norms[i]:12.4e} {rel:10.2e} {status:>12s}')

    # ---- Step 4: Trace manifold along weakest direction ----
    print(f'\n--- Manifold Trace (weakest singular direction) ---')

    # The last right singular vector = most degenerate direction
    degen_dir = Vt[-1]  # (18,) unit vector in the most degenerate direction
    print(f'  Degenerate direction components:')
    for i in range(N_PAR):
        if abs(degen_dir[i]) > 0.05:
            print(f'    {NAMES[i]:8s}: {degen_dir[i]:+.4f}')

    print(f'\n  Tracing manifold (±50 steps, step=0.1):')
    path = trace_manifold(theta_true, degen_dir, fwd,
                          n_points=50, step_size=0.1)

    # Show every 10th point
    print(f'  {"t":>6s} {"pat_MSE":>12s}  '
          f'{"N1":>7s} {"ax1":>7s} {"ay1":>7s} {"ng1":>7s} {"d_W":>7s} {"gap":>7s}')
    for t, theta, mse in path[::10]:
        print(f'  {t:+6.1f} {mse:12.2e}  '
              f'{theta[0]:7.3f} {theta[3]:7.3f} {theta[6]:7.3f} '
              f'{theta[9]:7.4f} {theta[12]:7.1f} {theta[13]:7.2f}')

    # ---- Step 5: Compare 9-D vs 13-D vs 18-D rank ----
    print(f'\n--- Subproblem Analysis ---')

    # 9-D: only speeds + angles (columns 0-8)
    J_9d = J[:, :9]
    s_9d = np.linalg.svd(J_9d, compute_uv=False)
    rank_9d = np.sum(s_9d > s_9d[0] * 1e-6)
    cond_9d = s_9d[0] / s_9d[-1]
    print(f'  9-D (speeds+angles):  rank={rank_9d}/9   '
          f'cond={cond_9d:.2e}   min_sv={s_9d[-1]:.2e}')

    # 13-D: speeds + angles + beam (columns 0-8, 14-17)
    cols_13 = list(range(9)) + list(range(14, 18))
    J_13d = J[:, cols_13]
    s_13d = np.linalg.svd(J_13d, compute_uv=False)
    rank_13d = np.sum(s_13d > s_13d[0] * 1e-6)
    cond_13d = s_13d[0] / s_13d[-1]
    print(f'  13-D (+beam):         rank={rank_13d}/13  '
          f'cond={cond_13d:.2e}   min_sv={s_13d[-1]:.2e}')

    # 18-D: everything
    print(f'  18-D (everything):    rank={rank}/18  '
          f'cond={cond:.2e}   min_sv={sigma[-1]:.2e}')

    print(f'\n--- Conclusion ---')
    if null_dim == 0:
        print('  The 18-D problem is locally identifiable (full rank Jacobian).')
        print('  All parameters can in principle be recovered.')
        print('  The practical difficulty is the extreme condition number.')
    else:
        print(f'  The 18-D problem has a {null_dim}-dimensional degeneracy.')
        print(f'  {rank} parameters are identifiable, {null_dim} are not.')
    print()
