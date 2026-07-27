#!/usr/bin/env python3
"""
run_experiments_v3.py — Run all experiments for the paper rewrite.

Usage:
  python paper/run_experiments_v3.py --battery     # 50-case random battery
  python paper/run_experiments_v3.py --error       # Jacobian error analysis
  python paper/run_experiments_v3.py --all         # everything
"""

import sys, os, time, json, argparse, io
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'reverse_problem_v2'))
sys.path.insert(0, HERE)
# Note: do NOT wrap sys.stdout here — breaks redirected output

import torch

from ml_staged_solver import (
    DiffFwd, AngleNet, RemainNet, DEVICE, P, T_PTS, T_OBS, SRC, THK,
    N_PAR, NAMES, LO, HI, RG, ANG_MODEL, REM_MODEL,
    extract_speeds_and_peaks, _build_peak_feats_single, canon,
)
from solve_preconditioned import solve, vec2pat, compute_jacobian_svd, ml_init


def load_models():
    ang_net = AngleNet()
    rem_net = RemainNet()
    ang_net.load_state_dict(torch.load(ANG_MODEL, map_location=DEVICE, weights_only=True))
    rem_net.load_state_dict(torch.load(REM_MODEL, map_location=DEVICE, weights_only=True))
    ang_net.to(DEVICE); rem_net.to(DEVICE)
    return ang_net, rem_net


def random_params(rng):
    """Generate a random parameter vector within bounds."""
    v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15:
            v[j] = np.copysign(0.15, v[j])
    return v


# ================================================================
#  Experiment 1: Random Battery
# ================================================================

def run_battery(n_cases=50, seed=2026):
    print('=' * 70)
    print(f' RANDOM BATTERY ({n_cases} cases)')
    print('=' * 70)

    ang_net, rem_net = load_models()
    rng = np.random.default_rng(seed)
    results = []

    for i in range(n_cases):
        v = random_params(rng)
        tc = canon(v)
        try:
            pat = vec2pat(tc)
            t0 = time.time()
            solved, mse, details = solve(ang_net, rem_net, pat,
                                         verbose=False, return_details=True)
            dt = time.time() - t0

            errs = np.abs(solved - tc)
            max_err = float(np.max(errs))
            per_param = {NAMES[j]: float(errs[j]) for j in range(N_PAR)}
            tag = 'PERFECT' if max_err < 1e-3 else 'CLOSE' if max_err < 0.01 else 'FAIL'

            result = {
                'case': i + 1,
                'max_err': max_err,
                'pat_mse': float(mse),
                'time': dt,
                'tag': tag,
                'per_param_err': per_param,
                'true_params': tc.tolist(),
                'solved_params': solved.tolist(),
                **details,
            }
            results.append(result)
            print(f'  Case {i+1:3d}/{n_cases}: max_err={max_err:.2e}  '
                  f'MSE={mse:.2e}  {dt:.0f}s  {tag}', flush=True)
        except Exception as e:
            results.append({'case': i + 1, 'tag': 'ERROR', 'error': str(e)})
            print(f'  Case {i+1:3d}/{n_cases}: ERROR ({e})', flush=True)

    # Summary statistics
    good = [r for r in results if r['tag'] in ('PERFECT', 'CLOSE')]
    n_perfect = sum(1 for r in results if r['tag'] == 'PERFECT')
    n_close = sum(1 for r in results if r['tag'] == 'CLOSE')

    print(f'\n--- Summary ---')
    print(f'  PERFECT (<1e-3): {n_perfect}/{n_cases}')
    print(f'  CLOSE (<0.01):   {n_close}/{n_cases}')
    if good:
        max_errs = [r['max_err'] for r in good]
        times = [r['time'] for r in good]
        print(f'  Max error:  median={np.median(max_errs):.2e}  '
              f'max={np.max(max_errs):.2e}')
        print(f'  Time:       median={np.median(times):.0f}s  '
              f'mean={np.mean(times):.0f}s')

        # Per-parameter statistics
        print(f'\n  Per-parameter median errors:')
        for j in range(N_PAR):
            errs_j = [r['per_param_err'][NAMES[j]] for r in good]
            print(f'    {NAMES[j]:8s}: median={np.median(errs_j):.2e}  '
                  f'max={np.max(errs_j):.2e}')

        # Timing breakdown
        if 'timings' in good[0]:
            print(f'\n  Timing breakdown (mean):')
            for phase in ['phase1_signs', 'phase2_adam', 'phase3_trf']:
                vals = [r['timings'][phase] for r in good if 'timings' in r]
                print(f'    {phase:16s}: {np.mean(vals):.1f}s')

    return results


# ================================================================
#  Experiment 2: Jacobian Error Analysis
# ================================================================

def run_error_analysis(n_cases=5, seed=2026):
    print('=' * 70)
    print(f' JACOBIAN ERROR ANALYSIS ({n_cases} cases)')
    print('=' * 70)

    ang_net, rem_net = load_models()
    fwd = DiffFwd().to(DEVICE)
    rng = np.random.default_rng(seed)
    results = []

    for i in range(n_cases):
        if i == 0:
            # Paper test case
            tv = np.array([1.5,-1.,2., 12.,-8.,5., 3.,10.,-6.,
                           1.5,1.55,1.6, 100.,6., 10.,5., 0.,0.], np.float64)
            tc = canon(tv)
        else:
            tc = canon(random_params(rng))

        pat = vec2pat(tc)
        solved, mse, details = solve(ang_net, rem_net, pat,
                                     verbose=False, return_details=True)

        # Compute Jacobian at the solved point
        J, U, sigma, V = compute_jacobian_svd(solved, fwd)
        cond = float(sigma[0] / max(sigma[-1], 1e-15))

        # Residual
        residual = vec2pat(solved).reshape(-1) - pat.reshape(-1)
        r_norm = float(np.linalg.norm(residual))

        # Predicted error bound: ||delta_theta|| <= ||r|| / sigma_min
        predicted_max = r_norm / max(sigma[-1], 1e-15)

        # Actual errors
        actual_errs = np.abs(solved - tc)
        actual_max = float(np.max(actual_errs))

        # Per-parameter: component-wise bound from J^+ r
        Jplus_r = V @ (np.diag(1.0 / (sigma + 1e-30)) @ (U.T @ residual))
        predicted_per_param = np.abs(Jplus_r)

        result = {
            'case': i,
            'cond_number': cond,
            'sigma_min': float(sigma[-1]),
            'sigma_max': float(sigma[0]),
            'singular_values': sigma.tolist(),
            'residual_norm': r_norm,
            'predicted_max_err': float(predicted_max),
            'actual_max_err': actual_max,
            'per_param': {
                NAMES[j]: {
                    'sigma_col_norm': float(np.linalg.norm(J[:, j])),
                    'predicted': float(predicted_per_param[j]),
                    'actual': float(actual_errs[j]),
                } for j in range(N_PAR)
            },
        }
        results.append(result)

        print(f'\n  Case {i}: κ={cond:.2e}  ||r||={r_norm:.2e}', flush=True)
        print(f'    Predicted max err: {predicted_max:.2e}', flush=True)
        print(f'    Actual max err:    {actual_max:.2e}', flush=True)
        print(f'    σ range: [{sigma[-1]:.2e}, {sigma[0]:.2e}]', flush=True)
        print(f'    {"Param":8s} {"σ_col":>10s} {"Predicted":>12s} {"Actual":>12s}')
        for j in range(N_PAR):
            print(f'    {NAMES[j]:8s} {np.linalg.norm(J[:,j]):10.2e} '
                  f'{predicted_per_param[j]:12.2e} {actual_errs[j]:12.2e}', flush=True)

    return results


# ================================================================
#  Main
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--battery', action='store_true')
    parser.add_argument('--error', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--n', type=int, default=50, help='Number of battery cases')
    args = parser.parse_args()

    if args.all:
        args.battery = args.error = True

    all_results = {}
    out_path = os.path.join(HERE, 'experiments_v3_results.json')

    if args.battery:
        all_results['battery'] = run_battery(n_cases=args.n)

    if args.error:
        all_results['error_analysis'] = run_error_analysis()

    if all_results:
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nResults saved to {out_path}')
    else:
        print('No experiments selected. Use --battery, --error, or --all')
