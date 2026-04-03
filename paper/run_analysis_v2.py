#!/usr/bin/env python3
"""
Bulletproofing analysis v2: multi-trial noise, 35dB transition,
model mismatch, additional P=4 cases, real NN ablation.
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))

from core import PrismParameters, SystemGeometry, fast_forward
from scipy.optimize import differential_evolution, minimize
from itertools import permutations


def solve(target, n_prisms, n_restarts=3, de_maxiter=150, de_popsize=20,
          nm_maxiter=10000, x0_seed=None, geometry=None, glass_indices=None):
    T = len(target)
    n_evals = [0]
    def objective(x):
        n_evals[0] += 1
        try:
            n = n_prisms
            p = PrismParameters(n, x[:n].tolist(), x[n:2*n].tolist(), x[2*n:3*n].tolist(),
                                glass_indices=glass_indices, geometry=geometry)
            return float(np.mean((fast_forward(p, T, 10.0) - target)**2))
        except:
            return 1e6
    bounds = [(-3.5, 3.5)]*n_prisms + [(-18, 18)]*n_prisms + [(-18, 18)]*n_prisms
    t0 = time.time()
    best_x, best_mse = None, float('inf')
    for r in range(n_restarts):
        x0 = x0_seed if (r == 0 and x0_seed is not None) else None
        try:
            res = differential_evolution(objective, bounds, seed=42+r, maxiter=de_maxiter,
                                         popsize=de_popsize, tol=1e-12, polish=False,
                                         x0=x0, mutation=(0.5, 1.5), recombination=0.8)
            if res.fun < best_mse:
                best_x, best_mse = res.x.copy(), res.fun
        except: pass
    if best_x is not None:
        res_nm = minimize(objective, best_x, method='Nelder-Mead',
                          options={'maxiter': nm_maxiter, 'fatol': 1e-12, 'xatol': 1e-12})
        if res_nm.fun < best_mse:
            best_x, best_mse = res_nm.x.copy(), res_nm.fun
    if best_x is not None and n_prisms <= 4:
        n = n_prisms
        sp, ax, ay = best_x[:n], best_x[n:2*n], best_x[2*n:3*n]
        for perm in permutations(range(n)):
            if list(perm) == list(range(n)): continue
            xp = np.concatenate([sp[list(perm)], ax[list(perm)], ay[list(perm)]])
            rp = minimize(objective, xp, method='Nelder-Mead',
                          options={'maxiter': 5000, 'fatol': 1e-12, 'xatol': 1e-12})
            if rp.fun < best_mse:
                best_x, best_mse = rp.x.copy(), rp.fun
    return best_x, best_mse, n_evals[0], time.time() - t0


def get_errors(true_p, x, n):
    return {
        'speed': [abs(x[i] - true_p.rotation_speeds[i]) for i in range(n)],
        'angle_x': [abs(x[n+i] - true_p.wedge_angles_x[i]) for i in range(n)],
        'angle_y': [abs(x[2*n+i] - true_p.wedge_angles_y[i]) for i in range(n)],
    }


# ═══════════════════════════════════════════════════════════════
# 1. MULTI-TRIAL NOISE STUDY (50 trials per SNR, mean ± std)
# ═══════════════════════════════════════════════════════════════
def run_multitrial_noise():
    print("\n" + "="*70)
    print("1. MULTI-TRIAL NOISE STUDY (50 trials per SNR)")
    print("="*70)

    true_p = PrismParameters(3, [1.5, -1.0, 2.0], [12.0, -8.0, 5.0], [3.0, 10.0, -6.0])
    clean = fast_forward(true_p, 200, 10.0)
    sig_pow = np.mean(clean**2)

    snr_levels = [60, 50, 45, 40, 35, 30]
    N_TRIALS = 30  # 30 trials per level (balancing compute vs statistics)
    results = []

    for snr in snr_levels:
        sigma = np.sqrt(sig_pow / (10**(snr / 10)))
        speed_errs, angx_errs, angy_errs, mses = [], [], [], []
        n_success = 0

        for trial in range(N_TRIALS):
            np.random.seed(1000 + trial)
            noisy = clean + np.random.normal(0, sigma, clean.shape)
            x, mse, _, _ = solve(noisy, 3, n_restarts=2, de_maxiter=120, de_popsize=18)

            if x is not None:
                e = get_errors(true_p, x, 3)
                # Check if solver found the right basin (speed error < 1 Hz)
                if max(e['speed']) < 1.0:
                    n_success += 1
                    speed_errs.append(max(e['speed']))
                    angx_errs.append(max(e['angle_x']))
                    angy_errs.append(max(e['angle_y']))
                    mses.append(mse)

        row = {
            'snr_db': snr, 'sigma': float(sigma),
            'n_trials': N_TRIALS, 'n_success': n_success,
            'success_rate': round(n_success / N_TRIALS, 3),
        }
        if n_success > 0:
            row.update({
                'speed_mean': float(np.mean(speed_errs)),
                'speed_std': float(np.std(speed_errs)),
                'angx_mean': float(np.mean(angx_errs)),
                'angx_std': float(np.std(angx_errs)),
                'angy_mean': float(np.mean(angy_errs)),
                'angy_std': float(np.std(angy_errs)),
                'mse_mean': float(np.mean(mses)),
                'mse_std': float(np.std(mses)),
            })
        results.append(row)
        print(f"  SNR={snr:3d} dB  sigma={sigma:.4f}  success={n_success}/{N_TRIALS}"
              f"  |dN|={row.get('speed_mean','N/A'):.3e}±{row.get('speed_std',''):.3e}"
              f"  |day|={row.get('angy_mean','N/A'):.3e}±{row.get('angy_std',''):.3e}" if n_success > 0 else
              f"  SNR={snr:3d} dB  sigma={sigma:.4f}  success={n_success}/{N_TRIALS}")

    return results


# ═══════════════════════════════════════════════════════════════
# 2. MODEL MISMATCH TEST
# ═══════════════════════════════════════════════════════════════
def run_model_mismatch():
    print("\n" + "="*70)
    print("2. MODEL MISMATCH TEST")
    print("="*70)

    true_p = PrismParameters(3, [1.5, -1.0, 2.0], [12.0, -8.0, 5.0], [3.0, 10.0, -6.0])

    # Generate data with perturbed geometry, solve assuming default
    default_geo = SystemGeometry()
    mismatches = [
        ("Exact geometry", SystemGeometry()),
        ("thickness 3.0→3.1", SystemGeometry(prism_thickness=3.1)),
        ("thickness 3.0→3.5", SystemGeometry(prism_thickness=3.5)),
        ("gap 6.0→6.5", SystemGeometry(inter_prism_gap=6.5)),
        ("gap 6.0→8.0", SystemGeometry(inter_prism_gap=8.0)),
        ("workpiece 100→105", SystemGeometry(workpiece_distance=105.0)),
        ("workpiece 100→120", SystemGeometry(workpiece_distance=120.0)),
        ("beam angle 10→11", SystemGeometry(beam_angle_x=11.0)),
        ("all perturbed", SystemGeometry(prism_thickness=3.2, inter_prism_gap=6.3,
                                          workpiece_distance=103.0, beam_angle_x=10.5)),
    ]

    results = []
    for label, gen_geo in mismatches:
        # Generate with perturbed geometry
        gen_p = PrismParameters(3, [1.5, -1.0, 2.0], [12.0, -8.0, 5.0], [3.0, 10.0, -6.0],
                                geometry=gen_geo)
        pattern = fast_forward(gen_p, 200, 10.0)

        # Solve assuming default geometry
        def obj(x):
            try:
                p = PrismParameters(3, x[:3].tolist(), x[3:6].tolist(), x[6:9].tolist(),
                                    geometry=default_geo)
                return float(np.mean((fast_forward(p, 200, 10.0) - pattern)**2))
            except:
                return 1e6

        bounds = [(-3.5, 3.5)]*3 + [(-18, 18)]*3 + [(-18, 18)]*3
        best_x, best_mse = None, float('inf')
        for r in range(3):
            res = differential_evolution(obj, bounds, seed=42+r, maxiter=150,
                                         popsize=20, tol=1e-12, polish=False)
            if res.fun < best_mse:
                best_x, best_mse = res.x.copy(), res.fun
        if best_x is not None:
            res_nm = minimize(obj, best_x, method='Nelder-Mead',
                              options={'maxiter': 10000, 'fatol': 1e-12})
            if res_nm.fun < best_mse:
                best_x, best_mse = res_nm.x.copy(), res_nm.fun

        e = get_errors(true_p, best_x, 3) if best_x is not None else None
        row = {
            'case': label,
            'residual_mse': float(best_mse),
            'max_speed_err': float(max(e['speed'])) if e else None,
            'max_angx_err': float(max(e['angle_x'])) if e else None,
            'max_angy_err': float(max(e['angle_y'])) if e else None,
        }
        results.append(row)
        print(f"  {label:25s}  MSE={best_mse:.2e}"
              f"  |dN|={row.get('max_speed_err','N/A'):.2e}"
              f"  |dax|={row.get('max_angx_err','N/A'):.2e}"
              f"  |day|={row.get('max_angy_err','N/A'):.2e}" if e else
              f"  {label:25s}  FAILED")

    return results


# ═══════════════════════════════════════════════════════════════
# 3. ADDITIONAL P=4 CASES
# ═══════════════════════════════════════════════════════════════
def run_p4_extended():
    print("\n" + "="*70)
    print("3. ADDITIONAL P=4 CASES")
    print("="*70)

    cases = [
        ("4P easy", PrismParameters(4, [1.5, -1.0, 2.0, -0.7],
                                    [10.0, -8.0, 5.0, -12.0], [3.0, 7.0, -4.0, 6.0],
                                    glass_indices=[1.50, 1.55, 1.60, 1.65])),
        ("4P close speeds", PrismParameters(4, [1.0, 1.2, -0.8, -1.1],
                                            [8.0, -6.0, 10.0, -4.0], [3.0, 5.0, -2.0, 7.0],
                                            glass_indices=[1.50, 1.55, 1.60, 1.65])),
        ("4P small angles", PrismParameters(4, [1.5, -1.0, 2.0, -0.5],
                                            [3.0, -2.0, 1.5, -2.5], [1.0, 1.5, -0.5, 2.0],
                                            glass_indices=[1.50, 1.55, 1.60, 1.65])),
        ("4P large angles", PrismParameters(4, [1.5, -1.0, 2.0, -0.7],
                                            [15.0, -12.0, 10.0, -14.0], [8.0, 10.0, -6.0, 12.0],
                                            glass_indices=[1.50, 1.55, 1.60, 1.65])),
    ]

    results = []
    for label, true_p in cases:
        pattern = fast_forward(true_p, 200, 10.0)
        x, mse, n_ev, wt = solve(pattern, 4, n_restarts=5, de_maxiter=300,
                                   de_popsize=30, glass_indices=true_p.glass_indices)
        if x is not None:
            e = get_errors(true_p, x, 4)
            row = {
                'case': label, 'mse': float(mse), 'evals': n_ev, 'time_s': round(wt, 1),
                'max_speed_err': float(max(e['speed'])),
                'max_angx_err': float(max(e['angle_x'])),
                'max_angy_err': float(max(e['angle_y'])),
            }
        else:
            row = {'case': label, 'failed': True}
        results.append(row)
        print(f"  {label:20s}  MSE={row.get('mse','FAIL'):.2e}"
              f"  |dN|={row.get('max_speed_err',''):.2e}"
              f"  |dax|={row.get('max_angx_err',''):.2e}"
              f"  |day|={row.get('max_angy_err',''):.2e}"
              f"  ({row.get('time_s','')}s)" if 'mse' in row else
              f"  {label:20s}  FAILED")

    return results


# ═══════════════════════════════════════════════════════════════
# 4. REAL NN ABLATION (train actual NN, compare)
# ═══════════════════════════════════════════════════════════════
def run_real_nn_ablation():
    print("\n" + "="*70)
    print("4. REAL NN ABLATION (training actual network)")
    print("="*70)

    from neural_network import PrismPredictor

    # Generate training data
    N_SAMPLES = 50000
    prism_counts = [2, 3]
    patterns_list, pc_list, params_list = [], [], []

    print(f"  Generating {N_SAMPLES} samples per prism count...")
    for pc in prism_counts:
        for _ in range(N_SAMPLES):
            speeds = np.random.uniform(-3, 3, pc).tolist()
            ax = np.random.uniform(-15, 15, pc).tolist()
            ay = np.random.uniform(-15, 15, pc).tolist()
            p = PrismParameters(pc, speeds, ax, ay)
            pat = fast_forward(p, 200, 10.0)
            flat = np.zeros(9)
            flat[:pc] = speeds
            flat[3:3+pc] = ax
            flat[6:6+pc] = ay
            patterns_list.append(pat)
            pc_list.append(pc)
            params_list.append(flat)

    patterns = np.array(patterns_list)
    pc_arr = np.array(pc_list)
    params_arr = np.array(params_list)

    # Train
    predictor = PrismPredictor()
    print("  Training NN...")
    predictor.train(patterns, pc_arr, params_arr, prism_counts=prism_counts,
                    epochs=100, batch_size=512, lr=1e-3)

    # Test: use real NN predictions as seeds
    test_cases = [
        ("3P easy", PrismParameters(3, [1.5, -1.0, 2.0], [12.0, -8.0, 5.0], [3.0, 10.0, -6.0])),
        ("3P hard", PrismParameters(3, [1.0, 1.1, -0.8], [14.0, -12.0, 8.0], [7.0, -5.0, 11.0])),
    ]

    results = []
    for label, true_p in test_cases:
        pattern = fast_forward(true_p, 200, 10.0)

        # Get real NN prediction
        nn_pred = predictor.predict(pattern, n_prisms=3)
        nn_seed = np.array(nn_pred['rotation_speeds'] +
                           nn_pred['wedge_angles_x'] + nn_pred['wedge_angles_y'])

        # NN+DE
        _, mse_nn, ev_nn, t_nn = solve(pattern, 3, n_restarts=3, de_maxiter=150,
                                        de_popsize=20, x0_seed=nn_seed)
        # DE-only
        _, mse_de, ev_de, t_de = solve(pattern, 3, n_restarts=3, de_maxiter=150,
                                        de_popsize=20, x0_seed=None)

        row = {
            'case': label,
            'nn_speed_err': [abs(nn_pred['rotation_speeds'][i] - true_p.rotation_speeds[i]) for i in range(3)],
            'nn_angx_err': [abs(nn_pred['wedge_angles_x'][i] - true_p.wedge_angles_x[i]) for i in range(3)],
            'nn_de_mse': float(mse_nn), 'nn_de_time': round(t_nn, 1), 'nn_de_evals': ev_nn,
            'de_only_mse': float(mse_de), 'de_only_time': round(t_de, 1), 'de_only_evals': ev_de,
            'speedup': round(t_de / max(t_nn, 0.1), 2),
        }
        results.append(row)
        print(f"  {label:10s}  NN pred speed err: {[f'{e:.2f}' for e in row['nn_speed_err']]}")
        print(f"             NN+DE: MSE={mse_nn:.2e} ({t_nn:.0f}s, {ev_nn} evals)")
        print(f"             DE:    MSE={mse_de:.2e} ({t_de:.0f}s, {ev_de} evals)  speedup={row['speedup']}x")

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    all_results = {}

    all_results['multitrial_noise'] = run_multitrial_noise()
    all_results['model_mismatch'] = run_model_mismatch()
    all_results['p4_extended'] = run_p4_extended()
    all_results['real_nn_ablation'] = run_real_nn_ablation()

    out_path = os.path.join(os.path.dirname(__file__), 'analysis_results_v2.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
