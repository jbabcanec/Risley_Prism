#!/usr/bin/env python3
"""
Comprehensive analysis for the inverse Risley prism paper.
Addresses: noise robustness, NN ablation, Yang comparison,
extended test battery, P=4 scaling, predicted vs measured errors.
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))

from core import PrismParameters, SystemGeometry, fast_forward
from scipy.optimize import differential_evolution, minimize


# ═════════════════════════════════════════════════════════════════════
# Utility: run the full solver (DE + NM + permutation) on a pattern
# ═════════════════════════════════════════════════════════════════════
def solve(target_pattern, n_prisms, n_restarts=4, de_maxiter=200,
          de_popsize=25, nm_maxiter=10000, x0_seed=None,
          geometry=None, glass_indices=None):
    """Solve inverse problem. Returns (best_params_array, best_mse, n_evals, wall_time)."""
    T = len(target_pattern)
    n_evals = [0]

    def objective(x):
        n_evals[0] += 1
        try:
            n = n_prisms
            p = PrismParameters(n, x[:n].tolist(), x[n:2*n].tolist(), x[2*n:3*n].tolist(),
                                glass_indices=glass_indices, geometry=geometry)
            return float(np.mean((fast_forward(p, T, 10.0) - target_pattern)**2))
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
        except:
            pass

    # NM polish
    if best_x is not None:
        res_nm = minimize(objective, best_x, method='Nelder-Mead',
                          options={'maxiter': nm_maxiter, 'fatol': 1e-12, 'xatol': 1e-12})
        if res_nm.fun < best_mse:
            best_x, best_mse = res_nm.x.copy(), res_nm.fun

    # Permutation search
    from itertools import permutations
    if best_x is not None and n_prisms <= 4:
        n = n_prisms
        speeds = best_x[:n]
        ax = best_x[n:2*n]
        ay = best_x[2*n:3*n]
        for perm in permutations(range(n)):
            if list(perm) == list(range(n)):
                continue
            x_perm = np.concatenate([speeds[list(perm)], ax[list(perm)], ay[list(perm)]])
            res_p = minimize(objective, x_perm, method='Nelder-Mead',
                             options={'maxiter': 5000, 'fatol': 1e-12, 'xatol': 1e-12})
            if res_p.fun < best_mse:
                best_x, best_mse = res_p.x.copy(), res_p.fun

    wall_time = time.time() - t0
    return best_x, best_mse, n_evals[0], wall_time


def extract_errors(true_params, recovered_x, n_prisms):
    """Extract per-parameter errors. recovered_x is [speeds(n), ax(n), ay(n)]."""
    n = n_prisms
    errors = {
        'speed': [abs(recovered_x[i] - true_params.rotation_speeds[i]) for i in range(n)],
        'angle_x': [abs(recovered_x[n+i] - true_params.wedge_angles_x[i]) for i in range(n)],
        'angle_y': [abs(recovered_x[2*n+i] - true_params.wedge_angles_y[i]) for i in range(n)],
    }
    return errors


# ═════════════════════════════════════════════════════════════════════
# 1. NOISE ROBUSTNESS
# ═════════════════════════════════════════════════════════════════════
def run_noise_study():
    print("\n" + "="*70)
    print("1. NOISE ROBUSTNESS STUDY")
    print("="*70)

    true_p = PrismParameters(3, [1.5, -1.0, 2.0], [12.0, -8.0, 5.0], [3.0, 10.0, -6.0])
    clean_pattern = fast_forward(true_p, 200, 10.0)
    signal_power = np.mean(clean_pattern**2)

    snr_levels = [np.inf, 60, 40, 30, 20, 10]  # dB
    results = []

    for snr_db in snr_levels:
        if snr_db == np.inf:
            noisy = clean_pattern.copy()
            sigma = 0.0
        else:
            sigma = np.sqrt(signal_power / (10**(snr_db / 10)))
            np.random.seed(42)
            noisy = clean_pattern + np.random.normal(0, sigma, clean_pattern.shape)

        best_x, mse, n_ev, wt = solve(noisy, 3, n_restarts=3, de_maxiter=150, de_popsize=20)

        if best_x is not None:
            errs = extract_errors(true_p, best_x, 3)
            row = {
                'snr_db': snr_db if snr_db != np.inf else 'inf',
                'sigma': float(sigma),
                'mse': float(mse),
                'max_speed_err': float(max(errs['speed'])),
                'max_angx_err': float(max(errs['angle_x'])),
                'max_angy_err': float(max(errs['angle_y'])),
                'evals': n_ev,
                'time_s': round(wt, 1),
            }
        else:
            row = {'snr_db': snr_db if snr_db != np.inf else 'inf', 'failed': True}

        results.append(row)
        snr_str = str(row['snr_db'])
        print(f"  SNR={snr_str:>5s} dB  sigma={sigma:.4f}  "
              f"MSE={row.get('mse', 'FAIL'):.2e}  "
              f"|dN|={row.get('max_speed_err', 'N/A')}  "
              f"|dax|={row.get('max_angx_err', 'N/A')}  "
              f"|day|={row.get('max_angy_err', 'N/A')}  "
              f"({row.get('time_s', '?')}s)")

    return results


# ═════════════════════════════════════════════════════════════════════
# 2. NN ABLATION: NN+DE vs DE-only
# ═════════════════════════════════════════════════════════════════════
def run_ablation():
    print("\n" + "="*70)
    print("2. NN ABLATION STUDY")
    print("="*70)

    cases = [
        ("2P easy", PrismParameters(2, [1.5, -1.0], [12.0, -8.0], [3.0, 10.0])),
        ("3P easy", PrismParameters(3, [1.5, -1.0, 2.0], [12.0, -8.0, 5.0], [3.0, 10.0, -6.0])),
        ("3P hard", PrismParameters(3, [1.0, 1.1, -0.8], [14.0, -12.0, 8.0], [7.0, -5.0, 11.0])),
    ]

    results = []
    for label, true_p in cases:
        pattern = fast_forward(true_p, 200, 10.0)
        n = true_p.n_prisms

        # Simulate NN estimate: true params + noise matching NN's typical MAE
        np.random.seed(42)
        nn_x = true_p.to_flat()
        nn_seed = np.zeros(3*n)
        nn_seed[:n] = nn_x[:n] + np.random.normal(0, 1.0, n)       # ~1 Hz noise
        nn_seed[n:2*n] = nn_x[3:3+n] + np.random.normal(0, 5.0, n)  # ~5 deg noise
        nn_seed[2*n:] = nn_x[6:6+n] + np.random.normal(0, 5.0, n)

        # NN+DE
        _, mse_nn, ev_nn, t_nn = solve(pattern, n, n_restarts=3, de_maxiter=150,
                                        de_popsize=20, x0_seed=nn_seed)
        # DE-only (no seed)
        _, mse_de, ev_de, t_de = solve(pattern, n, n_restarts=3, de_maxiter=150,
                                        de_popsize=20, x0_seed=None)

        row = {
            'case': label, 'n_prisms': n,
            'nn_de_mse': float(mse_nn), 'nn_de_time': round(t_nn, 1), 'nn_de_evals': ev_nn,
            'de_only_mse': float(mse_de), 'de_only_time': round(t_de, 1), 'de_only_evals': ev_de,
            'speedup': round(t_de / max(t_nn, 0.1), 2),
        }
        results.append(row)
        print(f"  {label:10s}  NN+DE: MSE={mse_nn:.2e} ({t_nn:.0f}s)  "
              f"DE-only: MSE={mse_de:.2e} ({t_de:.0f}s)  "
              f"speedup={row['speedup']}x")

    return results


# ═════════════════════════════════════════════════════════════════════
# 3. YANG PARAXIAL COMPARISON (P=2)
# ═════════════════════════════════════════════════════════════════════
def yang_paraxial_inverse(pattern, T_total=10.0):
    """
    Yang (2008) paraxial 2-prism inverse via FFT.
    In paraxial limit: x(t) ~ sum_i alpha_i * cos(2*pi*N_i*t + phi_i)
    So FFT peaks directly give speeds and amplitudes.
    """
    T = len(pattern)
    dt = T_total / T
    freqs = np.fft.rfftfreq(T, d=dt)

    fx = np.fft.rfft(pattern[:, 0])
    fy = np.fft.rfft(pattern[:, 1])
    mag = np.abs(fx) + np.abs(fy)

    # Find top 2 peaks (excluding DC)
    mag[0] = 0
    peaks = []
    for _ in range(2):
        idx = np.argmax(mag)
        peaks.append(idx)
        # Zero out neighborhood
        lo, hi = max(1, idx-3), min(len(mag), idx+4)
        mag[lo:hi] = 0

    peaks.sort()
    recovered_speeds = [freqs[p] for p in peaks]

    # Amplitude gives wedge angle (in paraxial: amplitude ~ (n-1)*alpha*d_w)
    # We don't have a clean calibration, so just return speed estimates
    # and note that amplitude recovery requires knowing geometry
    recovered = {
        'speeds': recovered_speeds,
        'peaks_idx': peaks,
        'fx_mag': [float(np.abs(fx[p])) for p in peaks],
        'fy_mag': [float(np.abs(fy[p])) for p in peaks],
    }
    return recovered


def run_yang_comparison():
    print("\n" + "="*70)
    print("3. YANG PARAXIAL COMPARISON (P=2)")
    print("="*70)

    cases = [
        ("Small angles (3,2 deg)", PrismParameters(2, [1.5, -1.0], [3.0, -2.0], [1.0, 1.5])),
        ("Medium angles (8,6 deg)", PrismParameters(2, [1.5, -1.0], [8.0, -6.0], [3.0, 5.0])),
        ("Large angles (15,12 deg)", PrismParameters(2, [1.5, -1.0], [15.0, -12.0], [5.0, 8.0])),
        ("Close speeds (1.0,1.1)", PrismParameters(2, [1.0, 1.1], [8.0, -6.0], [3.0, 5.0])),
    ]

    results = []
    for label, true_p in cases:
        pattern = fast_forward(true_p, 200, 10.0)

        # Yang FFT method
        yang = yang_paraxial_inverse(pattern)
        yang_speed_err = [abs(yang['speeds'][i] - abs(true_p.rotation_speeds[i]))
                          for i in range(2)]
        # Note: FFT can't distinguish sign of speed, and frequency resolution is limited

        # Our method
        best_x, mse, _, wt = solve(pattern, 2, n_restarts=2, de_maxiter=100, de_popsize=15)
        our_errs = extract_errors(true_p, best_x, 2) if best_x is not None else None

        row = {
            'case': label,
            'yang_speeds': yang['speeds'],
            'yang_speed_err': yang_speed_err,
            'our_mse': float(mse) if mse else None,
            'our_speed_err': our_errs['speed'] if our_errs else None,
            'our_angx_err': our_errs['angle_x'] if our_errs else None,
            'our_time': round(wt, 1),
        }
        results.append(row)
        print(f"  {label:30s}")
        print(f"    Yang FFT speeds: {yang['speeds']}  err={yang_speed_err}")
        print(f"    Ours:  MSE={mse:.2e}  speed_err={our_errs['speed'] if our_errs else 'N/A'}  ({wt:.0f}s)")
        if our_errs:
            print(f"           angx_err={our_errs['angle_x']}  angy_err={our_errs['angle_y']}")

    return results


# ═════════════════════════════════════════════════════════════════════
# 4. EXTENDED TEST BATTERY (adversarial cases)
# ═════════════════════════════════════════════════════════════════════
def run_extended_battery():
    print("\n" + "="*70)
    print("4. EXTENDED TEST BATTERY")
    print("="*70)

    cases = [
        # Adversarial 2-prism cases
        ("2P near-identical speeds",  PrismParameters(2, [1.0, 1.05], [10.0, -8.0], [5.0, 6.0])),
        ("2P tiny angles",            PrismParameters(2, [1.5, -1.0], [1.5, -1.0], [0.5, 0.8])),
        ("2P same-sign speeds",       PrismParameters(2, [1.5, 2.0], [12.0, -8.0], [3.0, 10.0])),
        ("2P near-zero speed",        PrismParameters(2, [0.1, -1.0], [12.0, -8.0], [3.0, 10.0])),
        # Adversarial 3-prism cases
        ("3P two close speeds",       PrismParameters(3, [1.0, 1.05, -2.0], [10.0, -8.0, 5.0], [3.0, 7.0, -4.0])),
        ("3P all positive speeds",    PrismParameters(3, [0.5, 1.0, 2.0], [10.0, -8.0, 5.0], [3.0, 7.0, -4.0])),
        ("3P large angle asymmetry",  PrismParameters(3, [1.5, -1.0, 2.0], [15.0, -1.0, 15.0], [1.0, 15.0, -1.0])),
        ("3P tiny angles all",        PrismParameters(3, [1.5, -1.0, 2.0], [2.0, -1.5, 1.0], [0.5, 0.8, -0.3])),
        ("3P harmonic speeds 1:2:3",  PrismParameters(3, [1.0, 2.0, 3.0], [10.0, -8.0, 5.0], [3.0, 7.0, -4.0])),
    ]

    results = []
    for label, true_p in cases:
        pattern = fast_forward(true_p, 200, 10.0)
        n = true_p.n_prisms

        best_x, mse, n_ev, wt = solve(pattern, n, n_restarts=3, de_maxiter=200,
                                        de_popsize=25)

        if best_x is not None:
            errs = extract_errors(true_p, best_x, n)
            row = {
                'case': label, 'n_prisms': n,
                'mse': float(mse),
                'max_speed_err': float(max(errs['speed'])),
                'max_angx_err': float(max(errs['angle_x'])),
                'max_angy_err': float(max(errs['angle_y'])),
                'evals': n_ev, 'time_s': round(wt, 1),
            }
        else:
            row = {'case': label, 'n_prisms': n, 'failed': True}

        results.append(row)
        s = f"  {label:30s}  P={n}  MSE={row.get('mse','FAIL'):.2e}"
        if 'max_speed_err' in row:
            s += f"  |dN|={row['max_speed_err']:.2e}  |dax|={row['max_angx_err']:.2e}  |day|={row['max_angy_err']:.2e}"
        s += f"  ({row.get('time_s','?')}s)"
        print(s)

    return results


# ═════════════════════════════════════════════════════════════════════
# 5. P=4 SCALING TEST
# ═════════════════════════════════════════════════════════════════════
def run_p4_test():
    print("\n" + "="*70)
    print("5. P=4 SCALING TEST")
    print("="*70)

    true_p = PrismParameters(4, [1.5, -1.0, 2.0, -0.7],
                             [10.0, -8.0, 5.0, -12.0],
                             [3.0, 7.0, -4.0, 6.0],
                             glass_indices=[1.50, 1.55, 1.60, 1.65])
    pattern = fast_forward(true_p, 200, 10.0)

    print(f"  True params: N={true_p.rotation_speeds}")
    print(f"               ax={true_p.wedge_angles_x}")
    print(f"               ay={true_p.wedge_angles_y}")
    print(f"  Search dim: {3*4}=12")

    best_x, mse, n_ev, wt = solve(pattern, 4, n_restarts=5, de_maxiter=300,
                                    de_popsize=30,
                                    glass_indices=true_p.glass_indices)

    if best_x is not None:
        errs = extract_errors(true_p, best_x, 4)
        print(f"  Recovered: N={best_x[:4].tolist()}")
        print(f"             ax={best_x[4:8].tolist()}")
        print(f"             ay={best_x[8:12].tolist()}")
        print(f"  MSE={mse:.2e}  evals={n_ev}  time={wt:.0f}s")
        print(f"  Max errors: |dN|={max(errs['speed']):.2e}  "
              f"|dax|={max(errs['angle_x']):.2e}  |day|={max(errs['angle_y']):.2e}")
        return {'mse': float(mse), 'evals': n_ev, 'time_s': round(wt, 1),
                'errors': {k: [float(v) for v in vs] for k, vs in errs.items()},
                'recovered': best_x.tolist()}
    else:
        print("  FAILED")
        return {'failed': True}


# ═════════════════════════════════════════════════════════════════════
# 6. PREDICTED vs MEASURED ERRORS (Hessian validation)
# ═════════════════════════════════════════════════════════════════════
def run_error_prediction():
    print("\n" + "="*70)
    print("6. PREDICTED vs MEASURED ERRORS (Hessian validation)")
    print("="*70)

    true_p = PrismParameters(3, [1.5, -1.0, 2.0], [12.0, -8.0, 5.0], [3.0, 10.0, -6.0])
    target = fast_forward(true_p, 200, 10.0)

    # Compute Hessian diagonal numerically using compact param vector
    # compact: [N1,N2,N3, ax1,ax2,ax3, ay1,ay2,ay3]
    n = 3
    true_compact = np.array(true_p.rotation_speeds + true_p.wedge_angles_x + true_p.wedge_angles_y)
    param_names = ['N1', 'N2', 'N3', 'ax1', 'ax2', 'ax3', 'ay1', 'ay2', 'ay3']
    h = 1e-4
    H_diag = []

    def mse_from_compact(x):
        p = PrismParameters(n, x[:n].tolist(), x[n:2*n].tolist(), x[2*n:3*n].tolist(),
                            glass_indices=true_p.glass_indices)
        return float(np.mean((fast_forward(p, 200, 10.0) - target)**2))

    for j in range(9):
        xp = true_compact.copy(); xp[j] += h
        xm = true_compact.copy(); xm[j] -= h

        fp = mse_from_compact(xp)
        fm = mse_from_compact(xm)
        f0 = mse_from_compact(true_compact)

        H_jj = (fp - 2*f0 + fm) / (h**2)
        H_diag.append(float(H_jj))

    # Get actual errors from a solve
    best_x, mse, _, _ = solve(target, 3, n_restarts=3, de_maxiter=200, de_popsize=25)
    actual_errors = [abs(best_x[j] - true_compact[j]) for j in range(9)]

    # Predicted errors: |delta_j| ~ sqrt(MSE / H_jj)
    predicted_errors = []
    for j in range(9):
        if H_diag[j] > 0:
            predicted_errors.append(np.sqrt(max(mse, 1e-20) / H_diag[j]))
        else:
            predicted_errors.append(float('inf'))

    print(f"  Achieved MSE: {mse:.2e}")
    print(f"  {'Param':>6s}  {'H_jj':>12s}  {'Predicted':>12s}  {'Actual':>12s}  {'Ratio':>8s}")
    print(f"  {'-'*58}")
    for j in range(9):
        ratio = actual_errors[j] / predicted_errors[j] if predicted_errors[j] > 0 else float('nan')
        print(f"  {param_names[j]:>6s}  {H_diag[j]:12.4e}  {predicted_errors[j]:12.4e}  "
              f"{actual_errors[j]:12.4e}  {ratio:8.2f}")

    kappa = max(H_diag) / min(d for d in H_diag if d > 0)
    print(f"\n  Condition number kappa = {kappa:.2e}")

    return {
        'mse': float(mse),
        'H_diag': {param_names[j]: H_diag[j] for j in range(9)},
        'predicted': {param_names[j]: predicted_errors[j] for j in range(9)},
        'actual': {param_names[j]: actual_errors[j] for j in range(9)},
        'kappa': float(kappa),
    }


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    all_results = {}

    all_results['noise'] = run_noise_study()
    all_results['ablation'] = run_ablation()
    all_results['yang'] = run_yang_comparison()
    all_results['extended'] = run_extended_battery()
    all_results['p4'] = run_p4_test()
    all_results['error_pred'] = run_error_prediction()

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'analysis_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
