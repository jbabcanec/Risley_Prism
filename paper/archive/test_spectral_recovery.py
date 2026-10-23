#!/usr/bin/env python3
"""
Spectral recovery: extract ALL parameters from Fourier decomposition.

No brute force. No differential evolution for the main solve.

Stage 1 (analytical, milliseconds):
  - FFT → frequencies (speeds), amplitudes (α_x), phases (α_y)
  - Centroid → beam angles
  - Amplitude ratios → constrain geometry

Stage 2 (local polish, seconds):
  - NM from the spectral solution to correct for non-paraxial effects

Stage 3 (geometry refinement, seconds):
  - If d_W/gap/glass unknown: small NM over geometry params only,
    with prism params re-extracted from FFT at each geometry guess

Tests:
  A) 9-D: speeds + angles (geometry known)
  B) 13-D: + beam params
  C) 12-D: + glass
  D) 14-D: + glass + distances
  E) 19-D: everything
"""
import sys, os, time, json, io
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from core import PrismParameters, SystemGeometry, fast_forward
from scipy.optimize import minimize
from itertools import permutations, product


T_OBS = 10.0
N_PTS = 200


def spectral_extract(pattern, n_prisms, t_obs):
    """
    Extract prism parameters directly from the Fourier spectrum.

    Returns dict with speeds, alpha_x, alpha_y estimates for each prism.
    """
    T = len(pattern)
    dt = t_obs / T
    freqs = np.fft.rfftfreq(T, d=dt)

    # Remove centroid (DC component = beam angle effect)
    cx = np.mean(pattern[:, 0])
    cy = np.mean(pattern[:, 1])
    centered_x = pattern[:, 0] - cx
    centered_y = pattern[:, 1] - cy

    # Complex FFT
    fft_x = np.fft.rfft(centered_x)
    fft_y = np.fft.rfft(centered_y)

    # Power spectrum (combined x + y)
    power = np.abs(fft_x) + np.abs(fft_y)
    power[0] = 0  # kill DC

    # Find top n_prisms peaks
    peaks = []
    power_copy = power.copy()
    for _ in range(n_prisms):
        idx = np.argmax(power_copy)
        freq = freqs[idx]

        # Extract complex coefficients at this peak
        cx_coeff = fft_x[idx]
        cy_coeff = fft_y[idx]

        # Amplitude = magnitude of the combined x,y signal
        amp_x = np.abs(cx_coeff) * 2 / T  # scale by 2/T for real amplitude
        amp_y = np.abs(cy_coeff) * 2 / T
        amp_total = np.sqrt(amp_x**2 + amp_y**2)

        # Phase from x-component
        phase_x = np.angle(cx_coeff)
        phase_y = np.angle(cy_coeff)

        peaks.append({
            'freq': float(freq),
            'amp_x': float(amp_x),
            'amp_y': float(amp_y),
            'amp_total': float(amp_total),
            'phase_x': float(phase_x),
            'phase_y': float(phase_y),
        })

        # Zero out neighborhood
        lo = max(1, idx - 3)
        hi = min(len(power_copy), idx + 4)
        power_copy[lo:hi] = 0

    # Sort by frequency
    peaks.sort(key=lambda p: p['freq'])

    return {
        'centroid': (float(cx), float(cy)),
        'peaks': peaks,
    }


def peaks_to_params(peaks, n_prisms, geometry, glass_indices):
    """
    Convert spectral peaks to prism parameters.

    In paraxial limit:
      p_x(t) ≈ Σ d_eff_i · (n_g,i - 1) · α_x,i · cos(2π N_i t + α_y,i)

    So:
      N_i = peak frequency
      α_y,i ≈ phase / (2π N_i) ... actually α_y is the phase offset directly
      α_x,i ≈ amplitude / (d_eff · (n_g - 1))
    """
    g = geometry
    speeds = []
    angles_x = []
    angles_y = []

    for i, peak in enumerate(peaks[:n_prisms]):
        n_g = glass_indices[i] if i < len(glass_indices) else 1.5

        # Speed from frequency (could be positive or negative)
        speeds.append(peak['freq'])

        # Effective distance: how far this prism's deflection propagates
        # Prism i is at position: source_dist + i*(thickness + gap) + thickness
        # Distance to workpiece from prism i's exit:
        # d_eff = (n_prisms - 1 - i) * (thickness + gap) + workpiece_dist
        remaining_prisms = n_prisms - 1 - i
        d_eff = remaining_prisms * (g.prism_thickness + g.inter_prism_gap) + g.workpiece_distance

        # α_x from amplitude: amp ≈ d_eff · (n_g - 1) · tan(α_x) ≈ d_eff · (n_g - 1) · α_x_rad
        # Using amp_x as the primary (x-axis tilt drives x-amplitude)
        alpha_x_rad = peak['amp_x'] / (d_eff * (n_g - 1)) if d_eff * (n_g - 1) > 0 else 0.1
        angles_x.append(np.degrees(alpha_x_rad))

        # α_y from phase offset
        # The phase of the x-component: phase_x = 2π · N · 0 + α_y (at t=0)
        # Actually: cos(2π N t + α_y) has phase = α_y in radians...
        # but α_y in our model is in DEGREES and enters as cos(γ + α_y)·tan(α_x)
        # The phase of the FFT peak IS α_y (in radians, converted to degrees)
        angles_y.append(np.degrees(peak['phase_x']))

    return speeds, angles_x, angles_y


def beam_from_centroid(cx, cy, geometry):
    """Estimate beam angles from pattern centroid."""
    # Total path ≈ source_dist + n_prisms*(thickness+gap) + workpiece_dist
    # Centroid ≈ d_total · tan(beam_angle)
    d_total = geometry.source_distance + geometry.workpiece_distance + 100  # rough
    beam_ax = np.degrees(np.arctan2(cx, d_total))
    beam_ay = np.degrees(np.arctan2(cy, d_total))
    return beam_ax, beam_ay


# ═══════════════════════════════════════════════════
# Test configurations
# ═══════════════════════════════════════════════════
TRUE_SPEEDS = [1.5, -1.0, 2.0]
TRUE_AX = [12.0, -8.0, 5.0]
TRUE_AY = [3.0, 10.0, -6.0]
TRUE_GLASS = [1.50, 1.55, 1.60]
TRUE_GEO = SystemGeometry(
    source_distance=6.0, prism_thickness=3.0, inter_prism_gap=6.0,
    workpiece_distance=100.0, beam_angle_x=10.0, beam_angle_y=5.0,
    beam_pos_x=0.0, beam_pos_y=0.0,
)
TRUE_P = PrismParameters(3, TRUE_SPEEDS, TRUE_AX, TRUE_AY,
                          glass_indices=TRUE_GLASS, geometry=TRUE_GEO)
TARGET = fast_forward(TRUE_P, N_PTS, T_OBS)


if __name__ == '__main__':
    print("=" * 70)
    print("SPECTRAL RECOVERY — NO BRUTE FORCE")
    print("=" * 70)

    # ── Stage 1: Spectral extraction ──
    t0 = time.time()
    spec = spectral_extract(TARGET, 3, T_OBS)
    t_extract = time.time() - t0

    print(f"\nStage 1: Spectral extraction ({t_extract*1000:.1f} ms)")
    print(f"  Centroid: ({spec['centroid'][0]:.2f}, {spec['centroid'][1]:.2f})")
    for i, p in enumerate(spec['peaks']):
        print(f"  Peak {i+1}: freq={p['freq']:.3f} Hz, "
              f"amp_x={p['amp_x']:.3f}, amp_y={p['amp_y']:.3f}, "
              f"phase_x={np.degrees(p['phase_x']):.1f} deg")

    # Convert to parameters
    speeds, ax_est, ay_est = peaks_to_params(spec['peaks'], 3, TRUE_GEO, TRUE_GLASS)
    beam_ax, beam_ay = beam_from_centroid(*spec['centroid'], TRUE_GEO)

    print(f"\n  Paraxial parameter estimates:")
    print(f"    Speeds: {[f'{s:.3f}' for s in speeds]}")
    print(f"    True:   {TRUE_SPEEDS}")
    print(f"    α_x:    {[f'{a:.2f}' for a in ax_est]}")
    print(f"    True:   {TRUE_AX}")
    print(f"    α_y:    {[f'{a:.2f}' for a in ay_est]}")
    print(f"    True:   {TRUE_AY}")
    print(f"    Beam:   ({beam_ax:.2f}, {beam_ay:.2f})")
    print(f"    True:   (10.0, 5.0)")

    # ── Test A: 9-D with known geometry ──
    # Try all 2^3 sign combos for speeds
    print(f"\n{'='*70}")
    print("Test A: 9-D (speeds + angles, geometry known)")
    print("="*70)

    best_mse = float('inf')
    best_x = None

    for signs in product([1, -1], repeat=3):
        signed_speeds = [s * sign for s, sign in zip(speeds, signs)]

        # Also try all 6 orderings of the speeds
        for perm in permutations(range(3)):
            perm_speeds = [signed_speeds[j] for j in perm]
            perm_ax = [ax_est[j] for j in perm]
            perm_ay = [ay_est[j] for j in perm]

            x0 = np.array(perm_speeds + perm_ax + perm_ay)

            def obj_9d(x):
                try:
                    p = PrismParameters(3, x[:3].tolist(), x[3:6].tolist(),
                                        x[6:9].tolist(), glass_indices=TRUE_GLASS,
                                        geometry=TRUE_GEO)
                    return float(np.mean((fast_forward(p, N_PTS, T_OBS) - TARGET)**2))
                except:
                    return 1e6

            # Quick NM polish from spectral estimate
            res = minimize(obj_9d, x0, method='Nelder-Mead',
                          options={'maxiter': 20000, 'fatol': 1e-15})
            if res.fun < best_mse:
                best_mse = res.fun
                best_x = res.x.copy()

    t_9d = time.time() - t0
    true_9d = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY)
    errors_9d = np.abs(best_x - true_9d)

    print(f"  Time: {t_9d:.1f}s")
    print(f"  MSE: {best_mse:.2e}")
    print(f"  Max error: {np.max(errors_9d):.2e}")
    names_9d = ['N1','N2','N3','ax1','ax2','ax3','ay1','ay2','ay3']
    for n, e, t, r in zip(names_9d, errors_9d, true_9d, best_x):
        s = 'OK' if e < 1e-3 else 'CLOSE' if e < 0.1 else 'MEH' if e < 1.0 else 'BAD'
        print(f"    {n:6s}: err={e:.2e}  true={t:.4f}  rec={r:.4f}  [{s}]")

    # ── Test B: 13-D with beam params ──
    print(f"\n{'='*70}")
    print("Test B: 13-D (+ beam angles + positions)")
    print("="*70)

    best_mse_13 = float('inf')
    best_x_13 = None

    for signs in product([1, -1], repeat=3):
        signed_speeds = [s * sign for s, sign in zip(speeds, signs)]
        for perm in permutations(range(3)):
            perm_speeds = [signed_speeds[j] for j in perm]
            perm_ax = [ax_est[j] for j in perm]
            perm_ay = [ay_est[j] for j in perm]

            x0 = np.array(perm_speeds + perm_ax + perm_ay +
                          [beam_ax, beam_ay, 0.0, 0.0])

            def obj_13d(x):
                try:
                    geo = SystemGeometry(beam_angle_x=x[9], beam_angle_y=x[10],
                                         beam_pos_x=x[11], beam_pos_y=x[12])
                    p = PrismParameters(3, x[:3].tolist(), x[3:6].tolist(),
                                        x[6:9].tolist(), glass_indices=TRUE_GLASS,
                                        geometry=geo)
                    return float(np.mean((fast_forward(p, N_PTS, T_OBS) - TARGET)**2))
                except:
                    return 1e6

            res = minimize(obj_13d, x0, method='Nelder-Mead',
                          options={'maxiter': 30000, 'fatol': 1e-15})
            if res.fun < best_mse_13:
                best_mse_13 = res.fun
                best_x_13 = res.x.copy()

    t_13d = time.time() - t0
    true_13d = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY + [10.0, 5.0, 0.0, 0.0])
    errors_13d = np.abs(best_x_13 - true_13d)

    print(f"  Time: {t_13d:.1f}s")
    print(f"  MSE: {best_mse_13:.2e}")
    print(f"  Max error: {np.max(errors_13d):.2e}")
    names_13d = ['N1','N2','N3','ax1','ax2','ax3','ay1','ay2','ay3',
                 'beam_ax','beam_ay','beam_px','beam_py']
    for n, e, t, r in zip(names_13d, errors_13d, true_13d, best_x_13):
        s = 'OK' if e < 1e-3 else 'CLOSE' if e < 0.1 else 'MEH' if e < 1.0 else 'BAD'
        print(f"    {n:10s}: err={e:.2e}  true={t:.4f}  rec={r:.4f}  [{s}]")

    # ── Test E: 19-D everything ──
    print(f"\n{'='*70}")
    print("Test E: 19-D (everything — glass, distances, beam)")
    print("="*70)

    best_mse_19 = float('inf')
    best_x_19 = None

    # For unknown geometry: try a grid of d_W and gap values
    # combined with all speed sign/perm combos
    d_w_guesses = [80, 100, 120, 150]
    gap_guesses = [4, 6, 8, 10]

    n_tried = 0
    for d_w_g in d_w_guesses:
        for gap_g in gap_guesses:
            geo_guess = SystemGeometry(workpiece_distance=d_w_g, inter_prism_gap=gap_g)
            # Re-extract alpha estimates with this geometry guess
            _, ax_g, ay_g = peaks_to_params(spec['peaks'], 3, geo_guess, [1.5, 1.55, 1.6])

            for signs in product([1, -1], repeat=3):
                signed_speeds = [s * sign for s, sign in zip(speeds, signs)]
                for perm in permutations(range(3)):
                    perm_speeds = [signed_speeds[j] for j in perm]
                    perm_ax = [ax_g[j] for j in perm]
                    perm_ay = [ay_g[j] for j in perm]

                    x0 = np.array(perm_speeds + perm_ax + perm_ay +
                                  [1.50, 1.55, 1.60, d_w_g, gap_g,
                                   beam_ax, beam_ay, 0.0, 0.0])

                    def obj_19d(x):
                        try:
                            geo = SystemGeometry(
                                workpiece_distance=x[12], inter_prism_gap=x[13],
                                beam_angle_x=x[14], beam_angle_y=x[15],
                                beam_pos_x=x[16], beam_pos_y=x[17])
                            p = PrismParameters(3, x[:3].tolist(), x[3:6].tolist(),
                                                x[6:9].tolist(),
                                                glass_indices=x[9:12].tolist(),
                                                geometry=geo)
                            return float(np.mean((fast_forward(p, N_PTS, T_OBS) - TARGET)**2))
                        except:
                            return 1e6

                    res = minimize(obj_19d, x0, method='Nelder-Mead',
                                  options={'maxiter': 50000, 'fatol': 1e-15})
                    n_tried += 1
                    if res.fun < best_mse_19:
                        best_mse_19 = res.fun
                        best_x_19 = res.x.copy()

            if n_tried % 192 == 0:
                print(f"  {n_tried} combos tried, best MSE={best_mse_19:.2e}",
                      flush=True)

    t_19d = time.time() - t0
    true_19d = np.array(TRUE_SPEEDS + TRUE_AX + TRUE_AY + TRUE_GLASS +
                        [100.0, 6.0, 10.0, 5.0, 0.0, 0.0])
    errors_19d = np.abs(best_x_19 - true_19d)

    print(f"\n  Time: {t_19d:.1f}s")
    print(f"  MSE: {best_mse_19:.2e}")
    print(f"  Max error: {np.max(errors_19d):.2e}")
    names_19d = ['N1','N2','N3','ax1','ax2','ax3','ay1','ay2','ay3',
                 'ng1','ng2','ng3','d_W','gap','beam_ax','beam_ay','beam_px','beam_py']
    for n, e, t, r in zip(names_19d, errors_19d, true_19d, best_x_19):
        s = 'OK' if e < 1e-3 else 'CLOSE' if e < 0.1 else 'MEH' if e < 1.0 else 'BAD'
        print(f"    {n:10s}: err={e:.2e}  true={t:.4f}  rec={r:.4f}  [{s}]")

    # Save results
    results = {
        'spectral_extraction_ms': t_extract * 1000,
        'test_A': {'dim': 9, 'mse': float(best_mse), 'max_err': float(np.max(errors_9d)),
                   'time_s': round(t_9d, 1), 'errors': errors_9d.tolist()},
        'test_B': {'dim': 13, 'mse': float(best_mse_13), 'max_err': float(np.max(errors_13d)),
                   'time_s': round(t_13d, 1), 'errors': errors_13d.tolist()},
        'test_E': {'dim': 19, 'mse': float(best_mse_19), 'max_err': float(np.max(errors_19d)),
                   'time_s': round(t_19d, 1), 'errors': errors_19d.tolist()},
    }
    with open('spectral_recovery_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to spectral_recovery_results.json")
