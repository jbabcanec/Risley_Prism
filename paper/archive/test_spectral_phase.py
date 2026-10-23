#!/usr/bin/env python3
"""
Test spectral phase extraction with interpolated frequencies.

Problem: FFT grid is 0.1 Hz. True speeds are off-grid, corrupting phases.
Fix: Parabolic interpolation → exact frequency → DFT at exact freq.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

from ml_staged_solver import N_PAR, LO, HI, RG, canon, T_PTS, T_OBS, P, SRC, THK
from solve_preconditioned import vec2pat


def interpolated_dft(pattern, n_prisms=3):
    """
    Extract frequencies, phases, and amplitudes using interpolated DFT.

    1. Standard FFT to find peak bins
    2. Parabolic interpolation for true peak frequency
    3. DFT at exact interpolated frequency for correct phase
    """
    T = len(pattern)
    dt = T_OBS / T
    freqs = np.fft.rfftfreq(T, d=dt)
    t_arr = np.arange(T) * dt  # time array

    fx = np.fft.rfft(pattern[:, 0])
    fy = np.fft.rfft(pattern[:, 1])
    power = np.abs(fx) + np.abs(fy)
    power[0] = 0

    results = []
    pw = power.copy()

    for _ in range(n_prisms):
        k = np.argmax(pw)
        if pw[k] <= 0:
            break

        # Parabolic interpolation for true peak frequency
        if 1 < k < len(pw) - 1:
            alpha = float(pw[k-1])
            beta = float(pw[k])
            gamma = float(pw[k+1])
            denom = alpha - 2*beta + gamma
            if abs(denom) > 1e-10:
                delta = 0.5 * (alpha - gamma) / denom
            else:
                delta = 0.0
        else:
            delta = 0.0

        f_interp = freqs[k] + delta * (freqs[1] - freqs[0])

        # DFT at exact interpolated frequency
        # C(f) = Σ x(t) * exp(-2πi f t)
        basis = np.exp(-2j * np.pi * f_interp * t_arr)
        Cx = np.sum(pattern[:, 0] * basis)
        Cy = np.sum(pattern[:, 1] * basis)

        amp_x = abs(Cx) * 2 / T
        amp_y = abs(Cy) * 2 / T
        phase_x = np.angle(Cx)
        phase_y = np.angle(Cy)

        results.append({
            'freq_bin': float(freqs[k]),
            'freq_interp': float(f_interp),
            'delta': float(delta),
            'amp_x': float(amp_x),
            'amp_y': float(amp_y),
            'phase_x': float(phase_x),
            'phase_y': float(phase_y),
        })

        # Zero out neighborhood
        lo = max(1, k - 2)
        hi = min(len(pw), k + 3)
        pw[lo:hi] = 0

    # Sort by frequency descending (canonical ordering)
    results.sort(key=lambda r: r['freq_interp'], reverse=True)
    return results


# Generate 30 random cases
rng = np.random.default_rng(2026)
cases = []
for i in range(30):
    v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))


print(f"{'='*90}")
print("INTERPOLATED DFT PHASE EXTRACTION — 30 cases")
print(f"{'='*90}")

freq_errs = []
phase_errs = []
ax_errs = []

for ci, tc in enumerate(cases[:10]):  # detailed on first 10
    pat = vec2pat(tc)
    peaks = interpolated_dft(pat)

    true_speeds = tc[:3]  # sorted by |N| descending
    true_ax = tc[3:6]
    true_ay = tc[6:9]
    true_ng = tc[9:12]
    true_dw = tc[12]
    true_gap = tc[13]

    print(f"\nCase {ci+1}: |speeds|={np.abs(true_speeds)}")

    for i, pk in enumerate(peaks):
        f_true = abs(true_speeds[i])
        f_est = pk['freq_interp']
        f_err = abs(f_est - f_true)
        freq_errs.append(f_err)

        # Phase → αᵧ (depends on sign of speed)
        # cos(2π N t + αᵧ_rad) where N = signed speed
        # If N > 0: FFT phase = αᵧ_rad
        # If N < 0: FFT phase = -αᵧ_rad (because cos(-ωt + φ) = cos(ωt - φ))
        # But we're computing DFT at |N| (positive freq), so:
        # Signal: A*cos(2π*sign*|N|*t + αᵧ_rad)
        # DFT at +|N|: phase = αᵧ_rad if sign > 0, phase = -αᵧ_rad if sign < 0
        # We don't know the sign yet, so we get ±αᵧ
        ay_true = true_ay[i]
        ay_est_pos = np.degrees(pk['phase_x'])
        ay_est_neg = -np.degrees(pk['phase_x'])

        # Also try atan2(phase_y, phase_x) approach
        # The combined (x,y) FFT: Cx + jCy = A * exp(j*αᵧ_rad) * stuff
        # Actually in 2D: x = A*cos(γ), y = A*sin(γ) where γ = 2πNt + αᵧ
        # So FFT_x at |N| ≈ (A/2)*exp(j*αᵧ) and FFT_y at |N| ≈ (A/2)*exp(j*(αᵧ-π/2))
        # Hmm, let me just check both approaches
        ay_from_atan2 = np.degrees(np.arctan2(pk['phase_y'], pk['phase_x']))

        err_pos = min(abs(ay_est_pos - ay_true), abs(ay_est_pos - ay_true + 360),
                     abs(ay_est_pos - ay_true - 360))
        err_neg = min(abs(ay_est_neg - ay_true), abs(ay_est_neg - ay_true + 360),
                     abs(ay_est_neg - ay_true - 360))
        best_phase_err = min(err_pos, err_neg)
        phase_errs.append(best_phase_err)

        # αₓ from amplitude
        d_eff = (P-1-i) * (THK + true_gap) + true_dw
        ng = true_ng[i]
        ax_est = np.degrees(np.arctan(pk['amp_x'] / (d_eff * (ng - 1.0))))
        ax_err = abs(ax_est) - abs(true_ax[i])
        ax_errs.append(abs(ax_err))

        sign_char = '+' if best_phase_err == err_pos else '-'
        print(f"  P{i+1}: f_true={f_true:.3f} f_est={f_est:.3f} (err={f_err:.4f})  "
              f"|αₓ|_est={ax_est:.2f} true={abs(true_ax[i]):.2f} (err={abs(ax_err):.2f})  "
              f"αᵧ_est={ay_est_pos:.1f}/{ay_est_neg:.1f}({sign_char}) true={ay_true:.1f} "
              f"(err={best_phase_err:.1f}°)")

# Run the rest silently
for ci in range(10, 30):
    tc = cases[ci]
    pat = vec2pat(tc)
    peaks = interpolated_dft(pat)
    true_speeds = tc[:3]
    for i, pk in enumerate(peaks):
        f_true = abs(true_speeds[i])
        f_err = abs(pk['freq_interp'] - f_true)
        freq_errs.append(f_err)
        ay_true = tc[6+i]
        ay_pos = np.degrees(pk['phase_x'])
        ay_neg = -ay_pos
        err_pos = min(abs(ay_pos-ay_true), abs(ay_pos-ay_true+360), abs(ay_pos-ay_true-360))
        err_neg = min(abs(ay_neg-ay_true), abs(ay_neg-ay_true+360), abs(ay_neg-ay_true-360))
        phase_errs.append(min(err_pos, err_neg))
        d_eff = (P-1-i)*(THK+tc[13])+tc[12]
        ng = tc[9+i]
        ax_est = np.degrees(np.arctan(pk['amp_x']/(d_eff*(ng-1.0)+1e-10)))
        ax_errs.append(abs(abs(ax_est)-abs(tc[3+i])))

freq_errs = np.array(freq_errs)
phase_errs = np.array(phase_errs)
ax_errs = np.array(ax_errs)

print(f"\n{'='*90}")
print(f"SUMMARY (90 extractions: 30 cases × 3 prisms)")
print(f"  Frequency error: median={np.median(freq_errs):.4f} Hz, "
      f"max={np.max(freq_errs):.4f}, <0.01: {np.sum(freq_errs<0.01)}/90")
print(f"  Phase → αᵧ err:  median={np.median(phase_errs):.1f}°, "
      f"max={np.max(phase_errs):.1f}°, <5°: {np.sum(phase_errs<5)}/90, "
      f"<10°: {np.sum(phase_errs<10)}/90")
print(f"  Amp → |αₓ| err:  median={np.median(ax_errs):.2f}°, "
      f"max={np.max(ax_errs):.2f}°, <1°: {np.sum(ax_errs<1)}/90, "
      f"<2°: {np.sum(ax_errs<2)}/90")
print(f"{'='*90}")
