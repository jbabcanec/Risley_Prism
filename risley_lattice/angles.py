"""Angles read off the spectrum.

ay_i is exactly the rotation phase offset of prism i in the forward model
(core.py builds the exit-face normal from cos/sin(gamma + ay) * tan(ax)),
so arg(c_i) at the fundamental equals ay_i, plus 180 deg iff ax_i < 0 --
an EXACT reparameterization symmetry (assumption A1), resolved by the
+-18 deg box. |c_i| is inverted for |ax_i| through a per-prism cubic gain
amp = a tan(ax) + b tan(ax)^3 calibrated with two forward evaluations.
"""
import numpy as np

from .model import N_PAR, T_PTS, T_OBS, DT, SRC, THK, vec2pat

T_GRID = np.arange(T_PTS) * DT
NG_MID, DW_MID, GAP_MID = 1.55, 125.0, 8.5
L_NOM = SRC + 3 * THK + 2 * GAP_MID + DW_MID


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def line_amp_at(pattern, f):
    """|c| of the line at frequency f by direct projection (mean removed)."""
    z = pattern[:, 0] + 1j * pattern[:, 1]
    z = z - z.mean()
    tg = np.arange(len(z)) * DT
    return np.abs(np.vdot(np.exp(2j * np.pi * f * tg), z)) / len(z)


def proj_phases(pattern, N_est):
    """Fundamental phases/amps by direct projection (fallback estimator)."""
    z = pattern[:, 0] + 1j * pattern[:, 1]
    z = z - z.mean()
    tg = np.arange(len(z)) * DT
    c = np.array([np.vdot(np.exp(2j * np.pi * f * tg), z) / len(z)
                  for f in N_est])
    return np.degrees(np.angle(c)), np.abs(c)


def calibrate_ax(N_est, fixed9):
    """Per-prism cubic gain amp = a*tan(ax)+b*tan(ax)^3 (2 fwd evals each)."""
    coefs = []
    for i in range(3):
        amps = []
        for ax_deg in (5.0, 15.0):
            v = np.zeros(N_PAR)
            v[:3] = N_est
            v[3 + i] = ax_deg
            v[9:] = fixed9
            amps.append(line_amp_at(vec2pat(v), N_est[i]))
        t1, t2 = np.tan(np.radians(5.0)), np.tan(np.radians(15.0))
        M = np.array([[t1, t1 ** 3], [t2, t2 ** 3]])
        a, b = np.linalg.solve(M, np.array(amps))
        coefs.append((a, b))
    return coefs


def invert_cubic(a, b, amp):
    """Solve a*tau + b*tau^3 = amp for tau in [0, tan(22 deg)]."""
    roots = np.roots([b, 0.0, a, -amp])
    real = [r.real for r in roots if abs(r.imag) < 1e-9 and
            -0.01 <= r.real <= np.tan(np.radians(22.0))]
    if real:
        return min(real, key=lambda r: abs(r - amp / max(a, 1e-9)))
    return float(np.clip(amp / max(a, 1e-9), 0.0, np.tan(np.radians(20.0))))


def angle_init(N_est, phases, amps, fixed9):
    """Deterministic (ax, ay) init from fundamental phases and amplitudes.
    Returns (ax, ay, list of branch-ambiguous prisms)."""
    coefs = calibrate_ax(N_est, fixed9)
    ax, ay, ambiguous = np.zeros(3), np.zeros(3), []
    for i in range(3):
        ph = wrap(phases[i])
        neg = abs(ph) > 90.0
        ay_i = wrap(ph - 180.0) if neg else ph
        mag = np.degrees(np.arctan(invert_cubic(*coefs[i], amps[i])))
        mag = min(mag, 17.5)
        ax[i] = -mag if neg else mag
        ay[i] = np.clip(ay_i, -18.0, 18.0)
        if abs(abs(ph) - 90.0) < 15.0:
            ambiguous.append(i)
    return ax, ay, ambiguous


def nominal_rest(pattern):
    """[ng x3, d_W, gap, bm_ax, bm_ay, bm_px, bm_py] init. Beam angles come
    analytically from the pattern DC: rotating first-order deflections
    average to zero, so the DC is the static ray."""
    dc = pattern.mean(0)
    bm_ax = float(np.clip(np.degrees(np.arctan(dc[0] / L_NOM)), -24.0, 24.0))
    bm_ay = float(np.clip(np.degrees(np.arctan(dc[1] / L_NOM)), -24.0, 24.0))
    return np.array([NG_MID, NG_MID, NG_MID, DW_MID, GAP_MID,
                     bm_ax, bm_ay, 0.0, 0.0])
