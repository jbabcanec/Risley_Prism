#!/usr/bin/env python3
"""
test_matrix_pencil.py -- Can matrix-pencil + integer-lattice identification
replace the FFT-peak / multi-triple brute force for speed extraction?

Method:
  1. z(t) = x(t) + i y(t): complex analytic signal. Each prism contributes at
     SIGNED frequency N_i (sign search eliminated by construction).
  2. Matrix pencil (ESPRIT) on z: frequencies + complex amplitudes far beyond
     FFT bin resolution. No windows, no bins, no phase corruption.
  3. Lattice ID: the extracted lines live on {k . N : k in Z^3}. Recover the
     three generators by greedy atom collection + small-integer consistency
     scoring (arithmetic only -- no forward-model evaluations).

Score on the standard seed-2026 30-case battery vs the FFT top-3 baseline.
Also check: arg(c) at each fundamental should equal ay_i (+180 deg if ax_i<0).

Run: python paper/test_matrix_pencil.py
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

from ml_staged_solver import N_PAR, LO, HI, RG, canon, vec2pat, T_PTS, T_OBS, FREQS

DT = T_OBS / T_PTS


# ---------------------------------------------------------------- matrix pencil
def matrix_pencil(z, dt, L=80, rtol=1e-8, max_M=60):
    """Estimate {f_m, c_m} with z(t_n) ~ sum_m c_m exp(2i pi f_m dt n)."""
    n = len(z)
    H = np.lib.stride_tricks.sliding_window_view(z, L + 1)      # (n-L, L+1)
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    M = min(int(np.sum(s > s[0] * rtol)), max_M)
    V = Vh.conj().T[:, :M]                                       # (L+1, M)
    Psi = np.linalg.lstsq(V[:-1], V[1:], rcond=None)[0]
    ev = np.linalg.eigvals(Psi)
    f = np.angle(ev) / (2 * np.pi * dt)
    A = ev[None, :] ** np.arange(n)[:, None]
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    return f, c


# ---------------------------------------------------------------- lattice ID
def _kbox(d, kmax=4):
    ax = [np.arange(-kmax, kmax + 1)] * d
    return np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, d)


def _assign(freqs, gens, kmax=4):
    """Best small-integer combo of gens for each freq. Returns K, resid."""
    KB = _kbox(len(gens), kmax)
    vals = KB @ np.asarray(gens)                                  # (nk,)
    d = np.abs(freqs[:, None] - vals[None, :])
    j = np.argmin(d, axis=1)
    return KB[j], d[np.arange(len(freqs)), j]


def lattice_id(freqs, amps, tol_add=3e-3, tol_fit=1.5e-3, kmax=4, verbose=False):
    """Recover 3 signed generators from spectral lines. Arithmetic only."""
    w = np.abs(amps)
    keep = (np.abs(freqs) > 0.02) & (w > w.max() * 1e-4)          # drop DC + floor
    f, w = freqs[keep], w[keep]
    o = np.argsort(-w)[:30]
    f, w = f[o], w[o]
    wn = w / w.sum()

    # greedy atom collection in amplitude order
    basis, extras = [], []
    for fi in f:
        if not basis:
            basis.append(fi); continue
        _, r = _assign(np.array([fi]), basis, kmax)
        if r[0] < tol_add:
            continue
        (basis if len(basis) < 3 else extras).append(fi)
    atoms = (basis + extras)[:7]
    if len(atoms) < 3:
        return None, atoms, 0.0                                    # rank-deficient

    from itertools import combinations
    best, best_g, best_K = -1e9, None, None
    for sub in combinations(range(len(atoms)), 3):
        g = np.array([atoms[i] for i in sub], float)
        K = None
        for _ in range(3):                                        # assign/refit
            K, r = _assign(f, g, kmax)
            ok = r < tol_fit
            if np.linalg.matrix_rank(K[ok]) < 3:
                break
            Wm = wn[ok, None]
            g, *_ = np.linalg.lstsq(K[ok] * Wm, f[ok] * Wm[:, 0], rcond=None)
        K, r = _assign(f, g, kmax)
        ok = r < tol_fit
        if np.linalg.matrix_rank(K[ok]) < 3:
            continue
        mass = wn[ok].sum()
        pars = (wn[ok] * (np.abs(K[ok]).sum(1) - 1)).sum()
        score = mass - 0.03 * pars
        if score > best:
            best, best_g, best_K = score, g, K
    if best_g is None:
        return None, atoms, 0.0
    order = np.argsort(-np.abs(best_g))
    if verbose:
        K, r = _assign(f, best_g, kmax)
        for i in range(min(12, len(f))):
            print(f"      f={f[i]:+8.4f}  |c|={w[i]:8.3f}  k={K[i]}  r={r[i]:.1e}")
    return best_g[order], atoms, best


# ---------------------------------------------------------------- FFT baseline
def fft_top3(pattern):
    fx = np.fft.rfft(pattern[:, 0]); fy = np.fft.rfft(pattern[:, 1])
    pw = np.abs(fx) + np.abs(fy); pw[0] = 0.0
    out = []
    for _ in range(3):
        i = int(np.argmax(pw))
        out.append(float(FREQS[i]))
        pw[max(1, i - 2):i + 3] = 0.0
    return np.sort(out)[::-1]


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- battery
if __name__ == '__main__':
    rng = np.random.default_rng(2026)
    cases = []
    for _ in range(30):
        v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
        for j in range(3):
            if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
        cases.append(canon(v))

    print(f"\n{'='*100}")
    print("MATRIX PENCIL + LATTICE ID  vs  FFT top-3   (signed speed recovery, 30 cases)")
    print(f"{'='*100}")
    print(f"{'#':>3} {'cyc':>5} {'sep':>6}  {'true speeds':>24}  {'pencil est':>24} "
          f"{'err':>8}  {'FFT':>4} {'PEN':>4}  {'phase':>6}")

    n_pen = n_fft = 0
    errs, ph_errs = [], []
    t0 = time.time()
    for ci, tc in enumerate(cases):
        pat = vec2pat(tc)
        sp = tc[:3]
        sep = min(abs(abs(sp[i]) - abs(sp[j])) for i in range(3) for j in range(i+1, 3))
        cyc = float(min(abs(sp)) * T_OBS)

        # FFT baseline: magnitudes only (it cannot see signs at all)
        fm = fft_top3(pat)
        fft_ok = np.max(np.abs(fm - np.sort(np.abs(sp))[::-1])) < 0.05

        z = pat[:, 0] + 1j * pat[:, 1]
        f, c = matrix_pencil(z, DT)
        g, atoms, score = lattice_id(f, c)

        ph_str = ""
        if g is None:
            ok, err = False, 999.0
        else:
            err = float(np.max(np.abs(g - sp)))
            ok = err < 0.02
            if ok:
                # phase check: arg(c) at fundamental == ay_i + 180*(ax_i<0)
                dphs = []
                for i in range(3):
                    j = int(np.argmin(np.abs(f - g[i])))
                    ph = np.degrees(np.angle(c[j]))
                    ph_true = tc[6 + i] + (180.0 if tc[3 + i] < 0 else 0.0)
                    dphs.append(abs(wrap(ph - ph_true)))
                ph_errs.append(max(dphs))
                ph_str = f"{max(dphs):6.2f}"
        n_pen += ok; n_fft += fft_ok
        if ok:
            errs.append(err)

        tag = "" if ok else ("  [close]" if sep < 0.08 else
                             "  [lowcyc]" if cyc < 2.5 else "  [**]")
        sp_s = " ".join(f"{v:+.3f}" for v in sp)
        g_s = " ".join(f"{v:+.3f}" for v in g) if g is not None else "  rank<3"
        print(f"{ci+1:>3} {cyc:>5.1f} {sep:>6.3f}  {sp_s:>24}  {g_s:>24} "
              f"{err:>8.1e}  {'ok' if fft_ok else '--':>4} {'OK' if ok else '--':>4}"
              f"  {ph_str:>6}{tag}", flush=True)
        if not ok and g is not None:
            print(f"      atoms: {' '.join(f'{a:+.3f}' for a in atoms)}")

    dt = time.time() - t0
    print(f"\n{'='*100}")
    print(f"  PENCIL+LATTICE (signed): {n_pen}/30      FFT top-3 (magnitude only): {n_fft}/30")
    if errs:
        print(f"  speed err among successes: median {np.median(errs):.1e}  max {np.max(errs):.1e}")
    if ph_errs:
        print(f"  phase->ay check (max over 3 prisms): median {np.median(ph_errs):.2f} deg, "
              f"max {np.max(ph_errs):.2f} deg")
    print(f"  {dt:.1f}s total ({dt/30:.2f}s/case)")
    print(f"{'='*100}")
