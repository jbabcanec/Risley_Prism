"""Matrix pencil and lattice-VarPro machinery.

The scan pattern's analytic signal z = x + iy is quasi-periodic on the
frequency lattice {k . N : k in Z^P}. This module fits that model:
amplitudes are linear, only the P generators are nonlinear (damped
Gauss-Newton). Two protections are load-bearing (each cures a documented
failure mode, see DIARY 2026-07-17):

  * RIDGE amplitudes: unregularized LS explodes canceling amplitude pairs
    on near-coincident lattice lines and the giant |c| poisons ranking;
  * MERGING of lattice lines closer than MERGE_TOL (min-|k|_1
    representative): the data cannot distinguish them, and ridge otherwise
    SPLITS large components across near-collinear columns.

`sharp=True` disables both for a final local refinement from a good basis.
"""
from itertools import product as iproduct

import numpy as np

SPEED_MAX = 3.6          # physical bound on |N| (parameter box is +-3.5)
SPEED_MIN = 0.10         # batteries clamp |N| >= 0.15; near-DC lines are junk
B_FIT = 3
B_REFINE = 4
MERGE_C = 0.12           # lines closer than MERGE_C / T_span are
                         # indistinguishable (0.012 Hz at the standard 10 s)


# ---------------------------------------------------------------- pencil
def matrix_pencil(z, dt, L=80, rtol=1e-8, max_M=60):
    """Estimate {f_m, c_m} with z(t_n) ~ sum_m c_m exp(2i pi f_m dt n)."""
    n = len(z)
    H = np.lib.stride_tricks.sliding_window_view(z, L + 1)
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    if s[0] < 1e-12:
        return np.array([]), np.array([])
    M = min(int(np.sum(s > s[0] * rtol)), max_M)
    V = Vh.conj().T[:, :M]
    Psi = np.linalg.lstsq(V[:-1], V[1:], rcond=None)[0]
    ev = np.linalg.eigvals(Psi)
    f = np.angle(ev) / (2 * np.pi * dt)
    A = ev[None, :] ** np.arange(n)[:, None]
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    return f, c


# ---------------------------------------------------------------- lattice
_KSETS = {}


def kset(ngen, B):
    key = (ngen, B)
    if key not in _KSETS:
        _KSETS[key] = np.array(
            [k for k in iproduct(range(-B, B + 1), repeat=ngen)
             if sum(abs(x) for x in k) <= B], float)
    return _KSETS[key]


def lattice_fit(z, t, mask, gens, B=B_FIT, iters=14, ridge_frac=0.05,
                sharp=False):
    """Damped GN VarPro on masked samples. Returns g, K, c, rel_resid.
    The order B is reduced if the design would be underdetermined."""
    g = np.asarray(gens, float).copy()
    zm, tm = z[mask], t[mask]
    merge_tol = MERGE_C / max(t[-1] - t[0], 1e-9)
    while B >= 1:
        K0 = kset(len(g), B)
        if len(K0) <= 0.75 * len(zm):
            break
        B -= 1
    scale = np.linalg.norm(zm - zm.mean()) + 1e-30
    lam = 1e-9 * len(zm) if sharp else (ridge_frac ** 2) * len(zm)

    def fit_amp(gv):
        vals = K0 @ gv
        if sharp:
            Kr = K0
        else:
            order = np.argsort(vals)
            reps = []
            grp = [order[0]]
            for j in order[1:]:
                if vals[j] - vals[grp[-1]] < merge_tol:
                    grp.append(j)
                else:
                    reps.append(min(grp, key=lambda q: np.abs(K0[q]).sum()))
                    grp = [j]
            reps.append(min(grp, key=lambda q: np.abs(K0[q]).sum()))
            Kr = K0[reps]
        E = np.exp(2j * np.pi * np.outer(tm, Kr @ gv))
        A = E.conj().T @ E + lam * np.eye(len(Kr))
        c = np.linalg.solve(A, E.conj().T @ zm)
        r = zm - E @ c
        return E, Kr, c, r, np.linalg.norm(r)

    E, K, c, r, res = fit_amp(g)
    if not np.isfinite(res):
        return g, K, np.zeros(len(K), complex), np.inf
    for _ in range(iters):
        J = 2j * np.pi * tm[:, None] * (E * c[None, :]) @ K
        Jr = np.concatenate([J.real, J.imag])
        rr = np.concatenate([r.real, r.imag])
        try:
            dg, *_ = np.linalg.lstsq(Jr, rr, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(dg)):
            break
        moved = False
        for _bt in range(4):
            g_try = g + dg
            E2, K2, c2, r2, res2 = fit_amp(g_try)
            if res2 < res:
                g, E, K, c, r, res = g_try, E2, K2, c2, r2, res2
                moved = True
                break
            dg = dg / 2
        if not moved or np.max(np.abs(dg)) < 1e-11:
            break
    return g, K, c, res / scale


def model_eval(t, g, K, c):
    return np.exp(2j * np.pi * np.outer(t, K @ g)) @ c


def parsimony(K, c):
    """Amplitude mass at |k|_1 >= 2 (0 = all mass on fundamentals/DC)."""
    a = np.abs(c)
    o = np.abs(K).sum(1)
    return float((a * np.maximum(0, o - 1)).sum() / (a.sum() + 1e-30))


def coverage_score(lines, w, g, fs, B=B_FIT, tol=6e-4, return_ok=False):
    """Amplitude-weighted small-integer coverage of all lines by basis g.
    Aliases f +- fs are accepted."""
    KB = kset(len(g), B)
    vals = KB @ np.asarray(g, float)
    best = np.full(len(lines), np.inf)
    orders = np.zeros(len(lines))
    for shift in (0.0, -fs, fs):
        d = np.abs((lines + shift)[:, None] - vals[None, :])
        j = np.argmin(d, axis=1)
        dj = d[np.arange(len(lines)), j]
        upd = dj < best
        best[upd] = dj[upd]
        orders[upd] = np.abs(KB[j[upd]]).sum(1)
    okm = best < tol * np.maximum(1.0, np.abs(lines))
    score = float((w * okm).sum() -
                  0.05 * (w * okm * np.maximum(0, orders - 1)).sum())
    return (score, okm) if return_ok else score


def fundamentals(K, g, c, n_gen=3):
    """Physical fundamentals: rank-extending largest-|c| lines with
    SPEED_MIN <= |freq| <= SPEED_MAX. Returns (freqs, rows, amps) or None."""
    idx = np.argsort(-np.abs(c))
    amax = np.abs(c[idx[0]]) if len(idx) else 0.0
    rows, freqs, amps = [], [], []
    for j in idx:
        k = K[j]
        if not k.any():
            continue
        if np.abs(c[j]) < 1e-5 * amax:
            break
        fq = float(K[j] @ g)
        if not (SPEED_MIN <= abs(fq) <= SPEED_MAX):
            continue
        if rows and np.linalg.matrix_rank(np.array(rows + [k])) == len(rows):
            continue
        rows.append(k); freqs.append(fq); amps.append(c[j])
        if len(rows) == n_gen:
            return np.array(freqs), rows, np.array(amps)
    return None
