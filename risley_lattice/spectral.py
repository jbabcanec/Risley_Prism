"""Signed speed extraction: de-glitch -> CLEAN line growth -> integer-
coverage basis selection -> lattice VarPro -> canonicalization.

The public entry point is extract_speeds(pattern, dt, n_gen). Signs come
free (complex analytic signal). The phase of the fundamental line is ay_i
(+180 deg iff ax_i < 0); |c| feeds ax recovery downstream.

Design notes (each choice cures a battery-documented failure mode):
  * CLEAN grows lines at order B=1, where a line explains only itself, so a
    compromise/subharmonic basis cannot absorb foreign lines (the failure of
    residual-guided greedy at B=3 is structural: a finer lattice always fits
    at least as well);
  * novelty preference keeps harmonics from exhausting the line slots before
    a weak fundamental appears;
  * basis selection is by ARITHMETIC coverage, immune to fit pathologies;
    the fit score only referees candidates within a small coverage band;
  * one fundamentals re-extraction happens on the chosen fit (it can rescue
    a mis-selected generator); never after refinement fits (re-extraction
    from refits grabs ridge-splitting twins);
  * the +-e_i amplitude comparison is a physical sign test (the conjugate
    leak is weaker than the main line, A2).
"""
from itertools import combinations

import numpy as np

from .lattice import (SPEED_MAX, SPEED_MIN, B_FIT, B_REFINE, matrix_pencil,
                      lattice_fit, model_eval, parsimony, coverage_score,
                      fundamentals)

MAX_LINES = 8


# ---------------------------------------------------------------- de-glitch
def deglitch_mask(pattern):
    d = np.abs(np.diff(pattern, axis=0)).sum(1)
    med = np.median(d)
    bad = d > max(10 * med, 5.0)
    mask = np.ones(len(pattern), bool)
    for i in np.where(bad)[0]:
        mask[max(0, i - 1):i + 3] = False
    if mask.sum() < 0.85 * len(mask):
        return np.ones(len(pattern), bool)
    return mask


def interp_masked(z, t, mask):
    if mask.all():
        return z
    zc = z.copy()
    zc[~mask] = np.interp(t[~mask], t[mask], z.real[mask]) \
        + 1j * np.interp(t[~mask], t[mask], z.imag[mask])
    return zc


# ---------------------------------------------------------------- CLEAN
def is_novel(fj, lines, tol=0.02):
    """Not a small-integer combo (|a|+|b|<=3, aliases included) of knowns."""
    if not lines:
        return True
    L = np.asarray(lines)
    combos = [a * L[i] + b * L[j]
              for i in range(len(L)) for j in range(i, len(L))
              for a in (-3, -2, -1, 0, 1, 2, 3)
              for b in (-3, -2, -1, 0, 1, 2, 3)
              if 0 < abs(a) + abs(b) <= 3]
    combos = np.asarray(combos)
    d = np.abs(combos - fj)
    d = np.minimum(d, np.abs(combos - fj - 20.0))
    d = np.minimum(d, np.abs(combos - fj + 20.0))
    return float(d.min()) > tol


def fft_lines(zr, dt):
    """Windowed-FFT peak candidates: coarse but unbiased at strong maxima
    and capacity-free -- the fallback line source when a dense comb
    spectrum overloads the pencil's model order. Parabolic interpolation
    on log|Z| refines each peak below bin resolution; CLEAN's joint GN
    supplies the final precision."""
    n = len(zr)
    w = np.hanning(n)
    Z = np.fft.fft(w * (zr - zr.mean()))
    fr = np.fft.fftfreq(n, dt)
    A = np.abs(Z)
    pk = []
    for i in range(1, n - 1):
        if A[i] > A[i - 1] and A[i] >= A[i + 1] and A[i] > 0.02 * A.max():
            lm, l0, lp = np.log(A[i - 1] + 1e-30), np.log(A[i] + 1e-30), \
                np.log(A[i + 1] + 1e-30)
            den = lm - 2 * l0 + lp
            d = 0.5 * (lm - lp) / den if abs(den) > 1e-12 else 0.0
            pk.append((fr[i] + d * (fr[1] - fr[0]), A[i]))
    pk.sort(key=lambda p: -p[1])
    pk = pk[:40]
    return (np.array([p[0] for p in pk]),
            np.array([p[1] for p in pk], complex))


def clean_lines(z, zc, t, mask, dt, max_lines=MAX_LINES, line_source=None):
    """Grow a B=1 line model: strongest candidate line of the residual
    (novel lines first), joint refit of all line frequencies. Returns
    freqs, amps, resid. line_source(resid, dt) -> (f, c) defaults to the
    matrix pencil."""
    source = line_source or matrix_pencil
    lines, res = [], 1.0
    g = K = c = None
    for _ in range(max_lines):
        resid_clean = zc - (model_eval(t, g, K, c) if g is not None else
                            np.full(len(t), z[mask].mean()))
        f, a = source(resid_clean, dt)
        cand = fallback = None
        for j in np.argsort(-np.abs(a)):
            fj = float(f[j])
            if abs(fj) < 0.05 or abs(fj) > 9.8:
                continue
            if lines and min(abs(fj - x) for x in lines) < 1.5e-3:
                continue
            if fallback is None:
                fallback = fj
            if is_novel(fj, lines):
                cand = fj
                break
        if cand is None:
            cand = fallback
        if cand is None:
            break
        g2, K2, c2, res2 = lattice_fit(z, t, mask, lines + [cand], B=1)
        if res is not None and res2 > 0.97 * res and res < 5e-3:
            break
        lines, res = list(g2), res2
        g, K, c = g2, K2, c2
        if res < 3e-4:
            break
    if g is None:
        return np.array([]), np.array([]), 1.0
    amps = np.zeros(len(lines), complex)
    for i in range(len(lines)):
        row = np.zeros(len(lines)); row[i] = 1.0
        j = int(np.argmin(np.abs(K - row).sum(1)))
        amps[i] = c[j]
    return np.asarray(lines, float), amps, res


# ---------------------------------------------------------------- selection
def select_bases(lines, amps, fs, n_gen=3, n_keep=6):
    """Rank candidate generator tuples among the fitted lines by coverage,
    with a consensus reweight that zeroes lines no top basis can explain
    (CLEAN artifacts). The top-|amps| tuple is always included."""
    w = np.abs(amps) / (np.abs(amps).sum() + 1e-30)
    idx = [i for i in range(len(lines))
           if SPEED_MIN <= abs(lines[i]) <= SPEED_MAX]
    subs = list(combinations(idx, n_gen))
    scored = []
    for sub in subs:
        scored.append((coverage_score(lines, w, lines[list(sub)], fs), sub))
    scored.sort(key=lambda s: -s[0])
    expl = np.zeros(len(lines), bool)
    for _cv, sub in scored[:8]:
        _s, okm = coverage_score(lines, w, lines[list(sub)], fs,
                                 return_ok=True)
        expl |= okm
    if not expl.all() and expl.any():
        w2 = np.where(expl, w, 0.0)
        w2 = w2 / (w2.sum() + 1e-30)
        scored = []
        for sub in subs:
            scored.append((coverage_score(lines, w2, lines[list(sub)], fs),
                           sub))
        scored.sort(key=lambda s: -s[0])
    keep = scored[:n_keep]
    topn = tuple(i for i in np.argsort(-np.abs(amps)) if i in idx)[:n_gen]
    if len(topn) == n_gen and \
            tuple(sorted(topn)) not in [tuple(sorted(k[1])) for k in keep]:
        covn = coverage_score(lines, w, lines[list(topn)], fs)
        keep.append((covn, topn))
    return [(cv, np.array(lines[list(s)])) for cv, s in keep]


# ---------------------------------------------------------------- main
def extract_speeds(pattern, dt, n_gen=3, verbose=False):
    """Signed speeds (sorted by descending |N|) + diagnostics dict."""
    t = np.arange(len(pattern)) * dt
    mask = deglitch_mask(pattern)
    z = pattern[:, 0] + 1j * pattern[:, 1]
    zc = interp_masked(z, t, mask)
    fs = 1.0 / dt
    info = {'masked': int((~mask).sum())}

    max_lines = MAX_LINES if n_gen <= 3 else 2 * n_gen + 3
    lines, amps, res_clean = clean_lines(z, zc, t, mask, dt,
                                         max_lines=max_lines)
    # front-end overload: a deeply non-paraxial spectrum can carry more
    # significant lattice lines than the pencil's model order (a dense comb
    # with tooth spacing = the minimal small-k lattice value), and the
    # pencil then returns cluster centroids instead of lines. res_clean
    # self-diagnoses this; retry once with capacity-free FFT-peak seeding
    # (joint GN still supplies precision). Fires on overloaded cases only.
    if res_clean > 0.3:
        lines2, amps2, res2 = clean_lines(z, zc, t, mask, dt,
                                          max_lines=max_lines,
                                          line_source=fft_lines)
        if res2 < res_clean:
            lines, amps, res_clean = lines2, amps2, res2
            info['overload_retry'] = True
    info['lines'] = lines
    info['res_clean'] = float(res_clean)
    if len(lines) < n_gen:
        info['fail'] = 'rank<n'
        info['gens_partial'] = [float(x) for x in lines
                                if SPEED_MIN <= abs(x) <= SPEED_MAX]
        return None, info

    # coverage is the primary gate; the fit score referees within a band;
    # candidates whose fit drifts far from the measured seed are wrong
    cands = select_bases(lines, amps, fs, n_gen=n_gen)
    if not cands:
        info['fail'] = 'rank<n'
        info['gens_partial'] = [float(x) for x in lines
                                if SPEED_MIN <= abs(x) <= SPEED_MAX]
        return None, info
    cov_max = max(cv for cv, _ in cands)
    fits = []
    for cv, gseed in cands:
        g, K, c, res = lattice_fit(z, t, mask, gseed, B=B_FIT)
        score = res * (1.0 + 2.0 * parsimony(K, c))
        moved = float(np.max(np.abs(g - gseed)))
        gated = (cv >= cov_max - 0.05) and moved < 0.06
        fits.append((not gated, score, cv, list(g), K, c, res))
    fits.sort(key=lambda f: (f[0], f[1]))
    _, _, cov_best, gens, K, c, res = fits[0]
    info['alts'] = [np.array(sorted(f[3], key=lambda x: -abs(x)))
                    for f in fits[1:]]

    # accidental-relation repair: a big line at |k|_1 >= 2 may be a
    # fundamental sitting on a near-relation -- trial-split it
    a = np.abs(c)
    ref = np.sort(a)[-n_gen]
    tried = 0
    for j in np.argsort(-a):
        if tried >= 2:
            break
        if np.abs(K[j]).sum() >= 2 and a[j] >= 0.2 * ref:
            tried += 1
            fj = float(K[j] @ np.array(gens))
            if not (SPEED_MIN <= abs(fj) <= SPEED_MAX):
                continue
            g2, K2, c2, res2 = lattice_fit(z, t, mask, list(gens) + [fj])
            if res2 < 0.6 * res:
                gens, K, c, res = list(g2), K2, c2, res2
                break

    # swap/add repair, only when coverage says lines are unexplained
    for _round in range(2):
        if cov_best >= 0.92 or len(gens) != n_gen:
            break
        rq = zc - model_eval(t, np.array(gens), K, c)
        fr, cr = matrix_pencil(rq, dt)
        fstars = []
        for j in np.argsort(-np.abs(cr)):
            fj = float(fr[j])
            if SPEED_MIN <= abs(fj) <= SPEED_MAX and \
                    min(abs(fj - x) for x in gens) > 0.02 and \
                    all(abs(fj - x) > 0.02 for x in fstars):
                fstars.append(fj)
            if len(fstars) >= 3:
                break
        improved = False
        for fstar in fstars:
            trials = [[fstar if i == q else gi for i, gi in enumerate(gens)]
                      for q in range(n_gen)]
            if min(abs(fstar - x) for x in gens) > 0.05:
                trials.append(list(gens) + [fstar])
            for tr in trials:
                g2, K2, c2, res2 = lattice_fit(z, t, mask, tr)
                if res2 < 0.55 * res and \
                        np.max(np.abs(g2 - np.asarray(tr))) < 0.06:
                    gens, K, c, res = list(g2), K2, c2, res2
                    improved = True
            if improved:
                break
        if not improved:
            break

    fnd = fundamentals(K, np.array(gens), c, n_gen=n_gen)
    if fnd is None or np.max(np.abs(fnd[0])) > SPEED_MAX:
        if len(gens) == n_gen and np.max(np.abs(gens)) <= SPEED_MAX:
            Nf = np.array(gens)
        else:
            info['fail'] = 'rank<n'
            info['resid'] = float(res)
            info['gens_partial'] = [float(x) for x in gens
                                    if SPEED_MIN <= abs(x) <= SPEED_MAX]
            return None, info
    else:
        Nf = fnd[0]

    # polish in the fundamental basis: the refined generators ARE the
    # speeds -- never re-extract fundamentals here
    def e_amp(Kf, cf, i, sign=1.0):
        row = np.zeros(n_gen); row[i] = sign
        return cf[int(np.argmin(np.abs(Kf - row).sum(1)))]

    def refine(Nf0, msk, res0):
        Nf1, res1, Kc = np.asarray(Nf0, float), res0, None
        Bf = B_REFINE if res0 > 2e-3 else B_FIT
        g3, K3, c3, res3 = lattice_fit(z, t, msk, list(Nf1), B=Bf)
        if res3 <= res1 and np.max(np.abs(g3 - Nf1)) < 0.07:
            Nf1, res1, Kc = g3, res3, (K3, c3)
        g4, K4, c4, res4 = lattice_fit(z, t, msk, list(Nf1), B=B_FIT,
                                       sharp=True)
        if res4 < res1 and np.max(np.abs(g4 - Nf1)) < 0.07:
            Nf1, res1, Kc = g4, res4, (K4, c4)
        return Nf1, res1, Kc

    Nf, res, Kc = refine(Nf, mask, res)

    # self-consistent glitch remask
    if Kc is not None and ((~mask).sum() > 0 or res > 2.5e-2):
        r_full = np.abs(z - model_eval(t, Nf, Kc[0], Kc[1]))
        mad = np.median(r_full[mask]) + 1e-30
        mask2 = r_full < 8 * mad
        if mask2.sum() >= 0.85 * len(mask2) and \
                (mask2 != mask).any() and mask2.sum() >= 100:
            Nf2, res2, Kc2 = refine(Nf, mask2, np.inf)
            if Kc2 is not None and res2 < res:
                Nf, res, Kc, mask = Nf2, res2, Kc2, mask2
                info['masked'] = int((~mask2).sum())

    # physical sign test: conjugate leak weaker than main line. The test
    # must ALWAYS run -- when the polish was rejected (Kc None), obtain the
    # amplitudes from a guarded sharp fit at the current basis (case 434:
    # the skipped test hid a 14x leak-inequality violation).
    if Kc is None:
        g_s, K_s, c_s, res_s = lattice_fit(z, t, mask, list(Nf), B=B_FIT,
                                           sharp=True)
        if np.max(np.abs(g_s - np.asarray(Nf, float))) < 0.07:
            Kc = (K_s, c_s)
    if Kc is not None:
        flips = [i for i in range(n_gen)
                 if np.abs(e_amp(*Kc, i, -1.0)) > np.abs(e_amp(*Kc, i, 1.0))]
        if flips:
            for i in flips:
                Nf[i] = -Nf[i]
            g5, K5, c5, res5 = lattice_fit(z, t, mask, list(Nf), B=B_FIT,
                                           sharp=True)
            if np.max(np.abs(g5 - Nf)) < 0.07:
                Nf, res, Kc = g5, res5, (K5, c5)

    order = np.argsort(-np.abs(Nf))
    if Kc is not None:
        phases_c = np.array([e_amp(*Kc, i) for i in range(n_gen)])
        info['phases'] = np.degrees(np.angle(phases_c))[order]
        info['amps'] = np.abs(phases_c)[order]
    info['resid'] = float(res)
    info['mask'] = mask
    N_out = Nf[order]
    info['N'] = N_out.copy()
    return N_out, info
