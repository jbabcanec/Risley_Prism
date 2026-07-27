#!/usr/bin/env python3
"""
spectral_speeds.py -- Signed speed extraction by CLEAN line growth + lattice
selection + VarPro polish. No sign combos, no triple brute force, no forward
model evaluations.

Model: z(t) = x(t) + i y(t) = sum_{k in Z^3} c_k exp(2i pi (k.N) t).

Pipeline:
  1. de-glitch: mask TIR-clip samples (pattern-jump detection);
  2. CLEAN growth at order B=1: repeatedly take the strongest matrix-pencil
     line of the residual and refit ALL line frequencies jointly (VarPro GN).
     At B=1 a line explains only itself, so no compromise/subharmonic basis
     can absorb foreign lines -- the failure mode of residual-guided greedy;
  3. basis selection: among the precise fitted lines, score each candidate
     triple by amplitude-weighted small-integer coverage of all lines
     (arithmetic only; folded aliases f+-1/dt included);
  4. full lattice fit at |k|_1 <= 3 (refine 4) from the best couple of bases;
     accidental-relation repair (big line at |k|_1>=2 -> trial generator);
  5. canonicalization: fundamentals = rank-extending largest-|c| lines;
     final polish in the fundamental basis.

Signs come free (complex signal). arg(c) at the fundamental is ay_i
(+180 deg iff ax_i < 0); |c| feeds ax recovery downstream.

API:  extract_speeds(pattern, dt) -> (N[3] signed, desc |N|, or None, info)
Test: python paper/spectral_speeds.py     (30-case seed-2026 battery)
"""
import sys, os, time
from itertools import product as iproduct, combinations
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

SPEED_MAX = 3.6          # physical bound on |N| (parameter box is +-3.5)
SPEED_MIN = 0.10         # battery clamps |N| >= 0.15; near-DC lines are junk
B_FIT = 3
B_REFINE = 4
MAX_LINES = 8
MERGE_TOL = 0.012        # lattice lines closer than this are indistinguishable


# ---------------------------------------------------------------- pencil
def matrix_pencil(z, dt, L=80, rtol=1e-8, max_M=60):
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


# ---------------------------------------------------------------- lattice fit
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

    Amplitudes solve a RIDGE system with near-coincident lattice lines MERGED
    (min-|k|_1 representative): unregularized, such lines produce exploding
    canceling amplitude pairs that poison |c|-based ranking. sharp=True turns
    both protections off for a final local refinement from a good basis.
    The order B is reduced if the design would be underdetermined."""
    g = np.asarray(gens, float).copy()
    zm, tm = z[mask], t[mask]
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
                if vals[j] - vals[grp[-1]] < MERGE_TOL:
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
    for _ in range(iters):
        J = 2j * np.pi * tm[:, None] * (E * c[None, :]) @ K
        Jr = np.concatenate([J.real, J.imag])
        rr = np.concatenate([r.real, r.imag])
        dg, *_ = np.linalg.lstsq(Jr, rr, rcond=None)
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
    a = np.abs(c)
    o = np.abs(K).sum(1)
    return float((a * np.maximum(0, o - 1)).sum() / (a.sum() + 1e-30))


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


# ---------------------------------------------------------------- CLEAN growth
def is_novel(fj, lines, tol=0.02):
    """Not a small-integer combo (|a|+|b|<=3, aliases included) of known lines."""
    if not lines:
        return True
    L = np.asarray(lines)
    combos = [a * L[i] + b * L[j]
              for i in range(len(L)) for j in range(i, len(L))
              for a in (-3, -2, -1, 0, 1, 2, 3) for b in (-3, -2, -1, 0, 1, 2, 3)
              if 0 < abs(a) + abs(b) <= 3]
    combos = np.asarray(combos)
    d = np.abs(combos - fj)
    d = np.minimum(d, np.abs(combos - fj - 20.0))
    d = np.minimum(d, np.abs(combos - fj + 20.0))
    return float(d.min()) > tol


def clean_lines(z, zc, t, mask, dt, max_lines=MAX_LINES):
    """Grow a B=1 line model: strongest pencil line of residual, joint refit.
    Novel lines (not combos of existing ones) get priority, so harmonics do
    not exhaust the slots before a weak fundamental appears.
    Returns fitted line freqs, complex amps, rel resid of the line model."""
    scale = np.linalg.norm(z[mask] - z[mask].mean()) + 1e-30
    lines, res = [], 1.0
    g = K = c = None
    for _ in range(max_lines):
        resid_clean = zc - (model_eval(t, g, K, c) if g is not None else
                            np.full(len(t), z[mask].mean()))
        f, a = matrix_pencil(resid_clean, dt)
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
            break                                    # no longer helping
        lines, res = list(g2), res2
        g, K, c = g2, K2, c2
        if res < 3e-4:
            break
    if g is None:
        return np.array([]), np.array([]), 1.0
    # per-line amplitude = |c| at k=+e_i
    amps = np.zeros(len(lines), complex)
    for i in range(len(lines)):
        row = np.zeros(len(lines)); row[i] = 1.0
        j = int(np.argmin(np.abs(K - row).sum(1)))
        amps[i] = c[j]
    return np.asarray(lines, float), amps, res


# ---------------------------------------------------------------- basis choice
def coverage_score(lines, w, g, fs, B=B_FIT, tol=6e-4, return_ok=False):
    """Amplitude-weighted small-integer coverage of all lines by basis g.
    Aliases f +- fs are accepted."""
    KB = kset(3, B)
    vals = KB @ g
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
    score = float((w * okm).sum() - 0.05 * (w * okm * np.maximum(0, orders - 1)).sum())
    return (score, okm) if return_ok else score


def select_bases(lines, amps, fs, n_keep=6):
    """Rank candidate generator triples among the fitted lines by coverage.
    The top-3 lines by amplitude (almost always the fundamentals) are always
    included. Returns [(coverage, basis array)] sorted by coverage."""
    w = np.abs(amps) / (np.abs(amps).sum() + 1e-30)
    idx = [i for i in range(len(lines))
           if SPEED_MIN <= abs(lines[i]) <= SPEED_MAX]
    subs = list(combinations(idx, 3))
    scored = []
    for sub in subs:
        scored.append((coverage_score(lines, w, lines[list(sub)], fs), sub))
    scored.sort(key=lambda s: -s[0])
    # consensus re-weight: a line no top-8 basis can explain is junk -- drop
    # its weight so CLEAN artifacts cannot poison the ranking
    expl = np.zeros(len(lines), bool)
    for _cv, sub in scored[:8]:
        _s, okm = coverage_score(lines, w, lines[list(sub)], fs, return_ok=True)
        expl |= okm
    if not expl.all() and expl.any():
        w2 = np.where(expl, w, 0.0)
        w2 = w2 / (w2.sum() + 1e-30)
        scored = []
        for sub in subs:
            scored.append((coverage_score(lines, w2, lines[list(sub)], fs), sub))
        scored.sort(key=lambda s: -s[0])
    keep = scored[:n_keep]
    top3 = tuple(i for i in np.argsort(-np.abs(amps)) if i in idx)[:3]
    if len(top3) == 3 and \
            tuple(sorted(top3)) not in [tuple(sorted(k[1])) for k in keep]:
        cov3 = coverage_score(lines, w, lines[list(top3)], fs)
        keep.append((cov3, top3))
    return [(cv, np.array(lines[list(s)])) for cv, s in keep]


def fundamentals(K, g, c):
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
        if len(rows) == 3:
            return np.array(freqs), rows, np.array(amps)
    return None


# ---------------------------------------------------------------- main
def extract_speeds(pattern, dt, verbose=False):
    t = np.arange(len(pattern)) * dt
    mask = deglitch_mask(pattern)
    z = pattern[:, 0] + 1j * pattern[:, 1]
    zc = interp_masked(z, t, mask)
    fs = 1.0 / dt
    info = {'masked': int((~mask).sum())}

    lines, amps, res_clean = clean_lines(z, zc, t, mask, dt)
    info['lines'] = lines
    info['res_clean'] = float(res_clean)
    if len(lines) < 3:
        info['fail'] = 'rank<3'
        return None, info

    # coverage is the primary gate (arithmetic, immune to fit pathologies);
    # the fit score only referees candidates within a small coverage band
    cands = select_bases(lines, amps, fs)
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
    info['alts'] = [np.array(sorted(f[3], key=lambda x: -abs(x))) for f in fits[1:]]

    # accidental-relation repair
    a = np.abs(c)
    ref = np.sort(a)[-3]
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

    # swap/add repair: a strong line left in the residual is a missed
    # fundamental -- try swapping it for each generator (or adding it).
    # Only when coverage says lines are unexplained: with a well-covering
    # basis this step corrupts more than it cures (fit-res is a noisy judge).
    for _round in range(2):
        if cov_best >= 0.92 or len(gens) != 3:
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
                      for q in range(3)]
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

    fnd = fundamentals(K, np.array(gens), c)
    if fnd is None or np.max(np.abs(fnd[0])) > SPEED_MAX:
        if len(gens) == 3 and np.max(np.abs(gens)) <= SPEED_MAX:
            Nf = np.array(gens)              # fall back to the fitted basis
        else:
            info['fail'] = 'rank<3'
            info['resid'] = float(res)
            return None, info
    else:
        Nf = fnd[0]

    # polish in the fundamental basis. The basis is already fundamental, so
    # the refined generators ARE the speeds -- re-extracting fundamentals
    # from refit amplitudes invites splitting hijacks; never do it here.
    def e_amp(Kf, cf, i, sign=1.0):
        row = np.zeros(3); row[i] = sign
        return cf[int(np.argmin(np.abs(Kf - row).sum(1)))]

    def refine(Nf0, msk, res0):
        Nf1, res1, Kc = np.asarray(Nf0, float), res0, None
        Bf = B_REFINE if res0 > 2e-3 else B_FIT
        g3, K3, c3, res3 = lattice_fit(z, t, msk, list(Nf1), B=Bf)
        if res3 <= res1 and np.max(np.abs(g3 - Nf1)) < 0.07:
            Nf1, res1, Kc = g3, res3, (K3, c3)
        g4, K4, c4, res4 = lattice_fit(z, t, msk, list(Nf1), B=B_FIT, sharp=True)
        if res4 < res1 and np.max(np.abs(g4 - Nf1)) < 0.07:
            Nf1, res1, Kc = g4, res4, (K4, c4)
        return Nf1, res1, Kc

    Nf, res, Kc = refine(Nf, mask, res)

    # self-consistent glitch remask: outlier samples against the fitted model
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

    # sign check: the conjugate leak (-N_i) is always weaker than the main
    # line (+N_i); if the fit says otherwise, the sign is flipped
    if Kc is not None:
        flips = [i for i in range(3)
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
        phases_c = np.array([e_amp(*Kc, i) for i in range(3)])
        info['phases'] = np.degrees(np.angle(phases_c))[order]
        info['amps'] = np.abs(phases_c)[order]
    info['resid'] = float(res)
    info['mask'] = mask
    return Nf[order], info


# ---------------------------------------------------------------- battery test
if __name__ == '__main__':
    from ml_staged_solver import N_PAR, LO, HI, RG, canon, vec2pat, T_PTS, T_OBS, FREQS
    DT = T_OBS / T_PTS

    def fft_top3(pattern):
        fx = np.fft.rfft(pattern[:, 0]); fy = np.fft.rfft(pattern[:, 1])
        pw = np.abs(fx) + np.abs(fy); pw[0] = 0.0
        out = []
        for _ in range(3):
            i = int(np.argmax(pw))
            out.append(float(FREQS[i]))
            pw[max(1, i - 2):i + 3] = 0.0
        return np.sort(out)[::-1]

    rng = np.random.default_rng(2026)
    cases = []
    for _ in range(30):
        v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
        for j in range(3):
            if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
        cases.append(canon(v))

    print(f"\n{'='*106}")
    print("SPECTRAL SPEEDS v4: CLEAN line growth + lattice basis selection (signed, 30 cases)")
    print(f"{'='*106}")
    print(f"{'#':>3} {'cyc':>5} {'sep':>6}  {'true speeds':>24}  {'estimate':>24} "
          f"{'err':>8} {'resid':>8} {'msk':>3}  {'FFT':>4} {'V4':>4}")

    n_v4 = n_fft = n_top3 = 0
    errs, tags = [], {'close': 0, 'lowcyc': 0, '**': 0}
    t0 = time.time()
    for ci, tc in enumerate(cases):
        pat = vec2pat(tc)
        sp = tc[:3]
        sep = min(abs(abs(sp[i]) - abs(sp[j])) for i in range(3) for j in range(i+1, 3))
        cyc = float(min(abs(sp)) * T_OBS)

        fm = fft_top3(pat)
        fft_ok = np.max(np.abs(fm - np.sort(np.abs(sp))[::-1])) < 0.05

        N, info = extract_speeds(pat, DT)
        if N is None:
            ok, err = False, 999.0
        else:
            err = float(np.max(np.abs(N - sp)))
            ok = err < 0.02
        ok3 = ok
        if not ok:
            for alt in info.get('alts', [])[:2]:
                if len(alt) == 3 and np.max(np.abs(alt - sp)) < 0.02:
                    ok3 = True
                    break
        n_v4 += ok; n_fft += fft_ok; n_top3 += ok3
        if ok:
            errs.append(err)

        tag = ""
        if not ok:
            key = ('close' if sep < 0.08 else 'lowcyc' if cyc < 2.5 else '**')
            tags[key] += 1
            tag = f"  [{key}]"
            ln = info.get('lines', np.array([]))
            if len(ln):
                tag += f"  lines: {' '.join(f'{x:+.3f}' for x in ln)}"
        sp_s = " ".join(f"{v:+.3f}" for v in sp)
        n_s = " ".join(f"{v:+.3f}" for v in N) if N is not None else "rank<3"
        print(f"{ci+1:>3} {cyc:>5.1f} {sep:>6.3f}  {sp_s:>24}  {n_s:>24} "
              f"{err:>8.1e} {info.get('resid', 1.0):>8.1e} {info['masked']:>3}  "
              f"{'ok' if fft_ok else '--':>4} {'OK' if ok else '--':>4}{tag}",
              flush=True)

    dt_all = time.time() - t0
    print(f"\n{'='*106}")
    print(f"  V4 CLEAN+lattice (signed): {n_v4}/30  (truth in top-3 bases: {n_top3}/30)"
          f"    FFT top-3 (mags): {n_fft}/30")
    print(f"  unsolved by class: {tags}")
    if errs:
        print(f"  speed err among successes: median {np.median(errs):.1e}  "
              f"max {np.max(errs):.1e}")
    print(f"  {dt_all:.1f}s total ({dt_all/30:.2f}s/case)")
    print(f"{'='*106}")
