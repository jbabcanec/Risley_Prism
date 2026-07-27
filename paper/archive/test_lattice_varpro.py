#!/usr/bin/env python3
"""
test_lattice_varpro.py -- Lattice VarPro speed extraction (v2 of the pencil idea).

The pattern is quasi-periodic: z(t) = sum_k c_k exp(2i pi (k.N) t), k in Z^3.
Fit that model DIRECTLY: amplitudes c_k are linear, only the 3 generators N are
nonlinear (Gauss-Newton, 3 parameters). Generators are grown greedily from the
strongest pencil line of the residual. No sign combos (complex signal), no
triple enumeration (greedy + canonicalization), no forward-model evaluations.

Why this beats per-line arithmetic (test_matrix_pencil.py, 8/30):
  - conjugate leaks (x/y gain asymmetry puts a line at -N_i) are the k=-e_i
    lines of the model, not competing atoms;
  - a merged/biased pencil line is re-estimated by GN inside the joint fit;
  - aliased high-order lines are modeled exactly (exp at exact k.N folds itself);
  - harmonics can seed a generator slot, because any basis of the right group
    fits identically -- the physical fundamentals are read off afterwards as
    the rank-extending largest-amplitude lines (first order dominates).

Run: python paper/test_lattice_varpro.py
"""
import sys, os, time
from itertools import product as iproduct
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

from ml_staged_solver import N_PAR, LO, HI, RG, canon, vec2pat, T_PTS, T_OBS, FREQS

DT = T_OBS / T_PTS
SPEED_MAX = 3.6          # physical box on |N| (LO/HI is +-3.5)
B_ORDER = 3              # lattice order bound |k|_1 <= B


# ---------------------------------------------------------------- matrix pencil
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


# ---------------------------------------------------------------- lattice model
def kset(ngen, B=B_ORDER):
    ks = [k for k in iproduct(range(-B, B + 1), repeat=ngen)
          if sum(abs(x) for x in k) <= B]
    return np.array(ks, float)


def lattice_fit(z, t, gens, B=B_ORDER, iters=12, rcond=1e-8):
    """VarPro: LS amplitudes over lattice lines, GN on the generators."""
    g = np.array(gens, float)
    K = kset(len(g), B)
    c = None
    for _ in range(iters):
        E = np.exp(2j * np.pi * np.outer(t, K @ g))
        c, *_ = np.linalg.lstsq(E, z, rcond=rcond)
        r = z - E @ c
        J = 2j * np.pi * t[:, None] * (E * c[None, :]) @ K
        Jr = np.concatenate([J.real, J.imag])
        rr = np.concatenate([r.real, r.imag])
        dg, *_ = np.linalg.lstsq(Jr, rr, rcond=None)
        g = g + dg
        if np.max(np.abs(dg)) < 1e-11:
            break
    E = np.exp(2j * np.pi * np.outer(t, K @ g))
    c, *_ = np.linalg.lstsq(E, z, rcond=rcond)
    r = z - E @ c
    return g, K, c, r


def next_gen_candidate(resid, dt, gens, fmax=SPEED_MAX):
    """Strongest pencil line of the residual that could be a new generator."""
    f, c = matrix_pencil(resid, dt)
    if len(f) == 0:
        return None
    order = np.argsort(-np.abs(c))
    for j in order:
        fj = f[j]
        if abs(fj) < 0.02 or abs(fj) > fmax:
            continue
        if any(min(abs(fj - gi), abs(fj + gi)) < 2e-3 for gi in gens):
            continue                      # partial-absorption remnant of a gen
        return float(fj)
    return None


def fundamentals(K, g, c):
    """Physical fundamentals: rank-extending largest-|c| lines. Signed freqs."""
    idx = np.argsort(-np.abs(c))
    rows, freqs = [], []
    for j in idx:
        k = K[j]
        if not k.any():
            continue
        if rows and np.linalg.matrix_rank(np.array(rows + [k])) == len(rows):
            continue
        rows.append(k); freqs.append(float(K[j] @ g))
        if len(rows) == 3:
            break
    if len(rows) < 3:
        return None
    N = np.array(freqs)
    return N[np.argsort(-np.abs(N))]


def solve_speeds(pattern, dt, verbose=False):
    t = np.arange(len(pattern)) * dt
    z = pattern[:, 0] + 1j * pattern[:, 1]
    z0 = z - z.mean()
    scale = np.linalg.norm(z0)
    gens, resid = [], z0
    K = c = None
    for _ in range(3):
        fnew = next_gen_candidate(resid, dt, gens)
        if fnew is None:
            break
        gens.append(fnew)
        g, K, c, r = lattice_fit(z, t, gens)
        gens = list(g)
        resid = r
        if np.linalg.norm(r) < 1e-7 * scale:
            break
    if K is None:
        return None, 1.0
    # plausibility split: a big-amplitude line with |k|_1 >= 2 may be a
    # fundamental sitting on an accidental near-relation -- trial-split it.
    if len(gens) == 3:
        rel = np.linalg.norm(resid) / scale
        amps = np.abs(c)
        big = np.sort(amps)[-3]
        for j in np.argsort(-amps):
            if np.abs(K[j]).sum() >= 2 and amps[j] >= 0.25 * big:
                fj = float(K[j] @ np.array(gens))
                g2, K2, c2, r2 = lattice_fit(z, t, gens + [fj])
                if np.linalg.norm(r2) < 0.67 * np.linalg.norm(resid):
                    gens, K, c, resid = list(g2), K2, c2, r2
                break
    N = fundamentals(K, np.array(gens), c)
    rel = np.linalg.norm(resid) / scale
    if verbose:
        amps = np.abs(c)
        for j in np.argsort(-amps)[:10]:
            print(f"      k={K[j].astype(int)}  f={K[j] @ np.array(gens):+8.4f}"
                  f"  |c|={amps[j]:9.4f}")
    return N, rel


# ---------------------------------------------------------------- battery
def fft_top3(pattern):
    fx = np.fft.rfft(pattern[:, 0]); fy = np.fft.rfft(pattern[:, 1])
    pw = np.abs(fx) + np.abs(fy); pw[0] = 0.0
    out = []
    for _ in range(3):
        i = int(np.argmax(pw))
        out.append(float(FREQS[i]))
        pw[max(1, i - 2):i + 3] = 0.0
    return np.sort(out)[::-1]


if __name__ == '__main__':
    rng = np.random.default_rng(2026)
    cases = []
    for _ in range(30):
        v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
        for j in range(3):
            if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
        cases.append(canon(v))

    print(f"\n{'='*104}")
    print("LATTICE VARPRO  (greedy generators + joint quasi-periodic fit; signed speeds, 30 cases)")
    print(f"{'='*104}")
    print(f"{'#':>3} {'cyc':>5} {'sep':>6}  {'true speeds':>24}  {'varpro est':>24} "
          f"{'err':>8} {'resid':>8}  {'FFT':>4} {'VP':>4}")

    n_vp = n_fft = 0
    errs = []
    t0 = time.time()
    for ci, tc in enumerate(cases):
        pat = vec2pat(tc)
        sp = tc[:3]
        sep = min(abs(abs(sp[i]) - abs(sp[j])) for i in range(3) for j in range(i+1, 3))
        cyc = float(min(abs(sp)) * T_OBS)

        fm = fft_top3(pat)
        fft_ok = np.max(np.abs(fm - np.sort(np.abs(sp))[::-1])) < 0.05

        N, rel = solve_speeds(pat, DT)
        if N is None:
            ok, err = False, 999.0
        else:
            err = float(np.max(np.abs(N - sp)))
            ok = err < 0.02
        n_vp += ok; n_fft += fft_ok
        if ok:
            errs.append(err)

        tag = "" if ok else ("  [close]" if sep < 0.08 else
                             "  [lowcyc]" if cyc < 2.5 else "  [**]")
        sp_s = " ".join(f"{v:+.3f}" for v in sp)
        n_s = " ".join(f"{v:+.3f}" for v in N) if N is not None else "rank<3"
        print(f"{ci+1:>3} {cyc:>5.1f} {sep:>6.3f}  {sp_s:>24}  {n_s:>24} "
              f"{err:>8.1e} {rel:>8.1e}  {'ok' if fft_ok else '--':>4} "
              f"{'OK' if ok else '--':>4}{tag}", flush=True)

    dt_all = time.time() - t0
    print(f"\n{'='*104}")
    print(f"  LATTICE VARPRO (signed): {n_vp}/30      FFT top-3 (magnitudes): {n_fft}/30"
          f"      pencil-v1 arithmetic: 8/30")
    if errs:
        print(f"  speed err among successes: median {np.median(errs):.1e}  "
              f"max {np.max(errs):.1e}")
    print(f"  {dt_all:.1f}s total ({dt_all/30:.2f}s/case)")
    print(f"{'='*104}")
