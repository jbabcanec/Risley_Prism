#!/usr/bin/env python3
"""
Fast multi-triple 9-D test: 30 Adam steps for screening (not 150).
Tests whether multi-triple search fixes the wrong-FFT-speed failures.

From diagnostic: 12/21 failures had spd_match=False (wrong FFT peaks).
Multi-triple should fix these by trying C(8,3)=56 frequency triples.
"""
import sys, os, time
import numpy as np
from itertools import combinations
from scipy.optimize import least_squares
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverse_problem_v2'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from ml_staged_solver import (
    AngleNet, RemainNet, ANG_MODEL, REM_MODEL,
    DEVICE, N_PAR, LO, HI, RG, canon,
    extract_speeds_and_peaks, _build_peak_feats_single, P,
    DiffFwd,
)
from solve_preconditioned import vec2pat, ml_init

print("Loading models...", flush=True)
ang = AngleNet(); rem = RemainNet()
ang.load_state_dict(torch.load(ANG_MODEL, map_location=DEVICE, weights_only=True))
rem.load_state_dict(torch.load(REM_MODEL, map_location=DEVICE, weights_only=True))
ang.to(DEVICE); rem.to(DEVICE)

rng = np.random.default_rng(2026)
cases = []
for i in range(30):
    v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
    for j in range(3):
        if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
    cases.append(canon(v))


def quick_adam(fwd, init, target_t, lo_t, hi_t, steps=30):
    """Very quick Adam screen — just enough to separate good from bad inits."""
    par = torch.tensor(init, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    opt = torch.optim.Adam([par], lr=0.01)
    best = float('inf'); best_p = init.copy()
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(fwd(par[None], high_precision=False).squeeze(0), target_t)
        loss.backward(); opt.step()
        with torch.no_grad(): par.clamp_(lo_t, hi_t)
        lv = loss.item()
        if lv < best: best = lv; best_p = par.detach().cpu().numpy().astype(np.float64).copy()
    return best_p, best


print(f"\n{'='*80}")
print("FAST MULTI-TRIPLE — 9-D (30 cases, 30 Adam screening steps)")
print(f"{'='*80}\n")

fwd = DiffFwd().to(DEVICE)
lo_t = torch.tensor(LO, dtype=torch.float32, device=DEVICE)
hi_t = torch.tensor(HI, dtype=torch.float32, device=DEVICE)

single_p = 0; multi_p = 0
t0 = time.time()

for ci, tc in enumerate(cases):
    pat = vec2pat(tc)
    tf = pat.reshape(-1)
    pf, pi_info = extract_speeds_and_peaks(pat)
    pk = _build_peak_feats_single(pat, pf, pi_info)
    fixed = tc[9:].copy()
    lo9 = LO[:9].astype(np.float64); hi9 = HI[:9].astype(np.float64)
    target_t = torch.tensor(pat, dtype=torch.float32, device=DEVICE)

    def make_res(fix):
        def r(x9): return vec2pat(np.concatenate([x9, fix])).reshape(-1) - tf
        return r
    res_fn = make_res(fixed)

    # --- (A) Single triple: 8 signs × ML → TRF ---
    s_best = 1e30; s_x = None
    for bits in range(8):
        signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
        speeds = signs * np.sort(pf)[::-1].astype(np.float64)
        ml = ml_init(ang, rem, speeds.astype(np.float32), pk, pat)
        try:
            res = least_squares(res_fn, ml[:9], jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            mse = float(np.mean(res.fun**2))
            if mse < s_best: s_best = mse; s_x = res.x.copy()
        except: pass

    # --- (B) Multi-triple: C(K,3) × 8 signs × ML → Adam screen → top-10 TRF ---
    all_pf = pi_info.get('all_peak_freqs', pf)
    triples = [tuple(np.sort(pf)[::-1])]
    for combo in combinations(range(len(all_pf)), P):
        triple = tuple(sorted([all_pf[i] for i in combo], reverse=True))
        if triple not in triples and all(f > 0.05 for f in triple):
            triples.append(triple)

    candidates = []
    for triple in triples:
        freq_arr = np.array(triple, dtype=np.float64)
        for bits in range(8):
            signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
            speeds = signs * freq_arr
            ml = ml_init(ang, rem, speeds.astype(np.float32), pk, pat)
            # Quick 30-step Adam screen
            refined, adam_mse = quick_adam(fwd, ml, target_t, lo_t, hi_t, steps=30)
            candidates.append((adam_mse, refined))

    candidates.sort(key=lambda x: x[0])

    m_best = 1e30; m_x = None
    for _, x0 in candidates[:10]:
        try:
            res = least_squares(res_fn, x0[:9], jac='2-point',
                bounds=(lo9, hi9), method='trf',
                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
            mse = float(np.mean(res.fun**2))
            if mse < m_best: m_best = mse; m_x = res.x.copy()
        except: pass
        if m_best < 1e-15: break

    s_err = float(np.max(np.abs(s_x - tc[:9]))) if s_x is not None else 999
    m_err = float(np.max(np.abs(m_x - tc[:9]))) if m_x is not None else 999
    s_ok = s_err < 1e-3; m_ok = m_err < 1e-3
    if s_ok: single_p += 1
    if m_ok: multi_p += 1

    tag = ""
    if m_ok and not s_ok: tag = " ***MULTI SAVED***"
    elif s_ok and not m_ok: tag = " ***SINGLE ONLY***"

    print(f"  {ci+1:2d}: S={'P' if s_ok else 'F'} M={'P' if m_ok else 'F'} "
          f"({len(triples)}tri,{len(candidates)}cands){tag}  [{time.time()-t0:.0f}s]",
          flush=True)

print(f"\n{'='*80}")
print(f"  Single triple:  {single_p}/30")
print(f"  Multi-triple:   {multi_p}/30")
print(f"  Time: {time.time()-t0:.0f}s ({(time.time()-t0)/30:.0f}s/case)")
print(f"{'='*80}")
