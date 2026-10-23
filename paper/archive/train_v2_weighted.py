#!/usr/bin/env python3
"""
Model v2: Sensitivity-weighted AngleNet.

The forward model is 35x more sensitive to alpha_x than alpha_y.
Current loss weights them equally → network wastes capacity on alpha_y.

Fix: weight the loss by forward model sensitivity.
Same speed as v1 (~20 min), fundamentally better priorities.
"""
import sys, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'reverse_problem_v2'))
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch.utils.data import DataLoader, TensorDataset

from ml_staged_solver import (
    AngleNet, DEVICE, N_PAR, LO, HI, RG, P, BATCH,
    load_or_gen_data, extract_all_speeds, fmt_time, canon,
    extract_speeds_and_peaks, _build_peak_feats_single,
    RemainNet, REM_MODEL,
)

ANG_V2 = os.path.join(HERE, 'ml_stage2_angles_v2.pt')

# Sensitivity weights from Jacobian analysis (averaged over 50 random cases)
# alpha_x is 35x more sensitive than alpha_y
# Output order: [ax1, ax2, ax3, ay1, ay2, ay3] (normalized)
SENS_WEIGHTS = torch.tensor(
    [95.7, 71.3, 69.4, 1.0, 1.2, 4.6],
    dtype=torch.float32, device=DEVICE)
# Normalize so mean = 1 (doesn't change optimization, just scale)
SENS_WEIGHTS = SENS_WEIGHTS / SENS_WEIGHTS.mean()


def train_weighted(model, train_ds, val_ds, epochs=80, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tr_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    va_dl = DataLoader(val_ds, batch_size=BATCH*2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(tr_dl))

    best_val = float('inf')
    t0 = time.time()

    for ep in range(epochs):
        model.train(); tl = 0; nb = 0
        for ang_in, tgt in tr_dl:
            ang_in = ang_in.to(DEVICE)
            tgt = tgt.to(DEVICE)
            pred = model(ang_in)

            # Weighted MSE: alpha_x errors cost 35x more than alpha_y
            diff_sq = (pred - tgt) ** 2  # (B, 6)
            loss = (diff_sq * SENS_WEIGHTS[None, :]).mean()

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tl += loss.item(); nb += 1

        # Validation: weighted loss
        model.eval(); vl = 0; vn = 0
        with torch.no_grad():
            for ang_in, tgt in va_dl:
                ang_in = ang_in.to(DEVICE); tgt = tgt.to(DEVICE)
                pred = model(ang_in)
                diff_sq = (pred - tgt) ** 2
                vl += (diff_sq * SENS_WEIGHTS[None, :]).mean().item()
                vn += 1

        vl /= vn
        improved = vl < best_val
        if improved:
            best_val = vl
            torch.save(model.state_dict(), ANG_V2)

        if (ep+1) % 5 == 0 or ep == 0:
            eta = (time.time()-t0)/(ep+1) * (epochs-ep-1)
            print(f"  Ep {ep+1:3d}/{epochs}  tr={tl/nb:.6f} val={vl:.6f} "
                  f"best={best_val:.6f}{'*' if improved else ''}  "
                  f"ETA {fmt_time(eta)}", flush=True)

    model.load_state_dict(torch.load(ANG_V2, map_location=DEVICE, weights_only=True))
    print(f"  Done: {fmt_time(time.time()-t0)}, best val={best_val:.6f}")
    return model


if __name__ == '__main__':
    print("="*70)
    print("MODEL v2: Sensitivity-Weighted AngleNet")
    print("="*70)
    print(f"Weights: ax ~70-96x, ay ~1-5x (ax/ay ratio: 35x)")

    print("\n--- Loading data ---")
    tp, ta, vp, va = load_or_gen_data()

    print("\n--- Peak features ---")
    tr_peaks = extract_all_speeds(tp)
    va_peaks = extract_all_speeds(vp)

    tr_spd_n = ((ta[:, :3] - LO[:3]) / RG[:3]).astype(np.float32)
    va_spd_n = ((va[:, :3] - LO[:3]) / RG[:3]).astype(np.float32)
    tr_in = np.concatenate([tr_spd_n, tr_peaks], axis=1)
    va_in = np.concatenate([va_spd_n, va_peaks], axis=1)

    tr_tgt = np.concatenate([
        (ta[:, 3:6] - LO[3:6]) / RG[3:6],
        (ta[:, 6:9] - LO[6:9]) / RG[6:9],
    ], axis=1).astype(np.float32)
    va_tgt = np.concatenate([
        (va[:, 3:6] - LO[3:6]) / RG[3:6],
        (va[:, 6:9] - LO[6:9]) / RG[6:9],
    ], axis=1).astype(np.float32)

    tr_ds = TensorDataset(torch.from_numpy(tr_in), torch.from_numpy(tr_tgt))
    va_ds = TensorDataset(torch.from_numpy(va_in), torch.from_numpy(va_tgt))

    print(f"\n--- Training (80 epochs, weighted loss) ---")
    model = AngleNet()
    model = train_weighted(model, tr_ds, va_ds, epochs=80, lr=1e-3)

    # --- Evaluate v1 vs v2 on 30 cases ---
    print(f"\n--- Evaluating v1 vs v2 on 30 test cases ---")
    from solve_preconditioned import vec2pat, ml_init
    from scipy.optimize import least_squares

    rem = RemainNet()
    rem.load_state_dict(torch.load(REM_MODEL, map_location=DEVICE, weights_only=True))
    rem.to(DEVICE)

    ang_v1 = AngleNet()
    ang_v1.load_state_dict(torch.load(os.path.join(HERE, 'ml_stage2_angles.pt'),
                                       map_location=DEVICE, weights_only=True))
    ang_v1.to(DEVICE)

    rng = np.random.default_rng(2026)
    cases = []
    for i in range(30):
        v = (LO + RG * rng.random(N_PAR).astype(np.float32)).astype(np.float64)
        for j in range(3):
            if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
        cases.append(canon(v))

    v1_p = 0; v2_p = 0; v1_ax_errs = []; v2_ax_errs = []

    for ci, tc in enumerate(cases):
        pat = vec2pat(tc)
        tf = pat.reshape(-1)
        pf, pi = extract_speeds_and_peaks(pat)
        pk = _build_peak_feats_single(pat, pf, pi)
        fixed = tc[9:].copy()
        lo9 = LO[:9].astype(np.float64); hi9 = HI[:9].astype(np.float64)

        def make_res(fix):
            def r(x9): return vec2pat(np.concatenate([x9, fix])).reshape(-1) - tf
            return r
        res_fn = make_res(fixed)

        for label, ang_net in [('v1', ang_v1), ('v2', model)]:
            best = 1e30; bx = None; best_ax_err = 999
            for bits in range(8):
                signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
                speeds = signs * np.sort(pf)[::-1].astype(np.float64)
                ml = ml_init(ang_net, rem, speeds.astype(np.float32), pk, pat)

                ax_err = float(np.max(np.abs(ml[3:6] - tc[3:6])))
                if ax_err < best_ax_err: best_ax_err = ax_err

                try:
                    res = least_squares(res_fn, ml[:9], jac='2-point',
                        bounds=(lo9, hi9), method='trf',
                        ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
                    mse = float(np.mean(res.fun**2))
                    if mse < best: best = mse; bx = res.x.copy()
                except: pass

            err = float(np.max(np.abs(bx - tc[:9]))) if bx is not None else 999
            if label == 'v1':
                if err < 1e-3: v1_p += 1
                v1_ax_errs.append(best_ax_err)
            else:
                if err < 1e-3: v2_p += 1
                v2_ax_errs.append(best_ax_err)

        sys.stdout.write("."); sys.stdout.flush()

    print(f"\n\n{'='*70}")
    print(f"  v1 (uniform loss):   {v1_p}/30 PERFECT  "
          f"median ax_err={np.median(v1_ax_errs):.1f} deg")
    print(f"  v2 (weighted loss):  {v2_p}/30 PERFECT  "
          f"median ax_err={np.median(v2_ax_errs):.1f} deg")
    print(f"{'='*70}")
