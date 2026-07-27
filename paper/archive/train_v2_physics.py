#!/usr/bin/env python3
"""
Model v2: Physics-first AngleNet training.

Key change: PRIMARY loss is forward-model MSE, not parameter MSE.
  L = ||F(theta_pred) - F(theta_true)||^2 + 0.1 * ||theta_pred - theta_true||^2

The network learns to be accurate WHERE THE FORWARD MODEL IS SENSITIVE.
A 1° error in alpha_x (high sensitivity) costs more than 5° in alpha_y (low sensitivity).
This naturally focuses learning on the bottleneck parameter.

Same data, same architecture, different loss. 20-min test.
"""
import sys, os, time, io
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'reverse_problem_v2'))
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ml_staged_solver import (
    AngleNet, DiffFwd, DEVICE, N_PAR, LO, HI, RG, P,
    T_PTS, T_OBS, N_TRAIN, N_VAL, BATCH,
    extract_speeds_and_peaks, load_or_gen_data, extract_all_speeds,
    fmt_time,
)

# Paths for v2 models (don't overwrite v1)
ANG_V2 = os.path.join(HERE, 'ml_stage2_angles_v2.pt')
ANG_V2_CKPT = os.path.join(HERE, 'ml_stage2_angles_v2_ckpt.pt')

LO_T = torch.tensor(LO, device=DEVICE, dtype=torch.float32)
RG_T = torch.tensor(RG, device=DEVICE, dtype=torch.float32)


def train_physics_first(model, train_ds, val_ds, fwd, epochs=80, lr=1e-3,
                        param_weight=0.1):
    """
    Physics-first training loop.

    Primary loss: ||F(assembled_pred) - F(true_params)||^2
    Secondary loss: param_weight * ||pred_angles - true_angles||^2
    """
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tr_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    va_dl = DataLoader(val_ds, batch_size=BATCH*2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(tr_dl))

    best_val = float('inf')
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        tl_param = 0; tl_phys = 0; nb = 0

        for ang_in, true_angles, true_full in tr_dl:
            ang_in = ang_in.to(DEVICE)
            true_angles = true_angles.to(DEVICE)
            true_full = true_full.to(DEVICE)

            pred_angles = model(ang_in)

            # Parameter loss (secondary)
            param_loss = F.mse_loss(pred_angles, true_angles)

            # Assemble full 18-D from predicted angles + true other params
            # pred_angles: (B, 6) = [ax_norm(3), ay_norm(3)]
            ax_pred = pred_angles[:, :3] * RG_T[3:6] + LO_T[3:6]
            ay_pred = pred_angles[:, 3:] * RG_T[6:9] + LO_T[6:9]

            # true_full: (B, 18) denormalized
            full_pred = true_full.clone()
            full_pred[:, 3:6] = ax_pred
            full_pred[:, 6:9] = ay_pred

            # Physics loss: forward model comparison
            with torch.no_grad():
                target_pat = fwd(true_full, high_precision=False)
            pred_pat = fwd(full_pred, high_precision=False)
            phys_loss = F.mse_loss(pred_pat, target_pat)

            # Combined loss: physics-first
            loss = phys_loss + param_weight * param_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            tl_param += param_loss.item()
            tl_phys += phys_loss.item()
            nb += 1

        # Validation (param loss only for comparison with v1)
        model.eval()
        vl = 0; vn = 0
        with torch.no_grad():
            for ang_in, true_angles, _ in va_dl:
                ang_in = ang_in.to(DEVICE)
                true_angles = true_angles.to(DEVICE)
                vl += F.mse_loss(model(ang_in), true_angles).item()
                vn += 1

        vl /= vn
        improved = vl < best_val
        if improved:
            best_val = vl
            torch.save(model.state_dict(), ANG_V2)

        ep_dt = time.time() - t0
        eta = ep_dt / (ep + 1) * (epochs - ep - 1)
        marker = ' *' if improved else ''
        if (ep+1) % 5 == 0 or ep == 0:
            print(f"  Ep {ep+1:3d}/{epochs}  phys={tl_phys/nb:.6f} "
                  f"param={tl_param/nb:.6f} val={vl:.6f} "
                  f"best={best_val:.6f}{marker}  ETA {fmt_time(eta)}",
                  flush=True)

        if (ep+1) % 10 == 0:
            torch.save({'epoch': ep, 'model': model.state_dict(),
                        'opt': opt.state_dict(), 'sched': sched.state_dict(),
                        'best_val': best_val}, ANG_V2_CKPT)

    model.load_state_dict(torch.load(ANG_V2, map_location=DEVICE, weights_only=True))
    print(f"  Done: {fmt_time(time.time()-t0)}, best val={best_val:.6f}")
    return model


if __name__ == '__main__':
    print("="*70)
    print("MODEL v2: Physics-First AngleNet Training")
    print("="*70)

    # Load data
    print("\n--- Loading data ---")
    tp, ta, vp, va = load_or_gen_data()

    print("\n--- Extracting peak features ---")
    t0 = time.time()
    tr_peaks = extract_all_speeds(tp)
    va_peaks = extract_all_speeds(vp)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Build datasets
    # AngleNet input: speeds_norm(3) + peaks(25) = 28
    # AngleNet target: angles_norm(6)
    # Extra: full 18-D params (denormalized) for physics loss
    tr_speeds_n = ((ta[:, :3] - LO[:3]) / RG[:3]).astype(np.float32)
    va_speeds_n = ((va[:, :3] - LO[:3]) / RG[:3]).astype(np.float32)

    tr_ang_in = np.concatenate([tr_speeds_n, tr_peaks], axis=1)
    va_ang_in = np.concatenate([va_speeds_n, va_peaks], axis=1)

    tr_ang_tgt = np.concatenate([
        (ta[:, 3:6] - LO[3:6]) / RG[3:6],
        (ta[:, 6:9] - LO[6:9]) / RG[6:9],
    ], axis=1).astype(np.float32)
    va_ang_tgt = np.concatenate([
        (va[:, 3:6] - LO[3:6]) / RG[3:6],
        (va[:, 6:9] - LO[6:9]) / RG[6:9],
    ], axis=1).astype(np.float32)

    # Full 18-D params (denormalized) for assembling forward model input
    tr_full = ta.astype(np.float32)
    va_full = va.astype(np.float32)

    tr_ds = TensorDataset(
        torch.from_numpy(tr_ang_in),
        torch.from_numpy(tr_ang_tgt),
        torch.from_numpy(tr_full))
    va_ds = TensorDataset(
        torch.from_numpy(va_ang_in),
        torch.from_numpy(va_ang_tgt),
        torch.from_numpy(va_full))

    # Train
    print(f"\n--- Training AngleNet v2 (physics-first, 80 epochs) ---")
    fwd = DiffFwd().to(DEVICE)
    model = AngleNet()
    model = train_physics_first(model, tr_ds, va_ds, fwd,
                                epochs=80, lr=1e-3, param_weight=0.1)

    # Quick eval on 30 test cases
    print(f"\n--- Quick evaluation ---")
    from solve_preconditioned import vec2pat, ml_init
    from ml_staged_solver import (RemainNet, REM_MODEL,
        extract_speeds_and_peaks, _build_peak_feats_single, canon)
    from scipy.optimize import least_squares

    rem = RemainNet()
    rem.load_state_dict(torch.load(REM_MODEL, map_location=DEVICE, weights_only=True))
    rem.to(DEVICE)

    # Load v1 for comparison
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

    v1_perfect = 0; v2_perfect = 0

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

        for label, ang_net, counter_name in [('v1', ang_v1, 'v1_perfect'),
                                              ('v2', model, 'v2_perfect')]:
            best = 1e30; bx = None
            for bits in range(8):
                signs = np.array([(1.0 if (bits>>j)&1==0 else -1.0) for j in range(P)], np.float64)
                speeds = signs * np.sort(pf)[::-1].astype(np.float64)
                ml = ml_init(ang_net, rem, speeds.astype(np.float32), pk, pat)
                try:
                    res = least_squares(res_fn, ml[:9], jac='2-point',
                        bounds=(lo9, hi9), method='trf',
                        ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
                    mse = float(np.mean(res.fun**2))
                    if mse < best: best = mse; bx = res.x.copy()
                except: pass

            err = float(np.max(np.abs(bx - tc[:9]))) if bx is not None else 999
            if err < 1e-3:
                if label == 'v1': v1_perfect += 1
                else: v2_perfect += 1

        sys.stdout.write("."); sys.stdout.flush()

    print(f"\n\n{'='*70}")
    print(f"  AngleNet v1 (param loss):   {v1_perfect}/30 PERFECT")
    print(f"  AngleNet v2 (physics loss): {v2_perfect}/30 PERFECT")
    print(f"{'='*70}")
