#!/usr/bin/env python3
"""
ml_solver.py  --  Pure ML Risley Prism Inverse Solver

Trains a neural network to predict all 18 prism system parameters
directly from a scan pattern.  No differential evolution.  No grid
search.  No Nelder-Mead.  Pure learned prediction.

Optional: gradient-based polish through a differentiable forward model.

Progress is saved every 5 epochs.  Safe to Ctrl-C and resume.

Usage:
  python paper/ml_solver.py              # train (or resume) + evaluate
  python paper/ml_solver.py --retrain    # wipe cache, start fresh
  python paper/ml_solver.py --eval       # evaluate saved model only
  python paper/ml_solver.py --no-refine  # skip gradient polish step
"""

import sys, os, time, json, io, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'reverse_problem_v2'))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from core import PrismParameters, SystemGeometry, fast_forward

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("PyTorch required.  pip install torch")
    sys.exit(1)


# ================================================================
#  Configuration  (edit these to trade speed vs accuracy)
# ================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

P       = 3        # prisms
T_PTS   = 200      # time samples
T_OBS   = 10.0     # observation window (s)
SRC     = 6.0      # source distance (fixed)
THK     = 3.0      # prism thickness (fixed)

N_PAR   = 18
NAMES   = ['N1','N2','N3','ax1','ax2','ax3','ay1','ay2','ay3',
           'ng1','ng2','ng3','d_W','gap','bm_ax','bm_ay','bm_px','bm_py']
LO = np.array([-3.5]*3 + [-18.]*6 + [1.3]*3 + [50.,2.,-25.,-25.,-5.,-5.], dtype=np.float32)
HI = np.array([ 3.5]*3 + [ 18.]*6 + [1.8]*3 + [200.,15.,25.,25.,5.,5.],   dtype=np.float32)
RG = HI - LO

# Training hypers
N_TRAIN     = 200_000      # training samples  (increase for accuracy)
N_VAL       = 20_000       # validation samples
BATCH       = 512
EPOCHS      = 80
LR          = 1e-3
SAVE_EVERY  = 5            # checkpoint interval (epochs)
REF_STEPS   = 200          # gradient polish iterations
REF_LR      = 0.002

# File paths
MODEL_PATH = os.path.join(HERE, 'ml_model.pt')
CKPT_PATH  = os.path.join(HERE, 'ml_checkpoint.pt')
DATA_PATH  = os.path.join(HERE, 'ml_data.npz')


# ================================================================
#  Utilities
# ================================================================

def norm(v):   return (v - LO) / RG          # → [0,1]
def denorm(v): return v * RG + LO            # [0,1] → real

_LO_T = _RG_T = _HI_T = None
def _tensors():
    global _LO_T, _RG_T, _HI_T
    if _LO_T is None:
        _LO_T = torch.tensor(LO, device=DEVICE)
        _RG_T = torch.tensor(RG, device=DEVICE)
        _HI_T = torch.tensor(HI, device=DEVICE)

def canon(v):
    """Sort prisms by |speed| descending for unique ordering."""
    v = v.copy()
    o = np.argsort(-np.abs(v[:3]))
    v[:3]  = v[:3][o]
    v[3:6] = v[3:6][o]
    v[6:9] = v[6:9][o]
    v[9:12]= v[9:12][o]
    return v

def vec2pat(v):
    """18-D parameter vector → (200, 2) scan pattern."""
    geo = SystemGeometry(
        source_distance=SRC, prism_thickness=THK,
        workpiece_distance=float(v[12]), inter_prism_gap=float(v[13]),
        beam_angle_x=float(v[14]), beam_angle_y=float(v[15]),
        beam_pos_x=float(v[16]), beam_pos_y=float(v[17]))
    pr = PrismParameters(P, v[:3].tolist(), v[3:6].tolist(), v[6:9].tolist(),
                         glass_indices=v[9:12].tolist(), geometry=geo)
    return fast_forward(pr, T_PTS, T_OBS)

def fmt_time(s):
    m, s = divmod(int(s), 60)
    return f'{m}m{s:02d}s' if m else f'{s}s'


# ================================================================
#  Data Generation  (cached to disk)
# ================================================================

def gen_data(n, seed=42):
    """Generate n valid (pattern, param) pairs."""
    rng = np.random.default_rng(seed)
    pats, pars = [], []
    t0 = time.time()
    fails = 0
    while len(pats) < n:
        v = LO + RG * rng.random(N_PAR).astype(np.float32)
        for j in range(3):
            if abs(v[j]) < 0.15:
                v[j] = np.copysign(0.15, v[j])
        try:
            pat = vec2pat(v)
            if np.all(np.isfinite(pat)) and np.max(np.abs(pat)) < 5000:
                pats.append(pat)
                pars.append(canon(v))
            else:
                fails += 1
        except Exception:
            fails += 1
        if len(pats) % 25_000 == 0 and len(pats) > 0:
            el = time.time() - t0
            rate = len(pats) / el
            eta = (n - len(pats)) / rate
            print(f"    {len(pats):>8,} / {n:,}   "
                  f"{rate:.0f}/s   ETA {fmt_time(eta)}   "
                  f"({fails} rejected)", flush=True)
    dt = time.time() - t0
    print(f"    Done: {n:,} in {fmt_time(dt)}  ({n/dt:.0f}/s, "
          f"{fails} rejected)")
    return np.array(pats[:n], np.float32), np.array(pars[:n], np.float32)


def load_or_gen_data():
    """Load cached data or generate fresh."""
    if os.path.exists(DATA_PATH):
        print(f"  Loading cached data from {os.path.basename(DATA_PATH)}...",
              flush=True)
        d = np.load(DATA_PATH)
        tp, ta, vp, va = d['tp'], d['ta'], d['vp'], d['va']
        if len(tp) >= N_TRAIN and len(vp) >= N_VAL:
            print(f"    {len(tp):,} train + {len(vp):,} val loaded")
            return tp[:N_TRAIN], ta[:N_TRAIN], vp[:N_VAL], va[:N_VAL]
        print(f"    Cached data too small, regenerating...")

    print(f"\n  Generating training data ({N_TRAIN:,} samples)...")
    tp, ta = gen_data(N_TRAIN, seed=42)
    print(f"  Generating validation data ({N_VAL:,} samples)...")
    vp, va = gen_data(N_VAL, seed=123)

    print(f"  Saving to {os.path.basename(DATA_PATH)}...", end=' ', flush=True)
    np.savez(DATA_PATH, tp=tp, ta=ta, vp=vp, va=va)
    sz = os.path.getsize(DATA_PATH) / 1e6
    print(f"({sz:.0f} MB)")
    return tp, ta, vp, va


# ================================================================
#  Feature Extraction
# ================================================================

def make_feats(pats):
    """(N,200,2) → raw (N,2,200), spec (N,404), peaks (N,19)."""
    N = len(pats)
    raw = pats.transpose(0, 2, 1).astype(np.float32)   # (N, 2, 200)

    fx = np.fft.rfft(pats[:, :, 0], axis=1)            # (N, 101) complex
    fy = np.fft.rfft(pats[:, :, 1], axis=1)

    spec = np.concatenate([
        np.log1p(np.abs(fx)), np.log1p(np.abs(fy)),
        np.angle(fx),         np.angle(fy),
    ], axis=1).astype(np.float32)                       # (N, 404)

    # ---- Explicit peak features (vectorised) ----
    # Extract top-3 peaks, then SORT BY FREQUENCY DESCENDING
    # to match canonical parameter ordering (|speed| descending).
    # Per peak: freq, amp_x, amp_y, sin_px, cos_px, sin_py, cos_py = 7
    # 3 peaks × 7 + 4 summary = 25 features
    N_PEAK_FEAT = 25
    dt = T_OBS / T_PTS
    freqs = np.fft.rfftfreq(T_PTS, d=dt)               # (101,)
    power = (np.abs(fx) + np.abs(fy)).copy()
    power[:, 0] = 0                                     # kill DC

    # Extract 3 peaks (order-of-extraction)
    arange_N = np.arange(N)
    peak_freq = np.zeros((N, P), dtype=np.float32)
    peak_ax   = np.zeros((N, P), dtype=np.float32)
    peak_ay   = np.zeros((N, P), dtype=np.float32)
    peak_phx  = np.zeros((N, P), dtype=np.float32)
    peak_phy  = np.zeros((N, P), dtype=np.float32)

    for p_idx in range(P):
        idx = np.argmax(power, axis=1)
        peak_freq[:, p_idx] = freqs[idx]
        peak_ax[:, p_idx]   = np.abs(fx[arange_N, idx]) * 2 / T_PTS
        peak_ay[:, p_idx]   = np.abs(fy[arange_N, idx]) * 2 / T_PTS
        peak_phx[:, p_idx]  = np.angle(fx[arange_N, idx])
        peak_phy[:, p_idx]  = np.angle(fy[arange_N, idx])
        for delta in range(-3, 4):
            cols = np.clip(idx + delta, 0, power.shape[1] - 1)
            power[arange_N, cols] = 0

    # Sort peaks by frequency DESCENDING (matches canonical |speed| order)
    sort_idx = np.argsort(-peak_freq, axis=1)           # (N, 3)
    rows = arange_N[:, None]
    peak_freq = peak_freq[rows, sort_idx]
    peak_ax   = peak_ax[rows, sort_idx]
    peak_ay   = peak_ay[rows, sort_idx]
    peak_phx  = peak_phx[rows, sort_idx]
    peak_phy  = peak_phy[rows, sort_idx]

    # Encode phases as sin/cos (avoids ±π wrapping discontinuity)
    peaks = np.concatenate([
        peak_freq,                                       # 3: frequencies
        peak_ax, peak_ay,                                # 6: amplitudes
        np.sin(peak_phx), np.cos(peak_phx),             # 6: phase x (sin+cos)
        np.sin(peak_phy), np.cos(peak_phy),             # 6: phase y (sin+cos)
        np.mean(pats[:,:,0], axis=1, keepdims=True),    # 1: centroid x
        np.mean(pats[:,:,1], axis=1, keepdims=True),    # 1: centroid y
        np.std(pats[:,:,0], axis=1, keepdims=True),     # 1: spread x
        np.std(pats[:,:,1], axis=1, keepdims=True),     # 1: spread y
    ], axis=1).astype(np.float32)                        # (N, 25)

    return raw, spec, peaks


# ================================================================
#  Neural Network
# ================================================================

class Net(nn.Module):
    """Triple-branch: temporal CNN + spectral MLP + peak MLP → 18 parameters."""

    def __init__(self):
        super().__init__()
        # Input norms
        self.rbn = nn.BatchNorm1d(2)
        self.sbn = nn.BatchNorm1d(404)
        self.pbn = nn.BatchNorm1d(25)

        # Temporal branch: 1D CNN  (2, 200) → 128
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, 7, 2, 3),   nn.BatchNorm1d(32),  nn.GELU(),
            nn.Conv1d(32, 64, 5, 2, 2),  nn.BatchNorm1d(64),  nn.GELU(),
            nn.Conv1d(64, 128, 3, 2, 1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, 3, 2, 1),nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())             # → 128

        # Spectral branch: MLP  404 → 128
        self.sml = nn.Sequential(
            nn.Linear(404, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU())

        # Peak branch: MLP  25 → 128  (structured peak features)
        self.pml = nn.Sequential(
            nn.Linear(25, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU())

        # Fusion → params  (128+128+128=384) → 18  (sigmoid → [0,1])
        self.head = nn.Sequential(
            nn.Linear(384, 384), nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(384, 192), nn.BatchNorm1d(192), nn.GELU(),
            nn.Linear(192, N_PAR), nn.Sigmoid())

    def forward(self, raw, spec, peaks):
        t = self.cnn(self.rbn(raw))       # (B, 128)
        s = self.sml(self.sbn(spec))      # (B, 128)
        p = self.pml(self.pbn(peaks))     # (B, 128)
        return self.head(torch.cat([t, s, p], dim=1))


# ================================================================
#  Differentiable Forward Model  (for gradient polish)
# ================================================================

class DiffFwd(nn.Module):
    """PyTorch port of fast_forward — batched, differentiable."""

    def __init__(self):
        super().__init__()
        self.register_buffer('times',
            torch.arange(T_PTS, dtype=torch.float64) * (T_OBS / T_PTS))

    def forward(self, par):
        """par: (B, 18) real-valued → patterns (B, 200, 2)."""
        par = par.double()
        B   = par.shape[0]
        dev = par.device
        D   = torch.pi / 180.0

        sp, ax, ay, gl = par[:,:3], par[:,3:6], par[:,6:9], par[:,9:12]
        dW, gp = par[:,12:13], par[:,13:14]
        bax, bay, bpx, bpy = par[:,14], par[:,15], par[:,16], par[:,17]

        z = torch.zeros(B, device=dev, dtype=torch.float64)
        sif = torch.stack([sp[:,0],sp[:,0], sp[:,1],sp[:,1], sp[:,2],sp[:,2]], 1)
        spx = torch.stack([z,ax[:,0], z,ax[:,1], z,ax[:,2]], 1)
        spy = torch.stack([z,ay[:,0], z,ay[:,1], z,ay[:,2]], 1)

        o  = torch.ones(B, 1, device=dev, dtype=torch.float64)
        ri = torch.cat([o,gl[:,0:1], o,gl[:,1:2], o,gl[:,2:3], o], 1)

        g, d = gp.squeeze(1), dW.squeeze(1)
        s, k = SRC, THK
        cd = torch.stack([z+s, z+s+k, s+k+g, s+2*k+g,
                          s+2*k+2*g, s+3*k+2*g, s+3*k+2*g+d], 1)

        tt = self.times[None, None, :]                       # (1,1,T)
        gam  = (360.0 * sif[:,:,None] * tt) % 360.0
        gr   = (gam + spy[:,:,None]) * D
        pt   = torch.tan(spx[:,:,None] * D)

        n0 = torch.cos(gr) * pt
        n1 = torch.sin(gr) * pt
        nn_ = torch.sqrt(n0**2 + n1**2 + 1.0)
        pxe = 90.0 - torch.acos(torch.clamp(n0/nn_, -1+1e-7, 1-1e-7)) / D
        pye = 90.0 - torch.acos(torch.clamp(n1/nn_, -1+1e-7, 1-1e-7)) / D

        zz  = torch.zeros(B, 1, T_PTS, device=dev, dtype=torch.float64)
        pxa = torch.cat([pxe, zz], 1)
        pya = torch.cat([pye, zz], 1)

        x = self._trace(bpx, bax, pxa, cd, ri)
        y = self._trace(bpy, bay, pya, cd, ri)
        return torch.stack([x, y], 2).float()

    # --- ray tracing internals ---

    def _trace(self, r0, th0, phi, cd, ri):
        B  = r0.shape[0]
        NI = phi.shape[1] - 1
        T_ = phi.shape[2]
        D  = torch.pi / 180.0

        tt0 = torch.tan(th0 * D)[:, None].expand(-1, T_)
        r0e = r0[:, None].expand(-1, T_)

        p3, z3, p4, z4 = self._sf(phi[:,0,:], cd[:,0:1].expand(-1, T_))
        pc, zc = self._ix(r0e, torch.zeros_like(r0e),
                          r0e + tt0, torch.ones_like(r0e), p3, z3, p4, z4)
        tc = th0[:, None].expand(-1, T_).clone()

        for i in range(NI):
            tp = torch.tan(phi[:,i,:] * D)
            Nn = torch.sqrt(tp**2 + 1.0)
            Nh0, Nh2 = tp / Nn, -1.0 / Nn

            tt = torch.tan(tc * D)
            Sn = torch.sqrt(tt**2 + 1.0)
            Si0, Si2 = tt / Sn, 1.0 / Sn

            nr = ri[:,i:i+1] / ri[:,i+1:i+2]
            cy = Nh2 * Si0 - Nh0 * Si2
            sq = torch.clamp(1.0 - nr**2 * cy**2, min=0.0)
            sv = torch.sqrt(sq + 1e-30)

            sf0 = nr * Nh2 * cy - Nh0 * sv
            sf2 = -nr * Nh0 * cy - Nh2 * sv
            sfn = torch.sqrt(sf0**2 + sf2**2 + 1e-30)
            ca  = torch.clamp(sf2 / sfn, -1+1e-7, 1-1e-7)
            tn  = torch.sign(sf0) * torch.acos(ca) / D
            tc  = torch.where(torch.abs(sf0) < 1e-12, torch.zeros_like(sf0), tn)

            p3, z3, p4, z4 = self._sf(phi[:,i+1,:], cd[:,i+1:i+2].expand(-1, T_))
            pc, zc = self._ix(pc, zc, pc + torch.tan(tc*D), zc + 1.0, p3, z3, p4, z4)

        return pc

    def _sf(self, phi, z0):
        fl = torch.abs(phi) < 1e-10
        ps = torch.where(fl, torch.ones_like(phi), phi)
        D  = torch.pi / 180.0
        tp = torch.tan(ps * D)
        p4 = torch.where(fl, torch.ones_like(phi), 1.0 / tp)
        z4 = z0 + torch.where(fl, torch.zeros_like(phi), torch.ones_like(phi))
        return torch.zeros_like(phi), z0, p4, z4

    def _ix(self, p1, z1, p2, z2, p3, z3, p4, z4):
        d = (p1-p2)*(z3-z4) - (z1-z2)*(p3-p4)
        d = torch.where(torch.abs(d) < 1e-15, torch.ones_like(d)*1e-15, d)
        pn = ((p1*z2-z1*p2)*(p3-p4) - (p1-p2)*(p3*z4-z3*p4)) / d
        zn = ((p1*z2-z1*p2)*(z3-z4) - (z1-z2)*(p3*z4-z3*p4)) / d
        return pn, zn


# ================================================================
#  Training  (with checkpointing + resume)
# ================================================================

def do_train(model, tr_raw, tr_spec, tr_peaks, tr_par,
             va_raw, va_spec, va_peaks, va_par):
    """Train (or resume training) the model."""
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    tr_dl = DataLoader(
        TensorDataset(torch.from_numpy(tr_raw),
                       torch.from_numpy(tr_spec),
                       torch.from_numpy(tr_peaks),
                       torch.from_numpy(norm(tr_par))),
        batch_size=BATCH, shuffle=True,
        pin_memory=(DEVICE.type == 'cuda'))
    va_dl = DataLoader(
        TensorDataset(torch.from_numpy(va_raw),
                       torch.from_numpy(va_spec),
                       torch.from_numpy(va_peaks),
                       torch.from_numpy(norm(va_par))),
        batch_size=BATCH * 2,
        pin_memory=(DEVICE.type == 'cuda'))

    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(tr_dl))

    start_ep = 0
    best_val = float('inf')

    # Resume from checkpoint?
    if os.path.exists(CKPT_PATH):
        print(f"  Resuming from checkpoint...", flush=True)
        ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start_ep = ck['epoch'] + 1
        best_val = ck['best_val']
        print(f"    Resuming from epoch {start_ep}, best val={best_val:.6f}")

    n_batches = len(tr_dl)
    n_params  = sum(p.numel() for p in model.parameters())
    if start_ep == 0:
        print(f"  Model params: {n_params:,}")
        print(f"  Batches/epoch: {n_batches}")

    t0 = time.time()

    try:
        for ep in range(start_ep, EPOCHS):
            ep_t0 = time.time()

            # ---- train ----
            model.train()
            train_loss = 0.0
            for bi, (r, s, pk, p) in enumerate(tr_dl):
                r  = r.to(DEVICE)
                s  = s.to(DEVICE)
                pk = pk.to(DEVICE)
                p  = p.to(DEVICE)
                loss = F.mse_loss(model(r, s, pk), p)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                train_loss += loss.item()

            # ---- validate ----
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for r, s, pk, p in va_dl:
                    r  = r.to(DEVICE)
                    s  = s.to(DEVICE)
                    pk = pk.to(DEVICE)
                    p  = p.to(DEVICE)
                    val_loss += F.mse_loss(model(r, s, pk), p).item()

            train_loss /= n_batches
            val_loss   /= len(va_dl)

            # Save best model
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                torch.save(model.state_dict(), MODEL_PATH)

            ep_dt = time.time() - ep_t0
            elapsed = time.time() - t0
            remaining = ep_dt * (EPOCHS - ep - 1)
            lr_now = sched.get_last_lr()[0]

            marker = ' *' if improved else ''
            print(f"  Ep {ep+1:3d}/{EPOCHS}  "
                  f"tr={train_loss:.6f}  va={val_loss:.6f}  "
                  f"best={best_val:.6f}{marker}  "
                  f"lr={lr_now:.1e}  "
                  f"{ep_dt:.1f}s/ep  "
                  f"ETA {fmt_time(remaining)}",
                  flush=True)

            # Checkpoint
            if (ep + 1) % SAVE_EVERY == 0 or ep == EPOCHS - 1:
                torch.save({
                    'epoch': ep, 'model': model.state_dict(),
                    'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'best_val': best_val,
                }, CKPT_PATH)

    except KeyboardInterrupt:
        print(f"\n  Training interrupted at epoch {ep+1}.")
        torch.save({
            'epoch': ep, 'model': model.state_dict(),
            'opt': opt.state_dict(), 'sched': sched.state_dict(),
            'best_val': best_val,
        }, CKPT_PATH)
        print(f"  Checkpoint saved. Run again to resume.")

    # Load best weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    total = time.time() - t0
    print(f"  Training complete: {fmt_time(total)}, best val={best_val:.6f}")
    return model


# ================================================================
#  Inference
# ================================================================

@torch.no_grad()
def predict(model, pattern):
    """pattern (200,2) numpy → params (18,) numpy in real units."""
    model.eval()
    raw, spec, peaks = make_feats(pattern[None])
    r  = torch.tensor(raw,   dtype=torch.float32, device=DEVICE)
    s  = torch.tensor(spec,  dtype=torch.float32, device=DEVICE)
    pk = torch.tensor(peaks, dtype=torch.float32, device=DEVICE)
    p_norm = model(r, s, pk).cpu().numpy()[0]
    return np.clip(denorm(p_norm), LO, HI)


_fwd_cache = [None]

def refine(pred_vec, target_pat, steps=REF_STEPS, lr=REF_LR):
    """Gradient polish through differentiable physics."""
    _tensors()
    if _fwd_cache[0] is None:
        _fwd_cache[0] = DiffFwd().to(DEVICE)
    fwd = _fwd_cache[0]

    tgt = torch.tensor(target_pat, dtype=torch.float32, device=DEVICE)
    par = torch.tensor(pred_vec, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    opt = torch.optim.Adam([par], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    for i in range(steps):
        opt.zero_grad()
        pred = fwd(par[None]).squeeze(0)
        loss = F.mse_loss(pred, tgt)
        loss.backward()
        opt.step()
        sch.step()
        with torch.no_grad():
            par.clamp_(_LO_T, _HI_T)

    return par.detach().cpu().numpy()


# ================================================================
#  Evaluation Display
# ================================================================

def show_result(pred, true, indent='    '):
    """Print per-parameter comparison table."""
    err = np.abs(pred - true)
    try:
        pp = vec2pat(pred)
        tp = vec2pat(true)
        mse = float(np.mean((pp - tp)**2))
        print(f"{indent}Pattern MSE: {mse:.2e}")
    except Exception:
        print(f"{indent}Pattern MSE: (forward model failed)")

    for i in range(N_PAR):
        pct = 100.0 * err[i] / abs(RG[i])
        q = ('OK' if pct < 1 else 'CLOSE' if pct < 5
             else 'MEH' if pct < 10 else 'BAD')
        print(f"{indent}{NAMES[i]:6s}  pred={pred[i]:9.4f}  "
              f"true={true[i]:9.4f}  err={err[i]:.2e} "
              f"({pct:5.1f}%)  [{q}]")


# ================================================================
#  Main
# ================================================================

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrain',   action='store_true', help='wipe all and retrain')
    ap.add_argument('--eval',      action='store_true', help='evaluate only')
    ap.add_argument('--no-refine', action='store_true', help='skip gradient polish')
    args = ap.parse_args()

    _tensors()

    print(f"Device: {DEVICE}")
    print('=' * 70)
    print(' PURE ML RISLEY INVERSE SOLVER')
    print('=' * 70)

    # --retrain: wipe cached files
    if args.retrain:
        for f in [MODEL_PATH, CKPT_PATH, DATA_PATH]:
            if os.path.exists(f):
                os.remove(f)
                print(f"  Deleted {os.path.basename(f)}")

    model = Net()

    # ---- Training ----
    if args.eval:
        if not os.path.exists(MODEL_PATH):
            print("  No saved model. Train first (run without --eval).")
            sys.exit(1)
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        print("  Loaded saved model.")
    else:
        need_train = (not os.path.exists(MODEL_PATH)
                      or os.path.exists(CKPT_PATH))  # resume partial

        if need_train:
            print(f"\n--- Data ---")
            tp, ta, vp, va = load_or_gen_data()

            print(f"\n--- Features ---")
            t0 = time.time()
            tr_raw, tr_spec, tr_peaks = make_feats(tp)
            va_raw, va_spec, va_peaks = make_feats(vp)
            print(f"  Extracted in {time.time()-t0:.1f}s  "
                  f"(raw: {tr_raw.shape}, spec: {tr_spec.shape}, "
                  f"peaks: {tr_peaks.shape})")

            print(f"\n--- Training ({EPOCHS} epochs, {N_TRAIN:,} samples) ---")
            model = do_train(model,
                             tr_raw, tr_spec, tr_peaks, ta,
                             va_raw, va_spec, va_peaks, va)

            # Clean up checkpoint after successful completion
            if os.path.exists(CKPT_PATH):
                os.remove(CKPT_PATH)
                print("  Checkpoint cleaned up (training complete).")

            del tp, ta, vp, va
            del tr_raw, tr_spec, tr_peaks, va_raw, va_spec, va_peaks
        else:
            print(f"\n  Loading saved model.")
            model.load_state_dict(
                torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
            model.to(DEVICE)

    # ---- Verify DiffForward ----
    print(f"\n--- DiffForward Verification ---")
    fwd = DiffFwd().to(DEVICE)
    tv = np.array([1.5, -1., 2.,  12., -8., 5.,  3., 10., -6.,
                   1.5, 1.55, 1.6,  100., 6.,  10., 5.,  0., 0.],
                  dtype=np.float32)
    with torch.no_grad():
        pt = fwd(torch.tensor(tv, device=DEVICE)[None]).squeeze(0).cpu().numpy()
    pn = vec2pat(tv)
    diff = np.max(np.abs(pt - pn))
    print(f"  Max |PyTorch - NumPy|: {diff:.2e}  "
          f"({'OK' if diff < 1e-4 else 'CHECK THIS'})")

    # ---- Paper test case ----
    print(f"\n{'='*70}")
    print(" TEST: Paper configuration (18-D full recovery)")
    print('=' * 70)

    true_c = canon(tv)
    pattern = vec2pat(tv)

    t0 = time.time()
    pred = predict(model, pattern)
    t_ms = (time.time() - t0) * 1000

    print(f"\n  Neural prediction ({t_ms:.0f} ms):")
    show_result(pred, true_c)

    if not args.no_refine:
        print(f"\n  Gradient polish ({REF_STEPS} steps)...")
        t0 = time.time()
        ref = refine(pred, pattern)
        t_ref = time.time() - t0
        print(f"  After polish ({t_ref:.1f}s):")
        show_result(ref, true_c)

    # ---- Random test battery ----
    print(f"\n{'='*70}")
    print(' TEST: 100 random configurations')
    print('=' * 70)

    rng_t = np.random.default_rng(999)
    errs, mses = [], []
    t0 = time.time()
    for _ in range(100):
        v = LO + RG * rng_t.random(N_PAR).astype(np.float32)
        for j in range(3):
            if abs(v[j]) < 0.15:
                v[j] = np.copysign(0.15, v[j])
        try:
            tc = canon(v)
            pat = vec2pat(v)
            pr  = predict(model, pat)
            errs.append(np.abs(pr - tc))
            pp  = vec2pat(pr)
            mses.append(float(np.mean((pp - pat)**2)))
        except Exception:
            pass
    t100 = time.time() - t0
    errs = np.array(errs)

    ms_per = t100 / len(errs) * 1000
    print(f"\n  {len(errs)} cases in {t100:.1f}s  ({ms_per:.0f} ms/case)")
    print(f"\n  {'Param':8s} {'Median':>10s} {'Mean':>10s} "
          f"{'Max':>10s} {'MeanPct':>8s}")
    for i in range(N_PAR):
        med = np.median(errs[:, i])
        mean = np.mean(errs[:, i])
        mx = np.max(errs[:, i])
        pct = 100.0 * mean / abs(RG[i])
        print(f"  {NAMES[i]:8s} {med:10.4f} {mean:10.4f} "
              f"{mx:10.4f} {pct:7.1f}%")

    print(f"\n  Pattern MSE:  median={np.median(mses):.2e}  "
          f"mean={np.mean(mses):.2e}  max={np.max(mses):.2e}")

    # ---- Save results ----
    results = {
        'method': 'Pure ML (neural network + optional gradient polish)',
        'n_train': N_TRAIN, 'n_epochs': EPOCHS,
        'device': str(DEVICE),
        'inference_ms': round(ms_per, 1),
        'per_param_mean_error': {
            NAMES[i]: round(float(np.mean(errs[:,i])), 6) for i in range(N_PAR)},
        'per_param_mean_pct': {
            NAMES[i]: round(float(100*np.mean(errs[:,i])/abs(RG[i])), 2)
            for i in range(N_PAR)},
        'pattern_mse_median': float(np.median(mses)),
        'pattern_mse_mean': float(np.mean(mses)),
    }
    rpath = os.path.join(HERE, 'ml_results.json')
    with open(rpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {os.path.basename(rpath)}")

    # ---- Comparison ----
    print(f"\n{'='*70}")
    print(' SPEED COMPARISON')
    print('=' * 70)
    print(f'  Differential Evolution (old):  100 - 4000 s per case')
    print(f'  Neural Network (this):         {ms_per:.0f} ms per case')
    if ms_per > 0:
        print(f'  Speedup:                       ~{int(2000 / (ms_per/1000)):,}x')
    print()
