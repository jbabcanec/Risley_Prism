#!/usr/bin/env python3
"""
ml_staged_solver.py  --  Staged ML Risley Prism Inverse Solver

Three-stage prediction pipeline:
  Stage 1: Speeds from FFT peaks (analytical, <1 ms)
  Stage 2: Angles from neural net conditioned on speeds (~2 ms)
  Stage 3: Glass/geometry/beam from neural net conditioned on speeds+angles (~2 ms)

Then: gradient polish through differentiable forward model (optional, ~2 s)

Total: ~5 ms prediction, ~2 s with polish.  No brute force anywhere.

Usage:
  python paper/ml_staged_solver.py              # train + evaluate
  python paper/ml_staged_solver.py --retrain    # wipe and retrain
  python paper/ml_staged_solver.py --eval       # evaluate only
"""

import sys, os, time, json, io, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'reverse_problem_v2'))
if hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

from core import PrismParameters, SystemGeometry, fast_forward

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("PyTorch required.  pip install torch"); sys.exit(1)


# ================================================================
#  Config
# ================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

P       = 3
T_PTS   = 200
T_OBS   = 10.0
SRC     = 6.0
THK     = 3.0
DT      = T_OBS / T_PTS
FREQS   = np.fft.rfftfreq(T_PTS, d=DT)   # (101,)

N_PAR   = 18
NAMES   = ['N1','N2','N3','ax1','ax2','ax3','ay1','ay2','ay3',
           'ng1','ng2','ng3','d_W','gap','bm_ax','bm_ay','bm_px','bm_py']
LO = np.array([-3.5]*3+[-18.]*6+[1.3]*3+[50.,2.,-25.,-25.,-5.,-5.], dtype=np.float32)
HI = np.array([ 3.5]*3+[ 18.]*6+[1.8]*3+[200.,15.,25.,25.,5.,5.],   dtype=np.float32)
RG = HI - LO

# Training
N_TRAIN     = 1_000_000
N_VAL       = 50_000
BATCH       = 512
EPOCHS_ANG  = 80       # angle network
EPOCHS_REM  = 100      # remainder network
LR          = 1e-3
REF_STEPS   = 300
REF_LR      = 0.005
PHYSICS_EVERY = 4      # compute physics loss every N batches
PHYSICS_WEIGHT = 0.1   # weight of physics loss relative to param loss

# Paths
DATA_PATH   = os.path.join(HERE, 'ml_data_1m.npz')
ANG_MODEL   = os.path.join(HERE, 'ml_stage2_angles.pt')
REM_MODEL   = os.path.join(HERE, 'ml_stage3_remain.pt')
ANG_CKPT    = os.path.join(HERE, 'ml_stage2_ckpt.pt')
REM_CKPT    = os.path.join(HERE, 'ml_stage3_ckpt.pt')


# ================================================================
#  Utilities
# ================================================================

def fmt_time(s):
    m, s = divmod(int(s), 60)
    return f'{m}m{s:02d}s' if m else f'{s}s'

def canon(v):
    v = v.copy()
    o = np.argsort(-np.abs(v[:3]))
    v[:3], v[3:6], v[6:9], v[9:12] = v[:3][o], v[3:6][o], v[6:9][o], v[9:12][o]
    return v

def vec2pat(v):
    geo = SystemGeometry(
        source_distance=SRC, prism_thickness=THK,
        workpiece_distance=float(v[12]), inter_prism_gap=float(v[13]),
        beam_angle_x=float(v[14]), beam_angle_y=float(v[15]),
        beam_pos_x=float(v[16]), beam_pos_y=float(v[17]))
    pr = PrismParameters(P, v[:3].tolist(), v[3:6].tolist(), v[6:9].tolist(),
                         glass_indices=v[9:12].tolist(), geometry=geo)
    return fast_forward(pr, T_PTS, T_OBS)


# ================================================================
#  Stage 1: Speed extraction (analytical — no ML)
# ================================================================

def extract_speeds_and_peaks(pattern):
    """
    FFT peak extraction.  Returns top-P peaks by power.

    Also returns all_peak_freqs (top-6) for multi-triple search.
    """
    fx = np.fft.rfft(pattern[:, 0])
    fy = np.fft.rfft(pattern[:, 1])
    power = (np.abs(fx) + np.abs(fy)).copy()
    power[0] = 0

    K = min(8, len(power) - 1)
    all_idxs = []
    pw_work = power.copy()
    for _ in range(K):
        idx = np.argmax(pw_work)
        if pw_work[idx] <= 0:
            break
        all_idxs.append(idx)
        lo, hi = max(1, idx - 2), min(len(pw_work), idx + 3)
        pw_work[lo:hi] = 0

    # Primary: top-P by power
    idxs = sorted(all_idxs[:P], key=lambda i: FREQS[i], reverse=True)

    peak_freqs = np.array([FREQS[i] for i in idxs], dtype=np.float32)
    amp_x  = np.array([np.abs(fx[i]) * 2 / T_PTS for i in idxs], dtype=np.float32)
    amp_y  = np.array([np.abs(fy[i]) * 2 / T_PTS for i in idxs], dtype=np.float32)
    phase_x = np.array([np.angle(fx[i]) for i in idxs], dtype=np.float32)
    phase_y = np.array([np.angle(fy[i]) for i in idxs], dtype=np.float32)

    centroid_x = float(np.mean(pattern[:, 0]))
    centroid_y = float(np.mean(pattern[:, 1]))

    return peak_freqs, {
        'amp_x': amp_x, 'amp_y': amp_y,
        'phase_x': phase_x, 'phase_y': phase_y,
        'centroid_x': centroid_x, 'centroid_y': centroid_y,
        'all_peak_freqs': np.array([FREQS[i] for i in all_idxs], dtype=np.float32),
    }


def resolve_speed_signs(peak_freqs, peak_info, pattern, ang_net=None, peak_feats=None):
    """
    Try all 2^P=8 sign combinations.  For each, use the angle network
    to predict angles, then evaluate the FULL forward model.
    Pick the combo with lowest pattern MSE.
    """
    best_speeds = peak_freqs.copy()
    best_mse = float('inf')

    for bits in range(2**P):
        signs = np.array([(1.0 if (bits >> i) & 1 == 0 else -1.0)
                          for i in range(P)], dtype=np.float32)
        speeds = signs * peak_freqs

        try:
            v = np.zeros(N_PAR, dtype=np.float32)
            v[:3] = speeds

            # Use angle network if available (much more accurate)
            if ang_net is not None and peak_feats is not None:
                speeds_norm = (speeds - LO[:3]) / RG[:3]
                ang_in = np.concatenate([speeds_norm, peak_feats]).astype(np.float32)
                with torch.no_grad():
                    ang_t = torch.tensor(ang_in[None], device=DEVICE)
                    ang_pred = ang_net(ang_t).cpu().numpy()[0]
                v[3:6] = ang_pred[:3] * RG[3:6] + LO[3:6]
                v[6:9] = ang_pred[3:] * RG[6:9] + LO[6:9]
            else:
                # Fallback: paraxial estimates
                for i in range(P):
                    d_eff = (P-1-i) * (THK+6.0) + 100.0
                    alpha_rad = peak_info['amp_x'][i] / (d_eff * 0.5 + 1e-10)
                    v[3+i] = np.degrees(alpha_rad)
                    v[6+i] = np.degrees(peak_info['phase_x'][i])

            v[9:12] = [1.5, 1.5, 1.5]
            v[12] = 100.0; v[13] = 6.0
            v[14] = np.degrees(np.arctan2(peak_info['centroid_x'], 100.0))
            v[15] = np.degrees(np.arctan2(peak_info['centroid_y'], 100.0))

            pat = vec2pat(v)
            mse = float(np.mean((pat - pattern)**2))
            if mse < best_mse:
                best_mse = mse
                best_speeds = speeds.copy()
        except Exception:
            pass

    return best_speeds


# ================================================================
#  Stage 1 (batch): extract speeds for entire dataset
# ================================================================

def extract_all_speeds(patterns):
    """Batch speed extraction for training data."""
    N = len(patterns)
    speeds_out = np.zeros((N, 3), dtype=np.float32)
    peak_feats = np.zeros((N, 19), dtype=np.float32)  # 3*(freq,ax,ay,sin_px,cos_px,sin_py,cos_py) - wait let me restructure

    # For training we use TRUE speeds (teacher forcing).
    # For inference we use extract_speeds_and_peaks + resolve_speed_signs.
    # Here we just extract peak features for the angle network input.

    fx = np.fft.rfft(patterns[:, :, 0], axis=1)
    fy = np.fft.rfft(patterns[:, :, 1], axis=1)
    power = (np.abs(fx) + np.abs(fy)).copy()
    power[:, 0] = 0
    arange_N = np.arange(N)

    p_freq = np.zeros((N, P), dtype=np.float32)
    p_ax   = np.zeros((N, P), dtype=np.float32)
    p_ay   = np.zeros((N, P), dtype=np.float32)
    p_phx  = np.zeros((N, P), dtype=np.float32)
    p_phy  = np.zeros((N, P), dtype=np.float32)

    for p_idx in range(P):
        idx = np.argmax(power, axis=1)
        p_freq[:, p_idx] = FREQS[idx]
        p_ax[:, p_idx]   = np.abs(fx[arange_N, idx]) * 2 / T_PTS
        p_ay[:, p_idx]   = np.abs(fy[arange_N, idx]) * 2 / T_PTS
        p_phx[:, p_idx]  = np.angle(fx[arange_N, idx])
        p_phy[:, p_idx]  = np.angle(fy[arange_N, idx])
        for delta in range(-3, 4):
            cols = np.clip(idx + delta, 0, power.shape[1] - 1)
            power[arange_N, cols] = 0

    # Sort by frequency descending
    sort_idx = np.argsort(-p_freq, axis=1)
    rows = arange_N[:, None]
    p_freq = p_freq[rows, sort_idx]
    p_ax   = p_ax[rows, sort_idx]
    p_ay   = p_ay[rows, sort_idx]
    p_phx  = p_phx[rows, sort_idx]
    p_phy  = p_phy[rows, sort_idx]

    # Build feature vector: 3 freqs + 6 amps + 6 sin/cos phases + 4 summary = 25
    cx = np.mean(patterns[:, :, 0], axis=1, keepdims=True)
    cy = np.mean(patterns[:, :, 1], axis=1, keepdims=True)
    sx = np.std(patterns[:, :, 0], axis=1, keepdims=True)
    sy = np.std(patterns[:, :, 1], axis=1, keepdims=True)

    peak_feats = np.concatenate([
        p_freq, p_ax, p_ay,
        np.sin(p_phx), np.cos(p_phx),
        np.sin(p_phy), np.cos(p_phy),
        cx, cy, sx, sy,
    ], axis=1).astype(np.float32)   # (N, 25)

    return peak_feats


# ================================================================
#  Stage 2 Network: Angles conditioned on speeds
# ================================================================

class AngleNet(nn.Module):
    """
    Input:  3 speeds (signed) + 25 peak features = 28
    Output: 6 angles (3 α_x + 3 α_y), normalized to [0,1]
    """
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(28)
        self.net = nn.Sequential(
            nn.Linear(28, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, 6), nn.Sigmoid())

    def forward(self, x):
        return self.net(self.bn(x))


# ================================================================
#  Stage 3 Network: Remaining params conditioned on speeds+angles
# ================================================================

class RemainNet(nn.Module):
    """
    Input:  raw pattern (2×200) + 9 known params (speeds+angles) + 25 peak features
    Output: 9 remaining params (glass×3, d_W, gap, beam_ax, beam_ay, beam_px, beam_py)
            normalized to [0,1]
    """
    def __init__(self):
        super().__init__()
        # CNN over raw pattern
        self.rbn = nn.BatchNorm1d(2)
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, 7, 2, 3),   nn.BatchNorm1d(32),  nn.GELU(),
            nn.Conv1d(32, 64, 5, 2, 2),  nn.BatchNorm1d(64),  nn.GELU(),
            nn.Conv1d(64, 128, 3, 2, 1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, 3, 2, 1),nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())             # → 128

        # Conditioning MLP: speeds(3) + angles(6) + peaks(25) = 34
        self.cond_bn = nn.BatchNorm1d(34)
        self.cond = nn.Sequential(
            nn.Linear(34, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU())

        # Fusion → 9 params
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, 9), nn.Sigmoid())

    def forward(self, raw, cond):
        r = self.cnn(self.rbn(raw))          # (B, 128)
        c = self.cond(self.cond_bn(cond))    # (B, 128)
        return self.head(torch.cat([r, c], dim=1))


# ================================================================
#  Differentiable Forward Model (from ml_solver.py)
# ================================================================

class DiffFwd(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('times',
            torch.arange(T_PTS, dtype=torch.float64) * DT)

    def forward(self, par, high_precision=True):
        dt = torch.float64 if high_precision else par.dtype
        par = par.to(dt); B = par.shape[0]; dev = par.device; D = torch.pi/180.
        sp,ax,ay,gl = par[:,:3],par[:,3:6],par[:,6:9],par[:,9:12]
        dW,gp = par[:,12:13],par[:,13:14]
        bax,bay,bpx,bpy = par[:,14],par[:,15],par[:,16],par[:,17]
        z = torch.zeros(B, device=dev, dtype=dt)
        sif = torch.stack([sp[:,0],sp[:,0],sp[:,1],sp[:,1],sp[:,2],sp[:,2]],1)
        spx = torch.stack([z,ax[:,0],z,ax[:,1],z,ax[:,2]],1)
        spy = torch.stack([z,ay[:,0],z,ay[:,1],z,ay[:,2]],1)
        o = torch.ones(B,1,device=dev,dtype=dt)
        ri = torch.cat([o,gl[:,0:1],o,gl[:,1:2],o,gl[:,2:3],o],1)
        g,d = gp.squeeze(1),dW.squeeze(1); s,k = SRC,THK
        cd = torch.stack([z+s,z+s+k,s+k+g,s+2*k+g,s+2*k+2*g,s+3*k+2*g,s+3*k+2*g+d],1)
        tt = self.times.to(dt)[None,None,:]
        gam = (360.*sif[:,:,None]*tt)%360.
        gr = (gam+spy[:,:,None])*D
        pt = torch.tan(spx[:,:,None]*D)
        n0=torch.cos(gr)*pt; n1=torch.sin(gr)*pt
        nn_=torch.sqrt(n0**2+n1**2+1.)
        pxe=90.-torch.acos(torch.clamp(n0/nn_,-1+1e-7,1-1e-7))/D
        pye=90.-torch.acos(torch.clamp(n1/nn_,-1+1e-7,1-1e-7))/D
        zz=torch.zeros(B,1,T_PTS,device=dev,dtype=dt)
        pxa=torch.cat([pxe,zz],1); pya=torch.cat([pye,zz],1)
        x=self._tr(bpx,bax,pxa,cd,ri); y=self._tr(bpy,bay,pya,cd,ri)
        return torch.stack([x,y],2)  # preserve dtype (float32 or float64)

    def _tr(self, r0, th0, phi, cd, ri):
        B=r0.shape[0]; NI=phi.shape[1]-1; T_=phi.shape[2]; D=torch.pi/180.
        tt0=torch.tan(th0*D)[:,None].expand(-1,T_); r0e=r0[:,None].expand(-1,T_)
        p3,z3,p4,z4=self._sf(phi[:,0,:],cd[:,0:1].expand(-1,T_))
        pc,zc=self._ix(r0e,torch.zeros_like(r0e),r0e+tt0,torch.ones_like(r0e),p3,z3,p4,z4)
        tc=th0[:,None].expand(-1,T_).clone()
        for i in range(NI):
            tp=torch.tan(phi[:,i,:]*D); Nn=torch.sqrt(tp**2+1.)
            Nh0,Nh2=tp/Nn,-1./Nn
            tt=torch.tan(tc*D); Sn=torch.sqrt(tt**2+1.)
            Si0,Si2=tt/Sn,1./Sn
            nr=ri[:,i:i+1]/ri[:,i+1:i+2]; cy=Nh2*Si0-Nh0*Si2
            sq=torch.clamp(1.-nr**2*cy**2,min=0.); sv=torch.sqrt(sq+1e-30)
            sf0=nr*Nh2*cy-Nh0*sv; sf2=-nr*Nh0*cy-Nh2*sv
            sfn=torch.sqrt(sf0**2+sf2**2+1e-30)
            ca=torch.clamp(sf2/sfn,-1+1e-7,1-1e-7)
            tn=torch.sign(sf0)*torch.acos(ca)/D
            tc=torch.where(torch.abs(sf0)<1e-12,torch.zeros_like(sf0),tn)
            p3,z3,p4,z4=self._sf(phi[:,i+1,:],cd[:,i+1:i+2].expand(-1,T_))
            pc,zc=self._ix(pc,zc,pc+torch.tan(tc*D),zc+1.,p3,z3,p4,z4)
        return pc

    def _sf(self, phi, z0):
        fl=torch.abs(phi)<1e-10; ps=torch.where(fl,torch.ones_like(phi),phi)
        D=torch.pi/180.; tp=torch.tan(ps*D)
        return (torch.zeros_like(phi),z0,
                torch.where(fl,torch.ones_like(phi),1./tp),
                z0+torch.where(fl,torch.zeros_like(phi),torch.ones_like(phi)))

    def _ix(self, p1,z1,p2,z2,p3,z3,p4,z4):
        d=(p1-p2)*(z3-z4)-(z1-z2)*(p3-p4)
        d=torch.where(torch.abs(d)<1e-15,torch.ones_like(d)*1e-15,d)
        return (((p1*z2-z1*p2)*(p3-p4)-(p1-p2)*(p3*z4-z3*p4))/d,
                ((p1*z2-z1*p2)*(z3-z4)-(z1-z2)*(p3*z4-z3*p4))/d)


# ================================================================
#  Data Loading
# ================================================================

def load_or_gen_data():
    if os.path.exists(DATA_PATH):
        print(f"  Loading cached data...", flush=True)
        d = np.load(DATA_PATH)
        tp, ta, vp, va = d['tp'], d['ta'], d['vp'], d['va']
        if len(tp) >= N_TRAIN and len(vp) >= N_VAL:
            print(f"    {len(tp):,} train + {len(vp):,} val loaded")
            return tp[:N_TRAIN], ta[:N_TRAIN], vp[:N_VAL], va[:N_VAL]

    from ml_solver import gen_data
    print(f"  Generating training data ({N_TRAIN:,})...")
    tp, ta = gen_data(N_TRAIN, seed=42)
    print(f"  Generating validation data ({N_VAL:,})...")
    vp, va = gen_data(N_VAL, seed=123)
    np.savez(DATA_PATH, tp=tp, ta=ta, vp=vp, va=va)
    return tp, ta, vp, va


# ================================================================
#  Generic Training Loop
# ================================================================

def train_net(model, train_ds, val_ds, model_path, ckpt_path,
              epochs, batch=BATCH, lr=LR, physics_ctx=None):
    """
    Generic training loop.  If physics_ctx is provided (Stage 3 only),
    adds a physics-informed loss every PHYSICS_EVERY batches:
      assemble full 18-D params → DiffForward → compare with input pattern.

    physics_ctx = {
        'fwd': DiffFwd instance,
        'lo': LO tensor, 'rg': RG tensor,   # for denormalization
    }
    """
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tr_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                       pin_memory=(DEVICE.type=='cuda'))
    va_dl = DataLoader(val_ds, batch_size=batch*2,
                       pin_memory=(DEVICE.type=='cuda'))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(tr_dl))

    start_ep = 0
    best_val = float('inf')

    if os.path.exists(ckpt_path):
        print(f"    Resuming from checkpoint...", flush=True)
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start_ep = ck['epoch'] + 1
        best_val = ck['best_val']
        print(f"    Resuming epoch {start_ep}, best={best_val:.6f}")

    n_params = sum(p.numel() for p in model.parameters())
    nb = len(tr_dl)
    if start_ep == 0:
        mode = "param + physics" if physics_ctx else "param only"
        print(f"    Params: {n_params:,}, batches/ep: {nb}, loss: {mode}")

    # Precompute denorm tensors for physics loss
    if physics_ctx:
        fwd = physics_ctx['fwd']
        lo_t = torch.tensor(LO, device=DEVICE, dtype=torch.float32)
        rg_t = torch.tensor(RG, device=DEVICE, dtype=torch.float32)

    t0 = time.time()
    try:
        for ep in range(start_ep, epochs):
            model.train(); tl = 0; pl = 0; p_count = 0
            for bi, batch_data in enumerate(tr_dl):
                inputs = [x.to(DEVICE) for x in batch_data[:-1]]
                target = batch_data[-1].to(DEVICE)
                pred = model(*inputs)
                loss = F.mse_loss(pred, target)

                # Physics loss: assemble predicted params, run DiffForward
                if physics_ctx and bi % PHYSICS_EVERY == 0:
                    # inputs[0] = raw pattern (B, 2, 200)
                    # inputs[1] = cond (B, 34): speeds_n(3) + angles_n(6) + peaks(25)
                    raw = inputs[0]
                    cond = inputs[1]
                    tgt_pat = raw.transpose(1, 2)    # (B, 200, 2)

                    # Denormalize known params from conditioning
                    speeds = cond[:, :3] * rg_t[:3] + lo_t[:3]
                    ax = cond[:, 3:6] * rg_t[3:6] + lo_t[3:6]
                    ay = cond[:, 6:9] * rg_t[6:9] + lo_t[6:9]

                    # Denormalize predicted remaining
                    remaining = pred * rg_t[9:] + lo_t[9:]

                    # Assemble full 18-D
                    full = torch.cat([speeds, ax, ay, remaining], dim=1)
                    pred_pat = fwd(full, high_precision=False)

                    p_loss = F.mse_loss(pred_pat, tgt_pat)
                    loss = loss + PHYSICS_WEIGHT * p_loss
                    pl += p_loss.item(); p_count += 1

                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                tl += loss.item()

            model.eval(); vl = 0
            with torch.no_grad():
                for batch_data in va_dl:
                    inputs = [x.to(DEVICE) for x in batch_data[:-1]]
                    target = batch_data[-1].to(DEVICE)
                    vl += F.mse_loss(model(*inputs), target).item()

            tl /= nb; vl /= len(va_dl)
            improved = vl < best_val
            if improved:
                best_val = vl
                torch.save(model.state_dict(), model_path)

            ep_dt = time.time() - t0
            eta = ep_dt / (ep - start_ep + 1) * (epochs - ep - 1)
            marker = ' *' if improved else ''
            phys_str = f"  phys={pl/max(p_count,1):.4f}" if p_count else ""
            if (ep+1) % 5 == 0 or ep == start_ep:
                print(f"    Ep {ep+1:3d}/{epochs}  tr={tl:.6f} va={vl:.6f} "
                      f"best={best_val:.6f}{marker}{phys_str}  "
                      f"ETA {fmt_time(eta)}", flush=True)

            if (ep+1) % 10 == 0:
                torch.save({'epoch': ep, 'model': model.state_dict(),
                            'opt': opt.state_dict(), 'sched': sched.state_dict(),
                            'best_val': best_val}, ckpt_path)

    except KeyboardInterrupt:
        print(f"\n    Interrupted at epoch {ep+1}.")
        torch.save({'epoch': ep, 'model': model.state_dict(),
                    'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'best_val': best_val}, ckpt_path)

    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    print(f"    Done: {fmt_time(time.time()-t0)}, best val={best_val:.6f}")
    return model


# ================================================================
#  Full Pipeline: predict all 18 params from a pattern
# ================================================================

@torch.no_grad()
def predict(ang_net, rem_net, pattern):
    """
    Full staged prediction.
    pattern: (200, 2) numpy
    Returns: (18,) numpy parameter vector in real units.
    """
    ang_net.eval(); rem_net.eval()

    # Stage 1: speeds from FFT + sign resolution via angle network
    peak_freqs, peak_info = extract_speeds_and_peaks(pattern)
    peak_feats = _build_peak_feats_single(pattern, peak_freqs, peak_info)
    speeds = resolve_speed_signs(peak_freqs, peak_info, pattern,
                                 ang_net=ang_net, peak_feats=peak_feats)

    # Stage 2: angles from network
    # Normalize speeds to [0,1]
    speeds_norm = (speeds - LO[:3]) / RG[:3]
    ang_input = np.concatenate([speeds_norm, peak_feats]).astype(np.float32)
    ang_t = torch.tensor(ang_input[None], device=DEVICE)
    ang_pred = ang_net(ang_t).cpu().numpy()[0]  # (6,) in [0,1]

    # Denormalize angles
    ax = ang_pred[:3] * RG[3:6] + LO[3:6]
    ay = ang_pred[3:] * RG[6:9] + LO[6:9]

    # Stage 3: remaining params from network
    raw = pattern.T[None].astype(np.float32)  # (1, 2, 200)
    raw_t = torch.tensor(raw, device=DEVICE)

    # Conditioning: speeds(3) + angles(6) + peaks(25) = 34
    speeds_norm32 = speeds_norm.astype(np.float32)
    angles_norm = np.concatenate([
        (ax - LO[3:6]) / RG[3:6],
        (ay - LO[6:9]) / RG[6:9],
    ]).astype(np.float32)
    cond = np.concatenate([speeds_norm32, angles_norm, peak_feats]).astype(np.float32)
    cond_t = torch.tensor(cond[None], device=DEVICE)

    rem_pred = rem_net(raw_t, cond_t).cpu().numpy()[0]  # (9,) in [0,1]

    # Denormalize remaining: glass(3), d_W, gap, beam_ax, beam_ay, beam_px, beam_py
    remaining = rem_pred * RG[9:] + LO[9:]

    # Assemble full 18-D vector
    result = np.zeros(N_PAR, dtype=np.float32)
    result[:3]  = speeds
    result[3:6] = ax
    result[6:9] = ay
    result[9:]  = remaining
    return np.clip(result, LO, HI)


def _build_peak_feats_single(pattern, peak_freqs, peak_info):
    """Build the 25-dim peak feature vector for a single pattern."""
    return np.concatenate([
        peak_freqs,                                    # 3
        peak_info['amp_x'], peak_info['amp_y'],        # 6
        np.sin(peak_info['phase_x']),                  # 3
        np.cos(peak_info['phase_x']),                  # 3
        np.sin(peak_info['phase_y']),                  # 3
        np.cos(peak_info['phase_y']),                  # 3
        [np.mean(pattern[:, 0])],                      # 1
        [np.mean(pattern[:, 1])],                      # 1
        [np.std(pattern[:, 0])],                       # 1
        [np.std(pattern[:, 1])],                       # 1
    ]).astype(np.float32)                              # 25 total


# ================================================================
#  Gradient Refinement + Full Solver
# ================================================================

_fwd = [None]

def _get_fwd():
    if _fwd[0] is None:
        _fwd[0] = DiffFwd().to(DEVICE)
    return _fwd[0]


def _ml_init(ang_net, rem_net, speeds, peak_feats, pattern):
    """Full 18-D ML prediction for given signed speeds."""
    ang_net.eval(); rem_net.eval()
    speeds_norm = ((speeds - LO[:3]) / RG[:3]).astype(np.float32)

    ang_in = np.concatenate([speeds_norm, peak_feats]).astype(np.float32)
    with torch.no_grad():
        ang_pred = ang_net(torch.tensor(ang_in[None], device=DEVICE)).cpu().numpy()[0]
    ax = ang_pred[:3] * RG[3:6] + LO[3:6]
    ay = ang_pred[3:] * RG[6:9] + LO[6:9]

    raw = pattern.T[None].astype(np.float32)
    angles_norm = np.concatenate([
        (ax - LO[3:6]) / RG[3:6], (ay - LO[6:9]) / RG[6:9]
    ]).astype(np.float32)
    cond = np.concatenate([speeds_norm, angles_norm, peak_feats]).astype(np.float32)
    with torch.no_grad():
        rem_pred = rem_net(
            torch.tensor(raw, device=DEVICE),
            torch.tensor(cond[None], device=DEVICE)
        ).cpu().numpy()[0]
    remaining = rem_pred * RG[9:] + LO[9:]

    v = np.zeros(N_PAR, dtype=np.float64)
    v[:3] = speeds; v[3:6] = ax; v[6:9] = ay; v[9:] = remaining
    return np.clip(v, LO, HI)


def _paraxial_init(speeds, peak_info):
    """
    Paraxial-formula initialization — more accurate than ML for angles.
    Uses default glass/geometry and derives angles from FFT amplitudes/phases.
    """
    v = np.zeros(N_PAR, dtype=np.float64)
    v[:3] = speeds

    n_g0 = 1.5
    d_W0 = 100.0
    gap0 = 6.0

    for i in range(P):
        d_eff = (P - 1 - i) * (THK + gap0) + d_W0
        alpha_rad = peak_info['amp_x'][i] / (d_eff * (n_g0 - 1) + 1e-10)
        v[3 + i] = np.degrees(np.clip(alpha_rad, -0.31, 0.31))
        v[6 + i] = np.degrees(peak_info['phase_x'][i])

    v[9:12] = n_g0
    v[12] = d_W0
    v[13] = gap0
    v[14] = np.degrees(np.arctan2(peak_info['centroid_x'], d_W0))
    v[15] = np.degrees(np.arctan2(peak_info['centroid_y'], d_W0))
    return np.clip(v, LO, HI).astype(np.float64)


def _quick_refine(fwd, init, target, steps=100):
    """Fast uniform-LR refinement for sign selection (float32)."""
    lo_t = torch.tensor(LO, device=DEVICE)
    hi_t = torch.tensor(HI, device=DEVICE)
    par = torch.tensor(init, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    opt = torch.optim.Adam([par], lr=0.01)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(fwd(par[None], high_precision=False).squeeze(0), target)
        loss.backward(); opt.step()
        with torch.no_grad(): par.clamp_(lo_t, hi_t)
    return par.detach().cpu().numpy(), loss.item()


def _heavy_refine(fwd, init, target_64, verbose=False):
    """
    Two-phase Adam refinement in float64.

    Phase 1 (lr=0.005, 5000 steps): fast convergence to ~1e-3 MSE.
    Phase 2 (lr=0.0005, 20000 steps): fine convergence through
      the tiny non-paraxial gradients that break the degeneracy.

    All 18 params optimized jointly (speeds included — they refine
    from the FFT estimate to sub-Hz precision).
    """
    lo_t = torch.tensor(LO, device=DEVICE, dtype=torch.float64)
    hi_t = torch.tensor(HI, device=DEVICE, dtype=torch.float64)
    par = torch.tensor(init, device=DEVICE, dtype=torch.float64).requires_grad_(True)

    best_loss = float('inf')
    best_params = init.copy()

    # ---- Phase 1: Adam coarse (find the basin) ----
    opt = torch.optim.Adam([par], lr=0.005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 5000, eta_min=5e-5)
    for step in range(5000):
        opt.zero_grad()
        loss = F.mse_loss(fwd(par[None], high_precision=True).squeeze(0), target_64)
        loss.backward(); opt.step(); sched.step()
        with torch.no_grad(): par.clamp_(lo_t, hi_t)
        lv = loss.item()
        if lv < best_loss: best_loss = lv; best_params = par.detach().cpu().numpy().copy()
    if verbose: print(f"      Phase 1 (Adam coarse): pat_MSE={best_loss:.2e}", flush=True)
    if best_loss < 1e-14: return best_params, best_loss

    # ---- Phase 2: Adam fine (push into basin floor) ----
    opt = torch.optim.Adam([par], lr=0.0005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10000, eta_min=5e-7)
    for step in range(10000):
        opt.zero_grad()
        loss = F.mse_loss(fwd(par[None], high_precision=True).squeeze(0), target_64)
        loss.backward(); opt.step(); sched.step()
        with torch.no_grad(): par.clamp_(lo_t, hi_t)
        lv = loss.item()
        if lv < best_loss: best_loss = lv; best_params = par.detach().cpu().numpy().copy()
    if verbose: print(f"      Adam done: pat_MSE={best_loss:.2e}", flush=True)
    return best_params, best_loss


def solve(ang_net, rem_net, pattern, verbose=False):
    """
    Full 18-D solver: FFT extraction → 8-sign search → per-parameter gradient polish.

    Strategy for handling the glass/geometry degeneracy:
      1. Solve well-conditioned params (speeds, angles, beam) fast (high LR)
      2. Let degenerate params (glass, geometry) converge slowly via
         non-paraxial gradients that weakly break the degeneracy
      3. Find ANY valid 18-D solution with pattern MSE → 0
    """
    fwd = _get_fwd()
    peak_freqs, peak_info = extract_speeds_and_peaks(pattern)
    peak_feats = _build_peak_feats_single(pattern, peak_freqs, peak_info)
    target_32 = torch.tensor(pattern, dtype=torch.float32, device=DEVICE)
    target_64 = torch.tensor(pattern, dtype=torch.float64, device=DEVICE)

    # ---- Phase 1: 8 sign combos × paraxial init × light gradient (pick basin) ----
    candidates = []
    for bits in range(8):
        signs = np.array([(1.0 if (bits >> i) & 1 == 0 else -1.0)
                          for i in range(P)], dtype=np.float64)
        speeds = signs * peak_freqs.astype(np.float64)
        init = _ml_init(ang_net, rem_net, speeds.astype(np.float32),
                        peak_feats, pattern)
        refined, mse = _quick_refine(fwd, init, target_32, steps=150)
        candidates.append((mse, refined))
        if verbose:
            print(f"    Sign {bits}: MSE={mse:.2e}", flush=True)

    candidates.sort(key=lambda x: x[0])
    if verbose:
        print(f"    Best sign combo: MSE={candidates[0][0]:.2e}")

    # ---- Phase 2: Multi-start heavy refinement ----
    # The sign combo gives correct speeds+angles, but glass/geometry are
    # degenerate.  Try multiple glass/geometry starting points to cover
    # the degenerate directions.  Pick the basin with lowest MSE.
    best_init = candidates[0][1]

    glass_starts = [
        [1.35, 1.35, 1.35],
        [1.50, 1.50, 1.50],
        [1.65, 1.65, 1.65],
        [1.40, 1.55, 1.70],
        [1.70, 1.40, 1.55],
    ]
    geo_starts = [
        (80.0,  4.0),
        (100.0, 6.0),
        (120.0, 8.0),
        (150.0, 10.0),
        (60.0,  3.0),
    ]

    best_solved = None
    best_final_mse = float('inf')

    for gi, (ng_init, (dw_init, gap_init)) in enumerate(
            zip(glass_starts, geo_starts)):
        trial = best_init.copy()
        trial[9:12] = ng_init
        trial[12] = dw_init
        trial[13] = gap_init
        solved, mse = _heavy_refine(fwd, trial, target_64, verbose=False)
        if verbose:
            print(f"    Start {gi+1}/{len(glass_starts)}: "
                  f"ng={ng_init[0]:.2f} dW={dw_init:.0f} gap={gap_init:.0f} "
                  f"→ MSE={mse:.2e}", flush=True)
        if mse < best_final_mse:
            best_final_mse = mse
            best_solved = solved

    if verbose:
        print(f"    Best: MSE={best_final_mse:.2e}")

    return best_solved, best_final_mse


def _reparam_refine(fwd, init, C_fft, target_64, n_steps=15000, verbose=False):
    """
    Reparameterized optimization: hold composite amplitudes C_i fixed
    (from FFT — ground truth), optimize (n_g, d_W, gap, beam_pos).
    α_x is DERIVED:

        α_x,i = arctan( C_i / (d_eff_i * (n_g,i - 1)) )

    This creates strong gradients for glass/geometry because changing n_g
    forces α_x to change to maintain C, which visibly changes the non-paraxial
    pattern.  The degeneracy direction becomes well-conditioned.
    """
    C_t  = torch.tensor(C_fft, device=DEVICE, dtype=torch.float64)
    N_t  = torch.tensor(init[:3], device=DEVICE, dtype=torch.float64)
    ay_t = torch.tensor(init[6:9], device=DEVICE, dtype=torch.float64)
    bax_t = torch.tensor(init[14:15], device=DEVICE, dtype=torch.float64)
    bay_t = torch.tensor(init[15:16], device=DEVICE, dtype=torch.float64)

    # Free params: glass(3) + d_W + gap + beam_pos(2) = 7
    free = torch.tensor(
        np.array([init[9], init[10], init[11], init[12], init[13],
                  init[16], init[17]], dtype=np.float64),
        device=DEVICE, dtype=torch.float64).requires_grad_(True)

    lo_free = torch.tensor([1.3,1.3,1.3, 50., 2., -5.,-5.],
                            device=DEVICE, dtype=torch.float64)
    hi_free = torch.tensor([1.8,1.8,1.8, 200.,15., 5., 5.],
                            device=DEVICE, dtype=torch.float64)

    opt = torch.optim.Adam([free], lr=0.002)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps, eta_min=1e-6)

    best_loss = float('inf')
    best_full = init.copy()

    for step in range(n_steps):
        opt.zero_grad()
        ng  = free[:3]
        dW  = free[3]
        gap = free[4]
        bpx = free[5:6]
        bpy = free[6:7]

        # d_eff per prism
        d_eff = torch.stack([(P-1-i) * (THK + gap) + dW for i in range(P)])

        # DERIVE α_x from C = d_eff * (n_g - 1) * tan(α_x)
        ax_rad = torch.atan(C_t / (d_eff * (ng - 1.0) + 1e-12))
        ax_deg = ax_rad * (180.0 / torch.pi)

        full = torch.cat([N_t, ax_deg, ay_t, ng, dW.unsqueeze(0), gap.unsqueeze(0),
                          bax_t, bay_t, bpx, bpy])
        pred = fwd(full[None], high_precision=True).squeeze(0)
        loss = F.mse_loss(pred, target_64)
        loss.backward()
        opt.step(); sched.step()

        with torch.no_grad():
            free.clamp_(lo_free, hi_free)

        lv = loss.item()
        if lv < best_loss:
            best_loss = lv
            best_full = full.detach().cpu().numpy().copy()

        if verbose and (step + 1) % 5000 == 0:
            print(f"      Reparam step {step+1}/{n_steps}: pat_MSE={lv:.2e}",
                  flush=True)

        if lv < 1e-14:
            if verbose: print(f"      Converged: pat_MSE={lv:.2e}")
            break

    if verbose:
        print(f"      Reparam done: pat_MSE={best_loss:.2e}", flush=True)
    return best_full, best_loss


def refine(pred_vec, target_pat, steps=REF_STEPS, lr=REF_LR):
    """Simple gradient polish (legacy interface)."""
    target_t = torch.tensor(target_pat, dtype=torch.float32, device=DEVICE)
    result, _ = _grad_refine(pred_vec, target_t, steps, lr)
    return result


# ================================================================
#  Display
# ================================================================

def show(pred, true, indent='    '):
    err = np.abs(pred - true)
    try:
        mse = float(np.mean((vec2pat(pred) - vec2pat(true))**2))
        print(f"{indent}Pattern MSE: {mse:.2e}")
    except Exception:
        print(f"{indent}Pattern MSE: (failed)")
    for i in range(N_PAR):
        pct = 100 * err[i] / abs(RG[i])
        q = 'OK' if pct < 1 else 'CLOSE' if pct < 5 else 'MEH' if pct < 10 else 'BAD'
        print(f"{indent}{NAMES[i]:6s}  pred={pred[i]:9.4f}  true={true[i]:9.4f}  "
              f"err={err[i]:.2e} ({pct:5.1f}%)  [{q}]")


# ================================================================
#  Main
# ================================================================

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrain', action='store_true')
    ap.add_argument('--eval',    action='store_true')
    ap.add_argument('--no-refine', action='store_true')
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    print('=' * 70)
    print(' STAGED ML RISLEY INVERSE SOLVER')
    print('=' * 70)

    if args.retrain:
        for f in [ANG_MODEL, REM_MODEL, ANG_CKPT, REM_CKPT]:
            if os.path.exists(f):
                os.remove(f)
                print(f"  Deleted {os.path.basename(f)}")

    ang_net = AngleNet()
    rem_net = RemainNet()

    if args.eval:
        if not os.path.exists(ANG_MODEL) or not os.path.exists(REM_MODEL):
            print("  Models not found. Train first."); sys.exit(1)
        ang_net.load_state_dict(torch.load(ANG_MODEL, map_location=DEVICE, weights_only=True))
        rem_net.load_state_dict(torch.load(REM_MODEL, map_location=DEVICE, weights_only=True))
        ang_net.to(DEVICE); rem_net.to(DEVICE)
        print("  Loaded both models.")
    else:
        need_train = (not os.path.exists(ANG_MODEL) or not os.path.exists(REM_MODEL)
                      or os.path.exists(ANG_CKPT) or os.path.exists(REM_CKPT))

        if need_train:
            print(f"\n--- Data ---")
            tp, ta, vp, va = load_or_gen_data()

            print(f"\n--- Peak Features ---")
            t0 = time.time()
            tr_peaks = extract_all_speeds(tp)
            va_peaks = extract_all_speeds(vp)
            print(f"  Extracted in {time.time()-t0:.1f}s")

            # ---- Stage 2: Angle network ----
            print(f"\n--- Stage 2: Angle Network ({EPOCHS_ANG} epochs) ---")

            # Input: normalized speeds (3) + peak features (25) = 28
            # Target: normalized angles (6): ax(3) + ay(3)
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

            tr_ds2 = TensorDataset(torch.from_numpy(tr_ang_in),
                                    torch.from_numpy(tr_ang_tgt))
            va_ds2 = TensorDataset(torch.from_numpy(va_ang_in),
                                    torch.from_numpy(va_ang_tgt))

            ang_net = train_net(ang_net, tr_ds2, va_ds2,
                                ANG_MODEL, ANG_CKPT, EPOCHS_ANG)
            if os.path.exists(ANG_CKPT): os.remove(ANG_CKPT)

            # ---- Stage 3: Remainder network ----
            print(f"\n--- Stage 3: Remainder Network ({EPOCHS_REM} epochs) ---")

            # Input: raw pattern (2, 200) + conditioning (34)
            # Conditioning: speeds_norm(3) + angles_norm(6) + peaks(25) = 34
            # Target: normalized remaining (9): glass(3), d_W, gap, beams(4)
            tr_raw = tp.transpose(0, 2, 1).astype(np.float32)
            va_raw = vp.transpose(0, 2, 1).astype(np.float32)

            tr_ang_n = np.concatenate([
                (ta[:, 3:6] - LO[3:6]) / RG[3:6],
                (ta[:, 6:9] - LO[6:9]) / RG[6:9],
            ], axis=1).astype(np.float32)
            va_ang_n = np.concatenate([
                (va[:, 3:6] - LO[3:6]) / RG[3:6],
                (va[:, 6:9] - LO[6:9]) / RG[6:9],
            ], axis=1).astype(np.float32)

            tr_cond = np.concatenate([tr_speeds_n, tr_ang_n, tr_peaks], axis=1)
            va_cond = np.concatenate([va_speeds_n, va_ang_n, va_peaks], axis=1)

            tr_rem_tgt = ((ta[:, 9:] - LO[9:]) / RG[9:]).astype(np.float32)
            va_rem_tgt = ((va[:, 9:] - LO[9:]) / RG[9:]).astype(np.float32)

            tr_ds3 = TensorDataset(torch.from_numpy(tr_raw),
                                    torch.from_numpy(tr_cond),
                                    torch.from_numpy(tr_rem_tgt))
            va_ds3 = TensorDataset(torch.from_numpy(va_raw),
                                    torch.from_numpy(va_cond),
                                    torch.from_numpy(va_rem_tgt))

            # Physics-informed loss: DiffForward compares predicted
            # pattern with input pattern, weighting params by observability
            diff_fwd = DiffFwd().to(DEVICE)
            phys = {'fwd': diff_fwd}

            rem_net = train_net(rem_net, tr_ds3, va_ds3,
                                REM_MODEL, REM_CKPT, EPOCHS_REM,
                                physics_ctx=phys)
            if os.path.exists(REM_CKPT): os.remove(REM_CKPT)

            del tp, ta, vp, va
        else:
            print(f"\n  Loading saved models.")
            ang_net.load_state_dict(torch.load(ANG_MODEL, map_location=DEVICE, weights_only=True))
            rem_net.load_state_dict(torch.load(REM_MODEL, map_location=DEVICE, weights_only=True))
            ang_net.to(DEVICE); rem_net.to(DEVICE)

    # ---- Paper test case ----
    print(f"\n{'='*70}")
    print(' TEST: Paper configuration (18-D full recovery)')
    print('=' * 70)

    tv = np.array([1.5,-1.,2., 12.,-8.,5., 3.,10.,-6.,
                   1.5,1.55,1.6, 100.,6., 10.,5., 0.,0.], np.float32)
    true_c = canon(tv)
    pattern = vec2pat(tv)

    # Quick ML prediction (for comparison)
    t0 = time.time()
    pred = predict(ang_net, rem_net, pattern)
    t_ms = (time.time() - t0) * 1000
    print(f"\n  ML prediction ({t_ms:.0f} ms):")
    show(pred, true_c)

    # Full solve: ML init + 8-sign search + gradient polish
    if not args.no_refine:
        print(f"\n  Full solve (8 sign combos + gradient polish)...")
        t0 = time.time()
        solved, solved_mse = solve(ang_net, rem_net, pattern, verbose=True)
        t_solve = time.time() - t0
        print(f"\n  Solved in {t_solve:.1f}s (pattern MSE={solved_mse:.2e}):")
        show(solved, true_c)

    # ---- Random battery ----
    n_test = 20 if not args.no_refine else 100
    print(f"\n{'='*70}")
    print(f' TEST: {n_test} random configurations (full solve)')
    print('=' * 70)

    rng_t = np.random.default_rng(999)
    errs, pat_mses, times_each = [], [], []
    for case_i in range(n_test):
        v = LO + RG * rng_t.random(N_PAR).astype(np.float32)
        for j in range(3):
            if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
        try:
            tc = canon(v); pat = vec2pat(v)
            t0 = time.time()
            if args.no_refine:
                pr = predict(ang_net, rem_net, pat)
            else:
                pr, mse = solve(ang_net, rem_net, pat)
            dt = time.time() - t0
            times_each.append(dt)
            errs.append(np.abs(pr - tc))
            pp = vec2pat(pr)
            pat_mses.append(float(np.mean((pp - pat)**2)))
            maxe = np.max(np.abs(pr - tc))
            tag = 'OK' if maxe < 1e-3 else 'CLOSE' if maxe < 0.01 else ''
            print(f"  Case {case_i+1:3d}: max_err={maxe:.2e}  "
                  f"pat_MSE={pat_mses[-1]:.2e}  {dt:.1f}s  {tag}", flush=True)
        except Exception as e:
            print(f"  Case {case_i+1:3d}: FAILED ({e})")

    errs = np.array(errs)
    avg_time = np.mean(times_each)

    print(f"\n  {len(errs)} cases, avg {avg_time:.1f}s/case")
    print(f"\n  {'Param':8s} {'Median':>10s} {'Mean':>10s} {'Max':>10s}")
    for i in range(N_PAR):
        med = np.median(errs[:, i])
        mean = np.mean(errs[:, i])
        mx = np.max(errs[:, i])
        print(f"  {NAMES[i]:8s} {med:10.6f} {mean:10.6f} {mx:10.6f}")

    print(f"\n  Pattern MSE:  median={np.median(pat_mses):.2e}  "
          f"mean={np.mean(pat_mses):.2e}  max={np.max(pat_mses):.2e}")

    n_perfect = np.sum(np.max(errs, axis=1) < 1e-3)
    n_close   = np.sum(np.max(errs, axis=1) < 0.01)
    print(f"\n  < 1e-3 max error: {n_perfect}/{len(errs)}")
    print(f"  < 1e-2 max error: {n_close}/{len(errs)}")

    # Save results
    results = {
        'method': 'Staged ML + gradient solve (8 sign combos, 5000 steps)',
        'n_train': N_TRAIN,
        'avg_time_s': round(avg_time, 1),
        'n_cases': len(errs),
        'n_perfect_1e3': int(n_perfect),
        'n_close_1e2': int(n_close),
        'per_param_max_error': {
            NAMES[i]: float(np.max(errs[:,i])) for i in range(N_PAR)},
        'pattern_mse_median': float(np.median(pat_mses)),
    }
    rpath = os.path.join(HERE, 'ml_staged_results.json')
    with open(rpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results → {os.path.basename(rpath)}")

    print(f"\n{'='*70}")
    print(' SPEED COMPARISON')
    print('=' * 70)
    print(f'  Differential Evolution:  100 - 4000 s/case')
    print(f'  End-to-end NN (V1):      3 ms/case  (speeds ~5%, glass ~22%)')
    print(f'  Staged ML (this):        {ms_per:.0f} ms/case')
    if ms_per > 0:
        print(f'  Speedup vs DE:           ~{int(2000/(ms_per/1000)):,}x')
    print()
