#!/usr/bin/env python3
"""
train_v3.py  --  Unified ML Training for Risley Prism Inverse Problem (V3)

Single 1D-CNN backbone that processes raw 200x2 scan patterns and predicts
all 15 non-speed parameters (speeds come from FFT, no ML needed).

Architecture:
  - 1D CNN backbone over raw pattern (2, 200) -> rich feature embedding
  - FFT spectral branch (amplitude + phase features)
  - Speed conditioning (3 signed speeds from FFT)
  - Fusion MLP with residual blocks -> 15 output params

Multi-phase training:
  Phase 1: Normalized parameter MSE (fast, learns rough mapping)
  Phase 2: Sensitivity-weighted MSE (focus on alpha_x, the bottleneck)
  Phase 3: Physics-informed loss every Nth batch (expensive, best signal)

Usage:
  python paper/train_v3.py --phase 1 --epochs 200 --data-size 5000000
  python paper/train_v3.py --phase 2 --epochs 100 --resume
  python paper/train_v3.py --phase 3 --epochs 50 --resume
  python paper/train_v3.py --eval
  python paper/train_v3.py --eval --heavy   # with heavy gradient refinement

Checkpoints are saved every 5 epochs and on interruption.  Fully resumable.
"""

import sys, os, time, json, io, argparse, logging
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'reverse_problem_v2'))
if hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                      errors='replace')
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
#  Logging
# ================================================================

LOG_PATH = os.path.join(HERE, 'train_v3.log')

def setup_logging(verbose=True):
    fmt = '%(asctime)s | %(message)s'
    handlers = [logging.FileHandler(LOG_PATH, encoding='utf-8')]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers,
                        force=True)
    return logging.getLogger('train_v3')

log = setup_logging()


# ================================================================
#  Physical Constants & Parameter Ranges
# ================================================================

P       = 3
T_PTS   = 200
T_OBS   = 10.0
SRC     = 6.0
THK     = 3.0
DT      = T_OBS / T_PTS
FREQS   = np.fft.rfftfreq(T_PTS, d=DT)   # (101,)
N_PAR   = 18

NAMES = ['N1','N2','N3','ax1','ax2','ax3','ay1','ay2','ay3',
         'ng1','ng2','ng3','d_W','gap','bm_ax','bm_ay','bm_px','bm_py']

# Full 18-param ranges
LO = np.array([-3.5]*3 + [-18.]*6 + [1.3]*3 + [50.,2.,-25.,-25.,-5.,-5.],
              dtype=np.float32)
HI = np.array([ 3.5]*3 + [ 18.]*6 + [1.8]*3 + [200.,15.,25.,25.,5.,5.],
              dtype=np.float32)
RG = HI - LO

# V3 predicts indices 3..17 (15 params: 6 angles + 3 glass + 2 geom + 4 beam)
# Speeds (indices 0..2) come from FFT
PRED_IDX   = list(range(3, 18))   # 15 params
N_PRED     = len(PRED_IDX)        # 15
PRED_NAMES = [NAMES[i] for i in PRED_IDX]
PRED_LO    = LO[PRED_IDX]
PRED_HI    = HI[PRED_IDX]
PRED_RG    = RG[PRED_IDX]

# Sensitivity weights for Phase 2:
# alpha_x has the tightest convergence basin (~1 deg), so upweight it.
# alpha_y matters too but has wider basin (~3 deg).
# Glass/geometry are degenerate -- downweight in param space, let physics fix them.
SENS_WEIGHTS = np.ones(N_PRED, dtype=np.float32)
SENS_WEIGHTS[0:3]  = 5.0    # ax1, ax2, ax3 -- critical
SENS_WEIGHTS[3:6]  = 3.0    # ay1, ay2, ay3 -- important
SENS_WEIGHTS[6:9]  = 0.5    # ng1, ng2, ng3 -- degenerate
SENS_WEIGHTS[9]    = 0.5    # d_W            -- degenerate
SENS_WEIGHTS[10]   = 0.5    # gap            -- degenerate
SENS_WEIGHTS[11:13] = 2.0   # bm_ax, bm_ay  -- moderate
SENS_WEIGHTS[13:15] = 1.0   # bm_px, bm_py  -- moderate
# Normalize so mean weight = 1
SENS_WEIGHTS /= SENS_WEIGHTS.mean()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================================================
#  Paths
# ================================================================

DATA_V3_PREFIX = os.path.join(HERE, 'ml_data_v3')  # + _{size}.npz
MODEL_PATH     = os.path.join(HERE, 'ml_v3_model.pt')
CKPT_PATH      = os.path.join(HERE, 'ml_v3_ckpt.pt')
RESULTS_PATH   = os.path.join(HERE, 'ml_v3_results.json')


# ================================================================
#  Utilities
# ================================================================

def fmt_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h: return f'{h}h{m:02d}m{s:02d}s'
    return f'{m}m{s:02d}s' if m else f'{s}s'


def canon(v):
    """Sort prisms by |speed| descending for unique ordering."""
    v = v.copy()
    o = np.argsort(-np.abs(v[:3]))
    v[:3], v[3:6], v[6:9], v[9:12] = v[:3][o], v[3:6][o], v[6:9][o], v[9:12][o]
    return v


def vec2pat(v):
    """18-D parameter vector -> (200, 2) scan pattern."""
    geo = SystemGeometry(
        source_distance=SRC, prism_thickness=THK,
        workpiece_distance=float(v[12]), inter_prism_gap=float(v[13]),
        beam_angle_x=float(v[14]), beam_angle_y=float(v[15]),
        beam_pos_x=float(v[16]), beam_pos_y=float(v[17]))
    pr = PrismParameters(P, v[:3].tolist(), v[3:6].tolist(), v[6:9].tolist(),
                         glass_indices=v[9:12].tolist(), geometry=geo)
    return fast_forward(pr, T_PTS, T_OBS)


# ================================================================
#  FFT Feature Extraction (batch, vectorized)
# ================================================================

def extract_fft_features(patterns):
    """
    Extract rich FFT-based features from patterns.

    Input:  patterns (N, 200, 2)
    Output: features (N, F) where F includes:
      - 3 peak frequencies
      - 6 peak amplitudes (x, y per peak)
      - 12 peak phases (sin/cos of phase_x, phase_y per peak)
      - 4 summary stats (centroid_x, centroid_y, std_x, std_y)
      = 25 features (same as v2, for compatibility)

    Also returns speed_signs: (N, 3) with +1/-1 sign for each speed.
    """
    N = len(patterns)
    fx = np.fft.rfft(patterns[:, :, 0], axis=1)  # (N, 101)
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

    # Sort by frequency descending (matches canonical |speed| order)
    sort_idx = np.argsort(-p_freq, axis=1)
    rows = arange_N[:, None]
    p_freq = p_freq[rows, sort_idx]
    p_ax   = p_ax[rows, sort_idx]
    p_ay   = p_ay[rows, sort_idx]
    p_phx  = p_phx[rows, sort_idx]
    p_phy  = p_phy[rows, sort_idx]

    # 25-dim feature vector
    cx = np.mean(patterns[:, :, 0], axis=1, keepdims=True)
    cy = np.mean(patterns[:, :, 1], axis=1, keepdims=True)
    sx = np.std(patterns[:, :, 0], axis=1, keepdims=True)
    sy = np.std(patterns[:, :, 1], axis=1, keepdims=True)

    peak_feats = np.concatenate([
        p_freq, p_ax, p_ay,
        np.sin(p_phx), np.cos(p_phx),
        np.sin(p_phy), np.cos(p_phy),
        cx, cy, sx, sy,
    ], axis=1).astype(np.float32)  # (N, 25)

    return peak_feats, p_freq


def extract_spectral_image(patterns):
    """
    Extract full spectral representation for CNN input.

    Input:  patterns (N, 200, 2)
    Output: spectral (N, 4, 101) -- log-amp and phase for x and y channels

    This gives the network access to ALL frequency information, not just
    the top-3 peaks.  The 1D CNN can learn to extract whatever features
    matter for each parameter.
    """
    fx = np.fft.rfft(patterns[:, :, 0], axis=1)  # (N, 101)
    fy = np.fft.rfft(patterns[:, :, 1], axis=1)

    spectral = np.stack([
        np.log1p(np.abs(fx)),   # log amplitude x
        np.log1p(np.abs(fy)),   # log amplitude y
        np.angle(fx),           # phase x
        np.angle(fy),           # phase y
    ], axis=1).astype(np.float32)  # (N, 4, 101)

    return spectral


# ================================================================
#  Data Generation
# ================================================================

def gen_data(n, seed=42):
    """Generate n valid (pattern, param) pairs.  Imported from ml_solver."""
    from ml_solver import gen_data as _gen_data
    return _gen_data(n, seed=seed)


def get_data_path(n_train):
    """Data file path for a given training set size."""
    if n_train <= 1_000_000:
        # Reuse existing 1M data if available
        legacy = os.path.join(HERE, 'ml_data_1m.npz')
        if os.path.exists(legacy):
            return legacy
    return f"{DATA_V3_PREFIX}_{n_train // 1000}k.npz"


def load_or_gen_data(n_train, n_val, force_regen=False):
    """Load cached data or generate fresh."""
    # Try to reuse existing 1M data
    legacy_path = os.path.join(HERE, 'ml_data_1m.npz')
    if not force_regen and os.path.exists(legacy_path):
        log.info("Loading cached data from ml_data_1m.npz...")
        d = np.load(legacy_path)
        tp, ta, vp, va = d['tp'], d['ta'], d['vp'], d['va']
        if len(tp) >= n_train and len(vp) >= n_val:
            log.info(f"  {len(tp):,} train + {len(vp):,} val available, "
                     f"using {n_train:,} + {n_val:,}")
            return tp[:n_train], ta[:n_train], vp[:n_val], va[:n_val]
        else:
            log.info(f"  Cached has {len(tp):,} train, need {n_train:,}")

    # Try v3 data file
    data_path = get_data_path(n_train)
    if not force_regen and os.path.exists(data_path):
        log.info(f"Loading cached data from {os.path.basename(data_path)}...")
        d = np.load(data_path)
        tp, ta, vp, va = d['tp'], d['ta'], d['vp'], d['va']
        if len(tp) >= n_train and len(vp) >= n_val:
            log.info(f"  Using {n_train:,} train + {n_val:,} val")
            return tp[:n_train], ta[:n_train], vp[:n_val], va[:n_val]

    # Generate fresh
    log.info(f"Generating {n_train:,} training samples...")
    tp, ta = gen_data(n_train, seed=42)
    log.info(f"Generating {n_val:,} validation samples...")
    vp, va = gen_data(n_val, seed=123)

    data_path = f"{DATA_V3_PREFIX}_{n_train // 1000}k.npz"
    log.info(f"Saving to {os.path.basename(data_path)}...")
    np.savez(data_path, tp=tp, ta=ta, vp=vp, va=va)
    sz = os.path.getsize(data_path) / 1e6
    log.info(f"  Saved ({sz:.0f} MB)")
    return tp, ta, vp, va


# ================================================================
#  V3 Network Architecture
# ================================================================

class ResBlock1d(nn.Module):
    """Residual block for 1D convolutions."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ResBlockMLP(nn.Module):
    """Residual block for MLP layers."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class RisleyNetV3(nn.Module):
    """
    Unified network for Risley prism inverse problem.

    Inputs:
      raw_pattern: (B, 2, 200)  -- raw x,y time series
      spectral:    (B, 4, 101)  -- log-amp + phase for x,y channels
      speeds:      (B, 3)       -- signed speeds from FFT (normalized to [0,1])
      peak_feats:  (B, 25)      -- peak features (freqs, amps, phases, stats)

    Output:
      params: (B, 15) in [0, 1] -- normalized predicted parameters
              [ax1,ax2,ax3, ay1,ay2,ay3, ng1,ng2,ng3, d_W, gap, bm_ax,bm_ay, bm_px,bm_py]
    """
    def __init__(self, base_ch=64, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim

        # ---- Branch 1: Raw pattern CNN (2, 200) -> embed ----
        self.raw_bn = nn.BatchNorm1d(2)
        self.raw_cnn = nn.Sequential(
            # (2, 200) -> (base_ch, 100)
            nn.Conv1d(2, base_ch, 7, stride=2, padding=3),
            nn.BatchNorm1d(base_ch), nn.GELU(),
            ResBlock1d(base_ch),
            # (base_ch, 100) -> (base_ch*2, 50)
            nn.Conv1d(base_ch, base_ch*2, 5, stride=2, padding=2),
            nn.BatchNorm1d(base_ch*2), nn.GELU(),
            ResBlock1d(base_ch*2),
            # (base_ch*2, 50) -> (base_ch*4, 25)
            nn.Conv1d(base_ch*2, base_ch*4, 3, stride=2, padding=1),
            nn.BatchNorm1d(base_ch*4), nn.GELU(),
            ResBlock1d(base_ch*4),
            # (base_ch*4, 25) -> (base_ch*4, 12)
            nn.Conv1d(base_ch*4, base_ch*4, 3, stride=2, padding=1),
            nn.BatchNorm1d(base_ch*4), nn.GELU(),
            # Global pool -> (base_ch*4,)
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        raw_out = base_ch * 4  # 256

        # ---- Branch 2: Spectral CNN (4, 101) -> embed ----
        self.spec_bn = nn.BatchNorm1d(4)
        self.spec_cnn = nn.Sequential(
            # (4, 101) -> (base_ch, 51)
            nn.Conv1d(4, base_ch, 7, stride=2, padding=3),
            nn.BatchNorm1d(base_ch), nn.GELU(),
            ResBlock1d(base_ch),
            # (base_ch, 51) -> (base_ch*2, 26)
            nn.Conv1d(base_ch, base_ch*2, 5, stride=2, padding=2),
            nn.BatchNorm1d(base_ch*2), nn.GELU(),
            ResBlock1d(base_ch*2),
            # (base_ch*2, 26) -> (base_ch*4, 13)
            nn.Conv1d(base_ch*2, base_ch*4, 3, stride=2, padding=1),
            nn.BatchNorm1d(base_ch*4), nn.GELU(),
            # Global pool -> (base_ch*4,)
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        spec_out = base_ch * 4  # 256

        # ---- Branch 3: Speed + peak feature MLP ----
        cond_in = 3 + 25  # speeds(3) + peak_feats(25) = 28
        self.cond_bn = nn.BatchNorm1d(cond_in)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU(),
        )
        cond_out = 128

        # ---- Fusion ----
        fusion_in = raw_out + spec_out + cond_out  # 256 + 256 + 128 = 640
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, embed_dim),
            nn.BatchNorm1d(embed_dim), nn.GELU(),
            nn.Dropout(0.1),
            ResBlockMLP(embed_dim),
            nn.Dropout(0.05),
            ResBlockMLP(embed_dim),
            nn.Dropout(0.05),
        )

        # ---- Prediction heads ----
        # Separate heads for different param groups (different difficulty levels)
        # Head 1: angles (6 params) -- hardest, biggest impact
        self.head_angles = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, 6), nn.Sigmoid(),
        )
        # Head 2: glass + geometry (5 params) -- degenerate but lower-D
        self.head_glass_geo = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Linear(64, 5), nn.Sigmoid(),
        )
        # Head 3: beam params (4 params) -- moderate difficulty
        self.head_beam = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Linear(64, 4), nn.Sigmoid(),
        )

    def forward(self, raw_pattern, spectral, speeds, peak_feats):
        # Branch embeddings
        r = self.raw_cnn(self.raw_bn(raw_pattern))          # (B, 256)
        s = self.spec_cnn(self.spec_bn(spectral))           # (B, 256)
        c = self.cond_mlp(self.cond_bn(
            torch.cat([speeds, peak_feats], dim=1)))         # (B, 128)

        # Fuse
        h = self.fusion(torch.cat([r, s, c], dim=1))        # (B, embed_dim)

        # Multi-head prediction
        angles    = self.head_angles(h)      # (B, 6)
        glass_geo = self.head_glass_geo(h)   # (B, 5)
        beam      = self.head_beam(h)        # (B, 4)

        return torch.cat([angles, glass_geo, beam], dim=1)   # (B, 15)


# ================================================================
#  Differentiable Forward Model (from ml_staged_solver.py)
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
        return torch.stack([x,y],2)

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
#  Dataset Preparation
# ================================================================

def prepare_datasets(patterns, params, n_train, n_val):
    """
    Prepare train and val TensorDatasets.

    Each sample: (raw, spectral, speeds_norm, peak_feats, target_norm)
    """
    log.info("Extracting FFT features...")
    t0 = time.time()
    peak_feats, _ = extract_fft_features(patterns)
    spectral = extract_spectral_image(patterns)
    log.info(f"  FFT features extracted in {time.time()-t0:.1f}s")

    # Raw pattern as (N, 2, 200)
    raw = patterns.transpose(0, 2, 1).astype(np.float32)

    # Normalize speeds to [0,1] (from true params, teacher forcing)
    speeds_norm = ((params[:, :3] - LO[:3]) / RG[:3]).astype(np.float32)

    # Target: 15 predicted params normalized to [0,1]
    target_norm = ((params[:, PRED_IDX] - PRED_LO) / PRED_RG).astype(np.float32)

    # Split
    tr_raw = torch.from_numpy(raw[:n_train])
    tr_spec = torch.from_numpy(spectral[:n_train])
    tr_speeds = torch.from_numpy(speeds_norm[:n_train])
    tr_peaks = torch.from_numpy(peak_feats[:n_train])
    tr_target = torch.from_numpy(target_norm[:n_train])

    va_raw = torch.from_numpy(raw[n_train:n_train+n_val])
    va_spec = torch.from_numpy(spectral[n_train:n_train+n_val])
    va_speeds = torch.from_numpy(speeds_norm[n_train:n_train+n_val])
    va_peaks = torch.from_numpy(peak_feats[n_train:n_train+n_val])
    va_target = torch.from_numpy(target_norm[n_train:n_train+n_val])

    # Also store full params for physics loss (denormalization)
    tr_params_full = torch.from_numpy(params[:n_train].astype(np.float32))
    va_params_full = torch.from_numpy(params[n_train:n_train+n_val].astype(np.float32))

    train_ds = TensorDataset(tr_raw, tr_spec, tr_speeds, tr_peaks,
                             tr_target, tr_params_full)
    val_ds   = TensorDataset(va_raw, va_spec, va_speeds, va_peaks,
                             va_target, va_params_full)

    log.info(f"  Train: {len(train_ds):,} samples, Val: {len(val_ds):,} samples")
    return train_ds, val_ds


# ================================================================
#  Training Loop
# ================================================================

def compute_loss(pred, target, phase, weights_t, fwd, params_full, raw,
                 batch_idx, physics_every, physics_weight, lo_t, rg_t,
                 pred_lo_t, pred_rg_t):
    """
    Compute loss depending on training phase.

    Phase 1: Plain MSE
    Phase 2: Sensitivity-weighted MSE
    Phase 3: Weighted MSE + physics loss every Nth batch
    """
    if phase == 1:
        return F.mse_loss(pred, target), 0.0

    elif phase == 2:
        diff = (pred - target) ** 2
        loss = (diff * weights_t).mean()
        return loss, 0.0

    elif phase == 3:
        # Weighted param loss
        diff = (pred - target) ** 2
        param_loss = (diff * weights_t).mean()

        # Physics loss: every Nth batch
        phys_loss_val = 0.0
        if batch_idx % physics_every == 0 and fwd is not None:
            # Denormalize predictions to real units
            pred_real = pred * pred_rg_t + pred_lo_t  # (B, 15)

            # Assemble full 18-D: speeds from true params + predicted rest
            speeds = params_full[:, :3]  # true speeds
            full = torch.cat([speeds, pred_real], dim=1)  # (B, 18)

            # Target pattern from raw input
            tgt_pat = raw.transpose(1, 2)  # (B, 200, 2)

            # Forward model
            try:
                pred_pat = fwd(full, high_precision=False)
                p_loss = F.mse_loss(pred_pat, tgt_pat)
                param_loss = param_loss + physics_weight * p_loss
                phys_loss_val = p_loss.item()
            except Exception:
                pass  # skip physics loss on rare NaN cases

        return param_loss, phys_loss_val

    else:
        raise ValueError(f"Unknown phase: {phase}")


def train(model, train_ds, val_ds, args):
    """
    Main training loop with checkpointing, resumption, and multi-phase support.
    """
    model.to(DEVICE)

    # Optimizer
    lr = args.lr
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    tr_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=0, pin_memory=(DEVICE.type == 'cuda'))
    va_dl = DataLoader(val_ds, batch_size=args.batch_size * 2,
                       num_workers=0, pin_memory=(DEVICE.type == 'cuda'))

    epochs = args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(tr_dl))

    start_ep = 0
    best_val = float('inf')
    phase = args.phase

    # Resume from checkpoint
    if args.resume and os.path.exists(CKPT_PATH):
        log.info(f"Resuming from checkpoint...")
        ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        if 'sched' in ck and ck.get('sched_epochs') == epochs:
            try:
                sched.load_state_dict(ck['sched'])
            except Exception:
                log.info("  Scheduler state mismatch, reinitializing")
        start_ep = ck.get('epoch', 0) + 1
        best_val = ck.get('best_val', float('inf'))
        prev_phase = ck.get('phase', 1)
        if prev_phase != phase:
            log.info(f"  Phase change: {prev_phase} -> {phase}, "
                     f"resetting scheduler (keeping model + optimizer)")
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(tr_dl))
            start_ep = 0
            best_val = float('inf')
        else:
            log.info(f"  Resuming phase {phase} from epoch {start_ep}, "
                     f"best_val={best_val:.6f}")
    elif args.resume and os.path.exists(MODEL_PATH):
        # No checkpoint but model exists -- load model, start fresh training
        log.info(f"Loading saved model for phase {phase} training...")
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

    # Sensitivity weights
    weights_t = torch.tensor(SENS_WEIGHTS, device=DEVICE, dtype=torch.float32)

    # Physics loss setup
    fwd = None
    if phase >= 3:
        fwd = DiffFwd().to(DEVICE)

    lo_t = torch.tensor(LO, device=DEVICE, dtype=torch.float32)
    rg_t = torch.tensor(RG, device=DEVICE, dtype=torch.float32)
    pred_lo_t = torch.tensor(PRED_LO, device=DEVICE, dtype=torch.float32)
    pred_rg_t = torch.tensor(PRED_RG, device=DEVICE, dtype=torch.float32)

    n_params = sum(p.numel() for p in model.parameters())
    nb = len(tr_dl)
    phase_names = {1: 'MSE', 2: 'weighted MSE', 3: 'weighted MSE + physics'}
    log.info(f"Phase {phase}: {phase_names[phase]}")
    log.info(f"  Parameters: {n_params:,}")
    log.info(f"  Batches/epoch: {nb}")
    log.info(f"  LR: {lr}, batch_size: {args.batch_size}")
    log.info(f"  Epochs: {start_ep} -> {epochs}")
    if phase >= 3:
        log.info(f"  Physics loss: every {args.physics_every} batches, "
                 f"weight={args.physics_weight}")

    t0 = time.time()
    try:
        for ep in range(start_ep, epochs):
            model.train()
            tl = 0; pl = 0; p_count = 0

            for bi, batch_data in enumerate(tr_dl):
                raw, spec, speeds, peaks, target, params_full = \
                    [x.to(DEVICE) for x in batch_data]

                pred = model(raw, spec, speeds, peaks)
                loss, phys_val = compute_loss(
                    pred, target, phase, weights_t, fwd, params_full, raw,
                    bi, args.physics_every, args.physics_weight,
                    lo_t, rg_t, pred_lo_t, pred_rg_t)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tl += loss.item()
                if phys_val > 0:
                    pl += phys_val; p_count += 1

            # Validation
            model.eval()
            vl = 0; v_angle = 0; v_glass = 0; v_beam = 0
            with torch.no_grad():
                for batch_data in va_dl:
                    raw, spec, speeds, peaks, target, params_full = \
                        [x.to(DEVICE) for x in batch_data]
                    pred = model(raw, spec, speeds, peaks)
                    vl += F.mse_loss(pred, target).item()
                    # Per-group losses for monitoring
                    v_angle += F.mse_loss(pred[:, :6], target[:, :6]).item()
                    v_glass += F.mse_loss(pred[:, 6:11], target[:, 6:11]).item()
                    v_beam  += F.mse_loss(pred[:, 11:], target[:, 11:]).item()

            n_va = len(va_dl)
            tl /= nb; vl /= n_va
            v_angle /= n_va; v_glass /= n_va; v_beam /= n_va

            improved = vl < best_val
            if improved:
                best_val = vl
                torch.save(model.state_dict(), MODEL_PATH)

            ep_dt = time.time() - t0
            eta = ep_dt / (ep - start_ep + 1) * (epochs - ep - 1)
            marker = ' *' if improved else ''
            phys_str = f"  phys={pl/max(p_count,1):.4f}" if p_count else ""
            cur_lr = sched.get_last_lr()[0]

            if (ep + 1) % 5 == 0 or ep == start_ep or improved:
                log.info(
                    f"  Ep {ep+1:4d}/{epochs}  "
                    f"tr={tl:.6f} va={vl:.6f} "
                    f"[ang={v_angle:.5f} gl={v_glass:.5f} bm={v_beam:.5f}]  "
                    f"best={best_val:.6f}{marker}{phys_str}  "
                    f"lr={cur_lr:.2e}  ETA {fmt_time(eta)}")

            # Checkpoint every 5 epochs
            if (ep + 1) % 5 == 0:
                torch.save({
                    'epoch': ep,
                    'model': model.state_dict(),
                    'opt': opt.state_dict(),
                    'sched': sched.state_dict(),
                    'sched_epochs': epochs,
                    'best_val': best_val,
                    'phase': phase,
                    'args': vars(args),
                }, CKPT_PATH)

    except KeyboardInterrupt:
        log.info(f"\n  Interrupted at epoch {ep+1}.")
        torch.save({
            'epoch': ep,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'sched_epochs': epochs,
            'best_val': best_val,
            'phase': phase,
            'args': vars(args),
        }, CKPT_PATH)

    # Reload best
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    total_time = time.time() - t0
    log.info(f"  Training complete: {fmt_time(total_time)}, "
             f"best val={best_val:.6f}")
    return model


# ================================================================
#  Inference: V3 Prediction
# ================================================================

def extract_speeds_and_peaks_single(pattern):
    """Single-pattern speed + peak extraction (for inference)."""
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

    idxs = sorted(all_idxs[:P], key=lambda i: FREQS[i], reverse=True)
    peak_freqs = np.array([FREQS[i] for i in idxs], dtype=np.float32)
    amp_x  = np.array([np.abs(fx[i]) * 2 / T_PTS for i in idxs], dtype=np.float32)
    amp_y  = np.array([np.abs(fy[i]) * 2 / T_PTS for i in idxs], dtype=np.float32)
    phase_x = np.array([np.angle(fx[i]) for i in idxs], dtype=np.float32)
    phase_y = np.array([np.angle(fy[i]) for i in idxs], dtype=np.float32)

    centroid_x = float(np.mean(pattern[:, 0]))
    centroid_y = float(np.mean(pattern[:, 1]))

    # Build 25-dim peak features
    peak_feats = np.concatenate([
        peak_freqs,
        amp_x, amp_y,
        np.sin(phase_x), np.cos(phase_x),
        np.sin(phase_y), np.cos(phase_y),
        [centroid_x], [centroid_y],
        [np.std(pattern[:, 0])], [np.std(pattern[:, 1])],
    ]).astype(np.float32)

    # Spectral image (4, 101)
    spec = np.stack([
        np.log1p(np.abs(fx)),
        np.log1p(np.abs(fy)),
        np.angle(fx),
        np.angle(fy),
    ]).astype(np.float32)

    return peak_freqs, peak_feats, spec, {
        'amp_x': amp_x, 'amp_y': amp_y,
        'phase_x': phase_x, 'phase_y': phase_y,
        'centroid_x': centroid_x, 'centroid_y': centroid_y,
    }


@torch.no_grad()
def predict_v3(model, pattern):
    """
    Full V3 prediction: FFT -> 8-sign search -> best ML prediction.

    pattern: (200, 2) numpy
    Returns: (18,) numpy parameter vector in real units, best_mse
    """
    model.eval()

    peak_freqs, peak_feats, spec, peak_info = \
        extract_speeds_and_peaks_single(pattern)

    raw_t = torch.tensor(pattern.T[None].astype(np.float32), device=DEVICE)
    spec_t = torch.tensor(spec[None], device=DEVICE)
    peaks_t = torch.tensor(peak_feats[None], device=DEVICE)

    best_result = None
    best_mse = float('inf')

    for bits in range(2**P):
        signs = np.array([(1.0 if (bits >> i) & 1 == 0 else -1.0)
                          for i in range(P)], dtype=np.float32)
        speeds = signs * peak_freqs

        speeds_norm = ((speeds - LO[:3]) / RG[:3]).astype(np.float32)
        speeds_t = torch.tensor(speeds_norm[None], device=DEVICE)

        pred_norm = model(raw_t, spec_t, speeds_t, peaks_t).cpu().numpy()[0]
        pred_real = pred_norm * PRED_RG + PRED_LO

        # Assemble full 18-D
        result = np.zeros(N_PAR, dtype=np.float32)
        result[:3] = speeds
        result[3:] = pred_real
        result = np.clip(result, LO, HI)

        try:
            pat_pred = vec2pat(result)
            mse = float(np.mean((pat_pred - pattern)**2))
            if mse < best_mse:
                best_mse = mse
                best_result = result.copy()
        except Exception:
            pass

    if best_result is None:
        # Fallback: use positive speeds
        speeds = peak_freqs.copy()
        speeds_norm = ((speeds - LO[:3]) / RG[:3]).astype(np.float32)
        speeds_t = torch.tensor(speeds_norm[None], device=DEVICE)
        pred_norm = model(raw_t, spec_t, speeds_t, peaks_t).cpu().numpy()[0]
        pred_real = pred_norm * PRED_RG + PRED_LO
        best_result = np.zeros(N_PAR, dtype=np.float32)
        best_result[:3] = speeds
        best_result[3:] = pred_real
        best_result = np.clip(best_result, LO, HI)
        best_mse = float('inf')

    return best_result, best_mse


# ================================================================
#  Gradient Refinement (for evaluation)
# ================================================================

def heavy_refine(fwd, init, target_64, verbose=False):
    """
    Two-phase Adam refinement in float64 for evaluation.
    """
    lo_t = torch.tensor(LO, device=DEVICE, dtype=torch.float64)
    hi_t = torch.tensor(HI, device=DEVICE, dtype=torch.float64)
    par = torch.tensor(init, device=DEVICE, dtype=torch.float64).requires_grad_(True)

    best_loss = float('inf')
    best_params = init.copy()

    # Phase 1: Adam coarse
    opt = torch.optim.Adam([par], lr=0.005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 5000, eta_min=5e-5)
    for step in range(5000):
        opt.zero_grad()
        loss = F.mse_loss(fwd(par[None], high_precision=True).squeeze(0), target_64)
        loss.backward(); opt.step(); sched.step()
        with torch.no_grad(): par.clamp_(lo_t, hi_t)
        lv = loss.item()
        if lv < best_loss: best_loss = lv; best_params = par.detach().cpu().numpy().copy()
    if verbose: log.info(f"      Phase 1: pat_MSE={best_loss:.2e}")
    if best_loss < 1e-14: return best_params, best_loss

    # Phase 2: Adam fine
    opt = torch.optim.Adam([par], lr=0.0005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10000, eta_min=5e-7)
    for step in range(10000):
        opt.zero_grad()
        loss = F.mse_loss(fwd(par[None], high_precision=True).squeeze(0), target_64)
        loss.backward(); opt.step(); sched.step()
        with torch.no_grad(): par.clamp_(lo_t, hi_t)
        lv = loss.item()
        if lv < best_loss: best_loss = lv; best_params = par.detach().cpu().numpy().copy()
    if verbose: log.info(f"      Phase 2: pat_MSE={best_loss:.2e}")
    return best_params, best_loss


def quick_refine(fwd, init, target_32, steps=200):
    """Fast float32 refinement for sign selection."""
    lo_t = torch.tensor(LO, device=DEVICE)
    hi_t = torch.tensor(HI, device=DEVICE)
    par = torch.tensor(init, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    opt = torch.optim.Adam([par], lr=0.01)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(fwd(par[None], high_precision=False).squeeze(0), target_32)
        loss.backward(); opt.step()
        with torch.no_grad(): par.clamp_(lo_t, hi_t)
    return par.detach().cpu().numpy(), loss.item()


def solve_v3(model, pattern, verbose=False, heavy=True):
    """
    Full solve: V3 ML prediction -> 8-sign search -> gradient refinement.

    Returns: (solved_params, pattern_mse)
    """
    fwd = DiffFwd().to(DEVICE)
    target_32 = torch.tensor(pattern, dtype=torch.float32, device=DEVICE)
    target_64 = torch.tensor(pattern, dtype=torch.float64, device=DEVICE)

    # Step 1: ML prediction with 8-sign search
    ml_pred, ml_mse = predict_v3(model, pattern)
    if verbose:
        log.info(f"    ML prediction: pat_MSE={ml_mse:.2e}")

    if not heavy:
        # Light refinement only
        refined, mse = quick_refine(fwd, ml_pred, target_32, steps=500)
        if verbose:
            log.info(f"    Quick refine: pat_MSE={mse:.2e}")
        return refined, mse

    # Step 2: Heavy refinement from ML init
    # Try the ML prediction + a few glass/geometry starting points
    glass_starts = [
        None,                          # keep ML prediction as-is
        [1.35, 1.35, 1.35],
        [1.50, 1.50, 1.50],
        [1.65, 1.65, 1.65],
        [1.40, 1.55, 1.70],
    ]
    geo_starts = [
        None,
        (80.0, 4.0),
        (100.0, 6.0),
        (120.0, 8.0),
        (150.0, 10.0),
    ]

    best_solved = None
    best_final_mse = float('inf')

    for gi, (ng_init, geo_init) in enumerate(zip(glass_starts, geo_starts)):
        trial = ml_pred.copy().astype(np.float64)
        if ng_init is not None:
            trial[9:12] = ng_init
        if geo_init is not None:
            trial[12], trial[13] = geo_init

        solved, mse = heavy_refine(fwd, trial, target_64, verbose=False)
        if verbose:
            tag = "ML" if gi == 0 else f"ng={ng_init[0]:.2f}"
            log.info(f"    Start {gi+1}/{len(glass_starts)} ({tag}): "
                     f"pat_MSE={mse:.2e}")
        if mse < best_final_mse:
            best_final_mse = mse
            best_solved = solved

    if verbose:
        log.info(f"    Best: pat_MSE={best_final_mse:.2e}")

    return best_solved, best_final_mse


# ================================================================
#  Evaluation
# ================================================================

def show(pred, true, indent='    '):
    """Print per-parameter comparison."""
    err = np.abs(pred - true)
    try:
        mse = float(np.mean((vec2pat(pred) - vec2pat(true))**2))
        log.info(f"{indent}Pattern MSE: {mse:.2e}")
    except Exception:
        log.info(f"{indent}Pattern MSE: (failed)")
    for i in range(N_PAR):
        pct = 100 * err[i] / abs(RG[i])
        q = 'OK' if pct < 1 else 'CLOSE' if pct < 5 else 'MEH' if pct < 10 else 'BAD'
        log.info(f"{indent}{NAMES[i]:6s}  pred={pred[i]:9.4f}  true={true[i]:9.4f}  "
                 f"err={err[i]:.2e} ({pct:5.1f}%)  [{q}]")


def evaluate(model, n_test=30, heavy=True, verbose=True):
    """Evaluate on random test cases."""
    model.eval()

    # Fixed test case: paper configuration
    log.info(f"\n{'='*70}")
    log.info(' TEST: Paper configuration')
    log.info('=' * 70)

    tv = np.array([1.5,-1.,2., 12.,-8.,5., 3.,10.,-6.,
                   1.5,1.55,1.6, 100.,6., 10.,5., 0.,0.], np.float32)
    true_c = canon(tv)
    pattern = vec2pat(tv)

    t0 = time.time()
    ml_pred, ml_mse = predict_v3(model, pattern)
    t_ms = (time.time() - t0) * 1000
    log.info(f"\n  ML prediction ({t_ms:.0f} ms, pat_MSE={ml_mse:.2e}):")
    show(ml_pred, true_c)

    if heavy:
        log.info(f"\n  Full solve (ML init + gradient refinement)...")
        t0 = time.time()
        solved, solved_mse = solve_v3(model, pattern, verbose=True, heavy=True)
        t_solve = time.time() - t0
        log.info(f"\n  Solved in {t_solve:.1f}s (pat_MSE={solved_mse:.2e}):")
        show(solved, true_c)

    # Random battery
    log.info(f"\n{'='*70}")
    log.info(f' TEST: {n_test} random configurations')
    log.info('=' * 70)

    rng_t = np.random.default_rng(999)
    errs, pat_mses, times_each = [], [], []
    ml_only_errs = []

    for case_i in range(n_test):
        v = LO + RG * rng_t.random(N_PAR).astype(np.float32)
        for j in range(3):
            if abs(v[j]) < 0.15: v[j] = np.copysign(0.15, v[j])
        try:
            tc = canon(v)
            pat = vec2pat(v)

            # ML-only prediction (for comparison)
            ml_pred, ml_mse = predict_v3(model, pat)
            ml_only_errs.append(np.abs(ml_pred - tc))

            t0 = time.time()
            if heavy:
                pr, mse = solve_v3(model, pat, verbose=False, heavy=True)
            else:
                pr, mse = solve_v3(model, pat, verbose=False, heavy=False)
            dt = time.time() - t0
            times_each.append(dt)
            errs.append(np.abs(pr - tc))
            pp = vec2pat(pr)
            pat_mses.append(float(np.mean((pp - pat)**2)))
            maxe = np.max(np.abs(pr - tc))
            tag = 'OK' if maxe < 1e-3 else 'CLOSE' if maxe < 0.01 else ''
            log.info(f"  Case {case_i+1:3d}: max_err={maxe:.2e}  "
                     f"pat_MSE={pat_mses[-1]:.2e}  "
                     f"ml_mse={ml_mse:.2e}  {dt:.1f}s  {tag}")
        except Exception as e:
            log.info(f"  Case {case_i+1:3d}: FAILED ({e})")

    if not errs:
        log.info("  No successful cases.")
        return

    errs = np.array(errs)
    ml_only_errs = np.array(ml_only_errs)
    avg_time = np.mean(times_each)

    log.info(f"\n  {len(errs)} cases, avg {avg_time:.1f}s/case")

    # ML-only stats
    log.info(f"\n  ML-only prediction errors (no refinement):")
    log.info(f"  {'Param':8s} {'Median':>10s} {'Mean':>10s} {'Max':>10s}")
    for i in range(N_PAR):
        med = np.median(ml_only_errs[:, i])
        mean = np.mean(ml_only_errs[:, i])
        mx = np.max(ml_only_errs[:, i])
        log.info(f"  {NAMES[i]:8s} {med:10.4f} {mean:10.4f} {mx:10.4f}")

    # After-refinement stats
    log.info(f"\n  After refinement:")
    log.info(f"  {'Param':8s} {'Median':>10s} {'Mean':>10s} {'Max':>10s}")
    for i in range(N_PAR):
        med = np.median(errs[:, i])
        mean = np.mean(errs[:, i])
        mx = np.max(errs[:, i])
        log.info(f"  {NAMES[i]:8s} {med:10.6f} {mean:10.6f} {mx:10.6f}")

    log.info(f"\n  Pattern MSE:  median={np.median(pat_mses):.2e}  "
             f"mean={np.mean(pat_mses):.2e}  max={np.max(pat_mses):.2e}")

    n_perfect = int(np.sum(np.max(errs, axis=1) < 1e-3))
    n_close   = int(np.sum(np.max(errs, axis=1) < 0.01))
    log.info(f"\n  < 1e-3 max error: {n_perfect}/{len(errs)}")
    log.info(f"  < 1e-2 max error: {n_close}/{len(errs)}")

    # Save results
    results = {
        'method': 'V3 unified CNN (phase-trained)',
        'n_test': len(errs),
        'avg_time_s': round(avg_time, 1),
        'n_perfect_1e3': n_perfect,
        'n_close_1e2': n_close,
        'per_param_max_error': {
            NAMES[i]: float(np.max(errs[:, i])) for i in range(N_PAR)
        },
        'per_param_median_error': {
            NAMES[i]: float(np.median(errs[:, i])) for i in range(N_PAR)
        },
        'ml_only_per_param_median_error': {
            NAMES[i]: float(np.median(ml_only_errs[:, i])) for i in range(N_PAR)
        },
        'pattern_mse_median': float(np.median(pat_mses)),
        'pattern_mse_mean': float(np.mean(pat_mses)),
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\n  Results saved to {os.path.basename(RESULTS_PATH)}")

    return results


# ================================================================
#  Main
# ================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description='V3 Unified ML Training for Risley Prism Inverse Problem')

    # Mode
    ap.add_argument('--eval', action='store_true',
                    help='Evaluate saved model (no training)')
    ap.add_argument('--retrain', action='store_true',
                    help='Delete saved model and retrain from scratch')

    # Training
    ap.add_argument('--phase', type=int, default=1, choices=[1, 2, 3],
                    help='Training phase: 1=MSE, 2=weighted, 3=physics')
    ap.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs for this phase')
    ap.add_argument('--lr', type=float, default=1e-3,
                    help='Peak learning rate')
    ap.add_argument('--batch-size', type=int, default=512,
                    help='Batch size')
    ap.add_argument('--resume', action='store_true',
                    help='Resume from checkpoint')

    # Data
    ap.add_argument('--data-size', type=int, default=1_000_000,
                    help='Training set size')
    ap.add_argument('--val-size', type=int, default=50_000,
                    help='Validation set size')
    ap.add_argument('--regen-data', action='store_true',
                    help='Force regenerate training data')

    # Architecture
    ap.add_argument('--base-ch', type=int, default=64,
                    help='Base CNN channels (64->~2.1M params, 96->~4.5M)')
    ap.add_argument('--embed-dim', type=int, default=512,
                    help='Fusion embedding dimension')

    # Physics loss (phase 3)
    ap.add_argument('--physics-every', type=int, default=4,
                    help='Compute physics loss every N batches')
    ap.add_argument('--physics-weight', type=float, default=0.1,
                    help='Weight of physics loss relative to param loss')

    # Evaluation
    ap.add_argument('--n-test', type=int, default=30,
                    help='Number of test cases for evaluation')
    ap.add_argument('--heavy', action='store_true',
                    help='Use heavy refinement in evaluation')
    ap.add_argument('--no-refine', action='store_true',
                    help='ML prediction only, no gradient refinement')

    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    log = setup_logging()

    log.info(f"{'='*70}")
    log.info(f" V3 UNIFIED RISLEY INVERSE SOLVER")
    log.info(f"{'='*70}")
    log.info(f"Device: {DEVICE}")
    log.info(f"Mode: {'eval' if args.eval else f'train phase {args.phase}'}")

    # Build model
    model = RisleyNetV3(base_ch=args.base_ch, embed_dim=args.embed_dim)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: RisleyNetV3 (base_ch={args.base_ch}, "
             f"embed_dim={args.embed_dim})")
    log.info(f"  Total parameters: {n_params:,}")

    if args.retrain:
        for f in [MODEL_PATH, CKPT_PATH]:
            if os.path.exists(f):
                os.remove(f)
                log.info(f"  Deleted {os.path.basename(f)}")

    if args.eval:
        # ---- Evaluation only ----
        if not os.path.exists(MODEL_PATH):
            log.info("  Model not found. Train first.")
            sys.exit(1)
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        log.info(f"  Loaded model from {os.path.basename(MODEL_PATH)}")

        evaluate(model, n_test=args.n_test,
                 heavy=(args.heavy and not args.no_refine))

    else:
        # ---- Training ----
        log.info(f"\n--- Data ---")
        n_total = args.data_size + args.val_size
        tp, ta, vp, va = load_or_gen_data(
            args.data_size, args.val_size, force_regen=args.regen_data)

        # Stack train + val for unified feature extraction
        all_pats = np.concatenate([tp, vp], axis=0)
        all_params = np.concatenate([ta, va], axis=0)

        log.info(f"\n--- Feature Extraction ---")
        train_ds, val_ds = prepare_datasets(
            all_pats, all_params, len(tp), len(vp))

        del tp, ta, vp, va, all_pats, all_params

        log.info(f"\n--- Training Phase {args.phase} ---")
        model = train(model, train_ds, val_ds, args)

        # Quick evaluation after training
        log.info(f"\n--- Quick Evaluation ---")
        evaluate(model, n_test=min(10, args.n_test), heavy=False)
