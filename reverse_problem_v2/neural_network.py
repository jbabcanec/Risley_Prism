#!/usr/bin/env python3
"""
PyTorch Neural Network for the inverse Risley prism problem.

Works with physical prism parameters (PrismParameters):
  - n_prisms: 1-3 physical wedge prisms
  - rotation_speeds: one per prism (both faces rotate in lockstep)
  - wedge_angles_x, wedge_angles_y: exit-face tilt per prism
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, List, Tuple, Optional


class RisleyNet(nn.Module):
    def __init__(self, input_dim: int = 600, prism_counts: List[int] = None):
        super().__init__()
        if prism_counts is None:
            prism_counts = [2, 3]
        self.prism_counts = prism_counts

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 768), nn.BatchNorm1d(768), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(prism_counts)))

        # Per-prism-count regression heads: output 3*n_prisms values
        self.reg_heads = nn.ModuleDict()
        for np_ in prism_counts:
            self.reg_heads[str(np_)] = nn.Sequential(
                nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.05),
                nn.Linear(128, 3 * np_),  # speeds + angles_x + angles_y
            )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        reg_outs = {np_: self.reg_heads[str(np_)](feat) for np_ in self.prism_counts}
        return logits, reg_outs


class PrismPredictor:
    SPEED_SCALE = 3.5
    ANGLE_SCALE = 16.0
    N_POINTS = 200

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: RisleyNet = None
        self.input_mean: np.ndarray = None
        self.input_std: np.ndarray = None
        self.prism_counts: List[int] = None
        self.pc_to_idx: Dict[int, int] = None

    @staticmethod
    def _build_features(patterns: np.ndarray) -> np.ndarray:
        N, T, _ = patterns.shape
        raw = np.concatenate([patterns[:, :, 0], patterns[:, :, 1]], axis=1)
        xc = patterns[:, :, 0] - patterns[:, :, 0].mean(axis=1, keepdims=True)
        yc = patterns[:, :, 1] - patterns[:, :, 1].mean(axis=1, keepdims=True)
        fft = np.concatenate([
            np.abs(np.fft.rfft(xc, axis=1))[:, 1:],
            np.abs(np.fft.rfft(yc, axis=1))[:, 1:],
        ], axis=1)
        return np.concatenate([raw, fft], axis=1).astype(np.float32)

    @classmethod
    def _build_features_single(cls, pattern: np.ndarray) -> np.ndarray:
        return cls._build_features(pattern[np.newaxis])[0]

    def _expand(self, np_: int, compact: np.ndarray) -> np.ndarray:
        s  = compact[:np_]       * self.SPEED_SCALE
        ax = compact[np_:2*np_]  * self.ANGLE_SCALE
        ay = compact[2*np_:]     * self.ANGLE_SCALE
        out = np.zeros(9, dtype=np.float32)
        out[:np_], out[3:3+np_], out[6:6+np_] = s, ax, ay
        return out

    # ── training ──────────────────────────────────────────────────────
    def train(self, patterns, prism_counts_arr, params_array, *,
              prism_counts=None, epochs=200, batch_size=512,
              lr=1e-3, val_split=0.12):

        if prism_counts is None:
            prism_counts = sorted(set(prism_counts_arr.tolist()))
        self.prism_counts = prism_counts
        self.pc_to_idx = {pc: i for i, pc in enumerate(prism_counts)}

        N = len(patterns)
        X = self._build_features(patterns)
        self.input_mean = X.mean(axis=0)
        self.input_std = X.std(axis=0)
        self.input_std[self.input_std < 1e-8] = 1.0
        X = (X - self.input_mean) / self.input_std

        Y_class = np.array([self.pc_to_idx[pc] for pc in prism_counts_arr], dtype=np.int64)

        n_val = int(N * val_split)
        idx = np.random.permutation(N)
        vi, ti = idx[:n_val], idx[n_val:]
        dev = self.device

        X_tr  = torch.tensor(X[ti],       dtype=torch.float32).to(dev)
        X_va  = torch.tensor(X[vi],       dtype=torch.float32).to(dev)
        Yc_tr = torch.tensor(Y_class[ti], dtype=torch.long).to(dev)
        Yc_va = torch.tensor(Y_class[vi], dtype=torch.long).to(dev)
        Pc_tr = torch.tensor(prism_counts_arr[ti], dtype=torch.long).to(dev)
        Pc_va = torch.tensor(prism_counts_arr[vi], dtype=torch.long).to(dev)
        Yp_tr = torch.tensor(params_array[ti], dtype=torch.float32).to(dev)
        Yp_va = torch.tensor(params_array[vi], dtype=torch.float32).to(dev)

        input_dim = X.shape[1]
        self.model = RisleyNet(input_dim, prism_counts).to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5, min_lr=1e-6)
        ce = nn.CrossEntropyLoss()

        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val, best_state, patience = float('inf'), None, 0
        print(f"  Training {len(ti)} / val {len(vi)} -- features {input_dim} -- prisms={prism_counts}")

        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(len(X_tr), device=dev)
            ep_loss, nb = 0.0, 0

            for s in range(0, len(X_tr), batch_size):
                bi = perm[s:s + batch_size]
                logits, reg_outs = self.model(X_tr[bi])
                loss_cls = ce(logits, Yc_tr[bi])

                loss_reg = torch.tensor(0.0, device=dev)
                nh = 0
                bpc, bp = Pc_tr[bi], Yp_tr[bi]
                for pc in prism_counts:
                    m = (bpc == pc)
                    if not m.any(): continue
                    pred = reg_outs[pc][m]
                    tp = bp[m]
                    tgt = torch.cat([
                        tp[:, :pc]     / self.SPEED_SCALE,
                        tp[:, 3:3+pc]  / self.ANGLE_SCALE,
                        tp[:, 6:6+pc]  / self.ANGLE_SCALE,
                    ], dim=1)
                    loss_reg = loss_reg + ((pred - tgt)**2).mean()
                    nh += 1
                if nh: loss_reg /= nh

                loss = loss_cls + 30.0 * loss_reg
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item(); nb += 1

            self.model.eval()
            with torch.no_grad():
                vl, vr = self.model(X_va)
                v_cls = ce(vl, Yc_va).item()
                v_reg, vn = 0.0, 0
                for pc in prism_counts:
                    m = (Pc_va == pc)
                    if not m.any(): continue
                    pred = vr[pc][m]; tp = Yp_va[m]
                    tgt = torch.cat([tp[:,:pc]/self.SPEED_SCALE, tp[:,3:3+pc]/self.ANGLE_SCALE, tp[:,6:6+pc]/self.ANGLE_SCALE], dim=1)
                    v_reg += ((pred-tgt)**2).mean().item(); vn += 1
                if vn: v_reg /= vn
                v_loss = v_cls + 30.0 * v_reg
                v_acc = (vl.argmax(1) == Yc_va).float().mean().item()

            t_loss = ep_loss / nb
            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)
            history['val_acc'].append(v_acc)
            sched.step(v_loss)

            if v_loss < best_val:
                best_val = v_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if (epoch+1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  t={t_loss:.4f} v={v_loss:.4f} acc={v_acc:.1%} lr={opt.param_groups[0]['lr']:.1e}")
            if patience >= 25:
                print(f"  Early stopping at epoch {epoch+1}"); break

        if best_state:
            self.model.load_state_dict(best_state); self.model.to(dev)
        print(f"  Best val loss: {best_val:.4f}")
        return history

    # ── inference ─────────────────────────────────────────────────────
    def predict(self, pattern, n_prisms=None):
        if self.model is None: raise ValueError("Not trained")
        feat = self._build_features_single(pattern)
        expected = self.input_mean.shape[0]
        if len(feat) != expected:
            from scipy.interpolate import interp1d
            t_o, t_n = np.linspace(0,1,len(pattern)), np.linspace(0,1,self.N_POINTS)
            resampled = np.column_stack([interp1d(t_o,pattern[:,0])(t_n), interp1d(t_o,pattern[:,1])(t_n)])
            feat = self._build_features_single(resampled)

        feat = ((feat - self.input_mean) / self.input_std).astype(np.float32)
        x = torch.tensor(feat).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, reg_outs = self.model(x)

        probs = torch.softmax(logits, 1).cpu().numpy()[0]
        if n_prisms is None:
            cls_idx = logits.argmax(1).item()
            n_prisms = self.prism_counts[cls_idx]

        compact = reg_outs[n_prisms][0].cpu().numpy()
        p9 = self._expand(n_prisms, compact)
        return {
            'n_prisms': n_prisms,
            'rotation_speeds': p9[:n_prisms].tolist(),
            'wedge_angles_x': p9[3:3+n_prisms].tolist(),
            'wedge_angles_y': p9[6:6+n_prisms].tolist(),
            'class_probs': {pc: float(probs[i]) for i, pc in enumerate(self.prism_counts)},
        }

    def predict_batch(self, patterns, n_prisms=None):
        N = len(patterns)
        X = self._build_features(patterns).astype(np.float32)
        X = (X - self.input_mean) / self.input_std
        x = torch.tensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, reg_outs = self.model(x)

        if n_prisms is not None:
            pred_np = np.full(N, n_prisms, dtype=np.int32)
        else:
            cls_idx = logits.argmax(1).cpu().numpy()
            pred_np = np.array([self.prism_counts[i] for i in cls_idx], dtype=np.int32)

        params = np.zeros((N, 9), dtype=np.float32)
        for pc in self.prism_counts:
            mask = pred_np == pc
            if not mask.any(): continue
            compact = reg_outs[pc][torch.tensor(mask)].cpu().numpy()
            for j, idx in enumerate(np.where(mask)[0]):
                params[idx] = self._expand(pc, compact[j])
        return pred_np, params

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save({'state_dict': self.model.state_dict(), 'input_mean': self.input_mean,
                     'input_std': self.input_std, 'input_dim': self.model.backbone[0].in_features,
                     'prism_counts': self.prism_counts}, os.path.join(path, 'model.pt'))

    def load(self, path):
        data = torch.load(os.path.join(path, 'model.pt'), map_location=self.device, weights_only=False)
        self.input_mean = data['input_mean']; self.input_std = data['input_std']
        self.prism_counts = data['prism_counts']
        self.pc_to_idx = {pc: i for i, pc in enumerate(self.prism_counts)}
        self.model = RisleyNet(data['input_dim'], self.prism_counts).to(self.device)
        self.model.load_state_dict(data['state_dict']); self.model.eval()
