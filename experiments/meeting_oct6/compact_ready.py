#!/usr/bin/env python3
"""Compact the 669,574-case ready dataset into a numpy table (one row per
case) so every downstream analysis is a fast array operation.

  python3 experiments/meeting_oct6/compact_ready.py --part k --nparts 4 --out DIR
  python3 experiments/meeting_oct6/compact_ready.py --merge --out DIR
"""
import os, sys, json, argparse
import numpy as np

SRC = os.path.join(os.path.dirname(__file__), '..', 'results',
                   'sweep_adaptive_ready_0000000.jsonl')
CERTS = ['relation', 'weak-prism', 'glitch-floor', 'close-pair',
         'rank-deficient', 'basin', 'no-generators']
HOWS = ['primary', 'recal', 'zero', 'flipweak', 'alt0', 'alt1', 'no-speeds',
        'other']
TS = [10.0, 20.0, 40.0, 80.0]


def how_code(h):
    h = (h or '').split('+')[0]
    return HOWS.index(h) if h in HOWS else HOWS.index('other')


def part(k, nparts, out):
    size = os.path.getsize(SRC)
    lo, hi = size * k // nparts, size * (k + 1) // nparts
    cols = {c: [] for c in ['case', 'T_solved', 'pair_sep', 'rel_gap',
                            'ax_min', 'cyc', 'ntrial', 'how_win', 'how_last',
                            'err_final', 'secs_total', 'cert_bits',
                            'T_req', 'sigma_N10', 'sigma_N20', 'sigma_N40',
                            'sigma_N80', 'treq10', 'rescue_T']}
    perr = []
    sflip = []
    with open(SRC, 'rb') as f:
        f.seek(lo)
        if k > 0:
            f.readline()
        while f.tell() <= hi:
            line = f.readline()
            if not line:
                break
            r = json.loads(line)
            tr = r['trail']
            cols['case'].append(r['case'])
            Ts = r.get('T_solved')
            cols['T_solved'].append(Ts if Ts is not None else np.nan)
            for m in ['pair_sep', 'rel_gap', 'ax_min', 'cyc']:
                v = r.get(m)
                cols[m].append(np.nan if v is None else v)
            cols['ntrial'].append(len(tr))
            last = tr[-1]
            cols['how_win'].append(how_code(last['how']) if Ts else -1)
            cols['how_last'].append(how_code(last['how']))
            cols['err_final'].append(last['err'])
            cols['secs_total'].append(sum(t.get('secs', 0) for t in tr))
            bits = 0
            for c in (last.get('cert') or []) if not Ts else []:
                key = c.split()[0].split(':')[0]
                if key in CERTS:
                    bits |= 1 << CERTS.index(key)
            cols['cert_bits'].append(bits)
            treq = last.get('T_req') if not Ts else None
            cols['T_req'].append(np.nan if treq is None else treq)
            sig = {t['T']: t.get('sigma_N') for t in tr}
            for T in TS:
                v = sig.get(T)
                cols[f'sigma_N{int(T)}'].append(np.nan if v is None else v)
            t0 = tr[0].get('T_req') if tr[0]['err'] >= 1e-3 else None
            cols['treq10'].append(np.nan if t0 is None else t0)
            cols['rescue_T'].append(Ts if (Ts and len(tr) > 1) else np.nan)
            if Ts and last.get('x18') is not None:
                perr.append(np.asarray(last['x18'], np.float64))
            else:
                perr.append(np.full(18, np.nan))
    arr = {c: np.asarray(v) for c, v in cols.items()}
    arr['x18'] = np.asarray(perr, np.float64)
    np.savez(os.path.join(out, f'part{k}.npz'), **arr)
    print(k, len(arr['case']))


def merge(out):
    parts = sorted(p for p in os.listdir(out) if p.startswith('part'))
    ds = [dict(np.load(os.path.join(out, p))) for p in parts]
    m = {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}
    _, idx = np.unique(m['case'], return_index=True)
    print('rows', len(m['case']), 'unique cases', len(idx))
    m = {k: v[idx] for k, v in m.items()}
    np.savez(os.path.join(out, 'campaign.npz'), **m)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int)
    ap.add_argument('--nparts', type=int, default=4)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    merge(a.out) if a.merge else part(a.part, a.nparts, a.out)
