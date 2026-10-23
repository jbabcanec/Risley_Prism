#!/usr/bin/env python3
"""Regenerate the ground-truth 18-vector of every campaign case from its
seed (case_at(case, 424242)) and append truth + TIR margin to the table."""
import os, sys, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from risley_lattice.model import case_at
from tir import tir_margin18

ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True)
ap.add_argument('--part', type=int); ap.add_argument('--nparts', type=int, default=4)
ap.add_argument('--merge', action='store_true')
a = ap.parse_args()
m = dict(np.load(os.path.join(a.dir, 'campaign.npz')))
if not a.merge:
    idx = np.array_split(np.arange(len(m['case'])), a.nparts)[a.part]
    tr = np.array([case_at(int(c), 424242) for c in m['case'][idx]])
    np.save(os.path.join(a.dir, f'truth{a.part}.npy'), tr)
else:
    tr = np.concatenate([np.load(os.path.join(a.dir, f'truth{k}.npy'))
                         for k in range(a.nparts)])
    m['truth'] = tr
    m['tir_margin'] = tir_margin18(tr)
    # sanity: stored ax_min must equal regenerated truth
    d = np.nanmax(np.abs(m['ax_min'] - np.abs(tr[:, 3:6]).min(1)))
    print('max |ax_min stored - regenerated| =', d)
    np.savez(os.path.join(a.dir, 'campaign.npz'), **m)
