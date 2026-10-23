#!/usr/bin/env python3
"""Trials in the ready dataset that pass the MSE < 1e-12 acceptance test
while the max parameter error is >= 1e-3 (pattern-equivalent but wrong).
Writes out/mse_accepted_wrong.json: [case, T, err, mse, how, ax_min, pair_sep]."""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'results', 'sweep_adaptive_ready_0000000.jsonl')
rows, trials = [], 0
for line in open(SRC):
    r = json.loads(line)
    for t in r['trail']:
        trials += 1
        if t['mse'] < 1e-12 and t['err'] >= 1e-3:
            rows.append((r['case'], t['T'], t['err'], t['mse'], t['how'],
                         r['ax_min'], r['pair_sep']))
json.dump(rows, open(os.path.join(HERE, 'out', 'mse_accepted_wrong.json'), 'w'))
print('trials', trials, 'accepted but wrong', len(rows))
