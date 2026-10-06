import csv, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
baseline=json.loads((HERE/'baseline_summary.json').read_text(encoding='utf-8'))
noise=json.loads((HERE/'noise_summary.json').read_text(encoding='utf-8'))
names=baseline['names']
levels=[1e-8,1e-6,1e-4,1e-3]
cases=list(dict.fromkeys(row['case_id'] for row in noise['rows']))
with (HERE/'baseline_coordinate_errors.csv').open('w',newline='',encoding='utf-8') as h:
    w=csv.writer(h);w.writerow(['case_id','status','wall_s','mse','max_native_error','physical_guards_pass']+names)
    for row in baseline['rows']:
        w.writerow([row['case_id'],row['status'],row['wall_s'],row.get('residual_mse'),row.get('native_max_error'),row.get('physical_guards_pass')]+row.get('native_absolute_errors',['']*18))
with (HERE/'noise_coordinate_max_by_case_eta.csv').open('w',newline='',encoding='utf-8') as h:
    w=csv.writer(h);w.writerow(['case_id','eta','pattern_count','max_native_error','worst_coordinate']+names)
    for case in cases:
        for eta in levels:
            rows=[r for r in noise['rows'] if r['case_id']==case and r['eta']==eta]
            errs=np.max([r['native_absolute_errors'] for r in rows],axis=0)
            w.writerow([case,eta,len(rows),float(np.max(errs)),names[int(np.argmax(errs))]]+errs.tolist())
with (HERE/'linearized_native_noise_gains.csv').open('w',newline='',encoding='utf-8') as h:
    w=csv.writer(h);w.writerow(['case_id','jacobian_condition']+names)
    for row in noise['sensitivity']:
        w.writerow([row['case_id'],row['jacobian_condition']]+row['derivative_linf_noise_gains'])
lines=[
    'Full18 baseline and local-noise characterization, 2026-10-02',
    '',
    'Protocol: shared cases.json; 200 time-tagged 2D positions, t=k/20.',
    'All 18 parameters unknown in the blind solvers. Uniform cases drawn across',
    'the original full bounds, without sorting physical order or clipping slow',
    'speeds; only strict sampled physical transmission filters were applied.',
    'Truth was used solely for scoring, never passed to the blind worker.',
    '',
    'Original solve18 standard baseline (60 second caps, two concurrent workers):',
]
for row in baseline['rows']:
    lines.append(f"  {row['case_id']}: {row['status']}; native max={row.get('native_max_error'):.9g}; MSE={row.get('residual_mse'):.6g}; wall={row['wall_s']:.3f}s; physical={row.get('physical_guards_pass')}")
random=[r for r in baseline['rows'] if r['case_id'].startswith('random')]
lines.extend(['',f"Random cases: {sum(r['recovered_below_0_001'] for r in random)}/{len(random)} recovered under native .001 criterion.",
              'The separate moderate positive control recovered all18 below .001.',
              'No failures are removed, permuted for scoring, or interpreted as impossibility.',
              '',
              'Local-noise experiment: six saved blind-recovered noiseless estimates;',
              'all18 released in each TRF refinement, max_nfev=1000; 30 second cap.',
              'Four noise patterns per level: fixed-seed dense uniform perturbation',
              'and signs of the pseudoinverse rows for three locally sensitive native',
              'coordinates. All amplitudes are bounded by eta. These are deterministic',
              'test patterns, not a probability model and not exhaustive worst cases.',
              'All noise jobs contain the actual recorded position arrays.',
              '',
              'This is WARM-START LOCAL PERTURBATION, not blind noisy recovery.',
              'Least-squares residuals often exceed the hard allowance at some samples.',
              'These results are measured algorithm sensitivity, not compatible-set',
              'diameters, deterministic accuracy bounds, or impossibility witnesses.',
              '',
              'Largest native-coordinate error among four tested patterns:',
              'case                         eta=1e-8       eta=1e-6       eta=1e-4       eta=1e-3'])
for case in cases:
    vals=[max(r['native_max_error'] for r in noise['rows'] if r['case_id']==case and r['eta']==eta) for eta in levels]
    lines.append(f"{case:26s}"+''.join(f'{v:15.7g}' for v in vals))
lines.extend(['',f"Noise jobs: {len(noise['rows'])}; returned={sum(r['status']=='returned' for r in noise['rows'])}; physical guards passed={sum(r.get('physical_guards_pass',False) for r in noise['rows'])}.",
              'Point guard validity uses the original interval package assumptions;',
              'it does not prove global uniqueness or accuracy.',
              '',
              'Files: baseline_summary.json and noise_summary.json retain full saved',
              'vectors, all18 errors, residuals, point margins, times and provenance.',
              'noise_coordinate_errors.csv gives each individual noisy result.',
              'noise_coordinate_max_by_case_eta.csv aggregates each native coordinate.',
              'linearized_native_noise_gains.csv gives local J-pseudoinverse row L1 gains',
              '(native-coordinate change per unit observation perturbation), not proofs.',
              'worker.py / run.py / summarize.py reproduce these isolated experiments.',
              'No original Dropbox project file was modified.'])
(HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
