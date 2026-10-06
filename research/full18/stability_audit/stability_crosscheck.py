"""Post-score saved optical research evidence; never fit or regenerate data.

The legacy core is imported read-only. All writes stay in this audit directory.
An independent directed-interval oracle checks all dataset truths and witness
endpoints, and any candidate whose declared hard-band classification changes.
Supplied binary64 times/parameters/observations are preserved exactly.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
SOURCE = Path(r'C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge')
sys.path[:0] = [str(WORK / 'stable_model'), str(SOURCE)]
from forward import vec2pat_stable, default_times
from oracle import reference, exact, bounds, error_upper
from risley_lattice.model import vec2pat as legacy_forward

started = time.perf_counter()
snapshots, issues, rows = {}, [], []
cache = {}
datasets = {}


def load(path):
    path = Path(path)
    raw = path.read_bytes()
    snapshots[str(path.relative_to(WORK))] = {
        'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return json.loads(raw)


def finite(value):
    return float(value) if math.isfinite(float(value)) else None


def maxabs(x):
    return float(np.max(np.abs(x)))


def interval_residual(y, ref):
    lower, upper = 0., 0.
    for pair, rp in zip(y, ref):
        for a, b in zip(pair, rp):
            lo, hi = bounds(exact(a) - b)
            upper = max(upper, abs(lo), abs(hi))
            lower = max(lower, 0. if lo <= 0 <= hi else min(abs(lo), abs(hi)))
    return {'lower': lower, 'upper': upper}


def oracle_class(enclosure, eta):
    if enclosure['upper'] <= eta:
        return 'inside'
    if enclosure['lower'] > eta:
        return 'outside'
    return 'undecided'


def predictions(theta, times, need_oracle=False):
    theta = np.asarray(theta, dtype=float)
    times = np.asarray(times, dtype=float)
    key = (theta.tobytes(), times.tobytes())
    if key not in cache:
        entry = {'theta': theta, 'times': times}
        try:
            entry['legacy'] = np.asarray(legacy_forward(theta, n_points=200, time_limit=10.))
        except Exception as exc:
            entry['legacy_error'] = repr(exc)
        try:
            stable, diag = vec2pat_stable(theta, times=times, return_diagnostics=True)
            entry['stable'], entry['margins'] = stable, diag['margins']
        except Exception as exc:
            entry['stable_error'] = repr(exc)
        cache[key] = entry
    entry = cache[key]
    if need_oracle and 'oracle' not in entry and 'oracle_error' not in entry:
        try:
            entry['oracle'], entry['oracle_margins'] = reference(theta, times, dps=70)
            for kind in ('legacy', 'stable'):
                if kind in entry:
                    entry[kind + '_oracle_error_upper'] = error_upper(entry[kind], entry['oracle'])
        except Exception as exc:
            entry['oracle_error'] = repr(exc)
    return entry


def audit(label, theta, observed, *, eta=0., times=None, source=None,
          truth=None, names=None, role='candidate', full_oracle=False,
          dataset=None, saved_residual=None, saved_compatible=None):
    y = np.asarray(observed, dtype=float).reshape(-1, 2)
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (18,) or y.shape != (200, 2):
        raise ValueError(f'{label}: wrong shapes {theta.shape}, {y.shape}')
    times = default_times() if times is None else np.asarray(times, dtype=float)
    entry = predictions(theta, times)
    row = dict(label=label, source=source, dataset=dataset, role=role, eta=float(eta),
               times_match_legacy_default=bool(np.array_equal(times, default_times())),
               saved_residual=saved_residual, saved_compatible=saved_compatible,
               observation_sha256=hashlib.sha256(y.tobytes()).hexdigest())
    for kind in ('legacy', 'stable'):
        if kind in entry:
            residual = entry[kind] - y
            row[kind + '_max_residual'] = finite(maxabs(residual))
            row[kind + '_rms'] = finite(np.sqrt(np.mean(residual**2)))
            row[kind + '_inside_eta'] = bool(np.isfinite(residual).all() and maxabs(residual) <= eta)
            row[kind + '_within_1e8_numerical_threshold'] = bool(np.isfinite(residual).all() and maxabs(residual) <= 1e-8)
        else:
            row[kind + '_error'] = entry.get(kind + '_error')
    if 'stable' in entry:
        row['stable_physical_margins'] = entry['margins']
    if 'legacy' in entry and 'stable' in entry:
        row['legacy_stable_max_difference'] = finite(maxabs(entry['legacy']-entry['stable']))
        row['eta_classification_changed'] = row['legacy_inside_eta'] != row['stable_inside_eta']
        row['numerical_threshold_classification_changed'] = row['legacy_within_1e8_numerical_threshold'] != row['stable_within_1e8_numerical_threshold']
    else:
        row['eta_classification_changed'] = None
    if truth is not None:
        errors = np.abs(theta-np.asarray(truth, dtype=float))
        row['absolute_native_errors'] = dict(zip(names, map(float, errors)))
        row['max_native_error'] = float(errors.max())
        row['all18_error_below_001'] = bool(np.all(errors < .001))
        row['coordinates_below_001'] = int(np.count_nonzero(errors < .001))
    # The reference is evaluated at every one of the 200 supplied times.
    need_oracle = full_oracle or row.get('eta_classification_changed') or row.get('numerical_threshold_classification_changed')
    if need_oracle:
        entry = predictions(theta, times, need_oracle=True)
        if 'oracle' in entry:
            row['oracle_scope'] = 'all 200 supplied binary64 timestamps, 70-digit directed intervals'
            enc = interval_residual(y, entry['oracle'])
            row['oracle_residual_enclosure'] = enc
            row['oracle_eta_classification'] = oracle_class(enc, eta)
            row['oracle_1e8_classification'] = oracle_class(enc, 1e-8)
            row['oracle_physical_margins'] = entry['oracle_margins']
            for kind in ('legacy', 'stable'):
                row[kind + '_oracle_error_upper'] = entry.get(kind + '_oracle_error_upper')
        else:
            row['oracle_error'] = entry.get('oracle_error')
    rows.append(row)
    return row


def add_truth_for_record(dataset, cid, y, eta, label, source):
    data, indexed = datasets[dataset]
    case = indexed[cid]
    row = audit(label, case['truth'], y, eta=eta, times=data['timestamps'],
                source=source, role='truth_against_actual_record', full_oracle=True,
                dataset=dataset)
    row['stored_noise_max_amplitude'] = maxabs(np.asarray(y)-np.asarray(case['observed']))
    row['stored_noise_within_eta'] = row['stored_noise_max_amplitude'] <= eta
    return row


for filename in ('cases.json', 'holdout.json', 'combined_holdout.json', 'collision_holdout.json'):
    data = load(WORK / filename)
    indexed = {c['id']: c for c in data['cases']}
    datasets[filename] = data, indexed
    for case in data['cases']:
        audit(f'{filename}:{case["id"]}:generation', case['truth'], case['observed'],
              times=data['timestamps'], source=filename, role='benchmark_truth',
              full_oracle=True, dataset=filename)
    print(json.dumps({'completed_truth_dataset': filename, 'count': len(indexed)}), flush=True)

data, indexed = datasets['combined_holdout.json']
for folder in ('noiseless', 'noise_1e6', 'noise_1e4', 'noise_1e3', 'noise_1e2'):
    paths = sorted((WORK / 'combined_validation' / folder).glob('*.json'))
    for path in paths:
        try:
            result = load(path)
            cid = result.get('id', path.stem)
            if cid not in indexed or 'theta' not in result:
                issues.append({'source': str(path.relative_to(WORK)), 'issue': 'No identified saved theta18'})
                continue
            case = indexed[cid]
            if 'actual_observations' in result:
                y = result['actual_observations']
            elif folder == 'noiseless':
                y = case['observed']
            else:
                raise ValueError('No saved actual noisy observation record; not reconstructed')
            eta = result.get('eta', result.get('noise_amplitude', 0.))
            source = str(path.relative_to(WORK))
            audit(source, result['theta'], y, eta=eta, times=data['timestamps'], source=source,
                  truth=case['truth'], names=data['names'], dataset='combined_holdout.json',
                  saved_residual=result.get('max_residual'),
                  saved_compatible=result.get('canonical_hard_band_compatible'))
            add_truth_for_record('combined_holdout.json', cid, y, eta, source+':truth', source)
        except Exception as exc:
            issues.append({'source': str(path.relative_to(WORK)), 'issue': repr(exc)})
    print(json.dumps({'completed_solution_folder': folder, 'saved_files': len(paths)}), flush=True)

for path in sorted((WORK / 'low_frequency').glob('random*.json')):
    result = load(path)
    if 'theta' not in result:
        continue
    cid = result.get('id', path.stem.replace('_release', ''))
    case = indexed[cid]
    audit(str(path.relative_to(WORK)), result['theta'], case['observed'],
          times=data['timestamps'], source=str(path.relative_to(WORK)), truth=case['truth'],
          names=data['names'], dataset='combined_holdout.json', full_oracle=True,
          saved_residual=result.get('max_residual'))

adverse = load(WORK / 'profiles' / 'adverse_observations_only.json')
adverse = {c['id']: c['observed'] for c in adverse['cases']}
for filename in ('result_native.json', 'result_replay.json', 'fresh_random_03.json', 'fresh_random_07.json'):
    path = WORK / 'gain_continuation' / filename
    if not path.exists():
        issues.append({'source': str(path.relative_to(WORK)), 'issue': 'File absent at snapshot'})
        continue
    result = load(path)
    cid = result.get('case_id', 'eta_1e-8_separated_speeds')
    source = str(path.relative_to(WORK))
    if filename.startswith('fresh_'):
        case = indexed[cid]
        audit(source, result['theta'], case['observed'], times=data['timestamps'], source=source,
              truth=case['truth'], names=data['names'], dataset='combined_holdout.json',
              full_oracle=True, saved_residual=result.get('strict_numerical_max_residual'))
    else:
        audit(source, result['theta'], adverse[cid], eta=1e-8, source=source, full_oracle=True,
              saved_residual=result.get('strict_numerical_max_residual'))

pairs = []
path = WORK / 'gain_continuation' / 'compatible_profiles.json'
if path.exists():
    result = load(path)
    record = adverse['eta_1e-8_separated_speeds']
    source = str(path.relative_to(WORK))
    base = audit(source+':candidate', result['candidate'], record, eta=result['eta'], source=source,
                 full_oracle=True, role='witness_endpoint')
    for i, profile in enumerate(result['profiles']):
        other = audit(source+f':profile_{i}', profile['theta'], record, eta=result['eta'], source=source,
                      full_oracle=True, role='witness_endpoint')
        separation = np.abs(np.asarray(profile['theta'])-np.asarray(result['candidate']))
        pairs.append({'label': source+f':profile_{i}', 'endpoint_labels': [base['label'], other['label']],
                      'eta': result['eta'], 'max_native_separation': float(separation.max()),
                      'coordinate_separations': dict(zip(result['names'], map(float, separation)))})

path = WORK / 'profiles' / 'random00_eta1e4.json'
result = load(path)
pair = result['actual_record_pair_witness']
source = str(path.relative_to(WORK))
endpoints = []
for side in ('endpoint_a', 'endpoint_b'):
    theta = pair[side]['theta']
    row = audit(source+':'+side, theta, result['actual_observations'], eta=result['eta'],
                source=source, full_oracle=True, role='witness_endpoint')
    endpoints.append(row['label'])
pairs.append({'label': source, 'endpoint_labels': endpoints, 'eta': result['eta'],
              'coordinate': pair['coordinate'], 'separation': pair['separation'],
              'unavoidable_coordinate_error': pair['unavoidable_coordinate_error']})
add_truth_for_record('cases.json', 'random_00', result['actual_observations'], result['eta'],
                     source+':truth', source)

path = WORK / 'ambiguity' / 'witnesses.json'
result = load(path)
source = str(path.relative_to(WORK))
for witness in result['witnesses']:
    w = witness['strict_interval_witness']
    eta = witness['noise_allowance']
    record = np.asarray(w['common_observations'], dtype=float).reshape(200, 2)
    endpoint_labels = []
    for side in ('base', 'alternative'):
        row = audit(source+':'+witness['id']+':'+side, w[side], record, eta=eta,
                    source=source, role='witness_endpoint', full_oracle=True)
        row['historical_rational_time_endpoint_error_upper'] = w.get(side+'_observation_error_upper')
        row['original_proof_time_convention'] = 'exact rational k/20; audit uses binary64 default times'
        stored = np.asarray(witness['canonical_float_outputs'][side], dtype=float).reshape(200, 2)
        entry = predictions(w[side], default_times())
        row['recomputed_legacy_vs_saved_legacy_max_difference'] = maxabs(entry['legacy']-stored)
        endpoint_labels.append(row['label'])
    pairs.append({'label': source+':'+witness['id'], 'endpoint_labels': endpoint_labels,
                  'eta': eta, 'coordinate_separations': dict(zip(witness['parameter_names'], w['parameter_difference'])),
                  'historical_minimax_coordinate_lower_bound': w['minimax_coordinate_lower_bound']})

by_label = {r['label']: r for r in rows}
for pair in pairs:
    ends = [by_label[x] for x in pair['endpoint_labels']]
    pair['stable_both_inside_eta'] = all(r.get('stable_inside_eta', False) for r in ends)
    pair['legacy_both_inside_eta'] = all(r.get('legacy_inside_eta', False) for r in ends)
    pair['oracle_both_inside_eta'] = all(r.get('oracle_eta_classification') == 'inside' for r in ends)

aggregates = {}
for folder in ('noiseless', 'noise_1e6', 'noise_1e4', 'noise_1e3', 'noise_1e2'):
    group = [r for r in rows if r['role']=='candidate' and r['label'].startswith('combined_validation\\'+folder+'\\')]
    aggregates[folder] = {
        'saved_candidate_count': len(group),
        'all18_below_001_count': sum(r.get('all18_error_below_001',False) for r in group),
        'legacy_inside_eta_count': sum(r.get('legacy_inside_eta',False) for r in group),
        'stable_inside_eta_count': sum(r.get('stable_inside_eta',False) for r in group),
        'legacy_within_numerical_1e8_count': sum(r.get('legacy_within_1e8_numerical_threshold',False) for r in group),
        'stable_within_numerical_1e8_count': sum(r.get('stable_within_1e8_numerical_threshold',False) for r in group),
        'max_native_error': max((r['max_native_error'] for r in group), default=None)}

summary = {
    'records': len(rows), 'distinct_parameter_time_evaluations': len(cache),
    'full_oracle_parameter_time_evaluations': sum('oracle' in v for v in cache.values()),
    'max_legacy_stable_difference': max((r.get('legacy_stable_max_difference') or 0 for r in rows),default=0),
    'max_stable_oracle_error_upper': max((r.get('stable_oracle_error_upper') or 0 for r in rows),default=0),
    'max_legacy_oracle_error_upper': max((r.get('legacy_oracle_error_upper') or 0 for r in rows),default=0),
    'eta_classification_changes': [r['label'] for r in rows if r.get('eta_classification_changed')],
    'numerical_1e8_classification_changes': [r['label'] for r in rows if r.get('numerical_threshold_classification_changed')],
    'noisy_truth_outside_stable_eta': [r['label'] for r in rows if r['role']=='truth_against_actual_record' and r['eta']>0 and not r.get('stable_inside_eta',False)],
    'witness_pair_count': len(pairs),
    'witness_pairs_inside_stable_eta': sum(p['stable_both_inside_eta'] for p in pairs),
    'witness_pairs_inside_oracle_eta': sum(p['oracle_both_inside_eta'] for p in pairs),
    'combined_validation': aggregates,
    'issues': len(issues)}

changed_inputs = []
for relative, metadata in snapshots.items():
    if hashlib.sha256((WORK / relative).read_bytes()).hexdigest() != metadata['sha256']:
        changed_inputs.append(relative)
summary['source_files_changed_during_audit'] = changed_inputs
out = {
    'scope': 'Read-only post-scoring of immutable saved records; no optimization, no data regeneration.',
    'time_convention': 'Exact supplied binary64 timestamps; default core grid for witnesses. Original rational-time witness proof retained separately.',
    'eta_classification': 'Literal max absolute x/y residual <= eta; eta=0 is exact equality and is distinct from the explicit 1e-8 numerical threshold.',
    'oracle_scope': 'All 200 timestamps at every benchmark truth and witness endpoint, plus changed candidate classifications; 70-digit directed intervals, not a formal verification of mpmath.',
    'parameter_scoring': 'All 18 native coordinates in physical order, absolute error <0.001; evaluator-independent.',
    'summary': summary, 'records': rows, 'witness_pairs': pairs,
    'input_snapshots': snapshots, 'issues': issues,
    'evaluator_sha256': {str(p.relative_to(WORK)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (WORK/'stable_model'/'forward.py',WORK/'stable_model'/'oracle.py')},
    'elapsed_seconds': time.perf_counter()-started}
HERE.mkdir(exist_ok=True)
(HERE/'results.json').write_text(json.dumps(out, indent=2, allow_nan=False), encoding='utf-8')
columns = ['label','role','dataset','eta','legacy_max_residual','stable_max_residual',
           'legacy_inside_eta','stable_inside_eta','eta_classification_changed',
           'legacy_stable_max_difference','stable_oracle_error_upper','legacy_oracle_error_upper',
           'oracle_eta_classification','max_native_error','all18_error_below_001',
           'stored_noise_max_amplitude','stored_noise_within_eta']
with (HERE/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
report = [
    'Stable evaluator cross-check of saved full18 evidence',
    'Original data and Dropbox source preserved; this script does not fit anything.',
    'Parameter errors compare saved coordinates directly with truth and are unchanged by the evaluator.',
    'Literal eta=0 compatibility requires exact equality; use the separate 1e-8 column for numerical fit.',
    'Oracle checks supplied binary64 times; historical exact rational-time witness proofs remain distinct.',
    '', json.dumps(summary, indent=2), '', 'Issues: '+json.dumps(issues, indent=2),
    '', 'Witness pairs: '+json.dumps(pairs, indent=2),
    '', 'See comparison.csv and results.json for every success, failure, endpoint, and truth/noise record.'
]
(HERE/'REPORT.txt').write_text('\n'.join(report),encoding='utf-8')
print(json.dumps({'summary': summary, 'elapsed_seconds': out['elapsed_seconds']}), flush=True)
