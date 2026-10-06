"""Post-selection full18 scoring for the entire frozen ten-case noise sweep.

This script has truths; the solver runners do not. It performs no optimization.
"""
import json,csv,hashlib
from pathlib import Path
import numpy as np
from verify_results import evaluate
from contract import WORK,ROOT

def main():
    data=json.loads((WORK/'combined_holdout.json').read_text())
    cases={r['id']:r for r in data['cases']}
    levels=[('noiseless',0.),('noise_1e6',1e-6),('noise_1e4',1e-4),('noise_1e3',1e-3),('noise_1e2',1e-2)]
    allrows=[];groups=[]
    for label,eta in levels:
        rows=[]
        for name,case in cases.items():
            file=WORK/'combined_validation'/label/(name+'.json')
            if not file.exists():
                rows.append({'id':name,'missing':True,'passes_001':False});continue
            r=json.loads(file.read_text())
            if r.get('theta') is None:
                rows.append({'id':name,'no_candidate':True,'passes_001':False,'result_file':str(file)});continue
            score=evaluate(r['theta'],case,r.get('actual_observations'))
            score.update(noise_amplitude=eta,result_file=str(file),status=r.get('status'),seconds=r.get('seconds'),cohort='fresh_combined_seed20261005')
            score['hard_band_compatible_under_ivx']=None if eta==0 else score.get('exact_model_vs_observed_interval_bound',float('inf'))<=eta
            score['canonical_hard_band_compatible']=None if eta==0 else score['canonical_max_residual']<=eta
            rows.append(score);allrows.append(score)
        complete=all(not r.get('missing') for r in rows)
        groups.append({'directory':label,'eta':eta,'total':len(cases),'complete':complete,
                       'within_001':sum(r.get('passes_001',False) for r in rows),
                       'strict_physical':sum(r.get('strict_physical',False) for r in rows),
                       'hard_band_compatible':None if eta==0 else sum(r.get('hard_band_compatible_under_ivx',False) for r in rows),
                       'rows':rows})
    report={'cohort':'fresh_combined_seed20261005','all18_unknown':True,'original_prior_unchanged':True,
            'original_recorded_source_hashes_unchanged':all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==h for k,h in data['source_hashes'].items()),
            'noise_pattern':'e[k,a]=eta*sin((k+1)*(sqrt(2)+a*sqrt(3))); one deterministic bounded pattern, not worst-case coverage',
            'native_target':.001,'solver_policy':'frozen pencil/all-orders then FFT fallback; excludes subsequent adaptive repairs',
            'groups':groups}
    (WORK/'ACCURACY_SUMMARY.json').write_text(json.dumps(report,indent=2))
    fields=['cohort','id','noise_amplitude','passes_001','max_native_error','worst_coordinate','canonical_max_residual',
            'exact_model_vs_observed_interval_bound','strict_physical','hard_band_compatible_under_ivx','seconds']+['error_'+n for n in data['names']]
    with (WORK/'FULL18_ACCURACY.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in allrows:
            line={k:r.get(k) for k in fields};line.update({'error_'+k:v for k,v in r['native_error'].items()});writer.writerow(line)
    table=['# Frozen full18 accuracy sweep','',
           'All ten cases remain in every denominator. All errors compare physical prism positions without relabeling. Noise uses one fixed bounded pattern per level; this is empirical recovery evidence, not a uniform guarantee.','',
           '| Observation allowance | All18 errors <= .001 | Strict-model compatible fits | Complete |',
           '|---|---:|---:|---|']
    for g in groups:table.append(f"| {g['eta']:g} | {g['within_001']}/{g['total']} | {'numerical noiseless test' if g['eta']==0 else str(g['hard_band_compatible'])+'/'+str(g['total'])} | {g['complete']} |")
    table+=['','| Case | Noiseless maximum error | eta 1e-6 | eta 1e-4 | eta 1e-3 | eta 1e-2 |','|---|---:|---:|---:|---:|---:|']
    for name in cases:
        values=[]
        for g in groups:
            r=next(r for r in g['rows'] if r['id']==name)
            values.append('missing' if 'max_native_error' not in r else f"{r['max_native_error']:.8g} ({r['worst_coordinate']})")
        table.append('| '+name+' | '+' | '.join(values)+' |')
    table+=['','Coordinate units: Hz for N; degrees for ax, ay and beam angles; dimensionless glass indices; native unspecified position units for distances, gap and source positions.',
            'Noiseless means no added noise on binary64 legacy-generated samples; it does not assert exact strict-model feasibility at eta=0.',
            'Full coordinate errors, residuals, physical checks and candidate provenance are in FULL18_ACCURACY.csv and ACCURACY_SUMMARY.json.']
    (WORK/'ACCURACY_SUMMARY.md').write_text('\n'.join(table)+'\n')
    print(json.dumps([{k:v for k,v in g.items() if k!='rows'} for g in groups],indent=2))

if __name__=='__main__':main()
