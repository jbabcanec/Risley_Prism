"""Score adaptive repairs separately from the frozen fifty-run experiment.

Candidate selection is by full-record residual and strict physical feasibility,
never by truth. The per-coordinate scoring below is post-selection only.
"""
import json
import numpy as np
from pathlib import Path
from verify_results import evaluate
from contract import WORK
from risley_lattice.model import vec2pat,LO,HI
from risley_lattice.fmodel import forward_point

def main():
    dataset=json.loads((WORK/'combined_holdout.json').read_text());groups=[]
    for label,eta in [('noiseless',0.),('noise_1e4',1e-4)]:
        rows=[]
        for case in dataset['cases']:
            ident=case['id'];original=WORK/'combined_validation'/label/(ident+'.json')
            saved=json.loads(original.read_text());y=np.array(saved['actual_observations'])
            folder=WORK/'low_frequency' if eta==0 else WORK/'low_frequency'/label
            files=[original]+sorted(folder.glob(ident+'*.json'));candidates=[]
            for path in files:
                result=json.loads(path.read_text())
                if not isinstance(result,dict) or result.get('theta') is None:continue
                v=np.asarray(result['theta'])
                if not np.all(v>=LO) or not np.all(v<=HI):continue
                try:forward_point(v)
                except Exception:continue
                residual=float(np.max(abs(vec2pat(v)-y)))
                candidates.append((residual,path,v))
            residual,path,v=min(candidates,key=lambda r:r[0])
            score=evaluate(v,case,y);score.update(selected_result=str(path),noise_amplitude=eta,
                policy_status='adapted after inspecting failures; not a pristine frozen holdout',
                examined_result_files=[str(p) for p in files])
            score['hard_band_compatible_under_ivx']=None if eta==0 else score.get('exact_model_vs_observed_interval_bound',float('inf'))<=eta
            rows.append(score)
        groups.append({'eta':eta,'total':10,'all18_within_001':sum(r['passes_001'] for r in rows),
            'hard_band_compatible':None if eta==0 else sum(r['hard_band_compatible_under_ivx'] for r in rows),'rows':rows})
    report={'cohort':'fresh_combined_seed20261005','selection_uses_truth':False,'denominators_retained':True,'groups':groups}
    (WORK/'ADAPTED_RESULTS.json').write_text(json.dumps(report,indent=2))
    print(json.dumps([{**{k:v for k,v in g.items() if k!='rows'},'repaired_rows':[{k:r[k] for k in ('id','max_native_error','worst_coordinate','canonical_max_residual','hard_band_compatible_under_ivx','selected_result')} for r in g['rows'] if r['id'] in ('random_02','random_03','random_07')]} for g in groups],indent=2))

if __name__=='__main__':main()
