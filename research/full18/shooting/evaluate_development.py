"""Post-selection scoring of four development cases, separate from solver."""
import json,csv
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
contract=json.loads((HERE.parent/'cases.json').read_text())
names=contract['names'];rows=[]
for case_id in ['moderate_unsorted','random_03','exact_collision','weak_first']:
    truth=np.asarray(next(c['truth'] for c in contract['cases'] if c['id']==case_id))
    for mode in ['blocks','constrained']:
        path=HERE/(case_id+'_'+mode+'.json')
        result=json.loads(path.read_text());best=result['best']
        row={'id':case_id,'method':mode,'status':result['status'],'seconds':result['seconds']}
        if best is not None:
            errors=abs(np.asarray(best['theta'])-truth)
            row.update(theta=best['theta'],native_errors=dict(zip(names,errors.tolist())),
                       max_native_error=float(max(errors)),passes_001=bool(max(errors)<.001),
                       mse=best['mse'],max_residual=best['max_residual'],physical=best['physical'])
        rows.append(row)
report={'rows':rows,'truth_used_only_for_post_selection_scoring':True,
        'combined_holdout_truth_not_read':True,'global_accuracy_certificate':False}
(HERE/'development_evaluation.json').write_text(json.dumps(report,indent=2))
with (HERE/'development_coordinate_errors.csv').open('w',newline='') as h:
    w=csv.writer(h);w.writerow(['id','method','status','seconds','mse','max_native_error','passes_001']+names)
    for row in rows:w.writerow([row.get(k) for k in ['id','method','status','seconds','mse','max_native_error','passes_001']]+[row.get('native_errors',{}).get(n) for n in names])
print(json.dumps([{k:r.get(k) for k in ['id','method','status','seconds','mse','max_native_error','passes_001']} for r in rows],indent=2))
