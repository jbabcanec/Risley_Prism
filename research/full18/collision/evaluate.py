"""Truth is used here only after solver output files have been written."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
import numpy as np
work=Path(__file__).resolve().parent
contract=json.loads((work.parent/'cases.json').read_text())
truth=np.array(next(c['truth'] for c in contract['cases'] if c['id']=='exact_collision'))
names=contract['names'];results=[]
for name in ('baseline_relations.json','result.json','refined.json','noise_1e-6.json','noise_1e-4.json','noise_1e-3.json'):
    path=work/name
    if not path.exists():continue
    result=json.loads(path.read_text());x=np.array(result['x18'])
    result['file']=name;result['max_native_error']=float(np.max(abs(x-truth)))
    result['all18_error_under_001']=bool(np.all(abs(x-truth)<.001))
    result['per_coordinate']=[dict(name=n,truth=float(t),estimate=float(e),absolute_error=float(abs(e-t))) for n,t,e in zip(names,truth,x)]
    results.append(result)
report={'source_sha256':{name:hashlib.sha256((work/name).read_bytes()).hexdigest() for name in ('recover.py','refine.py','baseline_relations.py')},'evaluations':results}
(work/'evaluation.json').write_text(json.dumps(report,indent=2))
for r in results:print(json.dumps({k:r.get(k) for k in ('file','seconds','mse','max_residual','max_native_error','all18_error_under_001')}))
