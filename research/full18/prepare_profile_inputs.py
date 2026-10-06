"""Package observed records only, with no hidden truth in solver inputs."""
from pathlib import Path
import json
import numpy as np
WORK=Path(__file__).resolve().parent
out=WORK/'profiles';out.mkdir(exist_ok=True)
data=json.loads((WORK/'observations.json').read_text())
row=next(c for c in data['cases'] if c['id']=='random_00')
k=np.arange(200)[:,None];axis=np.arange(2)[None,:]
y=np.asarray(row['observed'])+1e-4*np.sin((k+1)*(np.sqrt(2)+axis*np.sqrt(3)))
r=json.loads((WORK/'solver/noisy_v1_1e4/random_00.json').read_text());r['actual_observations']=y.tolist()
(out/'candidate_random00_eta1e4.json').write_text(json.dumps(r,indent=2))
w=json.loads((WORK/'ambiguity/witnesses.json').read_text())['witnesses']
records=[{'id':r['id'],'observed':np.array(r['strict_interval_witness']['common_observations']).reshape(200,2).tolist()} for r in w]
(out/'adverse_observations_only.json').write_text(json.dumps({'cases':records},indent=2))
print('Packaged actual noisy observation and two constructed shared records; no truth fields.')
