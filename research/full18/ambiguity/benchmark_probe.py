"""Test nonzero incidence and unequal glass near the low-noise geometry pair."""
import json,time
from pathlib import Path
import numpy as np
from profile_pairs import profile,NAMES

records=[];start=time.time()
for wedge,angle in ((3.,.05),(3.,.1),(2.5,.1),(2.5,.2)):
    base=np.array([2.71,-1.93,.83,wedge,-1.15*wedge,.9*wedge,7.,-11.,17.,1.31,1.32,1.33,190.,3.,angle,-angle,1.2,-2.1])
    rec=profile(base,12,.0022);rec.update(family='separated_speeds_unequal_glass_nonzero_incidence',wedge_scale=wedge,incidence=angle)
    records.append(rec)
    print(json.dumps({'wedge':wedge,'incidence':angle,'eta':rec['numerical_eta_midpoint']}),flush=True)
    Path('benchmark_candidates.json').write_text(json.dumps({'names':NAMES,'records':records,'elapsed_seconds':time.time()-start},indent=2))
