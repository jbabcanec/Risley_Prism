"""Additional separated-speed geometry witnesses with small source angles."""
import json,time
from pathlib import Path
import numpy as np
from profile_pairs import profile,jacobian,NAMES

start=time.time();rows=[]
for wedge in (2.,3.,4.,6.,8.):
    for index in (1.31,1.55,1.79):
        base=np.array([2.71,-1.93,.83,wedge,-1.15*wedge,.9*wedge,7.,-11.,17.,index,index,index,190.,3.,0.,0.,0.,0.])
        J=jacobian(base);free=[i for i in range(18) if i!=12]
        direction=np.linalg.lstsq(J[:,free],-J[:,12],rcond=None)[0]
        gain=np.max(np.abs(J[:,12]+J[:,free]@direction))
        rows.append((float(gain),wedge,index,base))
rows.sort(key=lambda x:x[0]);records=[]
for gain,wedge,index,base in rows[:8]:
    rec=profile(base,12,.0022);rec.update(family='separated_speeds_small_incidence',wedge_scale=wedge,glass=index,linear_gain_heuristic=gain)
    records.append(rec)
    print(json.dumps({'wedge':wedge,'glass':index,'eta':rec['numerical_eta_midpoint']}),flush=True)
    Path('geometry_candidates.json').write_text(json.dumps({'names':NAMES,'records':records,'elapsed_seconds':time.time()-start},indent=2))
