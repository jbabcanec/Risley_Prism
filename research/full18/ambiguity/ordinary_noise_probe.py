"""Non-small-wedge geometry tradeoff at a looser position-error allowance."""
import json
from pathlib import Path
import numpy as np
from profile_pairs import profile,NAMES

base=np.array([2.71,-1.93,.83,12.,-13.8,10.8,7.,-11.,17.,1.46,1.57,1.67,137.,7.,2.,-3.,1.2,-2.1])
record=profile(base,12,.05)
record.update(family='ordinary_distinct_speeds_large_wedges',wedge_scale=12.)
Path('ordinary_noise_candidate.json').write_text(json.dumps({'names':NAMES,'records':[record]},indent=2))
print(json.dumps({'eta':record['numerical_eta_midpoint'],'dw_error_floor':record['half_coordinate_separation'][12]}))
