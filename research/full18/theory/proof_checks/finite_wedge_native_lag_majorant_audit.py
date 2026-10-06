#!/usr/bin/env python3
"""Certify the spectral radius of the exact dyadic NONNEGATIVE MAJORANT only.
This does not enclose eigenvalues of any actual lag/optical/chord derivative.
"""
import json,math,numpy as np,mpmath as mp
from pathlib import Path
mp.iv.dps=60;iv=mp.iv;root=Path(__file__).resolve().parent
src=json.loads((root/'finite_wedge_native_integral_spectral.json').read_text());M=np.array(src['lag11_matrix_radius_upper']);ev,V=np.linalg.eig(M);j=np.argmax(ev.real);v=abs(V[:,j].real);v/=min(v)
Mv=[]
for i in range(6):Mv.append(sum(iv.mpf(float(M[i,k]))*iv.mpf(float(v[k])) for k in range(6))/iv.mpf(float(v[i])))
lo=math.nextafter(min(float(z.a) for z in Mv),-math.inf);hi=math.nextafter(max(float(z.b) for z in Mv),math.inf)
assert lo>1
out={'scope':'nonnegative entrywise-radius MAJORANT M only; no physical singularity or instability inference','positive_vector_binary':v.tolist(),'collatz_wielandt_spectral_radius_interval':[lo,hi],'no_positive_weight_can_make_this_exact_M_contractive':True}
(root/'finite_wedge_native_lag_majorant_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
