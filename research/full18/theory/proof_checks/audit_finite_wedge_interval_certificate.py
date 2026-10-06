#!/usr/bin/env python3
"""Independent verification of exported same-point derivative enclosures.
Accepts only the declared one-point 18x18 enclosure. Checks a trace obstruction
and rebuilds an exact-similarity Gershgorin certificate at 70 interval digits.
This does not generate a new physical case or choose an iteration parameter.
"""
from pathlib import Path
from fractions import Fraction
import json,math
import numpy as np
import mpmath as mp
mp.iv.dps=70
iv=mp.iv
root=Path(__file__).resolve().parent
p=json.loads((root/'finite_wedge_corrector_interval.json').read_text())
assert p['interval'] is True and p['kappa']==0.1
lo,hi=p['DT_interval_lower'],p['DT_interval_upper']
assert len(lo)==len(hi)==18 and all(len(row)==18 for row in lo+hi)
assert all(lo[i][j]<=hi[i][j] for i in range(18) for j in range(18))
# Fraction.from_float retains each exported binary endpoint exactly.
tl=sum(Fraction(lo[i][i]) for i in range(18));tu=sum(Fraction(hi[i][i]) for i in range(18))
assert tu < -18
trace_rho_lower=math.nextafter(float(-tu/18),-math.inf)

def mat(a):return np.array([[iv.mpf(x) for x in row] for row in a],dtype=object)
def upper(x):return math.nextafter(float(x.b),math.inf)
def infnorm(a):return max(upper(sum(abs(x) for x in row)) for row in a)
T=np.array([[iv.mpf([lo[i][j],hi[i][j]]) for j in range(18)] for i in range(18)],dtype=object)
S,R=mat(p['basis']),mat(p['inverse_basis']);Id=mat(np.eye(18))
E=Id-R@S;delta=infnorm(E);assert delta<1e-8
A=R@T@S
# The exact S^{-1} T S = (I-E)^{-1} A.
tail=upper(iv.mpf(delta)/(1-iv.mpf(delta))*infnorm(A))
eps=iv.mpf([-tail,tail]);B=np.array([[x+eps for x in row] for row in A],dtype=object)
discs=[]
for i in range(18):
    center=float(B[i,i].mid)
    radius=upper(abs(B[i,i]-iv.mpf(center))+sum(abs(B[i,j]) for j in range(18) if j!=i))
    discs.append((center,radius))
unstable=[]
for i,(center,radius) in enumerate(discs):
    cc,rr=Fraction(center),Fraction(radius)
    # Every comparison here is rational exact, not a rounded floating predicate.
    isolated=all(abs(cc-Fraction(c2))>rr+Fraction(r2) for j,(c2,r2) in enumerate(discs) if i!=j)
    if isolated and abs(cc)-rr>1:
        unstable.append({'disc':i,'real_eigenvalue_interval':[math.nextafter(float(cc-rr),-math.inf),math.nextafter(float(cc+rr),math.inf)]})
assert len(unstable)>=2
real_upper_exact=max(Fraction(center)+Fraction(radius) for center,radius in discs)
assert real_upper_exact<1
realupper=math.nextafter(float(real_upper_exact),math.inf)
out={'scope':'independent audit of same-point exported enclosure, not a finite-radius certificate','trace_from_entry_endpoints':[math.nextafter(float(tl),-math.inf),math.nextafter(float(tu),math.inf)],'spectral_radius_lower_from_trace':trace_rho_lower,'inverse_basis_defect_upper':delta,'similarity_entry_tail_upper':tail,'discs':discs,'isolated_real_unstable_eigenvalues':unstable,'all_eigenvalue_real_parts_upper':realupper}
(root/'finite_wedge_interval_certificate_independent_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
print('PASS: exact-rational trace obstruction and independently reconstructed interval Gershgorin certificate.')
