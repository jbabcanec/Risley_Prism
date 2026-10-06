#!/usr/bin/env python3
"""Analyze the fixed λ-integral enclosure, no new optical evaluation."""
import sys,json,time,math
from pathlib import Path
import numpy as np
import mpmath as mp
mp.iv.dps=45;iv=mp.iv;ROOT=Path(__file__).resolve().parent
raw=json.loads((ROOT/'finite_wedge_native_integral_envelope_arrays.json').read_text())
def lo(v):return math.nextafter(float(v.a),-math.inf)
def hi(v):return math.nextafter(float(v.b),math.inf)
def zero(shape):return np.full(shape,iv.mpf(0),dtype=object)
def arr(x):
    v=np.array(x,dtype=float);out=np.empty(v.shape[:-1],object)
    for idx in np.ndindex(out.shape):out[idx]=iv.mpf(v[idx].tolist())
    return out
Js=arr(raw['Js']);Jw=arr(raw['Jw']);D=arr(raw['Dlag']);R=zero((7,6))
for j in range(1,7):
    den=iv.sqrt(j*(j+1))
    for k in range(j):R[k,j-1]=1/den
    R[j,j-1]=-iv.mpf(j)/den
roots=[iv.mpc(1,0)]
for num in [1,7,49]:
    a=2*iv.pi*num*11/400;z=iv.mpc(iv.cos(a),iv.sin(a));roots += [z,iv.mpc(z.real,-z.imag)]
p=[iv.mpc(1,0)]
for z in roots:
    out=[iv.mpc(0,0)]*(len(p)+1)
    for j,v in enumerate(p):out[j]-=z*v;out[j+1]+=v
    p=out
p=np.array([z.real for z in p],object)
AD=zero((6,6,18));fd=zero((6,18))
for q in range(18):
    H=np.array([[Js[a,k+11*j,q] for j in range(7)] for a in range(2) for k in range(123)],object)
    rr=np.array([sum(p[j]*Js[a,k+11*j,q] for j in range(8)) for a in range(2) for k in range(123)],object)
    AD[:,:,q]=D@H@R;fd[:,q]=D@rr
M=np.array([[sum(abs(z) for z in AD[i,j]) for j in range(6)] for i in range(6)],object)
norm=max(hi(sum(row)) for row in M);fr=np.array([sum(abs(z) for z in row) for row in fd],object)
report={'native_radius':.001,'integration_cells':8,'first_record_radius_max':max(hi(sum(abs(z) for z in row)) for row in Js.reshape(400,18)),'second_record_radius_max':max(hi(sum(abs(z) for z in row)) for row in Jw.reshape(400,18)),'lag11_uniform_neumann_bound':norm,'lag11_uniform_forcing_radius':[hi(z) for z in fr],'lag11_matrix_radius_upper':[[hi(z) for z in row] for row in M],'lag11_neumann_pass':norm<1}
# A single deterministic positive-weight linear proposal is permitted for the
# same 6x6 enclosure; it changes no physical box or chord norm.
Mu=np.array([[hi(z) for z in row] for row in M]);rho_proposal=max(abs(np.linalg.eigvals(Mu)))
report['matrix_radius_spectral_radius_proposal']=float(rho_proposal)
w=np.linalg.solve(np.eye(6)-Mu,np.ones(6));report['M_matrix_positive_weight_proposal']=w.tolist()
weighted=False
if np.all(w>0):
    wi=np.array([iv.mpf(float(v)) for v in w],object)
    q=max(hi(sum(M[i,j]*wi[j] for j in range(6))/wi[i]) for i in range(6))
    report['weighted_neumann_defect_upper']=q;weighted=q<1
    if weighted:
        alpha=max(hi(fr[i]/wi[i]) for i in range(6))/(1-q)
        arad=np.array([iv.mpf(math.nextafter(alpha,math.inf))*v for v in wi],object)
        crad=np.array([sum(abs(R[i,j])*arad[j] for j in range(6)) for i in range(7)],object)
        report['a_coefficient_radius_componentwise_upper']=[hi(v) for v in arad]
        report['c_polynomial_radius_componentwise_upper']=[hi(v) for v in crad]
report['lag11_weighted_neumann_pass']=weighted
report['lag11_any_neumann_pass']=norm<1 or weighted
report['status']='lag11 certified; subsequent rotor, fit, and A2 tests still required' if report['lag11_any_neumann_pass'] else 'UNRESOLVED: even λ-integral centered derivative enclosure does not certify frozen-lag11 invertibility on full native box'
print(json.dumps(report,indent=2));(ROOT/'finite_wedge_native_integral_spectral.json').write_text(json.dumps(report,indent=2))
