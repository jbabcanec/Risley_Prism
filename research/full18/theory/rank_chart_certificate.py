"""Certify an 18-observation local chart at a saved blind full18 candidate.

No fit, no truth read, no prior restriction. The source Jacobian enclosure
retains its explicit ivx arithmetic assumptions. Final inverse products use
independent mpmath directed interval arithmetic with exact binary64 inputs.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from scipy.linalg import qr
from mpmath import iv

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
REPO=Path(r'C:\Users\josep\Dropbox\Babcanec Works\Mathematics\Wedge')
sys.path[:0]=[str(REPO),str(WORK/'stable_model')]
from risley_lattice.fmodel import forward_iv
from risley_lattice.model import LO,HI,RG,NAMES
from oracle import exact,abs_upper,bounds


def directed_inverse_defect(C,Jm,Jr):
    """Return outward E >= |C * [J] * diag(RG) - I| and row norm."""
    iv.dps=70
    entries=[[exact(Jm[k,j])*exact(RG[j])+
              iv.mpf([-1,1])*exact(Jr[k,j])*exact(RG[j])
              for j in range(18)] for k in range(18)]
    Cq=[[exact(x) for x in row] for row in C]
    E=[]
    for i in range(18):
        row=[]
        for j in range(18):
            value=sum((Cq[i][k]*entries[k][j] for k in range(18)),iv.mpf(0))
            value-=int(i==j)
            row.append(abs_upper(value))
        E.append(row)
    q=max(bounds(sum((exact(x) for x in row),iv.mpf(0)))[1] for row in E)
    c_norm=max(bounds(sum((abs(exact(x)) for x in row),iv.mpf(0)))[1] for row in C)
    gain=None
    if q<1:
        gain=bounds(exact(c_norm)/(1-exact(q)))[1]
    return dict(defect_magnitude_upper=E,contraction_q_upper=q,
                scaled_inverse_linf_norm_upper=c_norm,
                scaled_local_inverse_lipschitz_upper=gain)


def main():
    started=time.perf_counter()
    path=WORK/'combined_validation'/'noiseless'/'random_00.json'
    raw=path.read_bytes()
    saved=json.loads(raw)
    theta=np.asarray(saved['theta'],float)
    assert theta.shape==(18,) and np.all(theta>LO) and np.all(theta<HI)
    _,_,Jm,Jr,point_margins=forward_iv(theta,np.zeros(18))
    # Numerical selection supplies a proposed chart; only directed checking
    # below justifies it. No parameter, observation, or physical ordering changes.
    _,_,pivots=qr((Jm*RG[None,:]).T,mode='economic',pivoting=True)
    selected=np.asarray(pivots[:18],dtype=int)
    C=np.linalg.inv(Jm[selected]*RG[None,:])
    checks=[]
    for scale in (0.,1e-10,1e-9,1e-8,1e-7):
        radii=RG*scale
        try:
            fm,fr,jm,jr,margins=forward_iv(theta,radii)
            proof=directed_inverse_defect(C,jm[selected],jr[selected])
            row=dict(scaled_radius=scale,native_radii=radii.tolist(),
                     inside_native_box=bool(np.all(theta-radii>=LO) and np.all(theta+radii<=HI)),
                     physical_margins=margins,
                     selected_jacobian_midpoint=jm[selected].tolist(),
                     selected_jacobian_radius=jr[selected].tolist(),**proof)
            row['local_chart_certified_under_ivx_contract']=bool(row['inside_native_box'] and proof['contraction_q_upper']<1)
        except Exception as exc:
            row=dict(scaled_radius=scale,error=repr(exc),local_chart_certified_under_ivx_contract=False)
        checks.append(row)
        print(json.dumps({k:row[k] for k in ('scaled_radius','contraction_q_upper','local_chart_certified_under_ivx_contract','error') if k in row}),flush=True)
    output=dict(
        statement='An explicit nonsingular 18-row observation minor, and any reported small-box inverse chart. Not global uniqueness or a global error certificate.',
        source_candidate=str(path.relative_to(WORK)),candidate_sha256=hashlib.sha256(raw).hexdigest(),
        candidate_was_previously_selected_without_truth=True,truth_read=False,optimization_performed=False,
        native_parameter_order=list(NAMES),theta=theta.tolist(),native_scaling=RG.tolist(),
        times=np.arange(0,10.,.05)[:200].tolist(),
        selected_flat_output_rows=selected.tolist(),
        selected_observations=[dict(sample=int(k//2),axis='xy'[k%2],time=float(np.arange(0,10.,.05)[k//2])) for k in selected],
        output_order='x0,y0,x1,y1,...,x199,y199',
        approximate_inverse_of_scaled_minor=C.tolist(),
        interpretation='C is treated as an exact binary64 matrix. q<1 certifies this minor is nonsingular; a certified convex box also gives injectivity of the selected chart and the entire record within that box.',
        arithmetic='Jacobian enclosure: original ivx/fmodel under A-fp, A-libm, A-blas. Final C*[J]*diag(RG)-I and row sums: 70-digit mpmath.iv directed intervals. Neither software stack is formally verified.',
        genericity='The nonzero analytic minor implies rank18 on an open dense, full-measure subset only of the connected analytic strict-physical component containing this point. No other component or global uniqueness follows.',
        source_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (REPO/'risley_lattice'/'fmodel.py',REPO/'risley_lattice'/'ivx.py',REPO/'risley_lattice'/'model.py')},
        checks=checks,elapsed_seconds=time.perf_counter()-started)
    (HERE/'rank_chart_certificate.json').write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'output':str(HERE/'rank_chart_certificate.json'),'elapsed_seconds':output['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    main()
