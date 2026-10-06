"""Conditional 4D bounded-error geometry polytope and verified pair witnesses.

F(q,l)=b(q)+A(q)l, l=(d_W,gap,px,py). This is conditional on a fixed q;
it is NOT full18 confinement. Nonlinear truth is used only to construct
indistinguishable full18 endpoint examples, never to initialize recovery.
LP extrema are floating proposals. Only independently enclosed endpoints
are used for finite-pair lower bounds about the actual observation record.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from contract import ROOT,WORK
from risley_lattice.model import LO,HI,NAMES,vec2pat
from risley_lattice.separable import affine_geometry,LINEAR
from risley_lattice.fmodel import forward_point

def one_case(case, eta):
    theta=np.array(case['truth']); y=np.array(case['observed']).ravel()
    b,A=affine_geometry(theta); center=theta[LINEAR]
    row_l1=np.sum(np.abs(np.linalg.pinv(A)),axis=1)
    scale=np.maximum(row_l1*eta,1e-14)
    residual=(b+A@center-y)/eta
    matrix=A*scale[None,:]/eta
    # 10 percent interior budget leaves room for LP and enclosure arithmetic.
    aub=np.vstack([matrix,-matrix]); bub=np.r_[.9-residual,.9+residual]
    bounds=list(zip((LO[LINEAR]-center)/scale,(HI[LINEAR]-center)/scale))
    out={'id':case['id'],'eta':eta,'affine_rank':int(np.linalg.matrix_rank(A)),
         'affine_condition_number':float(np.linalg.cond(A)),'coordinate_pairs':[],
         'scope':'fixed nonlinear coordinates; full18 endpoint lower witnesses only'}
    for j in range(4):
        ends=[]
        for sign in (1.,-1.):
            c=np.zeros(4);c[j]=sign
            result=linprog(c,A_ub=aub,b_ub=bub,bounds=bounds,method='highs',
                options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            if not result.success:
                ends.append({'success':False,'message':result.message});continue
            v=theta.copy();v[LINEAR]=center+scale*result.x
            try:
                fm,fr,m=forward_point(v)
                exact_bound=float(np.max(np.abs(fm-y)+fr))
                ends.append({'success':True,'theta':v.tolist(),'interval_actual_record_bound':exact_bound,
                    'compatible_verified_under_ivx_arithmetic_contract':bool(exact_bound<=eta),
                    'margins':m,'canonical_actual_record_max':float(np.max(np.abs(vec2pat(v).ravel()-y)))})
            except Exception as exc:ends.append({'success':False,'message':repr(exc)})
        row={'coordinate':NAMES[LINEAR[j]],'endpoints':ends}
        if all(e['success'] for e in ends):
            delta=np.abs(np.array(ends[1]['theta'])-np.array(ends[0]['theta']))
            verified=all(e['compatible_verified_under_ivx_arithmetic_contract'] for e in ends)
            row.update({'diameter_native':float(delta[LINEAR[j]]),'minimax_native_floor':float(delta[LINEAR[j]]/2),
                'all18_difference':delta.tolist(),'both_fit_actual_record_verified':verified,
                'rules_out_uniform_001_for_actual_record':bool(verified and delta[LINEAR[j]]>.002)})
        out['coordinate_pairs'].append(row)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='*',default=['random_00','random_06','moderate_unsorted','wide_unsorted','weak_first']);p.add_argument('--etas',nargs='*',type=float,default=[1e-8,1e-6,1e-4,1e-3,.01,.1]);a=p.parse_args()
    dataset=json.loads((WORK/'cases.json').read_text());rows=[]
    for case in dataset['cases']:
        if case['id'] not in a.cases:continue
        for eta in a.etas:
            row=one_case(case,eta);rows.append(row)
            print(json.dumps({'id':row['id'],'eta':eta,'cond':row['affine_condition_number'],'pairs':[{'coordinate':r['coordinate'],'diameter':r.get('diameter_native'),'verified':r.get('both_fit_actual_record_verified')} for r in row['coordinate_pairs']]}),flush=True)
            (WORK/'geometry_polytope_results.json').write_text(json.dumps({'rows':rows,'notes':'LP extrema are numerical proposals; endpoint fmodel interval bounds validate lower witnesses under its documented floating-arithmetic assumptions. No outer nonlinear exclusion or LP upper certification.'},indent=2))

if __name__=='__main__':main()
