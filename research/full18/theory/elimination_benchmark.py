"""Conditional exact-formula 2D geometry elimination, numerical benchmark.

All q inputs are observation-derived candidates. Full native affine bounds are
retained. Floating polygon/LP results are diagnostics, not outward certificates.
No repository code or hidden-truth files are imported.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

HERE=Path(__file__).resolve().parent;WORK=HERE.parent
LINEAR=np.array([12,13,16,17]);LOW=np.array([50.,2.,-5.,-5.]);HIGH=np.array([200.,15.,5.,5.])

def affine(theta):
    """Closed three-prism pair map; b,A preserve physical order and both axes."""
    v=np.asarray(theta);t=np.arange(200)/20
    gamma=2*np.pi*v[:3,None]*t+v[6:9,None]*np.pi/180
    ss=np.sin(v[3:6,None]*np.pi/180)
    b=np.empty((200,2));A=np.zeros((200,2,4));minimum=np.inf
    for axis,rotor in enumerate((np.cos(gamma),np.sin(gamma))):
        source_t=np.tan(v[14+axis]*np.pi/180)
        tangent=np.full(200,source_t);mag=np.ones(200);constant=np.zeros(200);gap=np.zeros(200)
        for i in range(3):
            incoming=tangent/np.sqrt(1+tangent*tangent)
            H=np.sqrt(v[9+i]**2-incoming**2)
            s=ss[i]*rotor[i];c=np.sqrt(1-s*s)
            Q=incoming*c+H*s;rad=1-Q*Q
            if np.any(rad<=0):raise ValueError('TIR')
            R=np.sqrt(rad);Z=R*c+Q*s;P=c*H-s*incoming
            minimum=min(minimum,float(np.min(rad)),float(np.min(Z)),float(np.min(P)))
            if minimum<=0:raise ValueError('physical guard')
            tangent=(Q*c-R*s)/Z
            L=H*R/(Z*P);K=3*incoming*R/(Z*P)
            mag=L*mag;constant=L*constant+K;gap=L*gap
            if i<2:gap+=tangent
        b[:,axis]=6*source_t*mag+constant
        A[:,axis,0]=tangent;A[:,axis,1]=gap;A[:,axis,2+axis]=mag
    return b.ravel(),A.reshape(400,4),minimum

def clip_polygon(poly,row,bound):
    if not len(poly):return poly
    vals=poly@row-bound
    inside=vals<=0
    if inside.all():return poly
    if not inside.any():return np.empty((0,2))
    out=[]
    for j in range(len(poly)):
        k=(j+1)%len(poly);p,q=poly[j],poly[k];fp,fq=vals[j],vals[k]
        if inside[j]:out.append(p)
        if inside[j]!=inside[k]:out.append(p+(fp/(fp-fq))*(q-p))
    return np.array(out)

def polygon_elimination(b,A,y,eta,center):
    # Scaling is only numerical coordinates, not a smaller prior.
    scale=np.maximum(np.sum(abs(np.linalg.pinv(A)),axis=1)*eta,1e-10)
    M=A*scale;residual=y-b-A@center
    lo=(LOW-center)/scale;hi=(HIGH-center)/scale
    rows=[];rhs=[]
    for axis in range(2):
        ii=np.arange(axis,400,2);h=M[ii,2+axis]
        assert np.all(h>0)
        coeff=-M[ii,:2]/h[:,None]
        lower=(residual[ii]-eta)/h;upper=(residual[ii]+eta)/h
        # Virtual source-box lower and upper interval endpoints.
        lc=np.vstack([coeff,np.zeros((1,2))]);uc=lc
        lv=np.r_[lower,lo[2+axis]];uv=np.r_[upper,hi[2+axis]]
        rows.append((lc[:,None,:]-uc[None,:,:]).reshape(-1,2))
        rhs.append((uv[None,:]-lv[:,None]).ravel())
    G=np.vstack(rows);h=np.concatenate(rhs)
    # Put strong central constraints first to avoid repeated huge-coordinate
    # clipping. This permutation changes no halfplane or original bound.
    rownorm=np.linalg.norm(G,axis=1);zero=rownorm==0
    if np.any(h[zero]<0):return dict(feasible=False,halfplanes=len(h),scale=scale.tolist())
    G,h=G[~zero],h[~zero]
    order=np.argsort(h/np.maximum(np.linalg.norm(G,axis=1),1e-300))
    poly=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
    for k in order:
        poly=clip_polygon(poly,G[k],h[k])
        if not len(poly):break
    if not len(poly):return dict(feasible=False,halfplanes=len(h)+4,scale=scale.tolist())
    mid=np.mean(poly,axis=0);linear=np.zeros(4);linear[:2]=mid
    intervals=[]
    for axis in range(2):
        ii=np.arange(axis,400,2);hsrc=M[ii,2+axis]
        z=(residual[ii]-M[ii,:2]@mid)/hsrc;e=eta/hsrc
        lower=max(lo[2+axis],float(np.max(z-e)));upper=min(hi[2+axis],float(np.min(z+e)))
        linear[2+axis]=(lower+upper)/2;intervals.append([lower,upper])
    actual=center+scale*linear
    return dict(feasible=True,halfplanes=len(h)+4,vertices=len(poly),
                dg_lower=(center[:2]+scale[:2]*poly.min(axis=0)).tolist(),
                dg_upper=(center[:2]+scale[:2]*poly.max(axis=0)).tolist(),
                geometry=actual.tolist(),max_residual=float(np.max(abs(b+A@actual-y))),
                source_intersection_widths=[scale[2+i]*(x[1]-x[0]) for i,x in enumerate(intervals)],
                scale=scale.tolist())

def full_lp(b,A,y,eta,center):
    scale=np.maximum(np.sum(abs(np.linalg.pinv(A)),axis=1)*eta,1e-10)
    M=A*scale/eta;r=(y-b-A@center)/eta
    G=np.r_[M,-M];h=np.r_[r+1,1-r]
    bounds=list(zip((LOW-center)/scale,(HIGH-center)/scale))
    opts={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
    fit=linprog(np.zeros(4),A_ub=G,b_ub=h,bounds=bounds,method='highs',options=opts)
    if not fit.success:return {'feasible':False,'status':int(fit.status),'message':fit.message}
    extrema=[]
    for j in (0,1):
        pair=[]
        for sign in (1.,-1.):
            obj=np.zeros(4);obj[j]=sign
            opt=linprog(obj,A_ub=G,b_ub=h,bounds=bounds,method='highs',options=opts)
            if not opt.success:raise RuntimeError(opt.message)
            pair.append(float(center[j]+scale[j]*opt.x[j]))
        extrema.append(pair)
    v=center+scale*fit.x
    return {'feasible':True,'dg_lower':[z[0] for z in extrema],'dg_upper':[z[1] for z in extrema],
            'geometry':v.tolist(),'max_residual':float(np.max(abs(b+A@v-y)))}

def main():
    obspath=WORK/'profiles/adverse_observations_only.json'
    y=np.array(next(c for c in json.loads(obspath.read_text())['cases'] if c['id']=='eta_1e-8_separated_speeds')['observed']).ravel()
    seedpath=WORK/'profiles/blind_adverse_1e8.json';fitpath=WORK/'gain_continuation/result_native.json';profilepath=WORK/'gain_continuation/compatible_profiles.json'
    cases=[('blind_incompatible',json.loads(seedpath.read_text())['theta']),('gain_compatible',json.loads(fitpath.read_text())['theta'])]
    prof=json.loads(profilepath.read_text())['profiles']
    cases.extend((f'profile_{i}',prof[i]['theta']) for i in (0,1))
    reports=[]
    for name,theta in cases:
        theta=np.array(theta);b,A,margin=affine(theta);center=theta[LINEAR]
        start=time.perf_counter();poly=polygon_elimination(b,A,y,1e-8,center);tp=time.perf_counter()-start
        start=time.perf_counter();lp=full_lp(b,A,y,1e-8,center);tl=time.perf_counter()-start
        row={'case':name,'eta':1e-8,'minimum_physical_pair_guard':margin,'polygon':poly,'full4D_LP':lp,'polygon_seconds':tp,'lp_seconds':tl,
             'feasibility_agrees':poly['feasible']==lp['feasible']}
        if poly['feasible'] and lp['feasible']:
            error=max(np.max(abs(np.array(poly[k])-lp[k])) for k in ('dg_lower','dg_upper'))
            row['max_geometry_extremum_difference']=float(error)
            assert error<1e-7
        assert row['feasibility_agrees'];reports.append(row)
        print(json.dumps(row),flush=True)
    report={'theorem_scope':'exact algebraic elimination for conditional optics q; all affine native bounds retained',
            'benchmark_scope':'floating polygon and LP diagnostics, not interval or exact-rational certificates',
            'hidden_truth_inputs':False,'global14_outer_contractor_implemented':False,'rows':reports,
            'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (obspath,seedpath,fitpath,profilepath)},
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE/'elimination_benchmark.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
