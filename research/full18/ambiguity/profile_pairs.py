"""Construct finite-noise ambiguity witnesses for the original passive full18 map.

Search only: exact-time strict per-axis forward algebra in binary64, complex-step
Jacobians, fixed-coordinate nuisance profiling, then minimax LP polishing.
Endpoint proof is separate in verify_pairs.py. Never sorts physical prism order.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, linprog

NAMES = ['N1','N2','N3','ax1','ax2','ax3','ay1','ay2','ay3','ng1','ng2','ng3','d_W','gap','bm_ax','bm_ay','bm_px','bm_py']
LO=np.array([-3.5]*3+[-18.]*6+[1.3]*3+[50.,2.,-25.,-25.,-5.,-5.])
HI=np.array([3.5]*3+[18.]*6+[1.8]*3+[200.,15.,25.,25.,5.,5.])
RG=HI-LO
TIMES=np.arange(200)/20.

def forward(v):
    """Strict mathematical per-axis Snell map, all 200 passive timestamps."""
    v=np.asarray(v)
    ang=2*np.pi*v[:3,None]*TIMES+v[6:9,None]*np.pi/180
    tx=np.tan(v[3:6,None]*np.pi/180)
    cs=np.cos(ang)*tx; sn=np.sin(ang)*tx
    pos=[]
    zs=[6,9,9+v[13],12+v[13],12+2*v[13],15+2*v[13],15+2*v[13]+v[12]]
    for axis,(u,w) in enumerate(((cs,sn),(sn,cs))):
        q=np.sqrt(1+w*w); norm=np.sqrt(1+u*u+w*w)
        sf=[0,u[0]/norm[0],0,u[1]/norm[1],0,u[2]/norm[2]]
        cf=[1,q[0]/norm[0],1,q[1]/norm[1],1,q[2]/norm[2]]
        slopes=[0,u[0]/q[0],0,u[1]/q[1],0,u[2]/q[2],0]
        ratios=[1/v[9],v[9],1/v[10],v[10],1/v[11],v[11]]
        tc=np.tan(v[14+axis]*np.pi/180); p=v[16+axis]+6*tc; z=6
        for j in range(6):
            norm_in=np.sqrt(1+tc*tc); si0=tc/norm_in; si2=1/norm_in
            cy=-cf[j]*si0-sf[j]*si2
            root=np.sqrt(1-(ratios[j]*cy)**2)
            out0=ratios[j]*(-cf[j])*cy-sf[j]*root
            out2=-ratios[j]*sf[j]*cy+cf[j]*root
            tc=out0/out2
            step=(zs[j+1]+slopes[j+1]*p-z)/(1-slopes[j+1]*tc)
            p=p+step*tc;z=z+step
        pos.append(p)
    return np.stack(pos,axis=-1).reshape(-1)

def jacobian(v):
    jac=np.empty((400,18))
    for j in range(18):
        q=np.asarray(v,dtype=complex);q[j]+=1e-22j
        jac[:,j]=np.imag(forward(q))/1e-22
    return jac

def profile(base,coord,shift):
    target=forward(base);free=np.array([j for j in range(18) if j!=coord])
    fixed=base.copy();fixed[coord]+=shift
    def unpack(z):
        x=fixed.copy();x[free]=base[free]+RG[free]*z;return x
    def fun(z):return (forward(unpack(z))-target)*1e6
    def jac(z):return jacobian(unpack(z))[:,free]*RG[free]*1e6
    lower=(LO[free]-base[free])/RG[free];upper=(HI[free]-base[free])/RG[free]
    opt=least_squares(fun,np.zeros(17),jac=jac,bounds=(lower,upper),xtol=2e-14,ftol=2e-14,gtol=2e-14,max_nfev=100)
    alt=unpack(opt.x)
    # Sequential linear minimax polish. Every accepted step lowers the actual
    # nonlinear infinity residual; this is a candidate generator, not a proof.
    lp_records=[]
    for _ in range(6):
        r=forward(alt)-target;J=jacobian(alt)[:,free]*RG[free]
        scale=1/max(np.max(np.abs(r)),1e-12)
        A=np.vstack([np.column_stack([J*scale,-np.ones(400)]),np.column_stack([-J*scale,-np.ones(400)])])
        b=np.r_[-r*scale,r*scale]
        bounds=[(max((LO[j]-alt[j])/RG[j],-1e-3),min((HI[j]-alt[j])/RG[j],1e-3)) for j in free]+[(0,None)]
        sol=linprog(np.r_[np.zeros(17),1.],A_ub=A,b_ub=b,bounds=bounds,method='highs',options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
        if not sol.success:break
        best=np.max(np.abs(r));accepted=False
        for alpha in (1,.5,.25,.125):
            candidate=alt.copy();candidate[free]+=alpha*RG[free]*sol.x[:-1]
            error=float(np.max(np.abs(forward(candidate)-target)))
            if error<best:
                alt=candidate;accepted=True;lp_records.append(error);break
        if not accepted:break
    diff=forward(alt)-target
    return {'base':base.tolist(),'alternative':alt.tolist(),'fixed_coordinate':NAMES[coord],'shift':shift,
            'parameter_difference':(alt-base).tolist(),'half_coordinate_separation':(np.abs(alt-base)/2).tolist(),
            'numerical_max_position_difference':float(np.max(np.abs(diff))),
            'numerical_xy_max_difference':np.max(np.abs(diff.reshape(200,2)),axis=0).tolist(),
            'numerical_eta_midpoint':float(np.max(np.abs(diff))/2),'least_squares_nfev':opt.nfev,
            'lp_residual_history':lp_records,'search_is_not_certificate':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='profile_candidates.json');args=ap.parse_args()
    records=[];start=time.time()
    for wedge in (2.,3.,4.,6.,8.,12.):
        base=np.array([2.71,-1.93,.83,wedge,-1.15*wedge,.9*wedge,7.,-11.,17.,1.46,1.57,1.67,137.,7.,2.,-3.,1.2,-2.1])
        for coord in (12,13):
            rec=profile(base,coord,.0022);rec['family']='separated_speeds';rec['wedge_scale']=wedge;records.append(rec)
            print(json.dumps({'wedge':wedge,'coord':NAMES[coord],'eta':rec['numerical_eta_midpoint'],'nfev':rec['least_squares_nfev']}),flush=True)
            Path(args.out).write_text(json.dumps({'contract':'original ordered full18, 200 passive exact k/20 timestamps, native box','names':NAMES,'records':records,'elapsed_seconds':time.time()-start},indent=2))

if __name__=='__main__':main()
