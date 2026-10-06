#!/usr/bin/env python3
"""One fixed native +-0.001 box at the predeclared kappa=.1 witness.
Outward first jets certify centered corrected-record increments. This is a
box/domain enclosure attempt, not an empirical optical reconstruction.
Frozen left inverse proposals are reused from the prior point certificate.
"""
import sys,json,time,math
from pathlib import Path
import numpy as np
sys.argv.append('--interval') if '--interval' not in sys.argv else None
import finite_wedge_corrector_derivative as c
c.mp.iv.dps=45
iv=c.mp.iv; D=18; ROOT=Path(__file__).resolve().parent
saved=json.loads((ROOT/'finite_wedge_corrector_interval.json').read_text())
start=time.time(); report={'scope':'same kappa=0.1 witness, exact nominal F(theta_star), fixed native +-0.001 all18, no noise','native_radius':.001}
def lo(v): return math.nextafter(float(v.a),-math.inf)
def hi(v): return math.nextafter(float(v.b),math.inf)
def endpoints(v):return [lo(v),hi(v)]
def radius(v):return hi(abs(v))
def ni(A):return max(hi(sum(abs(v) for v in row)) for row in A)
def ia(A):return np.array([[iv.mpf(float(v)) for v in row] for row in A],dtype=object)
def serialize(A):return np.array([[endpoints(v) for v in row] for row in A],dtype=float).tolist()
def cinverse(A,proposal,name):
    W=ia(proposal);I=ia(np.eye(A.shape[0]));E=I-W@A;defect=ni(E);report[name+'_inverse_defect']=defect
    assert defect<1,(name,defect)
    out=(I+E+E@E)@W;tail=hi(iv.mpf(defect)**3/(1-iv.mpf(defect))*ni(W));eps=iv.mpf([-tail,tail])
    return np.array([[v+eps for v in row] for row in out],dtype=object)
def jvar(val,idx,box):
    dd=c.zero(D);dd[idx]=iv.mpf(1)/1000
    return c.J(val+(iv.mpf([-1,1])/1000 if box else 0),dd)
def native(box):
    pi=iv.pi;f=pi/180;n0=[iv.sqrt(iv.mpf(17)/8),iv.sqrt(iv.mpf(233)/125),iv.sqrt(iv.mpf(13)/5)]
    a0=iv.atan2(iv.mpf(1)/10,iv.sqrt(iv.mpf(99)/100))/f
    tx0=iv.atan2(iv.mpf(1),iv.mpf(3))/f
    nv=[iv.mpf(1)/20,iv.mpf(7)/20,iv.mpf(49)/20]+[a0]*3+[iv.mpf(0)]*3+n0+[tx0,iv.mpf(0),iv.mpf(1),iv.mpf(2),iv.mpf(3),iv.mpf(100)]
    z=[jvar(v,i,box) for i,v in enumerate(nv)]
    c.N=z[:3];c.e=[(v*f).sin() for v in z[3:6]];c.phi=[v*f for v in z[6:9]]
    c.t=[(v*f).sin()/(v*f).cos() for v in z[12:14]];c.b=z[14:16];c.g,c.d=z[16:18]
    c.Q=1+sum(v*v for v in c.t);c.nsq=[v*v for v in z[9:12]];c.h=[(c.nsq[j]*c.Q-(c.Q-1)).sqrt() for j in range(3)]
    return nv

def compile_records(box):
    nv=native(box);F=c.zero((2,200));F1=c.zero((2,200));F2=c.zero((2,200));Js=c.zero((2,200,D));Jw=c.zero((2,200,D))
    for k in range(200):
        fx=c.forward(k);ft=c.forward(k,True)
        for a in range(2):
            one=ft[a].c[0]+ft[a].c[1];two=one+ft[a].c[2]
            F[a,k]=fx[a].v;F1[a,k]=one.v;F2[a,k]=two.v;Js[a,k]=one.d-fx[a].d;Jw[a,k]=two.d-fx[a].d
        if k%50==0:print('box' if box else 'center','sample',k,'seconds',time.time()-start,flush=True)
    return F,F1,F2,Js,Jw
F,F1,F2,Js0,Jw0=compile_records(False)
_,_,_,Js,Jw=compile_records(True)
Rs=np.array([[sum(abs(z) for z in Js[a,k]) for k in range(200)] for a in range(2)],dtype=object)
Rw=np.array([[sum(abs(z) for z in Jw[a,k]) for k in range(200)] for a in range(2)],dtype=object)
report['corrected_first_record_radius_max']=max(hi(v) for v in Rs.flat)
report['corrected_second_record_radius_max']=max(hi(v) for v in Rw.flat)
report['first_record_radius_rows']=[[hi(v) for v in row] for row in Rs]
report['second_record_radius_rows']=[[hi(v) for v in row] for row in Rw]
report['corrected_first_record_center_jacobian_row_norm_max']=max(hi(sum(abs(v) for v in row)) for row in Js0.reshape(400,18))
report['corrected_second_record_center_jacobian_row_norm_max']=max(hi(sum(abs(v) for v in row)) for row in Jw0.reshape(400,18))
print('centered record radii',report['corrected_first_record_radius_max'],report['corrected_second_record_radius_max'],flush=True)
R=c.zero((7,6))
for j in range(1,7):
    den=iv.sqrt(j*(j+1))
    for k in range(j):R[k,j-1]=1/den
    R[j,j-1]=-iv.mpf(j)/den
H=np.array([[F1[a,k+11*j] for j in range(7)] for a in range(2) for k in range(123)],dtype=object)
P=ia(saved['frozen_left_inverse_proposals']['lag11']['real']);At=P@H@R
Dlag=cinverse(At,np.linalg.inv(c.marray(At)),'lag11_center')@P
# Certify exact frozen left inverse; preserve its actual enclosure.
report['Dlag_norminf']=ni(Dlag)
roots=[iv.mpc(1,0)]
for num in [1,7,49]:
    ang=2*iv.pi*num*11/400;z=iv.mpc(iv.cos(ang),iv.sin(ang));roots += [z,iv.mpc(z.real,-z.imag)]
pol=[iv.mpc(1,0)]
for z in roots:
    out=[iv.mpc(0,0)]*(len(pol)+1)
    for j,v in enumerate(pol):out[j]-=z*v;out[j+1]+=v
    pol=out
pol=np.array([v.real for v in pol],dtype=object)
# A(s)-I and center-forcing vector f(s)=D[(v-v0)+(H-H0)c0]
# are enclosed by their derivative ranges times the fixed native unit box.
Ader=c.zero((6,6,18));fder=c.zero((6,18));Ader0=c.zero((6,6,18));fder0=c.zero((6,18))
for q in range(18):
    dh=np.array([[Js[a,k+11*j,q] for j in range(7)] for a in range(2) for k in range(123)],dtype=object)
    dh0=np.array([[Js0[a,k+11*j,q] for j in range(7)] for a in range(2) for k in range(123)],dtype=object)
    rr=np.array([sum(pol[j]*Js[a,k+11*j,q] for j in range(8)) for a in range(2) for k in range(123)],dtype=object)
    rr0=np.array([sum(pol[j]*Js0[a,k+11*j,q] for j in range(8)) for a in range(2) for k in range(123)],dtype=object)
    Ader[:,:,q]=Dlag@dh@R;Ader0[:,:,q]=Dlag@dh0@R;fder[:,q]=Dlag@rr;fder0[:,q]=Dlag@rr0
    if q%6==0:print('spectral derivative column',q,flush=True)
Arad=np.array([[sum(abs(v) for v in Ader[i,j]) for j in range(6)] for i in range(6)],dtype=object)
frad=np.array([sum(abs(v) for v in row) for row in fder],dtype=object)
report['lag11_uniform_neumann_bound']=ni(Arad)
report['lag11_center_linearized_neumann_bound']=max(hi(sum(abs(v) for v in Ader0[i].flat)) for i in range(6))
report['lag11_uniform_forcing_radius']=[hi(v) for v in frad]
report['lag11_center_linearized_forcing_radius']=[hi(sum(abs(v) for v in row)) for row in fder0]
report['lag11_matrix_radius']=serialize(Arad)
report['lag11_certified_by_single_box_neumann']=report['lag11_uniform_neumann_bound']<1
# Even a failed norm bound is a valid enclosure, not proof of singularity.
report['status']='lag11 domain certified' if report['lag11_certified_by_single_box_neumann'] else 'UNRESOLVED: centered interval-jet lag11 Neumann bound >=1; cannot certify subsequent root/A2/chord domain'
report['elapsed_seconds']=time.time()-start
out=ROOT/'finite_wedge_native_chord_envelope.json';out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if not isinstance(v,list)},indent=2),flush=True)
# Store outward derivative boxes for the next deterministic stage without rerunning optics.
payload={'F1':serialize(F1),'F2':serialize(F2),'Js':np.array([[[endpoints(v) for v in row] for row in axis] for axis in Js],float).tolist(),'Jw':np.array([[[endpoints(v) for v in row] for row in axis] for axis in Jw],float).tolist(),'Js0':np.array([[[endpoints(v) for v in row] for row in axis] for axis in Js0],float).tolist(),'Jw0':np.array([[[endpoints(v) for v in row] for row in axis] for axis in Jw0],float).tolist(),'Dlag':serialize(Dlag),'Ader':np.array([[[endpoints(v) for v in row] for row in axis] for axis in Ader],float).tolist(),'fder':serialize(fder)}
(ROOT/'finite_wedge_native_chord_envelope_arrays.json').write_text(json.dumps(payload))
