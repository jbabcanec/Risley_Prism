#!/usr/bin/env python3
"""Cancellation-preserving λ-integral first jets, one unchanged native box.
A fixed eight-cell λ quadrature is interval integration, not point sampling:
each exact nonnegative integration weight multiplies a uniform coefficient
interval on its closed cell. No physical-input subdivision or search.
"""
import sys,json,time,math
from pathlib import Path
import numpy as np
sys.argv.append('--interval') if '--interval' not in sys.argv else None
import finite_wedge_corrector_derivative as c
c.mp.iv.dps=40;iv=c.mp.iv;D=18;ROOT=Path(__file__).resolve().parent
start=time.time()
def lo(v):return math.nextafter(float(v.a),-math.inf)
def hi(v):return math.nextafter(float(v.b),math.inf)
def ends(v):return [lo(v),hi(v)]
def jvar(val,idx):
    dd=c.zero(D);dd[idx]=iv.mpf(1)/1000
    return c.J(val+iv.mpf([-1,1])/1000,dd)
def native():
    f=iv.pi/180;n0=[iv.sqrt(iv.mpf(17)/8),iv.sqrt(iv.mpf(233)/125),iv.sqrt(iv.mpf(13)/5)]
    a0=iv.atan2(iv.mpf(1)/10,iv.sqrt(iv.mpf(99)/100))/f;tx0=iv.atan2(iv.mpf(1),iv.mpf(3))/f
    nv=[iv.mpf(1)/20,iv.mpf(7)/20,iv.mpf(49)/20]+[a0]*3+[iv.mpf(0)]*3+n0+[tx0,iv.mpf(0),iv.mpf(1),iv.mpf(2),iv.mpf(3),iv.mpf(100)]
    z=[jvar(v,i) for i,v in enumerate(nv)];N=z[:3];e=[(v*f).sin() for v in z[3:6]];phi=[v*f for v in z[6:9]]
    t=[(v*f).sin()/(v*f).cos() for v in z[12:14]];b=z[14:16];g,d=z[16:18];Q=1+sum(v*v for v in t);nsq=[v*v for v in z[9:12]]
    return N,e,phi,t,b,g,d,Q,nsq
class L:
    def __init__(self,x):self.c=x.c[:] if isinstance(x,L) else (x+[c.J(0)]*(4-len(x))) if isinstance(x,list) else [c.jj(x),c.J(0),c.J(0),c.J(0)]
    def __add__(a,b):b=ll(b);return L([a.c[i]+b.c[i] for i in range(4)])
    __radd__=__add__
    def __neg__(a):return L([-x for x in a.c])
    def __sub__(a,b):return a+-ll(b)
    def __rsub__(a,b):return ll(b)+-a
    def __mul__(a,b):
        b=ll(b);return L([sum((a.c[j]*b.c[k-j] for j in range(k+1)),c.J(0)) for k in range(4)])
    __rmul__=__mul__
    def inv(a):
        z=[1/a.c[0]]
        for k in range(1,4):z.append(-sum((a.c[j]*z[k-j] for j in range(1,k+1)),c.J(0))/a.c[0])
        return L(z)
    def __truediv__(a,b):return a*ll(b).inv()
    def __rtruediv__(a,b):return ll(b)*a.inv()
    def sqrt(a):
        z=[a.c[0].sqrt()]
        for k in range(1,4):z.append((a.c[k]-sum((z[j]*z[k-j] for j in range(1,k)),c.J(0)))/(2*z[0]))
        return L(z)
def ll(x):return x if isinstance(x,L) else L(x)
N,e,phi,t,b,g,d,Q,nsq=native()
def forward(k,lam):
    X=[ll(x) for x in t];p=[ll(b[a]+6*t[a]) for a in range(2)]
    for j in range(3):
        angle=N[j]*(2*iv.pi*k/20)+phi[j]
        ej=L([e[j]*lam,e[j]]);tilt=ej/(1-ej*ej).sqrt();u=[tilt*ll(angle.cos()),tilt*ll(angle.sin())]
        Hj=(ll(nsq[j]*Q)-sum(x*x for x in X)).sqrt();Pj=Hj-sum(u[a]*X[a] for a in range(2));Dj=1+sum(x*x for x in u)
        Ej=(Pj*Pj-Dj*ll((nsq[j]-1)*Q)).sqrt();Z=(Hj*Dj-Pj+Ej)/Dj;Xn=[X[a]+u[a]*(Pj-Ej)/Dj for a in range(2)]
        pe=[p[a]+X[a]*(3+sum(u[aa]*p[aa] for aa in range(2)))/Pj for a in range(2)]
        ext=ll(g if j<2 else d)-sum(u[a]*pe[a] for a in range(2));p=[pe[a]+Xn[a]*ext/Z for a in range(2)];X=Xn
    return p
Js=c.zero((2,200,D));Jw=c.zero((2,200,D));CELLS=8
CELLROOT=ROOT/'finite_wedge_native_integral_cells';CELLROOT.mkdir(exist_ok=True)
for cell in range(CELLS):
    a=iv.mpf(cell)/CELLS;b0=iv.mpf(cell+1)/CELLS;lam=iv.mpf([a.a,b0.b]);w1=2*b0-b0*b0-2*a+a*a;w2=(1-a)**3-(1-b0)**3
    cell_c2=c.zero((2,200));cell_c3=c.zero((2,200));cell_d2=c.zero((2,200,D));cell_d3=c.zero((2,200,D))
    for k in range(200):
        f=forward(k,lam)
        for axis in range(2):
            cell_c2[axis,k]=f[axis].c[2].v;cell_c3[axis,k]=f[axis].c[3].v
            cell_d2[axis,k]=f[axis].c[2].d;cell_d3[axis,k]=f[axis].c[3].d
            Js[axis,k]-=f[axis].c[2].d*w1;Jw[axis,k]-=f[axis].c[3].d*w2
    cellout={'cell':cell+1,'lambda_interval':ends(lam),'weight_R1_c2':ends(w1),'weight_R2_c3':ends(w2),'c2_values':[[ends(v) for v in row] for row in cell_c2],'c3_values':[[ends(v) for v in row] for row in cell_c3],'c2_native_first_jets':[[[ends(v) for v in row] for row in axis] for axis in cell_d2],'c3_native_first_jets':[[[ends(v) for v in row] for row in axis] for axis in cell_d3]}
    (CELLROOT/f'cell_{cell+1:02d}.json').write_text(json.dumps(cellout))
    print('integral cell',cell+1,'of',CELLS,'seconds',time.time()-start,flush=True)
# Intersect two mathematically independent interval extensions of the SAME derivative.
# Prior direct first-jet array also bounds it, so intersection is rigorous.
old=json.loads((ROOT/'finite_wedge_native_chord_envelope_arrays.json').read_text())
for name,A in [('Js',Js),('Jw',Jw)]:
    for idx in np.ndindex(A.shape):
        pair=np.array(old[name])[idx];lv=max(lo(A[idx]),float(pair[0]));uv=min(hi(A[idx]),float(pair[1]));assert lv<=uv
        A[idx]=iv.mpf([lv,uv])
old.pop('Ader',None);old.pop('fder',None)
old['Js']=[[[ends(v) for v in row] for row in axis] for axis in Js];old['Jw']=[[[ends(v) for v in row] for row in axis] for axis in Jw]
(ROOT/'finite_wedge_native_integral_envelope_arrays.json').write_text(json.dumps(old))
report={'lambda_cells':CELLS,'native_radius':.001,'first_record_radius_max':max(hi(sum(abs(z) for z in row)) for row in Js.reshape(400,18)),'second_record_radius_max':max(hi(sum(abs(z) for z in row)) for row in Jw.reshape(400,18)),'elapsed_seconds':time.time()-start}
(ROOT/'finite_wedge_native_integral_envelope.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
