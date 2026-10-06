#!/usr/bin/env python3
"""Independent same-point consistency audit, not a finite-radius certificate.
One prescribed kappa=1/10 witness; no search, tuning, or new physical case.
The independent optical trace uses UNIT directions and the paired transfer
identity; lambda derivatives use unscaled first/second tangents. Complex-step
parameter derivatives are numerical cross-checks, not rigorous enclosures.
"""
import importlib.util
from pathlib import Path
import json
import numpy as np
from itertools import product
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('original_derivative',ROOT/'finite_wedge_corrector_derivative.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
D=18
eta=np.array([1/20,7/20,49/20,1,1,1,0,0,0,3/2,7/5,5/3,1/3,0,1,2,3,100],dtype=float)
clock=np.arange(200)/20

class L:
    # (value, first lambda derivative, second lambda derivative), no factorials.
    def __init__(self,v,a=0,b=0):self.v,self.a,self.b=v,a,b
    def __add__(x,y):
        y=lift(y);return L(x.v+y.v,x.a+y.a,x.b+y.b)
    __radd__=__add__
    def __neg__(x):return L(-x.v,-x.a,-x.b)
    def __sub__(x,y):return x+-lift(y)
    def __rsub__(x,y):return lift(y)+-x
    def __mul__(x,y):
        y=lift(y);return L(x.v*y.v,x.a*y.v+x.v*y.a,x.b*y.v+2*x.a*y.a+x.v*y.b)
    __rmul__=__mul__
    def inv(x):return L(1/x.v,-x.a/x.v**2,2*x.a**2/x.v**3-x.b/x.v**2)
    def __truediv__(x,y):return x*lift(y).inv()
    def __rtruediv__(x,y):return lift(y)*x.inv()
    def sqrt(x):
        v=np.sqrt(x.v);return L(v,x.a/(2*v),x.b/(2*v)-x.a**2/(4*v**3))
def lift(x):return x if isinstance(x,L) else L(x)
def sq(x):return x*x
def dot(x,y):return sum(a*b for a,b in zip(x,y))

def optical(eta,jet=False):
    N,e,ph,h=eta[:3],eta[3:6]/10,eta[6:9],eta[9:12]
    t,b,g,d=eta[12:14],eta[14:16],eta[16],eta[17]
    Q=1+np.dot(t,t);ns=(h*h+Q-1)/Q
    q=[lift(x/np.sqrt(Q)) for x in t]
    p=[lift(b[a]+6*t[a]) for a in range(2)]
    for j in range(3):
        angle=2*np.pi*N[j]*clock+ph[j]
        # e/sqrt(1-e^2) has lambda-derivative e and lambda-second derivative 0 at 0.
        tilt=L(0,e[j],0) if jet else L(e[j]/np.sqrt(1-e[j]**2))
        u=[tilt*np.cos(angle),tilt*np.sin(angle)]
        H=(ns[j]-dot(q,q)).sqrt();P=H-dot(u,q);DD=1+dot(u,u)
        E=(sq(P)-DD*(ns[j]-1)).sqrt()
        cc=(P-E)/DD
        qnext=[q[a]+cc*u[a] for a in range(2)];Z=H-cc
        v=[x/H for x in q];w=[x/Z for x in qnext]
        # Independent position identity: internal-plus-external slope transfer.
        correction=dot(u,[p[a]+3*v[a] for a in range(2)])/(1-dot(u,v))
        ell=g if j<2 else d
        p=[p[a]+3*v[a]+ell*w[a]+(v[a]-w[a])*correction for a in range(2)]
        q=qnext
    if jet:return np.array([x.v+x.a for x in p]),np.array([x.v+x.a+x.b/2 for x in p])
    return np.array([x.v for x in p])

def feature(eta):
    N,e,ph,h=eta[:3],eta[3:6]/10,eta[6:9],eta[9:12]
    t,b,g,d=eta[12:14],eta[14:16],eta[16],eta[17]
    B=b+t*(6+2*g+d+3*sum(1/h))
    coeff=np.zeros((2,7),dtype=np.result_type(eta));coeff[:,0]=B
    for j in range(3):
        LL=d+(2-j)*g+sum(3/h[j+1:]);W=d+(2-j)*g+sum(3/h[j+1:]**3)
        M=(h[j]-1)*(LL*np.eye(2)+(W+LL/h[j])*np.outer(t,t)-np.outer(t,B)/h[j])
        rot=np.array([[np.cos(ph[j]),-np.sin(ph[j])],[np.sin(ph[j]),np.cos(ph[j])]])
        coeff[:,1+2*j:3+2*j]=e[j]*M@rot
    x,y=t;H=h[2];rho=x*x+y*y
    ar=2*H*d+d*(H+1)*rho-x*B[0]-y*B[1];ai=x*B[1]-y*B[0]
    aa=4*d*H**3+d*H**2*(3*H-1)*rho;bb=-4*H**2-2*(H**2+1)*rho
    cr=aa*x+bb*(B[0]-d*x);ci=-aa*y+bb*(-B[1]+d*y)
    dr=2*(H-1)*(ar*ar-ai*ai);di=4*(H-1)*ar*ai
    ir=(cr*dr+ci*di)/(dr*dr+di*di);ii=(ci*dr-cr*di)/(dr*dr+di*di)
    return np.r_[N,coeff.ravel(),ir],coeff,ir+1j*ii

def complex_jac(fn):
    out=np.zeros(np.shape(fn(eta))+(D,))
    for j in range(D):
        point=eta.astype(complex);point[j]+=1e-25j
        out[...,j]=np.imag(fn(point))/1e-25
    return out

def left(A):
    P=np.linalg.pinv(A);return np.linalg.solve(P@A,P)

F=optical(eta);F1,F2=optical(eta,True)
JF=complex_jac(optical);J1=complex_jac(lambda p:optical(p,True)[0]);J2=complex_jac(lambda p:optical(p,True)[1])
feat,co,I=feature(eta);G=complex_jac(lambda p:feature(p)[0])
origF=np.array([[c.forward(k)[a].v for k in range(200)] for a in range(2)])
origJF=np.array([[c.forward(k)[a].d for k in range(200)] for a in range(2)])
orig1=np.array([[c.forward(k,True)[a].c[0].v+c.forward(k,True)[a].c[1].v for k in range(200)] for a in range(2)])
orig2=np.array([[sum(q.v for q in c.forward(k,True)[a].c) for k in range(200)] for a in range(2)])
origJ1=np.array([[c.forward(k,True)[a].c[0].d+c.forward(k,True)[a].c[1].d for k in range(200)] for a in range(2)])
origJ2=np.array([[sum((q.d for q in c.forward(k,True)[a].c),np.zeros(D)) for k in range(200)] for a in range(2)])
errors={name:float(np.max(np.abs(a-b))) for name,a,b in [('F unit versus Q-scaled',F,origF),('DF complex step versus jets',JF,origJF),('F1 tangent versus polynomial',F1,orig1),('F2 tangent versus polynomial',F2,orig2),('DF1 complex step versus jets',J1,origJ1),('DF2 complex step versus jets',J2,origJ2),('G independent versus jets',G,c.G)]}
for name,error in errors.items():assert error<1e-9,(name,error)
R=np.zeros((7,6))
for j in range(1,7):R[:j,j-1]=1/np.sqrt(j*(j+1));R[j,j-1]=-j/np.sqrt(j*(j+1))
assert np.max(abs(R.T@np.ones(7)))<1e-15
assert np.max(abs(R.T@R-np.eye(6)))<1e-15
Hlag=np.array([[F1[a,k+11*j] for j in range(7)] for a in range(2) for k in range(123)])
Dl=left(Hlag@R)
rots=np.exp(2j*np.pi*np.array([1,7,49])*11/400)
roots=np.r_[1,np.column_stack((rots,np.conj(rots))).ravel()]
pol=np.polynomial.polynomial.polyfromroots(roots).real
assert abs(sum(pol))<1e-14
V7=np.column_stack([np.ones(200)]+[f(2*np.pi*N*clock) for N in eta[:3] for f in [np.cos,np.sin]])
L7=left(V7)
indices=[m for m in product(range(-2,3),repeat=3) if sum(abs(v) for v in m)<=2]
nums=np.array([m[0]+7*m[1]+49*m[2] for m in indices]);target=indices.index((0,0,2))
assert len(nums)==25 and len(set(nums%400))==25
V31=np.column_stack([np.exp(2j*np.pi*num*np.arange(200)/400) for num in nums]+[(np.arange(200)/199)*np.exp(2j*np.pi*num*np.arange(200)/400) for num in [1,-1,7,-7,49,-49]])
ell=left(V31)[target]
U=(co[0,5]+co[1,6]+1j*(co[1,5]-co[0,6]))/2
Cfit=ell@(F2[0]+1j*F2[1]);errors['F2 target self harmonic']=float(abs(Cfit-I*U**2));assert errors['F2 target self harmonic']<1e-10

def preprocess(ds,dw):
    recurrent=np.array([sum(pol[j]*ds[a,k+11*j] for j in range(8)) for a in range(2) for k in range(123)])
    dc=-R@Dl@recurrent
    dn=np.zeros((3,D))
    for j,z in enumerate(rots):
        dp=np.polynomial.polynomial.polyval(z,np.arange(1,8)*pol[1:])
        dn[j]=20/(2*np.pi*11)*np.imag(-sum(dc[k]*z**k for k in range(7))/(dp*z))
    cf=np.array([L7@(ds[a]-J1[a,:,:3]@dn) for a in range(2)])
    ww=dw-np.array([J2[a,:,:3]@dn for a in range(2)])
    dc2=ell@(ww[0]+1j*ww[1])
    du=(cf[0,5]+cf[1,6]+1j*(cf[1,5]-cf[0,6]))/2
    di=np.real(dc2/U**2-2*I*du/U)
    return np.vstack((dn,cf[0],cf[1],di))
Kident=preprocess(J1,J2);leftidentity=np.linalg.solve(G,Kident)
errors['extracted feature differential identity']=float(np.max(abs(Kident-G)))
errors['DA(F1prime,F2prime) identity infinity norm']=float(np.linalg.norm(leftidentity-np.eye(D),np.inf))
assert errors['DA(F1prime,F2prime) identity infinity norm']<1e-6
K=preprocess(J1-JF,J2-JF);T=np.linalg.solve(G,K)
evals=np.linalg.eigvals(T)
original=json.loads((ROOT/'finite_wedge_corrector_proposal.json').read_text())
errors['DT difference from source proposal infinity norm']=float(np.linalg.norm(T-np.array(original['DT_mid']),np.inf))
assert errors['DT difference from source proposal infinity norm']<0.001
out={'scope':'one fixed witness; floating independent derivative cross-check, no radius/noise/global certificate','errors':errors,'trace_proposal':float(np.trace(T)),'eigenvalues_proposal':[[float(v.real),float(v.imag)] for v in evals], 'spectral_radius_proposal':float(max(abs(evals))),'DT_independent':T.tolist()}
path=ROOT/'finite_wedge_derivative_independent_audit.json';path.write_text(json.dumps(out,indent=2))
for name,error in errors.items():print(name,':',error)
print('Independent DT trace',np.trace(T))
print('Independent eigenvalues',evals)
print('PASS: all independent same-point consistency assertions; no rigorous spectral conclusion from this script alone.')
print('Saved',path)
