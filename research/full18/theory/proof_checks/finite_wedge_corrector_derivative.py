#!/usr/bin/env python3
"""Exactly one original oblique witness, kappa=1/10. No parameter sweep.
Floating proposal first; --interval uses outward interval first jets and matrix
Neumann certificates. Coordinates eta=(N[3], e/kappa[3], phi[3] radians,
h[3], t[2], b[2], g,d). This is derivative-at-point only.
"""
import os,sys,json,math,time
from itertools import product
import numpy as np
import scipy.linalg as sla
import mpmath as mp
from finite_wedge_point_guards import audit
IV='--interval' in sys.argv
if IV:mp.iv.dps=55
ctx=mp.iv if IV else None
D=18

def r(x):return mp.iv.mpf(x) if IV else float(x)
def sin(x):return mp.iv.sin(x) if IV else math.sin(x)
def cos(x):return mp.iv.cos(x) if IV else math.cos(x)
def sqrt(x):return mp.iv.sqrt(x) if IV else math.sqrt(x)
def arr(x):return np.array(x,dtype=object if IV else float)
def zero(shape):return np.full(shape,r(0),dtype=object if IV else float)
def mid(x):
    if not IV and np.iscomplexobj(x):return complex(x)
    if hasattr(x,'imag') and not isinstance(x,(float,int,np.floating)):
        if IV:return complex(float(x.real.mid),float(x.imag.mid)) if not hasattr(x,'_mpi_') else float(x.mid)
    return float(x)
def real(x):return x.real

def imat(A):
    return np.array([[mp.iv.mpc(complex(x).real,complex(x).imag) if np.iscomplexobj(A) else r(float(x)) for x in row] for row in A],dtype=object) if IV else np.array(A)
def lower(x):return math.nextafter(float(x.a),-math.inf) if IV else float(x)
def upper(x):return math.nextafter(float(x.b),math.inf) if IV else float(x)
def norminf(A):return max(upper(sum(abs(x) for x in row)) for row in A)
def marray(A):return np.array([[mid(x) for x in row] for row in A])
def cplx(x,y):return mp.iv.mpc(x,y) if IV else complex(x,y)
def conj(x):return cplx(x.real,-x.imag)

class J:
    def __init__(self,v,d=None):self.v=v.v if isinstance(v,J) else r(v) if isinstance(v,(int,float)) else v;self.d=(v.d.copy() if isinstance(v,J) else zero(D)) if d is None else d
    def __add__(a,b):
        b=jj(b);return J(a.v+b.v,a.d+b.d)
    __radd__=__add__
    def __neg__(a):return J(-a.v,-a.d)
    def __sub__(a,b):return a+-jj(b)
    def __rsub__(a,b):return jj(b)+-a
    def __mul__(a,b):
        b=jj(b);return J(a.v*b.v,a.d*b.v+b.d*a.v)
    __rmul__=__mul__
    def __truediv__(a,b):
        b=jj(b);return J(a.v/b.v,(a.d-b.d*(a.v/b.v))/b.v)
    def __rtruediv__(a,b):return jj(b)/a
    def __pow__(a,n):
        if n==0:return J(1)
        if n<0:return 1/(a**(-n))
        out=J(1)
        for _ in range(n):out=out*a
        return out
    def sqrt(a):
        v=sqrt(a.v);return J(v,a.d/(2*v))
    def sin(a):return J(sin(a.v),a.d*cos(a.v))
    def cos(a):return J(cos(a.v),-a.d*sin(a.v))
def jj(x):return x if isinstance(x,J) else J(x)

class P:
    """lambda polynomial truncated at degree two; coefficients are first jets."""
    def __init__(self,x):self.c=x.c[:] if isinstance(x,P) else x if isinstance(x,list) else [jj(x),J(0),J(0)]
    def __add__(a,b):b=pp(b);return P([a.c[i]+b.c[i] for i in range(3)])
    __radd__=__add__
    def __neg__(a):return P([-x for x in a.c])
    def __sub__(a,b):return a+-pp(b)
    def __rsub__(a,b):return pp(b)+-a
    def __mul__(a,b):
        b=pp(b);return P([sum((a.c[j]*b.c[k-j] for j in range(k+1)),J(0)) for k in range(3)])
    __rmul__=__mul__
    def inv(a):
        c=[1/a.c[0]]
        for k in range(1,3):c.append(-sum((a.c[j]*c[k-j] for j in range(1,k+1)),J(0))/a.c[0])
        return P(c)
    def __truediv__(a,b):return a*pp(b).inv()
    def __rtruediv__(a,b):return pp(b)*a.inv()
    def sqrt(a):
        c=[a.c[0].sqrt()]
        for k in range(1,3):c.append((a.c[k]-sum((c[j]*c[k-j] for j in range(1,k)),J(0)))/(2*c[0]))
        return P(c)
def pp(x):return x if isinstance(x,P) else P(x)

PI=mp.iv.pi if IV else math.pi
vals=[r(1)/20,r(7)/20,r(49)/20]+[r(1)]*3+[r(0)]*3+[r(3)/2,r(7)/5,r(5)/3,r(1)/3,r(0),r(1),r(2),r(3),r(100)]
eta=[]
for j,v in enumerate(vals):
    dd=zero(D);dd[j]=r(1);eta.append(J(v,dd))
N=eta[:3];e=[x/10 for x in eta[3:6]];phi=eta[6:9];h=eta[9:12];t=eta[12:14];b=eta[14:16];g,d=eta[16:18]
Q=1+sum(x*x for x in t);nsq=[(x*x+Q-1)/Q for x in h]

# Exact first-order feature circuit and physical invariant.
B=[b[a]+t[a]*(6+2*g+d+3*sum(1/x for x in h)) for a in range(2)]
L=[d+2*g+3/h[1]+3/h[2],d+g+3/h[2],d]
W=[d+2*g+3/(h[1]**3)+3/(h[2]**3),d+g+3/(h[2]**3),d]
M=[[[ (h[j]-1)*(L[j]*(1 if a==bb else 0)+(W[j]+L[j]/h[j])*t[a]*t[bb]-t[a]*B[bb]/h[j]) for bb in range(2)] for a in range(2)] for j in range(3)]
coeff=[[B[a]] for a in range(2)]
for j in range(3):
    cp=phi[j].cos();sp=phi[j].sin()
    for a in range(2):coeff[a]+=[e[j]*(M[j][a][0]*cp+M[j][a][1]*sp),e[j]*(-M[j][a][0]*sp+M[j][a][1]*cp)]
# Complex arithmetic represented as pairs of real jets, avoiding interval casting.
def cmul(a,b):return [a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]]
def cadd(a,b):return [a[0]+b[0],a[1]+b[1]]
def cscale(a,s):return [x*s for x in a]
def cdiv(a,b):return cscale(cmul(a,[b[0],-b[1]]),1/(b[0]*b[0]+b[1]*b[1]))
T=t;TB=[t[0],-t[1]];BB=[B[0],-B[1]];rho=sum(x*x for x in t);H=h[2]
A=cadd([2*H*d+d*(H+1)*rho,J(0)],cscale(cmul(T,BB),-1))
C0=cadd(cscale(TB,4*d*H**3+d*H**2*(3*H-1)*rho),cscale(cadd(BB,cscale(TB,-d)),-4*H**2-2*(H**2+1)*rho))
I=cdiv(C0,cscale(cmul(A,A),2*(H-1)))
features=N+[x for row in coeff for x in row]+[I[0]]
G=arr([x.d for x in features])

# Compiler in Q-scaled directions, normalized back to L=1 each stage.
def forward(k,taylor=False):
    wrap=pp if taylor else jj
    X=[wrap(x) for x in t];p=[wrap(b[a]+6*t[a]) for a in range(2)]
    for j in range(3):
        angle=N[j]*(2*PI*r(k)/20)+phi[j]
        if taylor:
            ej=P([J(0),e[j],J(0)]);tilt=ej/(1-ej*ej).sqrt()
        else:tilt=e[j]/(1-e[j]*e[j]).sqrt()
        u=[tilt*wrap(angle.cos()),tilt*wrap(angle.sin())]
        Hj=(wrap(nsq[j]*Q)-sum(x*x for x in X)).sqrt()
        Pj=Hj-sum(u[a]*X[a] for a in range(2));Dj=1+sum(x*x for x in u)
        Ej=(Pj*Pj-Dj*wrap((nsq[j]-1)*Q)).sqrt()
        Z=(Hj*Dj-Pj+Ej)/Dj
        Xn=[X[a]+u[a]*(Pj-Ej)/Dj for a in range(2)]
        pe=[p[a]+X[a]*(3+sum(u[aa]*p[aa] for aa in range(2)))/Pj for a in range(2)]
        ext=wrap(g if j<2 else d)-sum(u[a]*pe[a] for a in range(2))
        p=[pe[a]+Xn[a]*ext/Z for a in range(2)];X=Xn
    return p

LEFT_PROPOSALS={}
def leftinverse(A,name):
    P0=np.linalg.pinv(marray(A).astype(complex if any(isinstance(x,complex) or (hasattr(x,'_mpci_') and not hasattr(x,'_mpi_')) for x in A.flat) else float))
    LEFT_PROPOSALS[name]={'real':P0.real.tolist(),'imag':P0.imag.tolist() if np.iscomplexobj(P0) else None}
    if not IV:return np.linalg.solve(P0@A,P0)
    P0=imat(P0);nn=A.shape[1];Id=imat(np.eye(nn));E=Id-P0@A;dd=norminf(E);np0=norminf(P0)
    assert dd<1e-6,(name,dd)
    # Sum 0,1,2 Neumann terms plus rigorous common entrywise tail enclosure.
    out=(Id+E+E@E)@P0;tail=upper(r(dd)**3/(1-r(dd))*r(np0))
    eps=mp.iv.mpf([-tail,tail])
    complexA=any((hasattr(x,'_mpci_') and not hasattr(x,'_mpi_')) for x in A.flat)
    if complexA:eps=mp.iv.mpc(eps,eps)
    out=np.array([[x+eps for x in row] for row in out],dtype=object)
    print('VERIFIED LEFT INVERSE',name,'defect',dd,'tail entry bound',tail,flush=True)
    return out

if __name__=='__main__':
    audit() # FIRST, before derivative calculation; abort without altering witness on failure.
    # Remaining off-model branch guards at this very same Taylor pair.
    rho0=rho.v;Bv=t[0].v*B[1].v-t[1].v*B[0].v
    f1=r(399563722409659)/317911540674000
    for guard in [rho0,abs(Bv),f1,h[2].v-1]:assert float(guard.a if IV else guard)>0
    for j in range(3):
        detM=M[j][0][0].v*M[j][1][1].v-M[j][0][1].v*M[j][1][0].v
        beta=1/(h[j].v*L[j].v)
        assert float(detM.a if IV else detM)>0 and float(beta.a if IV else beta)>0
    print('Strict ellipse and bivariate chart guards PASS: rho>0, Bv!=0, det M_i>0, beta_i>0, h3>1, audited f1!=0.',flush=True)
    start=time.time();F=zero((2,200));F1=zero((2,200));F2=zero((2,200));JF=zero((2,200,D));J1=zero((2,200,D));J2=zero((2,200,D))
    for k in range(200):
        fx=forward(k);ft=forward(k,True)
        for a in range(2):
            one=ft[a].c[0]+ft[a].c[1];two=one+ft[a].c[2]
            F[a,k]=fx[a].v;JF[a,k]=fx[a].d;F1[a,k]=one.v;J1[a,k]=one.d;F2[a,k]=two.v;J2[a,k]=two.d
            # Independent first order formula sanity checked on the same point.
            direct=coeff[a][0]+sum(coeff[a][2*j+1]*(N[j]*(2*PI*r(k)/20)).cos()+coeff[a][2*j+2]*(N[j]*(2*PI*r(k)/20)).sin() for j in range(3))
            if not IV:
                assert abs(one.v-direct.v)<1e-10
                assert max(abs(one.d-direct.d))<1e-9
            else:
                assert upper(abs(one.v-direct.v))<1e-40
                assert max(upper(abs(v)) for v in one.d-direct.d)<1e-40
        if k%50==0:print('Compiled sample',k,'elapsed',time.time()-start,flush=True)
    Rs=J1-JF;Rw=J2-JF
    # Frozen DC-constrained lag11 recurrence.
    R=zero((7,6))
    for j in range(1,7):
        den=sqrt(r(j*(j+1)))
        for k in range(j):R[k,j-1]=1/den
        R[j,j-1]=-r(j)/den
    Hlag=arr([[F1[a,k+11*j] for j in range(7)] for a in range(2) for k in range(123)])
    Jlag=Hlag@R;Dlag=leftinverse(Jlag,'lag11')
    roots=[cplx(r(1),r(0))]
    rotor=[]
    for num in [1,7,49]:
        ang=2*PI*r(num*11)/400;z=cplx(cos(ang),sin(ang));rotor.append(z);roots += [z,conj(z)]
    c=[cplx(r(1),r(0))]
    for z in roots:
        new=[cplx(r(0),r(0))]*(len(c)+1)
        for j,v in enumerate(c):new[j]-=z*v;new[j+1]+=v
        c=new
    c=[x.real for x in c]
    recurrence=arr([[sum(c[j]*Rs[a,k+11*j,q] for j in range(8)) for q in range(D)] for a in range(2) for k in range(123)])
    dc=-R@Dlag@recurrence
    rootsep=min(float(abs(z-w).a if IV else abs(z-w)) for i,z in enumerate(roots) for w in roots[i+1:])
    assert rootsep>0
    print('All lag roots distinct: min separation >',math.nextafter(rootsep,-math.inf),flush=True)
    dN=zero((3,D))
    for i,z in enumerate(rotor):
        ppval=sum(j*c[j]*(z**(j-1)) for j in range(1,8))
        assert float(abs(ppval).a if IV else abs(ppval))>0
        for q in range(D):dN[i,q]=r(20)/(2*PI*11)*(-sum(dc[j,q]*(z**j) for j in range(7))/ppval/z).imag
    V7=arr([[r(1)]+[v for num in [1,7,49] for v in (cos(2*PI*r(num*k)/400),sin(2*PI*r(num*k)/400))] for k in range(200)])
    L7=leftinverse(V7,'fundamental7')
    dcoef=[L7@(Rs[a]-J1[a,:,:3]@dN) for a in range(2)]
    # Exact complex quadratic plus confluent extraction.
    indices=[m for m in product(range(-2,3),repeat=3) if sum(abs(v) for v in m)<=2]
    nums=[m[0]+7*m[1]+49*m[2] for m in indices];target=indices.index((0,0,2))
    def node(num,k):return cplx(cos(2*PI*r(num*k)/400),sin(2*PI*r(num*k)/400))
    V31=np.array([[node(num,k) for num in nums]+[(r(k)/199)*node(num,k) for num in [1,-1,7,-7,49,-49]] for k in range(200)],dtype=object if IV else complex)
    ell=leftinverse(V31,'demixer31')[target]
    residualweak=[Rw[a]-J2[a,:,:3]@dN for a in range(2)]
    # U=(complex cosine coefficient - i complex sine coefficient)/2.
    U=cplx((coeff[0][5].v+coeff[1][6].v)/2,(coeff[1][5].v-coeff[0][6].v)/2)
    assert float(abs(U).a if IV else abs(U))>0
    Ival=cplx(I[0].v,I[1].v);dI=[]
    for q in range(D):
        dC=sum(ell[k]*cplx(residualweak[0][k,q],residualweak[1][k,q]) for k in range(200))
        dU=cplx((dcoef[0][5,q]+dcoef[1][6,q])/2,(dcoef[1][5,q]-dcoef[0][6,q])/2)
        dI.append((dC/(U*U)-2*Ival*dU/U).real)
    K=arr(list(dN)+list(dcoef[0])+list(dcoef[1])+[dI])
    # Same-point checks: fitted self harmonic and full left-inverse differential.
    Cfit=sum(ell[k]*cplx(F2[0,k],F2[1,k]) for k in range(200))
    Cexpected=Ival*U*U
    print('Self harmonic difference',Cfit-Cexpected,flush=True)
    if IV:assert upper(abs(Cfit-Cexpected))<1e-20
    rident=arr([[sum(c[j]*J1[a,k+11*j,q] for j in range(8)) for q in range(D)] for a in range(2) for k in range(123)])
    dcident=-R@Dlag@rident;dnident=zero((3,D))
    for i,z in enumerate(rotor):
        ppval=sum(j*c[j]*(z**(j-1)) for j in range(1,8))
        for q in range(D):dnident[i,q]=r(20)/(2*PI*11)*(-sum(dcident[j,q]*(z**j) for j in range(7))/ppval/z).imag
    cfident=[L7@(J1[a]-J1[a,:,:3]@dnident) for a in range(2)]
    wident=[J2[a]-J2[a,:,:3]@dnident for a in range(2)]
    diident=[]
    for q in range(D):
        dC=sum(ell[k]*cplx(wident[0][k,q],wident[1][k,q]) for k in range(200))
        dU=cplx((cfident[0][5,q]+cfident[1][6,q])/2,(cfident[1][5,q]-cfident[0][6,q])/2)
        diident.append((dC/(U*U)-2*Ival*dU/U).real)
    Kident=arr(list(dnident)+list(cfident[0])+list(cfident[1])+[diident])
    print('Preprocessing feature differential identity error',norminf(Kident-G),flush=True)
    if IV:assert norminf(Kident-G)<1e-20
    if not IV:
        assert abs(Cfit-Cexpected)<1e-10
        assert norminf(Kident-G)<1e-8
        print('A differential left identity error',np.linalg.norm(np.linalg.solve(G,Kident)-np.eye(D),np.inf),flush=True)
    RG0=np.linalg.inv(marray(G));RG=imat(RG0);E=imat(np.eye(D))-RG@G;edef=norminf(E)
    T0=RG@K
    if IV:
        assert edef<1e-4
        err=upper(r(edef)**3/(1-r(edef))*r(norminf(T0)))
        T0=(imat(np.eye(D))+E+E@E)@T0
        print('RECONSTRUCTION inverse defect',edef,'DT uniform entry error bound',err,flush=True)
        eps=mp.iv.mpf([-err,err]);TT=np.array([[x+eps for x in row] for row in T0],dtype=object)
    else:TT=np.linalg.solve(G,K);err=0
    trace=sum(TT[i,i] for i in range(D))
    print('TRACE enclosure',trace,flush=True)
    if IV:assert upper(trace)<-18
    Tmid=marray(TT);ev,vec=sla.eig(Tmid)
    print('DT infinity norm',norminf(TT),flush=True)
    print('eigenvalues',ev,flush=True)
    print('spectral radius proposal',max(abs(ev)),flush=True)
    # Real adapted eigenvector basis at this one point. Fixed binary proposals only.
    columns=[]
    for j,lam in enumerate(ev):
        if lam.imag>1e-8:columns += [vec[:,j].real,vec[:,j].imag]
        elif abs(lam.imag)<=1e-8:columns += [vec[:,j].real]
    S=np.column_stack(columns)
    if S.shape!=(18,18):print('No real eigenbasis formed',S.shape);sys.exit(1)
    Sinv=np.linalg.inv(S);adapted=imat(Sinv)@TT@imat(S)
    if IV:
        defect=norminf(imat(np.eye(D))-imat(Sinv)@imat(S));qadapt=upper(r(norminf(adapted))/(1-r(defect)))
    else:qadapt=norminf(adapted)
    print('ADAPTED REAL INFINITY NORM UPPER',qadapt,'basis condition',np.linalg.cond(S),flush=True)
    gersh=None
    if IV:
        # Sinv is a frozen binary proposal. Enclose the exact similarity, rather
        # than treating its small inversion defect as zero.
        Es=imat(np.eye(D))-imat(Sinv)@imat(S);sd=norminf(Es)
        se=upper(r(sd)/(1-r(sd))*r(norminf(adapted)))
        ep=mp.iv.mpf([-se,se]);Acert=np.array([[x+ep for x in row] for row in adapted],dtype=object)
        gersh=[]
        for i in range(D):
            cen=float(Acert[i,i].mid)
            rr=upper(abs(Acert[i,i]-r(cen))+sum(abs(Acert[i,j]) for j in range(D) if j!=i))
            gersh.append((cen,rr))
        print('Verified Gershgorin discs (real centers,radii):',gersh,flush=True)
        unstable=[]
        for i,(cen,rr) in enumerate(gersh):
            isolated=all(lower(abs(r(cen)-r(c2)))>upper(r(rr)+r(r2)) for j,(c2,r2) in enumerate(gersh) if j!=i)
            if isolated and lower(abs(r(cen))-r(rr))>1:
                unstable.append((i,lower(r(cen)-r(rr)),upper(r(cen)+r(rr))))
        print('CERTIFIED isolated real unstable eigenvalue intervals',unstable,flush=True)
        assert len(unstable)>=2
        realupper=max(upper(r(cen)+r(rr)) for cen,rr in gersh)
        print('CERTIFIED all eigenvalues Re(lambda) <=',realupper,flush=True)
        assert realupper<1
        # Native coordinate differential at the unchanged point.
        native=imat(np.eye(D));factor=r(180)/PI
        for j in range(3):native[3+j,3+j]=r(1)/10/sqrt(1-r(1)/100)*factor;native[6+j,6+j]=factor
        for j in range(3):
            native[9+j,:]=zero(D);nj=sqrt(nsq[j].v)
            native[9+j,9+j]=h[j].v/(nj*Q.v)
            for a in range(2):native[9+j,12+a]=t[a].v*(1-h[j].v*h[j].v)/(nj*Q.v*Q.v)
        for a in range(2):native[12+a,12+a]=factor/(1+t[a].v*t[a].v)
        native_basis=native@imat(S)
        print('NATIVE chart tangent conversion infinity norm <=',norminf(native),flush=True)
        print('NATIVE adapted basis infinity norm <=',norminf(native_basis),flush=True)

    out={'interval':IV,'frozen_left_inverse_proposals':LEFT_PROPOSALS,'frozen_reconstruction_inverse_proposal':RG0.tolist(),'eta_coordinates':['N1','N2','N3','e1/kappa','e2/kappa','e3/kappa','phi1','phi2','phi3','h1','h2','h3','tx','ty','bx','by','g','d'],'kappa':0.1,'DT_mid':Tmid.tolist(),'DT_entry_error_bound':err,'eigenvalues_proposal':[[z.real,z.imag] for z in ev],'spectral_radius_proposal':float(max(abs(ev))),'adapted_infinity_upper':qadapt,'basis':S.tolist(),'inverse_basis':Sinv.tolist(),'G_mid':marray(G).tolist(),'K_mid':marray(K).tolist(),'physical_I':[mid(I[0].v),mid(I[1].v)],'gershgorin_discs':gersh,'trace_interval':([math.nextafter(float(trace.a),-math.inf),upper(trace)] if IV else [float(trace),float(trace)]),'DT_interval_lower':([[math.nextafter(float(x.a),-math.inf) for x in row] for row in TT] if IV else None),'DT_interval_upper':([[upper(x) for x in row] for row in TT] if IV else None),'elapsed_seconds':time.time()-start}
    path=os.path.join(os.path.dirname(__file__),'finite_wedge_corrector_'+('interval' if IV else 'proposal')+'.json')
    with open(path,'w') as f:json.dump(out,f,indent=2)
    print('Saved',path,flush=True)
