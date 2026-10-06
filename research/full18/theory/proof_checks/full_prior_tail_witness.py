"""Exact rational interval audit of the critical-limit full-prior tail witness.
No sampling/parameter campaign: one explicitly specified algebraic proof point.
"""
from fractions import Fraction as F
from math import isqrt

PREC = 45
S = 10**PREC
class I:
    def __init__(self, lo, hi=None):
        if isinstance(lo,I): self.lo,self.hi=lo.lo,lo.hi;return
        self.lo=F(lo);self.hi=F(lo if hi is None else hi)
        assert self.lo<=self.hi
    def __add__(self,o):
        o=I(o);return I(self.lo+o.lo,self.hi+o.hi)
    __radd__=__add__
    def __neg__(self):return I(-self.hi,-self.lo)
    def __sub__(self,o):return self+-I(o)
    def __rsub__(self,o):return I(o)+-self
    def __mul__(self,o):
        o=I(o);v=[a*b for a in (self.lo,self.hi) for b in (o.lo,o.hi)];return I(min(v),max(v))
    __rmul__=__mul__
    def __truediv__(self,o):
        o=I(o);assert not(o.lo<=0<=o.hi);return self*I(1/o.hi,1/o.lo)
    def __rtruediv__(self,o):return I(o)/self
    def __pow__(self,n):
        assert n>=0
        ans=I(1)
        for _ in range(n):ans=ans*self
        return ans
    def sqrt(self):
        assert self.lo>=0
        a=isqrt(self.lo.numerator*S*S//self.lo.denominator)
        b=isqrt(self.hi.numerator*S*S//self.hi.denominator)
        return I(F(a,S),F(b+1,S))
    def contains_between(self,lo,hi):
        return F(lo)<self.lo and self.hi<F(hi)
    def show(self):return (float(self.lo),float(self.hi))

n=I(F(9,5));t=I(F(2,5));z=1/(1+t*t).sqrt();q=t*z
h=(n*n-q*q).sqrt()/z
p=6*t
us=[];checks=[]
for j in range(3):
    H=(n*n-q*q).sqrt()
    u=I(F(8,25)) if j==0 else z*z/(q*H+(n*n-1).sqrt())
    P=H-u*q
    if j==0:
        rad=P*P-(1+u*u)*(n*n-1)
        assert rad.lo>0
        E=rad.sqrt()
        zn=(H*(1+u*u)-P+E)/(1+u*u)
        qn=q+u*(P-E)/(1+u*u)
    else:
        # Exact critical-root identities established in the accompanying theorem.
        qn=1/(1+u*u).sqrt();zn=u*qn
    pe=p+q*(3+u*p)/P
    ell=2 if j<2 else 200
    flight=ell-u*pe
    assert P.lo>0 and (3+u*p).lo>0 and flight.lo>0 and zn.lo>0
    pn=pe+qn/zn*flight
    us.append(u);checks.append((u,P,zn,pe,flight))
    p,q,z=pn,qn,zn

B=t*(6+4+200+9/h)
F1_u=B;F1_e=B
for i,u in enumerate(us):
    D=200+(2-i)*2
    L=D+3*(2-i)/h;W=D+3*(2-i)/(h**3)
    M=(h-1)*(L+(W+L/h)*t*t-t*B/h)
    F1_u+=u*M
    F1_e+=u/(1+u*u).sqrt()*M

tan18=((I(5).sqrt()-1)/(10+2*I(5).sqrt()).sqrt())
assert all(u.hi<tan18.lo for u in us)
assert t.hi<(I(2).sqrt()-1).lo # beta <22.5deg<25deg
assert p.contains_between('17821','17822')
assert F1_u.contains_between('196','197')
assert F1_e.contains_between('192','193')
assert (p-F1_u).lo>17600 and (p-F1_e).lo>17600
for j,values in enumerate(checks,1):
    print('stage',j,dict(zip(('u','P','z','p_exit','flight'),(v.show() for v in values))))
print('baseline',B.show())
print('F',p.show(),'linear_u',F1_u.show(),'linear_e',F1_e.show())
print('tail_u',(p-F1_u).show(),'tail_e',(p-F1_e).show())
print('Exact rational interval assertions passed.')

class J:
    order=4
    def __init__(self,x):
        if isinstance(x,J): self.c=x.c[:];return
        self.c=[I(0) for _ in range(self.order+1)]
        if isinstance(x,list):
            for k,a in enumerate(x): self.c[k]=I(a)
        else:self.c[0]=I(x)
    def __add__(self,o):
        o=J(o);return J([a+b for a,b in zip(self.c,o.c)])
    __radd__=__add__
    def __neg__(self):return J([-a for a in self.c])
    def __sub__(self,o):return self+-J(o)
    def __rsub__(self,o):return J(o)+-self
    def __mul__(self,o):
        o=J(o);return J([sum((self.c[j]*o.c[k-j] for j in range(k+1)),I(0)) for k in range(self.order+1)])
    __rmul__=__mul__
    def inv(self):
        c=[1/self.c[0]]
        for k in range(1,self.order+1):c.append(-sum((self.c[j]*c[k-j] for j in range(1,k+1)),I(0))/self.c[0])
        return J(c)
    def __truediv__(self,o):return self*J(o).inv()
    def __rtruediv__(self,o):return J(o)*self.inv()
    def sqrt(self):
        c=[self.c[0].sqrt()]
        for k in range(1,self.order+1):c.append((self.c[k]-sum((c[j]*c[k-j] for j in range(1,k)),I(0)))/(2*c[0]))
        return J(c)

zj=J(1/(1+t*t).sqrt());qj=J(t)*zj;pj=J(6*t);nj=J(n)
for j in range(3):
    uj=J([I(0),us[j]])
    Hj=(nj*nj-qj*qj).sqrt()
    Pj=Hj-uj*qj
    Dj=1+uj*uj
    Ej=(Pj*Pj-Dj*(nj*nj-1)).sqrt()
    znext=(Hj*Dj-Pj+Ej)/Dj
    qnext=qj+uj*(Pj-Ej)/Dj
    pexit=pj+qj*(3+uj*pj)/Pj
    ell=2 if j<2 else 200
    pnext=pexit+(qnext/znext)*(ell-uj*pexit)
    qj,zj,pj=qnext,znext,pnext
for degree in range(1,5):
    pol=sum(pj.c[:degree+1],I(0));tail=p-pol
    print('Taylor degree',degree,'value',pol.show(),'tail',tail.show())
    bounds={1:('196','197'),2:('240','241'),3:('278','279'),4:('308','309')}
    assert pol.contains_between(*bounds[degree])
    assert tail.lo>17000
print('Every degree 1 through 4 uniform absolute tail exceeds 17,000.')

# The native sine-coordinate jets used in the existing oblique derivation.
ez=J(1/(1+t*t).sqrt());eq=J(t)*ez;ep=J(6*t)
for j in range(3):
    e=us[j]/(1+us[j]*us[j]).sqrt()
    eu=J([I(0),e,I(0),(e**3)/2,I(0)])
    eh=(nj*nj-eq*eq).sqrt();eP=eh-eu*eq;eD=1+eu*eu
    eE=(eP*eP-eD*(nj*nj-1)).sqrt()
    ezn=(eh*eD-eP+eE)/eD;eqn=eq+eu*(eP-eE)/eD
    epe=ep+eq*(3+eu*ep)/eP
    ep=epe+(eqn/ezn)*((2 if j<2 else 200)-eu*epe)
    eq,ez=eqn,ezn
for degree in range(1,5):
    pol=sum(ep.c[:degree+1],I(0));tail=p-pol
    print('Sine Taylor degree',degree,'value',pol.show(),'tail',tail.show())
    assert tail.lo>17000
print('Every sine-coordinate degree 1 through 4 uniform absolute tail also exceeds 17,000.')
