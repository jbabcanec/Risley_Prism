"""Exact rational-interval audit of the triple-critical filtered-tail witness.
One specified algebraic limit, no parameter search or simulation sweep.
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

n=I(F(9,5));t=I(F(23,50));a=I(F(8,25));nu=n*n-1
z=1/(1+2*t*t).sqrt();q0=[t*z,t*z];q=q0[:]
H=(n*n-sum((v*v for v in q),I(0))).sqrt()
c=(H-((1+a*a)*nu).sqrt())/(q[0]*a)
assert c.lo>1 and c.hi<I(2).sqrt().lo
co=(c+(2-c*c).sqrt())/2;si=(c-(2-c*c).sqrt())/2
U1=[a*co,a*si]
assert si.lo>0 and co.lo>si.hi
sin18=(I(5).sqrt()-1)/4
tan18=(I(5).sqrt()-1)/(10+2*I(5).sqrt()).sqrt()
assert si.hi<sin18.lo and a.hi<tan18.lo
# tan(25deg)>=tan(22.5deg)+(pi/72)sec^2(22.5deg), pi>157/50.
t22=I(2).sqrt()-1
lower_t25=t22+I(F(157,3600))*(1+t22*t22)
assert t.hi<lower_t25.lo
p=[6*t,6*t];states=[]
def dot(x,y):return sum((a*b for a,b in zip(x,y)),I(0))
for j in range(3):
    H=(n*n-dot(q,q)).sqrt()
    if j==0:U=U1
    else:
        aj=z*z/(H*q[0]+(nu*(q[0]*q[0]+z*z)).sqrt())
        U=[aj,I(0)]
    D=1+dot(U,U);P=H-dot(U,q)
    A=3+dot(U,p)
    pe=[pi+qi*A/P for pi,qi in zip(p,q)]
    ell=2 if j<2 else 200
    B=ell-dot(U,pe)
    if j==0:
        r=[qi+ui*P/D for qi,ui in zip(q,U)];w=H-P/D
    else:
        rx=(1-q[1]*q[1]).sqrt()/(1+U[0]*U[0]).sqrt()
        r=[rx,q[1]];w=U[0]*rx
    pn=[pi+ri/w*B for pi,ri in zip(pe,r)]
    assert P.lo>0 and A.lo>0 and B.lo>0 and w.lo>0
    assert dot(U,U).sqrt().hi<tan18.lo
    states.append(dict(H=H,P=P,D=D,U=U,q=q[:],z=z,r=r,w=w,A=A,B=B,pe=pe))
    q,z,p=r,w,pn

s1,s2,s3=states
assert s2['U'][0].contains_between('0.032','0.033')
assert s3['U'][0].contains_between('0.00030','0.00032')
A1=2*s1['P']*q0[0]*(U1[0]-U1[1])
A2=2*s2['P']/s1['D']*(dot(s1['r'],U1)/s2['H']+dot(s2['U'],U1))
A3=2*s3['P']/s2['D']*(dot(s2['r'],s2['U'])/s3['H']+dot(s3['U'],s2['U']))
K=A3.sqrt()*A2.sqrt().sqrt()*A1.sqrt().sqrt().sqrt()
C=s3['B']*(s3['U'][0]*s3['w']+s3['r'][0])/(s3['D']*s3['w']*s3['w'])*K
assert A1.lo>0 and A2.lo>0 and A3.lo>0 and C.lo>0
assert s1['B'].lo>F('0.60') and s2['B'].lo>F('1.76') and s3['B'].lo>F('199.98')
assert p[0].contains_between('645433','645434')
assert p[1].contains_between('343242','343243')
for j,s in enumerate(states,1):
    print('critical stage',j,'tilt',[u.show() for u in s['U']],
          'outgoing z',s['w'].show(),'flight',s['B'].show())
print('cos(phi_c)',co.show(),'sin(phi_c)',si.show())
print('cascade A1,A2,A3',A1.show(),A2.show(),A3.show())
print('E3 coefficient K',K.show(),'screen coefficient C',C.show())
print('limiting screen',[v.show() for v in p])
print('Exact rational interval assertions passed.')
