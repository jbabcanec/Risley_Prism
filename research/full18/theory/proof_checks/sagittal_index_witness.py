"""Exact integer interval proof check at one rational-rotor witness; no sweep."""
from fractions import Fraction
from math import isqrt
from itertools import permutations
BITS=240
S=1<<BITS
class I:
 def __init__(self,x=0,hi=None):
  if hi is not None: self.lo,self.hi=int(x),int(hi)
  elif isinstance(x,I): self.lo,self.hi=x.lo,x.hi
  else:
   f=Fraction(x); self.lo=f.numerator*S//f.denominator; self.hi=-((-f.numerator*S)//f.denominator)
 def __add__(self,o):
  o=I(o); return I(self.lo+o.lo,self.hi+o.hi)
 __radd__=__add__
 def __neg__(self): return I(-self.hi,-self.lo)
 def __sub__(self,o): return self+-I(o)
 def __rsub__(self,o): return I(o)+-self
 def __mul__(self,o):
  o=I(o); a=[self.lo*o.lo,self.lo*o.hi,self.hi*o.lo,self.hi*o.hi]; return I(min(a)//S,-((-max(a))//S))
 __rmul__=__mul__
 def __truediv__(self,o):
  o=I(o); assert o.lo*o.hi>0, ('zero denominator',o)
  a=[Fraction(x*S,y) for x in (self.lo,self.hi) for y in(o.lo,o.hi)]
  lo,hi=min(a),max(a); return I(lo.numerator//lo.denominator,-((-hi.numerator)//hi.denominator))
 def __rtruediv__(self,o): return I(o)/self
 def sqrt(self):
  assert self.lo>=0
  lo=isqrt(self.lo*S); h=self.hi*S; hi=isqrt(h); hi+=hi*hi<h; return I(lo,hi)
 def __repr__(self): return f'[{float(Fraction(self.lo,S)):.17g},{float(Fraction(self.hi,S)):.17g}]'
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def det(a,b): return a[0]*b[1]-a[1]*b[0]
def add(a,b): return [x+y for x,y in zip(a,b)]
def sub(a,b): return [x-y for x,y in zip(a,b)]
def smul(c,a): return [c*x for x in a]
def matvec(A,x): return [dot(r,x) for r in A]
def refract(q,u,n):
 H=(n*n-dot(q,q)).sqrt(); P=H-dot(u,q); D=1+dot(u,u)
 E=(P*P-D*(n*n-1)).sqrt(); Z=(H*D-P+E)/D
 B=add(q,smul(H-Z,u)); R=Z-dot(u,B)
 a=smul(1/H,q); v=smul(1/Z,B)
 A=[[I(int(i==j))+(a[i]-v[i])*u[j]/(1-dot(u,a)) for j in range(2)] for i in range(2)]
 return H,P,E,Z,B,R,a,v,A
def step(p,q,u,n,ell):
 H,P,E,Z,B,R,a,v,A=refract(q,u,n)
 h=H*(3+dot(u,p))/P
 nex=add(matvec(A,add(p,smul(3,a))),smul(ell,v))
 checks=[H,P,E,Z,R,3+dot(u,p),3+ell-h]
 return nex,B,A,v,checks
rots=[(Fraction(15,17),Fraction(8,17)),(Fraction(4,5),Fraction(3,5)),(Fraction(3,5),Fraction(4,5))]
z=[(Fraction(1),Fraction(0)) for _ in range(3)]
t=[I(Fraction(1,3)),I(Fraction(1,10))]; Q=1+dot(t,t); z0=1/Q.sqrt(); q0=smul(z0,t)
n=I(Fraction(3,2)); g=I(10); d=I(100); b=[I(1),I(2)]
rows=[]; cs=[]; guard=None
for k in range(200):
 u=[list(map(I,(Fraction(1,10)*c,Fraction(1,10)*s))) for c,s in z]
 p=add(b,smul(6,t)); q=q0[:]
 deriv=[[I(1),I(0)],[I(0),I(1)],[I(0),I(0)]]
 for j in range(2):
  p,q,A,v,checks=step(p,q,u[j],n,g)
  deriv=[matvec(A,x) for x in deriv]; deriv[2]=add(deriv[2],v)
  guard=min([x.lo for x in checks]+([guard] if guard is not None else []))
 y,B,A,v,checks=step(p,q,u[2],n,d)
 guard=min([x.lo for x in checks]+[guard])
 H=(n*n-dot(q,q)).sqrt(); l=add(q,smul(H,u[2])); C=det(q,u[2])
 identity=det(sub(y,p),l)-(3+d)*C
 assert identity.lo<=0<=identity.hi
 if k<5:
  row=[-det(x,l) for x in deriv]+[-C,det(sub(y,p),u[2])/(2*H)]
  rows.append(row); cs.append(dot(q,q))
 z=[(c*a-s*bb,s*a+c*bb) for (c,s),(a,bb) in zip(z,rots)]
D=I(0)
for perm in permutations(range(5)):
 inv=sum(perm[i]>perm[j] for i in range(5) for j in range(i+1,5)); term=I((-1)**inv)
 for i in range(5):term*=rows[i][perm[i]]
 D+=term
print('Interval determinant:',D)
print('Determinant strict sign:', 'positive' if D.lo>0 else 'negative' if D.hi<0 else 'UNRESOLVED')
print('Certified scaled integer endpoints:',D.lo,D.hi,'denominator 2^',BITS)
print('Minimum physical guard lower bound:',float(Fraction(guard,S)))
print('Distinct incoming squared transverse momenta:',cs)
assert D.lo>0 or D.hi<0
assert guard>0
assert all(cs[i].hi<cs[j].lo or cs[j].hi<cs[i].lo for i in range(5) for j in range(i))
