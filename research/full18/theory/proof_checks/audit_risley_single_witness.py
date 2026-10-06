import sympy as S
from math import factorial
D=5; Z=(0,0,0)
class J:
 def __init__(self,v,d=None): self.v=S.Rational(v);self.d=tuple(map(S.Rational,d or [0]*D))
 def __add__(a,b):
  b=jj(b);return J(a.v+b.v,[x+y for x,y in zip(a.d,b.d)])
 __radd__=__add__
 def __neg__(a):return J(-a.v,[-x for x in a.d])
 def __sub__(a,b):return a+-jj(b)
 def __rsub__(a,b):return jj(b)+-a
 def __mul__(a,b):
  b=jj(b);return J(a.v*b.v,[x*b.v+a.v*y for x,y in zip(a.d,b.d)])
 __rmul__=__mul__
 def __pow__(a,r):
  r=S.Rational(r);v=a.v**r;return J(v,[r*a.v**(r-1)*x for x in a.d])
 def __truediv__(a,b):return a*jj(b)**-1
 def __rtruediv__(a,b):return jj(b)*a**-1
 def nz(a):return a.v!=0 or any(a.d)
def jj(a):return a if isinstance(a,J) else J(a)
class P:
 def __init__(self,v):self.c=v if isinstance(v,dict) else {Z:jj(v)}
 def __add__(a,b):
  b=pp(b); ks=set(a.c)|set(b.c);return P({k:a.c.get(k,J(0))+b.c.get(k,J(0)) for k in ks})
 __radd__=__add__
 def __neg__(a):return P({k:-v for k,v in a.c.items()})
 def __sub__(a,b):return a+-pp(b)
 def __rsub__(a,b):return pp(b)+-a
 def __mul__(a,b):
  b=pp(b);o={}
  for k,x in a.c.items():
   for l,y in b.c.items():
    m=tuple(i+j for i,j in zip(k,l))
    if sum(m)<=3:o[m]=o.get(m,J(0))+x*y
  return P({k:v for k,v in o.items() if v.nz()})
 __rmul__=__mul__
 def __pow__(a,r):
  r=S.Rational(r)
  if r in [0,1,2,3]:
   out=P(1)
   for _ in range(int(r)):out=out*a
   return out
  c=a.c.get(Z,J(0));q=(a-P(c))*P(c**-1);out=P(1);pw=P(1)
  for j in range(1,4):pw=pw*q;out=out+P(S.binomial(r,j))*pw
  return P(c**r)*out
 def __truediv__(a,b):return a*pp(b)**-1
 def __rtruediv__(a,b):return pp(b)*a**-1
 def coeff(a,m):return a.c.get(m,J(0))
def pp(a):return a if isinstance(a,P) else P(a)
vals=[S.Rational(3,2),S.Rational(4,3),S.Rational(5,3),S.Integer(3),S.Integer(100)]
params=[J(v,[int(i==j) for i in range(D)]) for j,v in enumerate(vals)]
x=P(0);p=P(0)
for i in range(3):
 mon=tuple(int(k==i) for k in range(3));s=P({mon:J(1)});c=(1-s*s)**S.Rational(1,2);n=P(params[i]);ell=P(params[3 if i<2 else 4]);H=(n*n-x*x)**S.Rational(1,2);Q=c*x+s*H;R=(1-Q*Q)**S.Rational(1,2);b=c*Q-s*R;z=s*Q+c*R;den=c*H-s*x
 p=R*H/(z*den)*p+3*x*R/(z*den)+ell*b/z
 x=b
K=[p.coeff(tuple(int(k==i) for k in range(3))) for i in range(3)]
mons=[(3,0,0),(0,3,0),(0,0,3),(2,1,0),(0,2,1)]
rows=[]
for m in mons:
 den=J(1)
 for i in range(3):den=den*K[i]**m[i]
 rows.append((p.coeff(m)/den).d)
M=S.Matrix(rows);det=S.factor(M.det())
print('Independent exact pair trace Taylor, single point:',vals)
print('K values:',[v.v for v in K]);print('det:',det)
expect=-S.Rational(2342246801318629443,280149096378993020072641029913075712000)
print('Matches reported:',det==expect)
