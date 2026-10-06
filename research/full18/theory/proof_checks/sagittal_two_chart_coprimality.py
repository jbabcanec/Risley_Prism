"""One-witness, two-chart exact dyadic interval norm/resultant proof check.
No reconstruction experiment or parameter sweep. All arithmetic is outward.
Variable X=G-9/4; dropping the known exact zero constant is algebraic deflation.
"""
from fractions import Fraction as F
from math import isqrt
from itertools import permutations
import sys,time,json
BITS=int(sys.argv[1]) if len(sys.argv)>1 else 1024
S=1<<BITS
class I:
 __slots__=('lo','hi')
 def __init__(self,x=0,hi=None):
  if hi is not None:self.lo,self.hi=x,hi
  elif isinstance(x,I):self.lo,self.hi=x.lo,x.hi
  else:
   f=F(x);self.lo=f.numerator*S//f.denominator;self.hi=-((-f.numerator*S)//f.denominator)
 def __add__(self,o):
  o=I(o);return I(self.lo+o.lo,self.hi+o.hi)
 __radd__=__add__
 def __neg__(self):return I(-self.hi,-self.lo)
 def __sub__(self,o):return self+-I(o)
 def __rsub__(self,o):return I(o)+-self
 def __mul__(self,o):
  o=I(o);a=(self.lo*o.lo,self.lo*o.hi,self.hi*o.lo,self.hi*o.hi);return I(min(a)//S,-((-max(a))//S))
 __rmul__=__mul__
 def __truediv__(self,o):
  o=I(o);assert o.lo*o.hi>0,('division crosses zero',describe(o))
  # Each exact rational endpoint is rounded outward independently.
  lows=[];highs=[]
  for a in (self.lo,self.hi):
   for b in (o.lo,o.hi):
    num=a*S
    if b<0:num,b=-num,-b
    lows.append(num//b);highs.append(-((-num)//b))
  return I(min(lows),max(highs))
 def __rtruediv__(self,o):return I(o)/self
 def sqrt(self):
  assert self.lo>=0
  lo=isqrt(self.lo*S);h=self.hi*S;hi=isqrt(h);hi+=hi*hi<h
  return I(lo,hi)
 def contains0(self):return self.lo<=0<=self.hi
 def exact0(self):return self.lo==self.hi==0
 def ldexp(self,n):
  if n>=0:return I(self.lo<<n,self.hi<<n)
  a=1<<(-n);return I(self.lo//a,-((-self.hi)//a))
 def __repr__(self):return describe(self)
def describe(x):
 # Float formatting is display only, never used for proof.
 try:return f'[{float(F(x.lo,S)):.14g}, {float(F(x.hi,S)):.14g}]'
 except OverflowError:return f'[binary sizes {x.lo.bit_length()-BITS}, {x.hi.bit_length()-BITS}]'
def dot(a,b):return sum((x*y for x,y in zip(a,b)),I(0))
def cross(a,b):return a[0]*b[1]-a[1]*b[0]
def add(a,b):return [x+y for x,y in zip(a,b)]
def sub(a,b):return [x-y for x,y in zip(a,b)]
def scale(c,a):return [c*x for x in a]
def mv(A,x):return [dot(r,x) for r in A]
def optical(q,u):
 n=I(F(3,2));H=(n*n-dot(q,q)).sqrt();P=H-dot(u,q);D=1+dot(u,u)
 E=(P*P-D*(n*n-1)).sqrt();lam=(n*n-1)/(P+E);Z=H-lam;B=add(q,scale(lam,u));v=scale(1/Z,B);a=scale(1/H,q)
 A=[[I(int(i==j))+(a[i]-v[i])*u[j]/(1-dot(u,a)) for j in range(2)] for i in range(2)]
 return H,P,E,Z,B,v,a,A
def step(p,q,u,gap):
 H,P,E,Z,B,v,a,A=optical(q,u)
 h=H*(3+dot(u,p))/P
 guards=[H,P,E,Z,Z-dot(u,B),3+dot(u,p),3+gap-h]
 return add(mv(A,add(p,scale(3,a))),scale(gap,v)),B,A,v,guards

def padd(a,b,sgn=1):
 c=a[:]+[I(0)]*max(0,len(b)-len(a))
 for i,x in enumerate(b):c[i]=c[i]+(x if sgn==1 else -x)
 return c
def pscale(a,c):return [x*c for x in a]
def pmul(a,b):
 c=[I(0)]*(len(a)+len(b)-1)
 for i,x in enumerate(a):
  if x.exact0():continue
  for j,y in enumerate(b):
   if not y.exact0():c[i+j]=c[i+j]+x*y
 return c
def plinear(a,beta):return padd(pscale(a,beta),[I(0)]+a)
def amul(a,b,betas):
 """Multiply in I[X,h]/(h_j^2-X-beta_j), masks index h monomials."""
 c={}
 for m,p in a.items():
  for n,q in b.items():
   z=pmul(p,q);common=m&n
   while common:
    bit=common&-common;j=bit.bit_length()-1;z=plinear(z,betas[j]);common^=bit
   mask=m^n;c[mask]=padd(c.get(mask,[]),z)
 return c
def anormalize(a):
 maximum=max(max(abs(x.lo),abs(x.hi)) for p in a.values() for x in p)
 shift=BITS-maximum.bit_length()
 return {m:[x.ldexp(shift) for x in p] for m,p in a.items()},shift

def witness_rows():
 rotations=[(F(15,17),F(8,17)),(F(4,5),F(3,5)),(F(3,5),F(4,5))];rotor=[(F(1),F(0))]*3
 t=[I(F(1,3)),I(F(1,10))];q0=scale(1/(1+dot(t,t)).sqrt(),t);b=[I(1),I(2)];g=I(10);d=I(100)
 rows=[];alphas=[];guard=None
 for k in range(6):
  u=[list(map(I,(c/10,s/10))) for c,s in rotor]
  p=add(b,scale(6,t));q=q0[:];deriv=[[I(1),I(0)],[I(0),I(1)],[I(0),I(0)]]
  for j in range(2):
   p,q,A,v,checks=step(p,q,u[j],g);deriv=[mv(A,x) for x in deriv];deriv[2]=add(deriv[2],v)
   guard=min([x.lo for x in checks]+([guard] if guard is not None else []))
  y,B,A,v,checks=step(p,q,u[2],d);guard=min([x.lo for x in checks]+[guard])
  alpha=dot(q,q);H=(I(F(9,4))-alpha).sqrt();delta=cross(q,u[2]);r=cross(sub(y,p),u[2])
  # Add true geometry times first4 columns to final column. Exact sagittal
  # identity then makes final column r*(h-H), without floating cancellation.
  const=[-cross(x,q) for x in deriv]+[-delta,-H*r]
  linear=[-cross(x,u[2]) for x in deriv]+[I(0),r]
  rows.append((const,linear));alphas.append(alpha)
  rotor=[(c*a-s*bb,s*a+c*bb) for (c,s),(a,bb) in zip(rotor,rotations)]
 assert guard>0
 assert all(alphas[i].hi<alphas[j].lo or alphas[j].hi<alphas[i].lo for i in range(6) for j in range(i))
 return rows,alphas,guard

def determinant(rows):
 a={}
 for perm in permutations(range(5)):
  parity=sum(perm[i]>perm[j] for i in range(5) for j in range(i+1,5));p={0:I(-1 if parity%2 else 1)}
  for i,j in enumerate(perm):
   new={}
   for mask,value in p.items():
    new[mask]=new.get(mask,I(0))+value*rows[i][0][j]
    if j!=3:new[mask|1<<i]=new.get(mask|1<<i,I(0))+value*rows[i][1][j]
   p=new
  for mask,value in p.items():a[mask]=[a.get(mask,[I(0)])[0]+value]
 return a

def norm_chart(rows,alphas,chart):
 betas=[I(F(9,4))-alphas[k] for k in chart];a=determinant([rows[k] for k in chart]);a,shift=anormalize(a);scalings=[shift]
 print('chart',chart,'determinant terms',len(a),flush=True)
 for j in reversed(range(5)):
  even={m:p for m,p in a.items() if not m&(1<<j)};odd={m^(1<<j):p for m,p in a.items() if m&(1<<j)}
  ee=amul(even,even,betas);oo=amul(odd,odd,betas)
  for m,p in oo.items():ee[m]=padd(ee.get(m,[]),plinear(p,betas[j]),-1)
  a,shift=anormalize(ee);scalings.append(shift)
  print(' norm',j,'masks',len(a),'max Gdeg',max(len(p)-1 for p in a.values()),'scale',shift,flush=True)
 assert set(a)=={0};p=a[0];assert len(p)==65
 print(' leading coefficient',describe(p[-1]),'constant',describe(p[0]),'linear',describe(p[1]),flush=True)
 assert not p[-1].contains0(),'degree64 coefficient unresolved'
 assert p[0].contains0(),'known exact root failed interval containment'
 # The exact norm polynomial has constant0, proved from the physical identity.
 # Consequently these are intervals for EXACT quotient coefficients, not a
 # numerical polynomial with the constant artificially zeroed.
 return p[1:],{'chart':chart,'scalings':scalings,'leading':[str(p[-1].lo),str(p[-1].hi)],'linear':[str(p[1].lo),str(p[1].hi)],'constant':[str(p[0].lo),str(p[0].hi)]}

def resultant_certificate(a,b):
 """Interval Euclidean elimination; certified nonzero terminal constant proves
 coprimality. All pivots must exclude zero; all intermediate degrees retained.
 Every actual coefficient specialization in initial boxes follows same degrees.
 """
 records=[]
 while len(b)>1:
  assert not b[-1].contains0(),('uncertified divisor degree',len(b)-1,describe(b[-1]))
  r=a[:]
  for k in range(len(a)-len(b),-1,-1):
   q=r[k+len(b)-1]/b[-1]
   # Highest term is exactly canceled by polynomial division, so discard it.
   for j in range(len(b)-1):r[k+j]=r[k+j]-q*b[j]
   r.pop()
  # Generic remainders should have degree deg(b)-1; never trim merely because
  # an interval contains zero.
  if not r:raise AssertionError('exact zero remainder (structural gcd)')
  r0,shift=anormalize({0:r});r=r0[0]
  lead=r[-1]
  width=max(x.hi-x.lo for x in r)
  print(' remainder degree',len(r)-1,'leading',describe(lead),'max width log2',width.bit_length()-BITS,'scale',shift,flush=True)
  records.append({'degree':len(r)-1,'leading':[str(lead.lo),str(lead.hi)],'scaling':shift,'max_width_binary_exponent':width.bit_length()-BITS})
  assert not lead.contains0(),('remainder leading sign unresolved',len(r)-1)
  a,b=b,r
 assert not b[0].contains0()
 return records

if __name__=='__main__':
 started=time.time();rows,alphas,guard=witness_rows();print('precision',BITS,'six-sample guard',float(F(guard,S)),flush=True)
 p,meta1=norm_chart(rows,alphas,[0,1,2,3,4]);q,meta2=norm_chart(rows,alphas,[0,1,2,3,5]);remainder_records=resultant_certificate(p,q)
 result={'precision_bits':BITS,'six_sample_guard_lower_scaled':str(guard),'charts':[meta1,meta2],'euclidean_remainders':remainder_records,'elapsed_seconds':time.time()-started,'conclusion':'Both norms degree64; exact deflated degree63 quotients coprime.'}
 out='/workspace/shared/risley_theory/proof_checks/sagittal_two_chart_coprimality_certificate.json'
 with open(out,'w') as f:json.dump(result,f,indent=2)
 print('PASS',result['conclusion'],'time',result['elapsed_seconds'],'certificate',out,flush=True)
