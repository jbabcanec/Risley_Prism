"""Independent dyadic enclosure and two-segment check of one sagittal witness.
Uses no source-transport matrices: affine position derivatives are exact one-unit
position differences. This is one proof witness, not a reconstruction experiment.
"""
from fractions import Fraction as F
from math import isqrt
from itertools import permutations
PREC = 280
SCALE = 2**PREC

def floor_rat(q): return q.numerator // q.denominator
def ceil_rat(q): return -floor_rat(-q)
def down(q): return F(floor_rat(q*SCALE), SCALE)
def up(q): return F(ceil_rat(q*SCALE), SCALE)
class Enclosure:
    def __init__(self, low=0, high=None):
        if isinstance(low, Enclosure):
            self.low, self.high = low.low, low.high
        else:
            self.low, self.high = down(F(low)), up(F(low if high is None else high))
    def __add__(self, other):
        other=Enclosure(other)
        return Enclosure(self.low+other.low,self.high+other.high)
    __radd__=__add__
    def __neg__(self): return Enclosure(-self.high,-self.low)
    def __sub__(self, other): return self+-Enclosure(other)
    def __rsub__(self, other): return Enclosure(other)+-self
    def __mul__(self, other):
        other=Enclosure(other)
        choices=[x*y for x in (self.low,self.high) for y in (other.low,other.high)]
        return Enclosure(min(choices),max(choices))
    __rmul__=__mul__
    def reciprocal(self):
        assert self.high<0 or self.low>0
        return Enclosure(1/self.high,1/self.low)
    def __truediv__(self, other): return self*Enclosure(other).reciprocal()
    def __rtruediv__(self, other): return Enclosure(other)*self.reciprocal()
    def root(self):
        assert self.low>=0
        # floor(sqrt(x)*SCALE)=isqrt(floor(x*SCALE**2)).
        a=isqrt(floor_rat(self.low*SCALE*SCALE))
        b=isqrt(floor_rat(self.high*SCALE*SCALE))
        if F(b*b,SCALE*SCALE)<self.high: b+=1
        return Enclosure(F(a,SCALE),F(b,SCALE))
    def contains_zero(self): return self.low<=0<=self.high
    def describe(self): return f'[{float(self.low):.17g}, {float(self.high):.17g}]'
E=Enclosure

def dot(v,w): return sum((a*b for a,b in zip(v,w)),E(0))
def cross(v,w): return v[0]*w[1]-v[1]*w[0]
def plus(v,w): return [a+b for a,b in zip(v,w)]
def minus(v,w): return [a-b for a,b in zip(v,w)]
def scale(a,v): return [a*x for x in v]

def direction(q,u):
    n=E(F(3,2))
    H=(n*n-dot(q,q)).root()
    P=H-dot(u,q)
    D=1+dot(u,u)
    R=(P*P-D*(n*n-1)).root()
    # Cancellation-free root of D*lambda**2-2*P*lambda+n**2-1=0.
    lam=(n*n-1)/(P+R)
    B=plus(q,scale(lam,u))
    Z=H-lam
    assert (dot(B,B)+Z*Z-1).contains_zero()
    assert (Z-dot(u,B)-R).contains_zero()
    return q,u,H,P,R,B,Z

def transport(p, optical, gap):
    q,u,H,P,R,B,Z=optical
    numerator=3+dot(u,p)
    internal_parameter=numerator/P
    exit_position=plus(p,scale(internal_parameter,q))
    internal_axial=H*internal_parameter
    remaining=3+gap-internal_axial
    output=plus(exit_position,scale(remaining/Z,B))
    return output,[H,P,R,Z,numerator,remaining]

def upstream(optics,b,g):
    p=plus(b,scale(6,t))
    for optical in optics[:2]: p,_=transport(p,optical,g)
    return p

rotations=[(F(15,17),F(8,17)),(F(4,5),F(3,5)),(F(3,5),F(4,5))]
rotor=[(F(1),F(0))]*3
t=[E(F(1,3)),E(F(1,10))]
q0=scale(1/(1+dot(t,t)).root(),t)
b=[E(1),E(2)]; g=E(10); d=E(100)
rows=[]; radial_values=[]; min_guard=None
for k in range(200):
    q=q0[:]; optics=[]
    for cosine,sine in rotor:
        u=[E(cosine/10),E(sine/10)]
        optical=direction(q,u)
        optics.append(optical)
        q=optical[5]
    p=plus(b,scale(6,t))
    for j, optical in enumerate(optics):
        if j==2: last_entrance=p
        p,guards=transport(p,optical,g if j<2 else d)
        lower=min(x.low for x in guards)
        min_guard=lower if min_guard is None else min(min_guard,lower)
    y=p
    q,u,H,P,R,B,Z=optics[2]
    v=plus(q,scale(H,u)); delta=cross(q,u)
    residual=cross(minus(y,last_entrance),v)-(3+d)*delta
    assert residual.contains_zero()
    if k<5:
        # Exact derivatives, since the two-prism positional map is affine in b,g.
        dx=minus(upstream(optics,[b[0]+1,b[1]],g),last_entrance)
        dy=minus(upstream(optics,[b[0],b[1]+1],g),last_entrance)
        dg=minus(upstream(optics,b,g+1),last_entrance)
        rows.append([-cross(dx,v),-cross(dy,v),-cross(dg,v),-delta,
                     cross(minus(y,last_entrance),u)/(2*H)])
        radial_values.append(dot(q,q))
    rotor=[(c*a-s*bb,s*a+c*bb) for (c,s),(a,bb) in zip(rotor,rotations)]
result=E(0)
for order in permutations(range(5)):
    parity=sum(order[i]>order[j] for i in range(5) for j in range(i+1,5))
    term=E(-1 if parity%2 else 1)
    for i in range(5): term=term*rows[i][order[i]]
    result=result+term
assert result.high<0
assert result.low>F(-7061916467402617,10**24)
assert result.high<F(-7061916467402614,10**24)
assert min_guard>F(824,1000)
assert all(radial_values[i].high<radial_values[j].low or radial_values[j].high<radial_values[i].low
           for i in range(5) for j in range(i))
print('Independent determinant enclosure:',result.describe())
print('Exact endpoints:',int(result.low*SCALE),int(result.high*SCALE),'over 2**',PREC)
print('All-200 minimum guard lower bound:',float(min_guard))
print('Five incoming squared norms:',[x.describe() for x in radial_values])
print('PASS: strictly negative determinant, pairwise-distinct radicands, all 200 physical samples.')
