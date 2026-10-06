"""Exact arithmetic audit of the reduced compiler and matched-jet constants.
This is a proof check at the specified construction, not a numerical sweep.
"""
from fractions import Fraction as F
from math import prod

# Five-root combined-position degree recurrence.
a, b = 1, 0
for j in range(1, 4):
    H, Xprev, Xnext, P, Z = 2*j-1, 2*j-1, 2*j+1, 2*j, 2*j+1
    terms = (Z+P+a, max(Z+Xprev, H+Xnext)+1+a,
             max(Z+Xprev, 1+Xprev+Xnext)+b, 1+P+Xnext+b)
    a, b = max(terms), Z+P+b
    assert a == b+1
assert (a,b)==(28,27)
assert a*2**5 == 896
assert 896*(6*200+1) == 1076096

# Exact interlaced weights and their cancellation moments.
r = [F(2*l+1,2)+F(l*l,1000) for l in range(6)]
D = prod(r[j]-r[0] for j in range(1,6))
w = [D/abs(prod(r[l]-r[m] for m in range(6) if m!=l)) for l in range(6)]
assert w == [F(1), F(2505,503), F(5025020,506521), F(119405,12084),
             F(2515027515,511079689), F(6012113393,6132956268)]
assert all(F(9,10)<v<11 for v in w)
assert all(sum((-1)**l*w[l]*r[l]**m for l in range(6))==0 for m in range(5))
assert sum((-1)**l*w[l]*r[l]**5 for l in range(6)) == -D
assert D/120 == F(25377130631853,25000000000000) < F(51,50)

# h=.003, exactly centered original-clock phases and noise threshold.
h, tc = F(3,1000), F(199,40)
assert 360*h*(r[-1]-r[0])/2*tc == F(1079973,80000)
assert F(1079973,80000)+4 < 18
assert max(h*(r[2*j+1]-r[2*j]) for j in range(3))/2 == F(3027,2000000)
assert (F(597,20000)*F(22,7))**5 < F(73,10000000)
assert 4*h*max(r) < 20

# Conservative all-angle physical guard, proved by rational inequalities.
zmax, umax, qmax = F(111,50000), F(1,400), F(1,50)
assert zmax*zmax < umax*umax*(1-zmax*zmax)
Hlo = F(7499,5000) # 1.4998 < sqrt(9/4 - .02^2)
assert Hlo*Hlo < F(9,4)-qmax*qmax
Plo = Hlo-umax*qmax
assert Plo > F(149,100)
assert Plo*Plo/(1+umax*umax)-F(5,4) > F(99,100)**2
assert 3*F(3,2)*umax < qmax
small = umax*qmax/Hlo
assert (3-2*umax)/(1+small) > F(297,100)
assert (3+2*umax)/(1-small) < F(303,100)
assert F(303,100)*qmax/F(149,100)+F(503,100)*qmax/F(99,100) < F(15,100)
print('All exact rational audit assertions passed.')
