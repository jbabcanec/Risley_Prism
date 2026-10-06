#!/usr/bin/env python3
"""Bounded interval proof check at the existing rational witness.

Certifies only two linear-algebra ingredients, independent of wedge scale:
(1) a left inverse of the normalized, lag-11 DC-constrained Hankel map;
(2) a low-l1 exact last-self-harmonic extraction row.
It does not certify a nonlinear finite-wedge basin or perform a sweep.
Double precision proposes matrices; mpmath interval arithmetic verifies them.
"""
import math
from itertools import product
import numpy as np
import mpmath as mp
iv=mp.iv
iv.dps=50

def upper(x):
    return math.nextafter(float(x.b), math.inf)
def ivc(z):
    return iv.mpc(iv.mpf(float(np.real(z))),iv.mpf(float(np.imag(z))))
def row_abs_sum(row):
    return sum((abs(z) for z in row),iv.mpf(0))
def norminf(A):
    return max(upper(row_abs_sum(row)) for row in A)
def cert_left_inverse(P0,A):
    """P0 is an exactly represented floating proposal; A is interval matrix."""
    n,m=P0.shape;P=[[ivc(P0[i,k]) for k in range(m)] for i in range(n)]
    E=[]
    for i in range(n):
        row=[]
        for j in range(n):
            value=sum((P[i][k]*A[k][j] for k in range(m)),iv.mpc(0))
            row.append((1 if i==j else 0)-value)
        E.append(row)
    return P,E,norminf(E),norminf(P)

# Exact rational first-order witness, interval complex roots of unity.
z=[iv.mpc(iv.cos(2*iv.pi*j/400),iv.sin(2*iv.pi*j/400)) for j in range(400)]
h=[iv.mpf(3)/2,iv.mpf(7)/5,iv.mpf(5)/3]
t=[iv.mpf(1)/3,iv.mpf(0)];B=[iv.mpf(1411)/35,iv.mpf(2)]
L=[100+6+3/h[1]+3/h[2],100+3+3/h[2],iv.mpf(100)]
W=[100+6+3/h[1]**3+3/h[2]**3,100+3+3/h[2]**3,iv.mpf(100)]
M=[]
for j in range(3):
    M.append([[(h[j]-1)*(L[j]*(1 if a==b else 0)+(W[j]+L[j]/h[j])*t[a]*t[b]-t[a]*B[b]/h[j]) for b in range(2)] for a in range(2)])
F=[[sum((M[j][a][0]*z[(num*k)%400].real+M[j][a][1]*z[(num*k)%400].imag for j,num in enumerate([1,7,49])),iv.mpf(0)) for k in range(200)] for a in range(2)]
R=[[iv.mpf(0) for _ in range(6)] for _ in range(7)]
for j in range(1,7):
    den=iv.sqrt(j*(j+1))
    for k in range(j):R[k][j-1]=1/den
    R[j][j-1]=-j/den
lag=11;K=200-7*lag
J=[]
for axis in range(2):
    for k in range(K):
        J.append([sum((F[axis][k+j*lag]*R[j][a] for j in range(7)),iv.mpf(0)) for a in range(6)])
Jmid=np.array([[float(x.mid) for x in row] for row in J])
P0=np.linalg.pinv(Jmid)
P,E,d,Kp=cert_left_inverse(P0,J)
Efrob=upper(iv.sqrt(sum((abs(x)**2 for row in E for x in row),iv.mpf(0))))
Pfrob=upper(iv.sqrt(sum((abs(x)**2 for row in P for x in row),iv.mpf(0))))
assert d<1e-10 and Efrob<1e-10
corrected_inf=upper(iv.mpf(Kp)/(1-iv.mpf(d)))
corrected_two=upper(iv.mpf(Pfrob)/(1-iv.mpf(Efrob)))
assert corrected_inf<0.367
assert corrected_two<0.049
print('LAG11 VERIFIED: ||left inverse||_inf < 0.367; sigma_min(J0) > 1/0.049 > 20.4')
print('  defect_inf upper =',d,'proposal_linf upper =',Kp,'proposal_Frobenius upper =',Pfrob,flush=True)

# Exact quadratic/confluent design; scale confluent columns by k/199.
indices=[m for m in product(range(-2,3),repeat=3) if sum(abs(v) for v in m)<=2]
nums=[m[0]+7*m[1]+49*m[2] for m in indices]
target=indices.index((0,0,2))
V=[]
for k in range(200):
    V.append([z[(num*k)%400] for num in nums]+[(iv.mpf(k)/199)*z[(num*k)%400] for num in [1,-1,7,-7,49,-49]])
Vmid=np.array([[complex(float(x.real.mid),float(x.imag.mid)) for x in row] for row in V])
P0=np.linalg.pinv(Vmid)
P,E,d,Kp=cert_left_inverse(P0,V)
tau=upper(row_abs_sum(E[target])); l0=upper(row_abs_sum(P[target]))
assert d<1e-7
ell_bound=upper(iv.mpf(l0)+iv.mpf(tau)*iv.mpf(Kp)/(1-iv.mpf(d)))
assert ell_bound<1.002
print('DEMIXER VERIFIED: an exact target row has l1 norm < 1.002')
print('  full_left_inverse_defect upper =',d,'target_defect_l1 upper =',tau,'proposal_target_l1 upper =',l0,'corrected_target_l1 upper =',ell_bound)
print('All bounded interval assertions passed. No finite-wedge convergence radius claimed.')
