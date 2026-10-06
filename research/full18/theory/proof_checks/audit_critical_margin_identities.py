"""Narrow exact symbolic checks of the data-dependent critical-margin identities."""
import sympy as s
Z,R,d,B,mu,C,Ustar = s.symbols('Z R d B mu C Ustar')
Cmu=50-53*mu
# Eq. 8 using the exact critical-plane relation Z*u.F=Z*d-R*B.
left=Z*(d-Cmu)-R*B
right=(1-mu)*Z*(d-50)+mu*Z*(3+d-B)+(mu*Z-R)*B
assert s.expand(left-right)==0
# Eq. 9, polynomial identity with no physical assumptions needed for equality.
ux,uy,Fx,Fy,Ax,Ay=s.symbols('ux uy Fx Fy Ax Ay')
F2=Fx**2+Fy**2
uf=ux*Fx+uy*Fy
right=Ustar**2*((Ax-Fx)*(Ax+Fx)+(Ay-Fy)*(Ay+Fy))
right+=(Ustar**2-ux**2-uy**2)*F2+(ux*Fy-uy*Fx)**2+(uf-C)*(uf+C)
assert s.expand(Ustar**2*(Ax**2+Ay**2)-C**2-right)==0
# Eq. 15: compose reverse and forward spatial steps, clearing positive denominators.
x,y,X,Y,U,V,H,ell=s.symbols('x y X Y U V H ell')
p=s.Matrix([x,y]); Xin=s.Matrix([X,Y]); Xout=s.Matrix([U,V]); u=s.Matrix([ux,uy])
P=H-(u.dot(Xin)); Rexpr=Z-u.dot(Xout)
N=Z*P*p+(Z*Xin-H*Xout)*u.dot(p)+3*(Z*Xin-u.dot(Xin)*Xout)+ell*P*Xout
K=H*Rexpr*s.eye(2)-(Z*Xin-H*Xout)*u.T
Vexpr=H*Xout-Xin*u.dot(Xout)
error=K*N-(ell*Vexpr+3*Xin*Rexpr)*Z*P-H*Rexpr*Z*P*p
assert all(s.expand(component)==0 for component in error)
# Eq. 7: strict disk threshold root.
a,b,S,eps=s.symbols('a b S eps')
root=(s.sqrt(2*S*S-(a-b)**2)-(a+b))/2
assert s.simplify(2*root**2+2*(a+b)*root+a*a+b*b-S*S)==0
print('Critical-margin identities (8), (9), (15), and epsilon root (7) verified exactly.')
