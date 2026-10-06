import sympy as S
x,y,H,d=S.symbols('x y H d',real=True)
T=x+S.I*y;Tb=x-S.I*y;B=S.Rational(1411,35)+2*S.I;Bb=S.conjugate(B);R=x*x+y*y;k=H-1
A=2*H*d+d*(H+1)*R-T*Bb
V=d*(H+1)*T*T-T*B
Cn=4*d*H**3*Tb+d*H**2*(3*H-1)*R*Tb-4*H**2*(Bb-d*Tb)-2*(H**2+1)*R*(Bb-d*Tb)
w=V/S.conjugate(A);I=Cn/(2*k*A*A)
pt={x:S.Rational(1,3),y:0,H:S.Rational(5,3),d:100}
vars=[x,y,H,d]
J=S.Matrix([[S.re(S.diff(w,v).subs(pt)) for v in vars],[S.im(S.diff(w,v).subs(pt)) for v in vars],[S.re(S.diff(I,v).subs(pt)) for v in vars],[S.im(S.diff(I,v).subs(pt)) for v in vars]])
J=J.applyfunc(S.factor);det=S.factor(J.det())
print('det=',det);print('rank=',J.rank());print('w=',S.factor(w.subs(pt)));print('I=',S.factor(I.subs(pt)))
if det==0:print('kernel=',J.nullspace())
