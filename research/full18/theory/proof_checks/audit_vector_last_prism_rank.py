import sympy as S
x,y,H,d=S.symbols('x y H d',real=True)
T=x+S.I*y;Tb=x-S.I*y;B=S.Rational(1411,35)+2*S.I;Bb=S.conjugate(B);a=Tb/2;c0=(Bb-d*Tb)/2;k=H-1
# Independently audited direct formal Snell/transport coefficients.
U=k*(d*(1+a*T)-T*c0/H)
V=k*(d*(1+1/H)*T*T-T*B/H)/2
C=k*(d*H*a+d*(3*H-1)*a*a*T/2-c0*(1+a*T+T*a/H**2))
w=V/S.conjugate(U);I=C/U**2
pt={x:S.Rational(1,3),y:0,H:S.Rational(5,3),d:100}
rows=[]
for f in [w,I]:
 dv=[S.factor(S.diff(f,v).subs(pt)) for v in [x,y,H,d]]
 rows += [[S.re(z) for z in dv],[S.im(z) for z in dv]]
det=S.factor(S.Matrix(rows).det());expected=S.Rational(19617689107775384734706249006250000,141672457738889533504325313445087046934090001)
print('independent determinant:',det);print('matches:',det==expected)
R=x*x+y*y;A=2*H*d+d*(H+1)*R-T*Bb;V0=d*(H+1)*T*T-T*B
C0=4*d*H**3*Tb+d*H**2*(3*H-1)*R*Tb-4*H**2*(Bb-d*Tb)-2*(H**2+1)*R*(Bb-d*Tb)
print('formula residuals:',*[S.factor(z) for z in [U-k*A/(2*H),V-k*V0/(2*H),C-k*C0/(8*H**2)]])
