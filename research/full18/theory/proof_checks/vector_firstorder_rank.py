import sympy as S
T,h1,h2,h3,g,d=S.symbols('T h1 h2 h3 g d')
h=[h1,h2,h3]; L=[d+2*g+3/h2+3/h3,d+g+3/h3,d]; W=[d+2*g+3/h2**3+3/h3**3,d+g+3/h3**3,d]
vars=[T,h1,h2,h3,g,d]
f=[z for i in range(3) for z in (T*T*(W[i]/L[i]+1/h[i]),T/(h[i]*L[i]))]
pt={T:S.Rational(1,3),h1:S.Rational(3,2),h2:S.Rational(7,5),h3:S.Rational(5,3),g:3,d:100}
J=S.Matrix([[S.diff(fi,x).subs(pt) for x in vars] for fi in f]);det=S.factor(J.det())
print('det=',det); print('nonzero=',det!=0);print('physical n squared=',[(hh**2+T**2)/(1+T**2) for hh in h]); print('n_squared_witness=',[S.factor(((hh**2+T**2)/(1+T**2)).subs(pt)) for hh in h])
