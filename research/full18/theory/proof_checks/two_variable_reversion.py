"""One exact proof-witness check; no parameter sweep or numerical reconstruction."""
import sympy as s
h1, h2, h3 = s.Rational(3,2), s.Rational(7,5), s.Rational(5,3)
g, d = s.Integer(3), s.Integer(100)
h = [h1,h2,h3]
L = [d+2*g+3/h2+3/h3, d+g+3/h3, d]
W = [d+2*g+3/h2**3+3/h3**3, d+g+3/h3**3, d]
alpha = [W[i]/L[i]+1/h[i] for i in range(3)]
beta = [1/(h[i]*L[i]) for i in range(3)]
p,a,E,z,b = beta[1], alpha[1]-1, 3*(h3**-3-h3**-1), alpha[0]-1, beta[0]
k,c = 2+3*p, -d-3/h3
f1 = s.factor(z*k+3*p-3*p*a*a+3*p*p*E-b*(k*k*a/p+2*k*c))
f0 = s.factor(z*c-E+3*p*a*E+b*k*k*E/p-b*c*c)
x = s.symbols('x')
Q = p*x*x-a*x+E
f = z*(k*x+c)-E-3*((p*x)**3-p*x)-b*(k*x+c)**2
assert s.cancel(s.rem(f,Q,x)-(f1*x+f0)) == 0
assert f1 == s.Rational(399563722409659,317911540674000)
assert f0 == -s.Rational(399563722409659,3033507067500)
assert -f0/f1 == L[1] == s.Rational(524,5)
Lminus = s.factor(E/(p*L[1]))
assert Lminus == -s.Rational(1008,625)
qprime = s.factor(s.diff(Q,x).subs(x,L[1]))
assert qprime == s.Rational(16627,22925)
assert s.factor(p*f0*f0+a*f0*f1+E*f1*f1) == 0
multiplier = s.factor(-f1*qprime)
assert multiplier == -s.Rational(511042000961953861,560624774611650000)
print('All exact checks passed.')
print('f1 =', f1, '\nf0 =', f0)
print('L_plus =',L[1], '\nL_minus =',Lminus)
print('Qprime =',qprime, '\nJacobian multiplier =',multiplier)
