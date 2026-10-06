import sympy as S
T=S.Rational(1,3);h=[S.Rational(3,2),S.Rational(7,5),S.Rational(5,3)];g=S.Integer(3);d=S.Integer(100)
rows=[]
for i in range(3):
 L=d+(2-i)*g+3*sum(1/h[j] for j in range(i+1,3));W=d+(2-i)*g+3*sum(1/h[j]**3 for j in range(i+1,3))
 rowA=[];rowB=[]
 for j in range(6):
  dT=int(j==0);dhi=int(j==i+1)
  dL=S.Integer(j==5)+(2-i)*int(j==4);dW=dL
  if j in range(i+2,4):dL-=3/h[j-1]**2;dW-=9/h[j-1]**4
  rowA.append(2*T*dT*(W/L+1/h[i])+T*T*((dW*L-W*dL)/L**2-dhi/h[i]**2))
  rowB.append(dT/(h[i]*L)-T/(h[i]*L)*(dhi/h[i]+dL/L))
 rows.extend([rowA,rowB])
det=S.factor(S.Matrix(rows).det());print('independent manually differentiated determinant:',det)
print('matches:',det==S.Rational(9079553843083,47292531300183951601092000000))
