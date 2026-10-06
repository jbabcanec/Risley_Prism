from pathlib import Path
exec(Path(__file__).with_name('audit_vector_firstorder_rank.py').read_text().split("det=S.factor")[0])
J=S.Matrix(rows);Bx=S.Rational(1411,35);By=S.Integer(2);Bbar=Bx-S.I*By
rhs=[]
for i in range(3):
 L=d+(2-i)*g+3*sum(1/h[j] for j in range(i+1,3));W=d+(2-i)*g+3*sum(1/h[j]**3 for j in range(i+1,3))
 A=T*T*(W/L+1/h[i]);bb=T/(h[i]*L);r=1+A-Bx*bb;c=-By*bb
 cp=1-r*r+c*c;rp=2*r*c
 rhs += [rp-(1+(Bx/By)**2)*c-Bx/By*cp,-cp/By-c*Bx/By**2]
v=J.inv()*S.Matrix(rhs)
z=T;dz=v[0]+S.I*T;zb=T;dzb=v[0]-S.I*T;H=h[2];dH=v[3];dd=v[5]
a=zb/2;da=dzb/2;c0=(Bbar-d*zb)/2;dc0=-(dd*zb+d*dzb)/2;k=H-1
F1=d*(1+a*z)-z*c0/H
DF1=dd*(1+a*z)+d*(da*z+a*dz)-(dz*c0+z*dc0)/H+z*c0*dH/H**2
G=1+a*z+z*a/H**2
DG=da*z+a*dz+(dz*a+z*da)/H**2-2*z*a*dH/H**3
F2=d*(H*a+(3*H-1)*a*a*z/2)-c0*G
DF2=dd*(H*a+(3*H-1)*a*a*z/2)+d*(dH*a+H*da+3*dH*a*a*z/2+(3*H-1)*(2*a*da*z+a*a*dz)/2)-dc0*G-c0*DG
U=k*F1;DU=dH*F1+k*DF1;C=k*F2;DC=dH*F2+k*DF2
ans=S.factor((DC*U-2*C*DU)/U**3)
r=S.factor(S.re(ans));im=S.factor(S.im(ans))
print('independent Re=',r);print('independent Im=',im)
er=S.Rational(18848880260984398987381756622941300813984492817,49946733938194977003395356774800375463944000000)
ei=S.Rational(1444583062125308617841515913871558551982883,79280530060626947624437074245714881688800000)
print('matches:',r==er,im==ei)
