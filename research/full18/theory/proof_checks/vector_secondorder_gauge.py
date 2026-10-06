from pathlib import Path
exec((Path(__file__).with_name('vector_firstorder_rank.py')).read_text().split("print('det='")[0])
Bx=S.Rational(1411,35);By=S.Integer(2)
fp=[S.factor(fi.subs(pt)) for fi in f]
rhs=[]
for j in range(3):
 A,beta=fp[2*j:2*j+2]
 shear=-By*beta; diag=1+A-Bx*beta
 cp=1-diag**2+shear**2;rp=2*diag*shear
 Ap=rp-(1+(Bx/By)**2)*shear-(Bx/By)*cp
 bp=-cp/By-shear*Bx/By**2
 rhs.extend([Ap,bp])
gauge=J.inv()*S.Matrix(rhs)
print('gauge=',[S.factor(v) for v in gauge])
x,y,H,D=S.symbols('x y H D',real=True)
z=x+S.I*y;zb=x-S.I*y; bb=Bx-S.I*By; aa=zb/2;cc=(bb-D*zb)/2;k=H-1
U=k*(D*(1+aa*z)-z*cc/H)
C=k*(D*H*aa+D*(3*H-1)*aa**2*z/2-cc*(1+aa*z+z*aa/H**2))
I=C/U**2
pp={x:pt[T],y:0,H:pt[h3],D:pt[d]}
# derivative along the unique first-order shape-preserving curve, with dpsi=1
vel=[gauge[0],pt[T],gauge[3],gauge[5]]
ans=S.factor(sum(S.diff(I,v).subs(pp)*vv for v,vv in zip([x,y,H,D],vel)))
print('I_derivative=',ans)
print('real=',S.factor(S.re(ans)))
print('imag=',S.factor(S.im(ans)))
print('nonzero=',ans!=0)
# Convert the already computed tangent to original native coordinates.
B0=6+2*g+d+3*sum(1/hh for hh in h)
B0p=sum(S.diff(B0,v).subs(pt)*velv for v,velv in zip(vars,gauge))
bxp=-gauge[0]*B0.subs(pt)-pt[T]*B0p;byp=-pt[T]*B0.subs(pt)
np=[];lp=[];ph=[]
for j in range(3):
 nn=S.sqrt((h[j]**2+T**2)/(1+T**2))
 np.append(S.diff(nn,T).subs(pt)*gauge[0]+S.diff(nn,h[j]).subs(pt)*gauge[j+1])
 tv=S.Matrix([x,y]);bv=S.Matrix([Bx,By]);Mi=(h[j]-1)*(L[j]*S.eye(2)+(W[j]+L[j]/h[j])*(tv*tv.T)-(tv*bv.T)/h[j])
 ev={**pt,x:pt[T],y:0}
 Mp=S.zeros(2)
 for vv,vp in zip(vars,gauge):
  if vv!=T:Mp+=Mi.diff(vv).subs(ev)*vp
 Mp+=Mi.diff(x).subs(ev)*gauge[0]+Mi.diff(y).subs(ev)*pt[T]
 C=S.simplify(Mi.subs(ev).inv()*Mp)
 lp.append(S.factor(-S.trace(C)/2));ph.append(S.factor(-(C[1,0]-C[0,1])/2))
 print('conformal_check',j,S.simplify(C[0,0]-C[1,1]),S.simplify(C[0,1]+C[1,0]))
print('Tprime=',S.N(gauge[0],10))
print('native nprime=',[S.N(v,10) for v in np])
print('gprime,dprime=',S.N(gauge[4],10),S.N(gauge[5],10))
print('beta_degprime=',S.N(gauge[0]/(1+pt[T]**2)*180/S.pi,10),S.N(pt[T]*180/S.pi,10))
print('source_prime=',S.N(bxp,10),S.N(byp,10))
print('log_e_prime=',[S.N(v,10) for v in lp])
print('phi_deg_prime=',[S.N(v*180/S.pi,10) for v in ph])
print('wedge_deg_prime_per_epsilon=',[S.N(v*180/S.pi,10) for v in lp])
other=[*np,gauge[4],gauge[5],gauge[0]/(1+pt[T]**2)*180/S.pi,pt[T]*180/S.pi,bxp,byp,*[v*180/S.pi for v in ph]]
sup=max(abs(float(S.N(v))) for v in other)
print('limiting_native_sup_norm=',sup)
print('I_abs_deriv=',abs(complex(S.N(ans))))
print('I_abs_deriv_per_native_sup=',abs(complex(S.N(ans)))/sup)
print('I_value=',S.N(I.subs(pp),12))
uval=complex(S.N(U.subs(pp)));print('Uplus=',uval,'absU_squared=',abs(uval)**2);print('self2N_output_derivative_per_native_sup_per_epsilon_squared=',abs(uval)**2*abs(complex(S.N(ans)))/sup)
