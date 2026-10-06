# Physical-vector Risley inversion: exact structure and a finite-record identifiability theorem

Research draft for expert review, 2026-10-02, revision 4.

The main result is a constructive local theorem for the physical three-dimensional Snell model with all eighteen parameters unknown. First-order ellipse data has rank seventeen at an explicit admissible oblique-beam witness. Its remaining one-dimensional ambiguity is resolved locally by one physical second harmonic. A four-polynomial last-prism construction makes formal candidate generation generically finite. A finite-sample argument then proves rank eighteen for the exact 200-point observation map at all sufficiently small nonzero wedges near the witness.

This is an identifiability and inverse-structure result. It is not a global uniqueness theorem, a completed global solver, an explicit usable wedge threshold, or a numerical noise guarantee. Formal leading coefficients are not assumed to be exact measurements of a finite-angle trace. The formulas, witness certificates and analytic transfer below keep these distinctions explicit.

The canonical independent-axis model is a different mathematical model. Its results are preserved in section 9 and the supporting notes; its pure-third-harmonic argument is not used to prove physical-vector identifiability. No new recovery experiments were run for this revision.

The independent checks described here are mathematical and symbolic audits, not external peer review or a proof-assistant certificate. No physical experiment or successfully evaluated useful finite-noise certificate has yet been established. Literature positioning and the remaining validation gap are stated at the end.

## 1. Problem, native coordinates and exact physical assumptions

The native parameter vector is
\[
\theta=(N_{1:3},a_{1:3},\phi_{1:3},n_{1:3},d,g,
        \beta_x,\beta_y,b_x,b_y).
\]

The bounds are:

| Coordinates | Native bounds and meaning |
|---|---|
| \(N_i\) | Signed rotor speeds in \([-3.5,3.5]\) Hz |
| \(a_i,\phi_i\) | Signed wedge angles and initial phases in \([-18^\circ,18^\circ]\) |
| \(n_i\) | Glass indices in \([1.3,1.8]\) |
| \(d,g\) | Workpiece distance \([50,200]\), shared gap \([2,15]\) |
| \(\beta_x,\beta_y\) | Beam angles in \([-25^\circ,25^\circ]\) |
| \(b_x,b_y\) | Source offsets in \([-5,5]\) |

Source distance is 6 and each prism has axial vertex thickness 3. Lengths are native model units; they are not asserted to be millimetres. Physical prism order is retained. Internal trigonometric equations and angular derivatives use radians unless a conversion to native degrees is stated.

The ideal observation times are \(t_k=k/20\), \(k=0,\ldots,199\), ending at 9.95 seconds. The original executable's binary64 `arange` timestamps are a different exact input array. All finite-record theorems here use the ideal clock; they do not silently certify a substituted timestamp array.

Let \(q\in\mathbb R^2\) be the incident transverse direction cosine, \(z=\sqrt{1-|q|^2}>0\), and \(\mathbf t=q/z\) the beam slope. The native angle convention is
\[
\beta_x=\arctan t_x,\qquad \beta_y=\arctan t_y.
\]
The source position is \(b=(b_x,b_y)\) at axial coordinate zero. Equivalently, its incident unit direction is the normalization of \((\tan\beta_x,\tan\beta_y,1)\).

For rotor \(i\), put
\[
s_i=\sin(a_i)(\cos\gamma_i,\sin\gamma_i),\qquad
\gamma_i(t)=2\pi N_it+\phi_i,\qquad
m_i=s_i/\sqrt{1-|s_i|^2}.
\]
Its exit normal is \((-s_i,\sqrt{1-|s_i|^2})\). Flat entrance planes have axial coordinates \(6,\ 9+g,\ 12+2g\); the corresponding exit planes are
\[
Z_{\rm ax}=9+m_1^Tp,\quad
Z_{\rm ax}=12+g+m_2^Tp,\quad
Z_{\rm ax}=15+2g+m_3^Tp.
\]
The screen is \(Z_{\rm ax}=15+2g+d\). These equations fix the sign convention and retain actual tilted-surface intersection.

Use the transmitted forward Snell branch and exclude total internal reflection and grazing. For physical sequential surface traversal, require positive internal and external propagation to the next stated surface. The small-wedge proof witnesses satisfy all these strict conditions. These physical constraints are declared here; they are not silently added to historical canonical-model scores.

For componentwise observation allowances \(\eta_m\), the inference object is
\[
\mathcal C_\eta(y)=
\{\theta\text{ in the native physical domain}:
 |F_m(\theta)-y_m|\le\eta_m,\quad m=1,\ldots,400\}.
\tag{1}
\]
An exact-data fiber means \(\eta=0\). A positive-width compatible set is generally a continuum. The original requested accuracy of .001 in every native coordinate mixes index, degree and native-length units; its interpretation is not a dimensionless condition number.

## 2. Exact vector forward graph and geometry reduction

### Paired position transfer

For one prism, let \(q\) now denote its incoming external transverse direction, set
\[
H=\sqrt{n^2-|q|^2},\qquad v=q/H,
\]
and let \(w\) be the outgoing external slope. If \(p\) is the flat-entry position, \(m\) the exit-plane slope, and \(\ell\) the axial distance from the exit vertex to the next flat plane, the exact transfer is
\[
\boxed{
p_{\rm next}=p+3v+\ell w+
 (v-w)\frac{m^T(p+3v)}{1-m^Tv}.}
\tag{2}
\]
Indeed, the exit intersection obeys
\(Z_{\rm ax}=3+m^Tp_{\rm exit}\) and \(p_{\rm exit}=p+vZ_{\rm ax}\),
so \(Z_{\rm ax}=(3+m^Tp)/(1-m^Tv)\); subsequent propagation is to \(Z_{\rm ax}=3+\ell\). Substitution gives (2). A direction-only calculation would omit its source-intersection term.

### Polynomial graph with physical root selection

An equivalent graph avoids square-root evaluations. Let \(X\) be the incoming transverse direction cosine and \((Y,v_{\rm out})\) the outgoing unit direction. Introduce
\[
H^2+|X|^2=n^2,\quad H>0,\qquad
Y-X+m(v_{\rm out}-H)=0,\qquad |Y|^2+v_{\rm out}^2=1,
\]
\[
v_{\rm out}>0,\qquad
P=H-m^TX>0,\qquad
R=v_{\rm out}-m^TY>0.
\tag{3}
\]
Tangential optical momentum is conserved in both directions of the exit plane. The positive conditions choose the intended root. The standard vector refraction principle is documented in Ken Moore's [Ansys/Zemax technical reference](https://optics.ansys.com/hc/en-us/articles/42661810210707-What-is-a-ray); the inverse constructions below are derived research results, not claims from that reference.

With \(W=v_{\rm out}X-HY\) and \(V=v_{\rm out}X-(m^TX)Y\),
\[
v_{\rm out}Pp_{\rm next}
=v_{\rm out}Pp+W(m^Tp)+3V+\ell PY.
\tag{4}
\]
Introducing dot products and \(W,V,P\) as auxiliaries makes the local equations polynomial of degree at most three. Initial propagation is
\(z\,p_{\rm entry}=z\,b+6q\), with \(|q|^2+z^2=1,\ z>0\).
Positive traversal can be imposed using
\[
h=H(3+m^Tp)/P>0,\qquad 3+\ell-h>0.
\]
The complete derivation and branch requirements are in [physical_vector_inverse.md](theory/physical_vector_inverse.md).

### Four geometry variables remain affine

Directions are independent of source position and of \(g,d\). Composing (2), or (4), with \(\ell=(g,g,d)\) gives
\[
F_{\rm vec}(t)=A(t)b+B_0(t)+gC(t)+dD(t),
\tag{5}
\]
where \(A\) is a real \(2\times2\) matrix. All coefficients depend only on the fourteen optical coordinates: indices, wedges, phases, speeds and beam angles.

Conditional on those fourteen coordinates, componentwise observation strips, native geometry bounds and the surface-order inequalities form a four-variable linear feasibility problem in \((b_x,b_y,g,d)\). This is exact at arbitrary admissible wedges and beam angles, not an asymptotic reduction. Euclidean residual bounds instead produce second-order-cone constraints.

The two transverse source coordinates are coupled. The canonical model's scalar-offset interval/polygon elimination does not transfer unchanged. General affine LP profiling and rigorously checked dual exclusions do survive, with coefficients recomputed from vector optics.

## 3. An explicit first-order inverse leaves one trial-beam curve

Let \(B\) be the zero-wedge workpiece position. Write \(e_i=\sin a_i\). The first-order trace has the form
\[
F(t)=B+\sum_i e_iM_i
 \begin{pmatrix}\cos\gamma_i(t)\\ \sin\gamma_i(t)\end{pmatrix}
 +O(\|e\|^2).
\tag{6}
\]
The remainder and its derivatives are uniform on a compact strict physical chart. Equation (6) defines formal first-order quantities; their estimation from a finite-angle record requires a separate error analysis.

### Deriving the first-order matrices

At zero wedge, all external directions equal the incident \(q\). Define
\[
H_i=\sqrt{n_i^2-|q|^2},\qquad D_i=d+(3-i)g,
\]
\[
\mathcal T=I/z+qq^T/z^3,\qquad
\mathcal V_j=I/H_j+qq^T/H_j^3.
\]
The unperturbed exit position of prism \(i\) is
\[
R_i=b+q\left[\frac{6+(i-1)g}{z}
             +3\sum_{j\le i}H_j^{-1}\right].
\]
Vector Snell differentiation gives an outgoing transverse change \((H_i-z)s_i\). Differentiating (2) and the downstream flat propagation gives
\[
\boxed{
M_i=(H_i-z)\left[
D_i\mathcal T+3\sum_{j>i}\mathcal V_j
-\frac{qR_i^T}{H_i z}\right].}
\tag{7}
\]
The final term retains the effect of source position through the tilted exit intersection.

### Effective-index factorization

Set
\[
h_i=H_i/z,\quad
L_i=D_i+3\sum_{j>i}h_j^{-1},\quad
W_i=D_i+3\sum_{j>i}h_j^{-3}.
\]
Then
\[
B=b+\mathbf t\left[6+2g+d+3\sum_i h_i^{-1}\right],
\qquad
n_i=\sqrt{\frac{h_i^2+|\mathbf t|^2}{1+|\mathbf t|^2}}.
\tag{8}
\]
Substituting \(B\) eliminates \(b\) from (7):
\[
\boxed{
M_i=(h_i-1)
 \left[L_iI+(W_i+L_i/h_i)\mathbf t\mathbf t^T
             -\mathbf t B^T/h_i\right].}
\tag{9}
\]
Write
\[
\frac{M_i}{(h_i-1)L_i}
=I+\alpha_i^{\rm sh}\mathbf t\mathbf t^T
      -\beta_i^{\rm sh}\mathbf tB^T,\quad
\alpha_i^{\rm sh}=W_i/L_i+1/h_i,\quad
\beta_i^{\rm sh}=1/(h_iL_i).
\tag{10}
\]
The superscript “sh” denotes ellipse-shape coefficients. These \(\beta_i^{\rm sh}\) are not native beam angles.

### Uniform orientation and signed speed

Rotate into the frame \(\mathbf t=(T,0)\), \(T=|\mathbf t|\). The matrices \(M_i/(h_i-1)\) are upper triangular; their perpendicular diagonal is \(L_i\ge50\). Put
\(B=b+B_{\rm beam}\mathbf t\), where
\(B_{\rm beam}=6+2g+d+3\sum_i h_i^{-1}\).
The parallel diagonal is
\[
L_i+W_iT^2-
 \big[(B_{\rm beam}-L_i)T^2+Tb_\parallel\big]/h_i.
\]
Since \(h_i\ge n_i\ge13/10\), \(B_{\rm beam}-L_i\le36+90/13\),
\(W_i\ge50\), and \(|b|\le\sqrt{50}\), it is bounded below by
\[
50+\frac{2870}{169}T^2-\frac{50\sqrt2}{13}T
\ge50-\frac{125}{287}>49.5.
\tag{11}
\]
Thus \(\det M_i>0\) throughout this bounded zero-wedge source/hardware domain.

For a resolved nonzero frequency, form the observed cosine/sine matrix using the positive-frequency convention. Its determinant has the sign of the physical signed speed: phase contributes a rotation and signed wedge contributes its square. Zero wedges, zero speeds and unresolved frequency collisions remain exceptional cases; this orientation argument does not justify discarding them.

### Reconstruction conditional on a trial slope

Let \(C_i\) be the first-order cosine/sine matrix oriented to the recovered signed rotor direction: starting with the positive-frequency convention, reverse its sine column when the speed is negative. Then put \(S_i=C_iC_i^T\). For a trial beam slope with \(T>0\), rotate \(S_i,B\) into its parallel/perpendicular frame. On a chart with \(B_\perp\ne0\), define
\[
c_i=(S_i)_{12}/(S_i)_{22},\qquad
r_i=\sqrt{\det S_i}/(S_i)_{22}>0.
\]
The shape factors are
\[
\boxed{
\beta_i^{\rm sh}=-\frac{c_i}{TB_\perp},\qquad
\alpha_i^{\rm sh}
=\frac{r_i-1-(B_\parallel/B_\perp)c_i}{T^2}.}
\tag{12}
\]
These remove wedge amplitude and initial phase.

Require \(\beta_i^{\rm sh}>0\) and \(\alpha_3^{\rm sh}>1\). Reconstruct downstream:
\[
h_3=\frac1{\alpha_3^{\rm sh}-1},\qquad
d=\frac1{h_3\beta_3^{\rm sh}}.
\]
Reject a trial unless the reconstructed \(h_3>1\), as physical \(n_3>1\) requires. Put \(E_3=3(h_3^{-3}-h_3^{-1})<0\). The equation
\[
\beta_2^{\rm sh}L_2^2-(\alpha_2^{\rm sh}-1)L_2+E_3=0
\tag{13}
\]
has exactly one positive root: its leading coefficient is positive and its constant is negative. Then
\[
h_2=\frac1{\beta_2^{\rm sh}L_2},\qquad
g=L_2-d-3/h_3,
\]
Require \(h_2>1\) before setting \(E_2=3(h_2^{-3}-h_2^{-1})<0\), and retain all native-index filters from (8). Then
\[
L_1=d+2g+3/h_2+3/h_3,\qquad
h_1=\frac1{\beta_1^{\rm sh}L_1}.
\]
One scalar consistency equation remains in the two trial slope coordinates:
\[
\boxed{
\alpha_1^{\rm sh}-1-\frac{E_2+E_3}{L_1}
-\beta_1^{\rm sh}L_1=0.}
\tag{14}
\]
Recover \(n_i,b\) using (8), and wedges/phases from
\[
M_i^{-1}C_i=e_iR(\phi_i).
\tag{15}
\]
Both possible signs of \(e_i\) differ by a phase shift of \(\pi\); the narrow native phase interval selects at most one. Enforce every original index, distance, gap, beam, source, wedge and phase bound. Retain all admissible assignments of resolved rotors to physical prism positions. Prism permutations are not a scoring symmetry.

Thus the formal first-order inverse is generically a curve in the trial beam plane, with beam direction and source position still unknown. Its scalar zero set may be singular, disconnected or cut by native boundaries. Charts \(T=0\) and \(B_\perp=0\) need separate treatment. A single locally found curve segment is not a complete global inverse.

## 4. A rank-seventeen first-order witness and its missing direction

For a fixed beam orientation, use the six shape quantities
\[
A_i=T^2(W_i/L_i+1/h_i),\qquad Q_i=T/(h_iL_i).
\]
Choose
\[
(T,h_1,h_2,h_3,g,d)=(1/3,3/2,7/5,5/3,3,100),\qquad
b=(1,2).
\tag{16}
\]
The physical indices and baseline are
\[
(n_1^2,n_2^2,n_3^2)=(17/8,233/125,13/5),\qquad
B=(1411/35,2).
\]
The beam points along x, with \(\beta_x=\arctan(1/3)\approx18.435^\circ\) and \(\beta_y=0\), strictly inside the beam box.

With rows \((A_1,Q_1,A_2,Q_2,A_3,Q_3)\) and columns
\((T,h_1,h_2,h_3,g,d)\), the exact determinant is
\[
\boxed{
\frac{9079553843083}
{47292531300183951601092000000}\ne0.}
\tag{17}
\]
The supplied independent audit reconstructed it using exact rational arithmetic at this one proof witness. This is a nonvanishing certificate, not a survey or a conditioning estimate.

The first-order data contains at most seventeen real quantities: three real \(2\times2\) harmonic matrices, two baseline coordinates and three speeds. At (16), six independent shape directions, six amplitude/phase directions, two baseline directions and three speed directions attain this upper bound. The remaining first-order fiber is locally one-dimensional.

An explicit fiber tangent uses beam orientation \(\psi\) with \(\psi'=1\), while holding the baseline, speeds and all \(C_i\) fixed. At \(\psi=0\), normalize the observed covariances as
\[
S_i=\begin{pmatrix}r_i^2+c_i^2&c_i\\c_i&1\end{pmatrix},
\qquad r_i=1+A_i-B_xQ_i,\quad c_i=-B_yQ_i.
\]
Rotating the fixed observed matrices gives
\[
c_i'=1-r_i^2+c_i^2,\qquad r_i'=2r_ic_i,
\]
\[
\begin{aligned}
A_i'={}&2r_ic_i-[1+(B_x/B_y)^2]c_i\\
&-(B_x/B_y)(1-r_i^2+c_i^2),\\
Q_i'={}&-(1-r_i^2+c_i^2)/B_y-c_iB_x/B_y^2.
\end{aligned}
\tag{18}
\]
The nonsingular system in (17) determines the compensating tangent in
\((T,h_1,h_2,h_3,g,d)\). Amplitude and phase adjustments keep each \(C_i\) fixed. All eighteen native parameters remain part of the inverse.

## 5. One physical second harmonic breaks that fiber locally

In this subsection use complex slope \(\mathsf T=t_x+it_y\) and complex baseline \(B=B_x+iB_y\). Let \(h=h_3\), \(k=h-1\). Isolate the positive rotor component using the formal complex transverse tilt
\(u=(\sigma/2,-i\sigma/2)\), so \(u^Tu=0\). This is an analytic coefficient calculation, not a complex physical illumination.

Set
\[
a_0=\overline{\mathsf T}/2,\qquad
c_0=(\overline B-d\overline{\mathsf T})/2.
\]
The outgoing axial direction normalized by the incident axial cosine, and its transverse slope, are
\[
\zeta(\sigma)=a_0\sigma+
 \sqrt{1-2ha_0\sigma+(a_0\sigma)^2},
\qquad
Q(\sigma)=\frac{\mathsf T+[h-\zeta(\sigma)]\sigma}{\zeta(\sigma)}.
\]
With only the third tilt active, exact paired propagation gives
\[
F(\sigma)=B+d[Q(\sigma)-\mathsf T]
 +(\mathsf T/h-Q(\sigma))
       \frac{c_0\sigma}{1-a_0\sigma/h}.
\tag{19}
\]
Expansion \(F=B+U_+\sigma+C_{+2}\sigma^2+O(\sigma^3)\) yields
\[
\boxed{
U_+=k[d(1+a_0\mathsf T)-\mathsf Tc_0/h],}
\]
\[
\boxed{
C_{+2}=k\left[
dha_0+\frac{d(3h-1)}2a_0^2\mathsf T
-c_0(1+a_0\mathsf T+\mathsf Ta_0/h^2)\right].}
\tag{20}
\]
At zero slope, \(C_{+2}=-k\overline B/2\), agreeing with the physical off-axis quadratic coefficient.

The ratio
\[
\mathcal I_3=C_{+2}/U_+^2
\tag{21}
\]
eliminates the signed wedge amplitude and phase. The positive orientation proof guarantees \(U_+\ne0\):
\(|U_+|^2-|U_-|^2=\det M_3>0\).

Along the exact first-order fiber tangent (18), the supplied independent exact calculation gives
\[
\operatorname{Re}\mathcal I_3'=
\frac{18848880260984398987381756622941300813984492817}
{49946733938194977003395356774800375463944000000}>0,
\]
\[
\operatorname{Im}\mathcal I_3'=
\frac{1444583062125308617841515913871558551982883}
{79280530060626947624437074245714881688800000}>0.
\tag{22}
\]
Both values were independently reproduced from paired vector equations and a separately assembled shape-preserving tangent. One real projection of the complex second harmonic therefore supplies the missing local direction.

This is a statement about a formal second-order coefficient. Its transfer to the actual finite-angle sampled map is the next theorem, rather than an assertion that the coefficient has been observed without contamination.

### A generically finite polynomial candidate construction

The last prism supplies a further formal reduction. Let \(U,V\) be its intrinsic unit-wedge coefficients at the positive and negative signed rotor frequencies, and \(C\) its intrinsic positive self-second-harmonic coefficient. With complex wedge amplitude \(\xi=e_3e^{i\phi_3}\), the observed formal coefficients are
\[
U_{\rm obs}=\xi U,\qquad V_{\rm obs}=\overline\xi V,\qquad
C_{\rm obs}=\xi^2C.
\]
Use one consistent signed-frequency convention throughout. The invariants
\[
w=V_{\rm obs}/\overline{U_{\rm obs}}=V/\overline U,\qquad
\mathcal I=C_{\rm obs}/U_{\rm obs}^2=C/U^2
\tag{L1}
\]
cancel wedge amplitude and phase. The cancellation holds for the corresponding observed *formal leading* coefficients; it is not an assertion about ratios extracted without bias from a finite-angle trace.

Temporarily fix the leading baseline \(B\). Set
\(\mathsf T=t_x+it_y,\ H=h_3,\ k=H-1,\ R=|\mathsf T|^2\), and define
\[
\mathscr A=2Hd+d(H+1)R-\mathsf T\overline B,\qquad
V_0=d(H+1)\mathsf T^2-\mathsf TB,
\]
\[
\begin{aligned}
C_0={}&4dH^3\overline{\mathsf T}
 +dH^2(3H-1)R\overline{\mathsf T}\\
&-4H^2(\overline B-d\overline{\mathsf T})
 -2(H^2+1)R(\overline B-d\overline{\mathsf T}).
\end{aligned}
\]
The exact formal identities are
\[
U=k\mathscr A/(2H),\qquad V=kV_0/(2H),\qquad
C=kC_0/(8H^2).
\]
Hence four real polynomials in \((t_x,t_y,H,d)\) determine candidates:
\[
\boxed{
\operatorname{Re}(V_0-w\overline{\mathscr A})=0,\qquad
\operatorname{Im}(V_0-w\overline{\mathscr A})=0,}
\]
\[
\boxed{
\operatorname{Re}[C_0-2\mathcal I(H-1)\mathscr A^2]=0,\qquad
\operatorname{Im}[C_0-2\mathcal I(H-1)\mathscr A^2]=0.}
\tag{L2}
\]
Retain \(H>1,\ \mathscr A\ne0\), the original beam/distance bounds, and
\[
1.3^2(1+R)\le H^2+R\le1.8^2(1+R).
\tag{L3}
\]
For each regular-chart candidate, reconstruct \(h_2,g,h_1\) using section 3, enforce its remaining scalar consistency equation, and apply every original source, index, wedge, phase and physical branch constraint.

At the same witness, with fixed \(B=(1411/35,2)\), the Jacobian determinant of
\((\operatorname{Re}w,\operatorname{Im}w,
\operatorname{Re}\mathcal I,\operatorname{Im}\mathcal I)\)
with respect to \((t_x,t_y,H,d)\) is exactly
\[
\boxed{
\frac{19617689107775384734706249006250000}
 {141672457738889533504325313445087046934090001}>0.}
\tag{L4}
\]
The coefficient identities and this determinant were independently reconstructed at the existing proof point. No additional parameter search is involved.

Consequently the four-dimensional rational map is generically finite for this audited fixed baseline, and for generic baselines in the joint baseline family. This has not been proved for every arbitrary fixed baseline. The polynomial degrees are at most \((4,4,9,9)\). After saturating complex elimination away from
\(H(H-1)\mathscr A\overline{\mathscr A}=0\), a coarse generic isolated complex candidate bound is \(4\cdot4\cdot9\cdot9=1296\). Real filters (L3) do not permit retaining denominator-generated roots. The bound establishes neither uniqueness nor finite fibers at exceptional invariant values.

This replaces an open-ended formal curve search with a generically finite algebraic candidate problem. Exceptional fibers may still contain curves or higher-dimensional components. Zero wedges can make ratios undefined; zero/colliding frequencies can prevent labeling; the ellipse chart also fails at \(T=0\) or \(B_\perp=0\), even though (L2) remains evaluable. Preserve these components and their inequalities or use the exact semialgebraic formulation. With uncertain coefficient intervals the compatible set is generally continuous, so 1296 is not a bound on the number of noisy compatible systems.

The leading baseline is part of the enclosure: measured DC has quadratic and higher-order contamination and cannot be inserted as exact \(B\). All formal candidates remain subject to a finite-angle contamination bound and exact-model correction.

## 6. Exact 200-sample rank eighteen and its scope

Choose
\[
N=(1,7,49)/20\ {\rm Hz},\qquad \phi_i=0,\qquad
e_i=\varepsilon\rho_i
\]
with nonzero fixed \(\rho_i\), at the oblique hardware/source witness (16).
All degree-at-most-two harmonic indices give 25 sampled nodes
\[
z_m=\exp[2\pi i(m_1+7m_2+49m_3)/400],\qquad |m|_1\le2.
\tag{23}
\]
They are distinct. A difference of two indices has \(\ell_1\) norm at most four and cannot satisfy a nonzero base-seven relation. Labels lie in \([-98,98]\), so their differences cannot be nonzero multiples of 400.

Add \(kz_m^k\) at the six signed fundamental nodes for the speed derivatives. These 31 confluent columns have full rank on the 200 rows. To see this exactly, an annihilating polynomial of degree at most 30 would vanish at the 25 distinct nodes and have zero derivative at the six repeated nodes. It would have 31 zeros counted with multiplicity and must be zero.

Fixed real linear functionals of the actual 200 x/y samples can consequently select seventeen first-order channels and a real projection of the second harmonic in the limiting temporal space. The extractor is fixed at the proof witness; it is not a proposed algorithm given the unknown speeds, continuous-time derivatives or independent torus measurements.

Along the first-order fiber,
\[
U_{\rm obs}=e_3e^{i\phi_3}U_+\quad\hbox{is fixed},\qquad
\frac{dC_{\rm obs}}{d\psi}
=U_{\rm obs}^2\,\mathcal I_3'\ne0.
\tag{24}
\]
Choose local coordinates adapted to two baseline directions, fifteen other first-order directions and the remaining fiber. In scaled wedge coordinates, multiply the corresponding Jacobian columns by
\(1,\ \varepsilon^{-1},\ \varepsilon^{-2}\), respectively.
The selected \(18\times18\) matrix tends to a block triangular matrix with nonzero diagonal blocks by (17), the fundamental/confluent separation, and (22).

All \(C^1\) remainders vanish after this scaling on a compact strict analytic chart. The fiber preserves the zeroth and first-order maps, leaving its first nonzero response at quadratic order. Unrepresented higher harmonics can leak into a finite extractor, but their scaled contribution tends to zero; they are not assumed absent. Frequency derivatives of quadratic terms are \(O(\varepsilon^2)\) and vanish after the first-order \(\varepsilon^{-1}\) scaling, so repeated quadratic nodes are unnecessary.

**Physical-vector finite-record theorem.** For all sufficiently small positive \(\varepsilon\), the exact coupled-vector \(400\times18\) sampled Jacobian has rank eighteen at this admissible oblique family. A nonsingular eighteen-output minor and the inverse function theorem give local injectivity. All eighteen parameters, including beam direction and source position, are unknown.

No numerical upper bound on \(\varepsilon\), usable singular-value bound, global uniqueness or noise guarantee is proved by this argument.

### Component-qualified generic conclusions

A nonzero analytic minor implies generic rank eighteen on the connected analytic interior component containing this witness. This is not a statement about every physical component or every point of the native box.

For the same ideal clock, rotor-step circles and positive-root vector equations give a semialgebraic lift. Use an injective native algebraic chart, with the original signed speeds, phases and wedge information retained even at zero wedges. Outside a lower-dimensional exceptional subset of the component's eighteen-dimensional image, exact fibers consist of isolated regular points; semialgebraicity makes such fibers finite. “Generic” is relative to that image, not to arbitrary points of the 400-dimensional observation space. A finite fiber need not contain one system.

Included native boundary faces have lower-dimensional images. At excluded strict optical boundaries the forward map may be undefined; boundary persistence uses the finite-output projection of the closure of the forward graph minus the graph, not an assumed value of \(F\) at undefined points. These qualifications must remain when stating robust exact-data continuation.

The finiteness conclusion concerns \(F(\theta)=y\), or \(\eta=0\). With positive error allowance and strict residual slack, an interior compatible point has an open neighborhood of compatible systems. Neither generic finiteness nor local rank converts positive-noise compatibility into a unique parameter vector.

## 7. Data-derived certificates and native uncertainty

The following is a sufficient, conditional certificate framework. It has not been successfully evaluated for an actual record in this work. Every condition may fail; failure leaves the associated prior region unresolved. It does not authorize substituting a favorable fitted neighborhood or discarding an unexamined branch. The complete audited framework is in [physical_inverse_certificates.md](theory/physical_inverse_certificates.md).

### Rotor enclosure from all 200 observations

On a uniformly guarded prior region, require a certified first-order enclosure
\[
F_h(t_k)=B_h+\sum_i[
C_{hi}\cos(2\pi\nu_it_k)+D_{hi}\sin(2\pi\nu_it_k)]+R_{hk},
\qquad |R_{hk}|\le\tau_{hk},
\]
where \(h=x,y\) and the \(\nu_i\) are unknown positive frequency magnitudes. This uses general oblique ellipses and has a safe first-order remainder \(O(\varepsilon^2)\); it does not assume a known beam. Let
\(\delta=\max_{hk}(\eta_{hk}+\tau_{hk})\), including any explicitly bounded clock or evaluator discrepancy.

Form a real stacked Hankel matrix \(H_y\in\mathbb R^{386\times7}\) whose rows are
\[
(y_{hk},y_{h,k+1},\ldots,y_{h,k+6}),
\quad h=x,y,\quad k=0,\ldots,192,
\]
and let \(v_y\) contain \(y_{h,k+7}\). All 200 samples are used. A compatible seven-tone sequence obeys \(H_0c=-v_0\), where
\[
p(z)=z^7+\sum_{j=0}^6c_jz^j
\]
has roots \(1,e^{\pm2\pi i\nu_i/20}\).

For a verified left inverse \(D H_y=I\), put
\(\gamma=\|D\|_{\infty\leftarrow\infty}\), \(\widehat c=-Dv_y\).
If \(q_H=7\gamma\delta<1\), every compatible \(H_0\) has full column rank and
\[
\boxed{
\|c-\widehat c\|_\infty\le
\Delta_c=
\frac{\gamma\delta(1+\|\widehat c\|_1)}
 {1-7\gamma\delta}.}
\tag{C1}
\]
To prove this, write \(H_0=H_y+E,\ v_0=v_y+e\), with
\(\|E\|_\infty\le7\delta,\ \|e\|_\infty\le\delta\).
Then \((I+DE)(c-\widehat c)=-D(e+E\widehat c)\); the Neumann inverse bound gives (C1).

A floating pseudoinverse is only a proposal. If its left-inverse defect is bounded by \(\rho<1\), use the corrected inverse \((DH_y)^{-1}D\) and the bound \(\gamma\le\|D\|/(1-\rho)\). The corrected inverse must also define \(\widehat c\); any retained approximate-center error must be added to \(\Delta_c\). Arithmetic bounds must be outward.

For disjoint candidate contours, certify
\[
|\widehat p(z)|>
\Delta_c\sum_{j=0}^6|z|^j
\quad\hbox{on their boundaries},
\tag{C2}
\]
with root counts summing to seven. Rouche's theorem then encloses every compatible spectrum. Intersect with the unit circle, the exact root 1, conjugate-pair structure and the native speed arc. Keep every allowed pairing, direction and prism assignment. A certified interval trigonometric design must charge the full frequency intervals when enclosing each observed ellipse; fitting only their midpoints is insufficient.

The stacked real coordinates retain both conjugate frequency nodes even when a circular complex trace has only one signed exponential. Zero DC, vanishing amplitudes or colliding nodes can still defeat the rank test. A differenced six-tone chart can remove DC, but doubles worst-case noise and attenuates slow signals; it does not justify ignoring failures.

For a positive-frequency ellipse matrix \(E_i\),
\[
E_i=e_iM_iR(\phi_i)\operatorname{diag}(1,\operatorname{sign}N_i).
\]
The structural bound (11) gives \(\det M_i>0\), so an interval determinant of \(E_i\) excluding zero determines the speed sign. Otherwise keep both signs. The proof-witness frequencies in section 6 are never initialization data.

### Uniform exact strong inverse and scalar closure

Suppose a data-derived branch admits an injective physical chart
\(\theta=\Theta(z,\psi)\), \(z\in\mathbb R^{17}\), with original bounds and units retained. Choose seventeen fixed real record rows \(P_f\), including frequency-tangent information, and one fixed quadratic row \(P_g\). They are selected from the data-derived rotor chart and then held fixed. Define the exact functions
\[
f(z,\psi)=P_fF(\Theta(z,\psi)),\qquad
g(z,\psi)=P_gF(\Theta(z,\psi)).
\]
No truncated optical model replaces these equations. For the observed record, let
\[
a_0=P_fy,\quad c_0=P_gy,\quad
r_a=|P_f|\eta,\quad r_c=|P_g|\eta,
\]
\[
\mathcal A=a_0+[-r_a,r_a],\qquad
\mathcal C=[c_0-r_c,c_0+r_c].
\tag{C3}
\]
This rectangular feature box safely overapproximates correlated feature noise.

**Strong condition.** For every \((\psi,a)\in I\times\mathcal A\), the exact equations \(f(z,\psi)=a\) must have one and only one solution \(z=\zeta(\psi,a)\) in the retained physical strong domain \(\mathcal Z\). Nonsingularity at a fitted point is insufficient.

A sufficient certificate on compact convex \(\mathcal Z\) is a fixed preconditioner \(C_f\) satisfying
\[
\sup_{\mathcal Z,I}\|I-C_fJ_{f,z}\|\le q_f<1,
\qquad z-C_f[f(z,\psi)-a]\in\mathcal Z
\tag{C4}
\]
for every \(z,\psi,a\) in the declared product domain. The contraction and self-map establish existence and uniqueness uniformly. A parameter-uniform interval-Newton/Krawczyk theorem or a structural strong inverse may replace (C4). Subdivided charts must retain a complete branch cover.

Define
\[
G(\psi,a)=g(\zeta(\psi,a),\psi),\quad J=f_z,\quad
w=g_zJ^{-1},\quad
s=g_\psi-g_zJ^{-1}f_\psi.
\]
Implicit differentiation gives the exact identities
\[
\zeta_\psi=-J^{-1}f_\psi,\quad \zeta_a=J^{-1},
\qquad G_\psi=s,\quad G_a=w.
\tag{C5}
\]
Require a constant sign of \(s\) and \(|s|\ge\mu>0\) on the entire strong solution family. For positive sign, the uniform bracket
\[
\sup_{a\in\mathcal A}G(\psi_{\rm left},a)\le\inf\mathcal C,
\qquad
\inf_{a\in\mathcal A}G(\psi_{\rm right},a)\ge\sup\mathcal C
\tag{C6}
\]
gives exactly one scalar solution for every \((a,c)\in\mathcal A\times\mathcal C\). For negative sign, require
\[
\inf_{a\in\mathcal A}G(\psi_{\rm left},a)\ge\sup\mathcal C,
\qquad
\sup_{a\in\mathcal A}G(\psi_{\rm right},a)\le\inf\mathcal C.
\]
Strict inequalities give an interior margin. Partial brackets can still exclude or enclose intervals, but do not prove existence for every target.

These conditions reduce the exact eighteen-feature inverse to a scalar monotone root while solving all eighteen original parameters jointly. They are checkable hypotheses, not facts established for the current observations.

### Finite uncertainty in native coordinates

Let \(\lambda_j=\sup|w_j|\). Along the solution path between two targets in the convex, uniformly bracketed feature box,
\[
|\Delta\psi|\le
\frac{|\Delta c|+\sum_j\lambda_j|\Delta a_j|}{\mu}.
\tag{C7}
\]
In native coordinates the exact differential is
\[
d\theta=U\,da+v\,d\psi,\quad
U=\Theta_zJ^{-1},\quad
v=\Theta_\psi-\Theta_zJ^{-1}f_\psi.
\tag{C8}
\]
Thus, with \(\overline U_{ij}=\sup|U_{ij}|\) and
\(\overline v_i=\sup|v_i|\),
\[
\boxed{
|\Delta\theta_i|\le
\sum_j\overline U_{ij}|\Delta a_j|
+\frac{\overline v_i}{\mu}
 \left(|\Delta c|+\sum_j\lambda_j|\Delta a_j|\right).}
\tag{C9}
\]
For a tube diameter use \(2r_a,2r_c\); relative to the certified center-feature solution use \(r_a,r_c\). Being a feature-center solution does not itself establish compatibility with the full record.

The actual native null tangent \(v\) is essential. The pointwise invariant weak-conditioning quantity is \(|s|/\|v\|_{\rm native}\), not \(|s|\) alone: reparameterizing \(\psi\) scales both. The uniform bound \(\mu/\sup\|v\|\) is valid but need not be numerically invariant under nonlinear reparameterization. Strong-to-weak coupling \(w\) is also essential; omitting it would treat jointly unknown parameters as fixed calibration. A precision claim needs the native factors \(\overline v_i/\mu,\overline U\), with angle and length units specified.

For normalized quadratic observables, a fixed nonzero branch reference \(U_0\) permits a linear feature such as \(\operatorname{Re}(C_{+2}/U_0^2)\), while the fundamental is included among the strong features. Its variation is then charged through \(w\). If actual noisy ratios are used and
\(|U-\widehat U|\le r_U<|\widehat U|\), \(|C-\widehat C|\le r_C\), a finite bound is
\[
\left|\frac C{U^2}-\frac{\widehat C}{\widehat U^2}\right|
\le\frac{r_C}{(|\widehat U|-r_U)^2}
+\frac{|\widehat C|r_U(2|\widehat U|+r_U)}
 {(|\widehat U|-r_U)^2|\widehat U|^2}.
\tag{C10}
\]
Division by a small fundamental is therefore part of the noise budget.

### The witness derivative is not a recovery constant

At the oblique witness, the tangent with \(\psi'=1\) has a native mixed-coordinate sup norm tending to approximately \(981.7869465\), dominated by the workpiece-distance derivative. The normalized invariant derivative has magnitude approximately \(0.3778192691\) per radian of \(\psi\), or \(0.0003848281650\) per unit of that native normalized tangent. Also
\[
U_+\approx69.904973545+0.133333333i,\qquad
|U_+|^2\approx4886.7231041.
\]
Thus the corresponding self-second-harmonic directional gain is approximately
\(1.880548685\,\varepsilon^2\) per native sup unit, asymptotically.

These are rounded conversions of one audited proof tangent. They are not the smallest singular value, a full inverse Lipschitz constant, or an accuracy guarantee for all coordinates. The large tangent scale illustrates why \(v\) in (C8)-(C9) cannot be omitted. The native norm mixes degrees, refractive-index units and native lengths because the requested coordinate tolerance does. The full tangent, including its wedge-angle conversion, is preserved in [physical_oblique_inverse.md](theory/physical_oblique_inverse.md).

At this regular witness the native singular-direction hierarchy has five order-one directions, twelve order-\(\varepsilon\) directions and one order-\(\varepsilon^2\) direction, with chart-dependent constants. The five strong directions include three wedge-amplitude directions and two baseline combinations. Weak directions mix native parameters; this does not assign a separate accuracy power to each named coordinate.

Under uniform bounds on the normalized strong inverse, chart derivatives and Taylor remainders, the scalar margin can take the form \(s=\varepsilon^2s_2+O(\varepsilon^3)\). Twelve adapted modes then have error order \(\eta/\varepsilon\), and one has order \(\eta/\varepsilon^2\). A nonzero rational determinant or a wedge exponent alone does not supply the necessary constants.

A matching deterministic lower-bound power requires more than a bound on one feature. Along a first-order-preserving curve parameterized by a monotone native coordinate \(\theta_j=s\), certify
\[
\|D F(\theta(s))\theta'(s)\|_\infty\le C\varepsilon^2
\]
over **all 400 original real observations**, plus an available parameter interval of length \(r\). Endpoints then have native separation at least their scalar separation and data distance at most \(C\varepsilon^2|\Delta s|\). For a uniform scalar observation allowance \(\eta\), a midpoint-record argument gives minimax native error at least
\[
\frac12\min\left(r,\frac{2\eta}{C\varepsilon^2}\right).
\tag{C11}
\]
Alternatively retain a certified secant factor in the native endpoint separation. Unit tangent norm alone is insufficient because arc length does not lower-bound endpoint separation. This conditional lower bound matches powers with the Schur upper bound, not automatically numerical constants.

### Higher-order contamination remains part of the inverse

A finite record at nonzero wedges contains higher harmonics. First-order coefficient fitting, frequency estimation and second-harmonic demixing need a joint error analysis. Formal coefficients, torus coefficients and coefficients produced by a finite linear extractor are different objects. A fixed degree-two extractor may mix cubic terms into a quadratic row, leaving only an \(O(\varepsilon)\) normalized quadratic remainder unless stronger cancellation is proved.

A numerical Newton or Krawczyk certificate for eighteen selected/projected equations proves only those equations. Every candidate must independently satisfy all 400 residual constraints, native bounds and branch/traversal conditions. A local root does not exclude a distant branch.

### A complete outer-to-exact route

For any polynomial approximation \(\mathcal T\) of the declared physical model, suppose a boxwise componentwise bound
\[
F=\mathcal T+R,\qquad |R|\le\tau
\]
has been rigorously established, with all source and beam variables retained. Put
\[
F_\lambda=\mathcal T+\lambda R,\qquad
\mathcal C_\lambda=
\{\theta:\ |F_\lambda(\theta)-y|
\le\eta+(1-\lambda)\tau\},\quad0\le\lambda\le1.
\tag{25}
\]
For a fixed certified box and \(\tau\), these sets are nested:
\(\mathcal C_{\lambda_2}\subseteq\mathcal C_{\lambda_1}\) for
\(\lambda_2\ge\lambda_1\). The triangle inequality proves this directly:
\[
|F_{\lambda_1}-y|
\le |F_{\lambda_2}-y|+(\lambda_2-\lambda_1)|R|.
\]
Every exactly compatible parameter remains feasible at the same parameter vector throughout; \(\lambda=1\) recovers the original observation condition.

This supplies a sound hierarchy if the initial approximation gives a complete remainder-thickened cover. Nominal roots of a truncated model alone are not that cover. An unvalidated remainder box remains unresolved. Complex-analytic or real-path Taylor bounds require the relevant entire disk/path to remain on coherent guarded branches; endpoint transmission alone is insufficient. Direct interval bounds on \(F-\mathcal T\) are a possible looser fallback.

Exact geometry affinity (5) allows common four-variable LP profiling throughout a hierarchy that preserves affinity, with boxwise constant error bounds. Approximate duals are only proposals: independently checked support corrections and outward bounds must justify exclusion. Global coverage and useful computational complexity remain open.

## 8. Other physical structure and unavoidable weak excitation

### The exact centered gauge

For a centered axial input, the first prism exits at \((0,0,9)\), with deflection
\[
\Delta_1=\arcsin(n_1\sin a_1)-a_1.
\]
All downstream rays depend on \(n_1,a_1\) only through \(\Delta_1\). Its level set is an exact one-dimensional gauge, with tangent
\[
\frac{da_1}{dn_1}
=-\frac{\sin a_1}
 {n_1\cos a_1-\sqrt{1-n_1^2\sin^2a_1}}.
\tag{26}
\]
Full rank eighteen cannot hold there, and a universal uniquely recovering inverse over a prior containing this stratum is impossible.

For \(q=0\), exact source affinity gives
\(F(g_s,b)-F(g_0,b)=[A(g_s)-A(g_0)]b\)
along this gauge, where \(g_s\) denotes a gauge path, not the air gap. With wedges \(O(\varepsilon)\), source-transport matrices are \(I+O(\varepsilon^2)\). On a compact guarded gauge tube,
\[
\|F(g_s,b)-F(g_0,b)\|
\le C_G|b|\varepsilon^2|s|.
\tag{27}
\]
This creates an indistinguishability scale proportional to
\(\eta/(|b|\varepsilon^2)\), capped by the available gauge segment. A numerical minimax statement needs a specified norm, gauge parameterization and constant \(C_G\).

### An independent off-axis axial-beam rank construction

At zero nominal tilt but unknown nonzero source offset, write complex
\(h_j=\sin(a_j)e^{i\phi_j}\), \(k_j=n_j-1\). Leading second-harmonic coefficients are
\[
S_j=-\overline b\,k_jh_j^2/2,\qquad
M_{ij}=-\overline b\,k_ik_jh_ih_j/(2n_j),\quad i<j.
\]
Their ratios \(M_{ij}^2/(S_iS_j)=k_ik_j/n_j^2\) explicitly recover the indices on the native interval; leading fundamentals then recover gains, distance and gap. The leading complex negative-fundamental tilt coefficient is
\[
C_{-e_j}=-\frac{k_j}{2n_j}bq\,\overline{h_j}
+\text{higher wedge orders}.
\]
It supplies both beam-angle directions when \(b,h_j\ne0\).

With witness speeds \((1,5,25)/20\), a separate 31-column finite dictionary and a scaled-Jacobian argument prove exact local rank eighteen for sufficiently small off-axis wedges. This theorem does not require externally known source position or tilt. Its complete vector cubic recurrence, coefficient inverse, finite-readout contamination and proof are in [physical_vector_inverse.md](theory/physical_vector_inverse.md).

The oblique construction is the primary direction here because it gives an explicit first-order candidate curve, a quadratic ambiguity breaker, and a generically finite four-polynomial formal candidate route. Neither construction establishes globally unique exact finite-angle recovery.

## 9. Canonical model results and their physical limitation

The original research's `risley_lattice/fmodel.py`, especially `_trace_axis`, and the legacy evaluator `reverse_problem_v2/core.py` treat the transverse axes independently. Their mathematically exact results remain useful for that declared model but must not be described as physical vector-optics theorems.

At a centered axial source with rotationally symmetric reference geometry, physical vector optics obeys
\[
Z(\gamma+\tau\mathbf1)=e^{i\tau}Z(\gamma).
\]
Where a full-torus branch exists, its complex Fourier coefficient \(C_m\) vanishes unless \(\sum_i m_i=1\); real coordinates permit weights \(+1\) and \(-1\). The same rule holds for formal Taylor coefficients near a regular centered state. Pure \(\pm3e_i\) and all-plus cubic modes are forbidden there. Sampled temporal aliases do not restore a forbidden torus coefficient.

The canonical cubic proof uses precisely such pure third harmonics. For one centered prism, with \(r=\sin a\), \(s=r(\cos\gamma,\sin\gamma)\), \(k=n-1\), and \(\mathcal B=k(n^2-n+1)/2\),
\[
\text{canonical slope}=ks+\mathcal B(s_x^3,s_y^3)+O(r^5),
\]
\[
\text{physical slope}=ks+\mathcal B|s|^2s+O(r^5).
\]
The artificial canonical term is \((\mathcal B r^3/4)e^{-3i\gamma}\), and the leading maximum slope discrepancy is \(\mathcal B r^3/2\). It is the same order as the canonical identification signal. Small relative error in total deflection therefore does not justify transferring that cubic inverse to physical hardware.

The separately preserved canonical results are:

- Exact paired-axis transfer, affine geometry and polynomial ideal-clock lifting: [algebraic_record.md](theory/algebraic_record.md).
- Exact scalar source projection and finite-noise circuit feasibility: [geometry_circuits.md](theory/geometry_circuits.md), [elimination.md](theory/elimination.md).
- Full eighteen-parameter local rank from the canonical cubic construction: [full18_local_rank.md](theory/full18_local_rank.md).
- Reconstruction of five formal canonical cubic hardware invariants through one scalar closure: [cubic_scalar_inverse.md](theory/cubic_scalar_inverse.md).
- Verified-exclusion and full-cover accuracy frameworks: [global_initialization.md](theory/global_initialization.md), [noise_bounds.md](theory/noise_bounds.md).

An independent strict-monotonicity/connectedness proof of uniqueness for the five formal canonical invariants has been reported, but its full proof is not yet incorporated. This report does not use that reported result. Even such uniqueness would not imply full-record global uniqueness or validity under vector optics.

The complete previous report is retained as [REPORT_v3_archive.md](REPORT_v3_archive.md). Historical solver counts, stored-record ambiguity witnesses and floating-evaluation audits are in [EVIDENCE_REPORT.md](EVIDENCE_REPORT.md). They concern their stated canonical model and timestamp contract; they are not numerical validation of physical vector optics.

## 10. Evidence, provenance and remaining publication work

| Claim | Evidence status | Exact scope |
|---|---|---|
| Physical paired transfer, root-free graph and four-variable geometry reduction | Derived exactly | Coupled-vector equations with declared branch/traversal conditions |
| First-order factorization, orientation bound and trial-beam curve | Derived constructively | Formal first-order data; separate exceptional charts retained |
| First-order rank seventeen at (16) | Exact rational determinant independently reproduced | One proof witness, not a statistical survey |
| Second harmonic breaks the remaining fiber | Exact expansion and both derivative components independently reproduced | Formal quadratic invariant and local fiber |
| Last-prism formal candidate generation | Four-dimensional determinant and coefficient identities independently reproduced | Generic finite fibers with denominator exclusions; exceptional families retained |
| Exact finite-200 physical rank eighteen | Analytic scaled-Jacobian proof with confluent finite dictionary | Sufficiently small nonzero wedges near the witness |
| Component-relative generic rank and finite exact fibers | Analytic and semialgebraic consequences | Exact data, declared chart/component, no uniqueness implication |
| Quantitative native tangent values | Audited witness conversion, rounded values reported | One direction, not a minimum singular value or noise guarantee |
| Data-derived rotor and exact 17+1 uncertainty framework | Sufficient conditional theorems with uniform bounds and all-output obligations | Not evaluated successfully for an actual record |
| Globally complete practical inversion | Open | All admissible branches, weak excitation and supplied precision |

The oblique source memo identifies symbolic checks `proof_checks/vector_firstorder_rank.py`, `proof_checks/vector_secondorder_gauge.py` and `proof_checks/vector_last_prism_rank.py`. This report incorporates their supplied independent audit results; it does not claim they were executed in this workspace. Read-only filename searches did not find these three scripts locally, so runnable copies remain to be included in a standalone reproduction package. The audit is mathematical and symbolic, not a proof-assistant certificate. No claim of literature novelty is made.

A constructive inference method suggested by the theory would:

1. Enclose candidate frequencies, first-order ellipses and the leading baseline from the paired record. Charge frequency uncertainty, harmonic leakage and quadratic DC contamination; retain unresolved collisions and alternative explanations.
2. Determine signed speeds where ellipse determinants exclude zero, and preserve every plausible assignment to physical prism order.
3. Jointly enclose the assigned fundamentals and self-second harmonic. Form \(w,\mathcal I\) only where denominators are certified nonzero.
4. Use the four-polynomial last-prism construction or a complete trial-beam scalar-set construction. Retain all admissible roots, branches and exceptional components; apply the remaining first-order consistency equation and all native bounds.
5. Recover signed wedges, phases and source offsets from (15) and (8). With uncertain coefficients, carry regions rather than silently substituting exact equalities.
6. Correct and enclose surviving regions with the exact vector model, four-variable geometry LP and the conditional exact 17+1 certificates. Verify all 400 residuals and physical constraints. A failed correction remains unresolved unless a sound exclusion establishes incompatibility.
7. Report parameter ranges and distinct surviving explanations. A unique estimate requires local control plus exclusion or enclosure of every other compatible region.

This is a mathematical prescription, not a completed reliable implementation. Finite-angle harmonic contamination, frequency error, quantitative inverse bounds and a practical complete initial cover remain substantial obligations. Near zero wedges, frequency collisions, axial illumination, vanishing perpendicular offset in this chart, and critical optical branches, conditioning or chart validity deteriorates. The exact centered gauge prevents a universal unique inverse over the whole prior.

The user's original Dropbox research and observations remain unchanged. All new documentation is confined to this workspace. Work for this revision was theory integration and read-only review; no parameter campaign, recovery experiment or original project-code execution was performed. Companion notes referenced by relative paths are retained in the research workspace; the main report states the principal constructions, certificates and limitations.

## 11. Related work, proposed contribution and validation gap

Third-order Risley optics is established: Li's 2011 work derives a nonparaxial thick-prism model and a third-order expansion for beam steering. Steering to a requested direction is a different inverse task from blind recovery of eighteen physical parameters from a passive sampled screen trace. [Li, 2011](https://doi.org/10.1364/AO.50.000679)

Numerical identification of wedge angles, refractive indices and installation parameters also predates this work. Li et al. report genetic-algorithm calibration and experimental pointing improvements; incomplete access to that paper's full text prevents an exhaustive fixed-versus-estimated parameter comparison here. Yuan et al.'s 2023 telescope work explicitly calibrates two encoder zeros, two wedges and two indices using gimbal directions and encoder readings, providing a concrete six-parameter comparator with additional measurement information. [Li et al., 2017](https://doi.org/10.1364/AO.56.007358), [Yuan et al., 2023, section 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC10051434/)

Fourier-based target-free main-section zero calibration and three-prism analytical modeling have relevant precedent. These works supply foundations for harmonic analysis and optical modeling; they do not by themselves settle identifiability of the particular blind observation map considered here. [Li and Zhou, 2021](https://doi.org/10.1364/AO.440678), [Li et al., 2017](https://doi.org/10.1364/OE.25.007677), [Qin et al., 2024](https://doi.org/10.1016/j.optcom.2024.130915)

The closest audited calibration comparator is the Livox Mid-40 observation model of Brazeal, Wilkinson and Hochmair. It uses exact three-dimensional vector Snell refraction, two identical prisms sharing wedge/index parameters, azimuth/zenith observations and an extended Kalman filter. Its thirteen model parameters are reduced by fixing air index, wedge angle and one tilt. Its discussion of wedge/index/air correlation is not an invariance proof for the distinct configuration here. That model must not be called merely paraxial, and the present three-prism screen-position theorem does not refute its sensor model or establish its identifiability. [Brazeal et al., 2021](https://doi.org/10.3390/s21144722)

The general machinery also has established origins: separating linear geometry variables from nonlinear optics is related to classical variable projection, and confluent Prony systems already have a local accuracy theory. The specific optical hypotheses, exact hard-band geometry and completeness obligations distinguish the present derivations from simply applying those general methods. [Golub and Pereyra, 1973](https://doi.org/10.1137/0710036), [Batenkov and Yomdin, 2013](https://arxiv.org/abs/1106.1137)

The defensible proposed contribution is the combination of blind finite-record physical eighteen-parameter local identifiability, an explicit leading inverse curve with a quadratic ambiguity breaker and generically finite algebraic candidate construction, exact conditional geometry elimination, excitation obstructions, and data-conditioned native uncertainty certificates. This targeted comparison does not establish first-ever novelty. Source-access and comparison provenance are recorded in [related_work_notes.md](theory/related_work_notes.md).

This remains a **research draft requiring expert review**. Independent symbolic audits are not external peer review. No physical experiment or successfully evaluated useful finite-noise certificate has yet been established. Publication requires stronger literature comparison, quantitative finite-angle/remainder and conditioning certification, disclosure or treatment of all surviving branches, and validation of every original residual constraint. Future empirical publication validation is a separate remaining requirement; the current theory-first, spot-check-only scope does not initiate an experimental campaign.
