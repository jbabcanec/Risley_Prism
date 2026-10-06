# Oblique beam inversion in the full vector prism model

## Main result and scope

This appendix gives a constructive first-order inverse for the physical full vector Snell model with three rotating prisms. All eighteen parameters remain unknown. Generically, the formal first-order record determines a one-dimensional candidate curve. A single physical second harmonic breaks the remaining local ambiguity at an explicit admissible point. A finite-sample argument then proves rank eighteen for the **exact 200-point observation map** at sufficiently small nonzero wedges near that point.

The result is local and generic on the connected analytic branch containing the witness. It does **not** establish global uniqueness, an efficient complete global solver, a quantitative admissible wedge radius, or a uniform noise guarantee over the original parameter box. The first-order curve and second-order coefficient are derived structures, not exact coefficients available without contamination from a finite-wedge trace.

This is a full vector optical result. It must not be confused with the independent-axis canonical equations: centered pure-third-harmonic arguments in that model do not transfer to full vector optics. In the vector model, an exactly centered axial input has an exact first-prism index–wedge gauge. The construction here uses an admissible nonaxial input while still treating its direction and offset as unknown parameters.

The formulas, exact certificates and audit statements below are transcribed from the independently audited source memo supplied for this report. This appendix was prepared by documentation-only integration; no new numerical experiments or proof computations were run for it. Independent mathematical and symbolic audit is distinguished from a proof-assistant certificate.

**Notation and error dependence.** Measurement error is the nonnegative parameter \(\epsilon\), with \(|F_m-y_m|\le\epsilon\); unequal allowances may be written \(\epsilon_m\). The independent, dimensionless small-wedge scale is \(\kappa\), while \(\tau\) denotes a certified model-remainder bound. These quantities are not interchangeable: \(\epsilon\to0\) does not remove finite-angle contamination, and \(\kappa\to0\) does not establish recovery at fixed positive measurement error. Retained domains and certificate constants may depend on \(\epsilon\); statements over an error interval require uniform bounds on that entire family. Generic finite counts concern exact formal coefficients or exact-data fibers, whereas positive-error compatible sets are generally continuous.

The companion [global inverse completeness theorem](global_inverse_completeness.md) establishes an ambiguity-complete exact algorithm in principle for finite exact input encodings and its specified sample times. The [explicit six-root compiler](explicit_six_root_compiler.md) makes the branch-preserving polynomial construction explicit. These results retain exceptional and positive-dimensional fibers; they do not imply uniqueness, a practical complete solver, or useful numerical noise thresholds. Optical admissibility is enforced at measured times; between-sample admissibility is a separate condition.

## 1. Model, units and admissibility

The eighteen native coordinates are three signed speeds, three signed wedge angles, three initial rotor phases, three refractive indices, the workpiece distance \(d\), shared gap \(g\), two beam angles and two source offsets. The bounds are:

| Coordinates | Native bounds |
|---|---|
| Signed speeds \(N_i\) | \([-3.5,3.5]\) Hz |
| Wedge angles \(a_i\) and phases \(\phi_i\) | \([-18,18]\) degrees |
| Refractive indices \(n_i\) | \([1.3,1.8]\) |
| Workpiece distance \(d\) | \([50,200]\) |
| Shared gap \(g\) | \([2,15]\) |
| Beam angles \(\beta_x,\beta_y\) | \([-25,25]\) degrees |
| Source offsets \(b_x,b_y\) | \([-5,5]\) each |

The source distance is 6, and each prism's reference axial thickness is 3. Lengths are native model units, not asserted to be millimetres. Samples occur at

\[
t_k=k/20\ \mathrm{s},\qquad k=0,\ldots,199.
\]

Internal trigonometric calculations and angular derivatives use radians unless a degree conversion is explicitly stated. The theorem concerns these exact ideal sample times.

Let \(q\in\mathbb R^2\) be the incident external transverse direction cosine vector, let

\[
z=\sqrt{1-|q|^2}>0,\qquad t=q/z
\]

be its positive axial direction cosine and transverse slope. The use of \(t\) for the slope in formulas is distinct from the sample-time argument \(t_k\). For the native beam-angle convention used in the tangent calculation,

\[
\beta_x=\arctan(t_x),\qquad \beta_y=\arctan(t_y).
\]

Let \(b\) be the source transverse position and \(B\) the observed zero-wedge workpiece position. The \(i\)-th exit normal is

\[
\left(-s_i,\sqrt{1-|s_i|^2}\right),\qquad
s_i=e_i\begin{pmatrix}\cos\gamma_i\\\sin\gamma_i\end{pmatrix},\qquad
e_i=\sin a_i,
\]

with \(\gamma_i(\tau)=2\pi N_i\tau+\phi_i\). The entrance faces are normal to the axial direction. The external refractive index is one.

Snell square roots use the forward transmitted branch. Total internal reflection and grazing are excluded, and axial propagation is forward. Where physical sequential traversal is part of the model, positive propagation lengths and the stated surface order must also be imposed. Sufficiently small wedges around the witness below retain strict physical margins.

## 2. Exact paired position transfer

For an incoming external transverse direction cosine \(q\), define

\[
H=\sqrt{n^2-|q|^2},\qquad v=q/H,
\]

where \(v\) is the internal slope. Let \(w\) be the outgoing external slope and

\[
m=\frac{s}{\sqrt{1-|s|^2}}
\]

the exit-plane slope. If \(p\) is the entrance position and \(\ell\) is the distance from the exit vertex plane to the next flat plane, exact intersection and propagation give

\[
\boxed{
p_{\rm next}=p+3v+\ell w
+(v-w)\frac{m^T(p+3v)}{1-m^Tv}.
}
\tag{1}
\]

Indeed, the exit plane is \(Z=3+m^Tp_{\rm exit}\), while the internal line is \(p_{\rm exit}=p+vZ\). Therefore

\[
Z=\frac{3+m^Tp}{1-m^Tv},
\]

followed by propagation to \(Z=3+\ell\), which rearranges to (1). This identity retains the actual finite thickness and tilted-surface intersection. Direction refraction alone omits information relevant to inverse conditioning.

For fixed optical parameters, the directions and slopes are independent of \(b,g,d\). Equation (1) is affine in \(p\) and \(\ell\). Composition with \(\ell=(g,g,d)\) consequently preserves exact affine dependence of the final vector trace on \((b_x,b_y,g,d)\). With componentwise bounded measurement error, these four geometry variables admit a conditional linear feasibility problem. The two transverse coordinates are coupled; the canonical independent-axis source-interval polygon does not transfer unchanged.

## 3. First-order matrix derivation

At zero wedge, every external prism direction equals \(q\). Write

\[
H_i=\sqrt{n_i^2-|q|^2},\qquad D_i=d+(3-i)g,
\]

\[
\mathcal T=\frac{I}{z}+\frac{qq^T}{z^3},\qquad
\mathcal V_j=\frac{I}{H_j}+\frac{qq^T}{H_j^3}.
\]

The unperturbed position at exit \(i\) is

\[
R_i=b+q\left[
\frac{6+(i-1)g}{z}+3\sum_{j\le i}\frac1{H_j}
\right].
\tag{2}
\]

Vector Snell differentiation at zero wedge gives outgoing transverse direction change \((H_i-z)s_i\). Differentiating (1) and all downstream flat propagation gives

\[
\boxed{
M_i=(H_i-z)\left[
D_i\mathcal T+3\sum_{j>i}\mathcal V_j
-\frac{qR_i^T}{H_i z}
\right].
}
\tag{3}
\]

The formal first-order record is therefore

\[
F(\tau)=B+\sum_{i=1}^3e_iM_i
\begin{pmatrix}\cos\gamma_i(\tau)\\\sin\gamma_i(\tau)\end{pmatrix}
+O(\|e\|^2).
\tag{4}
\]

The source-intersection term in (3) is essential: it retains the effect of a nonzero source offset.

## 4. Effective-index factorization

Set

\[
h_i=H_i/z,\qquad
L_i=D_i+3\sum_{j>i}h_j^{-1},\qquad
W_i=D_i+3\sum_{j>i}h_j^{-3}.
\]

The baseline and physical indices reconstruct as

\[
B=b+t\left[6+2g+d+3\sum_i h_i^{-1}\right],
\qquad
n_i=\sqrt{\frac{h_i^2+|t|^2}{1+|t|^2}}.
\tag{5}
\]

Substitution of the observed \(B\) into (3) eliminates \(b\) exactly:

\[
\boxed{
M_i=(h_i-1)\left[
L_iI+(W_i+L_i/h_i)tt^T-tB^T/h_i
\right].
}
\tag{6}
\]

Equivalently,

\[
\frac{M_i}{(h_i-1)L_i}
=I+\alpha_i tt^T-\beta_i tB^T,
\qquad
\alpha_i=W_i/L_i+1/h_i,\quad
\beta_i=1/(h_iL_i).
\tag{7}
\]

Here \(\beta_i\) are scalar **shape coefficients**, not the native beam angles \(\beta_x,\beta_y\). Similarly, the \(\alpha_i\) in (7) are shape coefficients, distinct from any later fixed wedge scaling constants.

## 5. Uniform orientation and signed-speed recovery

Rotate coordinates so \(t=(T,0)\), with \(T=|t|\). The normalized first-order matrices are upper triangular. The perpendicular diagonal of \(M_i/(h_i-1)\) is \(L_i\ge50\). Write

\[
B=b+B_0t,\qquad B_0=6+2g+d+3\sum_j h_j^{-1}.
\]

The parallel diagonal is

\[
L_i+W_iT^2-\frac{(B_0-L_i)T^2+Tb_\parallel}{h_i}.
\]

Because

\[
h_i\ge n_i\ge13/10,\qquad
B_0-L_i\le36+90/13,\qquad W_i\ge50,\qquad |b|\le\sqrt{50},
\]

it satisfies the uniform lower bound

\[
\begin{aligned}
\text{parallel diagonal}
&\ge50+\frac{2870}{169}T^2-\frac{50\sqrt2}{13}T\\
&\ge50-\frac{125}{287}>49.5.
\end{aligned}
\tag{8}
\]

Thus \(\det M_i>0\) throughout this bounded source and hardware domain.

For a resolved nonzero frequency, fit its observed cosine–sine coefficient matrix using the positive-frequency convention. Its determinant has the sign of the physical signed speed: wedge amplitude contributes its square, and phase contributes a rotation. This determines the signed rotation direction at first order. Zero wedges, zero speeds and unresolved frequency collisions are exceptional cases and must be retained separately.

## 6. Explicit inversion conditional on a trial beam slope

Let \(C_i\) be the oriented first-order coefficient matrix: start with the observed cosine–sine matrix at positive frequency, and, after determining a negative physical speed, flip its sine column. Leave the matrix unchanged for positive speed. This convention gives \(C_i=e_iM_iR(\phi_i)\). Set \(S_i=C_iC_i^T\). For a trial \(t\) with \(T>0\), rotate \(S_i\) and \(B\) into the beam-parallel/perpendicular frame. This chart requires \(B_\perp\ne0\). Define

\[
c_i=\frac{(S_i)_{12}}{(S_i)_{22}},\qquad
r_i=\frac{\sqrt{\det S_i}}{(S_i)_{22}}>0.
\]

Then

\[
\boxed{
\beta_i=-\frac{c_i}{TB_\perp},\qquad
\alpha_i=\frac{r_i-1-(B_\parallel/B_\perp)c_i}{T^2}.
}
\tag{9}
\]

These quantities are independent of wedge amplitude and initial phase. Reject a trial if any shape coefficient \(\beta_i\le0\) or if \(\alpha_3\le1\), since physical \(h_i>1\) implies these necessary conditions. Recover hardware downstream:

\[
h_3=\frac1{\alpha_3-1},\qquad
d=\frac1{h_3\beta_3}.
\tag{10}
\]

Explicitly reject \(h_3\le1\); the earlier condition \(\alpha_3>1\) alone does not ensure \(h_3>1\). On this physical branch put \(E_3=3(h_3^{-3}-h_3^{-1})<0\). Since \(\beta_2>0\), the second-prism lever \(L_2\) is the unique positive root of

\[
\beta_2L_2^2-(\alpha_2-1)L_2+E_3=0.
\tag{11}
\]

Next recover

\[
h_2=\frac1{\beta_2L_2},\qquad
g=L_2-d-3/h_3,
\]

Reject \(h_2\le1\) before defining \(E_2=3(h_2^{-3}-h_2^{-1})<0\). Continue with

\[
L_1=d+2g+3/h_2+3/h_3,\qquad
h_1=\frac1{\beta_1L_1}.
\tag{12}
\]

One scalar consistency equation remains on the two trial beam coordinates:

\[
\boxed{
\alpha_1-1-\frac{E_2+E_3}{L_1}-\beta_1L_1=0.
}
\tag{13}
\]

Recover \(n_i\) and \(b\) from (5). Wedge and phase reconstruct from

\[
M_i^{-1}C_i=e_iR(\phi_i),
\tag{14}
\]

where \(R(\phi_i)\) is the planar rotation matrix. The signed wedge and phase must satisfy their original bounds. The two signs of \(e_i\) differ by a phase shift of \(\pi\); the narrow native phase interval selects at most one. Filter all index, distance, gap, source and beam bounds. Retain every admissible prism assignment: physical order is not a permutation symmetry.

The formal first-order inverse therefore reduces generically to a curve in the two-dimensional trial-beam plane while keeping beam parameters unknown. The scalar consistency zero set may be singular or disconnected, and all admissible branches must be retained. The explicit \(h_2,h_3>1\) checks enforce necessary physical conditions already implied by the original index bounds; they do not narrow the prior. The charts \(T=0\) and \(B_\perp=0\) require separate treatment.

## 7. Exact first-order rank-seventeen witness

For fixed beam direction, the six ellipse-shape quantities can be written

\[
A_i=T^2(W_i/L_i+1/h_i),\qquad
Q_i=T/(h_iL_i).
\tag{15}
\]

Consider

\[
(T,h_1,h_2,h_3,g,d)
=\left(\frac13,\frac32,\frac75,\frac53,3,100\right).
\tag{16}
\]

The corresponding physical indices satisfy

\[
(n_1^2,n_2^2,n_3^2)
=\left(\frac{17}{8},\frac{233}{125},\frac{13}{5}\right).
\]

Choose \(b=(1,2)\) and the beam direction along \(x\). Then

\[
B=\left(\frac{1411}{35},2\right).
\]

The nonzero beam angle is \(\arctan(1/3)\approx18.435^\circ\), inside the original \(25^\circ\) bound. The wedge and phase bounds remain \(18^\circ\).

With rows \((A_1,Q_1,A_2,Q_2,A_3,Q_3)\) and columns \((T,h_1,h_2,h_3,g,d)\), the exact determinant at (16) is

\[
\boxed{
\frac{9079553843083}
{47292531300183951601092000000}\ne0.
}
\tag{17}
\]

Six shape directions, six wedge amplitude/phase directions, two baseline directions and three speeds give seventeen independent first-order observables. Seventeen is also the maximum: three real \(2\times2\) harmonic matrices, two baseline coordinates and three speeds contain only seventeen numbers. Thus the remaining local first-order fiber is one-dimensional.

The source audit independently reconstructed (17) exactly. This was a single rational proof check, not a parameter survey.

## 8. A physical second harmonic that breaks the remaining fiber

In this section use complex notation \(T=t_x+it_y\) and \(B=B_x+iB_y\); the complex \(T\) replaces the real magnitude used in the preceding beam-aligned formulas. Set \(h=h_3\), \(k=h-1\). Isolate the positive rotor component by the formal complex tilt

\[
u=(\sigma/2,-i\sigma/2),\qquad u^Tu=0.
\]

Put

\[
a_0=\overline T/2,\qquad c_0=(\overline B-d\overline T)/2.
\]

The normalized outgoing vertical direction and transverse slope are exactly

\[
Z(\sigma)=a_0\sigma+
\sqrt{1-2ha_0\sigma+(a_0\sigma)^2},
\]

\[
Q(\sigma)=\frac{T+[h-Z(\sigma)]\sigma}{Z(\sigma)}.
\tag{18}
\]

With only the third tilt active, the screen point is

\[
F(\sigma)=B+d[Q(\sigma)-T]
+\left(T/h-Q(\sigma)\right)
\frac{c_0\sigma}{1-a_0\sigma/h}.
\tag{19}
\]

Expand \(F=B+U_+\sigma+C_{+2}\sigma^2+O(\sigma^3)\). The coefficients are

\[
\boxed{
U_+=k[d(1+a_0T)-Tc_0/h],
}
\tag{20}
\]

\[
\boxed{
C_{+2}=k\left[
dha_0+\frac{d(3h-1)}2a_0^2T
-c_0(1+a_0T+Ta_0/h^2)
\right].
}
\tag{21}
\]

At \(T=0\), equation (21) gives \(C_{+2}=-k\overline B/2\), the physical off-axis quadratic coefficient. The normalized observable

\[
\mathcal I_3=C_{+2}/U_+^2
\tag{22}
\]

cancels the signed wedge amplitude and phase. Positive \(\det M_3\) implies \(U_+\ne0\): its forward and backward complex coefficients satisfy

\[
|U_+|^2-|U_-|^2=\det M_3>0.
\]

These are formal coefficient identities. They are not an assertion that the corresponding uncontaminated coefficients are directly observed at finite wedge angle.

## 9. Exact derivative along the first-order ambiguity

Let \(\psi\) be beam orientation and take \(\psi'=1\) radian. Hold the observed baseline, frequencies and every first-order harmonic matrix fixed. At \(\psi=0\), their normalized covariances are

\[
S_i=\begin{pmatrix}r_i^2+c_i^2&c_i\\c_i&1\end{pmatrix},
\qquad r_i=1+A_i-B_xQ_i,\quad c_i=-B_yQ_i.
\]

Rotating the fixed observed covariance gives

\[
c_i'=1-r_i^2+c_i^2,\qquad r_i'=2r_ic_i.
\]

Since \(B_\parallel'=B_y\) and \(B_\perp'=-B_x\),

\[
\begin{aligned}
A_i'={}&2r_ic_i-[1+(B_x/B_y)^2]c_i\\
&-(B_x/B_y)(1-r_i^2+c_i^2),
\end{aligned}
\]

\[
Q_i'=-\frac{1-r_i^2+c_i^2}{B_y}
-\frac{c_iB_x}{B_y^2}.
\tag{23}
\]

Solving the exact six-dimensional Jacobian system gives the unique compensating tangent in \((T,h_1,h_2,h_3,g,d)\). Amplitude and phase compensation follows by keeping \(C_i\) fixed; normalization in \(\mathcal I_3\) incorporates it automatically.

At (16), the exact derivative satisfies

\[
\boxed{
\operatorname{Re}\mathcal I_3'=
\frac{18848880260984398987381756622941300813984492817}
{49946733938194977003395356774800375463944000000}>0,
}
\tag{24}
\]

\[
\boxed{
\operatorname{Im}\mathcal I_3'=
\frac{1444583062125308617841515913871558551982883}
{79280530060626947624437074245714881688800000}>0.
}
\tag{25}
\]

The source audit independently reproduced both values from the exact paired equations and an independently assembled shape-preserving tangent. Therefore one real projection of this physical second harmonic breaks the single first-order ambiguity locally.

## 10. A generically finite polynomial candidate construction

The last prism supplies a further formal reduction. Let \(U,V\) be its intrinsic unit-wedge coefficients at the positive and negative signed rotor frequencies, and \(C\) its intrinsic positive self-second-harmonic coefficient. With \(\xi=e_3e^{i\phi_3}\), the observed formal coefficients satisfy
\[
U_{\rm obs}=\xi U,\qquad V_{\rm obs}=\overline\xi V,\qquad
C_{\rm obs}=\xi^2 C.
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

For each regular-chart candidate, reconstruct \(h_2,g,h_1\) using Section 6, enforce its remaining scalar consistency equation, and apply every original source, index, wedge, phase and physical branch constraint.

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

## 11. Exact finite-sample full-rank theorem

Choose signed speeds

\[
(N_1,N_2,N_3)=(1,7,49)/20\ \mathrm{Hz},
\]

initial phases zero, and nonzero wedge sines \(e_i=\kappa a_i^*\), with fixed nonzero \(a_i^*\). The constants \(a_i^*\) are wedge scaling constants, not the shape coefficients in (7).

The integer harmonic indices \(m\in\mathbb Z^3\) with \(|m|_1\le2\) give 25 distinct sampled nodes

\[
z_m=\exp\!\left[\frac{2\pi i}{400}
(m_1+7m_2+49m_3)\right].
\]

Six repeated fundamental nodes account for the frequency-derivative columns

\[
t_k\exp(\pm2\pi iN_it_k).
\]

The resulting 31-column confluent Vandermonde matrix has full rank on the original 200 samples. A difference of two degree-at-most-two indices has \(\ell_1\) norm at most four and cannot satisfy a nontrivial base-seven relation \(m_1+7m_2+49m_3=0\). The corresponding sampled integer numerators cannot differ by 400, so no sampling alias occurs. The distinct-node confluent Vandermonde argument proves independence already on its first 31 rows.

Use fixed real linear functionals of the actual sampled outputs to select the seventeen first-order channels and a real projection of the second harmonic. The fixed extractor is a proof device at the witness; it does not assume unknown speeds are already known in an inference procedure. Along the first-order fiber,

\[
U_{\rm obs}=e_3e^{i\phi_3}U_+\quad\text{is fixed},
\]

\[
\frac{dC_{\rm obs}}{d\psi}
=U_{\rm obs}^2\frac{d\mathcal I_3}{d\psi}\ne0.
\tag{26}
\]

In scaled wedge coordinates, rescale two baseline columns by one, fifteen first-order columns by \(\kappa^{-1}\), and the remaining fiber column by \(\kappa^{-2}\). The selected \(18\times18\) matrix tends to a block-triangular matrix with nonzero diagonal blocks. Higher-order \(C^1\) remainders vanish after this rescaling. Analyticity and strict branch margins hold near the zero-wedge witness; continuity then transfers the limiting rank to the exact model.

**Theorem.** At the stated admissible oblique family, for all sufficiently small positive \(\kappa\), the exact physical-vector \(400\times18\) observation Jacobian has rank eighteen.

No continuous-time derivatives or unobserved Taylor coefficients are used as measurements in this proof. The coefficients organize a finite-dimensional limiting Jacobian argument on the actual 200 samples.

Analyticity gives generic local identifiability on the connected analytic component containing the witness. A semialgebraic lift gives generic finite fibers within a suitable injective chart/component after excluding critical and admissible boundary images. Neither argument proves global uniqueness or connectivity of every physical branch. No numerical upper threshold for the admissible \(\kappa\) is supplied.

## 12. Native tangent and conditioning

A nonzero unnormalized derivative is not a robust recovery constant. At the witness, with \(\psi'=1\) radian and \(N_i'=0\), the converted native tangent is approximately:

| Coordinate group | Tangent per radian of \(\psi\) |
|---|---|
| Refractive indices | \((-101.7158822,-89.84814165,-123.7641878)\) |
| Shared gap \(g\) | \(-559.7542323\) |
| Workpiece distance \(d\) | \(981.7869465\) |
| Beam angles, degrees | \((19.82590088,19.09859317)\) |
| Source offsets | \((-145.6929506,-39.31428571)\) |
| Initial rotor phases, degrees | \((-5.814083335,-5.595777712,-5.566239507)\) |

For equal wedge sines \(e_i=\kappa\), wedge-angle derivatives in degrees are

\[
(12494.56570,13624.89758,10861.29387)
\frac{\kappa}{\sqrt{1-\kappa^2}}.
\tag{27}
\]

The limiting sup norm in the original mixed native-coordinate convention is about \(981.7869465\). This norm mixes degrees, index units and native lengths because the original requested coordinate tolerance does so; it is not dimensionless physical conditioning.

The raw normalized invariant derivative has magnitude about \(0.3778192691\) per radian of \(\psi\), but only about \(0.0003848281650\) per unit of the native sup-normalized tangent. At the witness,

\[
U_+\approx69.904973545+0.133333333i,
\qquad |U_+|^2\approx4886.7231041.
\]

Thus the corresponding self-second-harmonic directional gain is asymptotically about

\[
1.880548685\,\kappa^2
\]

per native sup unit. This is **one directional gain**, not the smallest singular value, a full inverse Lipschitz constant, or an accuracy guarantee for every coordinate.

At this regular point, the native singular-direction hierarchy has five order-one directions, twelve order-\(\kappa\) directions and one order-\(\kappa^2\) direction, with chart-dependent constants. The five strong directions comprise three wedge-amplitude directions and two baseline combinations. Weak directions are mixtures of original parameters; the hierarchy does not assign an independent accuracy order to each native coordinate.

## 13. Constructive use and unresolved steps

A constructive inference method suggested by the theory would:

1. Enclose candidate frequencies, first-order ellipses and the leading baseline from the paired record. Charge frequency uncertainty, harmonic leakage and quadratic DC contamination; retain unresolved collisions and alternative explanations.
2. Determine signed speeds where ellipse determinants exclude zero, and preserve every plausible assignment to physical prism order.
3. Jointly enclose the assigned fundamentals and self-second harmonic. Form \(w,\mathcal I\) only where denominators are certified nonzero.
4. Use the four-polynomial last-prism construction or a complete trial-beam scalar-set construction. Retain all admissible roots, branches and exceptional components; apply the remaining first-order consistency equation and all native bounds.
5. Recover signed wedges, phases and source offsets from (14) and (5). With uncertain coefficients, carry regions rather than silently substituting exact equalities.
6. Correct and enclose surviving regions with the exact vector model, four-variable geometry LP and the conditional exact 17+1 certificates described in [physical_inverse_certificates.md](physical_inverse_certificates.md). Verify all 400 residuals and physical constraints. A failed correction remains unresolved unless a sound exclusion establishes incompatibility.
7. Report parameter ranges and distinct surviving explanations. A unique estimate requires local control plus exclusion or enclosure of every other compatible region.

This is a mathematical construction, not a claim that a complete practical implementation or validated finite-noise initializer has been delivered. At finite wedges, harmonics contain higher-order contamination, while unknown-frequency error affects demixing. A validated remainder enclosure and complete candidate cover are needed before claiming an ambiguity-complete inverse.

The exact vector geometry remains affine in the two source coordinates, \(g\) and \(d\), so a four-variable LP can profile it under componentwise error bounds. Canonical source-interval polygon formulas require a new derivation under transverse coupling. Every proposed candidate must satisfy all original parameter bounds, optical branch requirements and all 400 output constraints; a selected projected root alone is insufficient.

The present proof does not supply a useful uniform finite-noise radius, prove uniqueness of all curve intersections, exclude disconnected exact branches, or establish a practical complete full-prior algorithm. Conditioning deteriorates near zero wedges, frequency collisions, near-axial illumination, vanishing perpendicular offset in this chart and critical optical branches. The exact centered first-prism gauge remains a real obstruction to a universal unique-recovery theorem.

Exact global completeness in principle is supplied by [global_inverse_completeness.md](global_inverse_completeness.md) and [explicit_six_root_compiler.md](explicit_six_root_compiler.md), subject to their finite exact encoding and sampling assumptions. The practical bottleneck remains a computationally useful full-prior cover and reliable refinement of weak or singular branches. Local full rank, formal coefficient inversion, global set completeness and exact finite-angle uniqueness are distinct claims. Universal uniqueness on the original prior is obstructed by the centered gauge.

## 14. Reproducibility and audit status

The supplied source material includes three primary physical-vector proof-check scripts:

- `proof_checks/vector_firstorder_rank.py`: stated to reproduce the exact six-dimensional determinant (17).
- `proof_checks/vector_secondorder_gauge.py`: stated to reconstruct the same first-order null tangent, the exact nonzero second-harmonic derivative and the native-coordinate conversion.
- `proof_checks/vector_last_prism_rank.py`: named for the independently audited last-prism coefficient identities and exact four-dimensional determinant (L4).

The former missing-source gap is closed. The [proof-check README](proof_checks/README.md) and [manifest](proof_checks/manifest.json) document seven supplied source files: these three primary physical-vector checks, their three independent audit counterparts, and one separately scoped canonical-model check. All seven saved SHA256 hashes match the supplied hashes according to the manifest. Packaging verified byte identity, not mathematical correctness. The source files remain unchanged, including their historical use of `epsilon` for wedge scaling; the updated mathematical documentation uses \(\kappa\) for that scale. No script was executed, imported, compiled or mathematically retested during this documentation update.

According to the supplied independent audit, separate calculations reproduced the first-order determinant, exact self-harmonic expansion, both real and imaginary gauge derivatives, orientation bound, finite-sample block argument, and last-prism coefficient identities and determinant. These are small symbolic rational checks at the explicitly stated witness, not statistical tests or parameter campaigns. The audit is mathematical and symbolic, not a proof-assistant certificate, and no claim of literature novelty is made.

| Statement | Status and limit |
|---|---|
| Exact paired transfer and affine four-variable geometry | Derived exact-model identities on the admissible branch |
| Effective-index factorization and trial-beam curve | Derived first-order inverse structure; not an exact finite-wedge inverse |
| Uniform orientation inequality | Derived bound, reported independently checked |
| Rank-seventeen determinant | Exact rational certificate, reported independently reproduced |
| Nonzero second-harmonic fiber derivative | Exact rational certificates, reported independently reproduced |
| Last-prism four-polynomial candidate construction | Formal coefficient identities and exact determinant, reported independently reproduced; generically finite for the audited baseline and generic joint-family baselines, not every baseline |
| Coarse 1296 isolated-complex-candidate bound | Generic formal algebraic bound after excluding denominator components; neither uniqueness nor a bound on noisy compatible systems |
| Exact finite-200 rank eighteen | Finite-sample analytic argument from the audited limiting blocks; no numerical wedge threshold |
| Generic local identifiability | On the analytic component containing the witness |
| Generic finite fibers | Qualified by a suitable injective semialgebraic chart/component and exclusion of critical/boundary images |
| Native directional gain | Asymptotic one-direction calculation, not an inverse stability guarantee |
| Exact global set completeness | Established in principle by the companion theorem and explicit compiler for their finite exact encodings and sample times; no practical complete solver supplied |
| Universal exact-data uniqueness | False on the original prior because of the centered gauge; local uniqueness remains separately scoped |
| Useful numerical noise guarantees | Not established for the actual record |
