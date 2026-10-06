# Oblique beam inversion in the full vector prism model

## Main result and scope

This memo gives a constructive first order inverse for the physical full vector Snell model with three rotating prisms. All eighteen parameters remain unknown. The first order record determines a one dimensional candidate curve generically. A single physical second harmonic breaks that remaining local ambiguity at an explicit admissible point. A finite sample argument then proves full rank eighteen for the exact 200 point observation map at sufficiently small nonzero wedges near that point.

The result is local and generic on the connected analytic branch containing the witness. It does not establish global uniqueness, an efficient complete global solver, a quantitative admissible wedge radius, or a uniform noise guarantee over the original parameter box. The first order curve and second order coefficient are derived structures, not exact coefficients available without contamination from a finite wedge trace.

This is a full vector optical result. It must not be confused with the independent axis canonical equations: centered pure third harmonic arguments in that model do not transfer to full vector optics. In the vector model an exactly centered axial input has an exact first prism index versus wedge gauge. The oblique construction below deliberately uses an admissible nonaxial input while still treating its direction and offset as unknown parameters.

## Model and units

The eighteen native coordinates are three signed speeds, three signed wedge angles, three initial rotor phases, three refractive indices, the workpiece distance d, shared gap g, two beam angles, and two source offsets. The bounds are

- signed speeds in [-3.5,3.5] Hz;
- wedge angles and initial phases in [-18,18] degrees;
- refractive indices in [1.3,1.8];
- d in [50,200] and g in [2,15];
- beam angles in [-25,25] degrees;
- source offsets in [-5,5] on each axis.

Source distance is 6 and each prism's reference axial thickness is 3. Lengths are native model units, not asserted to be millimetres. Samples occur at t_k=k/20 seconds for k=0,...,199. Internal trigonometric calculations and angular derivatives use radians unless a degree conversion is explicitly stated.

Let q be the incident external transverse direction cosine vector, z=sqrt(1-|q|^2)>0, and t=q/z the transverse slope. For the native beam angle convention used in the tangent calculation, beta_x=atan(t_x), beta_y=atan(t_y). Let b be the source transverse position, and B the leading zero wedge workpiece position. B is a model coefficient inferred from the passive record with a contamination enclosure; it is not an additional controlled zero wedge measurement. The measured DC generally includes a quadratic offset from B.

The i-th exit normal is (-s_i,sqrt(1-|s_i|^2)), with s_i=e_i(cos gamma_i,sin gamma_i), e_i=sin(a_i), and gamma_i=2 pi N_i t+phi_i. Entrance faces are normal to the axial direction. Snell square roots use the transmitted forward branch. Small wedges around the witness retain strict physical margins.

## Exact paired position transfer

For an incoming external transverse direction q, define H=sqrt(n^2-|q|^2), internal slope v=q/H, outgoing external slope w, and exit plane slope m=s/sqrt(1-|s|^2). If p is the entrance position and ell is the distance from the exit vertex plane to the next flat plane, the exact exit intersection and propagation give

\[
p_{\rm next}=p+3v+\ell w
 +(v-w)\frac{m^T(p+3v)}{1-m^Tv}.
\]

To check this identity, the exit plane is Z=3+m^T p_exit and the internal line is p_exit=p+v Z. Hence Z=(3+m^Tp)/(1-m^Tv), followed by propagation to Z=3+ell. This identity preserves the actual finite thickness and tilted surface intersection. For sequential physical traversal, also enforce the appropriate surface order inequalities: the internal intersection height Z is positive and the following forward flight has 3+ell-Z>0, together with transmitted branch and denominator conditions. Positive refraction roots alone do not certify surface order. Direction refraction alone would omit information relevant to inverse conditioning.

## First order matrix derivation

At zero wedge all external prism directions equal q. Write

\[
H_i=\sqrt{n_i^2-|q|^2},\quad
D_i=d+(3-i)g,
\]
\[
\mathcal T=I/z+qq^T/z^3,\quad
\mathcal V_j=I/H_j+qq^T/H_j^3.
\]

The unperturbed position at exit i is

\[
R_i=b+q\left[\frac{6+(i-1)g}{z}
          +3\sum_{j\le i}\frac1{H_j}\right].
\]

Vector Snell differentiation at zero wedge gives the outgoing transverse direction change (H_i-z)s_i. Differentiating the exact paired position transfer and all downstream flat propagation therefore gives

\[
\boxed{M_i=(H_i-z)
 \left[D_i\mathcal T+3\sum_{j>i}\mathcal V_j
             -\frac{qR_i^T}{H_i z}\right].}
\]

Consequently the first order record is

\[
F(t)=B+\sum_i e_iM_i
 \begin{pmatrix}\cos\gamma_i(t)\\\sin\gamma_i(t)\end{pmatrix}
 +O(\|e\|^2).
\]

The source intersection term in M_i is essential. It is the term that retains the effect of a nonzero source offset.

## Effective index factorization

Set h_i=H_i/z and

\[
L_i=D_i+3\sum_{j>i}h_j^{-1},\qquad
W_i=D_i+3\sum_{j>i}h_j^{-3}.
\]

The baseline and physical indices reconstruct as

\[
B=b+t\left[6+2g+d+3\sum_i h_i^{-1}\right],
\qquad
n_i=\sqrt{\frac{h_i^2+|t|^2}{1+|t|^2}}.
\]

Substituting the observed B into the matrix eliminates b exactly:

\[
\boxed{M_i=(h_i-1)
 [L_iI+(W_i+L_i/h_i)tt^T-tB^T/h_i].}
\]

Equivalently,

\[
\frac{M_i}{(h_i-1)L_i}
 =I+\alpha_i tt^T-\beta_i tB^T,
\quad
\alpha_i=W_i/L_i+1/h_i,
\quad
\beta_i=1/(h_iL_i).
\]

The beta_i in this equation are scalar shape coefficients, not native beam angles.

## Uniform orientation and signed speed recovery

Rotate coordinates so t=(T,0), T=|t|. The normalized first order matrices are upper triangular. The perpendicular diagonal of M_i/(h_i-1) is L_i>=50. Writing B=b+B_0 t, the parallel diagonal is

\[
L_i+W_iT^2-[(B_0-L_i)T^2+Tb_\parallel]/h_i.
\]

Since h_i>=n_i>=13/10, B_0-L_i<=36+90/13, W_i>=50, and |b|<=sqrt(50),

\[
\text{parallel diagonal}\ge
50+\frac{2870}{169}T^2-\frac{50\sqrt2}{13}T
\ge50-\frac{125}{287}>49.5.
\]

Thus det(M_i)>0 throughout this bounded source and hardware domain.

For a resolved nonzero frequency, fit its observed cosine and sine coefficient matrix using the positive frequency convention. Its determinant has the sign of the physical signed speed, since wedge amplitude contributes its square and phase contributes a rotation. This resolves the signed rotation direction at first order. Zero wedges, zero speeds, and unresolved frequency collisions are exceptional cases and must be retained separately.

## Explicit inversion conditional on a trial beam slope

Let C_i be the oriented first order coefficient matrix and S_i=C_i C_i^T. For a trial t with T>0, rotate S_i and B into the beam parallel and perpendicular frame. Require B_perp!=0 in this chart and define

\[
c_i=(S_i)_{12}/(S_i)_{22},\qquad
r_i=\sqrt{\det S_i}/(S_i)_{22}>0.
\]

Then

\[
\boxed{\beta_i=-\frac{c_i}{TB_\perp},\qquad
\alpha_i=\frac{r_i-1-(B_\parallel/B_\perp)c_i}{T^2}.}
\]

These quantities are independent of wedge amplitude and initial phase. Reject a trial if any shape coefficient beta_i is nonpositive or if alpha_3<=1; physical h_i>1 implies these necessary conditions. Recover the hardware downstream:

\[
h_3=\frac1{\alpha_3-1},\qquad
 d=\frac1{h_3\beta_3}.
\]

Put E_j=3(h_j^{-3}-h_j^{-1})<0. The second prism lever L_2 is the unique positive root of

\[
\beta_2 L_2^2-(\alpha_2-1)L_2+E_3=0.
\]

Next,

\[
h_2=\frac1{\beta_2L_2},\qquad
 g=L_2-d-3/h_3,
\]
\[
L_1=d+2g+3/h_2+3/h_3,\qquad
h_1=\frac1{\beta_1L_1}.
\]

One scalar consistency equation remains on the two trial beam coordinates:

\[
\boxed{\alpha_1-1-\frac{E_2+E_3}{L_1}-\beta_1L_1=0.}
\]

Recover n_i and b from the formulas above. Recover wedge and phase from

\[
M_i^{-1}C_i=e_iR(\phi_i).
\]

The signed wedge and phase must satisfy their original bounds. The two possible signs for e_i differ by a phase shift of pi; the narrow native phase interval selects at most one. Also filter all index, distance, gap, source and beam bounds. Retain all admissible prism assignments; physical order is not a permutation symmetry.

Thus the formal first order record reduces the full inverse to a curve in the two dimensional trial beam plane, rather than fixing beam parameters externally. Charts with T=0 or B_perp=0 need separate treatment.

## Exact first order rank seventeen witness

For a fixed beam direction, the six independent ellipse shape quantities can be written

\[
A_i=T^2(W_i/L_i+1/h_i),\qquad Q_i=T/(h_iL_i).
\]

The point

\[
(T,h_1,h_2,h_3,g,d)=(1/3,3/2,7/5,5/3,3,100)
\]

has

\[
(n_1^2,n_2^2,n_3^2)=(17/8,233/125,13/5).
\]

Choose source b=(1,2), beam direction along x, and therefore

\[
B=(1411/35,2).
\]

The beam angle is atan(1/3), approximately 18.435 degrees, inside the original 25 degree beam bound. Wedge and phase bounds remain 18 degrees.

The exact determinant, with rows (A_1,Q_1,A_2,Q_2,A_3,Q_3) and columns (T,h_1,h_2,h_3,g,d), is

\[
\boxed{\frac{9079553843083}
{47292531300183951601092000000}\ne0.}
\]

Six shape directions, six wedge amplitude and phase directions, two baseline directions, and three speeds give seventeen independent first order observables. Seventeen is also the maximum: three real 2 by 2 harmonic matrices, two baseline coordinates and three speeds contain only seventeen numbers. The remaining local first order fiber has dimension one.

The determinant was independently reconstructed exactly. This was a single rational proof check, not a parameter survey.

## A physical second harmonic that breaks the remaining fiber

Use complex notation T=t_x+i t_y and B=B_x+i B_y, and set h=h_3, k=h-1. Isolate the positive rotor component by the formal complex tilt u=(sigma/2,-i sigma/2); then u^Tu=0. Put

\[
a_0=\overline T/2,\qquad
c_0=(\overline B-d\overline T)/2.
\]

The normalized outgoing vertical direction and transverse slope are exactly

\[
Z(\sigma)=a_0\sigma+
 \sqrt{1-2ha_0\sigma+(a_0\sigma)^2},
\]
\[
Q(\sigma)=\frac{T+[h-Z(\sigma)]\sigma}{Z(\sigma)}.
\]

The screen point with only the third tilt active is

\[
F(\sigma)=B+d[Q(\sigma)-T]
 +(T/h-Q(\sigma))\frac{c_0\sigma}{1-a_0\sigma/h}.
\]

Expanding F=B+U_+ sigma+C_{+2} sigma^2+O(sigma^3) gives

\[
\boxed{U_+=k[d(1+a_0T)-Tc_0/h],}
\]
\[
\boxed{C_{+2}=k\left[
 dha_0+\frac{d(3h-1)}2a_0^2T
 -c_0(1+a_0T+Ta_0/h^2)\right].}
\]

At T=0, C_{+2}=-k conjugate(B)/2, the physical off axis quadratic coefficient. The normalized observable

\[
\mathcal I_3=C_{+2}/U_+^2
\]

cancels the signed wedge amplitude and phase. Positive det(M_3) implies U_+ is nonzero, since its forward and backward complex coefficients satisfy |U_+|^2-|U_-|^2=det(M_3).


## Last prism algebraic candidate construction

The last prism provides a further reduction. Its complex first order coefficients U and V multiply the positive and negative signed rotor frequencies. The data invariants

\[
w=V/\overline U,\qquad \mathcal I=C/U^2
\]

cancel wedge amplitude and initial phase. Here C is the positive self second harmonic coefficient. In observed coefficients, use the same signed frequency convention in all three quantities.

Keep the leading baseline B fixed temporarily. Put T=t_x+i t_y, H=h_3, k=H-1 and R=T\overline T. Define

\[
A=2Hd+d(H+1)R-T\overline B,
\]
\[
V_0=d(H+1)T^2-TB,
\]
\[
\begin{aligned}
C_0={}&4dH^3\overline T+dH^2(3H-1)R\overline T\\
&-4H^2(\overline B-d\overline T)
-2(H^2+1)R(\overline B-d\overline T).
\end{aligned}
\]

The exact formal coefficients obey U=kA/(2H), V=kV_0/(2H), and C=kC_0/(8H^2). Consequently the four real polynomial equations are

\[
\boxed{\operatorname{Re}(V_0-w\overline A)=0,
\quad\operatorname{Im}(V_0-w\overline A)=0,}
\]
\[
\boxed{\operatorname{Re}(C_0-2\mathcal I(H-1)A^2)=0,
\quad\operatorname{Im}(C_0-2\mathcal I(H-1)A^2)=0.}
\]

Their real unknowns are (t_x,t_y,H,d), and their total degrees are at most (4,4,9,9). Retain H>1, A!=0, the original beam and distance bounds, and

\[
1.3^2(1+R)\le H^2+R\le1.8^2(1+R).
\]

For every retained candidate, reconstruct h_2,g,h_1 by the first order triangular formulas and apply their consistency equation, source, index, wedge, phase and physical branch bounds. Denominator clearing does not authorize retaining spurious roots with A=0.

At the existing witness, the Jacobian of (Re w, Im w, Re I, Im I) with respect to (t_x,t_y,H,d) has exact determinant

\[
\frac{19617689107775384734706249006250000}
{141672457738889533504325313445087046934090001}>0.
\]

Thus for the audited fixed baseline B=(1411/35,2), this four dimensional rational map is generically finite. The same holds for generic baselines in the joint baseline family; it is not proved here for every arbitrary fixed baseline. For complex elimination, saturate away H(H-1)A conjugate(A)=0; on real candidates H>1 and A!=0 enforce these exclusions. After excluding denominator components, a coarse generic isolated complex candidate bound is 4 times 4 times 9 times 9 = 1296. It does not prove uniqueness or finite fibers at exceptional invariant values. It converts formal leading candidate generation into a finite algebraic problem generically, rather than requiring a continuous trial beam curve search.

Degenerate outputs require a different result type. Zero wedge amplitude makes the normalized observed ratios undefined; zero or colliding rotor frequencies can prevent unique harmonic labeling. The trial beam ellipse chart also fails at T=0 or B_perp=0 even though the four polynomial equations can still be evaluated there. Exceptional polynomial fibers may contain curves or higher dimensional sets. Retain these components and their original inequalities, or pass them to the exact semialgebraic inverse; do not discard them because a generic finite root routine is inapplicable. With uncertain coefficient intervals the feasible set is generally continuous, so the 1296 bound does not bound the number of compatible noisy parameter vectors. In regular chart candidates, reconstruct hardware as described and reject only on an actual violated original constraint.

This additional exact witness uses the same physical point as the first order and quadratic gauge checks. The coefficient identities and the exact four dimensional determinant were independently reconstructed and matched. No new parameter search is involved. The polynomial construction concerns formal leading coefficients; approximate invariants from a finite wedge record still require a contamination enclosure and exact correction. In particular, treating the measured DC as exact B would introduce an unaccounted bias.

## Exact derivative along the first order ambiguity

Let psi be beam orientation and choose psi'=1 radian. Hold the observed baseline, frequencies and all first order harmonic matrices fixed. At psi=0, write their normalized covariances as

\[
S_i=\begin{pmatrix}r_i^2+c_i^2&c_i\\c_i&1\end{pmatrix},
\quad r_i=1+A_i-B_xQ_i,\quad c_i=-B_yQ_i.
\]

Rotating the fixed observed covariance gives

\[
c_i'=1-r_i^2+c_i^2,\qquad r_i'=2r_ic_i.
\]

Since B_parallel'=B_y and B_perp'=-B_x,

\[
A_i'=2r_ic_i-[1+(B_x/B_y)^2]c_i
 -(B_x/B_y)(1-r_i^2+c_i^2),
\]
\[
Q_i'=-(1-r_i^2+c_i^2)/B_y-c_iB_x/B_y^2.
\]

Solving the exact six dimensional Jacobian system gives the unique compensating tangent in (T,h_1,h_2,h_3,g,d). Amplitude and phase compensation follows from keeping C_i fixed. The normalization in I_3 incorporates it automatically.

At the same witness, the exact derivative has

\[
\operatorname{Re}\mathcal I_3'=
\frac{18848880260984398987381756622941300813984492817}
{49946733938194977003395356774800375463944000000}>0,
\]
\[
\operatorname{Im}\mathcal I_3'=
\frac{1444583062125308617841515913871558551982883}
{79280530060626947624437074245714881688800000}>0.
\]

Both values were independently reproduced from the exact paired equations and an independently assembled shape preserving tangent. Hence one real projection of this physical second harmonic breaks the single first order ambiguity locally.

## The actual finite sample full rank theorem

Choose signed speeds N=(1,7,49)/20 Hz, initial phases zero, and nonzero wedge sines e_i=epsilon alpha_i. The integer harmonic indices of total degree at most two give 25 distinct sampled nodes. Six repeated fundamental nodes account for the frequency derivative columns t_k exp(plus or minus 2 pi i N_i t_k). The resulting 31 column confluent Vandermonde matrix has full rank on the original 200 samples.

Distinctness follows from base seven: a difference of two degree at most two indices has l1 norm at most four and cannot satisfy m_1+7m_2+49m_3=0 except trivially. The corresponding sampled integer numerators cannot differ by 400, so no sampling alias occurs.

Use fixed real linear functionals of these actual samples to select the seventeen first order channels and a real projection of the second harmonic. Along the first order fiber,

\[
U_{\rm obs}=e_3e^{i\phi_3}U_+\quad\text{is fixed},
\]
\[
\frac{dC_{\rm obs}}{d\psi}
 =U_{\rm obs}^2\frac{d\mathcal I_3}{d\psi}\ne0.
\]

In scaled wedge coordinates, rescale two baseline columns by one, fifteen first order columns by epsilon^{-1}, and the remaining fiber column by epsilon^{-2}. Their selected eighteen by eighteen matrix tends to a block triangular matrix with nonzero diagonal blocks. Higher order C1 remainders vanish after this rescaling.

Therefore the exact full vector 400 by 18 observation Jacobian has rank eighteen at this admissible family for all sufficiently small positive epsilon. No continuous time derivatives or unobserved Taylor coefficients are used as measurements in this proof.

Analyticity gives generic local identifiability on the connected analytic component containing the witness. A semialgebraic lift gives generic finite fibers within a suitable injective chart/component, after excluding critical and admissible boundary images. Neither argument proves global uniqueness or connectivity of every physical branch.

## Native tangent and conditioning

A nonzero unnormalized derivative is not a robust recovery constant. With psi'=1 radian at the witness, N_i'=0 and the converted native tangent is approximately

- refractive indices: (-101.7158822,-89.84814165,-123.7641878);
- gap g: -559.7542323;
- workpiece distance d: 981.7869465;
- beam angles: (19.82590088,19.09859317) degrees per radian of psi;
- source offsets: (-145.6929506,-39.31428571);
- initial rotor phases: (-5.814083335,-5.595777712,-5.566239507) degrees per radian of psi.

For equal wedge sines e_i=epsilon, wedge angle derivatives in degrees are

\[
(12494.56570,13624.89758,10861.29387)
\frac{\epsilon}{\sqrt{1-\epsilon^2}}.
\]

The limiting sup norm in the original mixed native coordinate convention is therefore about 981.7869465. This norm mixes degrees, index units and native lengths because the original requested coordinate tolerance does so; it is not dimensionless physical conditioning.

The raw normalized invariant derivative has magnitude about 0.3778192691 per radian of psi, but only about 0.0003848281650 per unit of this native sup normalized tangent. At the witness,

\[
U_+\approx69.904973545+0.133333333i,\qquad |U_+|^2\approx4886.7231041.
\]

Thus the corresponding self second harmonic directional gain is approximately 1.880548685 epsilon^2 per native sup unit, asymptotically. This is one directional gain. It is not the smallest singular value, a full inverse Lipschitz constant, or an accuracy guarantee for every coordinate.

At this regular point the native singular direction hierarchy has five order one directions, twelve order epsilon directions, and one order epsilon squared direction, with chart dependent constants. The five strong directions comprise three wedge amplitude directions and two baseline combinations. The weak directions are mixtures of the original parameters. This hierarchy does not assign an independent accuracy order to every native coordinate.

## Data to parameter algorithm

The inputs are only the original paired 200 point record, its timestamps, the original parameter bounds and a declared observation error allowance. No frequencies, material indices, source coordinates or calibration boxes are supplied from truth.

1. Estimate candidate rotor frequencies and their real 2 by 2 first order ellipse matrices from the paired record. A structured harmonic or matrix pencil method is an initializer, not a completeness certificate. Include constant terms and relevant higher harmonic contamination. Retain alternative frequency explanations and all unresolved collisions; use the bounded speed prior to reject impossible aliases.
2. For resolved nonzero ellipses, determine speed signs from their determinant and retain every plausible physical prism assignment. The three ordered prisms are not interchangeable. Estimate the leading baseline from the record while allowing for its quadratic DC contamination.
3. Jointly extract or enclose the assigned self second harmonic and the fundamental coefficients. Form w=V/conjugate(U) and I=C/U^2 only where the denominator is certified nonzero. Frequency uncertainty must be propagated into these quantities, including the effect of using an estimated rather than exact harmonic dictionary.
4. Use either the four polynomial last prism construction or the two dimensional trial beam construction. In the latter, reconstruct all five hardware variables explicitly for every trial beam and impose the remaining scalar first order consistency equation; then use the normalized self second harmonic to select candidates on that curve. In the polynomial route, retain all admissible real last prism roots and apply the remaining first order consistency equation after reconstruction.
5. Recover signed wedge amplitudes, phases and source offsets from M_i^{-1}C_i and the baseline relation. Enforce every original bound, transmitted branch inequality and physical prism order. For noisy coefficients, maintain parameter regions rather than silently accepting exact equalities.
6. Correct every surviving candidate against all 400 exact full vector position residuals. Use the exact paired trace or an equivalent root free graph, exact conditional four variable geometry profiling, and validated Newton or interval certificates when a guarantee is claimed. The local graded Jacobian gives a correction mechanism; it does not prove that an unvalidated optimizer found every component.
7. Report compatible parameter ranges and distinct surviving explanations. A failed correction is unresolved unless a valid exclusion proves incompatibility. Certification of a unique parameter estimate requires both local control and exclusion or enclosure of other candidates.

The frequency and low order stages may have truncation bias, frequency dependent leakage, unresolved order ambiguity and errors from estimating the DC baseline. Their formal polynomial coefficients are not exact measured quantities at finite wedge amplitude. The leading solve is therefore an initializer until an explicit remainder bound and exact residual validation establish the claimed result. A candidate found using true frequencies or a tight box around hidden truth would not implement this algorithm.

## Constructive use and unresolved steps

For finite wedges, harmonics contain higher order contamination; unknown frequency error also affects demixing. A validated remainder enclosure and complete candidate cover are required before claiming an ambiguity complete inverse. Exact vector geometry remains affine in the two source coordinates, g and d, so a four variable LP can profile it. The canonical two dimensional source interval polygon does not transfer unchanged because the two transverse coordinates are coupled.

The present proof does not provide a usable uniform finite noise radius, prove uniqueness of all curve intersections, exclude other disconnected exact branches, or establish a practical complete full prior algorithm. Conditioning deteriorates near zero wedges, frequency collisions, near axial illumination, vanishing perpendicular offset in this chart, and critical optical branches. The exact centered first prism gauge remains a genuine obstruction to a universal unique recovery theorem.

## Reproducibility and audit status

The accompanying proof_checks/vector_last_prism_rank.py reproduces the four dimensional last prism determinant. proof_checks/vector_firstorder_rank.py reproduces the exact six dimensional determinant. proof_checks/vector_secondorder_gauge.py reconstructs the same first order null tangent, the exact nonzero second harmonic derivative, and the native coordinate conversion. These are small symbolic rational checks at the one explicitly stated proof witness, not statistical tests or parameter campaigns.

Independent calculations separately reproduced the last prism four dimensional determinant and coefficient identities, the first order determinant, the exact self harmonic expansion, both real and imaginary gauge derivatives, the orientation bound, and the finite sample block argument. This is independent mathematical and symbolic audit, not a proof assistant certificate. No claim of literature novelty is made here.
