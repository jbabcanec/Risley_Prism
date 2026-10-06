# Exact boundary compactification and all-branch continuation

## Scope and main advances

This addendum concerns the original eighteen unknowns, their entire original prior box, and all 200 passive samples at \(k/20\), \(k=0,\ldots,199\). It uses the exact vector Snell model and requires the stated strict sequential surface traversal. No wedge, speed, beam angle, offset, index, gap, or distance is assumed known. No sample is changed or added.

**Chart scope.** This memo uses the original product-box chart, with the three native refractive indices retained, from [global_inverse_completeness.md](global_inverse_completeness.md). Its optical boxes, fourteen-coordinate distances, and modulus constants refer to that chart. They are not stated in the coupled five-root index chart introduced in [reduced_algebraic_inverse_addendum.md](reduced_algebraic_inverse_addendum.md); transferring these quantitative statements to that chart requires the corresponding coordinate conversion and bounds.

**Source-name alias.** References to global_boundary_continuation.md identify this memo, saved locally as global_boundary_compactification.md.

The new conclusions are:

1. Every direction/position denominator has a data-independent positive lower bound on the weak transmitted branch. Critical refraction remains possible, but axial grazing and incoming-normal grazing do not occur in the closure of this particular bounded model.
2. Replacing only the critical-refraction and traversal strict inequalities by weak ones gives exactly the closure of the physical parameter set, not merely an outer relaxation. A hierarchical inward scaling of the three signed wedges proves this assertion simultaneously at every sample.
3. The forward map therefore has a unique continuous semialgebraic extension to an explicit compact physical closure. All boundary near-aliases become ordinary fibers of this extension; their observation values lie in a semialgebraic set of dimension at most 17.
4. An eighteen-coordinate projection of the original 400 outputs is a proper finite-sheet covering away from a compact boundary-plus-critical discriminant. Complete anchor enumeration plus certified continuation gives an all-branch candidate cover. A projected noise cube avoiding this discriminant gives explicit candidate balls of radius \(\sqrt{18}\epsilon/\eta\), where \(\eta\) is a certified inverse-conditioning margin.
5. A global, explicitly generated \(1/8\)-Hölder modulus replaces derivative bounds at critical refraction. It yields sound LP-dual optical-box exclusion and conditional finite exclusion across critical boundaries.

The compactification is unconditional for this prior. The small candidate-cover and uniqueness conclusions are conditional: an entire noise cube must avoid a computable discriminant, every anchor root must be enumerated, and an inverse-conditioning margin must be certified. None of these costs is proved small. Boundary points remain limiting systems, never physical solutions when a strict inequality is zero.

## 1. Definitions and exact one-prism identities

Use the algebraic chart in [global_inverse_completeness.md](global_inverse_completeness.md). Let \(\Pi\) be its closed eighteen-dimensional prior box. Write \(x\) for the fourteen optical chart coordinates and \(q=(p_x,p_y,g,d)\) for the four geometry coordinates. Let \(P\) be the strict physical domain at all 200 sample times, without imposing observations. Native units are unchanged by the monotone inverse charts.

For one prism and one sample, let the incoming external unit direction be \((X,z)\), with \(X\in\mathbb R^2\) and \(z>0\). Set

\[
\begin{aligned}
H&=\sqrt{n^2-|X|^2},&
u&=\text{instantaneous exit-plane slope},\\
D&=1+|u|^2,&
\nu&=n^2-1,&
P_n&=H-u\cdot X,\\
\Delta&=P_n^2-D\nu,&
R&=\sqrt{\Delta},\\
c&=(P_n-R)/D,&
B&=X+cu,&
Z&=H-c .
\end{aligned}
\tag{1}
\]

The symbol \(P_n\) denotes the incoming normal optical momentum, to distinguish it from the physical parameter set \(P\). At \(\Delta\ge0\) these equations select the nonnegative transmitted normal branch. Direct substitution gives

\[
|B|^2+Z^2=1,\qquad Z-u\cdot B=R,\qquad
(P_n-R)(P_n+R)=D\nu,\qquad c=\frac{\nu}{P_n+R}.
\tag{2}
\]

For incoming transverse position \(p\) and vertex-to-next-flat separation \(\ell\in\{g,g,d\}\), set

\[
A=3+u\cdot p,\qquad
p_{\rm exit}=p+\frac{XA}{P_n},\qquad
B_{\rm flight}=\ell-u\cdot p_{\rm exit},\qquad
p_{\rm next}=p_{\rm exit}+\frac{B}{Z}B_{\rm flight}.
\tag{3}
\]

The strict traversal requirements are \(A>0\) and \(B_{\rm flight}>0\). They are equivalent to the strict internal and external axial traversal conditions in the existing polynomial graph. In particular, the internal axial travel is \(HA/P_n\), and the external axial travel is \(B_{\rm flight}\).

The use of \(B\) for an outgoing transverse vector and \(B_{\rm flight}\) for a scalar flight margin is confined to (1)–(3); later modulus formulas use distinct error symbols.

## 2. Uniform guards, including the critical branch

Put

\[
\begin{aligned}
u_*&=\tan(\pi/10),&
n_-&=13/10,& n_+&=9/5,\\
h_*&=\sqrt{n_-^2-1},&
\nu_*&=n_+^2-1,\\
z_{0*}&=[1+2\tan^2(5\pi/36)]^{-1/2},\\
z_{j*}&=\sqrt{\nu_*+z_{(j-1)*}^2}-\sqrt{\nu_*},
\qquad j=1,2,3 .
\end{aligned}
\tag{4}
\]

All these constants are algebraic; every \(z_{j*}\) is strictly positive. The subtraction in (4) may be evaluated stably using

\[
z_{j*}=
\frac{z_{(j-1)*}^2}
{\sqrt{\nu_*+z_{(j-1)*}^2}+\sqrt{\nu_*}}.
\]

Before imposing \(\Delta\ge0\), \(H\ge h_*\) and \(P_n\ge h_*-u_*>0\), because \(|X|\le1\). Once \(\Delta\ge0\), the stronger estimate

\[
P_n\ge\sqrt{D\nu}\ge h_*
\tag{5}
\]

holds. Also \(0\le R\le P_n\) and

\[
0<c=(P_n-R)/D\le\sqrt{\nu/D}\le\sqrt{\nu}.
\]

Indeed, \(P_n-R\le\sqrt{P_n^2-R^2}\). Therefore

\[
Z=H-c\ge\sqrt{\nu+z^2}-\sqrt{\nu}>0.
\]

The last expression increases with \(z\) and decreases with \(\nu\). Induction proves

\[
H_j\ge h_*,\qquad P_{n,j}\ge h_*,\qquad Z_j\ge z_{j*}>0
\tag{6}
\]

at every sample on the entire weak transmitted branch. These bounds use the original \(n\), wedge, and beam bounds, not a new guarded subdomain.

The same guard was independently obtained in [full_prior_spectral_cover.md](full_prior_spectral_cover.md). \(R\) can equal zero. Thus derivatives through \(\sqrt{\Delta}\) may diverge, and the source-transport determinant can tend to zero; (6) does not supply a positive critical-normal margin or a uniform Jacobian inverse bound.

## 3. The weak set is exactly the physical closure

Define \(W_{\rm opt}\) recursively by the original optical prior and \(\Delta_j(k)\ge0\) for every prism and sample. Equations (1) and (6) make each direction and position expression continuous wherever it is needed. Thus \(W_{\rm opt}\) is compact and semialgebraic. Positions are well-defined on \(W_{\rm opt}\) times the complete geometry box even when traversal fails.

Define \(W\) by additionally imposing \(A_j(k)\ge0\) and \(B_{{\rm flight},j}(k)\ge0\) for all \(j,k\). \(W\) is compact and semialgebraic. Continuity gives \(\overline P\subset W\). The reverse inclusion requires a proof; arbitrary weak inequality replacement would not justify it.

### Inward derivatives at one weak prism

Fix the incoming \(X,z,p,n,\ell\) at a weakly admissible point and replace \(u\) by \(\lambda u\). Write \(a=u\cdot X\) and \(b=u\cdot p\). Then

\[
\begin{aligned}
P_n(\lambda)&=H-\lambda a,\\
\Delta(\lambda)&=(H-\lambda a)^2-(1+\lambda^2|u|^2)\nu,\\
A(\lambda)&=3+\lambda b,\\
\lambda u\cdot p_{\rm exit}(\lambda)
&=\frac{\lambda(Hb+3a)}{H-\lambda a}.
\end{aligned}
\tag{7}
\]

At an active critical constraint \(\Delta(1)=0\), one has \(R=0\),

\[
Z=\frac{H|u|^2+a}{1+|u|^2}>0,\qquad
\Delta'(1)=-2(aP_n+|u|^2\nu)=-2P_nZ<0.
\tag{8}
\]

At an active internal traversal constraint \(A(1)=0\),

\[
A'(1)=b=-3<0.
\tag{9}
\]

At an active external traversal constraint \(B_{\rm flight}(1)=0\), equation (7) gives

\[
B_{\rm flight}'(1)=-H\ell/P_n<0.
\tag{10}
\]

Thus \(\lambda=1-\delta\) with \(\delta>0\) enters the strict side of every active constraint. Every inactive constraint stays positive for sufficiently small \(\delta\). The formulas are valid even at critical refraction: the three constraint functions in (7) depend smoothly on incoming state and \(\lambda\) and do not require differentiating the current outgoing square root.

If \(u=0\), none of these constraints is active: \(\Delta=z^2>0\), \(A=3>0\), and \(B_{\rm flight}=\ell>0\). Consequently a zero wedge creates no exception to this inward argument.

### Three-prism hierarchical strictification

For a fixed \(\theta\in W\), scale the three signed wedge slopes by

\[
r_1(\tau)=(1-\tau^{16})r_1,\qquad
r_2(\tau)=(1-\tau^4)r_2,\qquad
r_3(\tau)=(1-\tau)r_3,
\tag{11}
\]

and keep every other chart coordinate fixed. This scales \(u_j(k)\) by the corresponding scalar at every sample. It preserves signed wedge bounds and all original priors.

For prism 1, its incoming state is fixed. Equations (8)–(10) put every active constraint strictly inside with leading margin of order \(\tau^{16}\). Inactive constraints remain strict. The current output root obeys

\[
|\sqrt s-\sqrt t|\le\sqrt{|s-t|},\qquad s,t\ge0.
\]

Because all other denominators have (6), the first outgoing direction and next-flat position differ from their weak limits by \(O(\tau^8)\).

At prism 2, perturbing the incoming state contributes \(O(\tau^8)\) to each smooth constraint in (7). The direct inward slope scaling supplies a positive order-\(\tau^4\) term at each active constraint by (8)–(10). It dominates that upstream perturbation. All second-prism constraints are therefore strict. The resulting outgoing direction/next-flat position perturbation is \(O(\tau^2)\).

At prism 3, the upstream constraint perturbation is \(O(\tau^2)\), while the direct inward scaling supplies a positive order-\(\tau\) term at each active constraint. The same conclusion follows.

There are only 200 samples. Each strictly negative derivative in (8)–(10) has a positive magnitude at the fixed weak point; taking the minimum over its finitely many active constraints and taking the minimum of the finitely many admissible small-\(\tau\) thresholds proves simultaneous strictness for all samples. The constants and threshold may depend on \(\theta\). No uniform threshold is asserted or needed.

Thus \(\theta(\tau)\) belongs to \(P\) for all sufficiently small positive \(\tau\) and tends to \(\theta\). We have proved

\[
W=\overline P.
\tag{12}
\]

A strict physical point on a prior face can also be perturbed into the interior of \(\Pi\) while retaining strict inequalities. Hence, for \(O=P\cap\operatorname{int}(\Pi)\),

\[
W=\overline O,\qquad W\setminus O=\partial O,\qquad
\dim(W\setminus O)\le17.
\tag{13}
\]

These are closures and boundary in the eighteen-dimensional chart space. This proof is specific to the positive index contrast, positive gap/distance, forward branch, and sequential geometry of this model. It should not be transplanted to a generic radical inverse problem.

The approximation is constructively one-dimensional. For an algebraically encoded weak point, substitute (11) into the existing exact radical-sign compiler. All resulting tests are univariate sign conditions in \(\tau\) with algebraic coefficients. Root isolation determines a positive interval \((0,\tau_0)\) on which every physical inequality is strict. Any residual tolerance strictly larger than the weak point's residual, and any positive requested parameter-neighborhood radius, can be added to those tests; continuity and (11) guarantee a sufficiently small interval satisfying them. Thus a weak-boundary ambiguity witness can be converted into actual physical noisy witnesses using univariate certification. Equality at the weak residual threshold is deliberately not promised.

## 4. Exact boundary near-alias characterization

Let \(\overline F:W\to\mathbb R^{400}\) be the continuous extension of the exact original sampled forward map. It is single-valued and semialgebraic. Compactness and (12) give

\[
\overline{F(P)}=\overline F(W).
\tag{14}
\]

For any fixed record \(y\),

\[
\inf_{\theta\in P}\|F(\theta)-y\|_\infty
=
\min_{\theta\in W}\|\overline F(\theta)-y\|_\infty.
\tag{15}
\]

The minimum on the right is attained, but its minimizing point can be nonphysical. In particular, equality of the consistency threshold with \(\epsilon\) does not decide physical feasibility at that endpoint; a strict physical witness is still required.

There is also an exact small-noise limit. Put \(a\) equal to either side of (15), define
\(M_y=\{\theta\in W:\|\overline F(\theta)-y\|_\infty=a\}\),
and let \(S_{a+h}\) be the strict physical compatible set with error bound \(a+h\). For every \(h>0\), every point of \(M_y\) belongs to \(\overline{S_{a+h}}\): its strictification has residual tending to \(a\). Conversely, compactness shows that every accumulation point from \(S_{a+h}\) as \(h\) decreases to zero belongs to \(M_y\). Hence \(\overline{S_{a+h}}\) converges in Hausdorff distance to the nonempty compact set \(M_y\). After the continuous native inverse chart, each coordinate uncertainty diameter therefore tends exactly to that coordinate diameter of \(M_y\). For an exact record, \(M_y\) is the entire weak exact fiber. A unique strict exact solution can still have a positive uncertainty floor caused by another weak exact solution. None of these assertions implies physical feasibility at \(\epsilon=a\).

A record is a limit of records from physical systems escaping to an excluded physical boundary exactly when it belongs to

\[
A_{\rm boundary}=\overline F(W\setminus P).
\tag{16}
\]

Both directions follow from a convergent subsequence in compact \(W\) and from the strictification sequence (11). The set \(W\setminus P\) is compact, semialgebraic, and of dimension at most 17. Therefore \(A_{\rm boundary}\) is compact semialgebraic of dimension at most 17. This dimension is relative to the eighteen-dimensional model image, not a claim that arbitrary noisy 400-dimensional records lie on it.

For any relatively open candidate region \(U\) in \(W\), if

\[
\overline F^{-1}\bigl(\operatorname{closed\_noise\_cube}(y,\epsilon)\bigr)
\subset U,
\tag{17}
\]

then the compact exterior \(W\setminus U\) has a strictly positive residual gap above \(\epsilon\). This is now an ordinary compact minimum; no unaccounted nonproper escape remains. Checking (17) remains a global task. Testing only exact physical solutions would omit the boundary alternatives in (16).

## 5. A compact projected discriminant

Choose a fixed rational linear map \(L:\mathbb R^{400}\to\mathbb R^{18}\) and put \(H=L\overline F\). A coordinate selector is especially convenient because componentwise noise \(\epsilon\) remains componentwise \(\epsilon\) after projection. A full-rank sample-Jacobian witness guarantees that some selector has a nonzero \(18\times18\) minor there. A particular selector must actually be chosen and certified; the witness does not say every selection works.

The theory below also permits an arbitrary fixed rational \(L\). It makes no assumption that every physical connected component contains the known rank witness.

On \(O\), \(H\) is real analytic and semialgebraic. Define

\[
K_H=\{\theta\in O:\det DH(\theta)=0\},\qquad
D_H=H(W\setminus O)\ \cup\ \overline{H(K_H)}.
\tag{18}
\]

The closure in (18) is in \(\mathbb R^{18}\). Since \(W\) is compact and \(H\) extends continuously,

\[
\overline{H(K_H)}=H(\overline{K_H}^{\,W}).
\]

Thus \(D_H\) is compact semialgebraic. Its dimension is at most 17. The boundary contribution has that dimension bound by (13). For the critical contribution, stratify the semialgebraic critical set into finitely many smooth pieces on which \(H\) has constant rank. Each restricted rank is at most 17; the constant-rank theorem bounds its image dimension by 17. Finite unions and closure preserve that bound.

This argument includes a component on which the chosen \(H\) is everywhere rank deficient: its entire projected image belongs to \(D_H\). Such a component has not been assumed absent. Positive-dimensional projected fibers and prior-boundary roots also occur only over \(D_H\). The lower-dimensional exceptional set must still be retained and solved, not discarded as unimportant.

## 6. Finite coverings and complete continuation

Let \(V\) be a connected component of \(\mathbb R^{18}\setminus D_H\). Then

\[
H:H^{-1}(V)\longrightarrow V
\tag{19}
\]

is either empty over all of \(V\) or a finite-sheet covering. Its domain lies in \(O\), every derivative \(DH\) there is invertible, and every fiber is finite. The number \(m(V)\) of roots is constant over \(V\) and includes roots in every physical connected component.

Proof. A compact set \(T\subset V\) has compact preimage in \(W\), and that preimage cannot meet \(W\setminus O\) or \(K_H\). Hence (19) is proper and a local diffeomorphism. A fiber is compact and locally discrete, so finite. Around its finitely many points choose disjoint inverse-function neighborhoods. If arbitrarily nearby output values had another preimage outside those neighborhoods, compactness would yield an additional preimage at the central value, a contradiction. Thus a smaller output neighborhood is evenly covered. The image is both open and closed in \(V\), proving the empty/all alternative and constant sheet count.

Consequences:

- Enumerating all roots at one anchor \(z_0\in V\), followed by path lifting, enumerates every root at every other point reached inside \(V\).
- A simply connected subregion of \(V\) admits one single-valued inverse branch per anchor root. A non-simply-connected chamber can permute roots by monodromy; lifting all roots still preserves completeness.
- Continuation from one successful fit does not enumerate all sheets. It is complete only after complete anchor enumeration.
- All unprojected observations remain constraints. For exact data \(y\), retain from \(H^{-1}(Ly)\) precisely the systems satisfying all 400 exact equations. For componentwise noise \(\epsilon\), lift the entire projected zonotope \(Ly+L[-\epsilon,\epsilon]^{400}\), or a certified enclosing cube, on every sheet and then filter all 400 original noise inequalities. Filtering the anchor fiber alone is not a valid noisy inverse. The projection does not change the requested inverse.
- The number of full-record exact aliases can change when the unprojected residual filters change. Only the projected root count is asserted constant on \(V\).

For a coordinate-selector L, one exact anchor-enumeration route is the existing fixed-eighteen-variable algebraic backend: impose the eighteen selected scalar observation equations, together with all original per-sample physical predicates. These constraints preserve the sample-local compilation, so its stated encoding and polynomial-in-K complexity qualifications apply. For a general dense L, the projected equations couple ray outputs from different samples. The topological covering theorem still holds, and exact finite quantifier elimination on the full coupled ray graph still enumerates the anchor fiber, but the earlier constant-size per-sample compilation and polynomial-in-K complexity bound do not automatically transfer. Establishing that sharper complexity for dense L would require a separate elimination argument. Both anchor routes can be very expensive. The present theorem allows a complete enumeration to be reused across a certified observation chamber; it does not make the first enumeration free.

## 7. An all-branch noise candidate-cover theorem

For clarity take \(L\) to select eighteen original scalar observations. Fix a full record \(y\), noise \(\epsilon\ge0\), \(z_0=Ly\), and

\[
T=z_0+[-\epsilon,\epsilon]^{18}.
\]

Assume the following are certified:

(a) \(T\) is disjoint from \(D_H\).

(b) Every root \(\theta_1,\ldots,\theta_m\) of \(H(\theta)=z_0\) in \(W\) has been enumerated.

(c) A number \(\eta>0\) satisfies \(\sigma_{\min}(DH(\theta))\ge\eta\) on \(H^{-1}(T)\), using Euclidean chart and projected-observation norms.

Because \(T\) is connected and disjoint from \(D_H\), it lies in one chamber. Its compactness gives a slightly expanded convex cube still in that chamber. The covering is trivial on this expanded cube; each anchor root has one inverse branch over \(T\). Assumption (c) always has some positive solution when \(m>0\), by compactness, but its value must be certified. If \(m=0\), the entire cube has no preimage and the original noisy inverse is empty.

Along the straight segment from \(z_0\) to \(z\in T\), an inverse branch satisfies

\[
D\theta(z)=[DH(\theta(z))]^{-1}.
\]

Integrating and using \(\|z-z_0\|_2\le\sqrt{18}\epsilon\) proves that every full-200-compatible physical system lies in

\[
\bigcup_{i=1}^m
\overline B\left(\theta_i,\frac{\sqrt{18}\epsilon}{\eta}\right).
\tag{20}
\]

A tighter branch-dependent \(\eta_i\) can be used on each sheet. Conditions (a)–(c), including the anchor enumeration, are real-algebraic/semialgebraic certificate conditions because the physical graph and its derivatives are semialgebraic on the strict branch. Eigenvalue lower bounds can equivalently be expressed by positive-semidefinite matrix inequalities with exact principal-minor tests. A simpler sufficient certificate is \(|\det DH|\ge d_*>0\) and \(\|DH\|_2\le M_*\) on the preimage, which gives \(\eta=d_*/M_*^{17}\). As in the existing complete inverse, effective exact certification assumes rational/algebraic data encodings; arbitrary real-number oracles do not provide equality decisions.

Equation (20) is an all-branch cover of the original inverse. It makes no small-angle approximation, assumes no parameter known, and changes no samples. For a general \(L\), replace the projected cube and \(\sqrt{18}\epsilon\) by any certified enclosing convex observation set and its Euclidean radius; a conservative projected cube has half-width \(\|L\|_{\infty\to\infty}\epsilon\) and Euclidean radius \(\sqrt{18}\|L\|_{\infty\to\infty}\epsilon\). The balls in (20) are in algebraic chart coordinates. For a native-coordinate guarantee, project each chart ball to intervals, intersect with the chart prior, and apply the exact monotone inverse chart to its endpoints; a chart radius must not be reported unchanged as an angular or frequency error.

The theorem can fail to apply because the noise cube meets \(D_H\), because anchor enumeration is expensive, because \(\eta\) is too small for useful localization, or because \(m\) is large. Those failures are not inconsistencies. The corresponding exceptional/noise regions remain in the exact backend.

## 8. Degree does not by itself exclude remote sheets

For a chamber \(V\) with nonempty preimage, the signed count

\[
d(V)=
\sum_{\theta\in H^{-1}(z)}\operatorname{sign}(\det DH(\theta)),
\qquad z\in V,
\tag{21}
\]

is independent of \(z\). Thus \(m(V)\ge|d(V)|\) and \(m(V)\equiv d(V)\pmod2\). A degree of one does not prove one root: additional positively and negatively oriented sheets can cancel.

A valid degree-based uniqueness certificate needs, for example, a certificate that \(\det DH\) has one common sign throughout \(H^{-1}(V)\), together with \(|d(V)|=1\). Alternatively, a complete anchor enumeration with \(m=1\) already proves one projected root on the chamber and hence at most one full-record root there. The prior local-rank witness alone supplies neither certificate.

## 9. Explicit global Hölder modulus

This section supplies checkable constants for optical-box exclusion, including boxes touching critical refraction. Distances below are infinity norm in the fourteen optical algebraic chart coordinates, while transverse/direction vector errors use Euclidean norm. Geometry \(q\) is held fixed in its full prior box; traversal need not hold during this relaxed comparison.

Let \(k_*=199\) and

\[
L_u=1+2u_*(1+k_*),\qquad
D_*=1+u_*^2,\qquad
P_*=n_++u_*,\qquad
c_*=\sqrt{\nu_*}.
\]

The exact rotor formula gives

\[
|u_j(k;x)-u_j(k;x')|
\le L_u\delta,\qquad
\delta=\|x-x'\|_\infty.
\tag{22}
\]

Indeed, its derivatives with respect to the three chart coordinates \((r,f,v)\) have norms bounded by \(1\), \(2u_*\), and \(2ku_*\). The initial normalized direction changes by at most \(\sqrt2\delta\).

The following recursive, nonnegative error expressions bound a pair of weak-optical traces. Start with \(d_0=\sqrt2\delta\). At prism \(j\) define

\[
\begin{aligned}
h_j&=\frac{n_+\delta+d_{j-1}}{h_*},\\
p_j&=h_j+u_*d_{j-1}+L_u\delta,\\
a_j&=2P_*p_j+2u_*L_u\nu_*\delta+2D_*n_+\delta,\\
r_j&=\sqrt{a_j},\\
c_j&=p_j+r_j+2c_*u_*L_u\delta,\\
d_j&=d_{j-1}+h_j+(1+u_*)c_j+c_*L_u\delta.
\end{aligned}
\tag{23}
\]

Here lower-case \(h_j,p_j,a_j,r_j,c_j,d_j\) in (23) are error bounds, not model parameters. They respectively bound the changes in \(H\), \(P_n\), \(\Delta\), \(\sqrt{\Delta}\), \(c\), and the full outgoing direction. The bounds use

\[
|\sqrt s-\sqrt t|\le\sqrt{|s-t|},
\]

\[
|H-H'|\le\frac{n_+\delta+|X-X'|}{h_*},
\]

\[
|\Delta-\Delta'|
\le 2P_*|P_n-P_n'|+\nu_*|D-D'|+D_*|\nu-\nu'|.
\]

All terms in (23) have explicit algebraic coefficients. No segment between \(x\) and \(x'\) is required to remain physically admissible.

For position magnitudes choose

\[
\begin{aligned}
R_0&=\sqrt2\left(5+6\tan(5\pi/36)\right),\\
L_1&=L_2=15,\qquad L_3=200,\\
A_{j*}&=3+u_*R_{j-1},\\
E_{j*}&=R_{j-1}+A_{j*}/h_*,\\
T_{j*}&=L_j+u_*E_{j*},\\
R_j&=E_{j*}+T_{j*}/z_{j*}.
\end{aligned}
\tag{24}
\]

These bound entrance, exit, and next-flat positions even for the relaxed geometry box. For the comparison errors start \(e_0=6\sqrt2\delta\), since \(b\) is fixed. Set

\[
\begin{aligned}
f_j={}&(1+u_*/h_*)e_{j-1}
      +(A_{j*}/h_*)d_{j-1}\\
     &+(R_{j-1}/h_*)L_u\delta
      +(A_{j*}/h_*^2)p_j,\\
e_j={}&(1+u_*/z_{j*})f_j
      +(E_{j*}L_u/z_{j*})\delta\\
     &+T_{j*}\left(1/z_{j*}+1/z_{j*}^2\right)d_j.
\end{aligned}
\tag{25}
\]

Then \(f_j\) bounds the exit-position difference and \(e_j\) bounds the next-flat-position difference. The omitted \(\ell\)-difference term is zero because geometry is fixed. Define \(\Omega(\delta)=e_3\). It is continuous, nondecreasing, explicit, and \(\Omega(0)=0\). Uniformly over all samples and \(q\) in its original box,

\[
\|\overline F(x,q)-\overline F(x',q)\|_\infty
\le\Omega(\|x-x'\|_\infty).
\tag{26}
\]

Only \(x,x'\in W_{\rm opt}\) are needed; the full geometry rectangle may include nonphysical traversal points, on which the continuous affine extension is being used solely as a relaxation.

For \(0\le\delta\le1\), induction in (23)–(25) gives an explicit \(C\) with

\[
\Omega(\delta)\le C\delta^{1/8}.
\tag{27}
\]

One valid explicit choice is \(C=\Omega(1)\). To verify it, inductively bound each nonnegative expression at stage \(j\) by its value at \(\delta=1\) times \(\delta^{1/2^j}\); the square-root operation halves the previous exponent, and for \(0\le\delta\le1\) all larger powers are bounded by the smaller power. The exponent \(1/8\) is a safe three-square-root cascade bound, not a claim of a sharp exponent at every boundary. The constants may be very large, particularly through \(z_{3*}\). Guarded subregions can use much stronger Lipschitz estimates.

For traversal outer relaxations, the same formulas give uniform moduli

\[
\begin{aligned}
\Phi_{{\rm internal},j}(\delta)
  &=u_*e_{j-1}+R_{j-1}L_u\delta,\\
\Phi_{{\rm external},j}(\delta)
  &=u_*f_j+E_{j*}L_u\delta .
\end{aligned}
\tag{28}
\]

They bound changes in \(A\) and \(B_{\rm flight}\) for fixed \(q\). Their maximum over \(j\) is a valid common \(\Phi(\delta)\), tending to zero.

## 10. Critical-boundary LP exclusion and finite stopping

Fix an optical anchor \(x_0\in W_{\rm opt}\). Let \(B\) be any optical set with \(\|x-x_0\|_\infty\le r\) for its points. The anchor need not be its box center. No convexity or positive critical-radicand margin is required. Any potentially physical \(x\in B\) belongs to \(W_{\rm opt}\).

Let \(Q_B\) be a compact common outer geometry polytope and let \(\lambda,\mu\) be an exactly verified LP-dual pair at \(x_0\) as in [profiled_box_exclusion.md](profiled_box_exclusion.md), with \(\|\lambda\|_1\le1\). If \(\alpha\) is the certified affine-geometry lower bound there, then

\[
\|F(x,q)-y\|_\infty\ge\alpha-\Omega(r)
\tag{29}
\]

for every physical \((x,q)\) with \(x\in B\). Hence \(\alpha-\Omega(r)>\epsilon\) excludes the box. The dual residual correction from that memo remains valid. Formula (29) replaces its derivative term \(Lr\) and remains valid at critical refraction.

A useful common polytope is

\[
Q_B=Q\cap
\{q:g_s(x_0,q)\ge-\Phi(r)\text{ for every traversal row }s\},
\tag{30}
\]

where each \(g_s\) is \(A_j(k)\) or \(B_{{\rm flight},j}(k)\); these are affine in \(q\). It contains all weakly or strictly physical geometries for all \(x\in B\). Every coefficient is evaluated on the weak transmitted branch with its unique nonnegative root.

There is now a finite-exclusion theorem on the full compact weak optical domain. Suppose \(U\) is an open retained optical region and every weak physical system with optics outside \(U\) has residual strictly greater than \(\epsilon\). The corresponding exterior in \(W\) is compact, so its residual has gap \(\gamma>0\). At a fixed optical anchor outside \(U\), the polytopes (30) decrease to the exact weak-traversal polytope as \(r\) decreases to zero. Compactness of \(Q\) proves that their optimum residuals converge to the exact weak-traversal optimum; if the latter polytope is empty, the relaxations are empty for sufficiently small \(r\). Combine this with \(\Omega(r)\to0\) and LP strong duality to obtain an excluding neighborhood at every exterior anchor. Compactness gives a finite subcover.

This argument handles critical normal boundaries and strict traversal faces without imposing derivative guards. It is qualitative: it does not bound how many boxes or how small their radii must be, and it does not prove the required retained cover \(U\). Optical points outside \(W_{\rm opt}\) can be excluded by the exact branch predicate; a complete implementation still needs coverage and certificates for that complement rather than silently omitting it.

Because \(W=\overline P\), a weak boundary system surviving (30) is a genuine limit of physical systems. It is not a spurious weak-only component. It can still prevent exclusion at a noise threshold even when no strict physical system attains that threshold; this is a real limiting ambiguity that must be reported accurately.

## 11. Representation symmetries and quotient cautions

The original parameter chart already removes the standard rotor representation aliases away from zero wedge. For one nonzero rotor, \(u(0)\) fixes \(|r|\) and its angle. The allowed phase sector \([-18,18]\) degrees and its \(\pi\)-shifted sector for negative \(r\) are disjoint, so the signed \(r\) and phase are unique. The ratio \(u(1)/u(0)\) is \(\exp(2\pi iN/20)\); \(N\in[-3.5,3.5]\) spans less than the 20-Hz alias period, so this ratio fixes signed speed uniquely. Thus two original rotor parameter triples producing the same two instantaneous slope vectors must coincide when the wedge is nonzero.

At zero wedge, phase and speed truly disappear from that rotor's slope sequence, and their continuous fibers must remain in the inverse. The index still affects oblique internal transport; a zero wedge is not permission to delete the slab.

A signed-speed reversal generally conjugates or reverses a trajectory, rather than preserving the fixed labeled \(x,y\) record. Global rotations/reflections transform the observed coordinates. Physical prism order is not a permutation symmetry. Consequently a quotient by unordered frequencies, speed signs, or prism permutations would discard possible physically distinct branches unless an additional exact invariance is proved for that record. The above rotor representation argument does not rule out distinct optical systems producing the same screen record; those are the remote aliases addressed by the covering and discriminant construction.

## 12. Exact progress and remaining burden

The new structural closure is substantial: on this original prior, critical refraction is a continuous finite boundary, every weak boundary point is a genuine physical limit, and explicit Hölder exclusion is available there. The old concern about denominator blow-up toward arbitrary axial/incoming-normal grazing is absent for this model. Only derivative conditioning, critical-normal collapse, traversal endpoints, and genuine multiple fibers remain.

The projected covering and noise-cube theorem is a rigorous all-branch reduction. Its checkable inputs are the exact boundary/critical discriminant, complete anchor roots, a noise cube avoiding that discriminant, and a positive singular-value certificate. The remaining hard computation has been localized to those objects; it has not been proved inexpensive, nor has a particular full18 record been supplied and certified here.

There is no unconditional unique inverse over the original box. Zero-wedge and centered gauges already forbid it. There is no justified claim that a rank witness controls every connected component, that degree one removes oppositely oriented sheets, or that continuing one fit proves completeness.

### Mathematical sources and provenance

The model-specific guard, strictification, compactification, modulus, and projected-noise cover are derived above. The algebraic chart and full-record graph are from [global_inverse_completeness.md](global_inverse_completeness.md) and [explicit_six_root_compiler.md](explicit_six_root_compiler.md). The affine LP dual is from [profiled_box_exclusion.md](profiled_box_exclusion.md).

The elementary proper-local-homeomorphism covering argument is proved in Section 6; a classical primary reference is Chung-Wu Ho, “A note on proper maps,” *Proceedings of the American Mathematical Society* 51 (1975), 237–241, DOI [10.1090/S0002-9939-1975-0370471-3](https://doi.org/10.1090/S0002-9939-1975-0370471-3). A short mathematical exposition with the same proof is [Proper local homeomorphisms and covering maps](https://www.matem.unam.mx/~omar/notes/propetale.html).

The semialgebraic closure, dimension, and stratification facts used above are standard finite-cell results; see Michel Coste, “An Introduction to Semialgebraic Geometry,” [author-listed notes](https://perso.univ-rennes1.fr/michel.coste/articles.html), and the constructive real-algebraic background already cited in global_inverse_completeness.md, [Saugata Basu's survey](https://www.math.purdue.edu/~sbasu/raag_survey2011.pdf). None of these sources is being cited as proving the optical-specific strictification theorem.

### Independent proof checks

An independent audit checked the three inward derivatives, the \(\tau^{16}/\tau^4/\tau\) hierarchy, the closure and projected covering hypotheses, the full-noise-cube rather than anchor-only lift, every term of the global modulus recurrence, the critical-boundary LP convergence argument, and the Hausdorff minimizer limit. The two inward derivative identities were also simplified symbolically to zero. These checks support the stated theorems; they do not instantiate the discriminant, enumerate roots for a supplied record, or establish practical conditioning.
