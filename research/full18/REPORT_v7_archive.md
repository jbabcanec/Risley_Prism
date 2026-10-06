# Physical-vector Risley inversion: exact global construction and symbolic-error guarantees

Research draft for expert review, 2026-10-03, revision 7.

This investigation gives a constructive, ambiguity-complete inverse in principle for the exact coupled three-dimensional Snell model with all eighteen native parameters unknown. The original six-root compiler remains a valid baseline; an equivalent coupled index chart reduces the forward compiler to five roots and degree 896. Exact source transport and mixed-strict circuit elimination reduce affine fibers without discarding degenerate cases. A model-specific compactification and conditional projected-sheet theorem now give all-branch continuation and noise candidate covers. A collision-safe three-variable spectral cover, sharp critical-tail limitations and data-only final-prism margin tests further separate complete outer bounds from useful localization. None of these results establishes a practical 200-sample runtime or useful constants for an actual record.

Measurement error is the symbolic parameter \(\epsilon\). For a fixed observed record, the inverse produces the full family \(\mathcal C_\epsilon(y)\), its exact consistency and unacceptable-error thresholds, and each threshold's endpoint status. One unrestricted point estimate guarantees .001 in every native coordinate exactly when every projected diameter is at most .002. The report gives sharp LP coefficients and matching conditional quadratic bounds on regular branches, and a complete one-sided classification by Puiseux powers, uncertainty floors and jumps at singular endpoints. The coordinate midpoint need not be a compatible physical system. Global uniqueness and a uniformly useful precision guarantee are false over the full original prior.

Three obligations organize the results:

1. **Exact finite-angle construction.** Sections 2A–2B give the six-root baseline, improved five-root compiler, exact source elimination and mixed-strict affine circuits. No small-angle truncation is used by the complete inverse.
2. **Every compatible system and ambiguity.** Complete-cell output retains all fibers. Section 2C proves exact physical compactification and a conditional all-branch cover requiring complete anchor enumeration, a discriminant-free projected noise cube and certified inverse margins. It is not an evaluated practical solver.
3. **Error-dependent native precision.** Section 7 gives the exact successful-error interval, sharp regular-branch sensitivity and explicit matching bounds for all eighteen coordinates, and the singular endpoint alternatives. Measurement error \(\epsilon\) is separate from small-wedge scale \(\kappa\). No arbitrary numerical noise-test levels organize the theory.

The local physical theory remains useful structure: a constructive leading inverse has rank seventeen at an admissible oblique-beam witness; one physical second harmonic resolves its remaining local ambiguity; a four-polynomial construction makes formal candidates generically finite; and an analytic finite-record argument proves rank eighteen at sufficiently small nonzero wedges near that witness. These conclusions do not turn finite-error compatible sets into finite lists or supply useful numerical constants by themselves.

The canonical independent-axis model is a different model. Its results and historical evidence remain separately scoped in section 9. The available original-protocol 200-sample record has verified canonical simulation provenance, not full-vector or hardware provenance. Eleven supporting algebraic and inequality lemmas have now passed bounded Lean verification; the complete optical inverse and new spectral/continuation theorems are not formalized. Independent symbolic audits and these bounded formal checks do not constitute external peer review. No physical experiment, practical all-branches computation or useful finite-error certificate for an actual record has yet been established. This revision integrates theory, preserves supplied proof sources and compiles only the isolated Lean support package; it runs no original project code, recovery experiments or new rank campaign.

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

For a fixed observed record \(y\), let \(\epsilon\ge0\) be a symbolic uniform bound on each of the 400 real measurement errors. More generally, fix known nonnegative weights \(\omega_m\) and use component allowances \(\epsilon_m=\epsilon\omega_m\). The inference object is
\[
\mathcal C_\epsilon(y)=
\{\theta\text{ in the native physical domain}:
 |F_m(\theta)-y_m|\le\epsilon\omega_m,\quad m=1,\ldots,400\}.
\tag{1}
\]
The record stays fixed as \(\epsilon\) varies; these sets are nested. Exact data means \(\epsilon=0\). Positive-error compatibility is generally continuous. If the true system obeys the declared model and error bound, it belongs to this set; an empty set signals inconsistency of that model, record and allowance.

The original requested accuracy of .001 in every native coordinate mixes hertz, index, degree and native-length units; it is not a dimensionless condition number. Throughout the physical theory, \(\kappa\) denotes a geometric small-wedge scale independent of \(\epsilon\), and \(\tau\) denotes a rigorously bounded approximation or remainder error. Historical canonical appendices and verbatim source scripts retain their locally defined notation.

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

## 2A. Constructive exact inverse and complete ambiguity accounting

The complete derivation is in [global_inverse_completeness.md](theory/global_inverse_completeness.md), with an explicit optical compiler in [explicit_six_root_compiler.md](theory/explicit_six_root_compiler.md). This section states the construction and its quantitative scope.

### A bijective chart and exact sampled rotor motion

With native angles in degrees, introduce
\[
r_j=\tan(\pi a_j/180),\quad
f_j=\tan(\pi\phi_j/360),\quad
v_j=\tan(\pi N_j/20),\quad
s_h=\tan(\pi\beta_h/180).
\]
Retain \(n_{1:3},d,g,b_x,b_y\). The resulting eighteen-coordinate box has algebraic endpoints:
\[
|r_j|\le\tan(\pi/10),\quad |f_j|\le\tan(\pi/20),\quad
|v_j|\le\tan(7\pi/40),\quad |s_h|\le\tan(5\pi/36),
\]
with the original rational index and length bounds. Each map is strictly monotone on its full prior interval. No signed or zero wedge, speed, phase or endpoint is dropped.

Writing a two-vector as a complex number, the instantaneous exit-plane slope is exactly
\[
U_j(k)=r_j\frac{(1+i f_j)^2(1+i v_j)^{2k}}
 {(1+f_j^2)(1+v_j^2)^k}.
\tag{G1}
\]
Its denominator is strictly positive; numerator degree is at most \(2k+3\), denominator degree \(2k+2\), and integer coefficient bit length \(O(k)\). This is the exact ideal-clock motion at every measured time, with no Fourier or Taylor truncation. Inverse native angles are the corresponding scaled arctangents. Native angular values are not claimed algebraic or semialgebraic functions of chart coordinates.

### Six positive roots compile one physical sample

Use beam slope \(t=(s_x,s_y)\), \(Q=1+|t|^2\), and initial scaled direction \(X_0=t,\ Z_0=1,\ L_0=1\). The scaled external vector \((X_j,Z_j)/L_j\) has squared norm \(Q\); its physical unit direction has the additional divisor \(\sqrt Q\). At prism \(j\), with instantaneous slope \(u_j\), set
\[
\begin{aligned}
D_j&=1+|u_j|^2,\\
H_j&=\sqrt{n_j^2 QL_{j-1}^2-|X_{j-1}|^2},\\
P_j&=H_j-u_j\cdot X_{j-1},\\
E_j&=\sqrt{P_j^2-D_j(n_j^2-1)QL_{j-1}^2},\\
Z_j&=H_jD_j-P_j+E_j,\\
X_j&=X_{j-1}D_j+u_j(P_j-E_j),\qquad
L_j=L_{j-1}D_j .
\end{aligned}
\tag{G2}
\]
Both roots are positive, both radicands strictly positive, and \(P_j,Z_j>0\). These conditions select transmission on the declared forward branch; the scaled direction's unnormalized normal residual is \(E_j/L_{j-1}>0\). Its physical unit-normal component is \(E_j/(L_{j-1}\sqrt{QD_j})\). There are exactly six ordered radicals \(H_1,E_1,H_2,E_2,H_3,E_3\).

Starting with \(p_0=b+6t\), exact positions follow
\[
p_{\rm exit}=p_{j-1}
 +\frac{X_{j-1}(3+u_j\cdot p_{j-1})}{P_j},
\qquad
p_j=p_{\rm exit}
 +\frac{X_j}{Z_j}(\ell_j-u_j\cdot p_{\rm exit}),
\tag{G3}
\]
where \((\ell_1,\ell_2,\ell_3)=(g,g,d)\). Preserve strict traversal
\(3+u_j\cdot p_{j-1}>0\) and
\(\ell_j-u_j\cdot p_{\rm exit}>0\).
The constructed position denominators are products of positive \(P_j,Z_j\). Rationalizing by algebraic conjugates is unnecessary and can introduce zero denominators.

Every sign test is eliminated exactly, including equality and cancellation. For one positive root \(r=\sqrt R\), reduce an expression to \(A+Br\), and put \(C=A^2-B^2R\). If \(B=0\), use \(\operatorname{sign}A\); if \(A=0\), use \(\operatorname{sign}B\). If the two nonzero signs agree, use their common sign. If they disagree, use \(\operatorname{sign}A\operatorname{sign}C\). Repeating this for all six roots preserves every weak and strict branch condition.

The appendix proves weighted final position degrees at most 37 in the numerator and 33 in the positive denominator. Six radical reductions give ordinary terminal polynomial degree at most \(37\cdot64=2368\) for interval observations, or \(74\cdot64=4736\) for a separate Euclidean ball at each sample. A shared sign circuit uses at most \(3^6=729\) terminal sign-polynomial occurrences per atom; 24 interval-model atoms give at most 17,496 before deduplication, excluding prior bounds. This is a shared-circuit count, not an expanded Boolean formula size.

An independent route uses the single-sample polynomial graph: incident unit direction, four transverse positions and three five-variable prism blocks give 26 auxiliaries, 26 scalar equations and 19 strict inequalities of degree at most three. Positive root, normal, axial and traversal conditions prove that its lift is unique. An alternate scaled-direction graph uses 25 auxiliaries and remains cubic. Either fixed graph can be compiled once by real quantifier elimination. Neither route requires an empirical rank witness.

### Intersect in shared coordinates, then return every cell

Substitute (G1) into each compiled sample predicate, clearing only strictly positive denominators. The same eighteen chart coordinates describe the hardware at every sample; ray-state auxiliaries are sample-specific. Intersect all sample predicates with the original box:
\[
\mathcal S_\epsilon(y)=
 \{x\text{ in the chart box}:\bigwedge_{k=0}^{K-1}Q_k(x;y,\epsilon)\}.
\tag{G4}
\]
The graph equivalence proves that this set is precisely the chart image of \(\mathcal C_\epsilon(y)\), in both directions.

Complete real-algebraic decomposition returns the union of all satisfying cells with their defining and root-order data. A point per component is insufficient. The retained union includes isolated points, continuous or disconnected fibers, collisions, zero speeds, zero-wedge phase and speed ambiguities, centered gauges and all singular strata. A cell need not be a whole connected component. No rank, genericity, minimum separation or positive minimum optical margin is assumed.

For finite exact input encodings the algorithm terminates even with open physical guards. Coordinate infima and suprema can be isolated without asserting attainment. This contrasts with unqualified interval subdivision near excluded critical-normal or traversal boundaries, where termination does not follow automatically; section 2C supplies a compactification and a conditional modulus-based finite-exclusion theorem. Conditions here apply at the measured times; physical admissibility at every intermediate real time is a separate problem.

The exact input model matters. Rational data and rational interval bounds give effective finite inputs. Algebraic data require controlled encodings: either a specified number field and real embedding, charging its degree and height, or separate polynomial/isolating-interval descriptions with constant-size local elimination per sample. Combining unrelated algebraic observations into one compositum while treating its degree as fixed is invalid. Arbitrary-real approximation oracles do not support unconditional exact equality decisions.

### Quantitative size and honest computational scope

The explicit compiler bounds substitution degree by \(D(6k+7)\); at \(K=200\), \(D=2368\), the conservative maximum is **2,843,968**. With rational observations of bit size \(\tau_{\rm in}\), the full formula has \(O(KS_*)\) polynomial occurrences, degree \(O(D_*K)\) and coefficient bit size \(O(C_*(K+\tau_{\rm in}))\), for fixed compiler constants. The general rational-clock variant depends on the largest integer lattice index, not only its binary length.

For fixed allowance, decomposition has eighteen physical variables. Keeping scalar \(\epsilon\) free produces a **nineteen-dimensional** joint set. Both dimensions are fixed, so sample-count complexity is polynomial in principle, with enormous exponents and constants. Classical CAD's arithmetic bound is \((sd)^{2^{O(n)}}\); bit complexity additionally charges coefficient height and exact algebraic arithmetic. Basu's Definition 1.1 distinguishes the arithmetic convention, Theorem 2.4 gives that CAD bound, and Theorem 2.16 supplies effective quantifier-elimination bounds and integer-height control. The polynomial bit conclusion uses the fixed-dimensional rational algorithms and height controls; it is not a bit-operation statement from Theorem 2.4 alone. [Basu's primary survey](https://www.math.purdue.edu/~sbasu/raag_survey2011.pdf)

This establishes a finite complete inverse and an effective asymptotic bound, **not a practical 200-sample solver**. Per-sample error bands and fixed-size per-sample balls fit the argument. A correlated error constraint coupling the entire record needs its own complexity argument.

### Four geometry variables and strict feasibility

At fixed fourteen optical coordinates, all ray directions are fixed and positions and traversal constraints are affine in \((b_x,b_y,g,d)\). Thus interval observations and the geometry box form \(M=O(K)\) mixed weak and strict halfspaces. Finite Helly theory makes geometry feasibility equivalent to feasibility of every subsystem of at most five rows: \(O(K^5)\) constant-size tests. Open halfspaces are retained. A common slack \(e>0\) on strict rows encodes their strict feasibility; replacing it by \(e\ge0\) is unsound.

When the strict fiber \(F_x\) is nonempty, its closure is exactly the bounded polytope obtained by weakening its strict rows. This follows by convex combination with any point of \(F_x\), including when the fiber has lower dimension. Linear extrema therefore occur on that polytope's vertices. Enumerating four independent active rows gives \(O(K^4)\) candidate vertices; Cramer's rule requires determinant sign control and every candidate must satisfy all weak rows. Crucially, the original strict fiber must remain feasible. Otherwise a weakened boundary can invent a solution, as \(q>0,\ q\le0\) demonstrates. Closure vertices may describe unattained extrema, not physical systems.

### Exact native recoverability at a given error allowance

For a nonempty compatible set, define native-coordinate endpoints
\[
\ell_i(\epsilon)=\inf_{\theta\in\mathcal C_\epsilon(y)}\theta_i,\qquad
u_i(\epsilon)=\sup_{\theta\in\mathcal C_\epsilon(y)}\theta_i,\qquad
D_i(\epsilon)=u_i(\epsilon)-\ell_i(\epsilon).
\tag{G5}
\]
The sharp conditional minimax radius for an unrestricted native point estimate is
\[
R_y(\epsilon)=
 \inf_c\sup_{\theta\in\mathcal C_\epsilon(y)}\|\theta-c\|_{\infty,\rm native}
 =\frac12\max_{1\le i\le18}D_i(\epsilon).
\tag{G6}
\]
The coordinate midpoint attains this supremum bound whether or not extrema are attained. For coordinate targets \(\delta_i\), the guarantee is possible exactly when \(D_i(\epsilon)\le2\delta_i\) for every coordinate. Thus **.001 in all eighteen native coordinates is equivalent to all eighteen diameters being at most .002**. If a diameter exceeds its threshold, two actual compatible systems separated by more than that threshold can be extracted. A nonzero-width ambiguity at \(\epsilon=0\) imposes an error floor.

Native angular endpoints are scaled arctangents of algebraic chart endpoints. A native midpoint with chart endpoints \(a,b\) has exact chart representative
\[
m=\frac{a+b}{\sqrt{(1+a^2)(1+b^2)}+1-ab}.
\tag{G7}
\]
For \(q=c\arctan x/\pi\), where \(c=180,360,20\) as appropriate, all chart endpoints lie in \((-1,1)\). A rational accuracy target \(\delta\), with \(2\pi\delta/c<\pi/2\), satisfies the native diameter test exactly when
\[
b-a\le \tan(2\pi\delta/c)(1+ab).
\tag{G8}
\]
The comparison constant is algebraic. Its representation cost can grow with the target's denominator; .001 is a fixed target. No transcendental equality oracle is needed.

An unrestricted midpoint can be outside the physical set. Requiring the estimator itself to be compatible is the different quantified problem
\(\exists c\in\mathcal C_\epsilon(y)\ \forall\theta\in\mathcal C_\epsilon(y):
|\theta_i-c_i|\le\delta_i\).
Fixed rational targets make this decidable with the same chart comparisons. An optimized feasible-center native radius is not claimed algebraic or attained on an open set.


## 2B. Reduced compiler and exact affine-fiber inverse

The [reduced algebraic addendum](theory/reduced_algebraic_inverse_addendum.md) sharpens the construction without removing any of the eighteen unknowns. Section 2A and the six-root appendix remain valid baselines. The original boxed chart remains the chart used by the coordinatewise native-envelope and boundary results unless a change is explicitly stated.

### Five forward roots in a coupled index chart

Retain the angular/rotor chart, beam slope \(t\), and \(Q=1+|t|^2\). Replace the three independent index coordinates by
\[
h=\sqrt{n_1^2Q-|t|^2}>0,\qquad
c_2=(n_2^2-1)Q,\quad c_3=(n_3^2-1)Q.
\tag{R1}
\]
The exact transformed index prior is
\[
(13/10)^2Q-|t|^2\le h^2\le(9/5)^2Q-|t|^2,\quad h>0,
\]
\[
((13/10)^2-1)Q\le c_j\le((9/5)^2-1)Q,\quad j=2,3.
\tag{R2}
\]
This is a **coupled semialgebraic prior**, not a product box. It has \(h>1,c_j>0\), and the unique inverse is
\[
n_1=\sqrt{\frac{h^2+|t|^2}{Q}},\qquad
n_j=\sqrt{1+\frac{c_j}{Q}}\quad(j=2,3).
\tag{R3}
\]
Native index ranges must optimize these expressions jointly, or use linked native-index output variables. Projecting \(h\) or \(c_j\) alone does not give an index range. In this chart the native index derivatives also involve beam coordinates; the old diagonal native chart derivative must not be reused for them.

Set \(X_0=t,\ Z_0=L_0=1,\ D_j=1+|u_j|^2\), use \(H_1=h\), and define \(c_1=h^2-1\) as an expression. For \(j=2,3\),
\[
H_j=\sqrt{Z_{j-1}^2+c_jL_{j-1}^2}.
\]
For every prism use
\[
\begin{aligned}
P_j&=H_j-u_j\cdot X_{j-1},&
E_j&=\sqrt{P_j^2-D_jc_jL_{j-1}^2},\\
Z_j&=H_jD_j-P_j+E_j,&
X_j&=X_{j-1}D_j+u_j(P_j-E_j),\\
L_j&=L_{j-1}D_j.
\end{aligned}
\tag{R4}
\]
Preserve \(E_j,P_j,Z_j>0\), positive roots and every traversal guard. The invariant
\(|X_j|^2+Z_j^2=QL_j^2\)
proves equivalence to the original recurrence. The five sequential forward roots are \(E_1,H_2,E_2,H_3,E_3\).

For a position \(p=A/T\), \(T>0\), combine the two spatial steps before propagating numerators:
\[
\begin{aligned}
T_{\rm next}&=Z_jP_jT,\\
A_{\rm next}
 &=Z_jP_jA+(Z_jX_{j-1}-H_jX_j)(u_j\cdot A)\\
 &\quad+3[Z_jX_{j-1}-(u_j\cdot X_{j-1})X_j]T
       +\ell_jP_jX_jT .
\end{aligned}
\tag{R5}
\]
Initially \(A=b+6t,T=1\). The cancellation is exact and introduces no conjugate division. Give ordinary inputs weight one and \(c_2,c_3\) weight two. The five root weights are \(2,3,4,5,6\); induction gives final position weights \(28,27\). Every interval-observation atom has weight at most 28. Five radical sign reductions therefore give degree \(28\cdot2^5=896\).

| Conservative compiler bound | Six-root baseline | Five-root reduction |
|---|---:|---:|
| Sequential forward roots per sample | 6 | 5 |
| Instantaneous interval-atom degree | 2368 | 896 |
| Terminal sign tests per atom | 729 | 243 |
| Occurrences with the 24-atom allowance | 17,496 | 5,832 |
| Maximum substituted degree at \(K=200\) | 2,843,968 | 1,076,096 |

Counts refer to shared sign circuits, not expanded Boolean syntax. A per-sample squared Euclidean residual has reduced terminal degree at most 1792. The rotor substitution bound remains \(D(6k+7)\). These are exact representation improvements; their bounds are still very large and do not prove practical decomposition.

### Source transport is invertible at every strict physical optical point

Here use actual incoming/outgoing direction slopes, independently of scaled numerator notation. Let \(a=X_{\rm in}/H\), \(v=X_{\rm out}/Z\),
\(P=H-u\cdot X_{\rm in}>0\), \(R=Z-u\cdot X_{\rm out}>0\).
The position identities give
\[
p_{\rm next}=\mathsf A p+3\mathsf A a+\ell v,\qquad
\mathsf A=(I-vu^T)(I-au^T)^{-1},
\]
\[
\det\mathsf A=\frac{HR}{ZP}>0,\qquad
\mathsf A^{-1}=I+\frac{(v-a)u^T}{1-u\cdot v}.
\tag{R6}
\]
Consequently each sample has the exact form
\[
F_k(x,b,w)=\mathsf T_k(x)b+\mathsf C_k(x)w+d_k(x),
\qquad w=(g,d),\qquad \det\mathsf T_k>0,
\tag{R7}
\]
where \(\mathsf T_k=\mathsf A_{3,k}\mathsf A_{2,k}\mathsf A_{1,k}\). Initial propagation by \(6t\) is a translation. This source-offset elimination requires no generic-rank hypothesis. Its inverse can become ill-conditioned when the outgoing critical-normal factor tends to zero; at a weak boundary with \(R=0\), retain the unreduced graph.

### Exact data leave two geometry variables, with every rank stratum retained

At \(\epsilon=0\), any anchor sample \(k_0\) gives
\[
b=\mathsf T_{k_0}^{-1}
 [y_{k_0}-\mathsf C_{k_0}w-d_{k_0}].
\tag{R8}
\]
Substitute this into every other sample to obtain \(D(x)w=z(x)\). The original source square, distance bounds and strict traversal conditions become affine weak/strict inequalities in \(w\). All must remain. Thus the exact geometry fiber is a mixed-strict polygon cut by linear equations.

Helly order is now three in \(\mathbb R^2\), giving \(O(K^3)\) constant-size feasibility tests. Each pulled-back row uses its own sample and the anchor, so a three-row test can involve four distinct traces. An explicit complete rank atlas is also available: rank two fixes \(w\) from an independent row pair and checks all remaining constraints; rank one retains the fully constrained affine-line interval with endpoint flags; rank zero requires \(D=z=0\) and retains the remaining polygon. Rank-deficient distance gauges are preserved. This does not reduce the fourteen optical unknowns.

For positive error, use a **shared anchor error**
\[
b=\mathsf T_{k_0}^{-1}
 [y_{k_0}+e-\mathsf C_{k_0}w-d_{k_0}],
\qquad e\in[-\epsilon,\epsilon]^2.
\tag{R9}
\]
The exact noisy geometry fiber has four variables \((w,e)\), hence Helly order five. Using independent anchor errors in different rows is invalid. From a transformed weak row
\(a_i^Tw+v_i^Te\le b_i(\epsilon)\),
the necessary relaxation
\[
a_i^Tw\le b_i(\epsilon)+\epsilon\|v_i\|_1
\tag{R10}
\]
screens in two distance variables; a strict original row gives a strict necessary row. Its infeasibility excludes the exact noisy fiber. Its feasibility generally does not establish compatibility, because different rows may require incompatible values of the shared \(e\).

### Explicit mixed-strict circuit elimination and symbolic thresholds

In an affine geometry fiber \(q\in\mathbb R^d\), write all rows as
\(a_i(x)^Tq\le\beta_i(x,y)+\epsilon\delta_i\),
marking strict traversal rows. Observation rows have \(\delta_i=1\); prior/traversal rows have \(\delta_i=0\), or the corresponding positively cleared values. A minimal positive circuit is a nonzero \(\lambda\ge0\) with
\(\sum_i\lambda_i a_i=0\)
and inclusion-minimal support. Its support has at most \(d+1\) rows and rank one less than its size. Smaller supports, including zero-normal singleton rows, are essential on rank-degenerate strata.

Let
\[
B_\lambda=\sum_i\lambda_i\beta_i,\qquad
D_\lambda=\sum_i\lambda_i\delta_i\ge0,
\]
and let \(s_\lambda\) record whether a positively weighted row is strict. The entire affine system is feasible exactly when every circuit satisfies
\[
B_\lambda+\epsilon D_\lambda\ge0\quad(s_\lambda=\mathrm{false}),
\qquad
B_\lambda+\epsilon D_\lambda>0\quad(s_\lambda=\mathrm{true}).
\tag{R11}
\]
The proof combines Farkas' alternative with the positive-circuit decomposition. If a marked row is forced to equality throughout the weak polyhedron, LP duality constructs a zero-pairing dependence involving that row, contradicting a strict circuit condition. The addendum gives the full sufficiency argument.

Enumerate supports of size at most \(d+1\); signed cofactors of independent coordinate columns supply candidate null vectors. Verify annihilation of every column and one common nonzero sign, and orient the vector positively. Zero minors lead to smaller supports rather than unchecked division. This is an explicit determinant/sign compiler of degree at most \(d+1\) in the affine-row coefficients, with \(O(K^{d+1})\) support tests. That coefficient degree is not a small total optical-chart degree.

For fixed optics, first test all permanent \(D_\lambda=0\) conditions, retaining strictness. If any fail, no geometry is feasible at any error level. Otherwise define
\[
\rho(x)=\max\left(0,\max_{D_\lambda>0}
 -\frac{B_\lambda}{D_\lambda}\right).
\tag{R12}
\]
The feasible error budgets are \([\rho,\infty)\), except that the endpoint is excluded exactly when a **strict positive-\(D_\lambda\) circuit attains \(\rho\)**. A strict circuit with a smaller threshold does not open the endpoint. For the full noisy fiber \(d=4\), unanchored five-row circuits involve at most five sampled traces; the anchored representation can involve six including its common anchor. Global minimization of \(\rho(x)\) and optical boundary treatment remain nonlinear tasks.


## 2C. Exact compactification, boundary control and all-branch continuation

The [boundary compactification addendum](theory/global_boundary_compactification.md) proves stronger, model-specific results on the **original boxed index chart**. They supersede uncertainty about whether its weak transmitted/traversal model contains spurious nonapproximable points. They do not make zero-margin boundary points physical. Quantitative moduli and chart balls below use this original chart; transferring them to the coupled five-root chart requires the corresponding coordinate conversion.

### Uniform forward denominators and exact physical closure

For one prism with incoming external unit direction \((X,z)\), define
\[
H=\sqrt{n^2-|X|^2},\quad D=1+|u|^2,\quad
\nu=n^2-1,\quad P_n=H-u\cdot X,
\]
\[
\Delta=P_n^2-D\nu,\quad R=\sqrt\Delta,\quad
c=\frac{P_n-R}{D}=\frac{\nu}{P_n+R},\quad
B=X+cu,\quad Z=H-c .
\tag{B1}
\]
Here \(P_n\) is optical momentum, not the physical parameter domain. On \(\Delta\ge0\),
\(|B|^2+Z^2=1\) and \(Z-u\cdot B=R\).
Set
\[
u_*=\tan(\pi/10),\quad h_*=\sqrt{(13/10)^2-1},\quad
\nu_*=(9/5)^2-1,\quad
z_0^*=[1+2\tan(5\pi/36)^2]^{-1/2},
\]
\[
z_j^*=\sqrt{\nu_*+(z_{j-1}^*)^2}-\sqrt{\nu_*}>0.
\tag{B2}
\]
The original prior itself implies, at every sample on the whole weak transmitted branch,
\[
H_j\ge h_*,\qquad P_{n,j}\ge h_*,\qquad Z_j\ge z_j^*>0.
\tag{B3}
\]
Indeed \(P_n\ge\sqrt{D\nu}\), \(c\le\sqrt\nu\), and
\(Z\ge\sqrt{\nu+z^2}-\sqrt\nu\).
Thus **axial and incoming-normal grazing cannot occur in this model's physical closure**. Critical outgoing-normal refraction \(R=0\) remains possible, so derivatives through \(\sqrt\Delta\) and inverse source transport can still become ill-conditioned.

Let \(\Pi\) be the original closed eighteen-dimensional chart box, \(P\) the strict physical set at all 200 times, and \(W\) the set obtained by imposing \(\Delta_j(k)\ge0\) and weak versions of the internal and external traversal inequalities. Equations (B1)-(B3) define a unique continuous semialgebraic forward extension \(\bar{\mathcal F}\) on this compact set. The key theorem is
\[
\boxed{W=\overline P.}
\tag{B4}
\]
It is stronger than simply replacing strict inequalities by weak ones.

For a fixed weak point, scale the signed wedge slopes inward by
\[
r_1(\tau)=(1-\tau^{16})r_1,\qquad
r_2(\tau)=(1-\tau^4)r_2,\qquad
r_3(\tau)=(1-\tau)r_3 ,
\tag{B5}
\]
holding every other parameter fixed. Here \(\tau\) is a local path parameter. At an active constraint for a single prism scaled by \(u\mapsto\lambda u\), the inward derivatives at \(\lambda=1\) are
\[
\Delta'=-2P_nZ<0,\qquad
(3+\lambda u\cdot p)'=-3,\qquad
B_{\rm flight}'=-H\ell/P_n<0.
\tag{B6}
\]
These constraint functions are smooth in the incoming state even when the current outgoing root is critical. The first prism's \(O(\tau^{16})\) forcing changes its output by \(O(\tau^8)\); the second prism's \(O(\tau^4)\) inward margin dominates this upstream change and produces \(O(\tau^2)\) output change; the third prism's \(O(\tau)\) margin dominates that. The finitely many sample constraints are simultaneously strict for sufficiently small positive \(\tau\). Zero wedges have no active constraints in this inward argument.

This proves (B4) without a uniform strictification threshold. For algebraic weak points, the exact sign compiler reduces certification of a suitable interval \(0<\tau<\tau_0\) to univariate root isolation. Parameter-neighborhood constraints and any residual allowance **strictly above** the weak residual can be included. Equality at that residual is not promised.

Writing \(O=P\cap\operatorname{int}\Pi\), one also has
\(W=\overline O\) and \(\dim(W\setminus O)\le17\).
Prior faces remain distinct from excluded critical/traversal faces: some prior-face points are strictly physical but lie outside \(O\).

### Boundary near-aliases and the exact limiting uncertainty

Compactness and (B4) give
\[
\overline{\mathcal F(P)}=\bar{\mathcal F}(W),\qquad
\inf_{\vartheta\in P}\|\mathcal F(\vartheta)-y\|_\infty
 =\min_{\vartheta\in W}\|\bar{\mathcal F}(\vartheta)-y\|_\infty=a.
\tag{B7}
\]
The minimizing point may be nonphysical. A strict physical witness is still necessary to decide consistency at \(\epsilon=a\).

Let \(M_y\subset W\) be the compact set of all minimizers in (B7). As \(h\downarrow0\), the closures of strict compatible sets at allowance \(a+h\) converge in Hausdorff distance to \(M_y\). Strictification approximates each minimizer within every larger allowance; compactness puts every limiting compatible sequence in \(M_y\). Native-coordinate uncertainty diameters therefore converge exactly to the corresponding diameters of \(T(M_y)\), where \(T\) is the native inverse chart. For an exact record this is the entire weak exact fiber. A unique strict exact solution can still have a positive limiting uncertainty due to another weak exact solution.

Records approached through excluded physical boundaries are exactly
\[
\mathcal A_{\rm boundary}=\bar{\mathcal F}(W\setminus P).
\tag{B8}
\]
This set is compact semialgebraic of dimension at most 17, relative to the eighteen-dimensional model image. If a relatively open retained region \(U\subset W\) contains the complete weak preimage of a closed noise cube, then the compact exterior \(W\setminus U\) has a positive residual gap above that allowance. Checking the full weak preimage remains a global task. There is no unrepresented escape; boundary alternatives must be included in the check.

### A projected discriminant and complete sheet continuation

Choose a fixed selector \(L\) of eighteen original scalar observations and put
\(\mathcal H=L\bar{\mathcal F}\).
A nonzero sample-Jacobian minor shows that some selector works locally at the known witness; the particular selector must be chosen and certified. Define
\[
K_{\mathcal H}=\{\vartheta\in O:\det D\mathcal H(\vartheta)=0\},
\qquad
\mathcal D_{\mathcal H}
 =\mathcal H(W\setminus O)\cup
   \overline{\mathcal H(K_{\mathcal H})}.
\tag{B9}
\]
The discriminant is compact semialgebraic with dimension at most 17. It includes prior boundaries, critical physical boundaries, critical projected values, and the image of any component where the chosen projection is everywhere rank deficient. Such components are retained, not assumed absent.

Over a connected component \(V\) of \(\mathbb R^{18}\setminus\mathcal D_{\mathcal H}\), the map
\[
\mathcal H:\mathcal H^{-1}(V)\longrightarrow V
\tag{B10}
\]
is either empty or a proper finite-sheet covering. Every derivative is invertible there; compact fibers are locally discrete and hence finite. Compactness prevents additional roots from appearing outside the inverse neighborhoods, proving the covering property. The projected root count \(m(V)\) is constant and includes every physical connected component.

Complete enumeration of the anchor fiber at one \(z_0\in V\), followed by lifting every root, gives all projected roots along paths in \(V\). One fitted root is insufficient. Simply connected subregions support individual inverse branches; nontrivial chambers can permute branches by monodromy. For a coordinate selector, the exact backend imposes the eighteen selected scalar equations together with all original per-sample physical predicates. This preserves the sample-local encoding and its stated polynomial-in-sample-count complexity qualifications, although its initial cost can be enormous. Dense projections require the separate coupled-graph treatment qualified below.

For exact data, filter every root of \(\mathcal H^{-1}(Ly)\) against **all 400 original equations**. For noisy data, lift the **entire projected noise set** on every sheet, then filter all 400 inequalities. Filtering the anchor fiber alone loses noisy systems. Only projected root counts are constant on a chamber; the unprojected residual filters can change the number of full-record exact aliases.

### Conditional all-branch noise cover

For the selector above, let \(z_0=Ly\) and
\(C_\epsilon=z_0+[-\epsilon,\epsilon]^{18}\).
Assume the following are certified:

1. The whole cube \(C_\epsilon\) is disjoint from \(\mathcal D_{\mathcal H}\).
2. Every anchor root \(\vartheta_1,\ldots,\vartheta_m\) in \(W\) is enumerated.
3. \(\sigma_{\min}(D\mathcal H)\ge\eta>0\) on the full preimage of the cube, in Euclidean chart and projected-observation norms.

Then every original full-record compatible system lies in
\[
\boxed{
\bigcup_{i=1}^m
 \overline B\!\left(\vartheta_i,\frac{\sqrt{18}\,\epsilon}{\eta}\right).}
\tag{B11}
\]
The cube lies in one chamber and has a slightly expanded convex neighborhood there. Each sheet is single-valued on it, and integration of
\(D\vartheta(z)=[D\mathcal H(\vartheta(z))]^{-1}\)
along straight segments proves (B11). If \(m=0\), the entire cube has no preimage and the original noisy inverse is empty. For \(m>0\), compactness guarantees some positive \(\eta\), but its useful value must be certified.

A sufficient singular-value certificate is
\(|\det D\mathcal H|\ge d_*>0\) and \(\|D\mathcal H\|_2\le M_*\), giving
\(\eta=d_*/M_*^{17}\).
These and the discriminant/anchor conditions are exact semialgebraic certificate problems under finite input encodings. They are not proved inexpensive.

The balls are in chart units. For native guarantees, project them to intervals, intersect the original chart prior, and apply the exact native maps. A chart radius is not directly an angular or frequency error; unchanged indices and geometry retain their native units. With the reduced index chart, use the nonlinear output expressions (R3) instead.

An arbitrary fixed rational projection also has the covering theorem; its projected noise is a zonotope, which may be enclosed in a cube of half-width \(\|L\|_{\infty\to\infty}\epsilon\). Its cover radius gains that factor. Dense projections couple outputs across samples, so the earlier sample-local polynomial-in-\(K\) compilation bound does not transfer automatically; the coordinate-selector construction avoids that additional complexity issue.

Failure of a cube-disjointness test, expensive anchor enumeration, many sheets or a tiny \(\eta\) means this certificate is unhelpful or unresolved, not that the record is inconsistent. Exceptional regions stay in the complete backend. Likewise, signed degree one is not a uniqueness proof: oppositely oriented sheets can cancel. A common determinant sign plus absolute degree one, or complete anchor enumeration with one root, supplies the missing qualification.

### Global critical-boundary modulus and exclusion

The boundary appendix gives explicit nonnegative recurrences for direction errors, position magnitudes and position errors over all 200 times, with geometry fixed anywhere in its prior box. They use the rotor Lipschitz bound
\[
L_u=1+2u_*(1+199)
\]
and \(|\sqrt s-\sqrt t|\le\sqrt{|s-t|}\), together with (B3). No connecting segment has to remain physically admissible.

Denote the resulting **forward** modulus by \(\omega_{\rm fwd}(\delta)\), to distinguish it from the inverse ambiguity modulus \(\Omega\) in (E3). Its complete generating recurrences are equations (23)-(25) of the boundary appendix. It is explicit, continuous and nondecreasing, with
\[
\|\bar{\mathcal F}(\xi,q)-\bar{\mathcal F}(\xi',q)\|_\infty
 \le\omega_{\rm fwd}(\|\xi-\xi'\|_\infty),\qquad
\omega_{\rm fwd}(\delta)\le C\delta^{1/8}\quad(0\le\delta\le1),
\tag{B12}
\]
where one valid generated constant is \(C=\omega_{\rm fwd}(1)\). The exponent is a safe three-critical-root bound, not a sharp universal exponent. Constants can be extremely conservative. The same recurrence gives a traversal modulus \(\phi_{\rm trav}(r)\to0\).

At any weak optical anchor \(\xi_0\), choose a common outer geometry polytope
\[
Q_B=Q\cap\{q:g_s(\xi_0,q)\ge-\phi_{\rm trav}(r)\ \forall s\},
\]
where all optical points under consideration are within distance \(r\) of the anchor. A verified LP dual with center lower bound \(\alpha\) gives
\[
\|\mathcal F(\xi,q)-y\|_\infty
 \ge\alpha-\omega_{\rm fwd}(r).
\tag{B13}
\]
Thus \(\alpha-\omega_{\rm fwd}(r)>\epsilon\) excludes the region even at critical refraction, without convexity or a positive critical-radicand margin. The earlier dual residual correction remains valid.

If every weak physical system outside an open retained optical region has residual strictly above \(\epsilon\), compactness gives a positive exterior gap. As \(r\downarrow0\), the common traversal polytopes decrease to the exact weak traversal fiber; compactness makes their residual optima converge, or proves eventual emptiness. Combining this with \(\omega_{\rm fwd}(r)\to0\) gives an excluding neighborhood at each exterior anchor and a finite subcover. This is a qualitative finite-exclusion theorem, not a useful count or runtime bound. Optical points outside the weak transmitted domain still require certified complementary coverage.

The conditional all-branch theorem is now established. Remaining hard tasks are complete anchor enumeration, discriminant and entire-noise-cube certification, useful inverse margins and numbers of candidate regions, and treatment of records meeting exceptional strata. Neither compactification nor continuation removes exact gauges or remote aliases.


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
e_i=\kappa\rho_i
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
\(1,\ \kappa^{-1},\ \kappa^{-2}\), respectively.
The selected \(18\times18\) matrix tends to a block triangular matrix with nonzero diagonal blocks by (17), the fundamental/confluent separation, and (22).

All \(C^1\) remainders vanish after this scaling on a compact strict analytic chart. The fiber preserves the zeroth and first-order maps, leaving its first nonzero response at quadratic order. Unrepresented higher harmonics can leak into a finite extractor, but their scaled contribution tends to zero; they are not assumed absent. Frequency derivatives of quadratic terms are \(O(\kappa^2)\) and vanish after the first-order \(\kappa^{-1}\) scaling, so repeated quadratic nodes are unnecessary.

**Physical-vector finite-record theorem.** For all sufficiently small positive \(\kappa\), the exact coupled-vector \(400\times18\) sampled Jacobian has rank eighteen at this admissible oblique family. A nonsingular eighteen-output minor and the inverse function theorem give local injectivity. All eighteen parameters, including beam direction and source position, are unknown.

No numerical upper bound on \(\kappa\), usable singular-value bound, global uniqueness or noise guarantee is proved by this argument.

### Component-qualified generic conclusions

A nonzero analytic minor implies generic rank eighteen on the connected analytic interior component containing this witness. This is not a statement about every physical component or every point of the native box.

For the same ideal clock, rotor-step circles and positive-root vector equations give a semialgebraic lift. Use an injective native algebraic chart, with the original signed speeds, phases and wedge information retained even at zero wedges. Outside a lower-dimensional exceptional subset of the component's eighteen-dimensional image, exact fibers consist of isolated regular points; semialgebraicity makes such fibers finite. “Generic” is relative to that image, not to arbitrary points of the 400-dimensional observation space. A finite fiber need not contain one system.

Included native boundary faces have lower-dimensional images. For this particular prior, section 2C proves a unique finite continuous extension to \(W=\overline P\); no undefined axial or incoming-normal grazing escape remains. Critical-normal and traversal boundary values are genuine limits, but zero-margin points remain nonphysical. Robust continuation must account for those weak fibers and all prior-boundary images in its discriminant.

The finiteness conclusion concerns \(F(\theta)=y\), or \(\epsilon=0\). With positive error allowance and strict residual slack, an interior compatible point has an open neighborhood of compatible systems. Neither generic finiteness nor local rank converts positive-noise compatibility into a unique parameter vector.

## 7. Symbolic-error profile and conditional native uncertainty

### The whole error profile, with the record held fixed

Keep \(y\) and nonnegative weights \(\omega_m\) fixed, both with finite rational or algebraic encodings, and leave \(\epsilon\ge0\) free in (G4). The joint set
\[
\mathfrak S_y=\{(\epsilon,x):\epsilon\ge0,\ x\in\mathcal S_\epsilon(y)\}
\tag{E1}
\]
is semialgebraic in nineteen variables. Complete decomposition therefore describes the full family, including parameter values where new components enter, merge, disappear from a chart, or meet physical boundaries. The compatibility family itself is nested: an actual compatible system cannot be lost when \(\epsilon\) increases.

The consistency set
\(\mathcal E_y=\{\epsilon\ge0:\mathcal C_\epsilon(y)\ne\varnothing\}\)
is an upper interval when nonempty. Its lower endpoint can be excluded because optical guards are strict and the best residual may have an unattained infimum. Empty compatibility is model/data inconsistency, not a successful accuracy guarantee.

On \(\mathcal E_y\), each chart-coordinate infimum and supremum is a bounded semialgebraic function of \(\epsilon\); lower envelopes are nonincreasing and upper envelopes nondecreasing. A finite decomposition of the error axis describes these as algebraic branches over the fixed input field, allowing jumps and excluded endpoints. At an algebraically specified allowance the chart endpoints are algebraic numbers. Native angular endpoints are scaled arctangents of those functions; **the native angular profile itself is not claimed semialgebraic or algebraic**.

Nevertheless, any fixed rational native tolerance has a semialgebraic decision profile because (G8) expresses the angular comparisons exactly. Thus
\[
\mathcal G_y(\delta)=
\{\epsilon\in\mathcal E_y:
 D_i(\epsilon)\le2\delta_i\ \text{for all }i=1,\ldots,18\}
\tag{E2}
\]
is effectively describable by real-algebraic operations under the finite input assumptions. It is an interval within the consistency range, possibly empty, a singleton, or with excluded endpoints. It need not start at zero. Its finite breakpoints are algebraic in the chart/input representation. The .001 guarantee can therefore be stated as a symbolic error interval with explicit endpoint inclusion, rather than tested at arbitrary noise levels.

The true diameters \(D_i(\epsilon)\) and risk \(R_y(\epsilon)\) are nondecreasing on the consistency range. Branch changes can produce jumps. A local bound or fitted condition number need not be monotone and cannot replace this full profile. This is an exact algorithmic characterization; no useful numerical error interval for the supplied observations has yet been computed.

### Exact consistency and unacceptable-error thresholds

The complete audited [parametric observation-error theorem](theory/parametric_noise_inverse.md) proves the following threshold construction for the original **uniform** coordinate allowance, \(\omega_m=1\). Write \(x\) for the eighteen chart coordinates, \(T(x)\) for the native inverse chart, and \(\mathcal F(x)=F(T(x))\). Let \(\mathcal P_x\) be the chart prior restricted to the strict sampled-time physical branch, and put
\[
r_y(x)=\|\mathcal F(x)-y\|_\infty,\qquad
a=\inf_{x\in\mathcal P_x}r_y(x).
\tag{P1}
\]
The domain is nonempty and each admissible system has a finite record. Consequently the consistency interval is exactly
\[
\mathcal E_y=[a,\infty)\quad\hbox{or}\quad(a,\infty).
\tag{P2}
\]
The finite nonnegative threshold \(a\) is algebraic for the stated input encodings. Projection of the nineteen-variable family computes it; an exact feasibility test at \(a\) determines endpoint inclusion. The prior box is bounded, but \(\mathcal P_x\) is open at some physical boundaries. A sequence approaching excluded critical-normal or traversal faces need not attain its best residual within the strict physical set. Section 2C proves that the same infimum is attained on the compact weak closure, while physical endpoint membership still requires a strict witness.

For fixed rational native targets \(\delta_i\), let \(V_\delta(x,x')\) be the disjunction that some native coordinate separation exceeds \(2\delta_i\), using (G8). Define the unacceptable set
\[
\mathcal B_y(\delta)=
\{\epsilon\ge0:\exists x,x'\in\mathcal S_\epsilon(y),\
 V_\delta(x,x')\}.
\tag{P3}
\]
It is an upward semialgebraic set. When nonempty it is
\([b,\infty)\) or \((b,\infty)\), with algebraic finite threshold
\[
b=\inf_{\substack{x,x'\in\mathcal P_x\\V_\delta(x,x')}}
 \max(r_y(x),r_y(x')).
\tag{P4}
\]
Its endpoint inclusion is decided separately at \(b\). If this set is empty in a general target problem, use \(b=+\infty\).

The exact successful-error set is
\[
\boxed{\mathcal G_y(\delta)=\mathcal E_y\setminus\mathcal B_y(\delta).}
\tag{P5}
\]
It is one interval, a singleton, or empty; it cannot contain disconnected safe intervals. For nonempty bounded \(\mathcal G_y\), return its supremum **and its membership flag**. A supremum is a maximum admissible error only if that endpoint belongs to \(\mathcal G_y\). At an inconsistent allowance, report inconsistency. At an unacceptable allowance, extract two actual physical witnesses whose separation exceeds the target. Unattained extrema and excluded boundary points are never returned as physical ambiguity witnesses.

For \(\delta_i=.001\), the exact chart comparison constants are
\[
b_{\rm wedge}=b_{\rm beam}=\tan(\pi/90000),\quad
b_{\rm phase}=\tan(\pi/180000),\quad
b_{\rm speed}=\tan(\pi/10000);
\tag{P6}
\]
unchanged-coordinate separation is compared with \(1/500\). These are algebraic constants with genuine representation costs, not negligible-cost numbers merely because the tolerance has a short decimal representation.

The threshold is always finite for this particular target. The axial, zero-wedge, zero-offset systems produce the zero record while their phase, speed, index, gap and distance coordinates vary. Two can differ in phase by 36 degrees. For any fixed observed \(y\), both are compatible once \(\epsilon\ge\|y\|_\infty\). Hence
\[
0\le a\le b\le\|y\|_\infty.
\tag{P7}
\]
For \(y=0\), the safe set is empty even at zero error.

### Computing the thresholds without a 37-variable pair decomposition

Direct elimination in (P3) uses two eighteen-coordinate systems and one shared error parameter: **37 variables**, not nineteen. It is a correct complete alternative, with a conservative CAD arithmetic bound \((sd)^{2^{O(37)}}\), plus coefficient-bit costs.

A sharper route uses eighteen CAD stages of at most nineteen variables, one per coordinate. In each stage order \(\epsilon\) first and \(x_i\) second, and project all retained full cells to their first two coordinates. Cylindricity makes the union of marked projected cells precisely the projected compatible set. On each error base cell, take the lowest and highest marked coordinate boundaries, even if those boundaries are excluded from the physical set; at point error cells use the actual fiber to preserve jumps. Boundedness makes all coordinate endpoints finite.

Convert these lower and upper envelope graphs to exact polynomial sign predicates with their root-order data. For each coordinate, the conjunction of its two envelope graphs and its diameter-violation comparison uses just \((\epsilon,l,u)\). Project \(l,u\) and unite the eighteen resulting sets. This computes exactly \(\mathcal B_y(\delta)\), including nonattainment: a diameter strictly exceeding the threshold guarantees a pair of actual points exceeding it.

All graph-conversion, formula-size and coefficient-height growth must be charged. The resulting bound has the same general fixed-dimensional form \((sd)^{2^{O(19)}}\) with adjusted fixed constants and polynomial bit factors, where \(s,d=O(K)\) with the very large compiler constants. Exact algebraic data and target constants retain their encoding costs. This improves the theoretical construction; it is not a practical threshold computation. No numerical threshold for the original observations has been computed.

### Singular endpoints: fractional powers, uncertainty floors and jumps

Fix a finite algebraic endpoint or algebraically chosen subdivision value \(\epsilon_0\) and one side on which the consistency set contains an interval. Put \(h=|\epsilon-\epsilon_0|>0\). On a sufficiently short interval, each bounded chart envelope is a continuous semialgebraic function. It has a convergent algebraic Puiseux expansion after a common integer ramification:
\[
L_i(\epsilon)=l_i^0+\sum_{n\ge1}a_{i,n}h^{n/q},\qquad
U_i(\epsilon)=u_i^0+\sum_{n\ge1}b_{i,n}h^{n/q}.
\tag{P8}
\]
Boundedness excludes negative powers. Limits and chart coefficients are algebraic under the exact algebraic input model. Coste explains the semialgebraic-germ/Puiseux identification and convergence in Chapter 1, printed p. 10 of [Real Algebraic Sets, ICTP primary mirror](https://indico.ictp.it/event/a02455/session/23/contribution/14/material/0/0.pdf); see also [Barone and Basu, section 2.5](https://www.math.purdue.edu/~sbasu/refined-04-06-11.pdf).

Each coordinate has one of three behaviors:

- If \(u_i^0>l_i^0\), native uncertainty tends to the positive floor \(T_i(u_i^0)-T_i(l_i^0)\).
- If \(U_i-L_i\) vanishes identically on that side, the coordinate is fixed throughout those fibers.
- Otherwise, if both limits equal \(z_i\), there are positive rational exponents \(\nu_i,\eta_i\) and \(A_i>0\) such that
  \[
  U_i-L_i=A_i h^{\nu_i}+O(h^{\nu_i+\eta_i}),\qquad
  D_i(\epsilon)=T_i'(z_i)A_i h^{\nu_i}
     +O(h^{\nu_i+\eta_i'}),\quad\eta_i'>0.
  \tag{P9}
  \]

The native exponent is preserved because
\[
T_i(U)-T_i(L)
 =(U-L)\int_0^1T_i'(L+t(U-L))\,dt
\]
and the second factor has positive limit. Native coefficients may contain \(1/\pi\); they are not all claimed algebraic. If every floor vanishes, the nonzero minimax radius has the smallest positive shrinking exponent among the coordinate widths, with the largest associated leading half-width coefficient.

Actual endpoint fibers must still be tested separately: their values can differ from one-sided limits. A singular isolated solution can have an exponent below one; a gauge or distant boundary-approaching near-alias can create a positive floor. Even uniqueness of the exact endpoint fiber does not rule out the latter. The selected algebraic branches permit exact Puiseux extraction in principle, without a claimed small universal exponent or practical extraction cost.

### Sharp regular-branch sensitivity in all native coordinates

The following uses the uniform error model and an **exact central record**. Let \(x_0\) be an interior chart point with strict physical margins, \(y=\mathcal F(x_0)\), and
\(J=D\mathcal F(x_0)\in\mathbb R^{m\times18}\), \(m=2K=400\), of full column rank. Put
\[
\mathcal K_J=\{h:\|Jh\|_\infty\le1\},\qquad c_i=DT_i(x_0),
\]
\[
\boxed{
\chi_i=\max_{h\in\mathcal K_J}|c_i h|
 =\min_{\lambda:J^T\lambda=c_i^T}\|\lambda\|_1,\qquad i=1,\ldots,18.}
\tag{P10}
\]
These are exact primal/dual linear-programming sensitivities. Full column rank makes \(\mathcal K_J\) compact and the dual feasible. The notation \(\chi_i\) corresponds to the memo's locally defined \(\kappa_i\); the report reserves \(\kappa\) for wedge scale. For angle-like coordinates, native derivatives include \(c/[\pi(1+x^2)]\), with \(c=180,360,20\) as appropriate. Unchanged coordinates have derivative one.

For a sufficiently small regular branch,
\[
D_i^{\rm local}(\epsilon)=2\chi_i\epsilon+O(\epsilon^2),\qquad
R_{\rm local}(\epsilon)=\epsilon\max_i\chi_i+O(\epsilon^2).
\tag{P11}
\]
The leading coefficients are sharp. The polytope describes the limiting scaled compatible set; a \(1-O(\epsilon)\) contraction gives actual nonlinear-compatible directions. The next subsection supplies explicit matching bounds. A nonexact record at a positive consistency threshold can have activated boundaries, fractional powers or jumps; (P11) is not asserted there.

### Explicit certified range and matching upper and lower bounds

Choose a **closed chart ball** \(\overline B(x_0,R)\) strictly inside the prior and sampled-time physical branch. Every root, axial, normal and traversal margin must have a positive lower bound throughout it. Assume certified constants
\[
\gamma=\min_{\|h\|_2=1}\|Jh\|_\infty>0,\qquad
\|\mathcal F(x_0+h)-y-Jh\|_\infty\le\tfrac H2\|h\|_2^2
\]
on the ball, with \(HR\le\gamma\). Define
\[
C_0=\frac{H}{2\gamma^2},\qquad
\mathcal S_\epsilon^R=\mathcal S_\epsilon(y)\cap\overline B(x_0,R).
\tag{P12}
\]
For
\[
0\le\epsilon\le\gamma R/2,\qquad C_0\epsilon\le1/2,
\tag{P13}
\]
where the second restriction is automatic if \(H=0\), the exact sandwich is
\[
\boxed{
(\epsilon-C_0\epsilon^2)\mathcal K_J
 \subseteq\mathcal S_\epsilon^R-x_0
 \subseteq(\epsilon+4C_0\epsilon^2)\mathcal K_J.}
\tag{P14}
\]
The intersection with the certified ball is essential; this statement alone says nothing about exterior compatible systems.

For the upper inclusion, the Taylor lower bound on the ball gives
\(r_y(x_0+h)\ge(\gamma/2)\|h\|_2\), hence
\(\|h\|_2\le2\epsilon/\gamma\).
The remainder then yields
\(\|Jh\|_\infty\le\epsilon+2H\epsilon^2/\gamma^2\).
For the lower inclusion, take \(z\in\mathcal K_J\), so \(\|z\|_2\le1/\gamma\), and
\(h=(\epsilon-C_0\epsilon^2)z\). It stays inside the ball and has residual at most
\[
\epsilon(1-C_0\epsilon)
 +C_0\epsilon^2(1-C_0\epsilon)^2\le\epsilon.
\]
Both signs of an extremal direction therefore correspond to actual compatible systems.

Suppose also
\[
|T_i(x_0+h)-T_i(x_0)-c_i h|
 \le\tfrac12 B_i^{\rm nat}\|h\|_2^2
\]
throughout the ball. Unchanged coordinates have \(B_i^{\rm nat}=0\); arctangent second derivatives supply explicit bounds for angle-like coordinates. For half the local native diameter \(r_i^R(\epsilon)=D_i^R(\epsilon)/2\),
\[
\boxed{
\chi_i\epsilon-
 \left(C_0\chi_i+\frac{B_i^{\rm nat}}{2\gamma^2}\right)\epsilon^2
 \le r_i^R(\epsilon)
 \le \chi_i\epsilon+
 \left(4C_0\chi_i+\frac{2B_i^{\rm nat}}{\gamma^2}\right)\epsilon^2.}
\tag{P15}
\]
The leading LP coefficients are **sharp**; the displayed quadratic constants are **valid sufficient constants, not claimed optimal**.

For global recovery, separately certify the exterior residual gap
\[
\rho=\inf_{x\in\mathcal P_x\setminus B(x_0,R)}r_y(x)>0.
\tag{P16}
\]
Only when \(\epsilon<\rho\) does every compatible system lie inside the open ball, making (P14)-(P15) global. Global uniqueness of an exact solution plus a full-rank Jacobian does not prove this gap: distant physical systems can approach an excluded boundary while their residual tends to zero. Algebraically specified ball boundaries permit exact gap computation through the global backend. A compact guarded domain would remove this escape route, but such restrictions must be explicitly justified.

Under all range, derivative and global-gap conditions, the sufficient all-eighteen .001 threshold inequalities are
\[
\chi_i\epsilon+
 \left(4C_0\chi_i+\frac{2B_i^{\rm nat}}{\gamma^2}\right)\epsilon^2
 \le .001\quad\forall i.
\tag{P17}
\]
If the corresponding lower expression in (P15) exceeds .001 for any coordinate, even the local branch is too wide for that target. These are matching conditional bounds; the exact threshold set remains (P5). Strict margins establish that derivative bounds are finite, but do not automatically supply useful numerical values. No \(\gamma,H,R,\rho,\chi_i\) certificate for the actual record has been evaluated here.

### Geometry conditioning and piecewise-affine exact profiling

Split the chart as \(x=(\xi,q)\), with fourteen optical coordinates \(\xi\) and \(q=(b_x,b_y,g,d)\). Geometry affinity gives
\[
\mathcal F(\xi,q)=M(\xi)q+c(\xi),\qquad
J=[A\ M],\quad A=\partial_\xi\mathcal F(x_0).
\]
At a full-rank point \(\operatorname{rank}M=4\). Let
\[
\Pi=I-M(M^TM)^{-1}M^T,\qquad S=A^T\Pi A.
\tag{P18}
\]
The Schur complement \(S\) is positive definite exactly when all fourteen optical directions remain identifiable after geometry is fitted. For \(e=Jh\),
\[
h_\xi=S^{-1}A^T\Pi e,\qquad
h_q=(M^TM)^{-1}M^T(e-Ah_\xi).
\]
Writing \(\sigma_\xi=\sigma_{\min}(\Pi A)>0\) and
\(\sigma_q=\sigma_{\min}(M)>0\), the unit linear error polytope obeys
\[
\|h_\xi\|_2\le\frac{\sqrt m}{\sigma_\xi},\qquad
\|h_q\|_2\le
 \frac{\sqrt m(1+\|A\|_2/\sigma_\xi)}{\sigma_q}.
\tag{P19}
\]
Native derivatives convert these chart bounds to the requested units; the exact LP constants (P10) can be sharper.

For fixed optical coordinates and finite \(\epsilon\), geometry profiling remains the exact four-dimensional mixed-strict linear feasibility problem. Band endpoints depend affinely on \(\epsilon\), but coordinate extrema are generally **piecewise affine** as active constraints change. They are not globally affine functions of error. The strict Helly and closure-vertex constructions retain \(\epsilon\) as an additional parameter, preserve original strict-fiber feasibility, and include unattained extrema. Linearized Schur conditioning supplements this exact feasibility analysis; it does not replace it.


### Certified optical-box exclusion with exact geometry profiling

The [profiled optical-box exclusion addendum](theory/profiled_box_exclusion.md) supplies a sound reusable certificate. It is a conditional method for accelerating global localization, not a proof that a useful cover of the present record has already been found.

Let \(B\) be a convex box in the fourteen optical coordinates \(\xi\), with algebraic center \(\xi_0\) and infinity-norm radius \(r\). Certify positive lower margins for all six radicands and required optical normal/axial quantities on the whole box. Then
\(\mathcal F(\xi,q)=M(\xi)q+c(\xi)\)
is analytic there. Its affine position extension is defined for every geometry \(q\) in the original closed box \(Q\), including geometries violating traversal order. Using that extension for lower bounds is a relaxation, not a declaration of physical validity.

Differentiate the exact rational rotor and six-root formulas, including
\(D\sqrt R=DR/(2\sqrt R)\), and propagate exact algebraic or outward rational intervals through positive denominators. Root and position denominator bounds must come from certified margins. For any fixed \(\lambda\in\mathbb R^{400}\), obtain
\[
L_{B,\lambda}\ge
 \sup_{\xi\in B,\ q\in Q_B}
 \|D_\xi\mathcal F(\xi,q)^T\lambda\|_1,
\tag{X1}
\]
where \(Q_B\subseteq Q\) is a nonempty compact polytope containing every physical geometry possible anywhere in \(B\). If an enlarged outer set is used, recompute derivative bounds over that set. Center positivity or an unverified floating derivative is insufficient.

The original \(Q\) is always valid. For a tighter common outer polytope, write each strict traversal condition as
\(g_s(\xi,q)=a_s(\xi)^Tq+b_s(\xi)>0\),
and certify
\[
E_s(B)\ge\sup_{\xi\in B,q\in Q}
 |g_s(\xi,q)-g_s(\xi_0,q)|.
\]
Then
\[
Q_B=Q\cap\{q:g_s(\xi_0,q)\ge-E_s(B)\ \forall s\}
\tag{X2}
\]
contains all potentially physical geometries. Its weak inequalities are intentional outer bounds. A point surviving only on a traversal boundary is not thereby a physical witness.

Write \(Q_B=\{q:A_Bq\le b_B\}\). Any exactly verified pair satisfying
\[
\|\lambda\|_1\le1,\quad\mu\ge0,\quad
 A_B^T\mu=-M(\xi_0)^T\lambda
\]
gives
\[
\alpha=\lambda^T(c(\xi_0)-y)-b_B^T\mu,\qquad
\|\mathcal F(\xi,q)-y\|_\infty\ge\alpha-L_{B,\lambda}r
\tag{X3}
\]
for every physical system in the optical box. The dual inequalities lower-bound the center pairing; convexity of \(B\) and (X1) bound its change. Therefore the box is excluded for the exact symbolic range
\[
\boxed{\epsilon<\alpha-L_{B,\lambda}r.}
\tag{X4}
\]
Equality is retained unless another certificate resolves it. Maximizing the dual center bound gives the exact four-geometry-variable Chebyshev LP optimum, with one additional residual variable; a valid certificate does not require an optimal dual.

For \(Q=q_c+\prod_i[-R_i,R_i]\), the support formula
\[
\alpha=\lambda^T(\mathcal F(\xi_0,q_c)-y)
 -\sum_iR_i|(M(\xi_0)^T\lambda)_i|
\tag{X5}
\]
avoids explicit geometry multipliers. A proposed general dual need not satisfy its equality exactly. If \(\|\lambda\|_1\le1,\mu\ge0\) are verified, put
\(e_{\rm dual}=M(\xi_0)^T\lambda+A_B^T\mu\).
Replace \(\alpha\) by a certified lower bound on
\[
\lambda^T(c(\xi_0)-y)-b_B^T\mu
 +e_{\rm dual}^Tq_c-\sum_iR_i|e_{{\rm dual},i}|.
\tag{X6}
\]
This support correction over \(Q\supseteq Q_B\) makes the residual equality error explicit. All coefficients, norm/sign checks and arithmetic require exact values or outward bounds. Separately,
\(\mu\ge0,\ A_B^T\mu=0,\ b_B^T\mu<0\)
is a Farkas certificate that \(Q_B\) is empty and the box contains no physical system.

An interval upper bound at most zero for a required strictly positive branch quantity can exclude a box only if the preceding expressions enclose every potentially physical branch. Potential positivity without a certified positive lower margin does not license derivative division. Such boxes need branch-preserving algebraic exclusion, a different valid residual bound, subdivision without an unproved termination claim, or retention in the complete backend. Neither traversal relaxation nor bounded prior width removes this boundary obligation.

### Finite-cover stopping conditions and the conditional count

Suppose finitely many optical boxes cover the entire original optical prior, including its boundary. Classify each as soundly excluded, retained with an enclosure of **every** compatible system in it, or unresolved with its exact restricted predicate kept. Coverage and sound exclusion then preserve every branch. A Newton iterate, isolated exact root or successful fit does not enclose a positive-error compatible continuum.

A sufficient .001 stopping certificate consists of that complete cover, at least one verified physical compatible witness, and native ranges over **every surviving enclosure** with all eighteen diameters at most .002. The coordinate midpoint then guarantees the requested errors. Conservative ranges failing this test do not prove impossibility; two actual compatible witnesses separated by more than .002 do. Unresolved boxes prevent a completed precision claim unless their contributions have also been bounded or resolved exactly. Keeping exact restricted predicates preserves a complete implicit inverse; it is not yet a tractable explicit cell representation.

Each exclusion (X4) carries its valid error interval. Partition the error axis at certificate and local-validation endpoints as needed, and retain strict versus weak boundary directions. An exclusion proved below a threshold is not reused at or above it.

A conditional finite-cover estimate applies on a compact optical exterior \(G\). Assume uniform positive optical guards on a validated neighborhood, a certified Lipschitz bound \(L\) from the selected optical infinity norm to the observation infinity norm, uniformly over all \(q\in Q\), a common guard radius \(r_{\rm guard}>0\), and a **proved geometry-relaxed residual gap**
\[
\min_{q\in Q}\|\mathcal F(\xi,q)-y\|_\infty
 \ge\epsilon+\Gamma\quad(\xi\in G),\qquad\Gamma>0.
\tag{X7}
\]
Here \(\Gamma\) is an exterior residual gap, distinct from the regular-branch injectivity constant \(\gamma\). A dual center certificate within \(\Gamma/4\) of optimum excludes any certified box of radius
\(r\le\min(r_{\rm guard},\Gamma/(4L))\)
with positive slack, using \(+\infty\) for the second term if \(L=0\). For constructive selection of algebraic centers in \(G\), require an effective semialgebraic description over the encoded algebraic constants. Without that extra input hypothesis, interpret the covering count as a geometric existence bound only. A region of optical infinity-norm diameter \(D\) then has a covering-box count of order
\[
\left(1+\frac{D}{\min(r_{\rm guard},\Gamma/(4L))}\right)^{14}
\tag{X8}
\]
up to dimension-dependent factors.

The exponent **14 counts optical covering boxes conditionally; it is not a total-runtime bound**. It excludes the work of proving guards, derivative bounds, the exterior gap and near-optimal verified duals. A gap on the relaxed geometry box is stronger than physical uniqueness. Tighter common polytopes may help, but their resulting gaps need proof. At critical boundaries guard radii can vanish and derivatives diverge; gauges and remote near-aliases can make \(\Gamma=0\).

Section 2C now proves a conditional all-branch noise cover and a critical-boundary finite-exclusion theorem. The remaining work is to certify the complete anchor fiber, boundary/critical discriminant, entire projected noise cube and useful inverse margins, and to resolve regions where these conditions fail. No small number of roots or useful covering radius is established for an actual record. The derivative-based count above remains valid on its guarded subregions; the global forward Hölder modulus supplies a different boundary-safe route. [Remaining mathematical and implementation gaps](theory/remaining_constructive_gap.md) separates these obligations.


### Conditional data precision and uniform worst-case precision differ

For the full sampled-time physical domain \(\mathcal P\), define the native inverse modulus
\[
\Omega(r)=\sup\left\{
 \|\theta-\theta'\|_{\infty,\rm native}:
 \theta,\theta'\in\mathcal P,\ 
 |F_m(\theta)-F_m(\theta')|\le r\omega_m\ \forall m
 \right\}.
\tag{E3}
\]
For unrestricted point estimates and arbitrary real observed records, the deterministic uniform minimax risk equals
\[
\mathcal R(\epsilon)=\sup_{y:\mathcal C_\epsilon(y)\ne\varnothing}
 R_y(\epsilon)=\tfrac12\Omega(2\epsilon).
\tag{E4}
\]
Two systems in one compatible set have data separation at most \(2\epsilon\omega\). Conversely any such pair shares the midpoint record; each record's coordinate midpoint achieves its conditional bound (G6). This proves the formula using suprema without assuming extrema are attained.

Equation (E4) is an information bound, not a claim about algorithms restricted to rational records or physically compatible outputs. Such restrictions define different decision problems. Exact ambiguity implies \(\Omega(0)>0\). For example, a zero-wedge rotor has an arbitrary phase throughout the allowed 36-degree interval, and its speed can vary without altering its tilt. Therefore uniformly guaranteeing .001 on the entire original prior is impossible even at zero measurement error. A particular informative record can still have a much smaller conditional profile.

In fact, with the requested mixed native sup norm, the unrestricted uniform risk over this full prior is exactly **75 native units for every \(\epsilon\ge0\)**. Take zero wedges, an axial beam and \(b=0\). Every sample is \(y=0\) for all \(d\in[50,200]\), while strict internal and external traversal remain valid. Thus \(\Omega(0)\ge150\); no native prior coordinate has width greater than 150, so \(\Omega(r)\le150\). Equation (E4) gives 75. This is a worst-case statement across records, not the uncertainty of a particular informative record. Useful precision conclusions must therefore condition on the record or a justified excitation domain.

### Data-derived sufficient certificates

The remainder of this section gives conditional finite-error bounds that may be more useful computationally than full decomposition. They have not been successfully evaluated for an actual record here. Every condition may fail; failure leaves the associated prior region unresolved. The complete audited local framework is in [physical_inverse_certificates.md](theory/physical_inverse_certificates.md).

For fixed feature rows and a fixed certified geometry region, the formulas retain their full dependence on \(\epsilon\), wedge scale \(\kappa\), optical margins and conditioning. Feature boxes and retained domains generally change with \(\epsilon\); derivative bounds and Schur margins below are therefore functions of \(\epsilon\) unless one certificate controls a specified whole interval of error allowances. Constants cannot be carried across that interval solely because one fitted point is nonsingular.


### Rotor enclosure from all 200 observations

On a uniformly guarded prior region, require a certified first-order enclosure
\[
F_h(t_k)=B_h+\sum_i[
C_{hi}\cos(2\pi\nu_it_k)+D_{hi}\sin(2\pi\nu_it_k)]+R_{hk},
\qquad |R_{hk}|\le\tau_{hk},
\]
where \(h=x,y\) and the \(\nu_i\) are unknown positive frequency magnitudes. This uses general oblique ellipses and, under the stated uniform bounds, has a first-order remainder \(O(\kappa^2)\); it does not assume a known beam. Let
\(\delta(\epsilon,\kappa)=\max_{hk}(\epsilon\omega_{hk}+\tau_{hk}(\kappa))\), adding any separately bounded clock or evaluator discrepancy. The Hankel and root-contour conditions define their own validity range. Sending \(\epsilon\) to zero does not remove finite-angle remainder error.

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
r_a=\epsilon|P_f|\omega,\quad r_c=\epsilon|P_g|\omega,
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

### Explicit all-eighteen bounds as functions of measurement error

With component allowances \(\epsilon\omega\), hold the data-derived feature rows fixed and define
\[
b_f=|P_f|\omega,\qquad b_g=|P_g|\omega,
\qquad r_a(\epsilon)=\epsilon b_f,\quad r_c(\epsilon)=\epsilon b_g.
\]
For the full certified solution family at that allowance, write the uniform quantities as
\(\overline U_{ij}(\epsilon)\), \(\overline v_i(\epsilon)\),
\(\lambda_j(\epsilon)\), and \(\mu(\epsilon)>0\). Define, in the original native units,
\[
\boxed{
K_i(\epsilon)=
 \sum_j\overline U_{ij}(\epsilon)b_{f,j}
 +\frac{\overline v_i(\epsilon)}{\mu(\epsilon)}
  \left(b_g+\sum_j\lambda_j(\epsilon)b_{f,j}\right),
 \quad i=1,\ldots,18.}
\tag{E5}
\]
The exact feature-center solution obeys
\[
|\theta_i-\theta_{i,\rm center}|\le\epsilon K_i(\epsilon)
\quad\text{for every compatible system in that certified branch},
\]
and the branch's native diameter satisfies
\[
D_{i,\rm branch}(\epsilon)\le2\epsilon K_i(\epsilon).
\tag{E6}
\]
These inequalities include changes in every jointly unknown parameter through \(U,v,w\). They do not assign calibration parameters known values. Angular rows of the native chart derivative include the degree conversions, and speed rows retain hertz.

If all globally compatible systems lie in that one certified branch, the sufficient all-eighteen target condition is
\[
\epsilon K_i(\epsilon)\le .001\quad\text{for every }i,
\tag{E7}
\]
together with all strong-inverse, scalar-bracket, physical and residual conditions. A feature-center estimate need not pass every original residual, although it gives the stated unrestricted estimation bound. Requiring a compatible returned system uses the feasible-center criterion stated after (G8).

For a finite complete cover by branches \(b\), let certified intervals be
\([L_i^{(b)}(\epsilon),U_i^{(b)}(\epsilon)]\). Then use
\[
D_i(\epsilon)\le
 \max_b U_i^{(b)}(\epsilon)-\min_b L_i^{(b)}(\epsilon).
\tag{E8}
\]
Small within-branch diameters alone do not control separation between branches. An unresolved branch blocks a claimed full-prior precision certificate, unless exact global elimination proves it incompatible.

If all certificate conditions hold uniformly for \(0\le\epsilon\le\epsilon_*\), use uniform constants \(K_i^*\) on that whole range. Then
\(\epsilon\le\min(\epsilon_*,\min_i .001/K_i^*)\)
is a sufficient threshold, with a zero \(K_i^*\) imposing no coordinate restriction. If constants are available only pointwise or on separate ranges, keep the implicit conditions (E7). No universal numerical \(K_i\) or \(\epsilon_*\) follows from a nonzero determinant.

The geometric conditioning is explicit: \(J^{-1}\) controls strong inversion, \(w=g_zJ^{-1}\) couples strong error into the weak equation, and \(\overline v_i/\mu\) converts the weak scalar margin into native coordinate motion. The normalized pointwise weak margin is \(|s|/\|v\|_{\rm native}\). Beam direction, offsets, prism separations, index contrast, frequency separation and distances to optical guards enter these quantities and the validity of the chart. No dependence on one favorable scalar invariant alone certifies all eighteen coordinates.


### The witness derivative is not a recovery constant

At the oblique witness, the tangent with \(\psi'=1\) has a native mixed-coordinate sup norm tending to approximately \(981.7869465\), dominated by the workpiece-distance derivative. The normalized invariant derivative has magnitude approximately \(0.3778192691\) per radian of \(\psi\), or \(0.0003848281650\) per unit of that native normalized tangent. Also
\[
U_+\approx69.904973545+0.133333333i,\qquad
|U_+|^2\approx4886.7231041.
\]
Thus the corresponding self-second-harmonic directional gain is approximately
\(1.880548685\,\kappa^2\) per native sup unit, asymptotically.

These are rounded conversions of one audited proof tangent. They are not the smallest singular value, a full inverse Lipschitz constant, or an accuracy guarantee for all coordinates. The large tangent scale illustrates why \(v\) in (C8)-(C9) cannot be omitted. The native norm mixes degrees, refractive-index units and native lengths because the requested coordinate tolerance does. The full tangent, including its wedge-angle conversion, is preserved in [physical_oblique_inverse.md](theory/physical_oblique_inverse.md).

At this regular witness the native singular-direction hierarchy has five order-one directions, twelve order-\(\kappa\) directions and one order-\(\kappa^2\) direction, with chart-dependent constants. The five strong directions include three wedge-amplitude directions and two baseline combinations. Weak directions mix native parameters; this does not assign a separate accuracy power to each named coordinate.

Under uniform bounds on the normalized strong inverse, chart derivatives and Taylor remainders, the scalar margin can take the form \(s=\kappa^2s_2+O(\kappa^3)\). Twelve adapted modes then have error order \(\epsilon/\kappa\), and one has order \(\epsilon/\kappa^2\). A nonzero rational determinant or a wedge exponent alone does not supply the necessary constants.

A matching deterministic lower-bound power requires more than a bound on one feature. Along a first-order-preserving curve parameterized by a monotone native coordinate \(\theta_j=s\), certify
\[
\|D F(\theta(s))\theta'(s)\|_\infty\le C\kappa^2
\]
over **all 400 original real observations**, plus an available parameter interval of length \(r\). Endpoints then have native separation at least their scalar separation and data distance at most \(C\kappa^2|\Delta s|\). For a uniform scalar observation allowance \(\epsilon\), a midpoint-record argument gives minimax native error at least
\[
\frac12\min\left(r,\frac{2\epsilon}{C\kappa^2}\right).
\tag{C11}
\]
Alternatively retain a certified secant factor in the native endpoint separation. Unit tangent norm alone is insufficient because arc length does not lower-bound endpoint separation. This conditional lower bound matches powers with the Schur upper bound, not automatically numerical constants.

### Higher-order contamination remains part of the inverse

A finite record at nonzero wedges contains higher harmonics. First-order coefficient fitting, frequency estimation and second-harmonic demixing need a joint error analysis. Formal coefficients, torus coefficients and coefficients produced by a finite linear extractor are different objects. A fixed degree-two extractor may mix cubic terms into a quadratic row, leaving only an \(O(\kappa)\) normalized quadratic remainder unless stronger cancellation is proved.

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
\le\epsilon\omega+(1-\lambda)\tau\},\quad0\le\lambda\le1.
\tag{25}
\]
For a fixed certified box and \(\tau\), these sets are nested:
\(\mathcal C_{\lambda_2}\subseteq\mathcal C_{\lambda_1}\) for
\(\lambda_2\ge\lambda_1\). The triangle inequality proves this directly:
\[
|F_{\lambda_1}-y|
\le |F_{\lambda_2}-y|+(\lambda_2-\lambda_1)|R|.
\]
Every parameter compatible with the exact model at allowance \(\epsilon\) remains feasible at the same parameter vector throughout; \(\lambda=1\) recovers the original observation condition.

This supplies a sound hierarchy if the initial approximation gives a complete remainder-thickened cover. Nominal roots of a truncated model alone are not that cover. An unvalidated remainder box remains unresolved. Complex-analytic or real-path Taylor bounds require the relevant entire disk/path to remain on coherent guarded branches; endpoint transmission alone is insufficient. Direct interval bounds on \(F-\mathcal T\) are a possible looser fallback.

Exact geometry affinity (5) allows common four-variable LP profiling throughout a hierarchy that preserves affinity, with boxwise constant error bounds. Approximate duals are only proposals: independently checked support corrections and outward bounds must justify exclusion. The exact global construction in section 2A establishes coverage and termination in principle. A practical complete cover using this approximation hierarchy, and useful runtime bounds, remain unestablished.

## 7A. Complete frequency covers and the limits of whole-prior spectral accuracy

The [full-prior spectral addendum](theory/full_prior_spectral_cover.md) supplies a complete three-variable frequency outer cover without a rank or separation assumption. The [critical filtered-remainder addendum](theory/critical_filtered_remainder.md) proves a sharp limitation for the standard finite-order filters. These results concern the exact coupled-vector model. All eighteen parameters remain unknown; the explicit optical families below are proof witnesses.

### Endpoint algebra controls every fixed Taylor order

At zero wedges every glass and exit radicand has a uniform positive lower bound over the remaining native prior. Section 2C gives positive forward denominator bounds at every admitted endpoint, including critical limits. For \(A>0\), \(A+d\ge0\), and integer \(m\ge0\), the endpoint square-root inequality is
\[
\left|\sqrt{A+d}-\sum_{j=0}^m{1/2\choose j}A^{1/2-j}d^j\right|
\le c_m\frac{|d|^{m+1}}{A^{m+1/2}},
\qquad c_m=\frac{\binom{2m}{m}}{4^m}.
\tag{S1}
\]
For negative \(d\), all nonconstant coefficients of \(\sqrt{1-\lambda}\), \(\lambda=-d/A\in[0,1]\), have the same sign. Its divided tail is bounded by its value at one, giving \(c_m\). For nonnegative \(d\), the integral remainder gives a smaller coefficient. In particular
\[
\sqrt{A+d}-\sqrt A-\frac d{2\sqrt A}
=-\frac{d^2}{2\sqrt A(\sqrt{A+d}+\sqrt A)^2}.
\]
This remains finite at \(A+d=0\). Likewise, for \(A\ge A_*>0\) and \(A+d\ge d_*>0\),
\[
\frac1{A+d}=\sum_{j=0}^m\frac{(-d)^j}{A^{j+1}}
+\frac{(-d)^{m+1}}{A^{m+1}(A+d)}.
\tag{S2}
\]
Propagating homogeneous jets, magnitude bounds and remainders through the finite optical circuit therefore gives, for each fixed \(m\), a constructively bounded finite \(C_m\) such that
\[
\|F-F^{[m]}\|_\infty\le C_m a^{m+1},
\qquad a=\max_j\|u_j\|_2\le a_*=\tan18^\circ.
\tag{S3}
\]
The constants cover the original nonfrequency prior and the weak optical extension. They require no physical intermediate path from zero wedges to the endpoint. This endpoint argument differs from the path-based derivative approach discussed in section 7. It does not assert that the constants are moderate or that the bounds decrease as \(m\) increases. Sine-coordinate jets satisfy the same type of theorem because their normal factors are bounded below by \(\cos18^\circ\); coefficients through degree two agree after the appropriate sine-to-slope coordinate change.

### An explicit collision-safe degree-six speed cover

Let \(\tau_1\) bound the first-order absolute tail on the entire region being considered; for a whole-prior claim its validity must cover that prior. In the bijective speed chart
\[
v_i=\tan(\pi N_i/20),\qquad z_i=\frac{1+iv_i}{1-iv_i},
\]
define
\[
Q_v(T)=(T-1)\prod_{i=1}^3
\left[(1+v_i^2)T^2-2(1-v_i^2)T+(1+v_i^2)\right]
=\sum_{j=0}^7Q_j(v)T^j.
\tag{S4}
\]
This annihilates both real first-order channels, including zero amplitudes, repeated nodes and zero speeds. Every coefficient has degree at most six in \(v\). Since \(|v_i|<1\), the coefficients alternate in sign and
\[
\sum_{j=0}^7|Q_j(v)|=-Q_v(-1)=128.
\]
For the original record put
\[
S_y(v)=\frac1{128}\max_{\substack{c=x,y\\0\le k\le192}}
\left|\sum_{j=0}^7Q_j(v)y_{c,k+j}\right|.
\]
The annihilation identity and the triangle inequality imply
\[
r_y(\theta)=\|F(\theta)-y\|_\infty
\ge S_y(v(\theta))-\tau_1.
\tag{S5}
\]
Thus for any symbolic allowance \(\epsilon\ge0\) and chosen exclusion gap \(\gamma>0\),
\[
\Omega^{\mathrm{spec}}_{\epsilon,\gamma}
=\{v\text{ in the complete speed chart}:S_y(v)\le\epsilon+\tau_1+\gamma\}
\tag{S6}
\]
contains every compatible speed triple. Outside it every physical system has residual strictly greater than \(\epsilon+\gamma\). Besides the speed bounds, this is exactly **772 scalar polynomial inequalities of degree at most six in three variables**: two channels, 193 windows and two signs. Algebraically encoded inputs admit a finite exact decomposition. The cover retains all speed signs and prism assignments because (S4) is symmetric in \(v_i^2\). It does not by itself localize the other fifteen parameters.

Repeated factors can be removed on exact collision strata. The 15 set partitions of \(\{0,1,2,3\}\) represent zero speeds in the block containing 0 and equal nonzero speed squares in the other blocks. For \(q\) nonzero blocks, keep one quadratic factor per block and the constant factor. The degree is \(L=2q+1\), coefficient norm \(2^L\), and all windows are \(k=0,\ldots,K-L-1\). Their union remains complete. On the all-zero-speed stratum the exact trace is constant and the filtered tail is zero; a consecutive observed difference greater than \(2\epsilon\) excludes that stratum.

### Finite absolute tails need not produce useful localization

A single algebraic critical-limit witness inside the original bounds quantifies the problem. It uses \(t_x=2/5,t_y=0\), all indices \(9/5\), zero source, \(g=2,d=200\), zero speeds/phases and first slope \(u_1=8/25\). The first prism transmits strictly; choose the second and third at their coplanar critical slope
\[
u_c(q,z)=\frac{z^2}{q\sqrt{n^2-q^2}+\sqrt{n^2-1}}.
\]
The limiting root zeros are nonphysical, but reducing each of the last two slopes relative to its updated critical value gives a strictly physical approaching family. Exact rational intervals in the supplied [witness source](theory/proof_checks/full_prior_tail_witness.py) enclose
\[
u_2\in(.185,.187),\quad u_3\in(.011,.012),\quad
F_x\in(17821,17822),
\]
with positive traversal and external-flight margins. First-order slope and sine approximations lie in \((196,197)\) and \((192,193)\), respectively. Consequently every sound whole-prior first-order absolute tail satisfies
\[
\tau_1>17600,\qquad C_1>171875
\quad\text{for the slope bound }C_1a^2.
\tag{S7}
\]
The degree-two, -three and -four slope Taylor values lie in \((240,241)\), \((278,279)\), \((308,309)\); the same audit verifies \(\tau_m>17000\) for \(m=1,2,3,4\) in both slope and sine coordinates. These are exact witness bounds, not numerical sweeps.

A normalized filter score never exceeds \(Y=\|y\|_\infty\). Hence when \(Y\le\epsilon+\tau_m\), the corresponding global-absolute-tail speed cover is the entire speed prior. In particular a sound whole-prior first-order implementation cannot localize speeds for \(Y\le17600\). The seven-column Hankel sufficient condition also fails when its total contamination allowance \(\delta\ge Y\): a left inverse of norm \(\gamma_H\) obeys \(1\le7\gamma_HY\), so \(7\gamma_H\delta<1\) is impossible. For \(Y=0\) the required rank already fails. These statements concern those certificates; parameter-dependent tails, exact profiling and filtered remainders remain possible.

### Direct filtered bounds and the sharp critical exponent

Let \(\mathcal A_v\) be a real annihilator of the degree-\(m\) formal Taylor trace, normalized to coefficient \(\ell_1\) norm one, of degree \(L\), with the constant node included. Its coefficients sum to zero, so their positive and negative masses are each one half. Let \(\omega_{\mathrm{inst}}(\delta)\) be a certified instantaneous forward modulus for fixed nonwedge parameters and distance \(\max_i\|u_i-u_i'\|_2\le\delta\). The square-root circuit gives a finite bound \(C\delta^{1/8}\). With
\[
d_L(v)=a_*\max_{\substack{1\le i\le3\\1\le r\le L}}|z_i^r-1|,
\]
the two convex averages of the screen points in a window give
\[
\|\mathcal A_v(F-F^{[m]})\|_\infty
=\|\mathcal A_vF\|_\infty
\le\frac12\omega_{\mathrm{inst}}(d_L(v)).
\tag{S8}
\]
For first order, \(b(v)=\min\{\tau_1,\omega_{\mathrm{inst}}(d_L(v))/2\}\) can replace \(\tau_1\) in (S5). Unlike a global absolute tail, it vanishes at zero speeds. This modulus uses an instantaneous vector-slope norm; the constants of the original chart modulus \(\omega_{\mathrm{fwd}}\) in section 2C cannot be reused without the appropriate norm/coordinate conversion.

For degree \(m\), the standard full-lattice annihilator uses every \(\ell\in\mathbb Z^3\) with \(|\ell|_1\le m\). Its degree is
\[
L_m=\frac{4m^3+6m^2+8m+3}{3}
=7,25,63,129,231\quad(m=1,\ldots,5).
\tag{S9}
\]
Thus only \(m=1,2,3,4\) give full annihilation windows in 200 samples. This observation does not rule out other higher-order nonlinear constructions.

The exponent \(1/8\) in (S8) is sharp uniformly for these standard filters. The second witness uses \(t_x=t_y=23/50\), all indices \(9/5\), \(g=2,d=200\), zero source and first wedge slope magnitude \(8/25\). Choose its first phase \(\phi_c\) algebraically to make the first exit critical. At later stages the correct three-dimensional critical slope is
\[
a_j=\frac{z_{j-1}^2}
{H_jq_{j-1,x}+\sqrt{n^2-1}\sqrt{q_{j-1,x}^2+z_{j-1}^2}},
\tag{S10}
\]
since the y direction is nonzero. The [critical witness source](theory/proof_checks/critical_filtered_tail_witness.py) verifies \(a_2\in(.032,.033)\), \(a_3\in(.00030,.00032)\), strictly positive noncritical guards and flight margins, and a finite screen limit with x in \((645433,645434)\), y in \((343242,343243)\).

Decreasing the first phase by \(s>0\) makes the exit radicands satisfy
\[
E_1^2=A_1s+O(s^2),\qquad
E_2^2=A_2E_1+O(s),\qquad
E_3^2=A_3E_2+O(E_1),
\]
with all three \(A_j>0\). The screen derivative with respect to \(E_3\) is strictly negative at the limit, giving
\[
F_x(s)=F_x(0)-C_*s^{1/8}+O(s^{1/4}),\qquad C_*>0.
\tag{S11}
\]
The limiting configuration is excluded; the nearby positive-\(s\) systems are physical. On the original clock take per-sample phase increments \((-h,h^2,h^3)\) and initial phases \((\phi_c-h,0,0)\), so \(N=(10/\pi)(-h,h^2,h^3)\). Uniformly over the finite 200 samples,
\[
F_{h,x,k}=F_x(0)-C_*h^{1/8}(k+1)^{1/8}+O(h^{1/4}).
\]
All speeds and all nodes in each fixed harmonic lattice are distinct and nonaliased for sufficiently small \(h>0\). The alternative rational speed chart \(v=(-r,r^2,r^3)\), with first phase \(\phi_c-2\arctan r\), supplies finitely algebraically encoded records with the same exponent.

For fixed \(m\), the annihilator tends to \((T-1)^{L_m}\), with coefficient norm tending to \(2^{L_m}\). Its first window therefore has leading term
\[
-\frac{C_*}{2^{L_m}}h^{1/8}
\left.\Delta^{L_m}x^{1/8}\right|_{x=1}.
\tag{S12}
\]
The finite difference is nonzero: its repeated integral has integrand
\(\alpha(\alpha-1)\cdots(\alpha-L_m+1)(1+\sum s_j)^{\alpha-L_m}\), \(\alpha=1/8\), of constant nonzero sign. This lower bound and (S8) prove
\[
\|\mathcal A_{m,h}(F_h-F_h^{[m]})\|_\infty=\Theta(h^{1/8})
\quad(m=1,2,3,4).
\tag{S13}
\]
Taking the minimum of four positive lower constants also covers adaptive selection among those degrees. The cutoffs and constants are unevaluated; this is neither a practical noise threshold nor a claim that an actual record lies near this family. Arbitrary data-dependent filters, exact nonlinear inverses and validated exclusion of critical regions are outside this obstruction.

For useful all-eighteen localization, every retained spectral/coefficient region still needs a certified cover of all its optical solutions, including singular fibers and original bounds, or an exact residual exclusion. Uncertain formal coefficients remain regions; fitted points do not substitute for them. The frequency cover alone, or retaining the whole unresolved prior as one candidate, supplies no useful eighteen-coordinate precision claim.

## 7B. Data-only critical-normal certificates and their limits

The [data-dependent critical-margin addendum](theory/data_dependent_critical_margin.md) gives a direct last-prism guard and conditional upstream tests. These are symbolic theorems; no identified full-vector original-protocol record is currently available here to instantiate them. The canonical record in section 9 is not used. In this subsection write \(\varrho_j=R_j/Z_j=1-u_j\cdot v_j\), \(v_j=X_{\mathrm{out},j}/Z_j\), for the normal ratio; the source memo locally calls this ratio \(\kappa_j\), distinct from this report's wedge scale.

### A last-prism margin obtained directly from the observations

Let \(p_{\mathrm{exit}}\) and \(p_{\mathrm{next}}\) be consecutive exit and next-flat positions, and let \(\ell=g,g,d\) by stage. Weak traversal and the exact outgoing intersection equation give
\[
B=\ell-u\cdot p_{\mathrm{exit}},\qquad
0\le B\le3+\ell,\qquad
\ell-u\cdot p_{\mathrm{next}}=\varrho B.
\tag{DC1}
\]
The upper bound follows from nonnegative internal axial travel \(3+u\cdot p_{\mathrm{exit}}\). At \(R=0\), (DC1) gives the critical hyperplane \(u\cdot p_{\mathrm{next}}=\ell\), without inverting the singular source transport.

For sample \(k\), define
\[
A_{k,c}(\epsilon)=|y_{k,c}|+\epsilon,\qquad
M_k(\epsilon)=u_*\sqrt{A_{k,x}^2+A_{k,y}^2},\qquad
u_*=\tan(\pi/10).
\]
Every compatible weak or strict system has
\[
\varrho_3(k)\ge
\max\left\{0,\frac{50-M_k(\epsilon)}{53}\right\}.
\tag{DC2}
\]
Indeed \(\varrho(3+d)\ge d-u\cdot F_k\ge d-M_k\); \((d-M_k)/(3+d)\) increases with \(d\ge50\), since \(M_k\ge0\). All eighteen parameters remain unknown. The constants 50 and 53 come from the native distance lower bound and the three-unit prism thickness.

If \(m_3(\epsilon)=\min_k[50-M_k(\epsilon)]/53>0\), then
\[
R_3(k)\ge z_3^*m_3(\epsilon)>0,\qquad
\det A_3(k)\ge\frac{h_*}{h_*+u_*}m_3(\epsilon),
\quad h_*=\sqrt{(13/10)^2-1}.
\tag{DC3}
\]
Here \(z_3^*\) is the uniform unit axial bound in section 2C, and \(\det A_3=H_3R_3/(Z_3P_3)\). These guards permit bounded backward transport through the final prism; they do not certify prisms 1 or 2.

The native phase/speed prior can sharpen the first two samples. Set \(\alpha_0=\pi/10,\ \alpha_1=9\pi/20\). For \(A=A_{k,x}, B=A_{k,y}\), the exact signed-sector support is
\[
M_k^{\mathrm{sharp}}=
u_*\begin{cases}
\sqrt{A^2+B^2},&B\le A\tan\alpha_k,\\
A\cos\alpha_k+B\sin\alpha_k,&B>A\tan\alpha_k.
\end{cases}
\]
For \(k\ge2\), the allowed signed sectors cover every direction, and \(M_k\) is exact for this instantaneous relaxation. The simpler disk bound is used in the following closed-form noise threshold.

Fix \(0\le\mu<50/53\), put \(C_\mu=50-53\mu>0\), \(a=|y_{k,x}|\), \(b=|y_{k,y}|\), and \(S=C_\mu/u_*\). If \(a^2+b^2<S^2\), then every
\[
0\le\epsilon<\epsilon_{k,\mu}
=\frac{\sqrt{2S^2-(a-b)^2}-(a+b)}2
\tag{DC4}
\]
excludes the closed near-critical set \(R_3(k)\le\mu Z_3(k)\). Taking the minimum of these positive thresholds over all 200 samples gives a uniform ratio guard. If the strict base-radius test fails at a sample, this particular all-sample certificate has no nonnegative noise interval. Equality at the endpoint does not supply strict exclusion.

The certificate has explicit polynomial identities. Under \(R\le\mu Z\), (DC1) implies
\[
Z(u\cdot F-C_\mu)
=(1-\mu)Z(d-50)+\mu Z(3+d-B)+(\mu Z-R)B\ge0.
\tag{DC5}
\]
Thus \(u\cdot F\ge C_\mu>0\). With \(A_c=|y_c|+\epsilon\), the identity
\[
\begin{aligned}
u_*^2(A_x^2+A_y^2)-C_\mu^2
={}&u_*^2\sum_{c=x,y}(A_c-F_c)(A_c+F_c)\\
&+(u_*^2-|u|^2)|F|^2+(u_xF_y-u_yF_x)^2\\
&+(u\cdot F-C_\mu)(u\cdot F+C_\mu)
\end{aligned}
\tag{DC6}
\]
has only nonnegative terms, contradicting the strict squared-radius test. The sign \(C_\mu>0\) is essential before squaring.

### Conditional upstream tests: sixteen corners per sample

Suppose downstream positive ratio guards have been certified throughout an optical set \(D\) containing every compatible system. Backward transport through those downstream prisms gives
\[
F=T_{\mathrm{down}}p_j+c_{\mathrm{down}}(w),
\qquad w=(g,d)\in[2,15]\times[50,200].
\]
Its inverse is defined by the guards; source offset is absent from the critical-plane row. For target \(\mu\ge0\), it suffices to prove, for every optical point \(x\in D\), every distance-box vertex \(w_v\), and every sign corner \(\sigma\in\{\pm1\}^2\),
\[
\ell_j(w_v)-u_j\cdot p_j(x,w_v,y_k+\epsilon\sigma)
>\mu[3+\ell_j(w_v)].
\tag{DC7}
\]
There are sixteen inequalities for each stage/sample. Affine interpolation in geometry/error corners and (DC1) imply \(\varrho_j>\mu\) throughout the compatible set. Nonstrict inequalities with a specified positive \(\mu\) also give a nonzero guard. The denominator in the associated linear-fractional form is positive, so its distance-box minimum is attained at a vertex. Certifying stage 2 needs stage 3; stage 1 needs both downstream stages. No inverse of the current possibly critical prism is used.

For local ray-graph variables, the exact reverse step is
\[
p_{\mathrm{prev}}=
\frac{Kp_{\mathrm{next}}-\ell V-3X_{\mathrm{in}}R}{HR},
\quad
K=HR I-(ZX_{\mathrm{in}}-HX_{\mathrm{out}})u^T,\quad
V=HX_{\mathrm{out}}-X_{\mathrm{in}}(u\cdot X_{\mathrm{out}}).
\tag{DC8}
\]
Positive downstream denominator clearing gives polynomial degrees at most five for stage 2 and eight for stage 1. These bounds refer to independent local ray-graph variables with all their physical equalities/guards, not dense elimination into eighteen native coordinates. Explicit sums-of-squares/product identities and equality multiples can certify the universal inequalities by coefficient comparison and signs. No small search degree or easy construction is guaranteed. An optical superset may safely drop cross-time constraints only while retaining every compatible trace, potentially making the result too conservative.

### Exclude, retain or leave unresolved

A verified weak full-record witness \(\theta_0\in W\) with some \(R_j(k)=0\) and residual \(r_0\) has a different consequence: for every \(\epsilon>r_0\) and every \(\rho>0\), the established strictification supplies actual physical systems with residual below \(\epsilon\) and \(R_j(k)<\rho\). Thus no uniform positive normal margin exists at that noise budget. At \(\epsilon=r_0\), strict endpoint feasibility requires a separate check.

A single strict physical witness below one specified near-critical threshold retains that target set only. It does not by itself disprove every positive margin; the stronger conclusion requires the weak critical witness and noise slack, or another verified sequence with normals tending to zero. The audited critical examples show why no positive-margin procedure can succeed for every possible record. They do not establish the status of an arbitrary record.

Accordingly, a record-specific result must either certify exclusion using (DC2)/(DC4) or upstream signs, certify retention with the stated full-record witness, or remain unresolved. A failed exposure-inequality search proves neither compatibility nor inconsistency. The optical question of intersection with the compact critical-boundary image remains substantive.

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
along this gauge, where \(g_s\) denotes a gauge path, not the air gap. With wedges \(O(\kappa)\), source-transport matrices are \(I+O(\kappa^2)\). On a compact guarded gauge tube,
\[
\|F(g_s,b)-F(g_0,b)\|
\le C_G|b|\kappa^2|s|.
\tag{27}
\]
This creates an indistinguishability scale proportional to
\(\epsilon/(|b|\kappa^2)\), capped by the available gauge segment. A numerical minimax statement needs a specified norm, gauge parameterization and constant \(C_G\).

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

## 8A. Exact finite-record collision barrier away from optical boundaries

The [finite-record collision addendum](theory/finite_record_collision_barrier.md) gives a rigorous robustness obstruction for a restricted admissible family. It does **not** assert that a particular supplied record has this ambiguity. Measurement allowance is \(\epsilon\) here (the source memo calls it \(\eta\)); \(A\) is a screen-length amplitude normalization, not error, wedge angle or measured signal norm.

If two physical systems satisfy
\[
\|F(\theta^+)-F(\theta^-)\|_\infty\le2\epsilon,
\]
their midpoint record is compatible with both and every speed estimator has worst-case native error at least
\[
\tfrac12\|N(\theta^+)-N(\theta^-)\|_\infty.
\tag{V1}
\]
This already obstructs the same target in all eighteen coordinates. A given \(y\) is affected only if it belongs to both observation boxes. No stochastic averaging or square-root-of-200 noise improvement enters this deterministic argument.

### Exact optical jet correction, not an uncharged linear approximation

To construct a lower-bound subclass, set
\[
n_1=n_2=n_3=3/2,\quad g=5,\quad d=100,\quad
b=(1,0),\quad\beta_x=\beta_y=0.
\]
All remain unknown in the original inverse problem and in its full sampled Jacobian. The nonzero offset removes the centered excitation assumption. For complex sine-vector wedges \(z_j\), the exact physical screen map has leading gains
\[
Z(z_1,z_2,z_3)=1+57z_1+(107/2)z_2+50z_3+O(\|z\|^2).
\]
The exact nonlinear map, including source-dependent quadratic terms, is used for the correction below.

Choose six interlaced rational nodes
\[
r_l=l+\tfrac12+\frac{l^2}{1000},\quad l=0,\ldots,5,
\qquad r=(.5,1.501,2.504,3.509,4.516,5.525).
\]
The positive system uses indices \(0,2,4\) in physical prism order; the negative system uses \(1,3,5\). Speeds are \(hr_l\), \(0<h\le.003\) Hz. Integer-relation proofs in the appendix show that each triple has 25 distinct degree-two sampled nodes, and the six repeated-fundamental columns give the required 31-column confluent design on the original clock. Distinctness is exact; it is not a favorable numerical spectral test.

Let
\[
D=\prod_{j=1}^5(r_j-r_0),\qquad
w_l=\frac{D}{|\prod_{m\ne l}(r_l-r_m)|},\qquad
B_0=D/5!<1.02 .
\]
Then \(\sum_l(-1)^lw_lr_l^m=0\) for \(m=0,\ldots,4\).
With \(T=t_c=199/40\), \(r_c=(r_0+r_5)/2\),
\(\chi=2\pi h r_ct_c\) and \(s=2\pi h(t-t_c)\), use
\[
z_j^\pm(t)=\frac{A}{K_j}c_l e^{i\chi}e^{ir_ls},
\qquad (K_1,K_2,K_3)=(57,107/2,50).
\tag{V2}
\]
Here \(l=2j-2\) or \(2j-1\). Let \(G_\pm(A,c;s)=Z(z_1^\pm(s),z_2^\pm(s),z_3^\pm(s))\) be the exact complex screen outputs for the two systems. Hold \(c_0=1\) and impose the five normalized complex jet equations
\[
J_m=i^{-m}e^{-i\chi}\left.\partial_s^m
\left[(G_+-G_-)/A\right]\right|_{s=0}=0,\qquad m=0,\ldots,4.
\]
These are equivalent to vanishing of the raw jets because their row factors are nonzero. At \(A=0,c=w\), the real derivative of these normalized equations in the ten real components of \(c_1,\ldots,c_5\) is the realification of
\[
M_{m,l}=(-1)^lr_l^m,\qquad m=0,\ldots,4,\quad l=1,\ldots,5,
\]
an invertible Vandermonde matrix with column signs. The exact implicit-function construction gives
\(c_l(A)=w_l+O(A)\).
Complex correction adjusts both wedges and phases; fixing their phases would leave insufficient real degrees of freedom for this argument.

The native parameters are
\[
\sin a_j^\pm=A|c_l|/K_j,\qquad
\phi_j^\pm=\arg c_l+2\pi h(r_c-r_l)t_c,
\tag{V3}
\]
with phases converted to degrees. A certified correction box of radius \(1/20\), a preliminary \(A\le1/100\), and finite exact derivative bounds \(C_0,C_1\) give
\[
A_0=\min(1/100,\ 1/(2C_1),\ (1/20)/(2C_0)).
\]
Zero denominators remove the corresponding restriction. The appendix states the exact contraction, derivative and compact optimization predicates. It also proves strict transmission/traversal margins throughout this amplitude box and phases strictly inside the original \(\pm18^\circ\) bounds. These are analytic guard/contraction bounds, not executed optical experiments. The constants and a useful numerical \(A_0\) remain unevaluated.

### Fifth-order separation of the complete nonlinear records

The corrected exact difference has zero slow-time derivatives through order four. A divided-difference integral over the five-simplex bounds its linear-amplitude limit by \(B_0|s|^5\). A finite certified bound \(C\) on the total derivative
\(\partial_A\partial_s^5\), including \(c'(A)\), gives
\[
\|F(\theta^+(A))-F(\theta^-(A))\|_\infty
 \le A(B_0+AC)(2\pi hT)^5.
\tag{V4}
\]
There is no additive \(O(A^2)\) optical-tail floor: the nonlinear correction term has the same fifth power. The bound holds throughout the entire 9.95-second observation interval, hence for all 400 original real coordinates.

Let \(A_1=\min(A_0,(2-B_0)/C)\), omitting the second restriction if \(C=0\). For every fixed \(h>0\), the existing off-axis scaled-rank proof and the exact node conditions give \(A_{\rm rank}(h)>0\) on which both **original full18 Jacobians** have rank eighteen. The \(O(A)\) amplitude/phase correction preserves the nonzero limiting scaled minor. Thus with
\[
A_\star(h)=\min(A_1,A_{\rm rank}(h)),
\]
every \(0<A\le A_\star(h)\) has record separation at most \(2A(2\pi hT)^5\). Neither \(A_\star\), the rank cutoff nor the nonlinear constants have been numerically evaluated. No practical-wedge threshold is implied by their positivity.

### Explicit native-speed obstruction and a fixed-amplitude fifth-root law

At \(h=3/1000\) Hz, the triples are
\[
N^+=(.001500,.007512,.013548),\qquad
N^-=(.004503,.010527,.016575)\quad\text{Hz}.
\]
The largest difference is .003027 Hz. Since
\[
(2\pi hT)^5=(597\pi/20000)^5<73/10^7,
\]
every
\[
0<A\le A_\star(3/1000),\qquad
\epsilon\ge(73/10^7)A
\tag{V5}
\]
admits the ambiguous midpoint record and forces speed risk at least **.0015135 Hz**, exceeding the requested .001. This is an explicitly scaled conditional construction; its admissible amplitude has not been established numerically for a practical device or supplied dataset.

There is also a fixed-positive-amplitude family. Uniform contraction/remainder bounds over a compact \(h\)-interval permit one positive \(A\). Choose it small enough for both systems to have rank eighteen at some \(h_0>0\). A selected nonzero minor for each system is analytic in \(h\), so its zero at \(h=0\), if any, has finite order. Hence both systems have rank eighteen throughout some punctured interval \(0<h\le h_1\), at that one fixed \(A\).

Over this restricted two-branch family,
\[
\operatorname{risk}_{N}(\epsilon)\ge
 \frac{1009}{2000}
 \min\left(h_1,
   \left[\frac{\epsilon}{A(2\pi T)^5}\right]^{1/5}\right).
\tag{V6}
\]
All wedges remain bounded away from zero by a positive multiple of \(A\), the offset is one, and every individual system has full18 rank. The limiting zero-speed configuration is singular; no uniform inverse-Lipschitz or singular-value bound is claimed on the punctured family. The positive \(A,h_1\) remain unevaluated.

The full prior already has the stronger constant worst-case floor described in section 7. This construction matters because the fifth-root barrier persists in a subclass excluding zero wedges, centered illumination and singular individual systems, far from critical optical boundaries. It is a separate finite-time spectral obstruction. Adding timestamps inside the same time interval does not remove it, since (V4) controls the whole interval; a longer observation window changes the bound.

The exponent five and divided-difference cancellation belong to established three-source superresolution. The additional assertions here concern exact nonlinear optical jet correction, original-prior realization and full18 regularity. No first-ever novelty or practical upper recovery guarantee is claimed. [Batenkov, Goldman and Yomdin, Super-resolution of near-colliding point sources](https://arxiv.org/abs/1904.09186)


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

Previous reports are retained as [REPORT_v6_archive.md](REPORT_v6_archive.md), [REPORT_v5_archive.md](REPORT_v5_archive.md), [REPORT_v4_archive.md](REPORT_v4_archive.md) and [REPORT_v3_archive.md](REPORT_v3_archive.md). Historical solver counts, stored-record ambiguity witnesses and floating-evaluation audits are in [EVIDENCE_REPORT.md](EVIDENCE_REPORT.md). They concern their stated canonical model and timestamp contract; they are not numerical validation of physical vector optics.

### Verified provenance of the available original-protocol record

The read-only [record provenance audit](RECORD_PROVENANCE.md) identifies an existing 200-pair record in `full18_research/observations.json`, `cases[0]`, ID `random_00` (line 266). Its shared timestamp array has 200 entries generated as binary64 \(k\cdot0.05\), ending at `9.950000000000001`; its intended clock is \(k/20\). The `observed` array is \(200\times2\), ordered `[x,y]`. The observation-file SHA256 is `fc61cf3e7181c78c206370e0fe022246ed538bbec857923a1354ae1cc5417d74`.

This record is **synthetic canonical independent-axis data**. Companion `cases.json:265` states that model explicitly; `contract.py:38` calls `vec2pat`, Dropbox `risley_lattice/model.py:49–58` bridges to `reverse_problem_v2/core.py`, and its lines 200–203 trace x and y independently. The saved generation hashes match those current Dropbox source files. The provenance note records their complete paths and hashes.

No full-vector or hardware record with the original 200-pair protocol was verified in the inspected locations. The 200-row Dropbox `hardware_box.csv` is explicitly synthetic two-prism data. Archived full18 `162450.inputs.npz` contains 1,600 pairs over 80 seconds, and the vector-layer-stripping JSON contains symbolic results rather than an observation record. This is a report on inspected evidence, not an exhaustive claim that no other record exists.

The canonical record is not full-vector ground truth, and this report does not instantiate the new physical data certificate with it. The theorem itself accepts any fixed array and makes conditional statements about compatible full-vector systems; it supplies neither a full-vector generating truth nor nonemptiness of that compatible set. A record-specific full-vector study first needs an identified appropriate record and a precise clock/error contract; the exact \(k/20\) proofs do not silently convert saved binary64 timestamps into exact rationals. No record was generated, altered or run through a solver during this provenance check.

## 10. Evidence, provenance and remaining publication work

| Claim | Evidence status | Exact scope |
|---|---|---|
| Physical paired transfer, polynomial graphs and four-variable geometry reduction | Derived exactly | Coupled-vector equations with explicit strict branch/traversal conditions |
| Exact full-prior inverse and complete ambiguity representation | Constructive theorem, independently reviewed | Finite exact encodings, ideal sample clock, sampled-time admissibility; complete cells, including singular and continuous fibers |
| Five-root compiler with six-root baseline retained | Exact bijective index change, combined recurrence and sign-degree bounds audited | Coupled prior, nonlinear native-index recovery; degree896 is not a practical runtime claim |
| Source transport and mixed-strict geometry elimination | Positive determinant and full circuit/threshold theorem | Exact-data Helly3; noisy shared-anchor fiber remains four-dimensional/Helly5; every rank stratum and endpoint flag retained |
| Exact physical compactification and boundary modulus | Uniform denominators and hierarchical inward strictification audited | Weak model equals actual closure; boundary values remain nonphysical; safe global forward exponent1/8 |
| Conditional all-branch continuation/noise cover | Proper finite-sheet argument and inverse integration | Complete anchor roots, full projected cube avoiding discriminant and certified eta required; all400 constraints filtered |
| Complete spectral speed cover and endpoint tails | Explicit normalized annihilator, endpoint algebra and exact witness audits | 772 degree-six constraints retain all collisions; global absolute-tail vacuity and sharp filtered exponent apply with their stated scope |
| Data-dependent critical margins | Direct physical identity and polynomial/product certificates independently audited | Data-only final-prism ratio/noise cutoff; upstream sixteen-corner signs remain conditional; no record-specific instantiation |
| Bounded Lean support | Eleven declarations compile successfully, independently inspected | Algebra/sign/square-root lemmas only; standard axioms, no admitted goals; full optical/global theorems not formalized |
| Available original-protocol observation record | Read-only schema, clock and source-hash provenance audit | 200-pair canonical independent-axis simulation, not identified full-vector or hardware data |
| Exact finite-record collision obstruction | Nonlinear jet correction and analytic-minor argument audited | Nonzero wedges, off-axis full18-regular systems; fifth-root restricted-family risk; amplitude cutoffs/constants unevaluated |
| Fixed-dimension complexity | Effective theoretical bound with input/height qualifications | Polynomial sample-count dependence with prohibitive constants and exponents; no practical K=200 bound |
| Symbolic-error compatible-set and exact thresholds | Constructive parametric theorem, independently audited | Consistency/unacceptable intervals, endpoint inclusion and nonemptiness retained; .001 iff all native diameters at most .002 |
| Singular endpoint classification | Convergent algebraic Puiseux-envelope argument | Fractional powers, nonzero floors and jumps; actual endpoint fibers tested separately |
| Sharp regular native sensitivity and matching finite bounds | LP duality and two-sided Taylor/polytope bounds | All18 native coefficients; certified closed ball, explicit error range; global only with positive exterior gap |
| Profiled optical-box exclusions and finite-cover stopping | Exact LP-dual, support-correction and Farkas certificates | Guarded boxes and valid common geometry outer sets; exponent14 counts boxes under additional gap assumptions, not total runtime |
| First-order factorization, orientation bound and trial-beam curve | Derived constructively | Formal first-order data; exceptional charts retained |
| First-order rank seventeen at (16) | Exact rational determinant independently reproduced | One proof witness, not a statistical survey |
| Second harmonic breaks the remaining fiber | Exact expansion and both derivative components independently reproduced | Formal quadratic invariant and local fiber |
| Last-prism formal candidate generation | Four-dimensional determinant and coefficient identities independently reproduced | Generic finite exact fibers with denominator exclusions; noisy and exceptional families retained |
| Exact finite-200 physical rank eighteen | Analytic scaled-Jacobian proof with confluent finite dictionary | Sufficiently small nonzero wedges near the witness |
| Native tangent values and epsilon-dependent local bounds | Audited witness conversion and conditional uniform certificates | Native units and all18 row factors retained; no successful useful finite-error evaluation for an actual record |
| Source reproduction package | Ten supplied sources saved with exact matching SHA256 hashes; three new sources pass syntax parsing | Previously audited proof sources retained; no new Python mathematical execution during packaging |
| Practical global recovery, experimental validation and external peer review | Not established | Distinct from exact completeness in principle |

The [proof-check reproduction package](theory/proof_checks/README.md) contains ten supplied sources: three physical construction scripts, their three independent audit counterparts, a separately scoped canonical witness audit, and the three new full-prior-tail, critical-filtered-tail and reduced-inverse audits. All ten saved sources match the supplied SHA256 hashes in the [manifest](theory/proof_checks/manifest.json). The three newly supplied files also passed AST-only syntax parsing under existing WSL Python 3.12.3, as recorded in [syntax_verification.json](theory/proof_checks/syntax_verification.json). They were neither imported nor executed; their mathematical successes are reported from the supplied independent audits, not a new local run. The critical-margin memo separately reports a targeted symbolic check of its identities; that check's script was not supplied as an additional reproduction source. Verbatim scripts retain their local notation. These symbolic audits are distinct from the bounded Lean verification below and from external peer review.

### Bounded Lean verification: eleven supporting results

The isolated [Lean package](formalization/README.md), [source](formalization/RisleySupport.lean), [successful compiler log](formalization/compiler-output.txt), [machine-readable verification](formalization/verification.json) and [independent audit](formalization/independent_audit.md) provide a separate level of evidence. Installed Lean 4.33.0 (commit `d8b18978322de05a8f3dba51ef03cf5461676c17`) with cached mathlib v4.33.0 (manifest commit `db584cd6d46c92f209a44c0f1c829460d327499d`) compiled all eleven theorem declarations with **exit code 0, no errors and no warnings**. The check imported standard mathlib only; it imported no theorem or code from the user's Risley project and installed nothing.

Each individual theorem's printed axiom inventory is exactly `[propext, Classical.choice, Quot.sound]`. There are no admitted goals, `sorryAx` dependencies or custom axioms. The final source SHA256 is `402d3f1555dabe0639f54d8af43a770f30ed0a134ac6308846f4444e74d573e8`; the successful compiler-log SHA256 is `9502f23a92627676e8463aa4602683e4234a24555167c12a03e4c33242fe9778`. The separately retained failed draft log is superseded development provenance, not successful verification evidence.

For the first five statements let \(a,v,u\in\mathbb R^2\), \(d_a=1-u\cdot a\), \(d_v=1-u\cdot v\), and \(T(a,v,u)=I+(a-v)u^T/d_a\). All remaining scalar variables are real.

| Checked theorem | Exact statement and hypotheses |
|---|---|
| `entrance_det` | \(\det(I-au^T)=d_a\), no hypotheses. |
| `transport_factorization` | \(T(a,v,u)(I-au^T)=I-vu^T\), assuming \(d_a\ne0\). |
| `transport_det` | \(\det T(a,v,u)=d_v/d_a\), assuming \(d_a\ne0\). |
| `transport_inverse` | \(T(a,v,u)T(v,a,u)=I\), assuming \(d_a,d_v\ne0\). |
| `transport_det_pos` | \(\det T(a,v,u)>0\), assuming \(d_a,d_v>0\). |
| `combined_spatial_recurrence` | Identity (F1) below, assuming \(H-\dot X\ne0,\ Z\ne0,\ T\ne0\); all component scalars otherwise arbitrary. |
| `critical_inward_identity` | \(-2[a(H-a)+b\nu]=-2(H-a)(Hb+a)/(1+b)\), assuming \(1+b\ne0\) and \((H-a)^2=(1+b)\nu\). |
| `critical_inward_negative` | The preceding left side is negative, with the same equality and nonzero denominator, and additionally \(H-a>0,\ (Hb+a)/(1+b)>0\). |
| `external_flight_inward_identity` | \(-H(Hb+3a)/(H-a)^2=-H\ell/(H-a)\), assuming \(H-a\ne0,\ \ell=(Hb+3a)/(H-a)\). |
| `external_flight_inward_negative` | The preceding quantity is negative, assuming that active equality and \(H-a>0,\ H>0,\ \ell>0\). |
| `sqrt_difference_bound` | \(\lvert\sqrt s-\sqrt t\rvert\le\sqrt{\lvert s-t\rvert}\), assuming \(s,t\ge0\). |

The exact component identity checked for the combined recurrence is
\[
\begin{aligned}
\frac{A_i}{T}
+\frac{X_i(3+\alpha/T)}{H-\dot X}
+\frac{Y_i}{Z}
\left[\ell-\left(\frac{\alpha}{T}
+\frac{\dot X(3+\alpha/T)}{H-\dot X}\right)\right]
={}&
\frac{Z(H-\dot X)A_i+(ZX_i-HY_i)\alpha}
     {Z(H-\dot X)T}\\
&+\frac{3(ZX_i-\dot X Y_i)T+\ell(H-\dot X)Y_iT}
     {Z(H-\dot X)T}.
\end{aligned}
\tag{F1}
\]
In the optical specialization \(\alpha=u\cdot A\) and \(\dot X=u\cdot X\), but the theorem proves the identity for arbitrary scalar components.

These proofs verify algebraic identities, signs under displayed assumptions and the square-root inequality. Differentiation itself, derivation of the full optical graph and full-prior guards, boundary strictification, compactification, continuation, complete inversion, collision and spectral theorems remain outside this formalization. This is narrow supporting evidence, not a formal certificate of the complete report.

The complete algorithm in section 2A supplies an exact fallback independently of local harmonic regularity. A possible practical implementation could use the following local structure to reduce its workload:

1. Enclose candidate frequencies, first-order ellipses and the leading baseline from the paired record. Charge frequency uncertainty, harmonic leakage and quadratic DC contamination; retain unresolved collisions and alternative explanations.
2. Determine signed speeds where ellipse determinants exclude zero, and retain every plausible physical prism assignment.
3. Jointly enclose assigned fundamentals and the self-second harmonic. Form normalized invariants only where their denominators exclude zero.
4. Use the four-polynomial or trial-beam construction to propose regions. Preserve all roots, exceptional components and native bounds; uncertain coefficients define regions, not exact candidate points.
5. Recover wedges, phases and offsets jointly; correct and enclose surviving regions with the exact vector equations and four-variable geometry profiling.
6. Apply uniform exact native-coordinate certificates where valid. Verify all 400 residuals and strict physical constraints. Failed local correction remains unresolved unless sound global exclusion proves incompatibility.
7. Form the symbolic-error native ranges over every surviving branch. Report the all-eighteen diameter test and any incompatible/ambiguous/error-limited outcome.

That accelerated implementation is not completed. Finite-angle remainders, frequency uncertainty, branch coverage and practical conditioning certificates remain substantive obligations for it. They do not negate the separate finite exact global construction. Near weak excitation, collisions and critical optical branches, local charts or useful precision can fail; exact ambiguity prevents a uniform unique inverse on the full prior.

The conditional all-branch theorem has been proved. Remaining mathematical certification for a record concerns its full anchor fiber, discriminant and entire projected noise cube, useful inverse margins and treatment of exceptional strata. No useful universal conditioning follows; section 8A gives a regular-family collision barrier. Implementation still needs manageable exact/validated arithmetic, verified LP and circuit certificates, complete sheet/cover bookkeeping, actual-record thresholds and native ranges. These distinct burdens are recorded in [remaining_constructive_gap.md](theory/remaining_constructive_gap.md).

The user's original Dropbox research and observations remain unchanged. New documentation and source copies stay in this workspace. This revision performed theory integration, read-only mathematical review, source-byte packaging/hash checks and the explicitly authorized bounded Lean compilation described below. It ran no parameter campaign, recovery experiment or original project code, and did not rerun the supplied Python mathematical audits. Companion notes and supporting source files remain in the research workspace; this report states the principal constructions, certificates and limitations.


## 11. Related work, proposed contribution and validation gap

Third-order Risley optics is established: Li's 2011 work derives a nonparaxial thick-prism model and a third-order expansion for beam steering. Steering to a requested direction is a different inverse task from blind recovery of eighteen physical parameters from a passive sampled screen trace. [Li, 2011](https://doi.org/10.1364/AO.50.000679)

Numerical identification of wedge angles, refractive indices and installation parameters also predates this work. Li et al. report genetic-algorithm calibration and experimental pointing improvements; incomplete access to that paper's full text prevents an exhaustive fixed-versus-estimated parameter comparison here. Yuan et al.'s 2023 telescope work explicitly calibrates two encoder zeros, two wedges and two indices using gimbal directions and encoder readings, providing a concrete six-parameter comparator with additional measurement information. [Li et al., 2017](https://doi.org/10.1364/AO.56.007358), [Yuan et al., 2023, section 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC10051434/)

Fourier-based target-free main-section zero calibration and three-prism analytical modeling have relevant precedent. These works supply foundations for harmonic analysis and optical modeling; they do not by themselves settle identifiability of the particular blind observation map considered here. [Li and Zhou, 2021](https://doi.org/10.1364/AO.440678), [Li et al., 2017](https://doi.org/10.1364/OE.25.007677), [Qin et al., 2024](https://doi.org/10.1016/j.optcom.2024.130915)

The closest audited calibration comparator is the Livox Mid-40 observation model of Brazeal, Wilkinson and Hochmair. It uses exact three-dimensional vector Snell refraction, two identical prisms sharing wedge/index parameters, azimuth/zenith observations and an extended Kalman filter. Its thirteen model parameters are reduced by fixing air index, wedge angle and one tilt. Its discussion of wedge/index/air correlation is not an invariance proof for the distinct configuration here. That model must not be called merely paraxial, and the present three-prism screen-position theorem does not refute its sensor model or establish its identifiability. [Brazeal et al., 2021](https://doi.org/10.3390/s21144722)

The general machinery also has established origins: separating linear geometry variables from nonlinear optics is related to classical variable projection, and confluent Prony systems already have a local accuracy theory. The specific optical hypotheses, exact hard-band geometry and completeness obligations distinguish the present derivations from simply applying those general methods. [Golub and Pereyra, 1973](https://doi.org/10.1137/0710036), [Batenkov and Yomdin, 2013](https://arxiv.org/abs/1106.1137)

The defensible proposed contribution is the combination of an exact finite-angle, ambiguity-complete algebraic inverse with a reduced five-root compiler, exact physical compactification and conditional all-branch continuation; a symbolic-error native-coordinate recoverability profile; blind finite-record eighteen-parameter local identifiability; an explicit leading inverse curve and quadratic ambiguity breaker; exact source/circuit geometry elimination; conditional native uncertainty certificates; a collision-safe spectral outer cover and sharp critical-tail limitations; data-only final-prism and conditional upstream margin certificates; and an exact nonlinear finite-record collision barrier. The global construction uses established real-algebraic machinery; its optical encoding, branch control and particular inverse structure are the derived material here. This comparison does not establish first-ever novelty. Source-access and comparison provenance are recorded in [related_work_notes.md](theory/related_work_notes.md).

This remains a **research draft requiring expert review**. Independent symbolic audits are not external peer review. No physical experiment, practical all-branches execution or successfully evaluated useful finite-error certificate has yet been established. Publication requires stronger literature comparison and independent review of the full proofs, source reproducibility, quantitative finite-angle/conditioning certification and validation against every original residual constraint. Future empirical publication validation is a separate requirement; the current theory-first scope does not initiate an experimental campaign.
