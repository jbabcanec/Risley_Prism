# Global constructive inverse theorem for the finite full-vector model

**Status:** research draft; an independently audited mathematical construction, not an implemented or successfully evaluated global solver. This appendix concerns coupled vector Snell optics and all eighteen unknown native parameters. Measurement error is denoted by the symbolic parameter \(\epsilon\); a componentwise allowance may be written \(\epsilon_{h,k}\). Any small-wedge asymptotic scale used in companion results is a different quantity, \(\kappa\).

The [explicit six-root compiler](explicit_six_root_compiler.md) supplies a constructive alternative to the fixed single-sample quantifier-elimination preprocessing below. The [oblique inverse](physical_oblique_inverse.md) and [exact physical certificate framework](physical_inverse_certificates.md) supply local structure and possible numerical accelerators. None is an evaluated full-prior certificate.

## 0. Result and scope

For the three-prism vector-Snell model, its full eighteen-dimensional native prior box, and samples

\[
t_k=k/20,\qquad k=0,\ldots,K-1,\qquad K=200,
\]

there is a finite, ambiguity-complete exact inverse algorithm. Under the finite exact input encodings specified below, it returns the entire set of compatible systems, including disconnected and positive-dimensional fibers, or certifies emptiness. It also decides whether a requested worst-case native-coordinate accuracy is information-theoretically possible for the given record and error allowance.

This is not a global uniqueness theorem: uniqueness is false on the original box. Nor does the theorem establish practical recovery. Its sample-count complexity is polynomial when the number of prisms and the clock are fixed, but its constants and exponents can be prohibitive. No usable runtime for \(K=200\) is proved.

The constructive step that supports this complexity statement is to compile the **fixed single-sample graph first**, substitute exact rational rotor iterates second, and then intersect the sample predicates in eighteen shared variables. Directly eliminating an unstructured graph containing all \(K\) ray traces would involve \(O(K)\) variables and would not establish the stated bound.

All optical admissibility conditions in this theorem concern the measured times. Admissibility at every real time between observations is a separate condition requiring a separate mechanism.

## 1. A bijective algebraic chart of the original box

Native angles are in degrees. Introduce

\[
r_j=\tan\frac{\pi a_j}{180},\qquad
f_j=\tan\frac{\pi\phi_j}{360},\qquad
v_j=\tan\frac{\pi N_j}{20},
\]

\[
s_x=\tan\frac{\pi\beta_x}{180},\qquad
s_y=\tan\frac{\pi\beta_y}{180}.
\]

Keep \(n_1,n_2,n_3,d,g,p_x,p_y\) unchanged. These are eighteen chart coordinates. Their prior box is

\[
|r_j|\le\tan(\pi/10),\quad
|f_j|\le\tan(\pi/20),\quad
|v_j|\le\tan(7\pi/40),
\]

\[
|s_x|,|s_y|\le\tan(5\pi/36),\quad
13/10\le n_j\le9/5,
\]

\[
50\le d\le200,\qquad 2\le g\le15,\qquad |p_x|,|p_y|\le5.
\]

Every endpoint is algebraic: trigonometric functions at rational multiples of \(\pi\) have effective minimal-polynomial and real-root-isolation representations. Each coordinate map is strictly monotone on its entire prior interval. Their product is a homeomorphism from the native box to the algebraic box. No signed wedge, signed speed, endpoint, or phase is discarded.

In particular, if \(r_j=0\), its phase coordinate \(f_j\) remains arbitrary and independent. That rotor's zero tilt does not constrain \(v_j\), either. Such fibers must remain in the inverse. A zero-wedge glass slab is not removed: at oblique incidence its index can still affect internal lateral transport.

With \(i^2=-1\), define the instantaneous transverse exit-plane slope

\[
U_j(k)=u_j(k)+i w_j(k)
=r_j\frac{(1+i f_j)^2(1+i v_j)^{2k}}
{(1+f_j^2)(1+v_j^2)^k}. \tag{1}
\]

The two rotor factors are exactly \(e^{i\pi\phi_j/180}\) and \(e^{2\pi i N_j k/20}\), with \(\phi_j\) in native degrees and \(N_j\) in hertz. Every denominator is strictly positive on the real chart. The numerator degree is at most \(2k+3\), its integer coefficient bit length is \(O(k)\), and the denominator degree is \(2k+2\). Equation (1) is exact at every measured time; it is not a harmonic truncation.

The inverse native chart is

\[
a_j=\frac{180}{\pi}\arctan r_j,\quad
\phi_j=\frac{360}{\pi}\arctan f_j,\quad
N_j=\frac{20}{\pi}\arctan v_j,\quad
\beta_h=\frac{180}{\pi}\arctan s_h.
\]

These inverse functions are not asserted to be semialgebraic. Exact native angular outputs can instead be represented as arctangents of algebraic chart values.

## 2. Fixed-size polynomial graph and unique physical lift

For one sample, treat the three pairs \(U_j=(u_j,w_j)\) as independent real formal inputs. Let

\[
b=(p_x,p_y),\qquad \ell_1=\ell_2=g,\qquad \ell_3=d.
\]

Introduce incident external direction \((X_0,Z_0)\), transverse positions \(p_0,\ldots,p_3\), and, for each prism, internal axial optical momentum \(H_j\), outgoing external direction \((X_j,Z_j)\), and internal axial traversal \(h_j\). Every \(X_j\) and \(p_j\) is a two-vector.

The initial equations are

\[
X_0=sZ_0,\qquad |X_0|^2+Z_0^2=1,\qquad Z_0>0,
\]

\[
Z_0p_0=Z_0b+6X_0.
\]

For \(j=1,2,3\), impose

\[
H_j^2+|X_{j-1}|^2=n_j^2,\qquad H_j>0,
\]

\[
X_j-X_{j-1}+U_j(Z_j-H_j)=0,
\]

\[
|X_j|^2+Z_j^2=1,\qquad Z_j>0,
\]

\[
P_j:=H_j-U_j\cdot X_{j-1}>0,\qquad
R_j:=Z_j-U_j\cdot X_j>0,
\]

\[
h_jP_j=H_j(3+U_j\cdot p_{j-1}),
\]

\[
H_jZ_j(p_j-p_{j-1})
=h_jZ_jX_{j-1}+H_j(3+\ell_j-h_j)X_j. \tag{2}
\]

Here \(P_j,R_j\) are expressions, not additional variables. For strict sequential surface traversal, also impose

\[
h_j>0,\qquad 3+\ell_j-h_j>0. \tag{3}
\]

There are **26 auxiliary real variables**: three incident-direction coordinates, eight position coordinates, and three five-variable prism blocks. There are **26 scalar equations** and **19 strict inequalities**, including (3). All graph equations and inequalities have total degree at most three in the instantaneous inputs and auxiliaries.

Exact observations impose \(p_3=y\). Componentwise bounded error imposes

\[
y_h-\epsilon_h\le p_{3,h}\le y_h+\epsilon_h,
\qquad \epsilon_h\ge0.
\]

A separate Euclidean error ball at each sample instead imposes

\[
|p_3-y|^2\le\epsilon^2,\qquad \epsilon\ge0,
\]

also of fixed polynomial degree. A common scalar error allowance uses \(\epsilon_h=\epsilon\); known coordinate weights can be retained explicitly rather than assigning arbitrary numerical noise levels.

### Equivalence and uniqueness of the lift

The initial normalization and \(Z_0>0\) select exactly one incident unit direction. The equation for \(H_j\), with \(H_j>0\), selects the glass branch. Exit tangential momentum constrains the outgoing vector to an affine line parallel to the normal \((-U_j,1)\). Intersecting that line with the unit sphere gives at most two roots with opposite normal components; \(R_j>0\) selects at most one. A double root has \(R_j=0\) and is excluded. The condition \(P_j>0\) enforces the incoming normal orientation.

Positive \(P_j\) makes the traversal equation unique. Positive \(H_jZ_j\) makes the position equation unique. Equation (2) is exactly the sum of internal displacement \(h_jX_{j-1}/H_j\) and external displacement \((3+\ell_j-h_j)X_j/Z_j\). Conditions (3) enforce forward surface order. Induction therefore gives a unique auxiliary lift whenever the physical trace exists, and every satisfying polynomial lift is that trace. No unwanted root introduced by squaring remains.

The physical convention must be stated. Omit (3) only if the intended mathematical model permits backward or interleaved intersection geometries. All later results apply to either explicitly fixed convention. Strict grazing, transmission, and direction guards are never silently weakened.

## 3. Compile first, substitute second

The [explicit six-root compiler](explicit_six_root_compiler.md) gives a finite alternative to the preprocessing in this section. For interval observations it bounds terminal instantaneous polynomial degree by \(2368\) and terminal sign-test occurrences by \(24\cdot729\), before deduplication and excluding the prior box. These are conservative sign-circuit bounds, not evidence of a manageable implementation. The quantifier-elimination route below is independently complete.

Let \(G(x,U,y,\epsilon,z)\) denote the single-sample graph, where \(x\) contains the indices, incident slopes, and geometry; \(U\) contains six instantaneous tilt components; and \(z\) is the 26-variable auxiliary block. Compile

\[
Q(x,U,y,\epsilon)\quad\Longleftrightarrow\quad
\exists z\;G(x,U,y,\epsilon,z). \tag{4}
\]

A fixed real quantifier-elimination algorithm produces a finite Boolean combination of polynomial sign tests with rational coefficients. Its number of polynomial occurrences \(S_*\), maximum degree \(D_*\), coefficient height \(H_*\), and Boolean-expression size \(L_*\) are constants for this fixed graph, possibly enormous. This is an algorithmically specified precomputation, not an assertion that a usable expanded \(Q\) has already been computed.

At sample \(k\), substitute (1), the observed data, and the error allowance into \(Q\). Clear rational denominators with positive products, preserving every strict or weak sign and every Boolean connective. Call the result \(Q_k(\theta;y,\epsilon)\), in the same eighteen chart coordinates. The compatible chart set is

\[
\mathcal S(y,\epsilon)=
\left\{\theta\text{ in the chart prior box}:
\bigwedge_{k=0}^{K-1}Q_k(\theta;y,\epsilon)\right\}. \tag{5}
\]

The graph equivalence proves both directions: (5) contains precisely all compatible physical systems. Apply the coordinatewise inverse chart to obtain the native set. For a fixed allowance the compatible set has eighteen shared chart variables. Keeping one common scalar \(\epsilon\ge0\) free instead gives the joint nineteen-variable semialgebraic profile \(\{(\theta,\epsilon):\theta\in\mathcal S(y,\epsilon)\}\). Both dimensions are fixed, so the polynomial-in-\(K\) complexity conclusion below survives, with different fixed constants and exponents for the nineteen-variable profile. No numerical specialization of \(\epsilon\) is needed to construct this family. Exact decisions at a specified input value require its finite encoding as described below.

For rational observations and specified rational allowances of bit size at most \(\tau\), each \(Q_k\) has at most \(S_*\) polynomial occurrences, degree at most \(CD_*(k+1)\), and coefficient bit size at most \(C_*(\tau+k+1)\), for fixed constants. Fixed-degree polynomials are being expanded in rational expressions whose degrees and heights are \(O(k)\). Dense expansion has polynomial size because the dimension remains eighteen. Thus the whole formula has

\[
O(KS_*)\text{ occurrences},\qquad
O(D_*K)\text{ degree},\qquad
O(C_*(\tau+K))\text{ coefficient height}.
\]

Algebraic prior endpoints can be encoded by fixed univariate rational-coefficient root predicates, adding only constant complexity.

Each sample has its own ray lift while all samples share the same eighteen system coordinates. Local compilation uses the valid equivalence

\[
\bigwedge_k\exists z_k\,G_k
\quad\Longleftrightarrow\quad
\exists(z_0,\ldots,z_{K-1})\,\bigwedge_kG_k.
\]

It does not exchange unrelated system parameters or assume one common auxiliary ray state across samples.

## 4. Complete output and termination

Run an exact sign-invariant cylindrical algebraic decomposition (CAD), or another complete real-algebraic decomposition, on the polynomial family in (5). Retain exactly the cells satisfying its Boolean predicate. Output every retained cell's defining polynomials and root-order data, **not merely one representative point per component**. Their union represents every compatible chart point, and the inverse chart represents every native system.

For finite exact inputs this algorithm terminates without assumptions of nonzero rank, genericity, separation, a positive minimum optical margin, or a zero-dimensional fiber. It distinguishes emptiness, a singleton, finitely many systems, and positive-dimensional sets. It retains frequency collisions, zero or equal speeds, zero-wedge phase fibers, centered gauges, critical points, prior-boundary points satisfying the strict optical guards, and disconnected physical branches.

A CAD cell decomposition need not be a connected-component decomposition of the retained union. Complete cells suffice; no one-cell/one-component assertion is needed.

Open physical conditions do not prevent termination of this exact algebraic algorithm. They do obstruct naive compactness arguments and automatic finite termination of interval subdivision: boxes approaching a grazing boundary may persist indefinitely. Excluded boundary roots are not physical solutions. Coordinate extrema need not be attained, but their infimum and supremum endpoints in chart coordinates are algebraic and computable by projection and endpoint isolation.

## 5. Complexity and input encodings

Write

\[
s=O(S_*K),\qquad d=O(D_*K),\qquad
L=O(C_*(\tau+K)).
\]

Classical CAD arithmetic complexity for a fixed allowance has the form \((sd)^{2^{O(18)}}\). Effective fixed-dimensional rational/integer real-algebraic algorithms, with intermediate and output coefficient-height controls included, imply a polynomial bit-complexity bound in the count, degree, and input height. This is a deduction using that effective arithmetic and height analysis, not a bit-operation bound attributed to CAD's arithmetic theorem alone. Therefore, for this fixed graph and rational inputs, the full decomposition can be bounded by

\[
C_*(\tau+K+1)^{C_{18}}, \tag{6}
\]

where the fixed algorithm-dependent constants may be prohibitive. For the joint symbolic-\(\epsilon\) profile, replace dimension eighteen by nineteen and use its corresponding fixed exponent \(C_{19}\); the dependence on \(K\) remains polynomial under the same input scope. Keeping \((sd)^{2^{O(18)}}\), or its nineteen-variable counterpart, and the polynomial coefficient-bit factor visible is more informative than reading (6) as a favorable numerical bound. No small exponent, tractable implementation, or practical \(K=200\) runtime is implied.

Basu's author-hosted survey specifies the default arithmetic-operation convention in Definition 1.1 (p. 3). Theorem 2.4 (p. 6) gives the arithmetic CAD bound. Theorem 2.16 (p. 12) gives explicit block-quantifier-elimination bounds together with integer intermediate/output coefficient-bit-height controls. Applying the latter to the fixed single-sample graph makes the preprocessing constants effective. The polynomial bit conclusion above uses effective fixed-dimensional rational algorithms and the required height controls; it is not attributed to Theorem 2.4 alone. See [Basu, survey on algorithms in real algebraic geometry](https://www.math.purdue.edu/~sbasu/raag_survey2011.pdf), and the primary result [Basu, Pollack and Roy, *On the Combinatorial and Algebraic Complexity of Quantifier Elimination*, JACM 43 (1996), 1002–1045](https://doi.org/10.1145/235809.235813). These are algorithmic results, not merely existential elimination statements.

### Algebraic observations

Arbitrary algebraic observations cannot be placed into one common number field while silently treating its degree as fixed. Two rigorous encoding choices are available:

1. **Fixed number field.** Supply the data in a fixed field with an explicit real embedding. Charge its degree and height in the complexity. The sharper \(O(K)\) substituted-degree bound then applies with field arithmetic accounted for.
2. **Separate isolated real roots.** Supply each observation or allowance endpoint by an integer polynomial of degree at most \(\Delta\), a rational isolating interval, and coefficient/endpoint bit size at most \(\tau\). Keep the constant number of such values in each sample as temporary variables, impose their isolated-root predicates, and eliminate them locally. The number of local variables remains constant, so count, degree, and height remain polynomial in \(K,\Delta,\tau\), with fixed enormous exponents. The final eighteen-variable decomposition retains polynomial dependence on those parameters without forming an exponentially large common compositum. The sharper linear-in-\(K\) degree bound need not survive without additional structure.

### Clock and observation restrictions

The theorem uses the exact clock \(k/20\). Rational times can be embedded in a common rational lattice, but the complexity then depends on the largest resulting integer sample index, not just its binary length. A very large binary-encoded denominator or index does not inherit a polynomial bit-size guarantee. The exact ideal clock is not automatically identical to a stored floating-point timestamp array.

Exact arbitrary-real or transcendental data supplied only by approximation oracles do not support unconditional equality and sign decisions. Exact algebraic observations are an input model, not an assertion that every physical exact record is algebraic: generic native real parameters can generate transcendental observations. Rational measurement intervals cover ordinary finite-precision records.

The polynomial-in-\(K\) claim covers interval residual constraints and other fixed-size per-sample semialgebraic predicates, including a separate Euclidean ball at every sample. A norm or correlated-noise condition coupling all \(K\) samples requires separate complexity analysis. The full \(O(K)\)-auxiliary semialgebraic graph still gives finite exact inversion, but the eighteen-variable polynomial-in-\(K\) proof cannot be copied without a valid compilation argument.

## 6. Symbolic error and the exact recoverability criterion

Fix a record and an error model, leaving \(\epsilon\) symbolic when describing the compatible-set family. At any specified allowance for which \(\mathcal S(y,\epsilon)\ne\varnothing\), define the native-coordinate extrema

\[
\ell_j(\epsilon)=\inf_{\theta\in\mathcal S(y,\epsilon)}
\theta_{\mathrm{native},j},\qquad
u_j(\epsilon)=\sup_{\theta\in\mathcal S(y,\epsilon)}
\theta_{\mathrm{native},j}.
\]

The prior box is bounded, so these extrema are finite even if the compatible set is open. Project the chart set, compute its algebraic endpoint infima and suprema, and apply the monotone inverse chart to obtain them.

For unrestricted native point estimates, the sharp conditional worst-case radius is

\[
\inf_c\sup_{\theta\in\mathcal S(y,\epsilon)}
\max_j|c_j-\theta_{\mathrm{native},j}|
=\frac12\max_j\bigl(u_j(\epsilon)-\ell_j(\epsilon)\bigr). \tag{7}
\]

One optimal center is

\[
c_j=\frac{\ell_j(\epsilon)+u_j(\epsilon)}2.
\]

The optimal individual coordinate radius is \((u_j-\ell_j)/2\). Every center must lie at least half a projected diameter from one endpoint in the supremum sense, and the coordinatewise midpoint attains that bound. Endpoint attainment is unnecessary. This is a worst-case guarantee conditional on the observed record, not an average-risk result.

For coordinate tolerances \(\delta_j\ge0\), one unrestricted point estimate guarantees all requested errors if and only if

\[
u_j(\epsilon)-\ell_j(\epsilon)\le2\delta_j
\quad\text{for all eighteen }j.
\]

In particular, the original **0.001 native-coordinate target** is attainable exactly when all eighteen projected diameters are at most \(0.002\). This mixes degrees, hertz, index units, and native lengths according to the original coordinate tolerance; it is not a dimensionless conditioning statement.

If a diameter exceeds \(2\delta_j\), two actual compatible systems separated by more than that threshold exist, even if endpoint extrema are unattained. Exact real-algebraic witness extraction can construct their chart points. Such a pair is an ambiguity certificate, not a solver failure.

### Exact midpoint representation

For an angle-like coordinate with algebraic chart endpoints \(a\le b\), its native midpoint has algebraic chart representative

\[
m=\frac{a+b}{\sqrt{(1+a^2)(1+b^2)}+1-ab}.
\]

The denominator is positive on the present charts, and the half-angle identity gives

\[
\arctan m=\frac{\arctan a+\arctan b}{2}.
\]

Thus an exact algebraic chart point represents the unrestricted minimax center, though its displayed native angular coordinates are arctangent expressions.

### Feasible-center distinction

The midpoint need not be a physically compatible system. If the estimate itself must belong to \(\mathcal S\), impose the different requirement

\[
\exists c\in\mathcal S\;\forall\theta\in\mathcal S:
|c_{\mathrm{native},j}-\theta_{\mathrm{native},j}|\le\delta_j
\quad\text{for every }j.
\]

For fixed rational native tolerances, these comparisons are algebraic by the arctangent-difference argument below, using \(\delta_j\) rather than \(2\delta_j\) for a radius test. Fixed-dimensional quantifier elimination decides the statement. Rational bisection approximates an optimized feasible-center radius; its native value is not claimed algebraic, nor is its infimum claimed attained on an open compatible set. Equation (7) must not be applied to this different problem.

### Exact native angular tolerance test

For a native coordinate \(q=c\arctan x/\pi\), with \(c\in\{180,360,20\}\), its diameter is

\[
\frac c\pi(\arctan b-\arctan a).
\]

All current chart intervals lie inside \((-1,1)\). Hence \(1+ab>0\) and the angular difference lies in \([0,\pi/2)\). For rational native tolerance \(\delta\), the diameter test is exactly

\[
b-a\le\tan\!\left(\frac{2\pi\delta}{c}\right)(1+ab), \tag{8}
\]

when the target angle \(2\pi\delta/c<\pi/2\); larger targets are automatically satisfied on these charts. The threshold is algebraic for rational \(\delta\). Its representation cost matters: a large binary denominator in a rational multiple of \(\pi\) need not yield an algebraic number of degree polynomial in that denominator's bit length. A fixed requested tolerance has fixed encoding cost. Native endpoint arctangent expressions admit certified rational enclosures to any positive requested accuracy; they are not generally algebraic numbers themselves.

If an admissible true system generates the record within its declared error allowance, it belongs to \(\mathcal S\), so the guarantee applies. An empty set indicates model/data/allowance inconsistency. No uniformly small error can be promised over the full prior: a compatible zero-wedge configuration has an invisible phase with native diameter 36 degrees. Exact gauges and near-degenerate configurations provide further genuine limitations.

## 7. Exact geometry profiling and mixed-strict Helly reduction

Fix the fourteen optical chart variables. Refraction directions and branch guards are independent of \(b,g,d\); surface-order inequalities remain geometry-dependent. The exact paired position recurrence is affine in

\[
q=(p_x,p_y,g,d)\in\mathbb R^4.
\]

Interval observations, the geometry box, and strict traversal therefore form \(M=O(K)\) affine weak or strict halfspaces. In particular,

\[
h_j=\frac{H_j(3+U_j\cdot p_{j-1})}{P_j}
\]

and \(3+\ell_j-h_j\) are affine in \(q\) at fixed optics.

Finite Helly's theorem applies to arbitrary convex sets, including open and mixed halfspaces: the full intersection is nonempty if and only if each subfamily of at most five halfspaces has nonempty intersection. Include the eight geometry-box halfspaces in this same family. Optical compatibility is therefore equivalent to \(O(M^5)=O(K^5)\) constant-size affine feasibility tests, retaining every strict flag. No nested \(O(K^9)\) projected-polygon enumeration is needed.

These projected conditions are not generally closed. For example,

\[
\exists x\;(x>0\;\wedge\;x\le t)\quad\Longleftrightarrow\quad t>0.
\]

It would be incorrect to call the optical conditions closed without extra margin assumptions. Solve each small problem by fixed-dimensional real quantifier elimination or an exact mixed-strict linear procedure. A common slack \(e\) is valid if strict rows become \(a_iq+e\le b_i\), weak rows remain unchanged, and **\(e>0\)** is required. A finite strictly feasible system admits some positive common slack; the converse is immediate. Do not replace \(e>0\) by \(e\ge0\), or assume a positive optimum is attained.

Each five-row test references at most five sampled ray-coefficient systems, so its auxiliary count is constant and its substituted degree is polynomial in \(K\). This gives another fixed-dimensional theoretical method. It is optional: the directly compiled eighteen-variable formulation already establishes completeness and polynomial sample-count dependence.

### Geometry extrema from closure vertices

At an optical point \(x\) with nonempty mixed-strict geometry fiber \(F_x\), let \(P_x\) be the polytope formed by weakening its strict rows. Then

\[
\overline{F_x}=P_x.
\]

One inclusion is immediate. For the other, choose \(q_*\in F_x\). For any \(q\in P_x\), the convex combinations \((1-t)q+tq_*\), \(0<t\le1\), satisfy all original strict and weak rows and converge to \(q\). The argument preserves equalities, whether explicit or represented by opposite weak rows, and does not assume an interior point in \(\mathbb R^4\). The geometry box makes \(P_x\) bounded.

Every linear coordinate extremum of a nonempty bounded polytope occurs at a vertex. At a vertex there are four linearly independent active row normals. Otherwise a nonzero vector annihilates every active normal, allowing sufficiently small feasible displacement in both signs and contradicting extremality. This proof also covers lower-dimensional polytopes and singleton fibers, because affine-hull equalities count among active rows.

Enumerate the \(O(M^4)\) four-row subsets \(I\). Where \(\det A_I(x)\ne0\), define the Cramer candidate

\[
q_I(x)=A_I(x)^{-1}b_I(x).
\]

Retain it only if it satisfies every weakened row, and retain the optical point only if its **original mixed-strict fiber is nonempty**. The full compatible set's coordinate infima and suprema equal those over these retained closure vertices and all admissible optical points. This is equality of extremal values: a closure vertex on an excluded strict boundary is not itself a compatible physical system.

The original-fiber feasibility test is essential. Weakening \(q>0\) and \(q\le0\) creates the false surrogate vertex \(q=0\), although the strict fiber is empty.

Cramer determinants can have either sign. Split on determinant sign or clear denominators using positive squares; do not multiply inequalities by unchecked determinants. Each candidate references four rows, and each validity test adds only one further row. Thus a branch-preserving test involves at most five sampled ray expressions and admits constant-auxiliary compilation. Together with the \(O(M^5)\) strict-feasibility tests, this gives a polynomial-count description in fourteen optical variables for geometry-range computations. It remains an enormous exact construction, not a completed practical implementation.

## 8. Completion levels and unresolved work

The established theoretical result includes a finite ambiguity-complete algorithm, the complete bounded native prior at measured times, a bijective chart retaining singular fibers, exact transmission and surface-order predicates, sample-local compilation, an input-sensitive complexity bound, and the sharp conditional minimax criterion.

Three levels must remain distinct:

1. **Completeness and termination in principle.** For finite exact input encodings, every compatible system and every singular or mixed-strict boundary case in the sampled-time model is represented.
2. **Fixed-prism asymptotic complexity.** The one-time compiler constants \(S_*,D_*,H_*,L_*\) and the fixed eighteen-variable elimination/decomposition exponents are retained; the joint symbolic-\(\epsilon\) profile uses nineteen variables. Polynomial sample-count dependence is a theoretical bound, not a favorable numerical estimate.
3. **Practical all-branch certification.** A tractable complete solver for an actual 200-sample record, useful runtime bounds, numerically effective full-prior certificates, and useful finite-wedge conditioning constants have not been established.

The [oblique formal inverse](physical_oblique_inverse.md) can propose candidate regions. Its generic finite polynomial candidate bound concerns leading formal coefficients, not the full finite-angle record. Certified exact residual, Jacobian, and affine-geometry bounds may exclude boxes or validate local solutions. Every exclusion must be valid for the exact branch-preserving model. Unverified asymptotic fits, unresolved frequency assignments, singular strata, and boxes meeting strict boundaries remain in the global compatible-set description. The algebraic backend supplies a complete theoretical fallback; it does not authorize deleting unresolved regions from a practical computation.

Global uniqueness classification for nondegenerate records, quantitative usable conditioning, and continuous-time physical admissibility remain separate obligations. The local-rank results are useful for generic behavior and initialization, but are not needed for the global completeness theorem.

**Audit and execution status.** The source theorem and its mathematical corrections were independently audited. This appendix records those derivations and scopes; the enormous compiler/decomposition algorithm has not been executed. No recovery experiments, parameter sweeps, or project-code changes are part of this result. This remains a research draft requiring expert review, not a publication-ready or practically complete recovery system.
