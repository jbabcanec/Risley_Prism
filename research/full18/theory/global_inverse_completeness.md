# Global constructive inverse theorem for the finite full-vector model

## 0. What this result does and does not establish

This is an ambiguity-complete inverse theorem for the original three-prism vector-Snell model, its full eighteen-dimensional prior box, and the samples k/20, k=0,...,199. It returns the complete set of compatible systems, including positive-dimensional and disconnected fibers, or certifies that no system is compatible. It decides whether a requested worst-case coordinate accuracy is information-theoretically possible for a given record. It is not a global uniqueness theorem: such a theorem is false on the stated box. Its finite exact algorithm has polynomial complexity in the number of consecutive samples when the number of prisms and the clock are fixed, but with enormous constants and exponents. No usable runtime is proved.

The constructive improvement over a bare appeal to real quantifier elimination is crucial: one compiles the fixed single-sample optical graph once, substitutes rational rotor iterates, and only then intersects the sample predicates in eighteen shared variables. Putting all K ray graphs into one unstructured elimination problem would have O(K) variables and would not establish this bound.

All physical-admissibility statements below concern the measured times. Admissibility for every real time between observations is an additional requirement not silently covered by this theorem.

## 1. A bijective algebraic chart of the original box

Angles in the original parameter list are in degrees. Use

r_j=tan(pi a_j/180), f_j=tan(pi phi_j/360), v_j=tan(pi N_j/20),
s_x=tan(pi beta_x/180), s_y=tan(pi beta_y/180).

Keep n_1,n_2,n_3,d,g,p_x,p_y unchanged. These are eighteen coordinates. All their prior intervals have algebraic endpoints:

|r_j|<=tan(pi/10), |f_j|<=tan(pi/20), |v_j|<=tan(7pi/40),
|s_x|,|s_y|<=tan(5pi/36),
13/10<=n_j<=9/5, 50<=d<=200, 2<=g<=15, |p_x|,|p_y|<=5.

Trigonometric functions at rational multiples of pi are algebraic, and their desired real values have effective minimal-polynomial/root-isolation representations. Each chart is strictly monotone on its whole prior interval. This is a product homeomorphism from the native box onto an algebraic box; it does not discard any signed wedge, signed speed, endpoint, or phase.

In particular, when r_j=0, f_j remains an arbitrary independent coordinate. Thus all zero-wedge phase fibers survive without reconstruction conventions. The speed v_j is also unconstrained by that rotor's tilt, although the common record may still constrain other quantities. A zero-wedge glass slab is not removed: its refractive index still affects internal lateral transport for an oblique incident ray.

Let i denote the imaginary unit and define the instantaneous exit-plane slope by

U_j(k)=u_j(k)+i w_j(k)
 = r_j (1+i f_j)^2 (1+i v_j)^(2k) / [(1+f_j^2)(1+v_j^2)^k].             (1)

Indeed, the two factors are exp(i phi_j) and exp(2pi i N_j k/20). Denominators are strictly positive for every real chart point. The numerator has total degree at most 2k+3 and integer coefficient bit length O(k); the denominator has degree 2k+2. Formula (1) is exact at every measured time, not a harmonic truncation.

The inverse native chart is a_j=(180/pi)atan r_j, phi_j=(360/pi)atan f_j, N_j=(20/pi)atan v_j, beta=(180/pi)atan s. These inverse functions are not asserted to be semialgebraic.

## 2. A fixed-size polynomial graph with a unique physical lift

For one sample regard all three instantaneous pairs U_j as independent formal inputs. Let b=(p_x,p_y), ell_1=ell_2=g, ell_3=d. Introduce incident external direction (X_0,Z_0), four transverse positions p_0,...,p_3, and for each prism an internal axial optical momentum H_j, outgoing external direction (X_j,Z_j), and internal axial traversal h_j.

The initial equations are

X_0=s Z_0, |X_0|^2+Z_0^2=1, Z_0>0,
Z_0 p_0=Z_0 b+6X_0.

For each j=1,2,3 impose

H_j^2+|X_(j-1)|^2=n_j^2, H_j>0,
X_j-X_(j-1)+U_j(Z_j-H_j)=0,
|X_j|^2+Z_j^2=1, Z_j>0,
P_j=H_j-U_j dot X_(j-1)>0,
R_j=Z_j-U_j dot X_j>0,
h_j P_j=H_j(3+U_j dot p_(j-1)),
H_j Z_j (p_j-p_(j-1))
 =h_j Z_j X_(j-1)+H_j(3+ell_j-h_j)X_j.                (2)

Here P_j and R_j are expressions, not additional unknowns. For the strict sequential-traversal model also impose

h_j>0, 3+ell_j-h_j>0.                                 (3)

There are 26 auxiliary real variables: incident direction 3, positions 8, and three prism blocks of 5 each. There are 26 scalar equations and 19 strict inequalities including (3). All have degree at most three in the instantaneous inputs and auxiliaries. Exact data impose p_3=y; componentwise bounded-error data impose y_l-eta_l<=p_(3,l)<=y_l+eta_l. Per-sample Euclidean bounded error instead imposes |p_3-y|^2<=eta^2 with eta>=0, still of fixed degree.

Proof of equivalence and uniqueness. Initial normalization and Z_0>0 uniquely determine the incident direction. H_j>0 uniquely selects the glass branch. At the exit, the tangential equations constrain the outgoing vector to an affine line parallel to the normal (-U_j,1). Intersecting that line with the unit sphere gives at most two roots; their normal components have opposite signs, and R_j>0 selects at most one. A double root has R_j=0 and is excluded. The normal incoming condition is P_j>0. Thus these equations are precisely the forward transmitted vector-Snell branch, with no spurious squared root. Positive P_j makes the equation for h_j unique; positive H_j Z_j makes the position equation unique. Equation (2) is the sum of internal displacement h_j X_(j-1)/H_j and external displacement (3+ell_j-h_j)X_j/Z_j. Conditions (3) select forward surface order. Induction proves a unique auxiliary lift whenever the optical trace exists, and proves that every satisfying polynomial lift is that trace.

The physical domain must be specified: omit (3) only if the intended mathematical model really allows those backward/interleaved intersection geometries. All later statements apply to either explicitly fixed convention. They never admit grazing or TIR boundaries by replacing strict inequalities with weak ones.

## 3. Compile first, substitute second

An explicit alternative to the QE precomputation is now derived and independently checked in explicit_six_root_compiler.md. It uses six positive square roots and a finite radical-sign table, with terminal instantaneous polynomial degree at most2368 for interval data and at most24*729 terminal sign tests before sharing. The QE construction below remains a second complete route; the explicit compiler avoids leaving preprocessing as an unspecified huge elimination.

Let G(x,U,y,eta,z) be the single-sample predicate above, where x consists of indices, incident slopes, and geometry, U consists of six instantaneous tilt components, and z is the 26-variable auxiliary block. Compile

Q(x,U,y,eta) <=> exists z G(x,U,y,eta,z).               (4)

A fixed real quantifier-elimination algorithm produces a finite Boolean combination of polynomial sign tests with rational coefficients. Its number S_* of polynomial occurrences, degree D_*, coefficient height H_*, and Boolean-expression size L_* are absolute constants for this fixed three-prism graph. They may be extremely large. This is an algorithmically specified precomputation, not a claim that a manageable explicit Q has already been computed or printed.

For each k, substitute (1) into Q and substitute the observed data/noise bounds. Clear rational denominators using positive denominator products, preserving every strict or weak sign and every Boolean connective. Let Q_k(theta) be the resulting predicate in the same eighteen chart coordinates. Define

S(y,eta)={theta in the chart prior box: AND_(k=0)^(K-1) Q_k(theta)}.       (5)

Equations (1)-(4) prove, in both directions, that (5) is exactly the set of all compatible physical systems. The native compatible set is its coordinatewise inverse-chart image.

For rational observations/noise bounds of bit size at most tau, each Q_k has at most S_* polynomial occurrences and degree at most C D_*(k+1), for an absolute C. Its coefficient bit size is at most C_*(tau+k+1). This follows by expanding fixed-degree polynomials in rational expressions whose numerator/denominator degrees and heights are O(k). Dense expansion has polynomial size because there are only eighteen variables. The whole formula therefore has O(K S_*) occurrences, degree O(D_* K), and coefficient height O(C_*(tau+K)). Algebraic prior endpoints can be encoded by fixed univariate rational-coefficient sign predicates; this adds only constant complexity.

No exchange of unrelated existential witnesses is made: each sample has its own ray lift, and all samples share exactly the same eighteen system coordinates. The equivalence (AND_k exists z_k G_k) <=> exists (z_0,...,z_(K-1)) AND_k G_k justifies the local compilation.

## 4. Complete output and termination

Run an exact sign-invariant cylindrical algebraic decomposition (CAD), or another complete real-algebraic decomposition algorithm, on the polynomial family in (5), and retain precisely those cells satisfying the predicate. Output the defining polynomial/root-order data of every retained cell, not merely one point per component. Their union represents every compatible chart point. The inverse chart gives every native system.

This algorithm always terminates for the stated finite exact encodings, with no nonzero-rank, genericity, separation, positive minimum-margin, or zero-dimensionality assumption. It distinguishes an empty set, a singleton, a finite set of multiple systems, and positive-dimensional sets. It retains frequency collisions, zero and equal speeds, all zero-wedge phase fibers, centered gauges, critical points, all prior-boundary points that satisfy strict optical conditions, and all disconnected physical branches. A CAD cell decomposition is not itself necessarily a connected-component decomposition of the entire retained union; output cells suffice for a complete inverse and no false one-cell/one-component claim is made.

Open physical conditions cause no nontermination in this exact algorithm. They do invalidate naive compactness arguments and interval branch-and-bound stopping arguments: cells approaching grazing can persist indefinitely under such numerical subdivision. An excluded boundary root is never returned as a physical solution. Coordinate infima and suprema need not be attained, but remain exact algebraic endpoints in chart coordinates and can be computed by projection and endpoint isolation.

## 5. Complexity, with its encoding assumptions exposed

Let s=O(S_* K), d=O(D_* K), and L=O(C_*(tau+K)). Classical CAD has arithmetic complexity (sd)^(2^O(18)); fixed-dimensional rational/integer real-algebraic algorithms have corresponding polynomial bit complexity, with dependence on coefficient height included. Thus, for this fixed graph and rational data, the full decomposition can be bounded by

C_* (tau+K+1)^{C_18},                                 (6)

where C_* and C_18 are fixed, algorithm-dependent constants, potentially prohibitive. A more honest uncollapsed statement retains (sd)^(2^O(18)) and the polynomial coefficient-bit factor. Formula (6) does not assign a small exponent, show a usable implementation, or imply that K=200 is computationally tractable.

For a verified foundational result, Basu's author-hosted survey states CAD complexity in Theorem 2.4 and explicit block-QE size/degree/arithmetic bounds plus integer-height control in Theorem 2.16. Applying the latter to the fixed sample graph proves the constant-precompilation assertion. These results are algorithmic, not merely existential elimination. Source: https://www.math.purdue.edu/~sbasu/raag_survey2011.pdf (Theorems 2.4, 2.16). The primary underlying work is Basu, Pollack and Roy, On the Combinatorial and Algebraic Complexity of Quantifier Elimination, JACM 43 (1996), 1002-1045, DOI 10.1145/235809.235813.

Arbitrary algebraic input needs care. One cannot put K unrelated algebraic observations into a common number field and silently assume its degree remains fixed. Two rigorous options are:

(a) Supply observations in a fixed number field with an explicit real embedding, and include its degree/height in complexity. The sharper O(K) substituted-degree statement applies directly, with field arithmetic charged correctly.

(b) Supply each observation/noise endpoint by an integer polynomial of degree at most Delta and a rational isolating interval, with coefficient/endpoint bit size tau. In each sample retain these constant-many algebraic values as temporary variables, enforce their isolated-root predicates, and eliminate them locally. This keeps the number of local variables constant, so the degree/count/height remain polynomial in K, Delta, tau (with fixed enormous exponents). The final eighteen-variable stage remains polynomial in those parameters. This gives polynomial complexity in dense univariate algebraic encodings without taking an exponentially large common compositum. It loses the sharper linear-in-K degree bound unless more structure is known.

The clock is exactly k/20. More general rational times can be put on a common rational lattice, but complexity depends on the maximum resulting integer sample index, not merely its binary length. A very large binary-encoded index or denominator does not inherit a polynomial bit-size guarantee. Arbitrary real/transcendental observations represented only by numerical oracles do not support these exact zero/sign decisions.

## 6. Noise and the exact recoverability criterion

Let S be the nonempty compatible set for the specified noise model. For each native coordinate j, let l_j=inf(theta_native,j) and u_j=sup(theta_native,j) over S. These extrema are well-defined because the prior box is bounded, whether or not S is closed. They are found by projecting the chart set to its corresponding coordinate, taking algebraic infimum/supremum endpoints, and applying the monotone inverse chart.

For unrestricted point estimates in native coordinates,

inf_c sup_(theta in S) max_j |c_j-theta_j| = (1/2) max_j (u_j-l_j).       (7)

One optimal center is c_j=(l_j+u_j)/2. More generally the optimal individual coordinate radius is (u_j-l_j)/2. Proof: every center must be at least half the projected diameter from one endpoint in the supremum sense; the coordinate midpoint achieves the bound. No attainment of l_j or u_j is required. This is conditional worst-case recovery for the observed record, not an average-risk statement.

For coordinate-dependent tolerances delta_j, one point estimate can guarantee all requested errors if and only if u_j-l_j<=2 delta_j for every j. Thus the algorithm either certifies a point-estimation guarantee, or produces an ambiguity certificate. If a diameter exceeds 2 delta_j, strictness of the inequality guarantees two actual feasible points separated by more than 2 delta_j, even when extrema are unattained. Exact real-algebraic witness extraction constructs such chart points.

The native midpoint has an exact algebraic chart representative. If a,b are the chart endpoints of an angle-like coordinate, its midpoint chart value is

m=(a+b)/[sqrt((1+a^2)(1+b^2))+1-ab].

The denominator is positive on all present charts, and the half-angle identity proves atan(m)=(atan(a)+atan(b))/2. Thus one can output an exact algebraic chart point for the unrestricted minimax center, even though its displayed native angle values are arctangent expressions.

The midpoint need not itself be a physically compatible system. If the requested estimate must be feasible, impose that additional constraint and solve, for fixed rational native tolerances, exists c in S such that for every theta in S all native coordinate differences are within tolerance. These comparison predicates are algebraic by the same arctangent-difference argument below (use delta rather than 2delta for a radius test). Fixed-dimensional QE decides the statement. An optimized feasible-center native radius can be approximated by rational bisection; its native value is not claimed algebraic, nor is the infimum claimed attained on an open S. Equality (7) must not be claimed for that different problem.

For an angle-like native coordinate q=c atan x/pi with c in {180,360,20}, its diameter is c(atan b-atan a)/pi for algebraic chart extrema a<=b. All present chart intervals lie inside (-1,1), so 1+ab>0 and the arctangent difference belongs to [0,pi/2). A rational-native tolerance delta is therefore exactly testable by

(b-a) <= tan(2pi delta/c)(1+ab),                      (8)

whenever the target difference 2pi delta/c is below pi/2; larger targets are trivially satisfied here. The threshold is algebraic for rational delta. Its algebraic encoding cost must be counted: tan(pi times a rational with a large binary denominator) need not have polynomial algebraic degree in that denominator's bit length. For fixed requested tolerances this is a fixed constant. Exact native endpoints can be represented as arctangents of algebraic numbers, and certified rational enclosures can be computed to any strictly positive accuracy. Do not replace this representation with the false statement that native angle values are algebraic.

If the measured record is generated by an admissible true system with error satisfying the chosen bounds, that true system belongs to S, so the guarantee applies. If S is empty, the result is model/data/noise inconsistency. The algorithm cannot promise small error uniformly over the prior: at a zero wedge its phase is completely invisible, already giving an exact native phase diameter of 36 degrees whenever that configuration is feasible. Other exact gauges and near-degeneracies add genuine ambiguity, not solver failures.

## 7. Optional exact geometry profiling and the strict Helly correction

Fix the fourteen optical chart variables. Directions and all refraction-branch conditions are independent of b,g,d; traversal-order conditions remain geometry-dependent. The paired position recurrence is affine in q=(p_x,p_y,g,d), so interval observations, the geometry box, and strict traversal order give M=O(K) affine weak or strict halfspaces in R^4. In particular h_j=H_j(3+U_j dot p_(j-1))/P_j is affine in q, and 3+ell_j-h_j is affine too.

Finite Helly's theorem for arbitrary convex sets, including open or mixed halfspaces, implies that their whole intersection is nonempty exactly when every subfamily of at most five halfspaces has nonempty intersection. Include the eight geometry-box halfspaces in this same family. Hence optical compatibility can be tested by O(M^5)=O(K^5) constant-size affine feasibility conditions, each with its original strict flags. This is a legitimate geometry elimination route and avoids nested O(K^9) projected-polygon enumerations.

These optical conditions are not generally closed. The elementary projection exists x (x>0 and x<=t) is equivalent to t>0. Thus a claim of O(K^5) closed constraints without extra margin assumptions is false. Each small subproblem can instead be solved by fixed-dimensional real QE, or an exact mixed-strict linear feasibility procedure. If one uses a common slack e, strict rows become a_i q+e<=b_i, weak rows stay unchanged, and one requires e>0. A finite strictly feasible set admits such a common positive slack; its converse is immediate. Do not relax e>0 to e>=0 or presume a positive optimum is attained.

Each five-row test references at most five sample ray-coefficient systems, hence a constant auxiliary count and degree polynomial in K after rotor substitution. This also leads to a fixed-dimensional theoretical algorithm. It may be useful for a specialized implementation, but the direct compiled eighteen-variable method already proves global completeness and polynomial sample-count complexity and does not need Helly.

### Geometry coordinate extrema from closure vertices

There is a useful further exact reduction. At an optical point x whose mixed strict/weak geometry fiber F_x is nonempty, let P_x be the polytope obtained by weakening its strict rows. Then closure(F_x)=P_x. The inclusion into P_x is immediate. Conversely choose q_* in F_x. For any q in P_x, the points (1-t)q+t q_* belong to F_x for every 0<t<=1, and converge to q. The geometry box makes P_x bounded. This argument preserves equalities, whether encoded directly or by opposite weak rows, and does not require an interior point in R^4.

Every linear coordinate extremum on the nonempty bounded polytope P_x occurs at a vertex. At each vertex there are four linearly independent active row normals. Otherwise a nonzero direction annihilating all active normals would permit a sufficiently small feasible displacement in both signs, contradicting extremality. This proof works for lower-dimensional polytopes and singleton fibers, since affine-hull equalities count among active rows.

Consequently enumerate the O(M^4) four-row subsets I. On det(A_I(x))!=0, the candidate q_I(x)=A_I(x)^(-1)b_I(x) is given by Cramer's rule. Retain it exactly when it satisfies every weakened row, and retain the optical point only when the original mixed-strict fiber is feasible. Coordinate extrema over the full compatible set equal the infimum/supremum of the corresponding coordinates of these retained closure vertices over all admissible optical points. This equality concerns values; a vertex lying on an excluded strict boundary is not a compatible physical system.

The explicit original-fiber feasibility check is indispensable. Weakening an empty strict fiber can manufacture vertices and false extrema. For example q>0 and q<=0 has no strict point but its weak closure surrogate has q=0.

Cramer formulas introduce determinant denominators of either sign. Split on determinant sign or clear using positive squares; never multiply an inequality by an unchecked determinant. Each candidate is specified by four rows, and each vertex-validity test references those four rows plus one test row. Hence each branch-preserving coefficient test involves at most five sampled ray expressions and can be compiled with a constant auxiliary count. Together with the O(M^5) strict-feasibility tests, this yields a polynomial-count fourteen-optical-variable description for geometry range computations. It remains an enormous exact method; the existence of this description is not a completed practical implementation.

## 8. Remaining work, stated precisely

What is established is a finite ambiguity-complete algorithm, exact handling of the entire bounded native parameter box at sampled times, a rational chart retaining singular fibers, exact branch/order predicates, an explicit sample-local compilation strategy, a defensible input-sensitive complexity theorem, and the sharp conditional minimax criterion.

What remains unestablished is a practical implementation of the compiled predicate, a tractable complete K=200 solver, numerically useful finite-wedge conditioning constants, and a global uniqueness classification for nondegenerate records. The local-rank results in the companion files remain useful for generic behavior and numerical initialization, but none is required for the global completeness theorem above. Continuous-time physical admissibility, if desired, needs a separately stated and proved mechanism.

### Three distinct levels of completion

1. Exact global completeness and termination in principle are established under finite exact input encodings. This includes every compatible system and every singular or mixed-strict boundary case in the sampled-time model.
2. Fixed-prism asymptotic complexity is established with the one-time compiler constants S_*,D_*,H_*,L_* explicitly retained and fixed eighteen-variable elimination/decomposition exponents exposed. Polynomial sample-count dependence is a theoretical complexity result, not a favorable numerical bound.
3. Practical, all-branches certification for the actual 200-sample instance is not established here. No computed usable compiled predicate, runtime estimate, or realistic full-box certificate is supplied. A claim that the practical inverse is solved would exceed the theorem.

The data-derived oblique first/second-order inverse in oblique_inverse.md can propose candidate regions. Its polynomial candidate bound concerns formal leading coefficients in a generic stratum, not the full finite record. Certified interval bounds on exact forward residuals, Jacobians, and affine geometry LPs may exclude boxes or validate local solutions. Every exclusion must be valid for the exact branch-preserving model. Unverified asymptotic coefficient fits, unresolved frequency assignments, singular strata, and boxes reaching strict boundaries remain in the global compatible-set description. The complete algebraic backend supplies the fallback proof obligation, not permission to delete those regions.

The sharp polynomial-in-K statement above covers interval residual bounds and fixed-size per-sample semialgebraic noise predicates, including a separate Euclidean ball at each sample. A global residual norm or correlated noise constraint coupling all K samples requires a separate complexity analysis; the full O(K)-auxiliary semialgebraic graph still gives finite exact inversion, but the eighteen-variable polynomial-in-K proof must not be copied without a valid compilation argument.

Finally, exact algebraic observations are an input model, not a claim that every physically generated exact record is algebraic. A generic native real parameter can produce transcendental samples. Rational measurement intervals cover ordinary finite-precision data; exact arbitrary-real records available only as approximation oracles do not admit unconditional equality decisions of the kind used here.
