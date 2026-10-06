# A certified adaptive inverse: explicit interface and quantitative stopping

## Scope and status

This addendum joins the audited reductions into one algorithm for the original three-prism full18 model and all 200 paired observations at times k/20. The native priors, signed rotors, sampled-time physical branch and coordinatewise observation allowance epsilon are unchanged. Its unconditional output is an ambiguity-complete inverse and an exact symbolic-noise recoverability decision, not an unconditional accurate estimate.

Two additional deductions are proved here:

1. A joint observation/traversal violation LP supplies a quantitative, critical-boundary-safe adaptive exclusion theorem. It replaces the unquantified convergence of shrinking traversal relaxations by an explicit modulus, without a Hoffman bound.
2. Combining the mixed-strict circuit compiler with one queried coordinate reduces the exact symbolic-envelope construction to at most sixteen outer variables: fourteen optics, epsilon and that coordinate. No 37-variable pair-product is required.

These are certificate-size and representation improvements. Neither supplies a moderately conditioned record, a small complete anchor enumeration, or cheap construction of every certificate. The complete solver still has an expensive exact fallback. A useful all-branches runtime for a qualifying full-vector record has not been established.

The ingredients used from the other addenda are mathematical/symbolic audited results. This synthesis and its two deductions are not a proof-assistant formalization or a literature-novelty claim. No simulation, sweep or numerical recovery is used.

## 1. Exact state and the two coordinate systems

Let x be the fourteen original optical chart coordinates: three wedge slopes, three half-angle phase charts, three speed charts, three indices and two incident slopes. Let q=(b_x,b_y,g,d) belong to its original compact rectangle Q. Let D be the compact weak optical domain, selecting the transmitted nonnegative-root branch at every sample. The audited uniform H,P,Z guards make the affine extension

    Fbar(x,q)=M(x)q+c(x)

well-defined and continuous on D times Q, even where traversal fails. Denote the 1200 affine traversal margins by g_s(x,q): the internal and external margins of three prisms at 200 samples. Then

    W={(x,q):x in D, q in Q, every g_s>=0}

is exactly the closure of the strict physical domain P. The outgoing critical-normal tests remain strict in P. They are weak only in W and D. Weak boundary points are limiting systems, not automatically physical solutions.

For algebraic compilation, replace the three indices by the audited h,c_2,c_3 coordinates. There are still fourteen optics; use their exact coupled prior and five-root compiler, with instantaneous interval-atom degree at most 896. The algorithm's geometric subdivision may stay in the original product x-box: its box bounds are translated through the index reconstruction equations into the reduced chart. No extra independent index variable is introduced merely to describe such a box.

All observations and fixed query tolerances must have finite exact rational/algebraic encodings. Arbitrary real-number approximation oracles do not support the equality decisions below. Algebraic degrees, embedding data, isolating intervals and coefficient heights are charged. Ordinary rational measurement endpoints satisfy this input model.

## 2. A single quantitative LP across every optical boundary

For fixed epsilon>=0 form the following affine functions of q:

    ell_i(x,q,epsilon)= +(Fbar_l(x,q)-y_l)-epsilon,
                         -(Fbar_l(x,q)-y_l)-epsilon,
                         -g_s(x,q).

There are 800 observation rows and 1200 traversal rows. Define

    v_epsilon(x)= min_(q in Q) max_i ell_i(x,q,epsilon).       (1)

This is an LP in the four q coordinates and one epigraph variable. In particular,

    v_epsilon(x)<=0
      iff some q makes (x,q) weakly physical and observation-compatible.  (2)

Equation (2) is deliberately a weak test. At v=0, strict feasibility requires the original mixed-strict system. Even v<0 does not change an optical equality R=0 into a physical ray.

Let Omega(r) be the audited global forward modulus and Phi(r) the maximum of the audited traversal moduli. Put

    Psi(r)=max(Omega(r),Phi(r)).

For x,x' in D and the same q, every ell_i changes by at most Psi(||x-x'||_infinity). Taking a maximum and then a minimum preserves this bound. Therefore

    |v_epsilon(x)-v_epsilon(x')|<=Psi(||x-x'||_infinity).       (3)

The bound is independent of epsilon. It holds through critical normal collapse and does not require the segment joining x and x' to be physical. On distances at most one, the explicit recurrence supplies C>0 with

    Psi(r)<=C r^(1/8).                                      (4)

For example combine the separate endpoint-at-one majorants for Omega and all traversal expressions. On a certified convex guarded neighborhood, direct differentiation instead supplies a Lipschitz majorant C r. A possibly nonconvex region needs a separately certified pairwise Lipschitz bound or local convex-box bounds; a derivative bound there alone is insufficient. Distances and constants refer to the stated fixed optical-chart scaling.

### Explicit dual certificate and symbolic validity interval

Write ell_i=a_i(x)^T q+b_i(x)-epsilon delta_i, where delta_i=1 on observation rows and zero on traversal rows. Write Q=q_c+product_j[-R_j,R_j]. For any algebraically encoded weights lambda_i>=0 with sum_i lambda_i=1, put

    A_lambda=sum_i lambda_i a_i(x_0),
    a_0=sum_i lambda_i b_i(x_0)+A_lambda^T q_c
                                      -sum_j R_j |A_lambda,j|,
    s_lambda=sum_i lambda_i delta_i in [0,1].

Then

    v_epsilon(x_0)>=a_0-s_lambda epsilon.                    (5)

The maximum of the right side over the simplex equals the LP optimum by finite-dimensional LP/minimax duality. A proposal need not be optimal: verifying its nonnegative weights and coefficient enclosures already gives a sound bound. Outward arithmetic supplies a conservative lower a_0 when needed.

If every x in D intersect B is within r of x_0 in D, the certificate

    a_0-s_lambda epsilon > Psi(r)                           (6)

excludes every weak and strict compatible system in B. It also handles an empty weak traversal fiber; a separate feasibility-convergence argument is unnecessary. When s_lambda>0, (6) is valid precisely on the certified interval

    epsilon < (a_0-Psi(r))/s_lambda.

Its endpoint is not excluded. When s_lambda=0 and a_0>Psi(r), the box is excluded at every noise level. A failed test proves neither feasibility nor ambiguity. At a fixed anchor v_epsilon is a convex, nonincreasing, piecewise-affine function of epsilon; fixed dual weights give reusable affine lower pieces.

Proof of (5): max_i ell_i is at least their lambda-weighted average, and the minimum of its linear q term over Q is the displayed support formula. Compact Q and the finite row set give an attained primal and dual optimum. Combining (3) and (5) proves (6).

### Sparse mixed-strict leaves and sharper endpoint tests

Use row-specific moduli omega_i(r) instead of their maximum whenever available. Every physical compatible point in B yields a geometry q satisfying the anchor outer system

    ell_i(x_0,q,epsilon)<=omega_i(r) for observation rows,
    ell_i(x_0,q,epsilon)< omega_i(r) for traversal rows,

together with the original box Q. The strict inequality follows from the original strictly positive traversal margin even when its perturbation attains the error bound. Reject this outer system using the existing mixed-strict circuit compiler: an infeasible minimal subfamily has at most five total rows, including geometry-box rows.

Equivalently, the weighted lower certificate excludes strictly physical systems when a_0-s_lambda epsilon exceeds sum_i lambda_i omega_i(r), or equals it with positive weight on a strict traversal row. Equality never establishes weak-system exclusion. This optional endpoint certificate is stronger than (6) and carries its own exact strictness flag. The recurrence gives traversal exponents 1, 1/2 and 1/4 at prisms 1,2,3; only screen rows require the global 1/8 exponent. Sparse or observation-free circuits can exploit these sharper moduli.


## 3. Candidate generation and critical-boundary treatment, in order

The following order makes useful certificates available before global decomposition; no stage is allowed to discard its unresolved complement.

1. **Compile the exact model once.** Cache positive-denominator expressions, the five-root sign circuits, the coupled index prior, every sampled observation/traversal row and exact native-coordinate reconstruction. Keep a weak predicate separately from the strict physical predicate.

2. **Try the data-only last-prism guard first.** For each sample, the audited radius test gives

       R_3/Z_3 >= max(0,[50-u_* sqrt((|y_x|+epsilon)^2
                                           +(|y_y|+epsilon)^2)]/53).

   Use the stronger phase-sector version at samples zero and one when worthwhile. A proposed positive target mu has the audited algebraic epsilon cutoff; equality remains unresolved. Then try the sixteen-corner upstream exposure certificates in order 2,1, only where downstream guards are certified. Failure means retain that region. A globally weak critical witness with residual strictly below epsilon instead proves that no positive critical margin is possible there, by certified hierarchical strictification.

3. **Generate candidates without claiming completeness from a fit.** The collision-safe degree-six spectral test uses all windows, all signs and assignments, and the exact zero/collision strata. It may exclude only with a remainder bound valid on the whole region being excluded. It may be vacuous. Formal first/second-order demixing and local corrections can propose neighborhoods; they never remove the complement. A region-dependent bound must not be exported to another region.

4. **Use complete continuation when its three hypotheses are certified.** Choose eighteen actual scalar outputs. Enumerate every projected anchor root, certify that the entire projected noise cube avoids the boundary-plus-critical discriminant, and bound the smallest projected Jacobian singular value by eta>0 on its entire preimage. The audited covering theorem then supplies every sheet, with chart radius at most sqrt(18) epsilon/eta about its anchor. Retain and filter all sheets using all 400 original observations. One anchor fit, an anchor-only noisy test, or an arbitrary selector without these certificates is insufficient.

5. **Run adaptive exact-geometry exclusion on the remaining optical domain.** Use (6), stronger guarded derivative bounds, circuit infeasibility certificates, or exact sign contradictions. At each surviving leaf compute an enclosure of the *entire* geometry fiber, not only a favored fitted geometry. If a local certificate cannot enclose all compatible geometries, retain the exact restricted problem there.

6. **Resolve every residual exceptional piece.** Use the exact reduced optical/circuit representation below, including equality strata. A prescribed finite adaptive-work budget followed by exact decomposition of the finite unresolved frontier gives an unconditional terminating algorithm. The fallback is a completeness safeguard, not the practical advance claimed in this note.

### The exceptional strata are explicit, but need not be expanded in advance

The sign data that must remain available are: critical roots R_j(k)=0; zero traversal margins; prior faces; positive-circuit rank/minor and pairing zeros; exact-data distance ranks 0,1,2; zero wedges and spectral zero/collision conditions; and, when continuation is used, the projected critical/boundary discriminant. Use lazy sign subdivision and retain realized equality cases. Enumerating every subset of 600 critical tests in advance is neither necessary nor an efficiency claim.

For physical inversion R_j(k)=0 and zero traversal margins are excluded. For compact exterior reasoning they are retained in W. On critical-normal boundaries do not invert the source transport. Keep the original four-variable affine geometry map, which remains well-defined there. No quotient by speed sign, unordered frequencies or prism permutation is applied.

## 4. Exact geometry elimination and the sixteen-variable envelope theorem

For strict optics and epsilon=0, one exact anchor sample eliminates b by the globally invertible source map. The remaining two-distance equations and inequalities have the audited rank-two, rank-one and rank-zero descriptions. They retain every source bound and strict traversal endpoint. Positive noise instead needs the two *shared* anchor error coordinates; treating them independently in different rows is only a relaxation.

For all epsilon, prefer the unanchored mixed-strict circuit compiler when eliminating geometry. Include every geometry-box, observation and traversal row. Each inclusion-minimal positive dependence has support at most five; singleton zero-normal rows and all lower-rank supports remain. The appropriate circuit pairing is weak when all supported rows are weak and strict when any supported row is strict. Enumerating supports, their determinant signs and their pairing tests is an exact finite elimination of q.

This produces a root-free predicate C(xhat,epsilon) in fourteen reduced optics and epsilon, equivalent to existence of a strict physical compatible q. Each retained optical cell is accompanied by the original explicit mixed-strict affine fiber, or the smaller exact-data distance fiber. Thus the output is a finite, all-branch parameterization by optical cells and convex geometry fibers; it is more informative than a list of fitted points and preserves continuous gauges. The explicitly parameter-dependent fiber varies with the optical point; no constant polytope per cell or full eighteen-dimensional CAD is being asserted.

### One-coordinate query compiler

For each of the eighteen *original* chart coordinates zeta:

- If zeta is a geometry coordinate, append q_j=zeta as two weak affine rows before eliminating all q.
- If it is an unchanged reduced optical coordinate, use its equality with zeta (or substitute it directly).
- If it is an original index, use the exact positive reconstruction equation. With Q_s=1+|s|^2 these are zeta^2 Q_s=h^2+|s|^2 for n_1 and zeta^2 Q_s=Q_s+c_j for n_j, together with zeta>0 and the original index bounds.

The resulting predicate C_j(xhat,epsilon,zeta) uses at most sixteen outer variables and has the exact projection

    {(epsilon,zeta):some strict compatible full18 system has coordinate zeta}. (7)

The proof is simply the exact mixed-strict circuit theorem applied to the augmented affine rows, followed by the bijective index reconstruction. It uses no rank or attainment assumption.

An epsilon-first, zeta-second cylindrical decomposition projects (7) and returns its lower and upper fiber endpoints L_j(epsilon), U_j(epsilon), including every jump and inclusion flag. These are infima and suprema even when the endpoint itself is not feasible. Finite algebraic graph conversion then makes the diameter test a three-variable calculation in (epsilon,L,U). Original index extrema are never inferred from h or c extrema alone.

There are O(K^5) candidate circuit supports, each involving at most five sampled ray traces; query rows add no traces. At K=200 there are already 2008 original affine rows including the box, so wholesale support enumeration is not the fast path. Pointwise exact LP separation returns sparse witnesses; cache and reuse those encountered supports. Complete support enumeration/sign decomposition is reserved for unresolved symbolic pieces. Lazy support discovery alone is not a proof of global completeness. Compiling each fixed-size support and substituting rotors still has degree O(K), with a large fixed compiler constant and charged coefficient heights. The single-sample bound 896 must **not** be reused as the total degree bound after these cross-sample circuit eliminations. Algebraic data need the same fixed-field or local-isolating-encoding treatment as the audited completeness theorem. Either charge the full coefficient-field degree/height, or eliminate the constant-many separately encoded observation values within each circuit before assembling the rational global predicate; an uncharged compositum of all observations is not permitted.

Consequently a conservative exact envelope route uses eighteen decompositions in dimension at most sixteen, followed by fixed-dimensional graph processing, instead of direct nineteen-variable decompositions or a 37-variable pair-product. Its schematic arithmetic bound is (sd)^(2^O(16)), with s=O(K^5), d=O(K) carrying these compiler constants and full coefficient/encoding costs. This is a dimension reduction, not a small-runtime estimate.

## 5. A conditional adaptive/output-sensitive stopping theorem

Fix an algebraically encoded epsilon. Let U be a relatively open semialgebraic subset of D supplied with validated all-fiber retained enclosures. Assume

    K_epsilon={x in D:v_epsilon(x)<=0} is contained in U.      (8)

This condition covers the entire **weak** noise fiber. Covering only strict exact solutions would not suffice. The enclosures may overlap and may contain continua; they must bound every strict compatible (x,q) above their retained region.

Start from the original optical bounding box and subdivide dyadically in all fourteen coordinates. Let h be a node's infinity-diameter. Process a node B as follows:

- If D intersect B is empty, emit an exact domain-emptiness certificate.
- If D intersect B is contained in U, emit its validated retained certificate.
- Otherwise choose an exact algebraic anchor x_B in D intersect B, solve (1), and emit (6) if v_epsilon(x_B)>Psi(h).
- Otherwise split the node.

Exact emptiness, anchor selection and containment are finite semialgebraic operations, whose costs are **not** hidden in the LP count. A numerical guess at an anchor or containment test does not establish coverage. Empty or touching boundary pieces must also be processed.

### Finite termination

Suppose Psi(h)<=C h^alpha for h<=1, with alpha=1/8 globally or alpha=1 on a certified guarded domain. Because K_epsilon is compact and contained in relatively open U, there is a relative inner region U_0 containing K_epsilon and numbers b,gamma>0 such that

    dist_infinity(U_0,D minus U)>=b,
    v_epsilon(x)>=gamma on D minus U_0.                       (9)

An empty K_epsilon may use U_0 empty and b=+infinity. An empty D is already resolved. The quantities in (9) must be certified if a numerical bound is claimed.

Put

    h_* = min(1,b,(gamma/(4C))^(1/alpha)).                     (10)

Every node of diameter h<h_* terminates. If it meets U_0, all of its D points lie in U. If it does not, its anchor has v>=gamma>C h^alpha and is excluded. Thus the total number of visited nodes is bounded, up to a dimension-dependent constant, by

    (1+D_0/h_*)^14,                                          (11)

where D_0 is the original optical root-box diameter. Initial scales above one add only finitely many charged levels. The factor four leaves room for dual lower approximations within C h^alpha; exact LP optima do not require that slack.

This proves a quantitative version of the earlier qualitative critical-boundary finite-exclusion statement. No convergence-rate assumption for a changing geometry polytope and no uniform positive critical margin are required. The gap gamma here is the **joint violation gap** in (1); it is not automatically a fixed fraction of a previously certified screen-residual gap. Relating those two gaps quantitatively would require an additional geometry error/conditioning bound. The crude global gap dependence is gamma^(-112), versus gamma^(-14) in a guarded Lipschitz region. The improvement is an explicit theorem, not favorable conditioning.

### Adaptive count

With exact optimal anchor values, define

    A_h={x in D:v_epsilon(x)<=2Psi(h),
                     dist_infinity(x,D minus U)<=h}.

Every internal, nonterminating node at scale h has all its D points in A_h: the failed exclusion and (3) prove the first inequality; failed containment provides a point of D minus U in that same box. At scales h<=1, A_h can be enlarged by replacing 2Psi(h) with 2C h^alpha. Let N_h(A_h) be the number of scale-h dyadic boxes meeting A_h. The number of visited nodes obeys

    N <= 1+2^14 sum_over_scales N_h(A_h).                      (12)

Using overlapping closed boxes changes only fixed-dimensional multiplicities. This is an output-sensitive bound in the near-unresolved geometry, not merely the number of isolated solutions. Smaller actual moduli and local dual weights sharpen it. If only approximate optimal duals are used, their certified error must be included, replacing the constant 2 accordingly.

### When the adaptive count is genuinely output-sensitive

A quantitative corollary exposes the additional structure needed. Normalize the root-box diameter to one, and suppose the nonempty weak-compatible optical set K_epsilon has certified minimal ball-cover numbers B_r(K_epsilon)<=M r^(-s), with 0<=s<=14, on the scales used. Here B_r counts infinity-norm radius-r balls, unlike the occupied-grid count N_h in (12). Suppose also the record-dependent error bound

    max(v_epsilon(x),0)>=a dist_infinity(x,K_epsilon)^beta

holds on D for certified a>0 and rational beta>0. This is a real-algebraic certificate condition after clearing the rational power. For a finite enumerated K_epsilon one can take s=0 and M its cardinality. For a continuum the covering estimate needs its own validated atlas or scale covers; dimension alone does not give a moderate constant.

Then A_h lies within t_h=(2C/a)^(1/beta) h^(alpha/beta) of K_epsilon. Cover K_epsilon at radius max(h,t_h), and cover each resulting enlarged ball by scale-h grid boxes. At sufficiently small scales this gives, up to constants depending on dimension, a,C,alpha,beta and s,

    N_h(A_h)=O(M h^(-p)),
    p=s+(14-s) max(0,1-alpha/beta).

Summing (12) down to h_* gives O(M h_*^(-p)) nodes when p>0, and O(M[1+log(1/h_*)]) when p=0, with the finite coarse levels charged separately. Thus a globally certified Lipschitz/error-bound regime with finitely many optical candidates (alpha=beta=1,s=0) has a genuine candidate-count-times-depth bound. A positive-noise compatible region often has s=14 and offers no such dimension saving. Pointwise full rank alone proves none of the global hypotheses.

Both (9) and all leaf certificates are record-dependent, finite, checkable statements. However (12) gives no small count without control of these near-unresolved sets, and it gives no bit-runtime bound without charging exact domain/containment/anchor operations and arithmetic heights. In particular, a record may have one narrow retained component yet require a hard global exclusion proof.

The theorem is for a fixed epsilon. As epsilon approaches a transition, its margins can vanish. Certificate inequalities such as (6) remain reusable on their exact epsilon intervals, but no uniform fourteen-dimensional adaptive bound over every epsilon follows automatically.

## 6. Exact outcomes, all18 accuracy and symbolic stopping

Every adaptive leaf is one of: excluded with a checkable certificate; validated retained with an all-fiber enclosure; or unresolved with its complete restricted predicate. The leaf cover is always sound. The word “complete” applies to a final inverse only after every unresolved piece has an exact cell/fiber representation. A resource-limited frontier is an enclosure, not a solved inverse.

At a specified noise level there are three exhaustive mathematical outcomes, decided by the exact fallback when the fast certificates do not suffice:

1. **Inconsistent:** the strict compatible set S_epsilon is empty.
2. **Recoverable to 0.001 in all18 native coordinates:** S_epsilon is nonempty and every native coordinate diameter is at most 0.002. The coordinatewise native midpoint guarantees the requested error for every compatible physical system.
3. **Ambiguous at that target:** two actual strict compatible systems differ by more than 0.002 in at least one original native coordinate. Their extraction is possible whenever a diameter exceeds the threshold, even if its endpoint is unattained.

The midpoint is an unrestricted point estimate and need not itself be a physical compatible system. If the output must be physical, that is a separately constrained center problem and the diameter criterion alone is insufficient.

For ordinary index/length coordinates the diameter comparison is U-L<=1/500. For angular chart extrema it is

    U-L <= b_j(1+UL),

with b_wedge=b_beam=tan(pi/90000), b_phase=tan(pi/180000), and b_speed=tan(pi/10000). These are exact algebraic constants, with their actual encoding degree charged. Thus every original coordinate, including all three separately reconstructed indices, is tested in its native units.

For symbolic epsilon let E be the nonempty-consistency interval and B the unsafe-diameter interval. The sixteen-variable projections and endpoint tests give exactly

    E=[a,infinity) or (a,infinity),
    B=[b,infinity) or (b,infinity),
    G=E minus B.

For the common 0.001 target B is nonempty and b<=||y||_infinity: the axial zero-wedge/zero-offset gauge gives separated phases with zero output. Hence G is empty, a singleton or one bounded interval. Report its algebraic endpoint encodings and whether each endpoint is included. A supremum is called a maximum only after endpoint membership is proved. At a resource-limited stage, report separate certified-safe, certified-unsafe and unresolved epsilon intervals rather than interpolating across a missing endpoint.

A weak critical or other boundary witness cannot replace a strict feasible point at residual equality. If its residual is strictly below epsilon, the audited univariate hierarchical strictification can turn it into actual physical witnesses. Two such weak points with native separation strictly above 0.002 can likewise be strictified while preserving both noise slack and excessive separation. This can produce genuine ambiguity certificates without numerical near-boundary extrapolation.

## 7. What the theorem closes, and the exact remaining obligations

**Established interface.** Every acceleration has a precise place, a strict validity interval and a sound fallback. Affine geometry is eliminated exactly, the full optical boundary is retained, adaptive exclusion has an explicit rate, and native all18 recovery is an exact envelope decision. A finite adaptive phase followed by the fixed-dimensional exact backend terminates under finite exact encodings, without genericity or conditioning assumptions.

**Highest-priority practical obligations:**

1. Obtain one qualifying paired full-vector 200-sample record, its coordinate units and bounded-error interpretation. The available canonical per-axis record is not such an input.
2. Construct and independently check one complete finite optical cover for that record: all retained fibers enclosed, every other leaf excluded, every exact/strict endpoint preserved. An attractive local fit is insufficient.
3. Establish useful record-dependent critical margins or retain/exactly solve the boundary branches; evaluate the actual moduli, dual margins and resulting tree counts. The new LP removes one missing rate, not the need to certify an exterior cover.
4. If using continuation, actually instantiate the selector/discriminant, enumerate all anchor roots and certify the whole noise tube. If using spectral proposals, certify their finite-angle remainder on each used region and retain collision/sign/assignment strata.
5. Implement sparse, shared sign circuits and exact four-variable LP/circuit fibers with certificate replay. Charge arithmetic/encoding growth and demonstrate practical total cost, including anchor/domain decisions. This is the major unproved algorithmic-performance obligation.
6. For a publication claim, establish precise novelty relative to algebraic inverse problems, exact parametric LP, certified continuation and near-collision estimation. For formal verification, formalize the full physical compiler, closure, circuit elimination and certificate checker; the nineteen checked Lean support lemmas do not yet formalize this inverse.

**Not a valid substitute for these obligations.** Exact gauges rule out a uniform positive recovery guarantee. The audited fixed-positive-amplitude, off-axis, full-rank fifth-root family rules out deriving a uniform inverse-Lipschitz constant merely from nonzero wedges and pointwise full18 rank. The sharp critical 1/8 filtered remainder and the large global absolute tail prevent promising a uniformly useful standard Prony cover. These obstructions do not prove every actual record hard; they show why a positive, quantitatively useful certificate must be genuinely record-dependent.

No conjecture of practical completion is promoted to a theorem here. The plausible implementation hypothesis is that some informative records admit small replayable trees and modest circuit/continuation certificates. Its missing evidence is exactly items 1–5 above. No record-specific success, noise threshold or runtime is asserted.
