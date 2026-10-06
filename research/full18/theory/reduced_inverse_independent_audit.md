# Independent audit: reduced inverse and exact matched-jet obstruction

Audit date: 2026-10-03. This audit checks the mathematical claims in `reduced_algebraic_inverse_addendum.md` and `finite_record_collision_obstruction.md` against the existing exact physical ray model. No parameter sweep or broad numerical test was used. The accompanying `proof_checks/audit_reduced_inverse_exact.py` contains exact rational checks, all passed.

## 1. Reduced algebraic inverse: pass

The three-index change is a bijection on the complete original prior when its coupled inequalities and positive sign for h are retained. There remain fourteen optical coordinates and eighteen total coordinates. The image is not a box. Original index extrema must be computed through the inverse algebraic expressions, not through h or c-coordinate extrema alone.

The scaled directions preserve the original physical branch. The five-root tower is valid. The combined spatial numerator is exactly obtained by expanding the two intersections and using P+u·X=H. Assigning the stated root weights gives numerator/denominator budgets (28,27). Consequently the terminal instantaneous interval degree is at most896, with at most243 sign-circuit leaves per original atom. The conservative K=200 degree is1,076,096. These are upper bounds, not practical algorithm timings.

For actual incoming internal slope a and outgoing slope v, the paired source map is

A=(I-vu^T)(I-au^T)^(-1),

det A=(1-u·v)/(1-u·a)=H R/(Z P)>0.

All signs used here are original strict branch signs. Thus any exact sample eliminates both source offsets globally within that branch. The result is sixteen variables before further profiling: fourteen optical and two distances. The remaining distance equations must retain every rank-zero, rank-one and rank-two stratum, every transformed source bound and every strict traversal inequality.

Positive noise requires two shared anchor residuals and restores eighteen variables. Eliminating those residuals separately row by row is merely an outer relaxation. The draft labels this correctly.

The mixed-strict positive-circuit theorem is correct. In particular, if the weak polyhedron is feasible but its strict rows cannot all be satisfied, one marked row is equality throughout the weak polyhedron. LP duality yields a nonnegative zero-pairing dependence with positive coefficient on that marked row; its decomposition contains a zero-pairing circuit touching a strict row. This proves the strict part without assuming an interior feasible point. Singleton zero-normal circuits and lower-rank supports are necessary and correctly retained.

The threshold formula correctly separates permanent zero-D conditions from positive-D ratios. The lower endpoint is excluded precisely when a strict positive-D circuit attains the final maximum with zero included. A strict circuit attaining a smaller ratio does not exclude that endpoint.

A sample-dependence bookkeeping qualification was supplied to the author: three anchored exact rows can depend on four distinct sample traces, and five anchored noisy rows can depend on six. Five unanchored original geometry rows depend on at most five traces. These remain fixed-size counts; they must not be conflated when quoting root-elimination constants.

No uniform inverse conditioning, quantitative global optical cover or practical complete inverse is established by these reductions.

## 2. Matched-jet obstruction: pass for the stated small-amplitude family

The construction uses nonzero offset b=1 and axial beam direction, with interior hardware and positive wedges. The exact all-angle physical margins follow from the stated elementary inequalities; the rational audit checks conservative lower bounds for internal/normal quantities and traversal heights.

The six rational weights, their zero moments of orders0 through4, their fifth moment -D, and B0=D/120 were independently recomputed exactly. The low-degree nonresonance proof is valid for both interlaced triples; the original20 Hz clock introduces no further alias. The available31-column confluent finite-record rank theorem therefore applies at every fixed positive h, for sufficiently small amplitude.

The five complex correction equations are ten real equations. Their limiting real Jacobian is the realification of an invertible complex Vandermonde matrix. This proves a real-analytic implicit solution; it does not require the nonlinear optical map to be holomorphic. Complex corrections supply both wedge and phase freedom.

The corrected output difference has exactly vanishing slow-time jets0 through4. Comparing it to its A=0 divided-difference limit and differentiating along the actual correction gives the fifth-order bound with no additive optical-tail floor. The derivative of the correction must remain included; the draft does so.

The h=.003 native phases satisfy the exact centered bound13.4996625 degrees before the less-than4-degree correction. All phases remain strictly within the original18-degree limit. The native speed differences are(.003003,.003015,.003027) Hz, so the two-point worst-case speed error is at least.0015135 Hz. The exact rational pi<22/7 estimate proves the stated coefficient7.3×10^-6 multiplying A in the midpoint noise allowance.

A_star, the nonlinear derivative constants and any rank cutoff remain unevaluated. Thus the construction is an exact, quantitatively normalized existence theorem, not a certified noise limit for a supplied dataset or a selected practical wedge size.

## 3. Fixed-amplitude regularity strengthening

The following strengthening is valid. Choose a single positive A small enough for the correction on a compact h interval and for both full18 ranks at one positive h0. The unique corrected parameter paths are real analytic in h on an open neighborhood of the interval, including h=0. Fix one nonzero18-by18 sampled-Jacobian minor for each system at h0. Each minor along its corrected path is a nonidentically-zero one-variable real-analytic function. Its zero at h=0, if present, has finite order. Therefore both minors are nonzero for every sufficiently small positive h.

Consequently the fixed-A fifth-root modulus obstruction can be restricted to full18-regular systems for sufficiently small noise while all wedge amplitudes stay bounded away from zero. This supplies no uniform lower bound for the minors, no uniform conditioning, and no evaluated punctured-interval cutoff. Those distinctions are essential.
