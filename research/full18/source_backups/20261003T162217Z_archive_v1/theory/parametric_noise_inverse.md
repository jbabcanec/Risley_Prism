# Parametric observation-error theorem for the full-vector finite inverse

> **Notation scope.** This memo uses \(\theta\) for the eighteen chart coordinates and \(\kappa_j\) for the sharp native-coordinate LP sensitivity constants. These symbols are local to this memo. The main report uses \(x\) for chart coordinates and \(\chi_j\) for those sensitivity constants, reserving \(\theta\) for native parameters and \(\kappa\) for the small-wedge scale. Throughout this memo, \(\epsilon\) denotes only the symbolic observation-error bound.

## Scope and result

Fix an exact rational or algebraically encoded record y=(y_0,...,y_(K-1)), with K=200 in the original problem. Let epsilon>=0 be a single symbolic bound on each measured coordinate error, in the model’s native screen-length units (not assumed to be millimetres). The result below constructs the entire compatible family as epsilon varies, its coordinate uncertainty functions, and the exact threshold for a prescribed native-coordinate recovery tolerance. It does not require numerical noise sweeps.

The physical domain, clock, and native bounds are those of [global_inverse_completeness.md](global_inverse_completeness.md): three prisms, times k/20, strict transmitted/non-grazing/forward-traversal branches at the sampled times. Every native parameter remains unknown. Results concern unrestricted point estimation in the native mixed coordinate convention. Requiring the estimate itself to be a compatible physical system is a different optimization problem.

The principal conclusions are:

1. The compatible family is semialgebraic in eighteen chart coordinates and one shared epsilon. It is nested as epsilon increases.
2. Consistent noise levels form one upward interval. Each chart-coordinate infimum and supremum is a monotone, piecewise algebraic function of epsilon, with all endpoint and jump cases represented exactly.
3. For fixed rational native tolerance targets, the unacceptable noise levels also form one upward interval with an algebraic threshold. Acceptable levels are the difference of those two intervals, hence one interval, a singleton, or the empty set.
4. A supremal admissible noise level need not be admissible. Its endpoint must be tested; no maximum is asserted without that test.
5. Full-rank regular branches have an explicit linear small-noise uncertainty law, but extending it to global recovery requires excluding every distant branch and every boundary-approaching sequence.

## 1. Exact nineteen-variable family

Use the bijective native chart

```text
r_j=tan(pi a_j/180), f_j=tan(pi phi_j/360), v_j=tan(pi N_j/20),
s_l=tan(pi beta_l/180),
```

together with unchanged n_j,d,g,p_x,p_y. Write theta for this eighteen-vector and T(theta) for its native-coordinate inverse. The angular/speed inverse coordinates have form

```text
T_j(x)=c_j atan(x)/pi, with c_j=180,360,20,
```

respectively for wedge/beam angle, phase, and speed. The remaining coordinates use T_j(x)=x. Every chart interval for an angle-like coordinate lies strictly inside (-1,1).

Let P be the bounded chart prior restricted to the sampled-time physical branch. Let F:P->R^(2K) be the exact vector-Snell observation map. Define

```text
r_y(theta)=max_(k,l)|F_(k,l)(theta)-y_(k,l)|,
S_epsilon={theta in P:r_y(theta)<=epsilon}.            (1)
```

The explicit six-root compiler gives a rational-coefficient polynomial sign predicate Phi(theta,epsilon), with algebraic input constants encoded exactly, equivalent to (1). Epsilon is a shared free variable, not K additional variables. Thus

```text
mathcal S={(theta,epsilon):epsilon>=0 and Phi(theta,epsilon)}
```

is a semialgebraic subset of R^19. All strict branch and intersection-order tests remain strict. No closure of P is substituted for P.

The rotor formulas, branch guards, and radical compiler are exactly those proved in [explicit_six_root_compiler.md](explicit_six_root_compiler.md). Keeping epsilon symbolic changes neither the six-root count nor the instantaneous weighted degree37 bound for interval observations: residual numerators are A_(3,l)-(y_(k,l) +/- epsilon)T_3. Their root-free instantaneous degree is at most2368. After rational rotor substitution, the conservative sample-k degree bound remains2368(6k+7).

For epsilon_1<=epsilon_2, S_(epsilon_1) is a subset of S_(epsilon_2). This elementary inclusion drives all threshold statements below.

## 2. Consistency threshold and open physical boundaries

Define

```text
E={epsilon>=0:S_epsilon is nonempty},
a=inf_(theta in P) r_y(theta).                        (2)
```

P is nonempty: axial incidence, zero wedges, and the stated positive distances/indices give an admissible example. Every individual admissible system has a finite record, so a is finite and E is nonempty. By monotonicity,

```text
E=[a,infinity) or E=(a,infinity),                     (3)
```

where the lower endpoint is included exactly when some physical system attains residual a. The scalar a is a nonnegative real algebraic number for the stated exact input encodings. It is computed by projecting Phi onto epsilon and isolating the left endpoint. Whether Phi(theta,a) has a solution is a separate exact decision.

Strict physical constraints make the distinction essential. Compactness of the prior box does not imply compactness of P, and a minimizing sequence can approach an excluded grazing or traversal boundary. Such a sequence does not constitute a compatible system at epsilon=a.

Equation (2) also gives a consistency diagnostic: no inference guarantee is asserted at epsilon outside E. A vacuous universal assertion over an empty compatible set is never counted as successful recovery.

## 3. Coordinate envelopes as functions of epsilon

For epsilon in E and each chart coordinate j, set

```text
L_j(epsilon)=inf_(theta in S_epsilon) theta_j,
U_j(epsilon)=sup_(theta in S_epsilon) theta_j.         (4)
```

These are finite because the chart prior is bounded. Their values need not be attained by a physical point. Their graphs are semialgebraic. For example the lower-envelope graph is expressed by

```text
epsilon in E,
for every theta, Phi(theta,epsilon) implies theta_j>=l,
for every h>0, there exists theta with Phi(theta,epsilon) and theta_j<l+h.
```

This formula characterizes the infimum exactly, including nonattainment. Real quantifier elimination produces its graph. Alternatively, decompose the projection of mathcal S to the (epsilon,theta_j) plane and select each fiber's lower and upper endpoints, retaining their inclusion flags.

Nesting proves L_j is nonincreasing and U_j is nondecreasing. Semialgebraic cell decomposition gives a finite subdivision of E into points and open intervals on which the envelopes are continuous algebraic branches. Refining at the finitely many algebraic discriminant/critical values makes each nonconstant branch real analytic on its open interval. One way to see the latter is to take square-free polynomial equations for the one-dimensional graph, exclude their finitely many projection-critical values, and apply the implicit-function theorem. Constant branches are already analytic.

All finite subdivision endpoints are algebraic for algebraic input. At a subdivision point the envelope's actual value is obtained from its own fiber; it is not replaced with either one-sided limit. New connected components can create jumps. Strict branch boundaries can also make a limiting component appear only on one side. No continuity in epsilon is assumed globally.

A finite semialgebraic stratification of the full family, or an epsilon-adapted cylindrical decomposition, represents every cell as epsilon varies. Coordinate envelopes alone do not describe the joint correlations among parameters; the full predicate remains the complete inverse.

## 4. Native uncertainty and the optimal point estimate

Because each native coordinate map T_j is strictly increasing and continuous,

```text
l_j(epsilon)=T_j(L_j(epsilon)),
u_j(epsilon)=T_j(U_j(epsilon)),
Delta_j(epsilon)=u_j(epsilon)-l_j(epsilon).            (5)
```

Native angle and speed envelopes contain arctangents of algebraic functions. They are not claimed semialgebraic or algebraic. They are monotone, and Delta_j is nondecreasing. Their finite piecewise continuity and analytic structure follows by composition with the smooth inverse charts.

For any nonempty compatible set, the conditional worst-case native sup-norm point-estimation radius is

```text
R(epsilon)=inf_c sup_(theta in S_epsilon)||c-T(theta)||_infinity
          =(1/2)max_j Delta_j(epsilon).               (6)
```

The coordinatewise native midpoint c_j=(l_j+u_j)/2 achieves this value. Each lower bound follows from the supremal projected coordinate separation; the midpoint attains all upper bounds simultaneously. The proof does not require endpoint attainment or convexity of S_epsilon.

The angular midpoint has an algebraic chart representation. For a=L_j and b=U_j, its chart coordinate is

```text
m_j=(a+b)/[sqrt((1+a^2)(1+b^2))+1-ab].                (7)
```

The denominator is positive on these charts, and atan(m_j)=(atan(a)+atan(b))/2. Thus the entire minimax center can be represented parametrically by semialgebraic chart functions, even though its displayed native angular values involve arctangents. It need not lie in S_epsilon.

For prescribed positive native tolerances delta_j, success means Delta_j(epsilon)<=2 delta_j for every j. This is a sharp information-theoretic criterion for the given record and bounded-error model, not merely a sufficient local conditioning bound.

## 5. Exact algebraic fixed-tolerance comparison

For an unchanged coordinate, two candidate values x,x' violate the diameter target precisely when |x-x'|>2delta_j.

For an angle-like native coordinate c_j atan(x)/pi, define

```text
b_j=tan(2pi delta_j/c_j),                              (8)
```

when the angular target lies below pi/2. Since 1+xx'>0 on the entire chart and |atan x-atan x'|<pi/2, the violating comparison is exactly

```text
|x-x'|>b_j(1+xx').                                   (9)
```

Absolute value is represented by two polynomial inequalities; the constants b_j are algebraic when delta_j is rational. Larger target angles are trivially satisfied on these prior charts and can be omitted.

For the specific common target delta_j=0.001 in each original native coordinate,

```text
b_wedge=b_beam=tan(pi/90000),
b_phase=tan(pi/180000),
b_speed=tan(pi/10000),
```

and unchanged-coordinate separation is compared with1/500. These exact constants can be represented by integer polynomials and isolating intervals. Their algebraic degrees and bit costs are part of the input/constant cost; they are not assumed small because0.001 has a short decimal representation.

This fixed-threshold algebraization does not imply that an optimized native uncertainty radius R(epsilon) is algebraic or semialgebraic.

## 6. The exact maximum-admissible-noise construction

Let V_delta(theta,theta') be the disjunction of the violating comparisons in Section5, one per native coordinate. Define

```text
B={epsilon>=0:exists theta,theta',
   Phi(theta,epsilon) and Phi(theta',epsilon) and V_delta(theta,theta')}. (10)
```

B is the set of error budgets under which two compatible systems are too far apart for the requested point-estimation accuracy. It is an upward semialgebraic set, because both witnessing systems remain compatible at every larger epsilon. If nonempty, it is [b,infinity) or (b,infinity), where

```text
b=inf_(theta,theta' in P:V_delta(theta,theta'))
     max(r_y(theta),r_y(theta')).                    (11)
```

It is also possible in general that B is empty, in which case set b=+infinity. Finite b is algebraic and computed by eliminating the two eighteen-variable system blocks in (10), retaining epsilon. Inclusion at b is decided by testing (10) at that exact algebraic endpoint.

The acceptable-noise set is exactly

```text
G=E minus B.                                        (12)
```

Consequently G is empty, a singleton, or one interval with endpoints among a,b and0, with each endpoint included or excluded according to the exact predicates. It cannot contain disjoint safe intervals. Since B is a subset of E, b>=a when B is nonempty.

Algorithm:

1. Construct Phi(theta,epsilon) once, retaining epsilon symbolically.
2. Project exists theta Phi to obtain E and its lower-endpoint status.
3. Project the two-system formula (10) to obtain B and its lower-endpoint status.
4. Form the exact set difference G=E minus B.
5. If G is empty, report that no consistent noise level guarantees all requested coordinate tolerances.
6. If G is nonempty and bounded above, return epsilon_sup=sup G, its algebraic representation, and whether epsilon_sup belongs to G. It is a maximum admissible error only when the membership test succeeds. Otherwise all allowable error budgets are strictly below the returned supremum, subject to the lower consistency endpoint.
7. If G is unbounded, report that explicitly rather than inventing a finite maximum.

At any epsilon in B, witness extraction returns two actual physical systems violating some diameter target. At epsilon not in E it returns inconsistency. At epsilon in G, it supplies the minimax center/envelopes and the sharp conditional guarantee. Neither an excluded physical-boundary point nor an unattained envelope is substituted for an actual ambiguity witness.

The boundary distinctions are real: a new distant branch admitted exactly at b makes b unsafe; a branch existing only above b can leave b safe. If a=b, the safe set can be a singleton or empty, depending on endpoint inclusion. No blanket continuity argument resolves these cases.

## 7. Gauges give a finite upper threshold for0.001 targets

The physical axial, zero-wedge, zero-offset system produces the identically zero screen record for every allowed index, gap, and distance. Its phases and rotation speeds do not change that record. In particular there are two such admissible systems whose first phases differ by36 degrees.

For any fixed y, these systems both belong to S_epsilon whenever

```text
epsilon>=||y||_infinity.
```

Therefore for the0.001 target, B is nonempty and

```text
0<=a<=b<=||y||_infinity.                             (13)
```

The threshold construction is always finite in this particular target problem, although G can be empty. If y is the zero record, B contains0 and G is empty: no measurement precision, including exact noise-free data, resolves the stated native parameters. This rules out a positive universal observation-noise threshold over the whole prior box.

Other exact gauges can produce a positive limiting uncertainty even for nonzero records. The full parametric construction includes them; it does not force a local inverse onto a nonidentifiable fiber.

## 8. Regular-branch linear sensitivity and geometry conditioning

The following is a local theorem with explicit hypotheses. Let theta_0 be an interior prior point with strict optical/traversal margins, let y=F(theta_0), and suppose the m-by18 chart Jacobian J=DF(theta_0), m=2K, has full column rank. Restrict to a sufficiently small neighborhood U containing no other zero-residual point. The exact forward map is analytic there and has a uniform quadratic Taylor remainder.

Define the symmetric compact linear uncertainty polytope

```text
K_J={h:||Jh||_infinity<=1}.
```

For native coordinate j let c_j be the jth row of DT(theta_0). Define

```text
kappa_j=max_(h in K_J)|c_j h|
       =min_(lambda:J^T lambda=c_j^T)||lambda||_1.    (14)
```

The second identity is linear-programming duality; full column rank ensures boundedness and dual feasibility. These constants account for native angle/length/index units, not merely unscaled chart singular values. DT is diagonal, with entries c/(pi(1+x^2)) for an angle-like coordinate and1 for unchanged coordinates.

As epsilon decreases to0, the local compatible set has scaled displacement set converging to K_J, with an O(epsilon) Hausdorff error. Indeed the Taylor remainder gives ||F(theta_0+epsilon h)-y||/epsilon=||Jh+O(epsilon||h||^2)||; injectivity gives a uniform bound on h for local feasible points, while a1-O(epsilon) contraction of K_J satisfies the nonlinear inequalities. Therefore

```text
Delta_j^local(epsilon)=2 kappa_j epsilon+O(epsilon^2),
R_local(epsilon)=epsilon max_j kappa_j+O(epsilon^2).  (15)
```

This law is conditional on an exact central record and a regular interior branch. For a nonexact record at a positive minimum-residual threshold, boundary activation can instead produce other powers or jumps; no universal linear law is asserted there.

For the optical/geometry split theta=(x,q), with x in R^14 and q=(p_x,p_y,g,d), write the stacked exact map as

```text
F(x,q)=M(x)q+c(x),
J=[A M], A=partial_x F(theta_0).
```

Full column rank of J implies rank(M)=4. Let

```text
Pi=I-M(M^T M)^(-1)M^T,
S=A^T Pi A.
```

Then S is positive definite exactly when the remaining fourteen optical directions are identifiable after fitting geometry. This is the Schur complement of M^T M in J^T J. If e=J h, then

```text
h_x=S^(-1) A^T Pi e,
h_q=(M^T M)^(-1)M^T(e-Ah_x).                         (16)
```

With sigma_x=sigma_min(Pi A)>0 and sigma_q=sigma_min(M)>0, the linear bounded-error set satisfies

```text
||h_x||_2<=sqrt(m)/sigma_x,
||h_q||_2<=sqrt(m)(1+||A||_2/sigma_x)/sigma_q.         (17)
```

These bounds can be converted to native units using DT. The exact LP constants (14) can be sharper. The Schur complement detects optical information that cannot be absorbed by re-fitting the affine geometry; the geometry matrix detects how reliably that fitted geometry is recovered.

For finite epsilon, geometry profiling remains an exact four-dimensional mixed-strict LP conditional on x and epsilon. The observation-band endpoints depend affinely on epsilon. The profiled geometry-coordinate extrema can instead be piecewise affine as active constraints change. The previous Helly/closure-vertex constructions extend with epsilon as one additional optical-side variable. The linearized Schur formulas do not replace this exact global feasibility test.

## 9. What is required to promote local sensitivity to global recovery

The local law (15) describes all compatible systems only if every compatible system lies in U. A sufficient condition is the certified exterior residual gap

```text
rho=inf_(theta in P outside U) r_y(theta)>0.          (18)
```

For0<=epsilon<rho, no exterior system is compatible. The gap itself is expressible and computable through the global semialgebraic backend when U has algebraic boundaries.

A unique exact compatible system and a full-rank Jacobian do not alone prove (18). P is open at physical boundaries; distant admissible sequences can approach an excluded boundary while their residual tends to0. A compact guarded physical domain together with global uniqueness would remove this particular escape route, but such guards must be stated and justified rather than silently added to the model.

A singular or positive-dimensional exact fiber may have uncertainty that stays positive as epsilon tends to0. Even an isolated singular solution can produce fractional-power sensitivity. The global envelopes and their branch/end-point data remain valid in all these cases. The regular local law is an optional interpretation and computational accelerator, never grounds for discarding an unverified branch.

## 10. Complexity, sources, and actual computability

For rational y of bit size tau, the compiled family has s=O(K) fixed-compiler polynomial occurrences, degree d=O(K), and coefficient bit height O(K+tau), with the explicit large constants from the six-root compiler. A full compatible-family decomposition has dimension19, rather than18 for fixed epsilon. Conservative CAD arithmetic complexity has form

```text
(sd)^(2^O(19)),
```

with the corresponding coefficient-bit factor. Fixed algebraic targets add their explicit degree/height costs. The unacceptable-set projection uses two system copies plus epsilon, dimension37, so a safe full CAD bound for that direct route is

```text
(sd)^(2^O(37)),
```

again with coefficient arithmetic charged. It would be misleading to claim that the pairwise ambiguity query itself has only19 variables. The37-variable construction is not necessary. There is a sharper coordinate-envelope route, with at most nineteen-dimensional CAD stages:

- For each j, build a CAD with epsilon first and theta_j second. Project every retained full cell to the first two coordinates. Mark a projected cell when at least one retained full cell lies over it. Cylindricity makes this exactly the projected compatible set, with no missing fibers.
- Over each epsilon base cell, find the lowest and highest marked coordinate cells. Their finite boundary sections give L_j and U_j, even when those boundary sections themselves are excluded from the compatible set. Bounded prior intervals ensure these extrema are finite. At zero-dimensional epsilon cells use that exact fiber, retaining jumps.
- Represent the resulting two-dimensional envelope graphs by polynomial sign predicates, using Thom/root encodings with their standard fixed-dimensional conversion or refinement. This is still an exact real-algebraic operation and its output-size cost is included.
- For each j, construct in just three variables(epsilon,l,u) the conjunction of the lower graph, upper graph, and the appropriate algebraic diameter-violation comparison. Project l,u, then take the union over j. This recovers exactly B, including cases of unattained extrema, because diameter strictly exceeding the target guarantees a pair of actual points with excessive separation.

Thus a complete threshold algorithm can use eighteen nineteen-variable decompositions followed by fixed low-dimensional graph processing; it need not form the37D pair-product. Formula and coefficient growth from graph conversion must be included, but fixed dimension preserves a bound of the same general form(sd)^(2^O(19)) with adjusted fixed constants and polynomial bit factors. The37D estimate is a conservative direct alternative, not the best bound proved here.

The number of variables and prism count are fixed, so these are polynomial-in-K/encoded-input-size theoretical bounds with extremely large fixed exponents. The arbitrary-algebraic-record encoding caveats from global_inverse_completeness.md remain in force; local encoding/elimination avoids silently adjoining K unrelated algebraic numbers to an uncharged common field. Rational-native target constants also have genuine algebraic-degree costs.

General real-algebraic elimination and decomposition guarantees are given in Saugata Basu's author-hosted survey, Theorems2.4 and2.16, including integer-height control: https://www.math.purdue.edu/~sbasu/raag_survey2011.pdf. Finite interval subdivision for definable functions is stated and proved in Michel Coste's author-hosted notes, Theorem2.1: https://perso.univ-rennes1.fr/michel.coste/polyens/OMIN.pdf. The threshold, nesting, minimax, LP-dual, and Schur statements here follow from the displayed constructions and proofs.

What is completed is the mathematical symbolic-epsilon characterization and an exact terminating threshold algorithm under the stated encodings. A numeric threshold requires an actual record and execution of a certified solver. No such numerical value, practical runtime, or completed full200-sample implementation is asserted. Exact zero/equality decisions for arbitrary-real oracle inputs remain unsupported. The model's physical validity between samples is still a distinct requirement.

## 11. Complete singular-endpoint behavior: Puiseux powers, floors, and jumps

Let epsilon_0 be any finite algebraic endpoint or algebraically chosen subdivision value (in particular, the decomposition values of Section 3), and choose one side on which E contains an interval. Put h=|epsilon-epsilon_0|>0 on that side and restrict to a sufficiently short interval without another subdivision point. Every chart envelope is then a bounded continuous semialgebraic function of h.

The exact real-algebraic fact used here is that a one-variable semialgebraic function, after restricting to a sufficiently short one-sided interval, is represented by an algebraic Puiseux series; over the real numbers that series converges. A direct primary author source is Michel Coste, Real Algebraic Sets, Chapter1, discussion of algebraic Puiseux series on page10: https://perso.univ-rennes1.fr/michel.coste/polyens/RASroot.pdf. That discussion identifies the graph with a real algebraic curve branch and explains convergence after substituting h=z^q. The same convergence and germ statements were verified in Coste's ICTP-hosted notes, Chapter 1, printed page 10: https://indico.ictp.it/event/a02455/session/23/contribution/14/material/0/0.pdf. The same semialgebraic-germ/algebraic-Puiseux identification is recorded in Barone and Basu, Section2.5: https://www.math.purdue.edu/~sbasu/refined-04-06-11.pdf.

Consequently, after a common positive integer ramification q, the envelopes have convergent expansions

```text
L_j(epsilon)=ell_j+sum_(n>=1) a_(j,n) h^(n/q),
U_j(epsilon)=u_j+sum_(n>=1) b_(j,n) h^(n/q).          (19)
```

Zero coefficients are allowed. Boundedness excludes negative powers. Their one-sided limits ell_j,u_j are finite, satisfy ell_j<=u_j, and are algebraic for the exact algebraic input model. Neither limit is automatically the value of an attained physical solution or the envelope value at epsilon_0.

There are exactly three coordinatewise asymptotic cases:

1. Positive floor: u_j>ell_j. Then the native uncertainty tends to the positive value T_j(u_j)-T_j(ell_j). Exact gauges can produce this case. Distant near-aliases approaching an excluded boundary can also produce it even if an exact endpoint fiber is unique or empty.
2. Identically zero uncertainty: U_j-L_j is identically zero on the chosen side. That coordinate is fixed throughout those fibers.
3. Shrinking, nonzero uncertainty: ell_j=u_j=z_j and U_j-L_j is not identically zero. Its convergent Puiseux expansion has a first nonzero term

```text
U_j-L_j=A_j h^(nu_j)+O(h^(nu_j+eta_j)),              (20)
```

with A_j>0 and positive rational nu_j,eta_j. Since T_j is analytic near z_j and T_j'(z_j)>0,

```text
Delta_j(epsilon)=T_j'(z_j) A_j h^(nu_j)
                +O(h^(nu_j+eta'_j)),                (21)
```

for some positive rational eta'_j. To prove preservation of the exponent, factor the difference as

```text
T_j(U)-T_j(L)=(U-L) integral_0^1 T_j'(L+t(U-L)) dt.
```

The second factor has positive limit T_j'(z_j) and a convergent Puiseux expansion. Thus native arctangent charts preserve the leading shrinking exponent, while possibly introducing nonalgebraic coefficients such as1/pi.

For the global minimax radius, a positive coordinate floor gives a positive limiting radius. If all floors vanish and at least one coordinate varies, the radius has a leading positive rational-power law: take the smallest nu_j and the largest associated leading half-width coefficient. The finite maximum is eventually determined by its leading Puiseux comparisons. If all widths vanish identically, the radius is zero on that side.

Actual endpoint values must still be tested separately; they can differ from one-sided limits, yielding jumps. These alternatives cover the singular one-parameter behavior permitted by the exact semialgebraic inverse. They exclude an unsupported universal O(epsilon) assertion: a singular isolated solution may have exponent below1, while a gauge may have a nonzero floor. The relevant exponents and leading chart coefficients can in principle be computed from the selected algebraic envelope branches by Newton-Puiseux methods and exact branch isolation. No small universal exponent or practical extraction cost is claimed.

## 12. Explicit matching regular upper and lower bounds

Here is a quantitative version of Section 8, specifying the assumptions behind both sides of the first-order law. Use the Euclidean norm in chart space and the sup norm in observation space. Assume:

- The closed chart ball of radius R around theta_0 lies strictly inside the prior and physical branch. In particular every square-root radicand, positive axial/normal direction, and traversal-order margin has a positive lower bound on this ball.
- y=F(theta_0), J=DF(theta_0) has full column rank, and

```text
gamma=min_(||h||_2=1)||Jh||_infinity>0.
```

- A finite Hessian bound H is certified on the ball, giving

```text
||F(theta_0+h)-y-Jh||_infinity <= (H/2)||h||_2^2.
```

- Shrink R if necessary so H R<=gamma.
- If a global assertion is desired, a certified exterior residual gap rho>0 holds outside the open radius-R ball, as in (18).

Strict branch margins and compactness of this ball make derivative bounds finite; they do not supply their numerical values automatically. With exact algebraic center and guards, bounds may themselves be certified by real-algebraic optimization or validated interval estimates. Let

```text
C_0=H/(2gamma^2),
```

and define the local compatible set explicitly by

```text
S_epsilon^R = S_epsilon intersect closed_ball(theta_0,R).
```

Choose any epsilon satisfying

```text
0<=epsilon<=gamma R/2,
C_0 epsilon<=1/2.                                    (22)
```

The final inequality is automatic when H=0. With K_J from Section 8, the local compatible set always satisfies

```text
(epsilon-C_0 epsilon^2)K_J
 subseteq S_epsilon^R-theta_0
 subseteq(epsilon+4C_0 epsilon^2)K_J.                 (23)
```

If the exterior residual gap is certified and epsilon<rho, every globally compatible point lies inside the open radius-R ball. In that case S_epsilon^R=S_epsilon, so the same sandwich is global. Without that gap, no conclusion about compatible points outside the certified ball is implied.

Proof of the upper inclusion. By definition every point of S_epsilon^R lies in the certified closed ball. The Taylor lower bound and H R<=gamma give

```text
r_y(theta_0+h)>=gamma||h||_2-(H/2)||h||_2^2
             >=(gamma/2)||h||_2.
```

Hence ||h||_2<=2epsilon/gamma. Substitution into the remainder bound gives

```text
||Jh||_infinity<=epsilon+2H epsilon^2/gamma^2
                =epsilon+4C_0 epsilon^2.
```

This is exactly the upper inclusion.

Proof of the lower inclusion. If z in K_J then ||z||_2<=1/gamma. Put h=(epsilon-C_0 epsilon^2)z. Conditions (22) keep h inside the certified ball. Its residual is bounded above by

```text
epsilon(1-C_0 epsilon)+C_0 epsilon^2(1-C_0 epsilon)^2<=epsilon.
```

Thus both positive and negative extremal directions of K_J are represented by actual compatible systems, proving a matching lower bound rather than only a Lipschitz upper estimate.

For each native coordinate assume also the certified bound

```text
|T_j(theta_0+h)-T_j(theta_0)-c_j h| <= (B_j/2)||h||_2^2
```

on this ball. Here c_j=DT_j(theta_0); B_j=0 for unchanged coordinates, and the arctangent second derivative gives an explicit finite B_j for angle-like coordinates. Let r_j^R(epsilon) be half the native coordinate diameter of S_epsilon^R, and let kappa_j be the exact primal/dual LP value in (14). From (23) and the two-sided native Taylor bound,

```text
kappa_j epsilon-(C_0 kappa_j+B_j/(2gamma^2))epsilon^2
 <= r_j^R(epsilon)
 <= kappa_j epsilon+(4C_0 kappa_j+2B_j/gamma^2)epsilon^2.              (24)
```

The lower inequality uses an LP maximizer z and its negative; these exist because K_J is compact. The upper inequality bounds every feasible h. Thus

```text
r_j^R(epsilon)=kappa_j epsilon+O(epsilon^2)
```

with explicit, certified remainder constants and a stated admissible epsilon range. The corresponding local sup-norm minimax radius has leading coefficient max_j kappa_j. These are also the global diameters and minimax radius when the exterior gap is certified and epsilon<rho. This proves the requested geometry-dependent linear behavior exactly where its hypotheses hold, while Sections 6, 7, 9, and 11 handle global branches, gauges, strict boundaries, and singular power laws.


The leading LP sensitivity constants are sharp. The displayed quadratic remainder constants are valid sufficient bounds; their optimality is not asserted.
