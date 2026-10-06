# Hard-noise bounds for every compatible full18 system

2026-10-02. This note derives a certifiable search target and contractors for the
original passive, single-plane, 200-sample problem. It does **not** claim that the
full-prior search has been completed. No additional measurements, known unknowns,
minimum wedge, speed separation, or smaller prior are introduced. The original
physical prism order is retained.

## 1. Exact problem, arithmetic contract, and audit result

Write the native parameter vector as

\[
 \theta=(N_1,N_2,N_3,a_{x1},a_{x2},a_{x3},a_{y1},a_{y2},a_{y3},
 n_1,n_2,n_3,d,g,b_x,b_y,p_x,p_y)\in B_0\subset\mathbb R^{18}.
\]

The original closed native box is: speeds in [-3.5,3.5] Hz; wedge/phase angles
in [-18,18] degrees; indices in [1.3,1.8]; distance in [50,200]; gap in [2,15];
beam angles in [-25,25] degrees; positions in [-5,5]. Lengths use the original
model's position units: this note does not assign millimetres. Let P be the
subset satisfying every strict optical branch condition. Let F map to all
400 scalar positions. Use either the exact clock t_k=k/20, k=0,...,199, or a
specified stored binary clock; record the choice. They are not identical real
inputs. A proof using one cannot silently certify the other.

For the actual stored observation vector y and supplied componentwise error
allowances eta_i >= 0, define the entire compatible set

\[
 C=C_\eta(y)=\{\theta\in B_0\cap P:\ |F_i(\theta)-y_i|\le\eta_i,
                  \quad i=1,\ldots,400\}.
\tag{1}
\]

All inequalities below concern this set, not a distribution of noise. A scalar
eta means eta_i=eta. If y was produced by a floating implementation with clamps
or a different clock, its discrepancy from F must be included in eta or bounded
separately. A least-squares RMS or MSE below eta is not the hard-band condition
in (1). Exclusion can use any subset of rows, but a compatible witness must
check every row and all physical guards.

The read-only audit in [../AUDIT.txt](../AUDIT.txt) found the saved frozen combined
counts correct: 7/10 actual native-coordinate passes without added noise and
6/10 at the tested deterministic 1e-4 pattern. Seven noisy outputs fit the saved
hard band, so fitting and 0.001 recovery already differ in these ten outputs.
These are finite experimental counts, not uniform guarantees. Source hash
claims cover their named files, not all dependencies. The final arithmetic in
the earlier profile checker, `abs(fm-y)+fr`, needs outward rounding before its
scalar is called a rigorous upper bound. The independently directed interval
common-record witnesses are unaffected. See section 8.

Original implementation references: `risley_lattice/fmodel.py:44` (failed
margin bound versus proved violation), `:116` (point enclosure), `:151` (first
derivatives), `:200` (second derivatives); `certify_det2.py:1-54` (the local
second-order argument), `:181` (local box search). The implementation treats
stored floating timestamps as exact data at `fmodel.py:170-171`.

## 2. What constitutes a proof of 0.001 recovery

Assume C is nonempty. Define coordinate infima and suprema, not necessarily
attained,

\[
 \ell_j=\inf_{\theta\in C}\theta_j,\qquad
 u_j=\sup_{\theta\in C}\theta_j.
\]

**Theorem 1 (native minimax radius).** If estimates may be arbitrary real
vectors, then

\[
 \inf_z\sup_{\theta\in C}\|z-\theta\|_\infty
       =\frac12\max_j(u_j-\ell_j).
\tag{2}
\]

Proof: for each coordinate, no scalar z_j can be closer than half its range
to both extremes (or sequences tending to them). The midpoint
z_j=(ell_j+u_j)/2 attains all those coordinate bounds together. The maximum
and supremum commute because there are finitely many coordinates. This uses
absolute native units, not prior-normalized units or a Euclidean norm.

For a prescribed estimate z the exact worst-case coordinate error is
max(|z_j-ell_j|,|z_j-u_j|). Thus a sound outer cover

\[
 C\subseteq U=\bigcup_{B\in\mathcal L}B
\tag{3}
\]

gives immediately checkable upper bounds

\[
 E_j(z)=\max_{B\in\mathcal L}
       \max\{|\underline B_j-z_j|,|\overline B_j-z_j|\}.
\tag{4}
\]

If all E_j(z)<=0.001, this z has that error bound against **every** compatible
system in the full original prior. For the original strict target use
E_j(z)<0.001. Every retained box must be included, including unresolved
physical-boundary boxes and disconnected components. Small individual boxes
around several distant solutions do not satisfy (4).

The midpoint of coordinate ranges lies in the native rectangular prior but
need not be physically admissible or compatible. If the requested deliverable
is a physically valid fitted hardware vector, use an independently verified
candidate z and test (4) around that candidate. A half-range claim alone does
not establish the existence of a physical center achieving it. If C is empty,
the proper result is inconsistent data/model/error allowance, not recovery.

**Corollary (two-system obstruction).** If a,b are both compatible with the
same actual record, every possible estimate has error at least
|a_j-b_j|/2 in coordinate j for at least one of the two possible systems.
In particular, |a_j-b_j|>0.002 refutes universal 0.001 accuracy on that record.
This needs only two witnesses; it does not enumerate C. Coordinate-wise
adversarial endpoints can differ, so these are not simultaneous error lower
bounds for one chosen endpoint.

## 3. Sound interval contractors, including rank-deficient boxes

Take a box B=c+H inside a single certified smooth optical branch, with
H=[hlo,hhi] and c in B. The full convex box must be in that branch; checking
only its center or presumed solutions does not justify Taylor's theorem along
every segment. Let fbar and Jbar be fixed reference numbers and obtain outward
interval enclosures V containing F(c)-fbar, D containing J(c)-Jbar, and
H_i(B) containing the Hessian of F_i on B. Here H is a displacement box and
H_i is a Hessian interval matrix; the different uses are explicit below.

Taylor's integral remainder gives, for every h in the displacement box,

\[
 F_i(c+h)=\bar f_i+(\bar J h)_i+R_i(h),\qquad
 R_i(h)\in V_i+D_i H+\tfrac12 H^T[\mathcal H_i(B)]H=: [R_i].
\tag{5}
\]

The quadratic term denotes any sound range enclosure of the indicated
quadratic form; interval multiplication is sufficient, though often loose.
The 1/2 follows from integrating 1-t on [0,1]. For a symmetric box of radii r,
one may use the elementary bound |R_i|<=rad(V_i)+sum_j sup|D_ij| r_j+
0.5 sum_jk sup|H_ijk| r_j r_k when V_i is centered at zero.

Every compatible h satisfies the finite linear strips

\[
 -\eta_i-(\bar f_i-y_i)-\overline R_i
 \ \le (\bar J h)_i\le
 \eta_i-(\bar f_i-y_i)-\underline R_i,
 \qquad h\in H.
\tag{6}
\]

An empty rigorously checked outer polytope excludes B. Certified linear
coordinate minimization/maximization contracts B. Inconclusive linear programs
or singular Jbar do not justify rejection. (6) remains valid without full
rank: it can exploit identifiable directions while retaining a continuum in
others. It is an outer relaxation, so LP feasibility does not prove actual
optical feasibility. LP numerical output must be checked by interval/exact
dual bounds, with the variable box used to cover imperfect dual equalities.
The companion [elimination.md](elimination.md) develops those certificates
directly for the exact four affine geometric variables.

### Signed directional remainder: tighter profiles than absolute Hessians

For a desired native direction v and any fixed multiplier lambda in R^400,
compute a sound **signed** enclosure

\[
 [R_\lambda]\supseteq
 \{\lambda^T(F(c+h)-\bar f-\bar J h):h\in H\}.
\]

Compute the Hessian of the combined scalar lambda^T F before taking absolute
values, or combine reference Hessians with outward product-error coverage.
This retains cancellation that sum_i |lambda_i| |H_i| destroys. Denote the
support function of a box by
s_H(w)=sum_j max(w_j hlo_j,w_j hhi_j). Compatibility implies

\[
 \boxed{\quad v^T h\le
 -\lambda^T(\bar f-y)+|\lambda|^T\eta
 +s_H(v-\bar J^T\lambda)-\underline R_\lambda.\quad}
\tag{7}
\]

Proof: write F(c+h)-y=e, |e|<=eta, and substitute
v^T h=lambda^T(e-(fbar-y)-R(h))+(v-Jbar^T lambda)^T h.
Bound the three variable terms separately. Apply (7) to -v for a lower bound.
No exact left inverse or full-rank assumption is needed. The residual term
s_H(v-Jbar^T lambda) is essential when lambda is merely an approximate dual.

For native coordinate profiles set v=+e_j and v=-e_j. A floating optimizer can
propose lambda, a Hessian reference, or an orthogonal change of coordinates;
the verifier then treats the submitted numbers as fixed and verifies (7).
Taking minima across several valid upper bounds is safe. A signed quadratic
range can be bounded by interval arithmetic, verified positive semidefinite
factorization, or a certified convex relaxation. For example, a quadratic
remainder (h1-h2)^2 is nonnegative; replacing its expanded coefficients by
absolute values would lose precisely that useful fact.

After processing all retained boxes, take the minimum of their certified
coordinate lower bounds and the maximum of their upper bounds. These are
global outer profile limits only if (3) was already established. Optimizing
profiles near one fit does not establish global coverage.

### Noise-set Krawczyk contractor and pairwise diameter

For any fixed 18-by-400 matrix C, interval J(B), and error box E=[-eta,eta],
every compatible point in B belongs to

\[
 K_\eta(B)=c-C([F(c)]-y-E)+(I-C[J(B)])H.
\tag{8}
\]

Intersect B with (8), retaining interval rounding. This follows by integrating
the Jacobian from c to the compatible point. It is a **necessary-set**
contractor. Even K_eta(B) inside B does not prove existence of a compatible
point in this overdetermined problem: a zero of C(F-y-e) need not satisfy all
400 original equations. In particular, do not promote a rectangular
preconditioned fixed-point existence argument to full-record existence.

If a nonnegative matrix K bounds |I-CJ(x)| for every x in B, and some positive
weights w satisfy Kw<=beta w with beta<1, then for any compatible a,b in B,

\[
 |a-b|\le 2|C|\eta+K|a-b|,
 \quad |a-b|\le (I-K)^{-1}2|C|\eta.
\tag{9}
\]

The inverse is nonnegative by its Neumann series. A directly checkable upper
vector d with 2|C|eta+Kd<=d also suffices under the same contraction condition.
All quantities must use compatible native scaling; if theta=c+S z, convert
the final vector by |S|. The second-order signed-Hessian construction in the
existing `certify_det2.py` gives a sharper K than |C| times an interval-Jacobian
radius. Formula (9) is a local hard-noise diameter bound. It is global only
when every compatible pair is covered by such a valid common convex branch
box, or all cross-component pairs are handled separately. Injectivity of F
on B only excludes equal exact outputs. It does not make a noisy compatible
set a singleton.

## 4. All-prior cover algorithm and a checkable finite certificate

Use the exact affine elimination in [elimination.md](elimination.md),
F(q,l)=b(q)+A(q)l, l=(d,g,px,py), to branch in
14 nonlinear variables when useful. The affine lane's certified outer LP or
two-dimensional geometric polygon must retain **every** feasible l for each
q in the nonlinear cell; selecting only an affine least-squares minimizer
would lose hard-band solutions. Keep the resulting outer geometric boxes or
polytopes attached to each nonlinear cell. Splitting in all 18 native
coordinates is a valid, slower fallback.

The certificate tree starts from B0 and maintains three disjoint bookkeeping
categories: rigorously rejected regions, retained terminal outer regions, and
unprocessed regions. Numerical scheduling need not be disjoint geometrically:
closed child boxes may share faces, which is harmless. Required proof rules:

1. Split: the children cover the parent exactly, including boundaries.
2. Reject: prove that the parent has no point in (1), using a physical
   impossibility certificate, a disjoint output interval, or a verified
   nonlinear/affine outer-feasibility contradiction.
3. Contract: prove C intersect parent is contained in the replacement set.
4. Accept as terminal outer region: retain the entire replacement, not merely
   its best fitted point. A terminal region need not contain any solution.
5. Budget stop: retain every unprocessed or inconclusive region explicitly.

**Theorem 2 (global cover invariant).** At every finite point in this process,
C is contained in the union of terminal and unprocessed regions. The proof
is induction from B0 using the four inclusion-preserving rules. Consequently
(4) is valid even before search termination, if it happens to pass; and no
failed optimizer, rank test, or unsuccessful interval evaluation can silently
discard a region.

A replayable certificate stores the exact observation bytes, clock and native
bounds; hard eta; model/arithmetic assumptions; split coordinates and
endpoints; outward box/profile bounds; branch signs and physical margin
proofs; fixed multiplier/preconditioner data and verified inequality bounds;
and the complete leaf ledger. Hashes identify these inputs but do not prove
the inequalities. A verifier need not rerun the optimizing search. A
certificate based on boxes around a submitted fit is a local certificate
until the rest of B0 has been rejected or included in the ledger.

Adaptive priorities can use the contribution of a cell to global profile
widths, interval derivative widths, affine-strip widths, and weak native
directions. These choose where to spend work, never which compatible points
to retain. Before Hessians, use cheap interval values, guard tests, selected
row strips, and exact affine feasibility. Fairness requires that unresolved
cells cannot be starved indefinitely by heuristics.

## 5. Termination: a precise sufficient condition and its limits

A finite cover at any requested mesh resolution is always available by
subdividing the bounded prior and retaining everything. That trivial fact is
not recovery. Global profile widths can remain large as cells shrink.

**Theorem 3 (finite accuracy closure, conditional).** Let A be an open
acceptable-error region around a chosen output, contained in its required
native error limits. Suppose every point of the compact set B0 minus A has
an open neighborhood that the sound rejection machinery can certify after
sufficient subdivision and arithmetic refinement. Suppose interval extensions
converge there, and fair bisection makes nested unresolved cell diameters tend
to zero. Then a finite subdivision rejects every cell outside A or leaves
only cells contained in A. Hence the cover test proves the requested error.

Proof: if infinitely many unresolved cells still meet B0 minus A, the finite
branching tree has an infinite nested path with a limit point in that compact
set. The assumed rejectable neighborhood of the limit point eventually
contains the cell, contradicting non-rejection. Cells straddling A are handled
by the same argument unless their limiting point is in A, in which case a
small enough cell lies in A. Equivalently one can use a finite subcover and a
Lebesgue-number argument. This is a termination theorem under explicit
separation/completeness conditions, not a complexity estimate for full18.

A sufficient regular case is a compact region with positive optical margins,
continuous F, and strict residual separation outside a slightly smaller
acceptable-error region. Uniformly positive residual gaps plus convergent
intervals eventually exclude those points. Stronger signed/affine contractors
can prove exclusions earlier but do not change the logical requirement.

The original physical set P uses strict inequalities and is **not generally
closed** inside B0. A physical interval that straddles zero is inconclusive,
not invalid. A failed denominator/TIR lower bound must never be rejection.
Positive margins checked around a fitted solution say nothing about distant
near-boundary cells. Compatible sequences can approach a forbidden branch
boundary; interval derivatives can diverge there. To claim finite global
termination, either certify that the data exclude a whole boundary
neighborhood (thereby deriving, not assuming, a positive margin), provide
an exact boundary argument, or keep those cells unresolved. Merely requiring
an epsilon optical margin would change the user's original prior.

Likewise no finite-time promise is made at exact decision boundaries, such as
a profile width exactly 0.002 or a constraint touching its allowed band with
no strict gap. A strict target and finite-precision certificates need strict
slack. Failure of a contraction condition, lack of rank, or an unresolved
search budget is not a proof that recovery is mathematically impossible.

## 6. Continua, degeneracy, and what finite enumeration can mean

With eta>0, an interior physical parameter whose residuals are all strictly
inside their bands has a neighborhood of compatible parameters by continuity.
Therefore the compatible set is usually uncountable even for a locally
injective, full-column-rank map. The finite deliverable is a certified cover,
coordinate profiles, or connected-component enclosure; it is not a list of
every compatible vector. Isolated solutions at eta=0 can sometimes be
enumerated by finitely many verified isolation boxes plus global exclusion.
That conclusion requires proving finiteness and handling singular components.

There is an exact continuum in the original prior. Set all three wedge angles
ax_i=0, beam angles bx=by=0, and source positions px=py=0. Then every surface is
flat, every ray direction is (0,1) in either axis, every sampled position is
zero, and all three physical guard values are 1. The three speeds, three
phases, three glass indices, distance and gap remain arbitrary in their full
native intervals: an 11-dimensional family with the exact same record.
This follows directly from the six-interface trace in `fmodel.py:66-113`.
For that record even eta=0 has distance range [50,200], so the unrestricted
minimax distance error is at least 75. This is an algebraic obstruction, not
an optimizer failure. It does not imply that nondegenerate records cannot be
recovered. The nonzero-wedge finite-noise witnesses below show why degeneracy
cannot be the only diagnostic either.

## 7. Implementable native-profile improvements

The following additions preserve the original experiment and all unknowns:

* Attach affine feasible geometry intervals/polytopes to each nonlinear cell,
  and intersect them with the signed nonlinear strip contractor (6).
* Use verified directional bounds (7) for d, gap, wedge/index combinations,
  and finally each native coordinate. Optimize the proposed multipliers on
  floating arithmetic, then verify their finite residual supports exactly.
* Form weighted Hessians before absolute bounds. Retain sign/positive
  semidefinite quadratic information, rather than always bounding a centered
  remainder by a symmetric interval.
* Run profile threshold decisions globally: to prove theta_j<=a, reject the
  subset theta_j>a using its intersection with every current leaf. A profile
  witness below or above a threshold supplies an inner bound, never an outer
  bound. Maintain both, with unresolved regions explicit.
* Combine a fast blind optimizer only as a supplier of physical witnesses,
  good local coordinates, and useful multipliers. Its failure counts do not
  enter the proof. All-prior exclusion is separate work.

These are constructive certificate rules. Their use on the complete original
18-dimensional prior remains an implementation/performance problem. No
successful full-prior closure is asserted in this note.

## 8. Existing records already refuting some requested guarantees

The independently interval-verified files
`../ambiguity/benchmark_verified_100.json` and
`../ambiguity/ordinary_noise_verified.json` store the exact two parameter
vectors, a single common 200-row observation record, strict positive guards,
and outward bounds on both discrepancies to that record. Their clock is
exact k/20, their model is the strict canonical per-axis trace, and their
submitted binary64 parameters are interpreted as exact binary rationals.

| Shared record | Hard allowance | Both-record error upper bound | Distance separation | Unavoidable distance error |
|---|---:|---:|---:|---:|
| nonzero wedges about 3 degrees, distinct fast speeds | 1e-8 | 9.722184929e-9 | about 0.0022 | about 0.0011 |
| wedges 12, -13.8, 10.8 degrees | 1e-4 | 7.817942063e-5 | about 0.05 | about 0.025 |

The first pair already refutes a uniform 0.001 guarantee over the full original
box at eta=1e-8; the second does so at eta=1e-4 with ordinary larger wedges.
Neither is an exact-noiseless ambiguity claim. They do not imply every record
has these errors. Their existence means no better search or proof method can
produce an honestly smaller global error bound on those particular records.
The appropriate output is their compatible uncertainty, not an unqualified
single-vector recovery guarantee.

The development record in `../profiles/random00_eta1e4.json` is different from
the identically named combined-holdout case. Its saved two gap-profile
endpoints also imply distance separation about 0.008162806 and an error floor
about 0.004081403 **once the final point-bound arithmetic is outward checked**.
This is a finite observed-record witness, not a global profile enumeration.
The independent strict constructed witnesses above do not depend on that
profile arithmetic issue.

`verify_noise_bounds.py` supplies small exact-rational algebraic controls for
the directional bound and incomplete-dual correction, a disconnected-cover
control, and exact endpoint-distance consequences of the saved strict
witnesses. It does not rerun optical verification and does not certify a
global optical cover. Its output records that distinction explicitly.
