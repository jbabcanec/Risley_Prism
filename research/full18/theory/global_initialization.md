# A complete native-domain construction for the passive full18 inverse

This note supplies the missing search contract. Spectral, phase-splitting and
continuation procedures can propose good answers, but an answer is globally
accurate only after every other compatible region has been accounted for.
The construction below retains the full native domain, zero and arbitrarily
slow rotors, signed wedges, collisions, all physical orders and all18 unknowns.
It gives one sound, implementable cover algorithm and precise termination
conditions. It does **not** claim that global exclusion has already been run
on the 200-sample instances or that its worst-case cost is practical.

The algorithm's essential object is the entire compatible hardware set. Its
three parts are exact fourteen-coordinate optical search, bounded linear
elimination of the remaining four coordinates, and certified retention or
exclusion of every native region. Candidate generation is optional: removing
every spectral and continuation heuristic leaves the same completeness
contract. The procedure can return a small certified compatible set, a proved
incompatibility or ambiguity, or an explicitly unresolved cover. It cannot
promise accurate recovery for records that do not determine all eighteen
coordinates.

## 1. Object to recover and representation

Let D be the original closed native 18-coordinate box. Let P be the strict
physical branch of the specified mathematical forward model, and let
F:P -> R^400 be its 200 passive 2D observations. For an explicitly specified
per-coordinate error allowance eta >= 0, define

    C_eta = {theta in D intersect P : |F(theta)-y| <= eta componentwise}.

Eta is an input or a symbolic certificate parameter. This construction does
not select 1e-8, or any other measurement precision, on the user's behalf.
The task is to bound the original native coordinates of every member of C_eta.

Use the exact sampled-rotor/optical lift in `algebraic_record.md`, rather than
assuming a finite list of observable spectral lines is exhaustive. At exact
t_k=k/20 its step variable lies on

    c_i^2+s_i^2=1, c_i >= cos(7*pi/20),

and its initial phase remains on the native +/-18 degree arc. Retain the
phase variables even when the signed wedge amplitude is zero; otherwise the
lift would incorrectly erase an unobservable native coordinate. The lift's
positive square-root/forward/intersection guards are strict inequalities,
not unannounced positive lower margins.

For practical boxes we may retain the original native coordinates and use
their outward-rounded images in the lift. Thus no inverse-atan phase bound
is needed near a zero wedge. In particular, a tiny lift-coordinate box need
not be a tiny native phase box. Native error acceptance always uses a valid
native-coordinate enclosure, including the entire phase fiber when required.

Split theta=(q,l), where q has the fourteen nonlinear coordinates and
l=(d_W,gap,p_x,p_y) lies in its complete original box L. On the strict branch,

    F(q,l) = b(q)+A(q)l.                                      (1)

The four variables in l remain unknown. Their elimination means solving a
bounded feasibility problem, not supplying calibrated geometry.

Physical order is already present in the ordered native coordinates. If a
frequency proposal is treated as an unordered triple, assign it to all six
positions. Sorting speeds without carrying the order label changes the model.
Repeated values may duplicate proposals; they do not authorize removal of
distinct physical arrangements. The full-domain cover does not rely on any
chosen speed ordering.

**Clock/model scope.** The finite polynomial lift cited above is for exact
k/20 timestamps and the strict mathematical ray equations. The original
binary64 program's actual timestamp array and angle/clipping operations are
not identical objects. A certificate for that data must either use the actual
timestamp/model convention or include a verified representation discrepancy.
No fitted residual threshold silently supplies that discrepancy. The native
uniform-time theorem must not be labelled a proof for a different numerical
map. Section 9 of `algebraic_record.md` also supplies a theoretical exact lift
for declared rational timestamps, including exact binary64 timestamps, using
their common denominator. Its algebraic clock constant can have enormous
degree; the small uniform-clock lift and its counts do not carry over. Neither
lift makes the rounded legacy implementation itself an exact real model.

## 2. Safe full-box pruning after affine elimination

For fixed physical q, define the exact bounded minimax residual

    d(q) = min_{l in L} ||b(q)+A(q)l-y||_infinity.              (2)

This is a four-variable linear program, with one epigraph variable. It is not
the existing least-squares acceptance score. Feasibility of C_eta at that q is
equivalent to d(q)<=eta. Rank deficiency of A is allowed: it creates a geometry
polytope, not an instruction to freeze geometry.

Write L=c+[-r,r]. For a fixed rational vector w in R^400, `elimination.md`
establishes the box rejection test

    lower_Q w^T[y-b(q)-A(q)c]
       - sum_j r_j upper_Q |w^T A_j(q)|
       - eta ||w||_1 > 0.                                  (3)

Here every bound must be outward-rounded and valid for every q in Q. The
geometry dual vector may be proposed by a floating-point LP; the strict
inequality (3), independently verified, is the actual rejection certificate.
No exact numerical nullspace identity is assumed. If physical interval
evaluation fails because a box crosses a guard, the result is UNKNOWN, not
rejection of the box.

For an immediately implementable second test, choose the box midpoint q0 and
let r_Q be its coordinate radii. Suppose Q is verified to lie in the strict branch and

    M_ij >= sup_{q in Q,l in L} |d F_i(q,l)/d q_j|.

Set e_i = sum_j M_ij r_Q,j. Every compatible pair with q in Q obeys

    l in L,
    |b(q0)+A(q0)l-y| <= eta*1 + e.                           (4)

Proof: integrate the q derivative along the segment from q0 to q with l fixed,
then apply the triangle inequality. The segment must lie in the verified
physical box; a positive guard at the midpoint alone does not suffice.

Therefore verified infeasibility of (4) rejects Q. Otherwise eight bounded LPs
(min/max for each of the four coordinates) give an outer geometry hull H_Q.
This produces a safe enclosure Q x H_Q without guessing l. A floating-point
LP result must itself be validated, for example through rational primal/dual
certificates or directed-rounding residual bounds. An unvalidated solver
status is a proposal, not a deletion rule.

At a descendant node the previously certified geometry hull H can replace L
in (2)-(5). Its validity is conditional on that node's q box and ancestor
allowances. Intersect each newly certified hull with H; never export a
conditional contraction to an unrelated optical box.

The same derivative argument proves

    |d(q)-d(q0)| <= max_i e_i, q in Q.                       (5)

It follows by applying the uniform bound to a minimizing l on each side;
compactness of L gives those minimizers. A verified lower bound for d(q0)
minus max_i e_i is another safe lower bound. This is useful for search ordering
and screening; (3) normally keeps more cancellation than this absolute bound.

Tests (3)-(5) require no lower speed, frequency separation, wedge excitation,
known refractive index or known beam source. Their bounds can become weak
near singular/critical optics; that is a computational limitation, not a
license to discard the region.

**Noise thresholds can be reported rather than assumed.** For a nonzero w,
the first two terms of (3), divided by ||w||_1, give a verified eta threshold
below which this particular box is excluded. Store these thresholds with the
tree. They expose the precision required by a proposed global proof and let
one replay a family of noise allowances without canonizing an arbitrary eta.
A geometry hull contracted at eta0 is automatically valid for eta<=eta0,
because the compatible sets are nested. It is not automatically valid for a
larger allowance. Store each contraction's allowance as well as each
rejection's threshold; a replay must validate every ancestor contraction.

## 3. What observed phases and frequency differences can safely do

The useful signed-wedge/phase cone is a constraint on the physical rotor lift:
at k=0, U=A cos(phi), V=A sin(phi), with |phi|<=18 degrees and signed A.
It is the union of two cones, and includes A=0. It is not automatically a
constraint on a fitted Fourier coefficient of the full nonlinear optical map.
Mixed harmonics, conjugate leakage, frequency collisions and finite-window
leakage can all move an observed coefficient outside a single-prism cone.

Thus the successful cancellation decomposition

    a exp(i phi1)+b exp(i phi2) approximately equals c_observed

is a proposal mechanism unless its nonlinear/finite-window remainder has been
bounded. Signed a,b are essential: restricting both to reinforce each other
missed an actual recovered collision example.

A safe phase-informed rejection can instead be built as follows. For any fixed
linear functional T of the 400 observations, enclose T F over a parameter/lift
box respecting its exact phase constraints. Reject only if that enclosure is
disjoint from T y plus the transformed error set. For scalar real T, the error
radius is eta ||T||_1. For real and imaginary Fourier components apply this
bound to their separate real rows. A local Taylor enclosure is valid only when
the entire expansion box has verified guards and a verified remainder. If it
does not, retain the box.

Similarly, a line at f can suggest missing speeds f-a*N_i-b*N_j. Such innovation
and sideband proposals repaired a real failure in `low_frequency/REPORT.md`.
They are not exclusion certificates for all unlisted frequencies. A certified
finite-window harmonic/annihilator test may reject a frequency box only under
its explicit tail/noise assumptions; see
`../newtheory/harmonic_invariants.md`. Its latent-rotor recurrence must not be
confused with a fixed finite annihilator for the nonlinear optical output,
which generally has infinitely many harmonics.
Without those bounds, retain a catch-all domain branch.

This yields a useful division of work: phases, amplitude gains, residual spectra
and frequency differences order the search and propose starting points; exact
physical constraints and validated residual bounds decide which regions may
be removed. The search remains complete when the spectrum has fewer than three
distinct visible lines, or when a needed physical line is not in the extracted
list at all.

## 4. A precise cover algorithm

Maintain an active frontier and retained unresolved leaves of native q boxes,
each with a geometry hull H subset L, plus a ledger of certified rejected boxes.
Initially the frontier is the whole
fourteen-coordinate box with H=L. The invariant is

    C_eta is contained in the union of all retained Q x H.   (6)

Use closed boxes with shared split faces; harmless overlap prevents losing a
solution exactly on a bisection plane. If disjoint ownership is desired, use
explicit half-open ownership while the validating enclosures remain closed.

```text
frontier <- {(full nonlinear box, full geometry box)}
verified_witnesses <- empty

repeat until budget exhausted or a requested certificate is obtained:
    choose a frontier box fairly
    # Alternate best-priority choice with oldest-FIFO choice.

    run exact physical/rotor constraint propagation
    if a verified physical impossibility certificate exists:
        delete box, record certificate, continue

    propose point fits from phase cones / spectra / innovations / continuation
    add a point to verified_witnesses only after full observation/guard proof
    # A failed or successful point fit does not delete any box.

    attempt verified dual rejection (3) and geometry contraction (4)
    if rejected: delete box, record certificate, continue
    replace geometry hull only by a proved enclosing hull

    if all retained enclosures meet the requested native error test:
        return that error certificate together with a nonemptiness witness

    if native leaf-resolution requested and reached:
        retain as an unresolved leaf, never as a compatible witness
    else:
        bisect a nonresolved native coordinate and keep both children

return verified points, ALL unresolved leaves/frontier boxes, and the ledger
```

This is a branch-and-bound algorithm on a feasibility set, not on a ranked
list of fitted candidates. At a fixed physical q the exact four-variable LP,
or the equivalent two-dimensional polygon plus source intervals proved in
`elimination.md`, determines all compatible geometry. Over a q box, the
validated outer LP retains every geometry possibly shared with some q in
that box. Solving separate rows with independently chosen geometry is not a
substitute: the same four hardware variables must satisfy all 400 rows.

Each cheap node operation has a fixed finite budget and returns REJECT,
CONTRACT or UNKNOWN. If a complete decision is requested, UNKNOWN invokes the
exact feasibility fallback in Section 6.4; if only a finite computational
budget is available, it remains in the cover. In particular a failed local
optimization, a zero determinant, or a spectral proposal outside the box is
not REJECT.

With a finite resolution request, use predetermined coordinate depth caps so
priority cannot repeatedly refine one coordinate forever. At every other
expansion choose the oldest eligible frontier node. Each existing node has
only finitely many predecessors and is eventually processed. Candidate
generation receives a finite per-node budget, so it cannot starve coverage.
Use cyclic coordinate bisection, or longest normalized side bisection with a
fixed tie rule, so every positive unresolved coordinate width tends to zero
along an infinite branch. Fair node selection alone would not establish that
fact: repeatedly splitting only one coordinate can leave all other widths
unchanged. Apply this requirement to geometry coordinates too when full18
leaf resolution is requested.
The priority can incorporate dual lower bounds, spectral mass, observed
phase-cone scores and expected geometry contraction without affecting (6).

**Proof of invariant (6).** It holds initially by the prior. A verified
exclusion removes no member of C_eta. A proved geometry contractor contains
every compatible l for q in its box. Bisection replaces a box by a union
containing it. Point proposals do not change the covering set. Induction proves
the invariant after every operation, including early termination by budget.

For full18 leaf resolution, subdivide geometry hulls as needed as well: a fine
q box with a wide geometry polytope is not a fine native 18-box. Alternatively
return that polytope explicitly as unresolved geometry ambiguity.

## 5. Sound output and acceptance rules

Let U be the union of all retained native enclosures. A parameter candidate
theta_hat has a GLOBAL error certificate epsilon_j only if

    U subset product_j [theta_hat_j-epsilon_j,theta_hat_j+epsilon_j].    (7)

Nonemptiness of C_eta must also be established; otherwise a vacuous implication
is not a recovery. If the output is claimed to be physically compatible, verify
theta_hat itself against all400 strips and strict guards. A numerical residual
or a local uniqueness box is neither global coverage nor this point proof.

A witness may be an exactly represented lifted solution with its native
readout, or an existence certificate for the full original constraints. It
need not be a rounded native floating-point vector. For eta>0 a point with
strict interior strip and physical margins can often be verified directly by
outward evaluation. At eta=0, an enclosure merely containing zero in each of
400 residual intervals does not prove any common exact solution; an exact
lifted witness or a valid full-system existence argument is required. The
overdetermined-system caveat in Section 7 applies to this witness step.

Coordinate infima/suprema enclosing C_eta can also certify a midpoint estimate
with radius half their span. That numerical midpoint need not itself be a
physical compatible system. Distinguish an accurate parameter estimate from a
verified compatible hardware explanation if this distinction arises.

Two verified compatible points separated by more than 2 epsilon in a native
coordinate prove that no estimate can attain epsilon for every compatible
system, by the triangle inequality. A wide unresolved cover without two such
witnesses proves only that the search has not yet established the requested
accuracy. It is not an impossibility result.

## 6. What terminates, and under exactly which hypotheses

### 6.1 Finite-resolution OUTER coverage needs no excitation assumption

For fixed positive native widths delta_j, a uniform grid of D contains at most

    product_j max(1,ceil((HI_j-LO_j)/delta_j))

cells. Process each cell once, reject only with a certificate, and retain all
other cells as unresolved. This always terminates and gives a sound finite
outer cover, including open-guard boundaries and continuous solution fibers.
It need not identify which retained cells actually contain a compatible point.

For dyadic bisection, a valid leaf-count bound is

    2^[sum_j ceil(log2(max(1,(HI_j-LO_j)/delta_j)))].

This separates a finite-resolution COVER theorem from a successful recovery
theorem. At delta_j=.001 the naive full18 tensor count is approximately
4.55e73 cells; even the fourteen-coordinate tensor count before geometry
resolution is approximately 2.33e56. Reduction alone does not make exhaustive
gridding viable. Effective validated pruning is the substantive computational
problem.

### 6.2 Cheap interval/LP exclusion terminates on separated regular regions

On a compact q region Q_* wholly inside the strict physical branch, assume
d(q)>=eta+gamma for every q in Q_*, with gamma>0. Continuity of b,A and uniform
continuity on Q_* imply a finite cover by boxes whose convergent interval/LP
lower bounds exceed eta. More explicitly, if the verified Lipschitz constant
in a chosen norm is K>0, boxes of radius h<gamma/(2K), with point-LP lower-bound
error below gamma/2, are rejected by (5).
For K=0 only the point-LP lower-bound error must be controlled.
In the dual proof, a maximizing real w with strict positive separation can
be approximated by a rational w closely enough to retain separation, since
the support expression is continuous in w. Thus requiring exactly represented
rational witnesses does not obstruct the strict-gap finite-cover argument.

The hypothesis is a measurable separation condition on a region to exclude,
not a globally inserted physical guard or frequency floor. A similar compact
cover argument proves eventual acceptance of (7) when every point outside an
inner target region (strictly inside (7)) has a neighborhood with a valid
exclusion certificate. A finite subcover supplies a common sufficient cell
size, and the fair schedule eventually reaches it. The required local
certificates, not mere low residual at theta_hat, are the premise.

### 6.3 Open boundaries and strip contact are real limitations

Strict R>0, forward and intersection guards define an open domain. Positivity
at each physical point does not give one uniform positive margin over it.
Likewise d(q)>eta pointwise does not imply inf d>eta. For the elementary
example P=(0,1], F(x)=eta+x, y=0, every physical point is incompatible but the
closed interval enclosure on [0,h] touches the strip at the excluded point 0.
A naive interval test can refine forever. Exact strict-inequality reasoning
can reject it; assuming x>=a positive floor changes the problem.

Consequently, without additional evidence the cheap interval procedure may
finish only with unresolved boundary cells. It must not silently discard those
cells or declare a unique global solution. Equality on a noise-strip boundary
can also prevent strict-margin proofs even away from physical singularities.

### 6.4 Exact lifted feasibility provides a complete fallback in principle

For the exact uniformly sampled mathematical model, `algebraic_record.md`
gives a real semialgebraic feasibility formula, preserving strict guards and
zero-wedge phase fibers. The rational-clock extension described in Section 1
also has this property, at a much larger algebraic cost. Data and eta
represented as rationals/algebraic numbers fit this formulation. Real algebraic decision
algorithms can decide the formula restricted to each native cell. They handle
strict inequalities, singularities and positive-dimensional components; no
nonzero Jacobian hypothesis is inserted. This is the existential/quantifier
elimination problem described in Basu's primary survey, sections1.2 and2.1:
https://www.math.purdue.edu/~sbasu/raag_survey2011_final-sep4-2014.pdf

Combining finite native subdivision with exact cell feasibility therefore gives
a finite algorithm returning precisely which resolution cells contain some
compatible hardware, and can supply sample points. This is a complete bounded
enumeration of CELLS, not a claim that one representative per cell is an exact
enumeration of a continuum of solutions. With rational native cell endpoints,
every angle/phase/speed cut is encoded by an algebraic circle-arc boundary,
and scalar cuts are polynomial inequalities. An arbitrary real oracle-valued
angle endpoint is not silently assumed to have this finite exact encoding.
Alternatively, a global error claim
for a fixed estimate can be tested by querying for a compatible point violating
any one of its36 coordinate inequalities. Use rational native endpoints for
these tests, or otherwise provide their exact algebraic arc encoding. Preserve
the correct arc branch when encoding those comparisons. A binary64 estimate
and binary64 tolerance have rational exact values and therefore meet the
endpoint condition, even though the resulting algebraic constants may be
expensive to represent.

The fallback is a logical completeness result, not an executed full18 algorithm
or practical runtime guarantee. Its lifted system is large, and generic CAD has
very high worst-case complexity. The purpose of the physical LP and interval
contractors is to make most such queries unnecessary, or to shrink them to a
small unresolved part. The source's complexity warning must not be replaced
with a promise that an off-the-shelf symbolic solver will finish this record.

## 7. Continuation is sound locally, but does not establish coverage

Predictor/corrector continuation can accelerate candidate discovery within a
verified physical region. A precise local statement is: for a specified square
system G(x,tau)=0, if a corresponding fixed-point map sends a closed box into
itself and has a verified contraction factor strictly below one, uniformly
for each tau in a step interval, then it has a unique fixed point there
for each tau. With an invertible derivative, the implicit-function theorem
gives the local smooth branch. Loss of that validation stops this path; it does
not delete other regions of the cover.

There are two important traps for this inverse problem.

* Interpolating the400 observation values does not generally produce a path
  of exact18-parameter solutions: an overdetermined target curve can leave
  the model image. A least-squares stationary path is a candidate path, not a
  path of exact inverse solutions.
* A fixed point of x -> x-C(F(x)-y), with C an18-by400 numerical left inverse,
  proves only C(F(x)-y)=0. That projected equation does not imply F(x)=y.
  All400 original equations/strips still require verification. The same issue
  applies to solving18 selected equations and ignoring the remaining382.

Harmonic weighting, gain coordinates and block-phase multiple shooting may
improve the corrector basin. They remain admissible accelerators when all
original constraints and unknowns are restored and checked at the end. A
successful local path does not prove that disconnected or singular compatible
components have been found. Keep the unvisited cover until exclusion proves
otherwise.

## 8. Why a universally complete finite point initializer is the wrong object

Set one allowed wedge to zero. Its rotation speed and phase no longer affect
its flat faces. Holding the other coordinates fixed yields a continuum of
native speeds/phases with the same passive record. With all wedges and source
angles zero and source position zero, the record is identically zero for broad
families of the remaining parameters as well, on strictly transmitted branches.
No finite dictionary of exact speed points can enumerate these fibers, and no
unique full18 estimate can satisfy a tiny tolerance for all their members.

Finite boxes or semialgebraic cells can represent such fibers without removing
them. This does not excuse failure on the nondegenerate recovered cases: it
specifies the necessary output contract of a global algorithm on the unchanged
full prior. For a particular informative record, the same machinery may certify
a small complete compatible set. It must establish that from the observations
and exclusion ledger, rather than importing an excitation assumption.

## 9. Concrete next implementation boundary

The candidate layer already exists in `collision/` and `low_frequency/` and has
actual full18 recovery evidence. The next missing implementation is a persistent
frontier with theorem-labelled rejection records:

    node: native nonlinear box, geometry hull, physical-order label if used
    rejection: exact rational dual w + verified margin (3), or exact feasibility
    contraction: validated8-LP geometry bounds with the interval tube (4)
    witness: all400 forward-strip checks + strict physical checks
    progress: remaining boxes, native coordinate hull, eta validity thresholds

Every reported global result must be reconstructible from those records and
the original full-domain root node. This is an implementable construction with
a proved covering invariant; the unfinished work is making its exclusions
efficient enough and executing them, not finding another local residual
threshold to rename as global recovery.
