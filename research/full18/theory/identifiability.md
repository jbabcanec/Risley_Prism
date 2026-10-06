# A constructive local inverse and the scope of full18 identifiability

This note concerns the original passive experiment: 200 known timed x/y pairs,
all eighteen native parameters unknown, the original unsorted parameter box,
and physical prism order preserved. It concerns the exact independent-axis
optical equations evaluated by `stable_model/forward.py`, not the legacy
floating program's clipping and signed-arccos guards. No additional measurements
or restrictions on unknown parameters are introduced.

The constructive conclusion is a verifiable **local eighteen-parameter inverse
chart**. A chart can use eighteen selected scalar coordinates from the existing
400-coordinate observation record. Its inverse is obtained by a contraction
iteration, with interval checks supplying existence, uniqueness within the
chart, and deterministic error bounds. The remaining observations supply
additional consistency constraints. This does not establish that every possible
explanation of the record lies in the chart.

## 1. An explicit, checkable construction

Write the exact timed record as `F: D -> R^400`. Here `D` is the open domain on
which every sampled interface strictly transmits, the outgoing axial direction
is positive, all grazing denominators are nonzero, and angle tangents exist.
The original native box is imposed in addition to these physical conditions.

Choose a candidate `x`, any eighteen output indices `S`, and positive parameter
scales `D0=diag(HI-LO)`. Define `H(theta)=F(theta)[S]`. Compute any proposed
inverse matrix `C` for `H'(x)D0`; it need not itself be computed rigorously.
Treat its stored entries as exact numbers during verification. Let `B` be a
convex native-coordinate box containing `x`, and certify:

1. `B` is inside the native box and all strict physical guards hold on `B`.
2. An interval derivative enclosure gives a nonnegative matrix `E` satisfying
   `|I - C H'(theta) D0| <= E` for every `theta in B`.
3. The certified upper bound `q=||E||_infinity` is less than one.

All products and row sums in these checks must be enclosed outward. A floating
singular value, determinant, or inverse residual without error bounds is a
proposal for this check, not its proof.

**Local chart theorem.** These conditions imply that `H` is injective on `B`,
that its derivative is nonsingular throughout `B`, and consequently that `F`
is injective on `B`.

**Proof.** For `theta,psi in B`, set `u=D0^(-1)(theta-psi)`. Integrating the
Jacobian along the segment contained in `B` gives

`C(H(theta)-H(psi)) = (I-R)u`, with `|R| <= E`.

If the left side is zero, `||u|| <= q||u||` forces `u=0`. The same argument
at a single point proves the square derivative nonsingular. Equality of the
full records implies equality of their selected coordinates. No vector-valued
mean-value theorem with an unjustified common intermediate point is needed.

**Explicit inverse iteration.** In scaled coordinates `theta=x+D0 u`, for a
selected observation vector `z`, iterate

`u_next = u - C(H(x+D0 u)-z)`.

Its derivative has norm at most `q`. If the centered scaled box `|u|<=r`
satisfies the outward-verified componentwise condition

`|C(H(x)-z)| + E r <= r`,

the map takes that box into itself and is a contraction. Iteration converges
to its unique fixed point. Since `C` is nonsingular, that point satisfies
`H(theta)=z`. The inclusion check is a substantive existence condition; small
Jacobian defect alone does not prove the supplied record is in the chart.
The remaining 382 coordinates must still be checked against the actual record.

This is a mathematical construction from the existing observations, not a
claim that an arbitrary initializer will find the relevant chart. Choosing
output rows does not fix unknown parameters, impose a prior, alter the physical
order, or discard the remaining consistency constraints.

### Concrete certificate checked in this workspace

`rank_chart_certificate.py` uses the already selected blind candidate
`combined_validation/noiseless/random_00.json`; it reads no truth file and
performs no optimization. QR pivoting proposes eighteen output rows. The
original `fmodel.forward_iv` encloses their Jacobian, and independent 70-digit
`mpmath.iv` directed products verify `I-C[J]D0` at the point and four small boxes.
The resulting `rank_chart_certificate.json` records the actual row indices,
timestamps, proposed inverse, interval Jacobians, guard margins, and `q` values.

The check succeeds with `q <= 1.6291860922e-6` at the point and
`q <= 0.2052163037` on the recorded positive-radius box approximately
`1e-10*(HI-LO)` wide in each half-coordinate. On that box the certified minimum
transmission, forward, and absolute grazing margins exceed respectively
`0.5791056124`, `0.8542857822`, and `0.9407958686`. The selected flat output rows
are `396,383,375,357,316,341,399,223,54,1,12,44,210,33,60,253,382,90` in
`x0,y0,x1,y1,...` order. The scaled local inverse Lipschitz bound is at most
`120.6460569`. Larger proposed radii `1e-9`, `1e-8`, and `1e-7` fail this
particular interval contraction test; failure to certify those boxes is not
proof of a singularity or an alternative explanation there.

This single bounded derivative check ran in about 0.49 seconds. It establishes
the needed nonzero minor and an explicit small injective chart. It does not
verify the inverse iteration's self-mapping inclusion for the observed record,
nor establish that the unknown true parameter lies in the chart. In particular,
the tiny chart size must not be presented as a certified reconstruction error.

Its Jacobian enclosure inherits the explicit `ivx.py` assumptions A-fp, A-libm,
and A-blas. Directed final multiplication does not remove those upstream
assumptions. Neither Python implementation has been formally verified. The
certificate is for exact supplied binary64 timestamps constructed by
`np.arange(0,10,.05)[:200]`, with mathematical trigonometric constants enclosed
by `fmodel`; it must not silently be relabeled an exact-rational-time result.
The analytic theorem below only needs a certified nonzero point minor; a
certified positive-radius box additionally gives the constructive chart above.

## 2. Exact affine structure makes the rank question smaller

Reorder the unknowns only for linear algebra, retaining their physical meaning:
let `ell=(d_W,gap,bm_px,bm_py)` and let `u` collect the other fourteen. The source
model has the exact identity

`F(u,ell)=b(u)+A(u)ell`.

This is exact geometry, not a paraxial approximation. The incoming and outgoing
directions depend on wedge, phase, speed, glass, and beam angle; intersections
are affine in source position and plane heights once those directions are fixed.
`separable.py` and `solver/poc3.py` implement this identity.

At a parameter point put `B=partial_u F` with `ell` held fixed. If `A` has rank
four and `P` is orthogonal projection onto the complement of its column space,

`rank [B A] = 4 + rank(PB)`.

**Proof.** Split observation space into `range(A)` and its complement. The
four affine columns span the first subspace; only the projected nonlinear
columns can add independent directions in the second. Thus all eighteen
parameters are locally regular precisely when `rank(A)=4` and `rank(PB)=14`.

This equality suggests a useful constructive chart. Select four rows `R` for
which `A_R` is invertible. At a fixed record `y`, solve exactly

`ell(u)=A_R(u)^(-1)(y_R-b_R(u))`.

On the remaining rows the derivative of the residual at a solution is

`B_rest - A_rest A_R^(-1) B_R`.

Choose fourteen rows on which this Schur matrix is invertible. The resulting
eighteen-row full Jacobian minor has determinant, up to row/column sign,
`det(A_R)` times that fourteen-row Schur determinant. Interval bounds can verify
both factors or directly verify the full inverse defect in Section 1.

This separates two distinct failures: insufficient geometric excitation of
the four affine columns, and weak or missing information about the fourteen
direction parameters. It does not remove glass/wedge coupling. At a nonzero
residual the derivative of a variable-projection least-squares objective also
contains residual-dependent terms; substituting `PB` for that complete
derivative away from a fit would be an error.

The source-position columns have disjoint x/y support. They may be eliminated
first, leaving a two-column distance/gap rank question; no information is lost.
Bounds on the four affine parameters must still be enforced. A temporary
active optimizer bound is not evidence that the corresponding physical
parameter is known.

## 3. Deterministic precision follows from the chart, not rank alone

For any two parameters in a certified chart whose full records are both
compatible with `y` at coordinatewise noise allowance `eta`, the preceding
segment argument gives

`|D0^(-1)(theta-psi)| <= E |D0^(-1)(theta-psi)| + 2 eta |C| 1`.

Since `q<1`, the matrix series for `(I-E)^(-1)` is nonnegative and converges.
Therefore a certified componentwise diameter bound is

`|theta-psi| <= 2 eta D0 (I-E)^(-1) |C| 1`.

An outward-enclosed version, or a verified supersolution of this inequality,
is a deterministic local error statement for every admissible noise pattern.
It contains no statistical coverage assumption. The coarser scalar bound is
`2 eta D0_ii ||C||_infinity/(1-q)` in coordinate `i`.

One may instead use a rectangular `18 x 400` left inverse and all record rows
in the same proof, often improving the bound. A valid certificate must account
for numerical forward-model error in the observation allowance. The stable
audit in `stability_audit/INTERPRETATION.txt` distinguishes that floor from sensor
noise and verifies the saved actual records.

Full rank does not imply useful precision. An inverse can exist while its
noise amplification is too large for 0.001 native-coordinate accuracy. The
already verified separated-speed, nonzero-wedge ambiguity pair at `eta=1e-8`
is directly relevant evidence of this distinction. No change of coordinates
or numerical optimizer can distinguish its two compatible physical endpoints
from that record alone.

Nor does a local chart certificate imply its box contains the true parameter.
A global accuracy claim needs a separate exhaustive argument excluding every
compatible point outside the chart, or a global covering whose surviving
coordinate ranges are sufficiently narrow. `theory/algebraic_record.md`
describes an exact-clock feasibility formulation; feasibility encoding alone
does not supply practical exhaustive elimination or uniqueness.

## 4. The precise analytic genericity statement

Every coordinate of `F` is real analytic on each open connected strict-physical
component: it is a composition of trigonometric functions, division by nonzero
denominators, and positive square roots. The artificial guards in the legacy
floating code are not included in this statement.

**Generic local identifiability theorem.** Fix one such connected component
`U`. If a selected eighteen-row Jacobian minor is nonzero at one point of `U`,
then full Jacobian rank eighteen holds on an open dense subset of `U`, with
its rank-deficient complement of Lebesgue measure zero.

**Proof.** That minor is a real analytic scalar function `m` on `U`. A nonzero
value makes it a nontrivial analytic function on the connected domain. Its
zero set has empty interior and measure zero. Where `m` is nonzero the full
Jacobian has rank eighteen. The rank-deficient set is contained in that zero
set; full rank is open by continuity. The inverse-function theorem for the
selected eighteen coordinates gives local injectivity there.

The concrete directed minor certificate supplies the needed witness under
its stated arithmetic contract. A numerical SVD alone would not. The theorem
does not propagate to a different connected analytic component without another
witness or a proof connecting the domains. It proves neither global uniqueness,
uniform conditioning, finite-noise accuracy, nor successful blind initialization.
Having 400 scalar observations for eighteen parameters does not independently
prove any of these properties.

Collision submanifolds require care: an open-dense statement in eighteen
dimensions does not automatically give a generic result restricted to
`N1=N2`. One must prove the selected minor is not identically zero on that
stratum, for example with a separately verified point there. Distinct speed
magnitudes are useful to some spectral initializers, not a proved necessary
condition for the full nonlinear model's local identifiability. The source
`AGENTS.md` and September 16 diary distinguish their verified local collision
certificates from global recovery.

## 5. Which apparent symmetries survive the actual conventions?

The native vector is
`[N1,N2,N3,ax1,ax2,ax3,ay1,ay2,ay3,ng1,ng2,ng3,d_W,gap,bm_ax,bm_ay,bm_px,bm_py]`.
Speeds are in Hz; wedge, phase, and beam angles are in degrees. Native speeds
lie in `[-3.5,3.5]`, wedge and phase in `[-18,18]`. Time origins, sample order,
and x/y labels are known.

* A formal sign change `ax -> -ax`, `ay -> ay+180 degrees` gives the same
  rotating normal. The phase box does not contain both representatives, so
  this is not a two-point ambiguity inside the native box when `ax != 0`.
  The usual 360-degree phase periodicity likewise supplies no duplicate there.
* At exact times `k/20`, an individually observed rotor has frequency aliases
  `N -> N+20m`. No two such frequencies lie in `[-3.5,3.5]`. This removes that
  elementary rotor alias; it does not prove the optical record determines
  rotor states or excludes more complicated alternative optical explanations.
  Exact binary64 timestamps are a distinct mathematical grid.
* Permuting physical prisms changes propagation distances and the sequence of
  refractions. It is not a universal symmetry of the nonparaxial model. Sorting
  speeds would change the parameter problem. Repeated identical elements can
  of course make a particular permutation vacuous.
* Reversing all rotor speeds and phases, accompanied by y-angle/source
  reflection, reflects the y record. With labeled x/y observations it does
  not generally reproduce the same record. Special reflection-invariant
  records must be treated separately.
* A time translation or reversal changes the known time labels. It is not an
  allowed reparameterization of this timed experiment.

## 6. Exact exceptional fibers and their nearby conditioning limits

These explain why the chart conditions are instance-specific; they are not
proposed replacements for the already stronger ordinary-wedge noise witness.

**A zero wedge.** If `ax_j=0`, its normal is flat at every sample, independent
of `N_j` and `ay_j`. Both parameters have exact fibers within the native box;
their Jacobian columns vanish. Thus full rank is at most sixteen. The glass
index need not disappear: a plane-parallel plate changes lateral displacement
for an oblique incoming beam.

For the first flat wedge there is an additional explicit fiber. For transverse
axis `a`, let `B_a` be the source angle in radians and
`q(n,B)=sin(B)/sqrt(n^2-sin(B)^2)`. The state at the first exit plane is

`p_a=bm_p_a + 6 tan(B_a) + 3 q(ng1,B_a)`, with outgoing slope `tan(B_a)`.

Changing `ng1` to `ng1'` while changing
`bm_p_a` to `bm_p_a+3[q(ng1,B_a)-q(ng1',B_a)]` preserves this complete outgoing
state and therefore every downstream observation. Locally inside the native
position and glass bounds this is a third independent fiber, so rank is at
most fifteen. It is not legitimate to infer that every flat prism's glass
index is unobservable without an analogous state-preserving argument.

**All speeds zero.** With `N1=N2=N3=0`, the optical output is constant in time.
The derivative columns for the other fifteen parameters are constant x/y
vectors. Each speed derivative is `2 pi t` times a fixed derivative with
respect to its rotor phase in radians. Consequently the complete Jacobian
lies in the span of `1*e_x`, `1*e_y`, `t*e_x`, `t*e_y`: its rank is at most
four, not merely two. Restricted to the zero-speed submanifold, the fifteen
static parameters determine only two output coordinates. Where the source
position derivative is invertible, the implicit-function theorem gives a
thirteen-dimensional exact static fiber. A single zero speed alone does not
justify this conclusion for the other moving-prism system.

**Nonzero parameters do not imply a uniform noise bound.** Consider two
admissible speed values of a single prism separated by a fixed amount, and
give that prism a common wedge `delta`. At `delta=0` their records agree.
On a compact strict-physical neighborhood the derivative with respect to
`delta` is bounded, so their record difference is at most `2M|delta|`.
For every positive `eta`, a sufficiently small nonzero wedge makes their
midpoint record compatible with both endpoints while their speed separation
stays fixed. An interval derivative bound can make `M` explicit. This argument
can retain nonzero, distinct speed magnitudes and nonzero other wedges.

There is a complementary finite-window limit with wedges fixed nonzero.
Choose two interior static configurations on a regular constant-record fiber,
using source position to compensate a small fixed material or geometry change.
Set both speed triples to `delta*nu`, with three distinct nonzero components
of `nu`. As `delta -> 0`, the two timed records converge to the same constant
record while the other-parameter separation remains fixed. Strict guards and
native bounds persist for small enough `delta`. Hence excluding exact zero
wedges and exact zero speeds alone cannot establish uniform positive-noise
precision on their noncompact complement.

## 7. What is proved, and what remains to construct

The interval inverse-defect test gives a concrete local chart and, with the
self-mapping check, a convergent reconstruction formula. Exact affine
elimination supplies a smaller rank test without reducing the unknown count.
A verified point minor establishes generic local regularity in its connected
analytic component. Native conventions remove several elementary coordinate
symmetries, while the stated exceptional fibers are exact model properties.

Still required for a global constructive solution are: a method that reaches
all relevant charts from an arbitrary admissible record; exhaustive exclusion
of other compatible charts or an explicit report of them; and global bounds
on the compatible set tight enough for the requested accuracy. Neither the
analytic genericity theorem nor the semialgebraic encoding supplies those
steps. A solver's nonconvergence is an algorithmic failure unless accompanied
by an actual incompatibility proof; a small residual is not a parameter-error
certificate; and a rigorously compatible separated pair is a precision limit
regardless of the optimizer used.

Supporting workspace files: `rank_chart_certificate.py` and its JSON output;
`../stable_model/forward.py`; `../stability_audit/results.json`;
`../structure/results.json` (numerical conditioning diagnostics only);
`../ambiguity/witnesses.json`; and `algebraic_record.md` (exact-clock lift).
Source mathematical identities and arithmetic contract are in
`risley_lattice/separable.py`, `fmodel.py`, `ivx.py`, and `certify_det.py`.
