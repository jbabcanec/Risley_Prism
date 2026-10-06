# Wedge research handoff

## Current status: October 6, 2026

The current research is [the exact full18 package](research/full18/README.md),
including [report revision 13, October 3](research/full18/REPORT.md).
It was previously saved outside this repository in the October 2 research
workspace and is now published alongside the project.

**A complete constructive inverse exists in principle. Efficient global
computation remains the bottleneck.** The coupled three-dimensional vector-Snell
construction uses exact sampled algebraic optics, preserves all compatible
systems and ambiguities, and computes native uncertainty envelopes in principle.
The original passive 200-position protocol is covered by this set-valued
construction under its finite-input, ideal-clock and sampled-time physical
contract. This is distinct from uniform unique recovery for every system.

The executable certificate engine implements bounded pieces of the construction.
It retains four unresolved leaves in its saved full-prior check. Nineteen Lean
support declarations do not formalize the complete inverse or engine.

Use the package README for the algorithm and evidence links. The scientific
sources and original workspace are preserved; no optical recovery campaign or
new mathematical proof was run for publication. Later algebraic correction
notes, their instability result and the inconclusive finite box remain scoped
as stated in the report.

## Historical handoff entries (through September 27)

The earlier progress record follows unchanged. Its uses of "latest", "current"
and "open" belong to those dates and do not supersede the research package above.

Latest direction, September 27: the user stopped the verification and
certificate-integration cycle and requested a direct analytic result or
routine for the full unknown system. Continue with
[the inverse-state derivation](paper/RAY_STATE_INVERSE.md) and the new
`risley_lattice/ray_state_inverse.py` / `ray_state_initialization.py`.
The callable sparse multiple-shooting routine retains all18 unknowns and
actual timestamps, but is unexecuted and returns only numerical candidates
or unresolved. No new optical recovery is demonstrated. The new analytic
correspondence and rotor-constraint regularity do not establish global
convergence. Pending v26 material has no frozen snapshot; proposed RMS-cover
integration was stopped. Do not restart audit/report/manifest work by default.
The latest DIARY entry records the scope and the all-active six-degree
finite-noise obstruction. Earlier material below is background.

Current development adds analytic sparse derivatives, improved spectral
initialization, stationarity-aware constraint updates and a constrained
trust-region solver. `experiments/ray_state_development.py` is the prepared
bounded three-case harness. It was executed on 2026-09-27 (evening) on its three
cases: moderate 6.5e-4, seven 1.6e-3 (both polish below 5e-9 with `solve.trf`),
collision fails at initialization. See that DIARY entry. Three cases are not a
success rate; no archived campaign was replayed.

Updated September 27, 2026 (controlled geometry returns, explicit phase-free clock conditions and a selected-edge source gate added; blind full18 reconstruction remains open). Start here, then read the latest entries in
[DIARY.md](DIARY.md), the authoritative research log.

V25 gives a physical sufficient condition for the two nonparallel geometry
edges left conditional in v24. The [normalized response proof](paper/NORMALIZED_GEOMETRY_EDGE_V25_2026_09_27.md)
uses the original M=BR/(vT), hence W=(b/v)/M=bT/(BR). It proves W_s>1/6
on strictly transmitted wedges through fifteen degrees, and W_s>1/5 on
the full native seven-degree domain. Actual terminal and middle return
pairs with nonzero increments then give four-affine rank, provided both
axes are present. The original frozen K>3/8 gives an explicit seven-degree
minor floor. This is conditional on the row pairs, not full18 identification.

The [near-return theorem](paper/ROBUST_GEOMETRY_RETURNS_V25_2026_09_27.md)
retains full native glass and beam and bounds derivatives of normalized
distance/gap features by (306,296,307) and (14,10,0). Desired changes must
exceed upstream leakage, and their determinant reserve must remain positive.
This proves an explicit tolerance for imperfect returns on the independent
seven-degree effective-sine cube. Zero targets and failed allowances remain
unresolved. The comparison is within the same unknown nonlinear candidate.

The [phase-free finite-clock result](paper/PHASE_FREE_RETURN_COVERAGE_V25_2026_09_27.md)
characterizes exact controlled returns by frequency orders and supplies a
nonempty analytic family N=(+/-20/Q,+/-10/Q,+/-20/(3Q)), Q=6,...,99.
Rows (0,Q,2Q) on both axes suffice. With |sin ax2|,|sin ax3|>=1/10,
speed errors <=1/(200000Q) Hz and supplied per-sample clock error <=1 ns,
the source-weighted minor exceeds 1/650000 for some axis combination.
Keeping all four squared axis-pair minors gives an explicit positive
four-affine Gram floor for one fixed six-row set, uniformly in unknown
phases. This is a thin native region; arbitrary-speed coverage, useful
conditioning and nonlinear recovery remain unproved.

The [selected-edge source carrier](paper/SELECTED_GEOMETRY_EDGES_V25_2026_09_27.md)
compiles a degree-four fixed oriented minor and six mass/minor/trace guards.
All guards must hold on the whole coefficient region to obtain a Gram
floor. The full200 symbolic program has 100,388 nodes, retaining its 94,750
original nodes and fourteen inputs, with no observations. The carrier does
not yet compile the phase-uniform four-minor sum. No optical numerical
wrapper, actual data certificate or optical evaluation was run.

Primary research extends the retained Boyd--Vandenberghe read to Appendix
A.5.4, printed 648-649, with exact original-page and excerpt bindings.
Its standard SVD/Gram statements support the interpretation; the specialized
prism perturbation proof is derived here. Frozen v1-v24, the prior covers,
archive ledger and T1--T10 labels are unchanged. Seven-degree transmission
and these conditional affine results do not establish a 5--7-degree blind
initialization tolerance. Full18 reconstruction remains the active goal.

The v25 release has eight final reports and 433 top-level checks. Author/
audit counts are 80/35 (normalized edges), 47/49 (near returns), 56/51
(clock coverage), and 55/60 (source carrier). The immutable snapshot is
`controlled_geometry_manifest_v25.json`; all 24 preceding snapshots remain
preserved. The next proposed step, not yet proved, is a real-analytic
zero-set argument for generic four-affine rank on the seven-degree prior.

The previous v24 [moving tangent test](paper/RMS_TANGENT_V24_2026_09_27.md)
gives a ridge-free lower bound for the original shared four-affine RMS
profile. It collects the actual gradient expressions before taking their
absolute-value support. Exact unrelated polynomial families exclude over
whole nonlinear intervals even though their residual convex hull contains
zero, which rules out any one common fixed linear row separator. This
explains a useful distinction from earlier fixed-dual supports. V23 already
allowed source-dependent effective duals; the new result is a simpler
alternative, not a stronger pointwise duality theorem. Author/audit 49/24
pass. Rank changes, zero weights, singleton coordinates and hidden interior
fits remain accounted for.

The [curvature source gate](paper/RMS_CURVATURE_SOURCE_V24_2026_09_27.md)
removes the ridge penalty where original curvature can be verified. A fixed
rational preconditioner S and positive m must pass all 32 signed diagonal-
dominance guards for S^T G(q) S>=m I on the entire nonlinear cell. Only then
can the quartic stationarity-defect surplus exclude the full affine fiber.
A positive surplus alone is insufficient. Author/audit 48/47 pass, including
independent metric-orientation and active-multiplier checks. No optical
curvature floor has been evaluated or proved by source compilation.

The separate [active-face theorem](paper/RMS_CURVATURE_TRANSPORT_V24_2026_09_27.md)
can use a positive free-coordinate block even when the full Gram is singular.
Whole-cell active-gradient guards certify a lower bound from its stationary
face point. Additional free-coordinate guards are required to claim that
point lies in the cube and attains the minimum. The generic exact checker
distinguishes both claims, detects hidden face switches and retains all
original rational coefficient domains. An audit found and repaired a pole
hidden by cancellation before the final 53/50 author/audit pass. This
checker is not an authenticated optical face compiler.

[Hardware-dependent source elimination](paper/HARDWARE_WEIGHTED_AFFINE_RANK_V24_2026_09_27.md)
reduces four-affine rank to a two-by-two distance/gap Schur matrix while
retaining the actual source gains. That matrix is a sum of weighted outer
products of same-axis gain-weighted row differences; two nonparallel edge
vectors give a conditional curvature target. Zero source masses have an
explicit division-free branch. The source compiler appends polynomial
moments and a rank determinant without adding divisions or observations.
Author/audit 163/71 pass. Bounded source elimination has at most nine
quadratic pieces in distance/gap; a selected boundary piece is not a claim
of a unique ambient Hessian.

The same algebra proves an exact native ambiguity when only prism j has
a nonzero wedge. Keeping nonlinear coordinates fixed, the affine direction
(-[3-j],1,-[j-1]*t0x,-[j-1]*t0y) in (d,g,p_x,p_y) preserves every output.
Small changes remain in interior native affine priors. This includes
nonnormal beams and moving rotors, but concerns this stated zero-wedge
stratum only. It is an original-model ambiguity, not the v22 relaxation
fiber. Earlier zero-wedge/affinity results are credited.

Eight reports total 505 proof and implementation controls. Full 200-sample
symbolic carriers preserve the 94,750-node coefficient prefix: tangent
111,185 nodes, curvature 129,791, and rank moments 103,199. The first two
retain 400 formal targets; the rank carrier introduces no target inputs.
No optical state, rotor, observation or numerical source certificate ran.
Snapshot `shared_curvature_manifest_v24.json` preserves all 23 predecessors,
the frozen cover/driver and exact primary-source read scopes. Boyd and
Vandenberghe's strong-convexity and Schur-complement sections support the
standard convex ingredients; the explicit source and face guards are
derived here. No global inverse theorem is imported from that reference.

T5 confinement, T7 useful computation, T8 integration and T9 validation
remain open. The next physical obligation is a useful whole-region lower
bound for gain-weighted edge separation or an admitted Gram/free-face floor,
together with an original-data residual gap. Rank alone does not confine
the fourteen nonlinear coordinates. All T1--T10 labels, the historical
ledger and the earlier seven-degree guarantees remain unchanged.

The earlier v23 [shared-source RMS test](paper/AFFINE_RMS_SOURCE_V23_2026_09_27.md)
retains the original nonlinear coefficients in F=beta(q)+A(q)z, with one
common four-coordinate affine box and all fourteen nonlinear coordinates
unknown. A fixed five-by-five polynomial determinant gives a lower exclusion
test over a whole nonlinear cell. Positive ridge removes the rank gate;
subtracting its full 4*delta cost makes the bound valid for the original
residual. Both axes use their original N*eta_axis^2 budgets, including when
rows are selected sparsely. Strict positivity excludes; survival does not
establish a physical fit. This retains the material dependence discarded by
the v22 uniform-gain relaxation. Author/independent controls pass 67/53.

The [exact affine RMS oracle](paper/AFFINE_RMS_PROFILE_V23_2026_09_27.md)
supplies verified box KKT proposals, including rank-deficient cases, by
searching at most 81 faces. Nonnegative axis weights form a complete strict
exclusion family at a fixed coefficient problem; a finite weight schedule
need not find a separator. Small ridge and rational box multipliers recover
every strict fixed-fiber exclusion. This does not supply the missing global
nonlinear separation gap or a useful covering count. Author/independent
controls pass 50/29, including 54 independently prescribed dense minima.

[Finite Snell secants](paper/FINITE_SNELL_SECANTS_V23_2026_09_27.md) express
exact differences between same-hardware rows using endpoint means and
differences. Both endpoints require all three half-margins and native faces
at most 15 degrees. Upstream states may differ; no intermediate path,
wedge division or fixed upstream slice is needed for the secants. Their
affine-column recurrence retains shared glass, geometry, beam and source.
A separate formal flat-face derivative exceeds 4*d/5 in its own glass
index along the baseline-preserving native source fiber. This identifies
information lost by the old relaxation; it is not an observed finite-record
jet or a full-rank inverse theorem. No secant source compiler is claimed.
Author/independent controls pass 65/49.

The [fixed source-projection obstruction](paper/GLASS_SOURCE_PROJECTION_OBSTRUCTION_V23_2026_09_27.md)
shows why subtracting one fixed mean generally cannot remove unknown source
position uniformly over glass. On the stated terminal-prism stratum, fixed
row weights cancel source for an open glass interval exactly when their
sum vanishes in each equal squared-face group. An analytic native 200-row
clock example has all distinct groups on both axes, leaving only zero
weights. Candidate-dependent weights, original cofactors and the new Gram
test remain available. This is an obstruction to that fixed projection,
not physical nonuniqueness. Author/independent controls pass 51/50.

The full 200-sample RMS source compiles to 119,064 nodes, preserving the
94,750-node coefficient prefix, all 400 formal targets and both formal RMS
allowances. No optical data, including zero placeholders, were instantiated.
No numerical optical API ran. Eight reports total 414 controls, which are
proof and implementation checks rather than recoveries. The v23 snapshot
`hardware_dependent_rms_manifest_v23.json` preserves all 22 prior snapshots.
Boyd and Vandenberghe's original convex-duality text supports the ridge-box
dual; exact sections and retained-source hashes are recorded in the RMS note.

Actual optical confinement, strict reconstruction, useful global cost,
integration and validation remain open. T1--T10 and the recovery ledger are
unchanged. Seven-degree transmission and the conditional five-unknown
inverse retain their established scope; a 5--7-degree blind initialization
tolerance remains unproved. The next step must turn these shared expressions
into useful whole-cell bounds without replacing their hardware dependence
by independent coefficient intervals.

The earlier v22 [global half-margin path](paper/GLOBAL_HALF_MARGIN_PATH_V22_2026_09_27.md)
connects arbitrary same-system, same-axis fifteen-degree endpoints when
all three exit-root squares are at least1/2 at BOTH endpoints. Conditional
clipping and an exact moving-boundary cancellation retain those margins
along the path. Its second coordinate is monotone; the third need not be.
With the separate [half-domain response bounds](paper/HALF_MARGIN_RESPONSE_V22_2026_09_27.md)
|F_sj|/d<(18,11,6), this gives a global row cost bounded by
d*((57/2)*|delta1|+18*|delta2|+6*|delta3|). No small-tube restriction
remains on this qualified set. This is not arbitrary fifteen-degree
transmission, convexity of the straight segment or an inverse guarantee.
Author/independent controls pass94/37 for the path and64/64 for response
and source transport. The independent path controls use234 unrelated exact
scalar clamp pairs; they are not optical evaluations.

The [phase-free graph constraint](paper/PHASE_FREE_GRAPH_ENERGY_V22_2026_09_27.md)
combines both observed axes using the exact sine/cosine return identity.
It keeps one common distance/wedge-product polytope and charges the
original separate-axis RMS error through a verified graph-Laplacian bound.
Both axes at every selected endpoint must qualify. It gives collective
speed/wedge exclusions without knowing phases; it does not reconstruct
frequencies or prove ordinary-speed confinement. One conditional matching
example excludes simultaneous near-return bands of radius1e-5Hz at lag5,
given its stated data energy and noise bounds. Author50/independent45 pass.
The symbolic source compiler binds each return to its original speed and
lag, retains all18 inputs and both-axis rows, and shares repeated lags.
The five-sample source fixture is compilation only. Numerical source
wrappers are implemented but remain unexecuted.

The [relaxation limitation](paper/GAIN_CONE_SHADOW_LIMITATION_V22_2026_09_27.md)
is decisive for choosing the next route. The covered universal-gain and
row-bound constraints depend on hardware only through the six rotor
coordinates, three signed wedge amplitudes, distance and two flat
baselines. At any admitted strict native-interior fit, six independent
changes in glass, gap and beam tilt can be compensated by source offsets
while preserving every covered constraint. This is an exact local
six-dimensional fiber of the relaxation, not physical nonuniqueness.
An additional seven-dimensional fiber concerns a particular affine shadow
only. Author47/independent35 pass. Retaining a physical DAG prefix does
not impose its output equations on independently relaxed output rows.
The next recovery mechanism must retain shared hardware-dependent
nonlinear response or original residual relations that break this fiber;
more uniform gain bounds alone cannot do so. The earlier calibrated cubic
response identities illustrate the lost information but do not supply it
from an arbitrary moving scan record.

Snapshot `global_half_response_manifest_v22.json` binds eight reports with
436 controls and preserves all21 earlier snapshots. No numerical optical
state, observation, rotor or source certificate was evaluated. Actual
confinement, strict reconstruction, useful global cost, integration and
validation remain open; T1--T10 and the archive ledger are unchanged.

The earlier v21 [quarter-margin comparison](paper/QUARTER_MARGIN_TRANSPORT_V21_2026_09_27.md)
uses the old convex air-direction identity to propagate v>1/2 throughout
the original three-prism quarter-margin branch. The all18 unflattened caps
improve from(2011,890,196) to(93,50,24)*d. A half-margin anchor and
(15/4)|delta1|+(25/12)|delta2|+(19/10)|delta3|<=5/32 prove a connecting
quarter-margin segment, strictly enlarging the v20 sufficient tube. This
is a local forward comparison, not an angle-error or inverse-noise guarantee.
Author51/independent60 controls pass; the actual source adapter is compiled
symbolically and its numerical API remains unexecuted.

[Output-conditioned responses](paper/OUTPUT_CONDITIONED_RESPONSE_V21_2026_09_27.md)
eliminate contact positions before taking bounds, giving
|F_sj|<(16,21/2,13/2)_j*|F|+d*(70,75/2,13)_j on the same admitted region.
The intercept has no source-position term. Author79/independent41 controls
verify the free derivatives, exact source cancellation and complete native
scalar envelopes. One draft exact-scalar strict comparison was corrected
to equality; the strict physical premise and theorem are unchanged.

[Endpoint transport](paper/OUTPUT_CONDITIONED_ROW_TRANSPORT_V21_2026_09_27.md)
handles |f'|<=a|f|+b using a symmetric endpoint comparison. It needs no
interior output band. On a proved connecting tube, if both endpoints have
|F|<=d/14, the finite comparison
caps become(498/7,153/4,377/28)*d, improving every unconditional cap.
Exact rational exponential bounds and absolute-value chords retain constant
output-row coefficients for the frozen whole-cell dual engine. Necessary
endpoint intervals and one original RMS support use the same error budget.
Generic controls retain a hidden interior fit that endpoint-only checks lose.
The endpoint module has no new physical source adapter; its derivative and
path premises remain explicit obligations.

Primary [comparison research](paper/PATH_COMPARISON_RESEARCH_V21_2026_09_27.md)
checks Bihari's original theorem and inverse-domain condition; the signed
endpoint comparison is independently proved. The [Risley literature read](paper/RISLEY_INVERSE_RESEARCH_V21_2026_09_27.md)
distinguishes Luo2026's two-offset calibration of known11.35-degree hardware
against external checkerboard geometry from Li2017's pointing inverse and
our blind18 problem. Full-text versus abstract-only scopes are recorded.

Endpoint author57/independent46 controls pass. The admitted tube bounds its
scalar comparison rate by63/80;25 Taylor terms suffice for width2^-80.
Snapshot `output_conditioned_transport_manifest_v21.json` binds all six
reports with334 checks and preserves all twenty earlier snapshots. These
checks are not new optical recoveries. No numerical optical API ran.

Angle context: full native +/-7-degree wedge transmission is proved for all
rotors and native beam/glass ranges. Joint recovery of three wedges and two
source coordinates is conditional on supplied remaining13coordinates and
finite rotor sign coverage; an exact nominal design has a constructive
oracle theorem. Blind full18 recovery, arbitrary15-degree inverse guarantees
and a5-7degree initialization tolerance are not established. The frozen
five-degree broader inner protocol and all earlier evidence remain intact.

The earlier v20 [physical row coupling](paper/ROW_COUPLING_V20_2026_09_27.md)
repairs the independent-gain cone's stationary false fits. From one half-margin
anchor, the explicit tube8|delta1|+4|delta2|+2|delta3|<=3/20 keeps the mixed
effective-sine segment transmitted through15-degree faces. Original unflattened
response caps(2011,890,196)*d give costs that vanish as states coincide.
Identical effective triples have identical same-axis outputs without that tube
premise. These conservative local constants do not establish global robustness.
The symbolic source compiler retains all18 inputs, actual rows/clock and
operation domains; numerical source certification remains unexecuted.

The [coupled fixed-feature profile](paper/COUPLED_ROTOR_PROFILE_V20_2026_09_27.md)
keeps shared products, baseline offsets, fitted rows and proved row bands in
one convex relaxation with the original separate-axis RMS budgets. Its exact
strict separation-certificate family is complete for that declared relaxation;
the numerical proposer is not complete, and feasible relaxation points are
not physical witnesses. Three generic duplicate-pair constraints jointly
exclude although each fits alone; the exact excess profile boundary1/6 survives.

[Joint product contraction](paper/SHARED_PRODUCT_CONTRACTION_V20_2026_09_27.md)
uses all necessary contrasts on one common distance/product polytope. Exact
residual support validates imperfect proposed multipliers. Generic examples
show a contradiction missed by independent product boxes, actual interval
contraction, and stronger distance bounds after shared-error cancellation.
Source validity of every cut remains an external obligation.

[Whole-cell dual transport](paper/COUPLED_PROFILE_TRANSPORT_V20_2026_09_27.md)
holds one dual fixed while enclosing all16 complete expressions on a shared
nonlinear DAG. Baseline cancellation is exact; bounded product residuals
retain the common polytope. It gives a conditional finite-cover mechanism,
not the missing global separation gap or a useful cost guarantee.

V20 author/independent checks pass80/51 for row coupling,56/50 for the profile,
51/40 for product contraction, and33/40 for transport. The transport audit
directly composes the actual profile inequality blocks with a whole-cell DAG.
Snapshot `coupled_row_profile_manifest_v20.json` binds eight reports with401
checks and preserves all nineteen earlier snapshots. The row audit repaired
a tuple-key serialization defect; the scalar theorem and source-only replay
pass. No optical or physical-rotor values/observations were evaluated.
Frozen v1-v19 and the cover/driver remain unchanged. Useful actual confinement,
strict physical reconstruction and global cost remain open. T1--T10 and the
archive ledger remain unchanged; earlier milestones follow.

The v19 [gain tightening](paper/REVERSE_FLATTENING_GAIN_TIGHTENING_V19_2026_09_27.md)
raises the qualified15-degree reverse-staircase lower gains to
(59/200,6/25,21/1000)*d, retaining all native unknown hardware. Directly
output-qualified rows retain the stronger third floor8337d/88000. The proof
first establishes positive contact brackets, then combines lower factors;
independent affine certificates cover the full geometry rectangle. Checks43/27.
These bounds apply to the staircase integrals, not general unflattened
early-prism derivatives or the old conditional inverse modulus.

The [rotor-product support](paper/ROTOR_PRODUCT_SUPPORT_V19_2026_09_27.md)
uses one signed dual with exactly zero mass on each axis. Its baseline
cancels and its selected support depends only on six speed/phase coordinates.
The common products r_j=d*abs(sin ax_j) range over an exact16-vertex polytope,
with all eight closed wedge sectors retained. This projects a necessary
constraint; it does not recover the twelve eliminated coordinates or replace
the full inverse by a six-dimensional problem. The original18-input carrier,
candidate bounds, clock and constraints remain. Full200 symbolic compilation
has167,781 nodes and400 rows; the selected dependency claim is checked directly.
Author/independent45/53 pass, including independent active-constraint vertex
enumeration of the common-product polytope.

The [four-affine gate](paper/AFFINE_BASELINE_SUPPORT_V19_2026_09_27.md)
implements the v18 research reduction for general signed duals, including
nonzero baseline masses. It keeps all16 corners, shared nonlinear expressions,
grouped/collected variants and every original domain. Max branches are selected
only from fresh whole-cell bounds. Full200 source compilation has168,976 nodes
and retains the complete161,154-node carrier prefix. Its generic upper improves
11 to7. Numerical source certification remains unexecuted.
Author/independent61/68 pass.

The [shared scenario selector](paper/SHARED_SCENARIO_DUAL_V19_2026_09_27.md)
automatically proposes one fixed rational row-weight vector across all declared
scenarios and corners. Exact replay checks finite supports and separate original
RMS budgets, then the unchanged weights must pass a fresh complete cell gate.
The32-scenario generic example succeeds where either single-scenario dual
fails elsewhere. A hidden-interior-fit example passes the scenario stage but
correctly survives the whole-cell gate. Finite scenarios never certify a
nonlinear cover, and solver status/objective is never proof.
Author/independent52/45 pass, including2,048 independent support-vertex checks.
Snapshot `shared_rotor_support_manifest_v19.json` binds eight reports with394
checks and preserves all earlier evidence; these are controls, not recoveries.

All v1-v18 evidence and the frozen cover/driver remain unchanged. No optical
state, rotor, observation, derivative, forward/inverse computation or source
certificate was numerically evaluated. Actual useful confinement, certified
reconstruction, integration and global cost remain open. T1--T10 and the
archive ledger are unchanged. Earlier milestones follow.

The v18 [reverse-flattening theorem](paper/REVERSE_FLATTENING_REFERENCE_CONE_V18_2026_09_27.md)
proves that flattening faces in order3,2,1 preserves half-margins at each
qualified15-degree row. It gives F-B0=sum gamma_j*A_j*u_j with one common
all-flat baseline per axis and gains between d/3125 and d*(11/2,11/2,6).
This is a staircase comparison, not an arbitrary mixed-candidate path.
Directly v16-output-qualified rows have the stronger terminal lower gain
8337d/88000 and terminal-flat reference bound10079d/9156<9d/8. The bound
improves the v17 terminal lower gain by2779/391. Checks53/36 pass.

The [flat-terminal source compiler](paper/FLAT_TERMINAL_REFERENCE_SOURCE_V18_2026_09_27.md)
authenticates the original source, clock, priors and physical order. Its
symbolic reference removes terminal speed, phase and wedge dependencies
while retaining terminal glass and all18 candidate inputs. The external
zero reference is valid even when zero is outside a narrowed candidate cell.
Full200 compilation has161,154 nodes and400 reference rows. Checks64/63.

The [harmonic source wrapper](paper/TERMINAL_HARMONIC_SOURCE_V18_2026_09_27.md)
now binds the previously generic v17 correlation to those references and
the actual unit-rotor recurrence, with400 uninstantiated formal targets.
It explicitly distinguishes tangent wedge inputs from sine amplitudes,
keeps both closed sign sectors, and checks the same-data original-N RMS
qualification before its future numerical gate. Sharper gains and all16
shared affine corners are retained. Checks64/51. Source compilation is
symbolic; the numerical optical certificate API remains unexecuted.

The [common-baseline constraint](paper/SHARED_FLAT_BASELINE_V18_2026_09_27.md)
projects a finite row-cone relaxation exactly onto one shared offset and
returns an attaining signed dual. Its whole-cell gate combines original
per-axis RMS budgets and shared features before enclosure. A separate
compact symbolic source binds B0 and d*A_j*u_j with18 inputs and7,912 nodes.
It authenticates the original optical carrier but does not execute its
constraints or prove physical feasibility. The generic200-row fixture
has error450 against budget200 while every pair has error at most9/2.
This is unrelated nonoptical evidence, not an actual recovered scan.
Author/independent52/57 pass, including324 independent exact projection
witnesses and9,072 support-vertex comparisons.

The focused [research follow-up](paper/COMMON_BASELINE_AFFINE_RESEARCH_V18_2026_09_27.md)
checks Boyd/Vandenberghe's primary full text and derives a source-specific
improvement: positive d factors outside every max gain term, so the actual
common-baseline support is exactly affine in all four geometry/source
coordinates. Its full maximum reduces to16 complete corners with common
nonlinear q retained. A generic bound improves11 to7. The source-corner
compiler and useful optical q enclosures remain the next obligations;
this does not upgrade the stable generic kernel's actual performance claim.
Research author/independent26/25 pass. Snapshot
`shared_reference_source_manifest_v18.json` binds ten reports with491 checks
and preserves all earlier evidence. Checks count reproducibility controls,
not solved optical cases or progress percentages.

These bounds concern physical wedge sizes through15degrees under explicit
conditions; they are not a15-degree initialization radius. Actual useful
optical confinement, reconstruction, integration and global cost remain
open. Frozen v1-v17 and the cover driver remain unchanged, and no optical
or physical-rotor numerical evaluation or observation was instantiated.
T1--T10 and the archive ledger are unchanged. Earlier milestones follow.

The v17 [terminal harmonic theorem](paper/TERMINAL_HARMONIC_CUT_V17_2026_09_27.md)
removes an extra reference-band premise: on v16-qualified15-degree rows,
flattening the last face gives |F0|/d<9861/6104<13/8. Its path response lies
between(1173/88000)*d and12*d. Paired X/Y rows then yield a two-sided finite
harmonic constraint with one original-N RMS allowance, without a Fourier
tail, speed separation or nonzero-wedge assumption. Only two closed amplitude
sign sectors are needed. Amplitude elimination is exact for this relaxation;
the four shared affine unknowns still use16 corners. The generic API expects
the unpolarized correlation and applies the amplitude sign itself. Optical
correlation/source association remains unimplemented. Checks38/50 pass.

The [general reference gate](paper/REFERENCE_FACE_TERMINAL_V17_2026_09_27.md)
keeps correlated polynomial/remainder models over one shared nonlinear and
affine cell, including outward references outside the band. It is conditional
on source/guard enclosures and only bounds compatible candidates, so it
cannot be substituted into unconditional residual feedback. Checks65/28.
The [terminal tube](paper/TERMINAL_TUBE_V17_2026_09_27.md) retains a shared
squared-error budget, complete-speed rotor support and implicit inverse
graphs without solved center roots. A generic whole-cell example excludes
while every two-row subproblem survives. Actual optical projection/range
coverage remains unproved. Checks70/47, with independent endpoint audits.

The [15-degree order hull](paper/FIFTEEN_DEGREE_ORDER_HULL_V17_2026_09_27.md)
inherits margins only between actual comparable output-qualified anchors.
Its row contrasts keep common d*|A_j| products; distance bounds require
positive amplitude floors. It also proves that magnitude bounds alone leave
eleven outer coordinates unrestricted when three nonzero wedges can be
arbitrarily weak. Checks65/49. Actual values and shared row relationships
are necessary to go beyond that obstruction.

Snapshot terminal_shared_cell_manifest_v17.json binds eight reports with412
checks and preserves v1-v16. No optical/physical-rotor numerical evaluation
or observation was instantiated. The new analytic constraints still need
source-verified useful whole-cell optical bounds and reconstruction. T1--T10
and the recovery ledger remain unchanged. Earlier milestones follow.

The v16 [output-margin theorem](paper/OUTPUT_FORCED_FIFTEEN_MARGIN_V16_2026_09_27.md)
supplies the v15 transmission premise from a coordinate bound:
|F|<=d/14-17501/25984 implies all three R_i²>=1/2 for physical wedges±15degrees.
F is an original uncentered compatible model coordinate, with measurement
and evaluation error charged. At d50 the threshold is10757/3712, about2.898
position units. Unknown distance remains in the necessary alternative:
half-margins hold OR d<14r+17501/1856 for a valid output cap r. Selected
qualifying rows suffice if their rotor sign coverage is proved separately.
This makes a conditional premise observable; it does not infer the thirteen
outer coordinates or cover every transmitted15-degree system. Checks83/41.

The [terminal theorem](paper/TERMINAL_OUTPUT_BAND_V16_2026_09_27.md) extends
the native18-degree last-wedge band to |F|/d<3/2, and terminal15degrees to
1019/597, through a sharper exact contact bound. Independent Sturm and
Bernstein proofs agree. Its center-free recurrence gate retains zero wedges
and upstream uncertainty; diameter bounds are per fixed upstream fiber.
Checks94/79. Novelty correction: the September17 regular-bounds proof
already implies27/19, so the valid v15 terminal band below was weaker than
prior work. The v15 all-three15-degree inner inverse is unaffected.

The [sample transport theorem](paper/MONOTONE_SAMPLE_TRANSPORT_V16_2026_09_27.md)
turns proved seven-degree response bounds into directed finite-sample costs.
Nonnegative edge flows preserve one shared RMS budget, canceling intermediate
row errors. They strengthen independently inflated pair tests in a generic
200-row example. The exact Lipschitz extension characterizes the relaxed
response class, not optical realizability. Checks52/22. The
[source bridge](paper/SUPPORT_FLOW_SOURCE_V16_2026_09_27.md) binds costs to
all18 inputs and the actual clock/axes using a symbolic feature program.
Full200 compilation has7,239 feature nodes and14,234 cost nodes. No optical
or physical-rotor numerical gate has been executed; checks40/45.

Snapshot data_margin_transport_manifest_v16.json binds eight reports with456
checks. Frozen v1-v15 and the cover driver remain unchanged. The remaining
bottleneck is useful confinement over shared unknown upstream cells, followed
by certified physical existence/reconstruction. T1--T10 and the recovery
ledger are unchanged. Earlier milestone descriptions follow.

The v15 [fifteen-degree theorem](paper/FIFTEEN_DEGREE_RESPONSE_V15_2026_09_27.md)
extends conditional five-coordinate injectivity to physical wedges in
[-15,15]degrees on the explicit common transmission-margin class R_i²>=1/2.
The same thirteen outer coordinates must be supplied at both endpoints,
and rotor sign coverage is required. All native hardware ranges remain
inside that coupled class; arbitrary transmitted15-degree configurations
are not covered. Positive source gain>1/5 and normalized wedge responses
>1/625 give a deterministic inverse modulus. Ordered witness-row paths
avoid assuming convexity of the whole multirow domain. Separately, the
last wedge has a unique same-fiber readout for an output satisfying
|F|/d<6965/5662 on its full strictly transmitted interval. Author/independent
controls pass156/52. This is a wedge-size result, not an initialization-error
radius or an executed recovery.

The [joint selector](paper/JOINT_CORNER_WEIGHTS_V15_2026_09_27.md) chooses
signed residual weights and five common slopes together, using55+2m
variables and219 inequalities. Only exact rational weights/slopes leave
the untrusted numerical optimizer. The [actual-source gate](paper/SEVEN_DEGREE_JOINT_ROWS_V15_2026_09_27.md)
then recompiles the combined original expression and verifies all16 corners.
It retains both original grouping and exact collection; the stronger
complete same-slope proof wins. A generic cancellation example improves
1/2 to1. Full200 symbolic compilation retains18 physical inputs,400 formal
targets and6,400 row corners in108,589 nodes. Numerical optical checking
remains unexecuted. Author/independent selector checks43/40 and source
checks39/28 pass, including a corrected zero-scale proof sentence.

The [alias-energy quotient](paper/ALIAS_ENERGY_QUOTIENT_V15_2026_09_27.md)
projects the v14 generator-conditioned energy onto signed alias-class sums.
Its finite200 dual needs no full-label separation or Gram inverse. All
eight wedge-sign sectors remain, with explicit DC and mixed-sign cancellation.
Same-sign native phase sectors yield a conditional generator-sum cap.
Useful finite-record tails, actual frequency cells and source/data binding
remain unproved; this is not blind frequency recovery. Checks58/57 pass.

Snapshot: fifteen_degree_joint_alias_manifest_v15.json, eight reports,
473 checks. Frozen v1-v14 evidence and the existing cover driver remain
unchanged. No optical values or observations were instantiated. T1--T10
labels and the recovery ledger are unchanged. The full18 objective remains
active; the following paragraphs preserve earlier milestones.

The v14 round changes the spectral strategy: a uniform degree-three/four
error below1e-5 at seven degrees is impossible for approximations at the
same physical generators. The [finite-record proof](paper/SEVEN_DEGREE_TAIL_BOUNDS_V14_2026_09_27.md)
uses a native one-active-last-wedge subfamily, equal2Hz rotors and a symbolic
alternating functional on all200 samples. It proves both per-axis RMS and
sup error above19711/1679616000>1e-5, without evaluating an optical scan.
The proof permits arbitrary approximating coefficients. It does not rule
out changed frequency dictionaries, higher order, adaptive support or
exact-model recovery. A smaller degree-six obstruction leaves1e-5 accuracy
open. A new generator-conditioned spectral-energy inequality concerns
torus L2; it cannot be used directly as finite-record error.

The [exact affine-corner reduction](paper/SEVEN_DEGREE_AFFINE_CORNERS_V14_2026_09_27.md)
covers the four distance/gap/source variables by16 corners while retaining
all18 unknowns. It removes source-wedge Taylor remainders. An independent
audit caught avoidable cancellation loss; corner substitution now preserves
the original expression structure. Full200 symbolic compilation has121,378
nodes,18 physical symbols and400 formal target symbols. No optical values
or target records were supplied. Independent generic examples give strictly
stronger minorants, but actual optical sharpness and cost remain unproved.

The [common-slope selector](paper/CORNER_PROFILE_PERSPECTIVE_V14_2026_09_27.md)
uses a55-variable,218-inequality linear formulation. Numerical optimization
proposes slopes only; every final intercept and profile consequence is
reconstructed exactly. All corner and anchor premises remain separate.

The [partial-catalog theorem](paper/PARTIAL_GENERATOR_CATALOG_V14_2026_09_27.md)
replaces complete line detection with a certified cap Gamma on omitted
generators. Unselected slots retain detected alternatives OR the correlated
weak branch d*|sin(ax)|<=400Gamma/3, with speed/phase unrestricted.
Residual frame and selective-functional bounds can certify Gamma; the
latter can tolerate aliases among nuisance columns. Pruning still requires
all-label separation. Failed premises, aliases and zero wedges remain
represented. Optical tails, regional frames and actual detections remain
unverified; the generic kernel is not a physical cover controller.

Snapshot: seven_degree_corner_spectral_manifest_v14.json, eight reports
with422 checks. All v1-v13
evidence remains preserved. T1--T10 labels, archive recoveries and the full18
objective are unchanged. The following paragraphs retain earlier milestones.

The v13 round proves additional structure throughout the seven-degree domain
and implements an actual-equation residual program. It does not demonstrate
blind full18 recovery or a seven-degree initialization-error tolerance.
The conditional joint inner inverse below still requires thirteen supplied
coordinates and finite rotor sign coverage.

The [generator-gap theorem](paper/SEVEN_DEGREE_GENERATOR_GAP_2026_09_27.md)
subtracts the same linear term from both coordinate responses before applying
the earlier monotonic Fourier argument. Every participating nongenerator
coefficient is at most (1-3/22400) times its associated generator magnitude.
Signed wedges and arbitrary phases are retained; zero wedges remain invisible.
The finite catalog selector needs complete support, magnitude bounds and
sampled separation of ALL order-K labels, including absent lines. It is a
conditional generic kernel, not a certified optical frequency catalog.

The [complex-strip proof](paper/SEVEN_DEGREE_COMPLEX_STRIP_2026_09_27.md)
gives phase strip width1/128 radians and complex signal norm below2000,
uniform over native real hardware/source ranges and physical wedges up to
seven degrees. It retains the original root branches. This first explicit
global strip is too conservative for a useful low-order tail: its order3
tail majorant exceeds33billion. That is a defect of this upper bound's
sharpness, not a lower bound on the actual physical tail.

The [harmonic-selection proof](paper/OPTICAL_HARMONIC_SELECTION_2026_09_27.md)
gives 44-mode odd or 73-mode circular truncations, depending on the stated
symmetry/filter conditions. A finite lag filter and Hankel rank test can
exclude candidate cells without estimating frequencies first, but need
useful structural/tail allowances. Aliases and collisions are retained.

The [residual-minorant bridge](paper/SEVEN_DEGREE_RESIDUAL_MINORANT_V13_2026_09_27.md)
symbolically reconstructs all400 original rows with18 physical inputs and
400 uninstantiated target symbols. Its partial Taylor theorem fixes only
the five inner coordinates and retains all thirteen outer intervals.
The generic kernel and the optical source interface were independently
audited, including786 exact generic comparisons and original-domain/noise
accounting. Actual optical minorants, anchor residuals, sign/transport
premises and feedback have not been numerically evaluated. Useful bounds,
global coverage and runtime remain open. No frozen driver is integrated.

Evidence snapshot: seven_degree_spectral_minorant_manifest_v13.json,
eight reports with480 checks. All v1-v12 sources/reports remain preserved;
only the three living logs advance. Primary research from the previous
round is retained. T1--T10 labels and the recovery ledger are unchanged.

The following paragraphs retain the earlier v12 and v11 milestones.

**All thirteen outer coordinates now have an explicit forward variation
bound on the seven-degree domain.** The
[hardware transport proof](paper/HARDWARE_TRANSPORT_2026_09_27.md) gives
|F_nj|<d*(12,12,10), beam-radian sensitivity<52d, degree sensitivity
<286d/315, distance sensitivity<145/64 and shared-gap sensitivity<12265/4032.
It includes the full native glass, geometry, beam and source ranges.
Author/independent checks pass51/39; the independent proof propagates
forward sensitivities rather than reusing the author's backward adjoints.
The [rotor transport proof](paper/SEVEN_DEGREE_ROTOR_TRANSPORT_2026_09_27.md)
retains signed speed/phase compensation and the trigonometric range cap.
For the exact200-sample grid, its time center is199/40, not5; centered
phase is Delta ay+1791 Delta N in degrees. Exact RMS moments avoid an
unnecessary pointwise time bound. Rotor author/independent checks pass53/60.
Adding these two transports covers all
thirteen outer unknowns; no optical point or record was evaluated.

The [outer-profile construction](paper/SEVEN_DEGREE_OUTER_PROFILE_V12_2026_09_27.md)
uses a sign-complete anchor only. Candidate rotors may be arbitrary native
values, including collisions. For a certified anchor residual rho,
candidate allowance eta and whole-cell variation E, every compatible
inner point obeys ||Delta p||infinity+(d0*tau/50)||Delta A||1
<=(8/3)(rho+eta+E). The exact kernel retains all32 coupled halfspaces and
all18 coordinates. A signed residual minorant can exclude a whole outer
cell; a separate feedback helper gives a true global profile lower bound
only when its minorant holds on the entire original domain. Generic
author/independent controls pass68/61, with729 independent support cases.
The kernel still reports source_binding_verified=False. No optical anchor
residual, residual minorant, exclusion or cover was executed. Coarse bounds
can leave localization vacuous; useful global cost remains open.

[Finite-record primary research](paper/FULL18_FINITE_RECORD_RESEARCH_2026_09_27.md)
reads Das--Yorke and Laskar and rechecks Aubel--Boelcskei, Moitra and
Batenkov--Goldman--Yomdin. It makes the missing spectral transfer explicit:
full Fourier orders3/4 have63/129 possible nodes. The latter exceeds the
cited200-sample matrix-pencil theorem's budget; the former allows no
integer stride above1 at its full node count and can still alias exactly.
A proved analytic-tail formula states the additional amplitude, separation,
complex-domain and harmonic-assignment premises. Positive derivatives and
analyticity alone cannot determine unknown frequencies from finite samples
in a generic unknown-waveform class; that obstruction is not an optical
ambiguity. All actual physical coefficient constraints remain necessary.

Evidence snapshot: seven_degree_outer_transport_manifest_v12.json,
seven reports with385 checks plus retained primary research.
Frozen v1-v11, the five-degree contraction, the conditional seven-degree
inner theorem and its nominal triangular readout remain unchanged. The
new work advances necessary conditions for full18, not global recovery,
practical optical execution or the T1--T10 completion statuses.

**Seven-degree joint inner theorem (09-27): finite rotor sign coverage now
rules out cancellation between unknown wedges and source positions.** The
[sign-design proof](paper/SEVEN_DEGREE_SIGN_DESIGN_2026_09_27.md) integrates
the actual full-record map between arbitrary inner endpoints, including
the changing source, and retains one positive averaged source gain per row.
If X rotor rows cover every strict sign triple with margin tau, the three
signed wedges and X source are globally identifiable throughout[-7,7]degrees;
any Y row then fixes the independent Y source. Coverage in both axes gives
the joint deterministic inverse modulus. All seven other hardware coordinates
and all six rotor coordinates are supplied and fixed between endpoints.
The theorem is uniform over every native glass/geometry/beam/source choice
in those fibers; it is not blind full18 identifiability.

A proved six-dimensional native rotor box is centered at speeds
(5/8,5/4,5/2)Hz and zero phases, with independent speed radii1/250Hz and
phase radii1degree. Both axes have sign margin tau=1/8 on the existing
200-sample k/20 grid. With speed radii1/2000Hz, each sign group has12
certified rows per axis. These statements also tolerate explicitly bounded
clock errors<=1/10000second. No numerical trigonometric or optical values
were evaluated. Author/independent controls pass50/62.

Writing A=sin(ax), the larger-box sup-record theorem is
||Delta p||_infinity+(d/400)||Delta A||_1<=(8/3)R_sup. For the smaller
box, max-axis RMS discrepancy gives the right side at most11R_RMS.
Two points compatible with the same eta-error record use2eta, whereas a
computed candidate with certified residual rho uses rho+eta. The bounds
do not assert that a candidate was computed or arbitrary data are feasible.
No fast recovery iteration for the whole neighboring rotor box follows
from this sign-coverage result alone.

**Constructive nominal-design readout:** the
[triangular seven-degree procedure](paper/SEVEN_DEGREE_TRIANGULAR_READOUT_2026_09_27.md)
uses exact speeds(5/8,5/4,5/2), zero phases and t=k/20. Four Y entries
at indices0,8,4,2 successively isolate pY,A1,A2,A3; X row2 then gives pX
without A3 feedback. The full flat offset includes d*tan(beam). Three
clipped monotone bisections start from the entire seven-degree interval.
At J=38, at most114 scalar oracle queries give errors<0.001degree per
wedge and<0.001 per source coordinate with exact compatible data and ideal
source arithmetic. The explicit finite-arithmetic contract uses J=40,
at most120 queries, and certified query/source/point/angle error budgets
2^-60. This is a proved procedure and oracle-count bound, not an executed
optical solver or practical bit/runtime result. Exact zero rotor components
are essential; this procedure is not established on the neighboring boxes.
Author/independent controls pass104/61, including clipped intermediate
targets and independent generic noise/rounding cases.

The [response envelopes](paper/SEVEN_DEGREE_RESPONSE_ENVELOPES_2026_09_27.md)
also prove F_sj/d<(49,44,56) and an exact common-factor representation
F_sj/d=Phi*w_j, Phi=1/(G*v3^3), with substantially tighter relative weight
bounds. Author/independent controls pass69/46. The
[finite-lag response cones](paper/FINITE_LAG_RESPONSE_CONES_2026_09_27.md)
preserve this common factor, direct derivative caps and synchronized rotor
increments in an exact rational support test. This extends some mixed-sign
necessary exclusions while all18 coordinates remain unknown; synchronized
cancellation strata and broad full18 confinement remain unresolved.
The generic support author/independent controls pass94/84.

[Primary sign-injectivity research](paper/SIGN_INJECTIVITY_RESEARCH_2026_09_27.md)
connects this approach to Muller et al., FoCM16(2016), DOI
10.1007/s10208-014-9239-3. The primary full text was read and retained.
Its generalized-polynomial hypotheses are not asserted for optics. A direct
zero-safe sign-class proof and117 free controls explain why full records can
remain injective when compressed moments fail, and define a bounded-secant
LP route for rotor designs missing strict sign patterns. That optical LP
route remains unexecuted.

Current evidence snapshot: seven_degree_sign_inverse_manifest_v11.json.
The older five-degree contraction and all v1-v10 evidence remain preserved.
The next targets include an inner recovery method for the neighboring
rotor boxes, broader finite-design coverage, and useful full18 confinement. The
older entries below retain the status and bounds at their own checkpoints.

**Seven-degree structural advance (09-27): actual full position responses
now have uniform positive bounds.** The [gain refinement](paper/SEVEN_DEGREE_GAIN_REFINEMENT_2026_09_27.md)
proves G=F_p>3/8, final outgoing vertical component>2/5 and normalized atan
source derivative W_z>1/50, z=p/d. The exact G floor improves the earlier
conservative one by a factor above290. With wedges and other hardware fixed,
the directly source-normalized estimator has deterministic physical source
error <=min(10,(8/3)*eta), for per-axis sup or RMS measurement error eta.
This is not a joint wedge/full18 stability bound. Author/independent checks42/62.

The [response-ratio proof](paper/SEVEN_DEGREE_RESPONSE_RATIOS_2026_09_27.md)
retains all source, beam and geometry terms and establishes
F_sj/(d*F_p)>1/50 for all three effective face-sine coordinates. These are
uniform whole-domain statements with native glass/source/geometry, all beam
angles, arbitrary rotors and every real time in the seven-degree branch
domain. Physical wedge-angle columns still include signed rotor factors.
The proof does not require positive signed propagation legs or aperture
clearance, nor does it establish them. Author/independent checks pass75/55.
Positive individual responses still
do not prove the joint covariance positive.

Combining the bounds gives F_sj>3d/400. This supplies a quantitative version
of the existing finite-lag ordering test on the seven-degree domain: a
componentwise ordered effective-sine increment has a corresponding minimum
output increment. It retains unknown wedge/rotor coordinates and does not
cover mixed-sign increments or prove complete rotor confinement.

A [source-normalized residual theorem](paper/SOURCE_NORMALIZED_RESPONSE_2026_09_27.md)
cancels every source-estimator derivative from fixed-design centered moments.
It replaces the atan-weighted covariance with a fixed Gram matrix plus a
response-ratio error and a residual-times-log-gain error. Explicit bounds
over the whole convex wedge region imply contraction and deterministic
compatible-set diameter bounds. Those actual optical regional bounds remain
unproved. Generic author/independent controls pass45/27; the independent
example includes a noncentered design and source clipping transitions.

The [generic covariance certificate](paper/SOURCE_COVARIANCE_CERTIFICATE_2026_09_27.md)
has57 author and45 independent checks. It collects both clipping-endpoint
matrix tests before Taylor enclosure and requires a separate upper norm
bound for contraction. The [actual derivative source](paper/SEVEN_DEGREE_COVARIANCE_SOURCE_2026_09_27.md)
constructs full200 u,S,r expressions with189834 nodes and all18 symbolic
inputs, including all4 native affine intervals. Controls38/50 cover source
binding, coordinate chain rules and zero wedges. No optical matrix enclosure,
record, derivative box, contractor or inverse iteration was evaluated.
Neither new component is integrated into the frozen v7 cover driver.

The v10 evidence snapshot is seven_degree_response_manifest_v10.json.
All earlier snapshots and the five-degree inverse cover remain preserved.
The unresolved target is joint seven-degree inversion and useful full18
confinement, not another proof of scalar response signs. Earlier paragraphs
below record their historical bounds; the new constants supersede the
weaker source/vertical envelopes without invalidating them.

**Seven-degree inverse follow-through (09-27): source elimination now has a
proved monotone readout, and the remaining wedge test is explicit.** The
[source-covariance note](paper/SEVEN_DEGREE_SOURCE_COVARIANCE_2026_09_27.md)
replaces the fitted Y DC feature with the raw mean of the transformed Y
record. At fixed wedges and supplied outer hardware/rotors, each axis's
derivative with respect to normalized source z=p/d exceeds
2048/2563893625 > 2^-21. This gives a unique
clipped source response through seven degrees; clipping an inconsistent
target does not fit it. The X moment change is an invertible recombination
of the four old X features. The Y change uses the available full record.

Eliminating an interior X source makes the three-wedge Jacobian a weighted
covariance; a saturated source gives its uncentered moment counterpart.
The baseline weighted Gram part is
positive; the actual coupled response-error part still needs a bound strong
enough to preserve positivity. Positive pointwise slopes alone are insufficient,
as an explicit generic countermodel shows. That example is not a native
optical counterexample. Author and independent exact controls pass 38/40.
This result retains the supplied-hardware and separated-rotor scope; neither
seven-degree inversion nor blind full18 recovery is proved.

The [generic Taylor gate](paper/COFACTOR_TAYLOR_2026_09_27.md) now bounds a
collected cofactor and its five signed cofactors with validated derivatives,
centered second-order remainders and corner cuts. Its abstract nonlinear
fixture excludes a whole interval where both the natural cofactor bound and
independent coefficient LP fail. Original source domains remain checked,
including canceled nodes; every varying input enters the remainder. Generic
derivative controls pass 60 plus 74 analytic probes, independent controls45,
and root cofactor controls32.

The [original optical binding](paper/SEVEN_DEGREE_COFACTOR_TAYLOR_2026_09_27.md)
uses unclipped coefficients on the seven-degree rational domain, intersected
with the caller's source box. Full200 symbolic construction has 88,560 source
nodes and 139,155 cofactor nodes, 400 rows and 814 inputs including formal
data. All fourteen nonlinear coordinates and four native affine intervals
remain. Source/compiler controls pass43/26 and the combined independent
audit42. Numerical optical enclosure APIs are implemented and source-reviewed
only; no optical point, record, derivative box or contractor was evaluated.
This new gate is not integrated into the frozen v7 cover driver. Its useful
optical widths and the actual wedge covariance inequality remain next targets.

Evidence: seven_degree_taylor_covariance_manifest_v9.json. It preserves v1-v8
and the five-degree cover. The broad inverse range remains |sin(ax_i)|<=7/80
(at least ±5 degrees), with seven other hardware coordinates and six rotor
settings supplied. This is a wedge range, not an initialization tolerance.
Earlier checkpoint paragraphs below remain historical where superseded.

**Seven-degree follow-up (09-27): full native beam/glass transmission is
proved on the three-wedge ±7-degree cube.** The new
[research note](paper/SEVEN_DEGREE_RESEARCH_2026_09_27.md) and
[independent audit](paper/SEVEN_DEGREE_BRANCH_INDEPENDENT_AUDIT_2026_09_27.md)
give uniform normalized exit-radicand lower bound759/16384, exit-root
lower1/5 and outgoing vertical component lower1/20, for arbitrary rotor
settings at all times. No outer hardware is fixed for this branch statement.
It does not prove positive signed propagation distances or aperture clearance.

The proof also covers the rational transformed box with wedge tangents
in±1/8 and beam tangents in±15/32, containing native±7-degree wedges and
the full±25-degree beam ranges. Positive margins allow smooth single-prism
Taylor bounds. Propagating those bounds through the actual coefficient and
inverse maps is still required. Strong monotonicity or a verified global
univalence criterion may improve on the existing contraction test, but no
such optical criterion is yet proved over the seven-degree cube.

These are exact scalar inequalities and symbolic source checks, with no
optical point or record evaluation. Transmission is now proved at7degrees;
the inverse guarantee remains the conditional five-degree result below.
The supporting research snapshot is seven_degree_research_manifest_v8.json.
All earlier evidence, including the five-degree inverse cover, is preserved.
Primary texts read include Ryu/Boyd on strong monotonicity, Nemirovski on
monotone variational inequalities, and Araya/Trombettoni/Neveu on convex
interval Taylor cuts. The [Taylor follow-up](paper/SEVEN_DEGREE_TAYLOR_RESEARCH_2026_09_27.md)
provides explicit single-prism Hessian majorants and a rank-free cofactor
remainder test. Author/independent branch checks pass30/35; derivative
primitive checks pass30. These counts verify their stated components and
do not measure completion of full18 recovery.

**Coverage and acceptance follow-through (09-27): the exact generic driver
and separate original-physics witness gate are implemented.** The broad
conditional range remains |sin(ax_i)|<=7/80, containing at least ±5 degrees
per wedge. Seven other hardware coordinates and all six rotor settings are
supplied; this is not an initialization-error tolerance or blind full18
recovery. General seven-degree and transmitted fifteen-degree inversion
remain open. The existing five-degree cover is unchanged.

The [coverage driver](paper/PROFILED_COVER_2026_09_27.md) preserves a closed
partition with every unresolved leaf accounted for. FIFO service, exact
checkpoint replay, increasing arithmetic resources and frozen-problem LP
retries implement the existing conditional positive-gap argument. A selected
cofactor is optional acceleration; the direct coefficient/LP fallback remains.
The actual verified LP gap must reach coefficient diameter plus a shrinking
tolerance before refinement. A surviving upper witness never excludes a leaf.
Author and independent abstract controls pass 75/48. Exact boundary equality,
undefined domains, persistent faults and finite budgets can remain unresolved.
The work budget counts new service quanta, not total time or memory.

The [optical source adapter](paper/PROFILED_COVER_SOURCE_2026_09_27.md)
compiles the full200 symbolic source and all400 rows, retaining fourteen
nonlinear coordinates and all four native affine priors. Its 29 source checks
and independent call-chain review bind rows, source restrictions, deterministic
allowances, threshold and optional amplitude premise. No optical observation
is instantiated, including at cover initialization. Numerical optical
start/advance/replay APIs were source-reviewed only.

The [acceptance compiler](paper/PHYSICAL_ACCEPTANCE_2026_09_27.md) reconstructs
unclipped original optical expressions, exact native-angle membership and
every source restriction. A closed-source witness is distinct from a strictly
transmitted physical witness: every original exit radicand must have a
strictly positive proved lower bound for physical-existence flags. Clipped
diagnostic roots cannot pass that test. Additional equalities require exact
zero enclosures. The implemented sufficient checker accepts exact rational
transformed coordinates and auxiliary witnesses; it can remain unresolved
on irrational auxiliaries or noncollapsing equality enclosures. It does not
certify rounded exported angles, uniqueness or measurement/storage provenance.
Full200 compilation is symbolic; no physical candidate is evaluated.
Author and independent source/primitive controls pass 73/94; the latter
also rechecks the original prism/rotor polynomial identities and source
mutation rejection. Numerical acceptance flag construction was inspected
at source level only.

The v7 snapshot is
`experiments/results/universal_attack_2026_09_27/profiled_cover_acceptance_manifest_v7.json`.
It preserves all six earlier snapshots and the five-degree cover. The new
driver and acceptance APIs close implementation gaps identified at v6,
while useful optical cell widths, a positive gap on the required regions,
global confinement/uniqueness and a complete recovery procedure remain open.
Conditional termination concerns the chosen extension profile; physical
infeasibility alone does not guarantee its gap. Earlier entries below record
their own checkpoint status and are superseded by this paragraph where needed.

**Correlated cofactor follow-through (09-27): selected row combinations now
retain cancellations before interval enclosure.** The broad conditional
five-degree inverse is unchanged; general seven-degree and transmitted
fifteen-degree full18 recovery remain open. This phase implements a route
already proved in the September 17 cofactor note, using the existing exact
LP and validated expression arithmetic. It does not claim a new circuit
completeness theorem or an evaluated optical exclusion.

The [cofactor compiler](paper/COFACTOR_EXPRESSION_2026_09_27.md) collects the
entire determinant combination symbolically, retaining shared nonlinear
atoms, before taking intervals. No determinant or gain is divided out.
The four native identity-prior rows preserve rank-deficient cases. Original
source nodes remain, so cancellation cannot erase a square-root or reciprocal
domain failure. Author and independent controls pass 64/38.

The [support extractor](paper/COFACTOR_SUPPORT_2026_09_27.md) turns a verified
positive affine dual into at most five active rows through exact sign-compatible
compression, without enumerating all row subsets. Native identity rows
guarantee rank padding; optionally trying unused optical rows first can help
regional bounds. Controls pass 62/61, including 32 independently checked kernel
directions. The [bounded cell pipeline](paper/COFACTOR_CELL_PIPELINE_2026_09_27.md)
uses a rational midpoint surrogate only to propose support, then recompiles
and encloses the actual whole-cell expression. Its controls and independent
audit pass 27/27. A deliberately false surrogate separation caused by
irrational rounding does not survive the final proof gate.

On the existing September 17 abstract family, the independent coefficient
LP has lower zero, while the composed cofactor gives surplus at least 1/4
throughout the full test interval. An independent shared-square-root fixture
also separates despite lower zero from independent coefficients. These are
generic algebraic controls, not optical measurements or new inverse ranges.

The [actual optical adapter](paper/OPTICAL_COFACTOR_COMPILER_2026_09_27.md)
compiles all 200 samples with formal, uninstantiated target/allowance inputs.
The selected distant-row circuit expands the 94,750-node source to 95,963
nodes and preserves 14 nonlinear plus 4 affine unknowns. Direct beam-offset
terms cancel for five optical rows; source-prior supports correctly retain
their surviving terms. Beam dependence inside transfer coefficients remains.
Source controls and independent audit pass 41/48. Numerical optical APIs
were source-reviewed only and remain unexecuted.

The immutable evidence is
`experiments/results/universal_attack_2026_09_27/correlated_cofactor_manifest_v6.json`.
One selected circuit and one bounded attempt are not complete global search.
Fair fourteen-coordinate coverage, useful optical cell bounds and final
original-physical acceptance remain obligations. No physical reconstruction,
new angle guarantee or useful all-case runtime follows; earlier snapshots
and the five-degree cover remain unchanged.

**Validated extension oracle (09-27): the missing coefficient interface is
implemented; numerical optical execution remains unperformed.** The broad
certified inner inverse still covers |sin(ax_i)|<=7/80, containing at least
±5 degrees for each prism. It recovers three wedges and two source positions
with seven other hardware settings and all six rotor settings supplied.
This is not an initialization-error tolerance or blind full18 guarantee.
General 7-degree recovery and the requested transmitted 15-degree full18
inverse remain open. The existing restricted small-angle full18 constructive
results below are unchanged.

The new [source-bound extension compiler](paper/CENTER_EXTENSION_COMPILER_2026_09_27.md)
builds the previously proved clipped extension as an immutable arithmetic
program. The complete 200-sample symbolic source produces 94,750 expression
nodes, 400 coefficient rows and 8,000 named outputs. All fourteen nonlinear
coordinates, physical prism order, speed collisions, zero wedges and the
complete four-variable native affine box remain. Actual source equations,
priors, input domains, row mappings and extension mode are replay-bound.
Narrower source affine constraints are recorded separately; a full-box upper
witness does not establish their membership.

The [generic arithmetic](paper/CENTER_EXPRESSION_ARITHMETIC_2026_09_27.md)
uses exact-integer outward dyadic operations and supports zero square roots.
Explicit nonnegative radicand guards matter: an algebraically zero expression
can otherwise retain a negative interval lower bound at every precision.
The author control passes 103 checks and independent audit 102. Symbolic
source specification, compiler control and independent compiler audit pass
62, 34 and 104 respectively. No optical coefficient program was evaluated.

The [accuracy bridge](paper/CENTER_PROFILE_ACCURACY_BRIDGE_2026_09_27.md)
uses the actual verified LP gap, including unfinished optimization error.
Whole-cell coefficient intervals can be used directly without adding the
large global spatial modulus. Rowwise transport can improve scalar spatial
allowances; these comparisons were executed on abstract affine problems.
The [final binding](paper/CENTER_AFFINE_BINDING_2026_09_27.md) connects the
actual source enclosure, targets, deterministic row allowances and LP proof.
The data-required amplitude premise and strict exclusion gate share the same
profile threshold. Their author controls pass 50 and 16 checks; the joint
independent audit passes 78. The complete numerical optical adapter was
source-reviewed only, including its final combined exclusion gate.

An extension upper fit or clipped root cannot certify original physical
membership or the original root margin. No new observed-data contraction,
physical reconstruction, optical interval-width/runtime measurement or
inverse angle guarantee follows. Increasing precision at a fixed-width
cell does not remove genuine cell variation. The adaptive driver, fair
coverage/progress control and final original-physics acceptance remain open.
The new immutable snapshot is
`experiments/results/universal_attack_2026_09_27/center_oracle_manifest_v5.json`:
eight source-bound reports, all four earlier snapshots preserved, and the
five-degree cover unchanged.

Research at that checkpoint favored selected five-row cofactor separators compiled as whole
expressions, retaining correlations before interval enclosure. September 17
already proves pointwise circuit completeness with native-prior rows; the
next task was source-bound support extraction and scalar-DAG compilation,
without determinant division, now implemented above. A fair fallback when
chosen supports fail remains open.
Smooth interval Taylor cuts require justified derivative domains and cannot
silently cross critical roots or clipping seams. Another coefficient-box
upper policy would not improve the existing exact optimistic lower bound.

**Output/physical-fiber follow-through (09-27): stronger certified components,
with earlier results explicitly credited.** The broad angle statement is
still the conditional inner inverse at about5degrees per prism, with outer
hardware and rotors supplied. It is not a5-degree initialization tolerance
or blind full18 guarantee. General7-degree blind recovery and the requested
15-degree endpoint remain open.

The new [refined output inequality](paper/OUTPUT_TRANSMISSION_REFINEMENT_2026_09_27.md)
proves |Y|+C*E2>=165063/3125, C=1024942546639/995400000, for every
axis/sample on the original rational angular outer prior. The retained
terminal inequality has threshold150. These are position-unit thresholds,
not angles. At absolute compatible model-output cap50, the normalized
second-root floor improves to561437452800/216262877340829 (about.002596).
Both later roots are positive below52.82016. No actual record is classified.
The source-checked implementation preserves all18 unknowns, all original
obligations and critical branches where permitted by the outputs. Full200
symbolic construction adds400 absolute-value variables,400 equalities and
1600 inequalities, preserving the earlier weaker cuts. The root control
passes43 checks, the scalar/sign proof has an independent17-check audit,
and the complete refined source adapter has an independent108-check audit.

The [exact affine-profile backend](paper/AFFINE_FIBER_PROFILE_2026_09_27.md)
keeps distance, gap and both source coordinates free in their complete box.
Exact rational LP primal/dual witnesses give interval-family residual bounds
without dividing by gain or assuming full rank. Source extraction binds all
400 output rows; row selection is caller-bound. Limited resources return
sound unresolved bounds. Its140-check control and93-check independent audit
pass, including independently reconstructed branch LP witnesses. This is
an abstract interval-coefficient optimizer and symbolic source adapter;
no optical inverse or actual optical contractor was run.

The [profile bridge](paper/PROFILED_PHYSICAL_FIBER_PROGRESS_2026_09_27.md) and
[data-dependent extension](paper/PROFILED_DATA_MARGIN_PROGRESS_2026_09_27.md)
connect that backend to explicit clipped-center formulas and revised
constants. At that checkpoint the extension coefficient oracle and its
validated arithmetic were implementation obligations, now addressed above.
A fit to an extension is not physical
membership. The associated controls pass22/51 and root independent review20.

Novelty correction: September17 already established the |Y|<=50 root
separator, full4 affine fiber,14D search, zero-gain handling and bounded-data
linear regularity. The new work strengthens constants and implements the
checked interval LP/graph interfaces; it does not first prove those ideas.
Likewise [quarter-power sharpness](paper/CRITICAL_HOLDER_SHARPNESS_2026_09_27.md)
was already known. The new46-check construction places all wedges strictly
below15degrees and prevents leading-order cancellation for every native
geometry/source. None of these results closes T5 or useful global T7.

The [new primary-source review](paper/AFFINE_FIBER_EXTERNAL_RESEARCH_2026_09_27.md)
checks adjustable robust optimization, AE interval systems, nonsmooth
variable projection and interval least squares against this model's exact
assumptions. It recommends a small optional finite policy of affine witnesses
to tighten upper bounds across coefficient uncertainty. It explicitly cannot
improve the exact exclusion lower bound on the same independent coefficient
box. Correlations from the same upstream hardware remain the main obstacle;
rank/full-convexity assumptions from other solvers are not imported here.

The [finite-policy addendum](paper/COEFFICIENT_PARTITION_PROFILE_ADDENDUM_2026_09_27.md)
proves a rank-independent approximation bound and an exact16-box example
with upper5/3 reduced to1 while the lower remains5/9. The policy chooses
one shared hardware vector for each coefficient box; it cannot masquerade
as a single identified hardware system for the whole parent. Inner solver
gaps must be certified as well as coefficient widths. No policy wrapper is
implemented. The practical priority remains a validated center coefficient
evaluator or stronger correlated optical-cell bounds.

The independent policy audit passes70 checks, including the important
physical-image quantifier correction and a zero-width/zero-pivot counterexample.
The consolidated snapshot is
`experiments/results/universal_attack_2026_09_27/data_physics_bridge_manifest_v4.json`:
12 passing source-bound reports, preserved earlier snapshots and unchanged
five-degree cover. No optical recovery trial is included.

**Optical follow-through (09-27): stronger actual source equations, with independent audits.**
The [prior-wide margin theorem](paper/OPTICAL_MARGIN_SHARPENING_2026_09_27.md)
reduces the rational angular outer graph's final slope cap to10000/27
(about370.37) and output cap to32216945/432 (about74576.26), factors56.4
and1131.2 below the old builder bounds. No hardware is fixed and no extra
transmission margin is imposed. A separate native18-degree wedge/25-degree
beam theorem gives a final slope cap46875/256; its constants are NOT applied
to the larger rational angular outer graph. Original and independent
margin controls pass73 and86 checks. A further independent theorem proves
[every prism's transfer gain is below11/8](paper/UNIVERSAL_TRANSFER_GAIN_INDEPENDENT_AUDIT_2026_09_27.md);
that additional gain improvement is not yet substituted into the frozen overlay.

The [source-checked optical compiler](paper/OPTICAL_RELAXATION_COMPOSITION_2026_09_27.md)
now combines four layers: prior-derived forward-direction inequalities,
[Lorentz supports and glass-root corner envelopes](paper/OPTICAL_CONE_ENVELOPES_2026_09_27.md),
[paired-axis energy/material constraints across samples](paper/OPTICAL_SHARED_ENERGY_GRAPH_2026_09_27.md),
and [division-free shared distance/gap identities](paper/OUTPUT_GEOMETRY_BRIDGE_2026_09_27.md).
All18 coordinates, native priors, source constraints, zero wedges and rank-zero
geometry branches remain. Full200 symbolic compilation retains17243 original
variables plus27 auxiliaries, with4800 margin cuts and bounded selections of
the other layers. It does not run a contractor on optical equations or
instantiate observations. Root composition controls pass40 checks and the
independent composition audit passes38. Cell-local envelopes stay bound to
their source rectangles and cannot be attached to a broader hull.

The [progress note](paper/COMPOSED_OPTICAL_PROGRESS_RECOMMENDATION_2026_09_27.md)
derives an explicit Holder1/4 output modulus through later critical exits,
and a clipped mathematical extension agreeing on transmitted states. This
supports a future enclosure fallback using only18 physical-coordinate
subdivisions. The constant1.2e11 is intentionally conservative and impractical
alone. That earlier phase supplied no extension evaluation or implementation,
physical fit, observed
separation bound, practical convergence, or wider-angle inverse guarantee is
claimed. Its26 algebra checks and29 independent review checks pass. The
broad inverse remains conditional at about5degrees; general
7/15-degree blind full18 recovery is open. New snapshot:
`experiments/results/universal_attack_2026_09_27/optical_symbolic_manifest_v3.json`.
Earlier manifests, reports and the five-degree cover remain preserved.

**Joint follow-through (09-27): correlation now survives between components.**
The [joint cell pipeline](paper/JOINT_CELL_PIPELINE_2026_09_27.md) composes
the shared polynomial network, guarded rotor ports, closed sign/CID branches
and [singleton-aware verified LP](paper/VERIFIED_LP_CONTRACTOR_2026_09_27.md).
Child-local cuts are discarded before hulls; unresolved branches survive
budget exhaustion. Original constraint priorities rotate between capped
passes so later constraints are not permanently starved by an inert prefix.
Actual-LP witness rebinding prevents a proof for another LP from authorizing
contraction here. Independent source, budget and branch-scope audits accompany
the controls.

The [new rotor bound](paper/ROTOR_BERNSTEIN_BOUND_2026_09_27.md) keeps pairwise
cancellation in an exact rational squared-norm polynomial, then uses a
Bernstein quotient bound with validated fallbacks. A specified adjacent-index
control improves the squared bound20,004 times. Combining it with verified LP
on three correlated RHS variables raises a shared lower bound from1 to
1.999900009999 and excludes the whole synthetic region p<=1.999; either
interval propagation or the uncut LP stalls. A separate rotor-only graph
control narrows a shared interval more than35 times beyond the equally
budgeted sign-split LP. These are **nonoptical algebraic controls**, not
unknown optical-hardware contractions. Full18 reconstruction and the
15-degree inverse remain open; the established conditional five-degree
theorem is unchanged. No optical record, forward call or recovery trial was
executed. Long-span exact polynomial bounds are costly: one degree398
two-normal calculation took about11.9seconds, not a full-graph benchmark.

The previous immutable manifest remains unchanged. The new follow-through
snapshot is `experiments/results/universal_attack_2026_09_27/` +
`joint_shared_cell_manifest_v2.json`. It separately records the new proofs,
controls, independent audits and intentional living-log updates. It also
records one earlier finite-escape note that differs from the previous
snapshot, with a new10-check scalar-audit report bound to its current hash;
the old report and manifest were preserved rather than overwritten.

**Latest parallel work (09-27): use the checked components together.** Root
and three agents used all four available slots for mathematical construction,
outside research and independent audits. The active goal is full18 compatible
reconstruction, including at least15-degree transmitted systems. No physical
record or inverse trial was evaluated; stopped campaigns were not restarted.
The broad established inverse remains the conditional five-degree inner
theorem. New15/18-degree results concern feasibility, not blind recovery.

The [wide-angle retraction](paper/WIDE_ANGLE_OPERATOR_ATTACK_2026_09_27.md)
is now proved nonexpansive in an explicit weighted maximum norm. It extends
through18degrees on every positive common-margin class, with deteriorating
weights as the margin vanishes. Euclidean nonexpansiveness is false. A
[separate exact full18 saddle family](paper/WIDE_ANGLE_RESEARCH_2026_09_27.md)
has all target wedges active, separated target speeds and ordinary continuous
glass: a wrong flat candidate has zero gradient but positive residual. A
phase-pair curvature check gives a strict escape direction. This establishes
no global convergence. Both original structural results
passed [independent mathematical audit](paper/WIDE_ANGLE_INDEPENDENT_AUDIT_2026_09_27.md).

The follow-through now proves a
[finite feasible observed-data escape](paper/FINITE_FEASIBLE_SADDLE_ESCAPE_2026_09_27.md)
for that specified15-degree family, including deterministic per-axis RMS
noise<=1e-5. An observed mixed-Hessian test selects a phase; a rounded rational
direction and two feasible steps of length1e-7 guarantee that one decreases
the actual full-record summed objective. An explicit interval-comparison
accuracy suffices to certify decrease>=1/(1.2*10^15). This conservative gain
is tiny. No optical step was executed, no practical iteration count is claimed,
and repeating the step is not a proved global reconstruction method.

The [rotor event theorem](paper/FINITE_RECORD_ATTACK_2026_09_27.md) eliminates
terminal speed, phase and amplitude exactly for supplied fixed-upstream normal
rectangles, including isolated feasible speeds and zero wedges. Its rational
prototype has explicit resource caps; full native algebraic-coefficient
execution and practical complexity remain open. The
[short disk certificates](paper/ROTOR_DISK_CERTIFICATES_2026_09_27.md) now extend
to [whole continuous speed/upstream cells](paper/ROTOR_CELL_EXCLUSION_2026_09_27.md)
when outward normal bounds are supplied. The cell checker also gives necessary
cuts on shared RHS variables and passed926 independent exact controls.

[Shared polynomial contractors](paper/SHARED_POLYNOMIAL_CONTRACTORS_2026_09_27.md)
retain all18 unknowns, propagate constraints through one shared-variable graph,
and apply constructive disjunction without dropping unresolved slices. Exact
controls demonstrate shared-coordinate contraction and exclusion. Root's
rotor/contractor integration control passes after exact affine RHS substitution;
independent RHS intervals can lose this correlation. These are synthetic
coefficient controls. Sharp bounds and progress on unknown optical hardware
have not been demonstrated by them.

The [graph/rotor interface](paper/REDUCED_ROTOR_PORTS_2026_09_27.md) is now
implemented in both directions: whole graph cells supply each rotor's tangent
coordinates without fixing upstream hardware; selected supports give linear
constraints back to the shared graph. The adapter checks the actual rotor
identities, all18 coordinates, sampling, physical order and native binary
glass endpoints. Local cuts are bound to their source cell and, when needed,
their wedge sign. An independent125-check audit passes. This closes the
interface obligation; strong contraction from observed data and global
progress still need to be established.

The [primary research notes](paper/GLOBAL_INVERSION_RESEARCH_2026_09_27.md)
support CID/ACID propagation and certified cuts as the next joint-domain path.
Related [finite-sample spectral research](paper/FINITE_RECORD_RESEARCH_2026_09_27.md)
needs proved optical-tail/amplitude/collision premises. The
[sparse-SOS graph cover](paper/SPARSE_GLOBAL_RESEARCH_2026_09_27.md) passes
running-intersection and support audits for the full200-sample graph, but its
12,014 blocks expose substantial computational cost; no hierarchy was solved.
The [composed-clipping criterion](paper/COMPOSED_CLIPPING_CERTIFICATE_2026_09_27.md)
can optimize weights directly. However, the specified conversion of saved
Euclidean bounds to an absolute comparison matrix fails on all637 existing
five-degree leaves. Their original certificates remain valid. Sharper signed
or correlated bounds are necessary for that route; reweighting alone does
not recover information already discarded by absolute values.

**Latest practical requirement (09-23): at least FIFTEEN DEGREES per
prism.** The five-degree result is an intermediate result, not the user's
required endpoint. Preserve the broad remaining hardware requirements;
do not silently exchange larger wedges for narrow beam, glass or rotor
priors. A 15-degree inverse guarantee is not established. The existing
10-degree total-internal-reflection example lies inside the requested
15-degree box, so the extension must distinguish physically transmitted
configurations and their margins from combinations producing no outgoing
ray. Existence of useful transmitted 15-degree families does not prove
their inversion. Global uniqueness remains deferred.

**Latest wider-angle progress (09-23): a quantitative admissibility chart,
not a fifteen-degree scan inverse.** The
[common-margin chart](paper/QUANTITATIVE_ADMISSIBILITY_CHART_2026_09_23.md)
parameterizes the entire three-wedge domain with the declared exit-normal
margin, using triangular conditional intervals across both axes and all
retained times. Glass stays continuous in 1.3..1.8 and both original beam
ranges remain. A new convex-combination identity propagates the SAME
positive margin to outgoing rays. The chart preserves physical feasibility
without assuming the whole independent fifteen-degree cube transmits.
Its conditional widths have positive explicit lower bounds. This strengthens
the earlier strict-domain topology; reverse-order flattening was already known.
The contractor and algebra controls are in prism_margin_domain.py and the
09-23 results directory. Actual chart execution on an optical record was not run.

The [exact individual-prism inverse](paper/EXACT_PRISM_INVERSE_2026_09_23.md)
has bounded angular partials through fifteen degrees, with no small-wedge
expansion. It requires the intermediate ray directions, which the screen
record does not directly provide. For the fifteen-degree class with exit
normal squares at least .01, a separate telescoping argument bounds source
readout error by less than 138 times the record-error allowance, ONLY when
all other parameters are fixed correctly. This is not joint parameter error.
The missing step remains an observation-driven operator with a proved global
convergence/residual guarantee. The chart's triangular clipping was not known
nonexpansive on09-23; the09-27 weighted-norm result above closes that specific
gap. The five-degree data contraction still cannot be transplanted.
No optical trials, campaign, or main-manuscript edits were made.

**Material realism (09-23):** the user expects ordinary optical materials
and air/vacuum-like surroundings. Keep prism refractive indices as
continuous unknowns; do not hard-code a glass identity or a convenient
index to obtain the 15-degree result. Indices around 4, 5 or 6 are a
diagnostic warning outside this intended setup, not an acceptable way to
fit the scan. The existing native prism prior is 1.3..1.8 and the forward
model currently fixes the surrounding medium to 1.0; the latter is an
explicit model assumption, not recovery of an unknown medium. No new
material bounds or medium inference are established by this preference.
The existing TIR obstruction already uses the native upper glass endpoint
1.8, so the wider-angle proof obligation remains within the intended range.

For scope questions: the present five-degree inner theorem covers screen
distance50..200, gap2..15, independent incoming beam angles±25deg, glass
1.3..1.8 and source positions±5. The source-to-first-prism distance6 and
prism thickness3 are fixed model constants. Nominal rotor magnitudes are
.3..3.5Hz, pairwise magnitude separation at least.5Hz, with a small proved
candidate buffer; both rotation directions are allowed. The sampling
proof allows every rotor phase, while the current native output prior
still uses phase offsets±18deg. Speed, phase and the seven free hardware
coordinates remain candidate inputs to this inner theorem; their blind
recovery is not thereby solved. Lengths are in consistent model units.

**Latest user priority (09-22): defer global uniqueness.** The immediate
milestone is constructive recovery: find one admissible full18 explanation
from the actual 200-sample moving scan, then verify its physical margins
and agreement with the observations under deterministic error bounds.
Local certification may be used with its explicit neighborhood hypotheses.
Global exclusion of all other explanations is later work and must not
block this milestone. A verified candidate is not yet identification of
the generating hardware to native <.001. The full global criterion remains
an eventual obligation; existing campaign/manuscript stops are unchanged.

**Latest wedge release (09-22): at least FIVE DEGREES per prism, over the
full broad hardware range.** The user rejected the old .716-degree cap.
The [new exact-slope construction](paper/BROAD_WEDGE_RELEASE_2026_09_22.md)
now certifies all signed wedge sines within 7/80, containing ±5 degrees,
with the original glass1.3..1.8, distance50..200, gap2..15, source±5 and
independent beam±25-degree ranges. Generic separated speeds and arbitrary
phases remain the sampling assumptions; no special rotor schedule was
substituted. All seven free hardware variables and rotors are still fixed
candidate inputs. It recovers the three wedges and two sources within
that fixed fiber in the compatible noiseless case, from any feasible
inner start. For noisy or incorrect outer candidates, the projected
inner candidate still requires the full-record compatibility check.

The complete cover has 637 hardware/beam leaves, 269,360 effective-state
subboxes, no pending boxes, worst q<.994878 and physical margin>.470793.
The exact partition, all contraction/readout constants and dependency
ancestry passed independent audits. Final report:
`experiments/results/constructive_turbo_2026_09_22/wedge_release_bounds_affine_cover4.json`;
independent audit: `broad_wedge_cover_audit.json` in the same directory.

The new map uses atan(y/d-B0/d), exact angular and centered-position
cancellation, monotone scalar source inversion in p/d, and known glass
and geometry-lever preconditioning. Geometry-affine quantities use a
proved eight-vertex polytope reduction. This is an exact nonlinear
construction, not a truncated forward approximation. No optical record,
physical inverse iteration, or recovery trial was evaluated.

**Precision and remaining gaps:** at native per-axis RMS error1e-5,
8192 feasible updates each accurate to1e-12 and final normalized source
readout error1e-12, conservative conditional bounds are wedge-angle vector
error<.011790 degrees, source X<.015707 and source Y<.029137 in native
length units. These are looser than the old small-wedge bounds. Actual
update arithmetic remains an execution gate. The five-degree full-record
compatibility correction has NOT been composed, and outer hardware/rotor
selection remains open. The .716-degree full-record theorem below remains
valid with its original constants; it is not automatically extended by
this wedge release. Full18 recovery is not complete.

**Latest compatibility result (09-22): one full-record affine correction
turns the feature profile into a fixed-fiber decision procedure.** For ANY
candidate choice of the seven free hardware coordinates and rotors in the
declared ranges, the
[profile-affine theorem](paper/PROFILE_AFFINE_COMPATIBILITY_2026_09_22.md)
gives the following contract at native sup error eta=1e-5. After the
certified source-first construction and acceptance of an exact affine LP
gap <=1e-8, it either returns a physical output with full-record residual
<1.046e-5, or excludes EVERY eta-compatible source/wedge choice at those
fixed outer coordinates. An unaccepted numerical LP proposal remains
unresolved. This is not an exclusion of other hardware/rotor choices.

The proof uses a data-derived center, with every compatible inner point
within 7e-6 in each wedge sine and 7e-5 in each source. A whole-prior,
all-rotor Taylor calculation bounds the affine remainder by
2.241659173e-7. The exact LP primal/dual gap gives the candidate-or-exclusion
dichotomy without assuming that the proposed outer hardware is correct.
At fixed wedges, the two affine source variables can also be eliminated
by exact interval inequalities; this does not necessarily make the LP
faster than its original five-variable form.

Uniform 192-bit LP value/Jacobian error is <5.685e-54; final native-output
rounding adds <1.894e-26. Native prior boundaries are handled explicitly.
The earlier feasible source-first update error <=1e-12 and final source
readout error <=1e-12 remain separate execution gates. No optical record,
physical inverse iteration or optical LP was evaluated. Wedge sines
<=1/80 and the declared absolute-speed separation remain restrictions.
The seven remaining hardware coordinates and unknown rotors still need
a constructive selection method; broad full18 recovery is not complete.

**Preceding inner-solver improvement (09-22): two explicit source formulas and
a three-wedge contraction.** The
[source-first construction](paper/NATIVE_SOURCE_FIRST_CHART_2026_09_22.md)
replaces both source iterations by affine division and clipping. At wedge
sines up to 1/80, its global contraction is q<=.803148672 over the same full
glass, geometry, source and independent beam ranges below. There is no
near-true center for these five coordinates. The seven other hardware
coordinates and the rotors remain fixed candidate inputs.

With the CORRECT remaining hardware and rotors, per-axis input error 1e-5,
128 feasible updates each accurate to 1e-12, and final source readout error
1e-12, the conditional wedge-angle vector error is <.000394185 degrees.
The [source readout bounds](paper/SOURCE_FIRST_READOUT_2026_09_22.md) give
px error <5.735e-5 and py error <6.961e-5. Thus the data-derived candidate
is within 7e-6 in every wedge sine and 7e-5 in each source coordinate of a
compatible member in the same fiber. This is a different map from the
five-coordinate projection below; its noisy candidate can differ.
No optical record or physical inverse iteration was evaluated.

An [exact physical counterexample](paper/FEATURE_PROFILE_RESIDUAL_2026_09_22.md)
shows why selected-feature agreement alone is insufficient: its unique
five-feature candidate has full-record residual 2*eta although a physical
candidate with residual eta exists in the same fiber. This is a limitation
of that fixed feature profile, not of physical compatible reconstruction.
The source-first localization permits a separate full-record correction;
the seven-variable/rotor selection problem remains the outer obligation.

**Preceding hardware breakthrough (09-22): five coordinates recovered from
DIRECT native features without a local hardware center.** The
[native-feature construction](paper/NATIVE_FEATURE_CHART_2026_09_22.md)
keeps all higher optical terms tied to the same physical parameters.
For each candidate choice of seven free coordinates
(n1,n2,n3,d,g,t_x,t_y), a global projected contraction determines all
three wedge sines and both source positions from five fitted scan features.
It includes the full original glass, distance50..200, gap2..15, sources±5
and two independent beam ranges±25deg. Wedge sines are bounded by1/80
(about.716deg), with either sign and zero included. The certified q is
.948900193; at1/160 (~.358deg), q is.652224271.

The 200-sample feature design needs candidate absolute speeds in
[.2995,3.5005]Hz with absolute gaps>=.499Hz; every phase is allowed.
This covers component errors<=.0005Hz about the .3..3.5Hz/absolute-gap.5
class. Rotors are candidate inputs to this chart, not recovered by it.
Exact comparison algebra gives Gram>=.72. Complete interval covers of
36/181 geometry boxes and every independent beam pair prove the physical
contraction; independent audits checked2304/11584 pair bounds. These are
full covers, not sampled hardware cohorts.

The [physical projector](paper/PROJECTED_FEATURE_CHART_2026_09_22.md) needs
at most4 scalar quadratic pieces and independently checks32 vertex
inequalities. Enclosed algebraic gains yield rational feasible iterates
with certified projection error.512 feasible updates with total update
error<=1e-12 give fixed-point errors below6.13e-11 at1/80. This is a
conditional execution bound: no optical feature iteration ran. At input
error1e-5, the feature-coordinate bound is about.000399462, NOT native
hardware accuracy or a full-record residual.

**What remains:** selecting the seven free hardware coordinates and rotors,
retaining noise slack/physical correlations, and certifying the complete
scan residual. A projected fixed point can fail to fit noisy features at
a prior boundary; even matching those features does not prove a full scan
match. Exact-flat-coefficient beam/distance reductions are separate and do
not reduce this native problem to four unknowns. The new broad hardware
chart must not be silently combined with the older local-hardware blind
theorem below. No completed broad full18 recovery or T5 closure is claimed.

Current evidence: constructive_turbo_2026_09_22/direct_feature_a160_final.json,
direct_feature_a80_final.json, direct_feature_cover_audit.json,
native_feature_chart.json and projected_features.json. The attempted
stronger q<.8 cover is explicitly INCOMPLETE; it does not improve the
current global q. See DIARY for the free-polynomial resonance/noise
obstructions that motivated this direct physical feature construction.

**Preceding turbo result (09-22): broad blind rotor entry and scan
compatibility now compose at hard input error1e-5.** The
[broad compatibility proof](paper/CONSTRUCTIVE_BLIND_COMPATIBILITY_2026_09_22.md)
uses no unknown speed centers or physical ordering. Centered two-lag Hankel
pencils and32 finite three-tone GN updates bound speed error<1.985e-5Hz and
midpoint phase error<.000564rad. At most six constrained affine minimax
programs then give one physical output with full-record residual<2.042008e-5,
within the declared output budget2.1e-5. This is compatibility, not native
accuracy to the generator; output tolerance is larger than input noise.

SCOPE. The speed class has .3<=|N|<=3.5Hz, signed pair gaps>=.5Hz and
modulo1 gaps>=.02Hz. True phases±17deg, declared wedge signs(+,+,-), and
the prior shape halfbox near0.45deg remain. In particular hardware distance
is100±.05: broad hardware localization is not solved. Native speed endpoints
and all candidate parameter bounds are preserved. Affine/output arithmetic
are uniformly certified; the LP gap is verified per input with exact
primal/dual bounds. The [frontend arithmetic certificate](paper/FRONTEND_ARITHMETIC_2026_09_22.md)
now checks rational Hankel/eigenvector proposals,32 interval frequency
updates and phase sectors on accepted inputs. It charges1e-12 per update;
the full residual bound is2.04200751301657e-5. Universal numerical
acceptance/runtime is still unproved. No optical records
or LPs were run and useful whole-solver runtime is unproved. Independent
audits passed for the spectral bounds, GN, physical algebra and arithmetic.

The hardware attack also produced two global COEFFICIENT results:
[a unique normal-incidence cubic index inverse](paper/GLOBAL_NORMAL_CUBIC_2026_09_22.md)
over the full glass prior (with all8 hardware formulas), and
[observable hardware coordinates](paper/OBSERVABLE_HARDWARE_ROUTE_2026_09_22.md)
that eliminate wedges/sources without zero-coordinate exclusions. Two
globally monotone beam inversions reduce the remaining physical unknowns
to(n1,n2,n3,d,g) when the last wedge is nonzero. These need justified
finite-record Taylor coefficient bounds; they are not yet broad noisy
hardware recovery. T5 and practical full-prior recovery remain open.

**Preceding user correction and result (09-22): remove the impractical
restrictions; the bird is not free yet.** The
[larger-wedge composition](paper/CONSTRUCTIVE_RELEASE_2026_09_22.md) now
connects observed DFT phase entry, two structured rotor refits and joint
full18 correction near0.45-degree wedges, using the original200 samples.
Preserving correlations, cancelling leading rotor mismatch and retaining
fourteen separate error bounds gives q<0.485978. At hard input error about
7.62958e-12,120 outersteps give allnative errors<.001 (largest bound.0009).
The separate whole-record residual bound is4.45413e-5, not the input eta.

This is an initialized exact-execution mathematical result, not a complete
practical solver. The new inverse arithmetic/output rounding are not yet
certified and no optical record was evaluated. True speeds remain within
5e-6Hz of(3.1,1.2,.1), phases±17deg, with fixed order/signs and true shape
in half the declared broad local box (distance100±.05). Many coordinates
already have prior ranges below.001. Original-full-prior entry and practical
noise/compatibility remain open. Independent component and composition
audits support the result. Evidence: `constructive_release_2026_09_22/joint14.json`.

The [new exact physical pair](paper/CONSTRUCTIVE_PRESEED_RELEASE_AMBIGUITY_2026_09_22.md)
proves a uniform native.001 generating-distance guarantee is impossible
at hard noise1e-5 on the saved wedge ladder through7.18deg: two generators
with distance difference.0022 have overlapping noise balls. This is an
all-rotor interval/coefficient proof, not an optical trial or noiseless
nonuniqueness. It does NOT block one compatible explanation, which remains
the immediate user objective. Do not turn this accuracy obstruction into
an excuse to defer constructive compatibility.

**Preceding result (09-22): all finite tolerance links close on a restricted
full18 family; practical recovery remains open.** The
[complete tolerance proof](paper/CONSTRUCTIVE_FULL18_TOLERANCES_2026_09_22.md)
links data-derived Prony/quartic rotor entry, physical preseed entry,
joint rotor/shape correction, validated arithmetic, finite stopping and
whole-record output error. Tail subtraction before each rotor refit removes
the earlier fixed-rotor bias. The coupled error factor is<.178;40 outersteps
give every native error<2.574e-9 and full-record residual<2.074e-10.

Scope is essential: epsilon=2^-24 (largest nominal wedge~4.3e-7deg),
total input discrepancy<=1e-40, true speeds within5e-6Hz of(3.1,1.2,.1),
fixed physical order, phases±17deg and the declared small shape family.
The frequency prior is already narrower than native.001. Input noise and
output compatibility are distinct: the latter is1e-9. These are sufficient
mathematical bounds, not practical or necessary hardware specifications.
All18 coordinates vary, but the original full18 prior is not covered.
Validated kernels and interval/polynomial controls pass independent audits;
no optical record was evaluated and runtime is unbenchmarked. The old
binary64 driver and solve18 are not certified implementations of this new
variant. Global uniqueness remains deferred; T5 stays open.

**Earlier larger-wedge conditional result (09-22).**
The [tolerance ledger and proof](paper/CONSTRUCTIVE_TOLERANCES_2026_09_22.md)
now supply exact rotor conditioning, a finite quintic preseed coefficient
allowance, nonlinear twelve-shape coefficient boxes, and validated optical
Taylor remainders and derivatives. Their composition proves local physical
correction at fixed exact speeds (3.1,1.2,.1) and arbitrary fixed exact phases.
For a whole inner shape family near 0.45-degree wedges, hard position error
1e-12 gives all twelve native errors below0.000394, with contraction q<0.503.
The common starting center allows true distance to vary by0.05, so the
guarantee is not just an already-accurate prior. It assumes the declared local
family and exact evaluation of a projected correction, not the prototype's
floating execution. Whole-record noisy compatibility is a separate check.
Blind rotor/basin entry and arithmetic remain open for that larger-wedge
case; the subsequent complete chain above uses a smaller family. All
controls use coefficients/interval boxes; no optical observation records or
archive recovery runs were used. The stringent precision allowance is not
yet a useful full-prior operating specification.

**Constructive full18 theorem and executable prototype.** The
[cubic extension](paper/CUBIC_CONSTRUCTIVE_INVERSE_2026_09_22.md)
removes the nonzero-source restriction: a scalar quintic and explicit formulas
give a normal-incidence hardware preseed, and the40-row cubic physical jet has
rank12 even at centered source. A129-column quartic frequency fit and the same
200 samples retain the O(epsilon^2) full18 defect-contraction proof on a
nonempty small-wedge/near-normal/nonaliased open class. ALL18 remain unknown.

The complete numerical proposal/correction prototype is now implemented in
`constructive_inverse.py`, `constructive_rotors.py`, `physical_jet.py` and
`normal_cubic_init.py`. A declared NONOPTICAL polynomial record with all18
unknowns is inverted to2.15e-11 native discrepancy. An artificial omitted
degree5 term gives a0.001297 seed error, reduced below1e-9 after two correction
updates. Independent audits and coefficient/rational/polynomial controls pass.
The opt-in interval point validator exists but no optical recovery/validation
was run. Finite local fixed-rotor wedge/noise bounds are now given above;
blind full18 and numerical arithmetic bounds remain unproved. The prototype
is not integrated into solve18 and archive counts are unchanged.

[Exact coefficient sensitivity](paper/QUADRATIC_JET_SENSITIVITY_2026_09_22.md)
shows why all coefficient rows and the cubic terms help, while proving a real
small-wedge noise cost. These are coefficient-box tangent bounds, not nonlinear
native-record guarantees. The selected12 quadratic inverse has a documented
wrong-root control; its unused coefficients expose the mismatch.

The preceding [native moving-record construction](paper/CONSTRUCTIVE_MOVING_SCAN_2026_09_22.md),
section7, proves data-initialized convergence with ALL eighteen coordinates
unknown on a nonempty open class of sufficiently small wedges, near-normal
beam incidence and nonaliased cubic rotor frequencies. Prony seeding and
cubic frequency refinement feed an explicit [quadratic physical inverse](paper/QUADRATIC_PHYSICAL_INITIALIZER_2026_09_22.md)
and a local twelve-coordinate extension for unknown beam angles. The
approximate inverse has a C1 defect O(epsilon^2) under A=epsilon*alpha;
an exact-model correction cancels that defect and contracts. The same
200 samples are used; no frozen responses or truth initialization are
assumed. Fixed physical-order branches are enumerated outside the loop.

Three independent audits and exact symbolic/rational controls pass.
At N=(2.4,.7,.2) the63 cubic spectral columns are exactly orthogonal
on the ideal200-point grid. Its original helper is `quadratic_init.py`;
the source-independent cubic extension above is now the stronger proposal
route. This remains a scoped theorem, not full-prior recovery.

**Completion targets and current status:**
[FINISH_LINE.md](paper/FINISH_LINE.md) lists ten necessary targets, their
acceptance criteria and dependencies. The major open scientific gates are
global finite-scan localization, complete exceptional/boundary decisions,
and useful global computation. Alternative research routes are not additional
finish-line requirements. No completion percentage is justified.

**Latest (09-22): direct physical readout; T5 STILL OPEN.**
[Structured inverse](paper/STRUCTURED_LAST_PRISM_INVERSE_2026_09_22.md)
recovers A,n,d,Q at known normal incidence from eight signed frozen
values using one strictly monotone scalar equation, bypassing the free
25-coefficient fit. If Q=0 is known, one odd pair also proves normal
incidence, reducing the exact protocol to five values. Directed-interval
controls are reproducible, but their noise widths still become poor.
[Independent physical audit](paper/STRUCTURED_SLICE_NOISE_AUDIT_2026_09_22.md)
proves a separate real limitation: two moderate-wedge slices with index
separation .0022 differ everywhere by less than 3.508e-6. At allowance
11/6272000, those slice data cannot guarantee native error <.001. This
is not a full18 moving-record obstruction. The original finite scan,
unknown incidence/offset, and global competitor exclusion remain open.
The eventual global theorem must control the full-record residual over wrong
nonlinear regions; simply removing free coefficient fitting is insufficient
for that deferred guarantee.

**Earlier (09-22): finite algebraic shape theory; T5 STILL OPEN.**
[Last-prism inverse](paper/LAST_PRISM_IMPLICIT_INVERSE_2026_09_22.md): the full
response satisfies a bidegree-(4,4) polynomial in squared tilt and position,
with 25 coefficients. In the stated regular regime any 33 distinct squared
tilts determine it exactly, and a simple cubic root, one monotone scalar
inverse and explicit formulas recover the physical parameters. Normal
incidence and zero offset are included (the zero-offset branch uses a
smaller curve and two quadratic readouts). [Chain ordering](paper/CHAIN_POSITION_ALGEBRAIC_2026_09_22.md)
uses the extra algebraic degree contributed by flat downstream plates;
193 frozen values identify the last rotor and 81 order the remaining pair.
These are frozen-state data, not the native 200 moving-rotor samples.

**Do not oversell the linear readout.** Exact symbolic controls and rational
rank certificates pass, but a separate 80/110-digit conditioning control
finds a smallest nonzero singular value about 7.6e-27 even with 66 signed
tilt values and scaled Chebyshev columns. Free-coefficient fitting has a
measured linearized noise gain about 8.6e25. The five-parameter physical
Jacobian is vastly better conditioned; these diagnostics are neither
guarantees nor a physical impossibility result. The next useful step must
retain the physical coefficient constraints or target only physical
invariants. Full free-coefficient fitting is not a usable noisy solver.

[Targeted readout](paper/TARGETED_FINITE_SCAN_READOUT_2026_09_22.md) gives
explicit joint deterministic error bounds, a local full-prior analytic
domain, and exact common-rotor continuation through safe flattening paths.
It also states why these do not yet infer the frozen slices from 200
isolated samples. Global rotor confinement and a useful physical norming
bound remain open. The older claims of certified ten-second rotor
confinement, an intrinsic function-space floor, and equality between the
measured projection coefficients and Taylor coefficients were corrected.

**Scheduling instruction:** prioritize constructive recovery as above and
address global uniqueness afterward. T10 (historical archive closure and
final manuscript claims) remains deferred until T1-T9 are complete.
The standing stops on optical campaigns remain.
The user now requires one named existing target per sprint, with fixed
acceptance criteria and an explicit closed/still-open verdict. Supporting
lemmas do not replace a target or justify crossing it off.

**Latest (09-21): constructive inversion and the inert-prism theorem; T5
still open.** Theory note sections 2.6 and 2.7. The ideal record is
inverted by layer stripping: rotor by the rank-extending largest-line rule
(the one risley_lattice/lattice.py::fundamentals already uses; Lemma D
proves it), then the last prism reads the complete ray entering it at
every upstream state (the only unstable step), then prisms 2 and 1 are
explicit (F = 2 c_1^2 T/(2 c_2 - c_1^2 T), A = c_1/F, exact to 1e-40).
Theorem 3: with one inert prism (zero wedge anywhere, or stationary first
prism) the record determines everything except the speed and phase of a
zero wedge and, for an inert first prism, the three-dimensional fiber
through the ray entering prism 2; the index of a later flat plate is
determined, weakly. Competitors with a stationary tilted prism 2 or 3 or
with a rational relation are not excluded. Rational-relation strata are
locally regular at the tested system. Stored speeds are rational; the
stability theorem covers them.

**Latest (09-20): Theorem 1 tightened, Theorem 2 proved in qualitative
form; T5 still open.** In
[IDENTIFIABILITY_THEORY_2026_09_18.md](paper/IDENTIFIABILITY_THEORY_2026_09_18.md):
Lemma D (each generator strictly dominates every lattice line involving its
rotor, from tilt monotonicity by one integration by parts, explicit margin
0.18 |A_i| (g_x + g_y)) removes the visibility condition, so the
exceptional set of the ideal theorem is only a zero wedge or a rational
relation; the generator rule is: largest line, then largest line outside
the module generated so far (old rule false on cases 12, 16, 18, 27; new
rule correct on all 28). The assignment of rotors to chain positions, left
implicit before, is proved from the number of plate terms in each
single-rotor response. Theorem 2: for a fixed non-exceptional system and a
long enough record every compatible system of a compact class is confined,
Lipschitz in the allowance; compactness proof, constants not explicit, so
the T5 criterion at 200 samples is not met. Exact two-axis certificate
(experiments/lemma_c_two_axis.py): the last prism's ten truncated
coefficients determine its seven parameters inside the prior on two exactly
rational instances, no floating point. Correction: the second one-axis
solution of the 09-18 example has index 1.825, outside the prior. The
decomposition F = E C + R is a local tool (rho = 0.896 at the prior
corner); explicit constants at K = 200 have no known route other than
exhaustive computation, which the user has rejected. Later the same day:
Assumption T is now Theorem T, exact for all three prisms (position
increasing in tilt i if and only if l_i cos phi_i > p_i tan b_i cos c_i cos
theta_i, l_i an explicit effective remaining distance; verified to 2e-7;
experiments/tilt_monotone_exact.py), with a regime corollary and margins of
at least 48 on the battery. Section 6 of the theory note is the final
statement of what is proved, what is certified per instance, and why
explicit 200-sample constants are not reachable without exhaustive
computation.
Continuous formulation added (section 3.0a): from the continuous trace on
any time window, uniqueness (Theorem 1', by analyticity in time) and
stability on a compact class (Theorem 2') hold with no long-record clause
and no sampling-rate condition; those clauses are artifacts of sampling.

**Latest (09-18, later): the unified theory is now STATED as a theorem with
its proof structure, and two of its three Risley-specific lemmas are
proved.** Read
[IDENTIFIABILITY_THEORY_2026_09_18.md](paper/IDENTIFIABILITY_THEORY_2026_09_18.md).
Layer 1 (ideal record): outside an explicit exceptional set (zero wedge,
rational relation among signed speeds, visibility failure) the continuous
infinite record determines all eighteen parameters: Bohr uniqueness,
Lemma A (polarization: a lattice line with coefficient sum k has x-to-y lag
k pi/2, so generators are the positively handed first-order lines and all
even-sum lines are exactly linear; verified to the last digit), Lemma B
(positive first harmonic of every visible prism), Lemma C (shape from the
flat-configuration Taylor coefficients; normal incidence closed forms
verified to 1e-9, dilation ambiguity broken by a strictly decreasing
r(n); general incidence is a finite elimination not yet done). Layer 2
(finite record with allowance) is a stability theorem whose constants
(analyticity strip from the TIR margin, discrepancy of the sampled orbit)
are the remaining mathematics; it predicts exactly the observed failure
classes. Later the same day: Lemma C is proved at general incidence by analytic
continuation (three singularity radii of each prism's single-tilt response
give its amplitude, index and the beam angle in closed form, verified
exactly), so Layer 1 is complete modulo Assumption T. Layer 2 measured:
the rotor step has small explicit constants at 10 s; the function-space
route for the shape floors at about 1e-2 of the scale for any record
length (700 lattice lines above 1e-7, sup bound sums them), so the
finite-record shape statement is exactly the injectivity modulus of the
twelve-parameter sampled family, with the decomposition F = E o C + R and
the explicit global inversion of the low-order coefficient map C as the
open algebra. Evening: Assumption T became a per-instance certificate
(experiments/certify_tilt_monotone.py; 28/28 admissible truths, the two
TIR-touching truths refused), and the last prism's five-coefficient map is
proved injective on given data by exact elimination
(experiments/lemma_c_elimination.py; degree-284 resultant, Sturm count,
one solution, 48 s). Remaining for a per-instance global modulus of the
finite-record shape map: the remainder bound along the weak direction and
the twelve-parameter coefficient map; symbolic all-data proofs are out of
reach. The user has rejected search techniques as the route.

**Latest T5 attack (09-18): shape-free ordering test for the rotor
coordinates; blind families identified; target still open.** Read
[T5_ROTOR_ORDERING_VERDICT_2026_09_18.md](paper/T5_ROTOR_ORDERING_VERDICT_2026_09_18.md).
The workpiece position is increasing in every tilt sine (313,000 admissible
prior points including adversarial corners; proof open). Hence, with wedge
signs fixed, any pair of samples whose three tilt differences have certain
sign orders the data; the tilt-difference signs are exact functions of the
speed and phase box (product form), so the test is shape-free. It is sound
on all 30 battery truths and excludes 63% of random speed boxes at 0.25 Hz
width and 99.5% at 0.02 Hz with the shape at full prior. Its blind set is
exact: rotors in sync with mixed wedge signs carry zero constraints (the
flat-plate compensation family), so the rotor-only search does not
terminate on cases 0 and 1; on the all-positive-wedge cases 5 and 7 it
does not terminate either, because its certain constraints at
intermediate widths are strongly correlated and carry few bits. A sampling proxy shows magnitudes exclude
those families once rotor boxes reach 0.25 Hz / 9 deg; the rigorous
enclosure needed is specified (exact affine endpoints, corners in each own
index, beam angle and own index subdivided with mean-value slack from the
new exact-range Jacobian risley_lattice/chainjac.py) but not built. Gain
survey: the tilt gain is monotone in d_W, own index and gap only. No
archive input, replay or recovery campaign ran.

**Latest T5 attack: executed global search with exact monotone ranges; the
obstruction is measured, target still open.** Read
[T5_MONOTONE_SEARCH_VERDICT_2026_09_17.md](paper/T5_MONOTONE_SEARCH_VERDICT_2026_09_17.md).
A proved monotonicity lemma for the per-axis chain gives exact direction and
guard ranges over any parameter box from two corner evaluations, without
interval dependency loss or phase-wrap refusals (risley_lattice/monotone.py,
36,000 containment tests, zero violations). A branch-and-bound on the full
prior with per-sample exact ranges plus an affine LP was executed on a
synthetic battery scan: every configuration, including oracles with speeds
and phases given, excludes exactly half its boxes and never reaches a
survivor. The measured cluster radius of any per-sample test is 16 to over 32
box widths along phases and glass indices, scale invariant. The certified
Jacobian (the only joint primitive) refuses every box with free speeds down
to 1e-2 of the prior and, with speeds fixed, every box wider than 0.0625;
where it works its exclusion radius is 1 to 16 box widths. For box methods
the obstruction is the nonlinearity scale against fifteen coupled
coordinates, not the algebraic degree; the missing piece is a joint
multi-sample test informative at one quarter to one half of the prior with
wrapped phases. No archive input, replay or recovery campaign ran.

**Latest T5 computational attack: exact pruning checkers and smaller optical
graph; target still open.** Read
[T5_PRUNING_VERDICT_2026_09_17.md](paper/T5_PRUNING_VERDICT_2026_09_17.md).
Moving Farkas identities certify regional row deletion; the exact synthetic
record compresses 200 scalar bands to four. A positive-source normal-cone
reduction gives two-dimensional geometry certificates rejecting entire support
families. Both checkers now compose in an executed exact synthetic replay.
A unit-ray optical graph confines rotor dependence to alignment equations,
reduces maximal selected auxiliaries from 162 to 148, and preserves original
zero-wedge cases through generic auxiliary perturbations and exact limits.
Uniform prior margins remove inactive physical guard rows from enumeration;
only second/third exit floors remain potentially active. Separating the three
speed variables gives time-index exponent three in the candidate bound.
At K199,m400 the new exhaustive upper counts are about1.60e235 and8.04e237,
still impractical and not runtimes or lower bounds. The five final reports
pass1050 exact controls; a compression API domain-validation finding and its
original evidence are preserved. No optical inverse/evaluation, observation
or archive input occurred. No major target closed; T10 remains deferred.

**Preceding full-power T5 attack: finite witness construction proved; target still
open.** Read [T5_FULL_POWER_VERDICT_2026_09_17.md](paper/T5_FULL_POWER_VERDICT_2026_09_17.md).
Independent infinitesimal levels, finite saturated critical systems, exact
univariate representations and ordered limits now replace the assumed global
feasibility oracle in theory. At most eighteen active scalar optical blocks
enter each kernel; compatible-center accuracy adds an affine slack and keeps
fourteen nonlinear coordinates. Strict equality and nonattained endpoints have
explicit rules. The construction is not implemented for optical observations,
and even its sharper affine multihomogeneous candidate bounds are astronomical.
No major target closed. A new exact-grid quadrature/reflection family at speeds
(1,1,-3) has twenty rotor states but only ten scalar slots for eleven varying
parameters, with native worst-case floor .01 and uniformly strict optical roots.
The full physical prior domain is also proved contractible; compatible fibers
need not be. No optical evaluation, inverse decision or archive input occurred.

**Preceding T5-only sprint: completion test not passed.** Read
[T5_SPRINT_VERDICT_2026_09_17.md](paper/T5_SPRINT_VERDICT_2026_09_17.md).
A complete affine chart cover represents all 36 attained native extrema on
the compact moderate-observation class, including rank loss. Conditional rotor
reconstruction and buffered witness exchange are exact, but their global
nonlinear subqueries and strict boundary decisions remain unsolved. A finite-
horizon route audit rules out an invalid stopping inference. No major target
was crossed off. No optical evaluation, observation input or archive work ran.

**Latest regular-region mathematical attack:** read
[REGULAR_ATTACK_SYNTHESIS_2026_09_17.md](paper/REGULAR_ATTACK_SYNTHESIS_2026_09_17.md).
The last optical slope has a unique guarded quartic inverse for fixed upstream
arguments in the bounded-output band. Five-row cofactor inequalities exactly
eliminate the four affine unknowns while preserving their full priors. Neither
result yet confines all fourteen nonlinear coordinates. A separate signed
beam/glass/rotor inequality excludes an explicit ordinary regular region when
its retained observation violates the derived bound. A new exact cubic
glass/wedge compensation proves an all-time finite-noise accuracy obstruction
with distinct speeds and wedges bounded away from zero. Compactness also does
not make an arbitrary counterexample-guided loop terminate at strict equality;
the existing exact decidability theorem is not a new result of this pass.
Useful complete global computation remains open. No optical evaluations,
archive reads, recoveries or original-observation classifications occurred.

**Preceding continued mathematical attack:** read
[ATTACK_CONTINUATION_2026_09_17.md](paper/ATTACK_CONTINUATION_2026_09_17.md).
Observed coordinates satisfying |y|+eta<=50 exclude both critical exit layers
across the full18 prior at those sampled rays. Applying the condition everywhere
restores linear interval contraction and removes sampled critical-closure
ghosts on that observation class. The resulting conservative computation is
still impractical. Finite-lag inequalities also exclude whole speed/wedge
regions cheaply. A separate exact-grid theorem proves a 0.01 noiseless native
error floor at some nonconstant scans with every speed and wedge nonzero.
Literal stored timestamps have a separately bounded finite-noise transfer;
floating optical output discrepancies remain uncertified. T5-T7 remain open.
No optical campaign, archive classification or recovery count changed.

**Preceding executed mathematical attack:** read
[ATTACK_SYNTHESIS_2026_09_17.md](paper/ATTACK_SYNTHESIS_2026_09_17.md).
Physical subdivision now has an explicit anchor-free hidden-state contraction
theorem with a sharp fourth-root worst-case rate, and a sound full18 regional
residual bound. Its conservative complexity guarantee fails the practical gate.
An exact stationary-prefix/source ambiguity has three nonzero wedges and a
nonconstant scan; a separate function-field lemma retains the last optical
branch in the position observation. None supplies practical all-case recovery
or a new original-observation classification. The stopped optical campaigns
remain stopped. Read the synthesis for the scope and remaining obligation.

**Preceding outside attack:** read
[OUTSIDE_SYNTHESIS_2026_09_17.md](paper/OUTSIDE_SYNTHESIS_2026_09_17.md).
It supersedes the proposed global-bound direction below with a more specific
candidate: regular reverse angular optics, forward position propagation,
exact planar affine fibers, and global coordinate projections. It proves
new structural bounds and an explicit weak-phase noise obstruction, but no
practical all-case recovery or new archived observation classification.
The earlier exact compiler/backend evidence remains intact.

## 1. The user's actual objective

Recover **all eighteen physical parameters** of the Risley prism system, or
give a rigorous mathematical resolution of the archived failures. The original
numerical target is native-coordinate maximum error **strictly below 0.001**.
The user wants a broadly applicable mathematical method that handles collisions,
weak wedges and degeneracies. They reject solving a smaller calibrated problem,
endless guess-and-check repairs, and relaxing the threshold to claim success.

All eighteen coordinates remain unknown:

`[N1,N2,N3, ax1,ax2,ax3, ay1,ay2,ay3, ng1,ng2,ng3, d_W,gap, bm_ax,bm_ay,bm_px,bm_py]`.

Analytically eliminating a coordinate is acceptable only if its full compatible
range is preserved and it is reconstructed and certified as an output. No
coordinate may be supplied from simulation truth or silently calibrated.

### Collaboration failure to avoid repeating

The user is frustrated because the previous agent repeatedly delivered
representations, infrastructure, reports and plans without recovery or a new
population-wide failure bound. The agent explicitly acknowledged that the last
implementation pass recovered **zero additional cases** and established
**zero new population-wide failure bounds**. Do not describe those engineering
changes as having solved the research objective.

The baseline handoff followed **"write a handoff"**. The subsequent user asks
for a powerful outside perspective and a theory/algorithm handling the entire
problem. Three independent mathematical subagents, primary-source research,
exact algebra controls and read-only evidence audits were used. No external
collaborator was contacted, and no stopped optical campaign was restarted.

## 2. Standing execution restrictions

- The population replay is stopped. Do not restart it.
- The user also rejected further individual-case hunting and small targeted
  recovery cohorts. Historical experiment allowances are superseded.
- The latest work scope has been symbolic mathematics, exact algebra controls,
  synthetic algebraic solver controls, and read-only audits of saved evidence.
  No optical forward evaluations or optical inverse decisions were run in the
  latest pass. Do not infer new authorization for those experiments from this
  handoff or from historical proposed experiment lists.
- Use deterministic bounds for guarantees. Statistical/Fisher estimates are
  benchmarks, not certificates. A timeout, failed certificate or optimizer
  failure is never a proof of physical impossibility.
- Preserve failed experiments, frozen reports, hash bindings and historical
  source versions. Do not overwrite evidence to make a result look successful.
- Read [AGENTS.md](AGENTS.md). It contains important ordering, model, precision
  and Lean-build rules. Its historical solver descriptions are less current
  than the diary and evidence reports.

## 3. What is actually established about the archive

The retained historical campaign contains 669,574 cases: 651,172 recorded
numerical recoveries and **18,402 failures**. Those are historical outcomes,
not a success rate for the present implementation.

Among the 18,402 failures:

- **2,643** have proved physical-branch violations under the audited model.
- **15,759** have certified locally injective boxes around their known
  synthetic truths, including exact speed collisions. Two required weighted
  norms. This proves local identifiability, not blind recovery or global
  uniqueness. The truth-centered boxes cannot be used as hidden initialization.
- The latest saved-recovery recount is **504**, leaving **15,255** admissible
  cases without a saved recovery. This includes adaptive work and partial
  checkpoints; it is not the performance of one frozen algorithm. Older
  summaries showing 391 or 500 describe earlier scopes. Do not add overlapping
  cohorts or treat the stopped queue as completed evidence.
- At deterministic allowance `eta=1e-5` and native tolerance `.001`, saved
  pairs establish **6,701** actual-observation accuracy obstructions. These
  finite-noise bounds do **not** explain the original noiseless failures.

Sources: [REWRITE_PLAN.md](paper/REWRITE_PLAN.md),
[FAILURE_THEORY.md](paper/FAILURE_THEORY.md),
[CLOSURE_WORK.md](experiments/CLOSURE_WORK.md), and the diary.
The closure document retains older cohort counts intentionally; use the
rewrite plan and diary for the later aggregate recount.

### Data and model contract

The original cloud observation arrays and exact executable revision were
unavailable for the latest four historical recoveries; their observations
were regenerated with the current model. The archive audit records additional
provenance discrepancies. A reconstructed ideal record is not silently the
original archived observation.

The algebraic construction uses the strict reduced per-axis mathematical
model and exact uniform `dt=1/20`. Stored floating-point timestamps, floating
forward outputs, clipping and model discrepancy need an explicit deterministic
comparison contract. Exact rationalization of rounded observations at zero
allowance can make the mathematical equations inconsistent. Conversely, adding
noise allowance changes the inverse question; it cannot retroactively explain
a noiseless optimizer failure.

## 4. What the latest implementation pass produced

Read [EXACT_DECISION_ENGINE.md](paper/EXACT_DECISION_ENGINE.md) for the complete
input/output contract and limitations.

### Sparse exact optical equations

[exact_smt_graph.py](risley_lattice/exact_smt_graph.py) exports polynomial real
constraints preserving positive-root branches, intersection guards and the
exact original prior. It retains all18 physical unknowns and adds **30 trace
auxiliaries per sample**. For symbolic index `k=1`, the graph has 55 real
declarations, 457 definitions, 116 assertions and 30,303 bytes.

After definitions are inlined, maximum optical assertion degree is 14; prior
constants raise it to 20. This is an auxiliary-variable representation, **not
an eighteen-variable degree-14 inverse**. Higher sample indices raise rotor
degree. Only a symbolic one-sample artifact was built and checked; one sample
is not claimed to identify eighteen parameters.

### Exact native-coordinate accuracy

[exact_native_accuracy.py](risley_lattice/exact_native_accuracy.py) encodes all
eighteen error comparisons. Transformed angular coordinates use the tangent
difference identity. A rational-multiple-of-pi tolerance defines its exact
tangent by `Imag((1+i*T)^q)=0` and a rigorous rational isolating interval.
Repeated squaring keeps the equation circuit compact. It does not make its
algebraic degree or decision cost small. Isolation has a 256-term budget and
can fail for extreme inputs.

Caller-supplied rational `1/1000` and the exact binary64 value of `.001` are
different thresholds. The exported example uses rational `1e-6` merely as an
encoding control; the archive acceptance criterion was not changed.

### Exact-real backend and full18 query construction

[exact_real_backend.py](risley_lattice/exact_real_backend.py) uses isolated
Z3 5.1.0.0 workers for polynomial real arithmetic. It provides exact
rational/algebraic model values and bounded execution. Resource failures are
UNKNOWN. SAT/UNSAT are **trusted Z3 results**, not independently replayable
proof certificates. A proof-enabled probe did not produce a usable proof.

[exact_recovery_queries.py](risley_lattice/exact_recovery_queries.py) builds:

1. Existence of a physically compatible system.
2. Existence of a compatible estimate such that every compatible physical
   system is within the specified errors in all18 native coordinates.

The competing physical coordinates and all competing trace auxiliaries are
universally quantified. Dependency-ordered definitions are scoped inside the
quantifier. The positive compatible-estimate conjunct fixes valid shared
algebraic constants and prevents vacuous guarantees. Feasibility must be SAT
before accuracy UNSAT can be interpreted as an obstruction.

That obstruction is specifically **no accurate compatible estimate**; it is
not automatically impossibility for arbitrary output vectors. Exact output
extraction is implemented, but has not been exercised on an optical instance.
Construction exceptions and UNKNOWN remain unresolved computations.

**No full optical `solver.check()` was executed.** The saved feasibility and
accuracy queries only passed syntax, type and scope checks. They contain no
observation values. The accuracy query has one universal quantifier over 48
variables for its single symbolic sample. No universal observation-space
partition, universal selector, practical optical runtime or new archive
resolution has been established.

### An actual, modest algebraic reduction

[exact_threshold_axis.py](risley_lattice/exact_threshold_axis.py) and
[STRUCTURED_THRESHOLD_ELIMINATION.md](paper/STRUCTURED_THRESHOLD_ELIMINATION.md)
eliminate the third prism's exit square root from the threshold expression.
The physical requirement that its radicand is strictly positive remains.
Zero and opposite-sign cases are preserved by an exact sign rule.

The resulting axis template uses eight constructed roots rather than nine,
9,861 polynomial atoms, and degree upper bound 16,896 rather than 17,408.
Arithmetic nodes and compressed size increase somewhat. This is about a 3%
improvement in that axis degree bound, not a new bound for the full18 inverse
or demonstrated acceleration of recovery. The new template has not been
composed into a new full18 bound artifact.

### Evidence locations

- [decision_2026_09_16](experiments/results/decision_2026_09_16): threshold
  construction/audits, native-accuracy controls, symbolic queries, independent
  source review and final provenance check.
- [exact_smt_2026_09_16](experiments/results/exact_smt_2026_09_16): sparse graph
  export and dependency/scope controls.
- [exact_backend_2026_09_16](experiments/results/exact_backend_2026_09_16):
  dependency origin/hash and synthetic backend controls.

Validation completed: six threshold identities and 75 scalar sign controls;
small independent Sturm checks; initial twelve synthetic backend decisions and
nine input rejections; a final-source resource-handling regression check; eight
tangent-to-backend decisions (four SAT, four UNSAT). The original backend
control report predates a resource-error classification fix; its old hash is
intentional and is supplemented by the final-source report.

The final read-only check verified forty source/input hash bindings, six output
hashes, eighteen Python syntax parses and fifty-eight document links. These are
evidence of implementation checks, not optical recovery tests.

## 5. The substantive research gap and proposed priority

**Global confinement is missing.** Local injectivity says that nearby systems
can be distinguished. It does not locate the correct neighborhood from a scan
or exclude distant compatible alternatives. Exact decidability supplies a
logical route but no usable computational bound for this problem.

The recommendation discussed with the user was to prioritize a model-specific
global exclusion/confinement bound, derived from the optical structure and
informed by saved numerical failures. This is a proposed research target,
not an accomplished theorem or a proven practical solution.

Existing tools to build on, without rebuilding them:

- [GLOBAL_INVERSE_DESIGN.md](paper/GLOBAL_INVERSE_DESIGN.md): exact affine
  dependence on four still-unknown coordinates, joint compatible polytopes,
  verified dual exclusion and residual lower bounds.
- [ALGEBRAIC_PROGRESS.md](paper/ALGEBRAIC_PROGRESS.md): those bounds have
  excluded limited regions in earlier controls; they have not covered the
  whole prior or demonstrated efficient global localization. Some wider
  regions produced interval-domain refusals, not exclusions.
- [UNIVERSAL_FULL18_THEORY.md](paper/UNIVERSAL_FULL18_THEORY.md): complete
  logical formulation, exact representation, degeneracy and complexity scope.
- [FAILURE_THEORY.md](paper/FAILURE_THEORY.md): weak-wedge/glass compensation,
  collision mechanisms and saved finite-noise evidence. The usable uniform
  mixed-derivative bound for the weak-wedge family remains open.

A meaningful next result must supply explicit, checkable hypotheses and a
bound strong enough to exclude a substantial set of alternative explanations
or confine them tightly enough for all18 accuracy. Connect that bound to the
existing local certificates. Retain all physical orders, zero wedges, speed
collisions and strict branch conditions. If alternatives genuinely prevent
accuracy, prove the obstruction under the same observation/error contract.

Do not count any of the following as closing this gap: another declaration
that quantifier elimination terminates; parsing a larger symbolic query;
certifying another known-truth neighborhood; silently assuming a nonzero
spectral gap; or treating failure to find an alternative as uniqueness.
Practical performance eventually needs an optical evaluation, but no such
experiment is authorized or queued by this handoff.

## 6. Fundamental scope limits

Universal unique recovery on the entire original prior is mathematically
impossible: exactly zero wedges can make speeds/phases invisible, and exact
flat-plate compensation gives other ambiguities. A complete method must
represent those alternatives. These examples do **not** classify the 15,759
locally injective archived systems, which need their own global argument.

There is also no established uniform inexpensive runtime, useful universal
inverse-stability constant, or causal mathematical explanation of every
historical numerical failure. Keep existence, uniqueness, conditioning,
algorithmic convergence and model validity as distinct claims.

## 7. New structural results from the outside attack

- Reverse refraction uses `d=sqrt(n^2-h^2)` with `d^2>=69/100` over the original
  index prior. The critical exit root becomes a strict branch guard. The note
  gives a global secant bound, bounded Hessian, exact convex hull of the single
  root graph, and an explicit bidirectional quadratic prism block. The synthesis
  proves its uniform root-hull gap is at most
  `(42761/40960)*(width_n^2+width_h^2)`. This is not the convex hull of the full
  optical model or a full18 inverse error bound.
- Positive source-position gain permits exact elimination of both beam offsets
  into a polygon in `(d_W,gap)`, retaining both source reconstruction intervals.
  All18 unknowns remain. The nonlinear global-confinement problem remains.
- An explicit strict-prior near-TIR family disproves one uniform compensated
  gain/glass mixed-derivative bound over the full prior. Arbitrarily small
  nonzero wedges also exhibit the divergence by continuity. Positive-margin
  compact-family bounds remain valid.
- Exact all18 accuracy can be decided from coordinate extrema and their
  attainment flags followed by one compatible-center feasibility problem.
  Computing those extrema is still a global task. Open-boundary cases prevent
  a blanket finite-termination claim for a counterexample-only loop.
- A quantitative weak-phase theorem covers an infinite small-angle family at
  eta=1e-5 for ideal mathematical observations. The narrow 2-degree/5-degree
  version covers zero saved failure specifications after the additional
  0.008-degree weak-wedge condition. Do not claim an archive increase from it.
- The subsequent nonuniform envelope theorem is implemented as exact rational
  scalar inequalities in `experiments/outside_phase_envelopes.py`. Frozen before
  a full read-only applicability sweep, it proves all-time physical margins for
  11,605 of the 15,759 admissible saved specifications and phase-noise lower
  bounds for 444 ideal traces at eta=1e-5 (443 overlap earlier saved noise flags).
  The remaining 4,154 envelopes are inconclusive. No scan was regenerated or
  fitted. No new discrepancy bound to actual floating observations was proved;
  actual-observation classification and noiseless recovery counts remain
  unchanged. Read `OUTSIDE_PHASE_ENVELOPES_2026_09_17.md` and the sweep's saved
  report through the synthesis before using these counts.

Detailed proofs, scope and primary sources are in the synthesis and its linked
notes. Reproducible exact controls and read-only audit reports are under
`experiments/results/outside_2026_09_17/`. Earlier control failures and their
source versions are retained. This is mathematical progress, not a demonstrated
new optical solver.

## 8. Workspace and execution notes

- Workspace: `C:/Users/josep/Dropbox/Babcanec Works/Mathematics/Wedge`.
- Shell: PowerShell. The working tree contains substantial existing research
  changes. Do not reset, clean, commit or push them merely to organize handoff.
- Bundled Python:
  `C:/Users/josep/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`.
- Exact backend dependency: `.codex-deps/z3_exact` (ignored); installation
  version, wheel origin and hash are recorded in the backend results directory.
- Symbolic controls load modules directly with `importlib.util` to avoid
  incidental package imports. No inverse computation runs on importing the
  query module. Calling `classify_exact_data` does execute decisions; do not
  mistake it for a parser-only check.
- Frozen evidence binds source hashes. Version corrections and update their
  evidence deliberately; never silently invalidate a saved report.
- Lean build directory: `C:/Users/josep/lean/risley`, outside Dropbox. Keep
  edited formal sources synchronized; do not build `.lake` inside this repo.
- The `memory/MEMORY.md` path mentioned by AGENTS.md was not present at the
  workspace-relative location checked. Do not invent its contents.
- No new research computation or pending approval is created by this handoff.

The immediate successor should read the diary and the global-bound evidence,
choose a concrete mathematical bottleneck, and work toward a substantive bound
or constructive recovery result. Keep user updates short and state exactly
which research claim changed.
