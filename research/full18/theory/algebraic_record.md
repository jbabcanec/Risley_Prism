# Exact finite-record algebraization of the native passive inverse

This note derives an exact polynomial lift of the **original ordered, independent-axis three-prism model**, with all eighteen native parameters unknown. It is not the different full-vector 3-D model. It uses one passive record with two screen coordinates at each time, without derivatives, extra screens, parked states, or calibrated hardware. No global identifiability or practical global solver is inferred merely from algebraization.

The main theorem below uses the exact clock `t_k=k/20`, `k=0,...,K-1`, with `K=200` for the original experiment. Section 9 states precisely why that clock is different from the stored binary64 `numpy.arange` clock and gives a theoretical exact extension to rational timestamps.

## 1. Model and full native domain

The native vector, in its original physical order, is

\[
 (N_1,N_2,N_3,\alpha_1,\alpha_2,\alpha_3,
 \phi_1,\phi_2,\phi_3,n_1,n_2,n_3,d,g,\beta_x,\beta_y,p_x,p_y).
\]

`alpha_i` is `ax_i`, the signed wedge angle; `phi_i` is `ay_i`, its initial rotor phase. Both are in degrees in the API. `N_i` is in cycles per second. The phase at time `t` is `2*pi*N_i*t + pi*phi_i/180`. The bounds are

\[
\begin{gathered}
-3.5\le N_i\le3.5,\quad -18\le\alpha_i,\phi_i\le18,\quad
1.3\le n_i\le1.8,\\
50\le d\le200,\quad2\le g\le15,\quad
-25\le\beta_x,\beta_y\le25,\quad -5\le p_x,p_y\le5.
\end{gathered}
\]

Source distance is 6 and nominal prism thickness is 3; the two intervening gaps share the unknown `g`. The third prism-to-screen distance is the unknown `d`. These are the original model constants and native bounds, also saved in `../cases.json`. Prisms are never sorted by speed or otherwise permuted.

The displayed decimal native limits are interpreted as exact decimal rationals in this theorem. If the contract instead uses the exact stored binary64 endpoints (notably for the glass indices), substitute their integer-ratio values in the inequalities. The construction and proofs are unchanged; those two endpoint contracts should not be conflated in an exact claim.

The mathematical forward map is the strict transmitted, forward branch: square roots take their positive values, total internal reflection and zero longitudinal direction are refused, and line-plane denominators must be nonzero. This agrees with the strict mathematical model documented in the repository's `risley_lattice/fmodel.py` and the independently evaluated implementation `../stable_model/forward.py`. It does not claim bit-for-bit agreement with the legacy `acos`/degree/tangent implementation, nor include rays returned only because a legacy implementation clipped an invalid radicand. The forward accuracy audit in `../stable_model/README.md` is a separate numerical result.

No positive lower margin away from critical transmission is added. Also, no additional positive-travel-length constraint is imposed: doing so would change the original model's stated guards.

## 2. Replace sampled trigonometry by a bounded rotor recurrence

Let

\[
\theta_i=\pi N_i/10,\qquad c_i=\cos\theta_i,\quad s_i=\sin\theta_i.
\]

The complete native speed range is encoded by

\[
c_i^2+s_i^2=1,\qquad c_i\ge c_*:=\cos(7\pi/20)>0. \tag{1}
\]

The arc is `[-63 degrees,63 degrees]`, shorter than a semicircle. Every solution of (1) reconstructs exactly one native speed,

\[
N_i={10\over\pi}\operatorname{atan2}(s_i,c_i)\in[-3.5,3.5]. \tag{2}
\]

This handles zero, equal, opposite, and endpoint speeds. A nonzero speed separation is not required. Sampling at 20 Hz could identify speeds modulo 20 Hz in an unrestricted one-rotor representation, but no such pair of speeds lies in this native arc. This statement is about the rotor representation; it is not a uniqueness theorem for the screen inverse.

To preserve signed wedges and phase fibers, retain separate variables

\[
A_i=\sin(\pi\alpha_i/180),\quad
C_i^\phi=\cos(\pi\phi_i/180),\quad S_i^\phi=\sin(\pi\phi_i/180)
\]

and impose

\[
-a_*\le A_i\le a_*,\qquad
(C_i^\phi)^2+(S_i^\phi)^2=1,\qquad C_i^\phi\ge h_* , \tag{3}
\]

where `a_*=sin(18 degrees)` and `h_*=cos(18 degrees)`. Introduce `F_ik,G_ik`, with

\[
F_{i0}=A_i C_i^\phi,\qquad G_{i0}=A_i S_i^\phi,\tag{4}
\]
\[
F_{i,k+1}=c_iF_{ik}-s_iG_{ik},\qquad
G_{i,k+1}=s_iF_{ik}+c_iG_{ik}. \tag{5}
\]

These equalities give exactly

\[
F_{ik}=\sin\alpha_i\cos(2\pi N_i k/20+\phi_i),\qquad
G_{ik}=\sin\alpha_i\sin(2\pi N_i k/20+\phi_i), \tag{6}
\]

where angles inside trigonometric expressions in this note are understood in radians after conversion. In particular, `F_ik^2+G_ik^2=A_i^2` follows from the equalities and need not be separately constrained.

This is precisely a rescaling of the requested tangent rotor coordinates `U=tan(alpha) cos(gamma)`, `W=tan(alpha) sin(gamma)`: divide by `sqrt(1+tan(alpha)^2)=1/cos(alpha)`, positive throughout the native wedge interval. `U,W` obey the same recurrence. The rescaling makes `F,G` the actual face-normal sine components used in the two independent scalar refraction calculations; it is not a small-angle approximation.

The inverse readout is unique for every retained phase variable:

\[
\alpha_i={180\over\pi}\arcsin A_i,\qquad
\phi_i={180\over\pi}\operatorname{atan2}(S_i^\phi,C_i^\phi). \tag{7}
\]

At `A_i=0`, (4)-(5) give `F_ik=G_ik=0` for every sample, while `c_i,s_i,C_i^phi,S_i^phi` remain free on their allowed arcs. Thus the complete native speed and phase continuum of a zero wedge remains present. This does not assert that its glass index becomes invisible: a flat plate can still change the ray's displacement. Nor does it impose the artificial gauge `phi_i=0` when the wedge vanishes.

For nonzero wedges, the familiar unrestricted replacement `(alpha,phi) -> (-alpha,phi+180 degrees)` is outside the native phase interval. Keeping the signed `A_i` and the phase arc therefore avoids an incorrect identification of native hardware.

An optional smaller existential representation keeps only `F_i0,G_i0`, with

\[
F_{i0}^2+G_{i0}^2\le a_*^2,\qquad
G_{i0}^2\le\tan^2(18^\circ)F_{i0}^2. \tag{8}
\]

When `F_i0!=0`, reconstruct `A_i=sign(F_i0)*sqrt(F_i0^2+G_i0^2)` and `phi_i=atan(G_i0/F_i0)` in its native arc. When `F_i0=0`, (8) forces `G_i0=0`; restoring **all** original solutions requires appending the whole native phase interval. We use (3)-(5), not this quotient, in the theorem and counts.

## 3. Collapse a flat-entry / tilted-exit prism without losing branches

Fix one prism, one transverse axis, and one sample. Let `X` be the incoming unit transverse air direction. Its longitudinal component is positive; the other transverse axis is handled independently, so no joint 3-D normalization is introduced. Set `f=F_ik` on x and `f=G_ik` on y. Introduce six variables `C,H,Q,R,a,v` and the equations

\[
\begin{aligned}
C^2+f^2&=1,& C&>0,\\
H^2+X^2&=n_i^2,& H&>0,\\
Q&=CX+fH,\\
R^2+Q^2&=1,& R&>0,\\
a&=CQ-fR,\\
v&=fQ+CR,& v&>0.
\end{aligned} \tag{9}
\]

These are six equations of degree at most two. Their interpretation, derived directly from the original interface equations, is:

* `C=sqrt(1-f^2)` is the positive face-normal longitudinal component; its plane slope is `f/C`.
* At the flat entrance, the glass ray is `(X/n_i,H/n_i)`, so its internal slope is `X/H`.
* At the tilted exit, `Q` is the signed Snell transverse quantity before the air radical, `R=sqrt(1-Q^2)`, and the outgoing unit ray is `(a,v)`.

In particular,

\[
a^2+v^2=(C^2+f^2)(Q^2+R^2)=1,\qquad Cv-fa=R.\tag{10}
\]

The formulas agree in sign with `../stable_model/forward.py`: its interface quantity is `cy=-(C*X+f*H)/n_i=-Q/n_i` immediately before the exit. Substituting the glass-to-air ratio `n_i` gives exactly `a=CQ-fR`, `v=fQ+CR`. This independently checks the collapsed map proposed during this investigation.

Introduce one more variable

\[
D=HC-Xf,\qquad D>0.\tag{11}
\]

Its positive sign follows from the full native bounds, rather than an extra parameter restriction:

\[
D\ge\sqrt{1.3^2-1}\cos18^\circ-\sin18^\circ>0.48.
\tag{12}
\]

For example, the rational comparisons `sqrt(.69)>.83`, `cos18>.951`, `sin18<.3091` imply the final bound `.951*.83-.3091=.48023`. They use `|X|<=1`, which holds inductively from (10). Also `D^2+Q^2=n_i^2`; this identity alone would not fix the sign of `D`, so (12) or the explicit guard remains necessary in an equivalence argument.

Let `P` be the ray coordinate at this prism's flat entry plane, and let `P_next` be the coordinate at the next nominal parallel plane: the following entry plane for prisms 1 and 2, and the screen for prism 3. Let `b=g,g,d`, respectively. The exact position transfer is

\[
vD P_{next}-R(HP+3X)-b\,aD=0.\tag{13}
\]

This equation has degree three. To derive it, intersect the internal ray with the tilted exit plane to obtain

\[
P_e={C(HP+3X)\over D}.
\]

Then propagate by `P_next=P_e+(b-(f/C)P_e)*a/v` and use `Cv-fa=R`. Thus

\[
P_{next}={HR\over vD}P+{3XR\over vD}+b{a\over v}.\tag{14}
\]

All denominators multiplied in (13) are strictly positive. The internal line-plane grazing denominator equals `D/(HC)>0`; the outgoing propagation to a flat plane has no additional grazing zero. The expression `1-(f/C)*(a/v)=R/(Cv)>0` used in the transfer is also positive. Consequently the collapsed transfer respects the original nonzero-grazing guards throughout this native domain.

## 4. Assemble a finite polynomial feasibility problem

Represent the source direction separately on each axis by `(x_0,z_0)` with

\[
x_0^2+z_0^2=1,\qquad z_0>0,\qquad
-b_*\le x_0\le b_*,\quad b_*:=\sin25^\circ.\tag{15}
\]

It reconstructs `beta=(180/pi)*atan2(x_0,z_0)` exactly in its native interval. The two axes have separate pairs; they are not the components of one jointly normalized three-dimensional vector. Introduce each first entry coordinate `E` by

\[
z_0E=z_0p+6x_0.\tag{16}
\]

For each sample and axis, apply (9), (11), and (13) in the original prism order. Use `X=x_0,P=E` at the first prism, then use the preceding outgoing `a` and `P_next` as the next prism's input. The second and third prisms therefore share hidden rays with their preceding stages, not independent adjustable ray directions. Impose the native intervals on `n_i,d,g,p_x,p_y`.

For the actual observation `Y_ak` and componentwise hard allowance `eta>=0`, impose

\[
Y_{ak}-\eta\le P_{3ak}\le Y_{ak}+\eta.\tag{17}
\]

The 400 scalar observations for `K=200` are all used. `eta=0` is allowed. Rational decimal measurements or exact binary64 measurements are rational coefficients. No statistical distribution is presumed by (17).

**Theorem (exact finite-record equivalence).** For exact timestamps `k/20`, there exists a full native eighteen-vector whose strict original forward record obeys (17) if and only if the polynomial equalities and inequalities (1), (3)-(5), (9), (11), (13), (15)-(17), and the stated scalar native bounds have a real solution. Native solutions are reconstructed by (2), (7), (15), and the retained scalar variables. Every native solution, including zero wedges, zero speeds, equal speeds, and native interval endpoints, is represented. This is a feasibility equivalence, not an injectivity or successful-recovery theorem.

**Forward proof.** Given native hardware on its strict physical branch, define its step and phase circles, signed wedge sine, and source unit pairs. The elementary addition formulas give (4)-(6). At each sample, the original two entry/exit interface equations reduce to (9), with their positive roots and positive outgoing longitudinal component. The native bounds give (12), and the line-plane calculation gives (13). The same source entry and screen points satisfy (16)-(17). Thus every native admissible record supplies a lifted solution.

**Reverse proof.** Given a lifted solution, the arc constraints give unique native speeds, phases, wedge angles and source angles by (2), (7), and (15). Induction on the sample index using (4)-(5) gives exactly their rotating faces on the prescribed clock. At the first prism, the source unit pair has the correct positive longitudinal branch. At any prism, `C>0` and `H>0` select the unique face cosine and flat-entry radical; `R>0` selects the unique transmitted exit radical. Equations (9)-(10) then give exactly the original outgoing air ray, with `v>0`. Induction on prism order carries this fact through all three prisms. The positive denominator conditions make (13) reversible to the original ray-plane intersection, so its positions cannot be extraneous roots introduced by denominator clearing. Equation (16) fixes the source entry and (17) supplies the declared hard-error fit. No step divides by wedge amplitude, rotor speed, a frequency difference, or a data matrix determinant. Separate phase variables remain valid when a wedge is zero. This proves the converse and covers the stated degeneracies.

## 5. Exact constants, degree, and conservative variable counts

All constants in this lift are real algebraic numbers. Useful exact descriptions are

\[
a_*={\sqrt5-1\over4},\qquad h_*={\sqrt{10+2\sqrt5}\over4},\qquad
\tan^2 18^\circ={5-2\sqrt5\over5}.
\]

`c_*=cos63 degrees` is the unique root of `T_10(c)=0` in `(45/100,46/100)`. `b_*=sin25 degrees=cos65 degrees` is the unique root of `T_18(b)=0` in `(42/100,43/100)`, where `T_j` is the Chebyshev polynomial. The isolating intervals choose the intended roots. An exact implementation can work over this fixed real algebraic coefficient field, or add root-defining variables and their isolating intervals. Counts below use the coefficient-field option.

| Group | Variables | Equalities |
|---|---:|---:|
| Three prism globals `(c,s,A,Cphi,Sphi,n)` | 18 | 6 circle equations |
| Two source unit pairs | 4 | 2 |
| Geometry `(d,g,p_x,p_y)` | 4 | 0 |
| First entry positions `(E_x,E_y)` | 2 | 2 |
| Three two-component rotor sequences | `6K` | `6K` including initialization |
| Local `(C,H,Q,R,a,v,D)` at each of `6K` prism/axis/sample blocks | `42K` | `42K` |
| Three propagated positions on both axes per sample | `6K` | `6K` |
| **Total** | **`54K+28`** | **`54K+10`** |

For 200 samples this is **10,828 variables and 10,810 equalities**, of total polynomial degree at most **three**. The count difference of eighteen is consistent with the retained hardware dimension; it is not a proof of independent equations or identifiability at singular points.

There are `32` scalar global inequalities under the stated convention (counting each endpoint bound separately), `30K` local positive-root/branch inequalities, and `4K` observation inequalities: **`34K+32=6,832`** at `K=200`. Some guards and many repeated quantities can be removed using derived identities; these are explicit conservative counts, not a claim of a minimal encoding. At zero wedges and other singular hardware, the solution set may contain continua. The lift retains them instead of replacing them by arbitrary representative hardware.

The raw map `N -> sin(pi*N/10)` is not itself claimed to have a polynomial graph in the original real variable `N`. The polynomial problem uses exact circle coordinates with an explicit one-to-one arc readout. The native angular readout can be transcendental even when a chosen lifted sample has algebraic coordinates.

## 6. Structural elimination of hidden rays and geometry

The lift is triangular before imposing observations. For fixed fourteen nonlinear hardware coordinates (three speeds, signed wedges, phases, indices, and two source angles), its rotor values and every strictly admissible local optical variable are uniquely determined. Hidden rays are auxiliary variables, not additional free measurements or unknown hardware. They cannot independently absorb observation errors.

One can eliminate rotor sequences explicitly:

\[
\begin{aligned}
F_{ik}&=T_k(c_i)F_{i0}-s_i U_{k-1}(c_i)G_{i0},\\
G_{ik}&=s_i U_{k-1}(c_i)F_{i0}+T_k(c_i)G_{i0},
\end{aligned}\tag{18}
\]

for `k>=1`, with the initialization used at `k=0`. Here `U_j` is the Chebyshev polynomial of the second kind. This exchanges `6K` sequence variables for polynomial degrees up to `K` when `F_i0,G_i0` are retained, or `K+1` after substituting the signed-amplitude/phase products. It does not remove the global nonlinear dependence.

Each prism introduces three positive radicals per axis/sample: `C,H,R`. Thus eliminating three prisms by repeated squaring can have extension degree up to `2^9=512` for one axis/sample over its rotor/source/index field. This upper bound is not a global degree bound for the 400-output simultaneous inverse, and squaring alone loses branch information. Keeping the low-degree positive-root lift is a more transparent exact specification.

Geometry can be eliminated more effectively. Define per prism

\[
L_i={H_iR_i\over v_iD_i}>0,\qquad
K_i={3X_iR_i\over v_iD_i},\qquad T_i={a_i\over v_i}.
\]

Equation (14), applied three times starting at `p+6*T_0`, gives at each axis/sample

\[
P_3=B+Jg+Td+\kappa p,\tag{19}
\]
\[
\begin{aligned}
\kappa&=L_3L_2L_1>0,\\
B&=6T_0\kappa+K_3+L_3K_2+L_3L_2K_1,\\
J&=L_3L_2T_1+L_3T_2,\qquad T=T_3.
\end{aligned}
\]

For fixed nonlinear hardware this leaves a bounded four-variable linear feasibility problem with `4K+8=808` inequalities, exactly retaining the unknown `(d,g,p_x,p_y)`. Because `kappa>0`, source positions can further be eliminated by interval intersection:

\[
{Y-\eta-B-Jg-Td\over\kappa}\le p\le
{Y+\eta-B-Jg-Td\over\kappa},\qquad -5\le p\le5.\tag{20}
\]

For each axis every lower endpoint must be at most every upper endpoint. This yields a two-dimensional halfplane problem in `(d,g)`. There are at most `2(K+1)^2+4` inequalities including tautologies and the four distance/gap bounds; enumerating all pairs is usually unnecessary. The companion [elimination.md](elimination.md) proves this equivalence, reconstructs all source positions, and provides a uniform nonlinear-box rejection inequality using rational dual witnesses. Its measured explicit polygon construction is slower than the four-variable LP, so dimension reduction is not automatically a runtime improvement.

None of this elimination assumes a nonzero determinant, a full-rank geometry matrix, distinct speeds, or nonzero wedges. Near a critical exit, `R_i` can approach zero and `kappa` can become small; no uniform positive lower bound for `kappa` is asserted over the full strict domain.

## 7. A complete feasibility algorithm, and why it is not a usable global solver yet

The theorem specifies a finite problem over exact algebraic coefficients. There is a constructive complete route in principle:

1. Build precisely the listed equations, weak inequalities and strict inequalities, preserving sample indices and physical prism order. Represent observations and error allowance by their declared exact numbers and the fixed constants by isolating polynomials/intervals.
2. Form a complete sign-invariant real-algebraic decomposition for these polynomials: project with coefficients, discriminants and resultants sufficient for sign invariance; isolate the resulting univariate real roots; then lift recursively through their sections (roots) and sectors (intervals between roots). Use a complete projection/lifting scheme with the requisite handling of nullification and lower-dimensional cells. A reduced projection rule without its hypotheses/fallback is not a completeness argument.
3. Evaluate the required sign/equality formula on every resulting cell's exact algebraic sample. Keep every satisfying cell, including positive-dimensional cells, rather than just isolated solutions. A sample from any satisfying cell reconstructs feasible hardware by the theorem. If every cell fails, the finite-record problem is infeasible. Native readout maps the retained cells to all hardware fibers; it need not be algebraic in native degree/Hz coordinates.

The classical projection/lifting mechanism is described in H. Hong's primary paper, [An improvement of the projection operator in cylindrical algebraic decomposition (ISSAC 1990)](https://dl.acm.org/doi/10.1145/96877.96943). The optical lift and its equivalence proof above are derived here; that reference does not provide a Risley recovery theorem or a practical implementation for this system.

The algorithm treats strict signs as strict signs. It does **not** replace `R>0` by a chosen `R>=epsilon`, discard zero wedges as nongeneric, or require an isolated solution. Those changes could silently remove native solutions. A cell decomposition can in principle handle both open feasible regions and exact equality/tangency cases.

This is a finite decision prescription, not code that has been implemented or timed on the 10,828-variable lift. General elimination may have prohibitive complexity; the number of variables, coupled sampling equations, high degrees after hidden-variable elimination, and lower-dimensional cases make direct decomposition an implausible near-term recovery method. No useful complexity bound for this particular inverse has been established here.

A more plausible implementation would subdivide fourteen-dimensional optical boxes, propagate the triangular positive-root equations by certified interval bounds, and apply the exact bounded geometry elimination/dual pruning from [elimination.md](elimination.md). Spectral and local-fit information may order boxes or produce witnesses but cannot justify deleting an unvisited region. Failed interval guards are inconclusive unless they prove the entire box is invalid. Such a search is not automatically a finite terminating decision algorithm: boundary tangencies, zero residual gaps, singular fibers, and the strict critical-transmission boundary require retained unresolved boxes or a complete algebraic fallback. See [global_initialization.md](global_initialization.md) for the complementary search contract.

Global recovery to a requested tolerance needs more than one compatible point: it must bound the diameter of **all** compatible native hardware, handle the declared gauges/degenerate continua explicitly, and account for measurement error. Algebraization supplies a correct finite feasibility specification on which such a proof could be based; it does not supply those diameter bounds.

## 8. What is proved, checked, and still missing

The new mathematical results in this note are the finite degree-three lift with a two-way native interpretation, the explicit preservation of all signed/zero rotor fibers, the native arc's unique speed readout, and the counted low-degree construction. The independently checked collapsed prism signs and exact transfer reproduce the original scalar-axis optics. The affine geometry structure agrees with the separately derived [elimination.md](elimination.md).

The companion `check_algebraic_record.py` was executed successfully. Its twelve exact checks cover polynomial identities for rotor norm, collapsed ray norm, prism rotation, ray-plane transfer, the three-prism affine expansion, algebraic root-isolating intervals, the stated counts, a rational native grazing lower bound, and the scalar clock discrepancy below. All passed; the exact output and script hash are saved in `algebraic_record_checks.json`. These are local algebra checks supporting the derivation, not a machine-checked proof of the entire equivalence theorem or a completed global inverse. The script imports no original project code, reads no observations or hidden hardware vectors, and runs no optical cases or fits.

An independent agent also checked the local two-way optical interpretation, native sign of `D`, zero-wedge fibers, and counts. It emphasized the same two qualifications retained here: stored timestamps are not the nominal exact clock, and the original strict algebraic-intersection model does not impose positive travel lengths. This independent review is supplementary scrutiny, not a formal proof certificate. The separate interval Jacobian rank witness in `rank_chart_certificate.json` uses the original binary64 timestamps; it is not automatically a rank witness for this note's exact `k/20` lift.

Still missing are a computationally effective exhaustive feasibility/diameter algorithm on the full native domain, a nondegenerate global uniqueness theorem for the original finite passive record, and complete treatment of compatible sets near optical boundaries in a practical solver. Zero-wedge speed/phase continua are unavoidable within the stated prior; no algorithm can uniquely recover those coordinates from this passive record alone. Existing successful numerical recoveries and compatible-pair noise lower bounds remain separate evidence and are not upgraded by this algebraization.

## 9. Clock and floating-point contract

The theorem's rotor recurrence is exact for rational times `k/20`. The repository constructs its usual times with floating `numpy.arange`; the exact real values of those binary64 numbers are not all `k/20`. In the existing 200-sample audit their maximum discrepancy was about `1.78e-15` seconds. That is small numerically but cannot be dropped from an exact equivalence claim. The strict stable forward audit and prior independent interval pair verifiers intentionally preserved those original binary64 timestamps. Their conclusions should not be described as instances of the simpler `k/20` theorem without a separate timing-error enclosure.

For a concrete exact arithmetic discrepancy, the stored `t_1=.05` is `3602879701896397/2^56`, while the stored `t_3=3*.05` is `10808639105689192/2^56`. Thus `t_3-3*t_1=2^-56`, not zero. The symbolic spot-check script verifies these fractions without evaluating any optical case.

There is a theoretical algebraization for any **declared rational timestamps**, including the exact binary64 values. Choose an integer `Q` so every time is `m_k/Q` with integer `m_k`. For `Q>=20`, use the base step

\[
c_i=\cos(2\pi N_i/Q),\quad s_i=\sin(2\pi N_i/Q),\quad
c_i^2+s_i^2=1,\quad c_i\ge\cos(7\pi/Q)>0.
\]

The native arc again selects a unique speed `N_i=Q*atan2(s_i,c_i)/(2*pi)`. Compute the needed integer rotor powers by exact complex multiplication (or repeated squaring) and rotate the same initial signed-amplitude vector. Everything after the rotor construction is unchanged. The arc constant remains algebraic because its angle is a rational multiple of pi, but it can have an enormous algebraic degree when `Q` is the common binary64 denominator. Repeated squaring reduces the number of power equations but does not make that coefficient field or the elimination manageable. The degree-three and variable counts in Section 5 are specifically for the sequential exact `k/20` lift, not this alternative construction.

For arbitrary real, incommensurate timestamps this particular finite rotor-power polynomialization has not been shown. Known clock uncertainty should instead be explicitly included in a rigorous forward-error or enlarged-observation analysis; silently snapping timestamps or treating numerical trigonometric constants as exact algebraic coefficients would change the problem.
