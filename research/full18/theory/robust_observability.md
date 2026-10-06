# A sharp, finite-record robustness theorem for the canonical full18 inverse

## Finding

The existing rank proof can be strengthened to an anisotropic, deterministic local observability theorem whose constants are constructed from the actual 200 timestamps, the five-invariant scalar inverse, the beam identity, and explicit optical remainder bounds. It also gives matching information-theoretic lower bounds. The central new result is a five-dimensional cubic noise barrier, not another assertion that a Jacobian is nonsingular.

This theorem is local on a certified parameter chart. It must not be presented as a complete global inversion of the native box. The original bounds permit zero wedges, equal and zero speeds, harmonic collisions, and near-critical optics. No uniform full18 recovery or uniform positive observability constant is possible on that box.

## 1. Coordinates that expose all eighteen uncertainty scales

Use radians internally and state all degree conversions when returning native bounds. Let h=(n1,n2,n3,g,d), e_i=sin(alpha_i), and

K_i(h)=(n_i-1)[d+(3-i)g+3 sum_{j>i}(1/n_j)].

Define a_i=K_i(h)e_i, B(h)=d+2g+6+3 sum_i(1/n_i), and b_h=p_h+B(h) beta_h. The local chart is z=(a_1:a_3,b_x,b_y,N_1:3,phi_1:3,beta_x,beta_y,h). It is an actual invertible coordinate change near the centered small-wedge family, not removal of an unknown. Inverting gives alpha_i=arcsin(a_i/K_i), p_h=b_h-B(h)beta_h.

At the centered reference beta=p=0, a_i=epsilon A_i with nonzero fixed A_i, the exact derivative has five strong directions, six order-epsilon phase/speed directions, two order-epsilon-squared compensated-beam directions, and five order-epsilon-cubed hardware directions. Use

S_epsilon=diag(1 repeated 5, epsilon repeated 6, epsilon^2 repeated 2, epsilon^3 repeated 5).

The coordinate change and its inverse have epsilon-uniformly bounded derivatives on a compact interior hardware box. Its possibly large dimensional constants must be retained. Thus the same singular-value exponents hold in native coordinates after fixed unit normalization. The first five exponents are zero, the next six are one, the next two are two, and the last five are three.

The existing proof scales nine columns by epsilon^-1 because it uses A=a/epsilon. That bookkeeping should not be confused with native wedge sensitivities.

## 2. An explicit finite-record demixer, including unknown frequencies

For a candidate frequency box centered at N0, let M={m in Z^3: |m|_1<=3}. At t_k=k/20 form all exp(2 pi i (m dot N0)t_k), with conjugate pairs represented by real sine/cosine columns. Add t_k exp(±2 pi i N0_i t_k) for each fundamental. There are 63 real harmonic columns and six real confluent columns. Centering and normalizing time is permitted if its conversion factors are retained.

The 200 by 69 real matrix H is completely determined by candidate frequencies and the actual clock. One need not estimate derivatives of noisy observations. Its Gram entries are finite geometric sums and their first two derivatives. For example,

sum_{k=0}^{199} exp(i k x) = exp(199 i x/2) sin(100 x)/sin(x/2),

with its continuous value at multiples of 2 pi. Consequently rank, inverse/left-inverse residual, and individual demixer row l1 norms can be certified by finite interval or exact-trigonometric arithmetic. A proposed left inverse D may be checked through ||I-DH||<1; its numerical nullspace is never assumed exact.

A practical conditioning condition is a certified lower bound on this explicit Gram matrix together with finite row l1 norms of the particular needed functionals. Merely checking distinct nodes is insufficient for a useful bound. Alias distances are modulo 20 Hz on this clock, and composite frequencies m dot N, not just three base frequencies, must be checked. Near a collision the constant is allowed to diverge.

Choose eighteen fixed linear functionals P of the two-axis record from these demixers: nine fundamental tangent channels, two baseline channels, two compensated-beam channels, and five selected cubic channels. Cubic channels correspond to 3N_i, 2N1+N2, and 2N2+N3. For a positive complex Fourier coefficient, a selected cubic monomial contributes c_m product(e_i^m_i)/8 times its known phase factor. Equivalently its positive real cosine amplitude has factor 1/4. All conversion factors must be kept.

This construction includes frequency derivative columns; it does not assume that frequencies or phases have already been measured exactly. Other cubic channels are nuisance columns annihilated by the relevant extraction rows. A single channel-selection prescription should be fixed before any noise validation.

## 3. Structural inverse and quantitative certificate

Let F3 be the full degree-three canonical optical expansion in (e,beta,p), retaining all source terms. At a reference zc define

T0 = P D_z F3(zc) S_epsilon^-1,       L = T0^-1 P.

T0 is not an arbitrary exact-optics Jacobian minor. Its leading block inverse is constructed from:

1. Three nonzero fundamental amplitudes and their phase/frequency tangent coefficients;
2. The exact source/beam identity B q_p - q_beta = -d(n3-1)(3n3+1)/2;
3. The five cubic hardware invariants and their audited monotone scalar reconstruction.

The beam factor is uniformly at least 36.75 in magnitude on d>=50 and n3>=1.3, before wedge-amplitude and Fourier factors. The K_i are positive on the hardware box. Hardware conditioning is quantified by the scalar closure slope and triangular reconstruction derivatives, not inferred from injectivity alone.

For a convex chart box Z on a strict physical branch, certify

q = sup_{z in Z} ||L D_z F(z) S_epsilon^-1 - I||_infinity < 1.       (C)

This is directly checkable without assuming a singular value of the unknown true Jacobian. Split it into variation of the explicitly differentiated cubic polynomial plus a validated differentiated remainder:

q <= sup_Z ||L[D F3(z)-D F3(zc)] S^-1|| + sup_Z ||L D(F-F3)(z) S^-1||.

Near the centered family the second term is O(epsilon^2). This order requires differentiated remainder bounds: a forward remainder alone does not imply it. The first term measures chart radius, amplitude uncertainty, candidate frequency/phase uncertainty and source displacement explicitly. It cannot be discarded merely because epsilon is small. If desired, use a scaled interval matrix and componentwise nonnegative majorant rather than a single infinity norm.

THEOREM. If (C) holds and z,z' in Z produce records compatible with y under componentwise noise eta, then

||S_epsilon(z-z')||_infinity <= 2 ||L||_(infinity<-infinity) eta/(1-q).

For heterogeneous allowances replace ||L|| eta by max_j sum_m |L_jm| eta_m. For one exact truth and one point fitted to y with residual eta, the same factor two applies. The constant concerns all 400 scalar measurements; independent verification of those measurements remains required.

PROOF. Integrate D F along the segment between z and z'. With u=S(z-z'), obtain L(F(z)-F(z'))=(I+Ebar)u, ||Ebar||<=q. Rearrangement yields the bound. This also proves injectivity in Z. It does not by itself prove a compatible point exists, that data lie in the selected-output image, or that no compatible point lies outside Z.

This certificate gives a usable local stopping test and safe cell-contraction rule for a complete covering method. A record-specific full18 result requires applying it to every surviving chart and retaining all unresolved/singular pieces.

## 4. Actual native error rates

Suppressing explicitly computable constants, in the local small-noise regime:

- hardware h: eta/epsilon^3;
- beta_x,beta_y: eta/epsilon^2;
- speeds and phases: eta/epsilon;
- leading amplitudes a and baseline combinations b: eta.

Native wedges are recovered by alpha_i=arcsin(a_i/K_i). Thus their errors contain both O(eta) and O(epsilon times eta/epsilon^3)=O(eta/epsilon^2). Native source offsets p=b-B beta likewise usually have O(eta/epsilon^2) uncertainty; at a nearby beta=O(epsilon) point their hardware-induced contribution has the same order. Reporting only eta for wedges or offsets would ignore nuisance compensation.

A useful local asymptotic regime is eta <= c epsilon^3, with c determined by the chart margins and contraction constants. This is the condition that the weakest estimates remain within the certified hardware chart and that beam errors remain O(epsilon). No chosen numerical eta can be silently substituted for the user's input precision.

## 5. Matching lower bounds: five genuinely weak directions

Fix frequencies/phases and source beta=p=0. Let h(s) vary inside a compact hardware neighborhood and set e_i(s)=a_i/K_i(h(s)), so all three leading amplitudes remain exactly fixed. Then

F(h(s)) = fundamental(a,N,phi) + epsilon^3 C(h(s);A,N,phi) + O(epsilon^5).

A uniform derivative bound along this family gives ||dF/ds||_infinity <= C_h epsilon^3 ||dh/ds||. C_h is calculated from the explicit cubic recurrence plus a derivative remainder enclosure. Therefore two hardware vectors distance delta apart can yield records distance at most C_h epsilon^3 delta apart. Their midpoint record is compatible with both under eta whenever C_h epsilon^3 delta<=2eta. No deterministic estimator can have worst-case error below delta/2 for this experiment.

Equivalently the exact output F(h0) has a compatible five-dimensional hardware ball of radius min(r,eta/(C_h epsilon^3)), with amplitude-compensated wedges, when the entire path stays physical and within native bounds. This lower bound applies even if an oracle supplies exact frequencies/phases and source coordinates; unknown nuisances cannot remove it.

Analogous compensated-beam and phase/speed paths give epsilon^-2 and epsilon^-1 barriers, in the local regime where the source perturbation remains O(epsilon). Together with Section 3 these yield matching powers, not necessarily identical constants. At the nondegenerate witness, boundedness of both the scaled Jacobian and its explicit left inverse proves the singular-value hierarchy stated in Section 1 by standard min-max inequalities.

## 6. Unknown frequencies and cubic contamination must be charged

For a first-order harmonic initializer, the ignored canonical cubic response is O(epsilon^3). Under a certified fundamental fit this creates frequency and phase bias O(epsilon^2), plus measurement contribution O(eta/epsilon). These statements are local, require retained labels, and do not establish global frequency recovery.

A cubic-extraction row that annihilates both a fundamental and its frequency derivative has fundamental leakage O(epsilon T^2 deltaN^2), rather than O(epsilon T deltaN). After division by epsilon^3 to obtain an invariant, this is O(T^2 deltaN^2/epsilon^2). At deltaN=O(epsilon^2) it is O(epsilon^2), matching the canonical quintic contamination. Without confluent annihilation, leakage is O(deltaN/epsilon^2), and an apparently accurate initializer can destroy hardware inference.

A schematic invariant error budget is

Delta I <= C_demix [eta/epsilon^3 + C5 epsilon^2 + C_f T^2 deltaN^2/epsilon^2] + C_phase deltaPhi + C_amp(relative amplitude errors),

with all source terms either modeled in F3 or charged separately. For a first-order amplitude estimate, cubic contamination creates O(epsilon^2) relative amplitude bias. The full projected exact-model certificate avoids treating estimated invariants as independent noisy observations.

A one-shot cubic hardware estimate has worst-case error bounded in the form C_eta eta/epsilon^3 + C_tail epsilon^2, within its valid chart. Their balance occurs at epsilon proportional to eta^(1/5), with error proportional to eta^(2/5). This is a property of the uncorrected cubic approximation; it is not a fundamental limit for the exact-model inverse. Exact refinement can remove truncation bias, subject to its branch/coverage certificate.

## 7. Canonical versus full-vector optics

All preceding claims concern the original independent-axis Snell model. Let Delta_model be the difference between it and a physical full-vector propagation model on the candidate box. Any robustness claim for physical data must add a validated bound on Delta_model to the observation error before applying the inverse theorem.

There is no reason to identify this discrepancy with the canonical quintic remainder. Mixed x/y geometric terms in a different vector law can arise at cubic order even at centered incidence. If model discrepancy is O(epsilon^3), its hardware bias after inversion is generally O(1), so decreasing wedge does not resolve it. At tilted incidence lower-order discrepancy can occur. Those orders must be derived from a common physical convention before being asserted as facts for the particular vector model. If no such discrepancy bound is available, keep physical-model conclusions separate.

## 8. What a minimal decisive validation should do

No sweep or new recovery campaign is needed. At the already-proved witness and one stipulated nonzero epsilon:

1. Verify the selected 18-channel construction and exact leading block factorization; reproduce the beam coefficient and scalar closure derivative symbolically.
2. Certify actual finite-clock demixer row norms and its left-inverse residual. The rank witness may be badly conditioned: distinct nodes alone are not a robustness demonstration.
3. Certify an explicit strict optical margin and a differentiated quintic remainder on one named local box.
4. Produce one actual q<1 and resulting all18 uncertainty bound at a supplied eta, or report failure of this test. This is the decisive missing quantitative result.
5. Optionally verify one symbolic amplitude-preserving hardware path and its cubic leading variation, giving a lower-bound coefficient that can be compared with the upper certificate.

The biggest gap is not the shape of the theorem, which follows constructively as above. It is obtaining a useful q and finite-record demixer constant at a physically meaningful wedge and input precision, followed by a manageable complete global cover. The native-box-wide monotone hardware invariant theorem does not supply that cover or resolve finite-data harmonic labeling.

## Primary references and novelty boundary

Batenkov and Yomdin, On the Accuracy of Solving Confluent Prony Systems, https://arxiv.org/abs/1106.1137 and https://epubs.siam.org/doi/10.1137/110836584, already develop local perturbation analysis for confluent Prony systems. Batenkov, Stability and super-resolution of generalized spike recovery, https://arxiv.org/abs/1409.3137, already relates recovery stability to node separation and sample count. Potts and Tasche, Parameter estimation for exponential sums by approximate Prony method, https://www-user.tu-chemnitz.de/~potts/paper/prony.pdf, provides a standard frequency-estimation baseline.

The potentially publishable contribution here is not Prony conditioning itself: it is the original optical map's 5/6/2/5 sensitivity hierarchy, matching five-dimensional cubic lower bounds, explicit integration of a globally injective five-invariant hardware reconstruction, and a finite-record all18 certificate including unknown source and rotor coordinates. Global complete recovery remains a separate theorem/algorithmic gap.

# Physical-vector addendum: the decisive excitation obstruction

The vector-optics audit supplied a stronger result after the canonical theorem above was drafted. For centered axial incidence, the first physical prism exits at a fixed vertex and has deflection

delta_1=arcsin(n_1 sin(alpha_1))-alpha_1.

Consequently an entire admissible curve of (n1,alpha1) with the same delta1 gives exactly the same outgoing ray at every rotor time, hence exactly the same downstream observations. This is a positive-wedge, arbitrary-downstream-optics structural gauge. Its tangent is

d alpha1/d n1 = -sin(alpha1)/[n1 cos(alpha1)-sqrt(1-n1^2 sin(alpha1)^2)].

A full-vector centered analogue of the canonical rank witness cannot have rank eighteen. Distinct speeds and many samples do not remove this gauge.

## Rigorous near-axis noise barrier

Let g_s be an interior segment of this exact gauge, parameterized by n1 so that n1(g_s)-n1(g_0)=s. Let xi=(beta_x,beta_y,p_x,p_y) be the unknown off-axis excitation, expressed in any explicitly fixed dimensionless units. On a compact physical tube where all needed derivatives exist, define

C_G = sup_{|s|<=r,0<=u<=1,xi in Xi} ||D_xi partial_s F_vec(g_s,u xi)||_(infinity<-chosen excitation norm).

This quantity is computed from optical mixed derivatives and the exact symmetry, rather than from a lower bound on an unknown singular value. Since partial_s F_vec(g_s,0)=0, the fundamental theorem of calculus gives the exact identity

F_vec(g_s,xi)-F_vec(g_0,xi)
= integral_0^s integral_0^1 [D_xi partial_v F_vec(g_v,u xi)] xi du dv.

Thus the forward separation is at most C_G |s| ||xi||. A midpoint observation makes both systems compatible with componentwise allowance eta whenever C_G |s| ||xi||<=2eta. Every estimator therefore has worst-case n1 error at least

(1/2) min(r, 2eta/[C_G ||xi||]),

with the natural exact-gauge interpretation at xi=0. This is a rigorous excitation-dependent necessary condition. It makes no unsupported claim about which off-axis component breaks the gauge, or about the sharp power of wedge after nuisance profiling.

If a compatible-set cover still includes the centered gauge segment, it cannot certify a small index uncertainty. A nonzero estimated beam displacement is insufficient: its entire uncertainty set must exclude all relevant degenerate excitation strata, and a separate constructive upper bound must show that the particular excitation actually reveals the gauge direction.

The vector audit reports that the previously suggested alpha1^2 |beta| channel was a direction-only second-harmonic term after removing lower-order components. It is not yet a theorem for the complete positional record with unknown source coordinates. Finite-slab lateral walk contributes additional terms at nonzero tilt, and source-offset compensation must be derived before assigning a sharp exponent.

## Explicit mismatch effect on the canonical certificate

Let Delta(theta)=F_vec(theta)-F_can(theta). If the physical truth theta* and a canonical fit thetahat lie in one certified canonical chart, the exact same proof yields

||S(z_hat-z_*)|| <= [2||L|| eta + ||L Delta(theta*)||]/(1-q).

Using projected discrepancy L Delta is more informative than a uniform forward discrepancy; it asks whether the error enters the identifying channels. If the mismatch lies at cubic order in a hardware channel, division by epsilon^3 produces order-one hardware bias. The canonical fit may instead have no compatible parameter at all; physical consistency is not guaranteed by the local inequality.

For a single centered rotating physical wedge, let s=sin(alpha)(cos gamma,sin gamma), k=n-1 and B=(n-1)(n^2-n+1)/2. The audit derives

T_can=(k s_x+B s_x^3, k s_y+B s_y^3)+O(|s|^5),
T_vec=k s+B |s|^2 s+O(|s|^5).

The canonical pure-third-rotor harmonic B sin(alpha)^3/4 is absent from the true circular vector response. Full-vector rotational covariance requires complex x+i y Fourier indices to have total rotor index sum one at centered axial incidence; the real channels allow the conjugate total minus one. The canonical pure3N and positive-sum cubic hardware channels are therefore inappropriate as physical invariants. A physical inverse needs new mixed-sign or explicitly off-axis invariants, not a declaration that the omitted terms are small.

The audit has a promising noncentered construction: with zero input tilt and nonzero transverse offset b, second-harmonic and mixed-sum coefficients reportedly give ratios identifying the indices, with signal O(|b| alpha^2). These remain separate prospective invariants until their full recurrence, finite-record extraction, and unknown-tilt tangent rank are independently established. Ratios square small quantities and will require denominator-aware noise bounds; algebraic injectivity alone will not establish robust observability.

## Revised publication recommendation

Lead with a model-sensitive identifiability theorem: the canonical independent-axis model has the 5/6/2/5 full18 sensitivity hierarchy, whereas physical vector rotational symmetry causes an exact positive-wedge centered gauge and off-axis-dependent conditioning. Then develop an excitation-aware vector inverse or explicitly present the canonical model as the object of study. Do not combine the canonical hardware uniqueness theorem and the vector physical interpretation without a new derivation.

## Noise convention

These are deterministic componentwise-error bounds. The demixer's relevant noise norm is its row l1 norm; a square-root-of-200 improvement cannot be claimed without an additional stochastic-noise assumption. An independent Gaussian-noise analysis would use row l2 norms/Fisher information and a stated confidence level, but would not remove the structural cubic or excitation barriers.

The scalar hardware worker supplied proposed differential log-invariant constants (including a relative closure slope bound) while this memo was completed. They should be integrated only after their independent audit. Even a uniformly bounded inverse derivative on the image does not automatically prove a global Euclidean endpoint Lipschitz bound if the straight connecting invariant segment leaves that image. A safe finite-noise route is interval triangular reconstruction plus a monotone root bracket with all domain filters retained.
