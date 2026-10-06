# Passive eighteen-parameter Risley inversion: canonical and physical vector results

2026-10-02. This report follows the user's revised priority: construct a general method, using computation only for small checks of specific derivations. Case campaigns and new validation have stopped. Completed empirical work is preserved in [EVIDENCE_REPORT.md](EVIDENCE_REPORT.md), rather than used as the organizing principle here.

> **Two models, two separate proofs.** Sections 2-10 concern the original canonical model, which refracts the transverse axes independently. Its third-harmonic rank construction does not transfer to physical three-dimensional optics: centered axial rotational symmetry forbids the modes it uses. Section 11 gives a separate physical vector result at an unknown nonzero source offset, using quadratic modes and an independent finite-record argument. Neither theorem establishes global uniqueness, useful finite-noise accuracy or a negligible discrepancy between the models.

## Result and proposed architecture

The construction is: **exact optical formulation -> exact bounded geometry elimination -> globally sound optical-region exclusion -> native-coordinate uncertainty certificate**.

The exact canonical model admits a paired-prism transfer law, elimination of four geometry coordinates, and a finite polynomial formulation of the sampled record. Its local theorem identifies eighteen canonical parameters on a sufficiently small nonzero wedge family using the ideal 200-sample record. Five formal canonical cubic hardware invariants admit reconstruction through one scalar equation. A remainder-thickened hierarchy connects polynomial approximations to the exact canonical observation constraints without losing their solutions.

For physical vector optics, an exactly centered axial source has a first-prism index/wedge gauge. An unknown fixed nonzero source offset supplies additional quadratic harmonics. Their formal coefficient identities have an explicit smooth hardware inverse, while negative fundamental channels separate the two unknown beam-tilt directions. A separate 31-column finite-record argument then proves full eighteen-parameter local rank on a sufficiently small off-axis witness family. These are leading-coefficient and local-existence results, not an assumption that noisy finite-amplitude data directly reveal those coefficients.

These are structural results toward one general method. They do not supply an efficient completed global solver, full-record global uniqueness, or a uniformly accurate inverse over the whole prior. An independent proof of uniqueness for the five formal canonical cubic invariants has been reported; its full strict-monotonicity and connectedness argument is pending incorporation, so this version does not rely on that claim. Useful certified remainders, noise constants and complete global exclusion remain necessary for either model. The physical theorem uses its own coefficient identities rather than transferring the canonical proof.

## 1. Original problem and conventions

The native vector is

\[
\theta=(N_{1:3},\alpha_{1:3},\phi_{1:3},n_{1:3},d,g,\beta_x,\beta_y,p_x,p_y).
\]

`alpha` is original `ax`, the signed wedge; `phi` is `ay`, the initial rotor phase. Speeds are signed Hz; wedge, phase and source angles use degrees in the API. Bounds are

\[
|N_i|\le3.5,\quad |\alpha_i|,|\phi_i|\le18^\circ,\quad
1.3\le n_i\le1.8,\quad d\in[50,200],\quad g\in[2,15],
\quad|\beta_h|\le25^\circ,\quad|p_h|\le5.
\]

Lengths use the original native units, not an assumed millimetre calibration. Source distance 6 and prism thickness 3 are fixed. The two gaps share unknown `g`. Physical prism order matters. The canonical equations refract x and y independently. The original note AC uses a different, full-vector 3D model.

The ideal clock is `t_k=k/20`, `k=0,...,199`. The binary64 `arange` clock in the executable is a different exact input array. The polynomial recurrence below uses the ideal clock; an interval implementation can use the stored timestamps directly. Neither clock nor legacy evaluation error is silently ignored.

For the full 400-scalar record, the object to recover is

\[
\mathcal C_\eta(y)=\{\theta\in B_0\cap\mathcal P:
|F_m(\theta)-y_m|\le\eta_m\text{ for every }m\}.
\tag{1}
\]

`B0` is the complete native box; `P` is the original strict transmitted, forward, nongrazing branch. A scalar allowance means every `eta_m=eta`. The native target is `.001` in each coordinate. Input precision is a supplied constraint, not a quantity that can be reduced until a proof succeeds.

The original mathematical implementation is `risley_lattice/fmodel.py`, especially `_trace_axis`; `reverse_problem_v2/core.py` is the legacy floating evaluator. The newest substantive source notes are `DIARY.md` from line 10032 onward and `paper/P2_THEORY_2026_09_28/BRIEF.md`. The source notes `Y_global_quartic_shape.md`, `AA_two_screen_angular_inverse.md`, `AB_two_screen_geometry.md` and `AC_vector_layer_stripping.md` concern coefficient inversion or controlled protocols with different information from this passive record. Source interpretation and completed empirical evidence are documented in [EVIDENCE_REPORT.md](EVIDENCE_REPORT.md).

### Physical vector optics: a symmetry obstruction to transferring the proof

For a centered axial incident beam and rotationally symmetric reference/screen planes, let \(Z(\gamma)=x(\gamma)+iy(\gamma)\) be the physical vector trace, with \(\gamma=(\gamma_1,\gamma_2,\gamma_3)\). Rotating every prism by the same angle \(\tau\) rotates the entire optical configuration and output:

\[
Z(\gamma+\tau\mathbf 1)=e^{i\tau}Z(\gamma).
\tag{V1}
\]

This follows from rotational covariance of vector refraction and ray-plane intersections under the stated symmetric source and plane conditions. It does not assume that each transverse axis refracts independently.

Whenever a common physical branch defines a torus Fourier expansion, write
\(Z(\gamma)=\sum_m C_m e^{im\cdot\gamma}\). Substituting in (V1) and using uniqueness of Fourier coefficients gives

\[
[e^{i\tau(m_1+m_2+m_3)}-e^{i\tau}]\,C_m=0
\quad\hbox{for every }\tau.
\]

Consequently

\[
C_m=0\quad\hbox{unless }m_1+m_2+m_3=1.
\tag{V2}
\]

Real coordinate traces combine \(Z\) and its conjugate, so their allowed weights are \(+1\) and \(-1\). The same selection rule applies to a formal Taylor expansion of the equivariant trace near a regular centered state. If a physical branch is not defined on the whole torus, a global torus expansion is not assumed merely from sampled transmission.

In particular, the pure torus modes \(m=\pm3e_i\) are forbidden in physical vector optics under these conditions. The all-plus cubic modes \(2e_1+e_2\) and \(2e_2+e_3\) used in the canonical proof also have forbidden weight \(3\). Mixed difference modes such as \(2e_i-e_j\) have weight \(1\) and are allowed. At the 20 Hz sample rate, carriers coincide when \((m-m')\cdot N\in20\mathbb Z\). A sampled temporal line labelled \(3N_i\) may therefore represent an allowed mixed mode; aliasing does not restore the forbidden pure torus coefficient.

The canonical independent-axis map produces the pure third-harmonic modes used to isolate \(I_{300},I_{030},I_{003}\) in section 7. Their presence therefore cannot supply a rank certificate for the physical vector map. The canonical theorem remains valid for its declared equations. It proves neither vector-model rank nor an acceptably small discrepancy between the two models. Unknown nonzero source offsets or tilt break the centered symmetry, but do not justify transferring a rank witness evaluated at a centered axial source; source derivative columns have their own transformation rules.

This obstruction does not prove that rank eighteen is impossible for the vector model: allowed mixed modes may carry the missing information. Rotational covariance is also not an observational gauge, because rotating the system rotates the measured trace, and a common phase shift need not preserve the restricted native phase prior.

Section 11 supplies a separate vector construction: an unknown nonzero source offset excites quadratic modes, and an independent finite-record argument retains all eighteen unknowns. It does not transfer the centered canonical cubic proof. Any use of the canonical model as a physical approximation still requires a certified model-discrepancy enclosure in addition to measurement error. The remainder bounds in section 9 compare a canonical Taylor polynomial with the exact canonical model; they are not bounds on this physical discrepancy.

## 2. Exact paired-prism law

The effective exit-face sine is `f_i=sin(alpha_i)cos(gamma_i)` on x and `sin(alpha_i)sin(gamma_i)` on y, where `gamma_i=2pi N_i t+phi_i` after conversion to radians. This is exact, not a small-angle approximation.

For one prism, axis and sample, let `X` be the incoming unit transverse air direction and `p` the flat-entry coordinate. Define

\[
c=\sqrt{1-f^2},\quad H=\sqrt{n^2-X^2},\quad Q=cX+fH,
\quad R=\sqrt{1-Q^2},
\]
\[
a=cQ-fR,\quad v=fQ+cR,\quad D=cH-fX.
\tag{2}
\]

Take positive roots and require `R>0`, `v>0`. Then `a^2+v^2=1`, `cv-fa=R`, and the outgoing slope is `T=a/v`. The full native bounds imply

\[
D\ge\cos18^\circ\sqrt{1.3^2-1}-\sin18^\circ>.48.
\tag{3}
\]

Intersecting the internal ray with the exit plane gives `p_exit=c(Hp+3X)/D`. Propagating to the next nominal parallel plane yields

\[
\boxed{p_{\rm next}=L p+K+\ell T,
\qquad L=\frac{RH}{vD}>0,\quad K=\frac{3XR}{vD}.}
\tag{4}
\]

The next separation is `ell=g,g,d` for prisms 1, 2, 3. This retains Snell branches and never divides by wedge amplitude, speed or a frequency difference. It adds no positive-travel-distance constraint absent from the original verifier. The derivation in both directions is in [algebraic_record.md](theory/algebraic_record.md).

## 3. Exact geometry elimination

Starting at `p_h+6tan(beta_h)` and composing (4) gives, at every sample and axis,

\[
F=A p_h+B+Cg+Ed,
\tag{5}
\]
\[
\begin{aligned}
A&=L_3L_2L_1>0,\\
B&=6A\tan\beta_h+L_3L_2K_1+L_3K_2+K_3,\\
C&=L_3L_2T_1+L_3T_2,\qquad E=T_3.
\end{aligned}
\]

These coefficients depend only on the fourteen optical variables: three speeds, wedges, phases, indices, and two source angles. The unknown geometry remains `ell=(d,g,p_x,p_y)` in its complete native box. Stacked observations have the exact form

\[
F(q,\ell)=b(q)+M(q)\ell.
\tag{6}
\]

For fixed `q`, hard-noise feasibility is a bounded four-variable LP with 808 inequalities. Rank deficiency returns a geometry polytope; it does not authorize freezing a parameter.

Because `A>0`, source position on each axis belongs to the intersection of `[-5,5]` and

\[
\left[\frac{y_k-B_k-C_kg-E_kd-\eta}{A_k},
\frac{y_k-B_k-C_kg-E_kd+\eta}{A_k}\right]
\tag{7}
\]

over all samples. The intersection is nonempty exactly when every lower endpoint is no greater than every upper endpoint. These are halfplanes in `(g,d)`. Their intersection with the native rectangle is the complete compatible geometry polygon; (7) then reconstructs all source positions. No distinct-speed, nonzero-wedge or matrix-rank assumption is used.

For noiseless data, normalize and subtract sample 0 separately on each axis. The 398 equations are `Delta z=g Delta U+d Delta V`. Rank-two charts determine `(g,d)` from two rows and impose augmented three-by-three minors on the remaining rows. Rank-one and rank-zero consistency charts must also be retained.

### Optical-only feasibility without unstable normalization

Let `Y_k=y_k-B_k`. Multiplying the source constraints by positive `A_k` and the pair constraints by positive `A_k A_l` gives

\[
-C_kg-E_kd\le5A_k+\eta-Y_k,
\quad C_kg+E_kd\le5A_k+\eta+Y_k,
\tag{8}
\]
\[
-(C_kA_l-C_lA_k)g-(E_kA_l-E_lA_k)d
\le\eta(A_k+A_l)-Y_kA_l+Y_lA_k.
\tag{9}
\]

Add both axes and the geometry rectangle. These unnormalized inequalities avoid division by small `A`, which can approach zero near critical transmission.

For a finite family of halfplanes `n_i dot(g,d)<=h_i`, infeasibility has a certificate with at most three rows. All rank cases are explicit: a zero normal with negative right side; two opposing collinear normals with negative positively weighted right side; or a rank-two triple. For the latter, dependence coefficients are the three cyclic determinants. They must have a common sign up to simultaneous reversal, and their weighted right side must be negative. Degenerate zero coefficients reduce to the smaller cases.

Consequently, **absence of every such bad circuit is an exact finite-noise feasibility predicate in the fourteen optical variables**, with geometry reconstructed afterward. Positive denominator clearing makes it polynomial in the lifted optical variables. This is a principled symbolic elimination target. It is not a recommendation to enumerate all triples, nor a tractability or uniqueness theorem. A practical LP can propose a sparse active circuit for independent verification and region subdivision.

The detailed geometry derivation and general dual bounds are in [elimination.md](theory/elimination.md). A small polygon/LP check agreed numerically; the naive polygon was slower, so the unnormalized four-variable LP remains the practical backend.

The division-free inequalities, exhaustive circuit cases and their proof are written separately in [geometry_circuits.md](theory/geometry_circuits.md). No combinatorial circuit campaign was run.

## 4. Exact sampled-rotor and optical formulation

For ideal `t_k=k/20`, set

\[
u_i=\cos(\pi N_i/10),\quad w_i=\sin(\pi N_i/10),\quad
u_i^2+w_i^2=1,\quad u_i\ge\cos(7\pi/20)>0.
\tag{10}
\]

The arc gives a unique native speed by `N_i=(10/pi)atan2(w_i,u_i)`. Keep signed `S_i=sin(alpha_i)` and a separate phase-circle pair with cosine at least `cos18deg`. Initialize `(F_i0,G_i0)=S_i(cos phi_i,sin phi_i)` and impose

\[
F_{i,k+1}=u_iF_{ik}-w_iG_{ik},\qquad
G_{i,k+1}=w_iF_{ik}+u_iG_{ik}.
\tag{11}
\]

Use `F_ik` as the x face sine and `G_ik` as the y face sine. Equations (2), positive-root inequalities and

\[
vD p_{\rm next}=R(Hp+3X)+\ell aD
\tag{12}
\]

give a polynomial lift. Adjacent prisms share the actual preceding rays and positions; hidden states cannot independently absorb measurement error. Source direction circles, first-entry equations and all 400 observation strips complete the model.

There is a proved equivalence in both directions between lifted solutions and full native compatible hardware. No native case is removed: a zero wedge leaves its whole phase/speed fiber; slow and equal rotors remain; physical order is unchanged. Retaining phase variables prevents an artificial gauge choice at zero amplitude.

An explicit lift of degree at most three has 10,828 variables, 10,810 equalities and 6,832 inequalities for 200 samples. Most variables are triangular auxiliaries. These counts neither prove independent equations nor promise efficient computation. Complete real-algebraic decomposition provides a decision prescription in principle, including positive-dimensional cells, but no practical global complexity bound has been established. The stored-clock extension has different counts and coefficient complexity. See [algebraic_record.md](theory/algebraic_record.md).

## 5. One globally sound search algorithm

Maintain a cover of every compatible native system, starting with the complete optical and geometry boxes.

1. Propagate interval optical equations and coefficient bounds on each optical region. Reject only a proved whole-region physical violation. A failed branch enclosure is UNKNOWN, not rejection.
2. Form common-geometry outer LP constraints. All observation rows must constrain the same four geometry variables. Source-sign splits or centered interval support bounds handle coefficient products safely.
3. Use approximate numerical duals only as proposals. Verify an outward or exact exclusion/contracting inequality before discarding or shrinking a region.
4. Split unresolved optical regions fairly and contract geometry conditionally on each region. Preserve singular fibers and boundary regions. Spectral estimates may prioritize work; they do not remove unvisited alternatives.
5. Independently verify compatible candidates against every observation and guard. Use the complete surviving cover for a native accuracy claim. If cells cannot be settled, return them as unresolved or use an exact feasibility backend.

For example, write the geometry box as `c+[-r,r]`, and `f_c(q)=b(q)+M(q)c`. Every compatible point satisfies, for any fixed rational observation-space vector `z`,

\[
z^T[y-f_c(q)]\le\eta\|z\|_1+\sum_jr_j|z^TM_j(q)|.
\tag{13}
\]

A validated strict violation over an entire optical box excludes it. The defect `M^Tz` is retained; no exact numerical nullspace equality is assumed. More generally, for an outer LP `G ell<=h`, `lambda>=0` certifies infeasibility when

\[
\overline{\lambda^Th-\min_{\ell\in L_0}(G^T\lambda)^T\ell}<0.
\tag{14}
\]

This support correction makes approximate duals usable without making them exact by assertion. Signed combined expressions can retain cancellations that separate absolute-value bounds lose.

Finite interval-search termination follows on compact, uniformly guarded regions with a positive incompatibility gap and convergent bounds. It does not follow automatically at strip boundaries, singular fibers or the open critical-transmission boundary. Complete-set bookkeeping and the exact fallback are specified in [global_initialization.md](theory/global_initialization.md).

## 6. Native accuracy and conditioning

For the compatible set, let `l_j` and `u_j` be coordinate infima and suprema. An arbitrary real-valued estimate has minimax native maximum-coordinate error

\[
\frac12\max_j(u_j-l_j).
\tag{15}
\]

Its coordinate midpoint need not be physically compatible. For a verified physical estimate, every retained region must lie within its requested native error box. Multiple narrow boxes around distant systems do not establish recovery. Two verified compatible systems separated by more than `.002` in a coordinate refute uniform `.001` estimation for that record.

On a certified smooth box `theta=c+h`, local Taylor information gives useful global-search contractors. With reference value `f`, Jacobian `J`, signed remainder enclosure `R_z`, any native direction `v` and proposed multiplier `z` obey

\[
v^Th\le-z^T(f-y)+|z|^T\eta+s_H(v-J^Tz)-\underline R_z.
\tag{16}
\]

The support term covers inverse/dual error; the signed Hessian remainder preserves cancellation. Apply both coordinate signs, then combine bounds over every surviving global region. A local chart is not evidence that the true hardware lies there.

The exact affine/Schur rank structure, a constructive local inverse and the precise connected-component analytic genericity result are in [identifiability.md](theory/identifiability.md). [noise_bounds.md](theory/noise_bounds.md) proves the full-cover accuracy test, nonlinear contractors and termination qualifications. A checked 18-row minor anchors local rank under stated arithmetic assumptions; it does not prove global injectivity.

## 7. Full eighteen-parameter local identifiability on the finite record

The first-order response factors through at most fourteen real quantities: two baselines, three speeds, three shared phases and six real axis/prism amplitudes. Higher optical orders must supply the missing information. On the centered, normally incident stratum, odd symmetry makes cubic interactions the next informative order.

**Canonical-model theorem.** Let

\[
h_*=(n_1,n_2,n_3,g,d)=(3/2,4/3,5/3,3,100),\qquad
N_*=(1,7,49)/20,\qquad \phi_*=0.
\]

Set both source angles and positions to zero at the witness, and write

\[
e_i=\sin\alpha_i=\varepsilon A_i/K_i(h),\qquad
K_i=(n_i-1)\left[d+(3-i)g+3\sum_{j>i}1/n_j\right],
\tag{17}
\]

with fixed nonzero amplitudes, for example \(A_i=1\). For all sufficiently small positive \(\varepsilon\), the exact canonical independent-axis map to the 200 x/y positions at \(t_k=k/20\) has Jacobian rank eighteen at this family. All eighteen native coordinates are free in taking the derivative, including the unknown beam angles and source offsets. That canonical map is locally injective there. The forbidden-mode argument (V2) prevents interpreting this particular proof as a physical vector-optics rank certificate.

The theorem gives no explicit upper threshold for \(\varepsilon\), noise tolerance or global uniqueness. Small wedges place the witness inside the native bounds and on the strict physical branch by continuity from flat normal incidence.

### Cubic recurrence and the hardware determinant

Use radians and homogeneous expansions \(X=x_1+x_3+O(5)\), \(p=p_1+p_3+O(5)\). For one effective face sine \(s\), put \(\kappa=n-1\). Expansion of the exact law gives

\[
b_1=x_1+\kappa s,\qquad
b_3=x_3+\kappa x_1s^2+\frac{\kappa}{2n}x_1^2s+
 \frac{n\kappa}{2}s^3,
\]
\[
q_1=p_1+3x_1/n+\ell b_1,
\]
\[
\begin{aligned}
q_3={}&p_3+3x_3/n+\ell b_3+\ell b_1^3/2+3x_1^3/(2n^3)\\
&-\kappa s^2p_1-(\kappa/n)x_1sp_1
 -(3\kappa/n)x_1s^2-(3\kappa/n^2)x_1^2s.
\end{aligned}
\tag{18}
\]

Initialize all four homogeneous components to zero and compose the three prisms with separations \(g,g,d\). The final position is

\[
p_1=\sum_iK_is_i,\qquad p_3=\sum_{|m|=3}c_m(h)s^m.
\]

Normalize the five coefficients indexed by \(m=300,030,003,210,021\):

\[
I_m(h)=c_m(h)/\prod_i K_i(h)^{m_i}.
\tag{19}
\]

At \(h_*\), the gains are \((2201/40,524/15,200/3)\). In the displayed invariant order and hardware order \((n_1,n_2,n_3,g,d)\), exact rational differentiation gives

\[
\det D_hI(h_*)=
-\frac{2342246801318629443}
 {280149096378993020072641029913075712000}\ne0.
\tag{20}
\]

This determinant was independently obtained from the exact paired trace, and reproduced by one targeted rational check in [cubic_rank_check.json](theory/cubic_rank_check.json). It establishes a nonzero determinant, without interpreting cubic coefficients as directly measured data.

### Unknown source coordinates and the finite-sample argument

To differentiate the beam/source coordinates, initialize (18) with
\(x_1=\beta,\ x_3=-\beta^3/6,\ p_1=p+6\beta,\ p_3=2\beta^3\).
Let \(B_{\rm beam}=d+2g+6+3\sum_i1/n_i\). If \(q_p,q_\beta\) denote the coefficients of \(p s_3^2,\beta s_3^2\), then

\[
q_p=1-n_3,\qquad
B_{\rm beam}q_p-q_\beta=-\frac d2(n_3-1)(3n_3+1).
\tag{21}
\]

The second expression equals \(-200\) at the witness and is nonzero throughout the native index/distance bounds. DC and the \(2N_3\) harmonic therefore distinguish each source offset from its beam angle. Centering the witness does not remove their derivative columns.

For the chosen speeds, all degree-at-most-three temporal nodes are

\[
z_m=\exp[2\pi i(m_1+7m_2+49m_3)/400],\qquad |m|_1\le3.
\tag{22}
\]

There are 63 distinct nodes. Their integer labels lie in \([-147,147]\), so no modulo-400 collision occurs. A difference of two labels has digits in \([-6,6]\): a nonzero third digit dominates \(42+6\), and a nonzero second digit with third digit zero dominates 6. Thus the labels are unique.

Speed derivatives add \(kz_m^k\) at the six signed fundamental nodes, giving 69 confluent columns. Their first 69 rows are independent: an annihilating row polynomial of degree at most 68 would have 63 zeros, six also with zero derivative, hence 69 zeros counted with multiplicity. It must be zero. Fixed real linear functionals of the 200 samples can therefore separate the needed fundamental, cubic and beam channels. Frequencies \(3N_i,\ 2N_1+N_2,\ 2N_2+N_3\) isolate the five selected cubic monomials, up to nonzero trigonometric factors.

Use local coordinates \((A_{1:3},N_{1:3},\phi_{1:3},h,\text{source}_4)\). Scale the nine amplitude/phase/speed columns by \(\varepsilon^{-1}\), the five hardware columns at fixed \(A\) by \(\varepsilon^{-3}\), and each compensated beam column \((\delta\beta,\delta p)=(1,-B_{\rm beam})\) by \(\varepsilon^{-2}\); retain ordinary offset columns. The projected eighteen-by-eighteen Jacobian tends to a block triangular matrix. Its blocks are the nine independent fundamental directions, the nonzero determinant (20), and two nonzero beam/source blocks (21).

The exact centered optical response has next remainder \(O(\varepsilon^5)\); beam/source derivatives have next remainder \(O(\varepsilon^4)\). Analyticity on a strict branch and the finite number of samples justify the differentiated expansions. All scaled remainders tend to zero. The limiting determinant is nonzero, so continuity and the inverse function theorem prove the stated exact finite-record result. The complete derivation is in [full18_local_rank.md](theory/full18_local_rank.md).

### Genericity and conditioning

Analyticity gives generic rank eighteen on the connected admissible interior component containing this witness. For the ideal clock, use the injective native algebraic arc coordinates from section 4, preserving zero-wedge phase/speed fibers. Outside a lower-dimensional subset of that component's eighteen-dimensional image, an exact output fiber consists of isolated points; semialgebraicity makes it finite. This concerns \(F(\theta)=y\), not positive-width observation strips, and does not establish uniqueness or any claim about other components. Boundary persistence, when needed, uses the finite-output projection of the forward graph's frontier because the map may be undefined at excluded critical boundaries.

The same proof exposes five weak native directions. At fixed speeds/phases, vary \(h\) and compensate the wedges by

\[
\delta\alpha_i=-\tan\alpha_i\,\delta\log K_i
\tag{23}
\]

in radians. First-order amplitudes remain fixed and the output derivative is \(O(\varepsilon^3)\). Local rank is therefore compatible with severe finite-precision uncertainty. With positive observation allowance and strict residual slack, an interior compatible point has an open neighborhood of compatible systems.

## 8. Five formal cubic invariants reduce to one scalar closure

This is a global reconstruction of the five-dimensional **formal cubic hardware map of the canonical independent-axis model**. It does not assert that finite-amplitude, unknown-beam observations already supply these five invariants, or that these invariants identify physical vector-optics hardware.

Write
\[
P_1=I_{300},\quad P_2=I_{030},\quad P_3=I_{003},\quad
J_{12}=I_{210},\quad J_{23}=I_{021}.
\]
Define
\[
\mathcal F(n)=1+\frac n{(n-1)^2},\qquad
\delta(n)=3(1/n-1/n^3),
\]
\[
L_3=d,\quad L_2=d+g+3/n_3,\quad
L_1=d+2g+3/n_2+3/n_3,
\]
\[
M_3=L_3,\quad M_2=L_2-\delta(n_3),\quad
M_1=L_1-\delta(n_2)-\delta(n_3).
\]

The cubic recurrence gives the exact identities

\[
2P_iL_i^2=\mathcal F(n_i)-(L_i-M_i)/L_i,
\]
\[
J_{23}=\frac{3d(n_3+1)-2L_2}{2dn_3L_2^2},\qquad
J_{12}=\frac{3n_2M_2+3L_2-2L_1}{2n_2L_2L_1^2}.
\tag{24}
\]

Both mixed invariants are positive on the original hardware box. For \(n>1\), \(\mathcal F\) is strictly decreasing and

\[
\mathcal F^{-1}(T)=1+\frac{1+\sqrt{4T-3}}{2(T-1)}.
\tag{25}
\]

The native index interval corresponds to \(T\in[61/16,139/9]\).

Set the sole branch variable \(t=n_3\in[1.3,1.8]\). Reconstruct successively

\[
d(t)=\sqrt{\mathcal F(t)/(2P_3)},
\qquad
L_2(t)=\frac{3d(t)(t+1)}
 {1+\sqrt{1+6J_{23}d(t)^2t(t+1)}},
\]
\[
g(t)=L_2(t)-d(t)-3/t,\qquad
n_2(t)=\mathcal F^{-1}\!\left(2P_2L_2(t)^2+
 \frac{\delta(t)}{L_2(t)}\right),
\]
\[
V(t)=2L_2(t)-d(t)+3/n_2(t)-3/t.
\tag{26}
\]

The remaining mixed invariant supplies one scalar equation:

\[
\boxed{\Psi(t)=2J_{12}n_2L_2V^2+2V-3L_2
 -3n_2[L_2-\delta(t)]=0.}
\tag{27}
\]

At each retained root, recover

\[
n_1=\mathcal F^{-1}\!\left(2P_1V^2+
 \frac{\delta(n_2)+\delta(t)}{V}\right).
\tag{28}
\]

Require every radicand/domain condition, \(P_3>0\), the native inverse-index intervals, and all original bounds on \(n_i,g,d\). Then \(V=L_1>0\). Equations (26) select the unique positive distance and quadratic \(L_2\) roots; (25) selects the unique index on \(n>1\). Substitution in (24) proves that filtered roots of (27) correspond one-to-one with all hardware vectors realizing these five formal invariants. These reconstruction formulas alone do not prove monotonicity of \(\Psi\), a single admissible root or exclusion of exceptional continuous fibers.

**Proof-integration status.** Independent strict-monotonicity and connectedness arguments have been reported to establish global uniqueness of this five-invariant canonical map on the stated native hardware box. Their complete proof is not yet included here. This report preserves the reconstruction and its proved correspondence without using the newly reported uniqueness conclusion. Even a completed uniqueness theorem for these five formal invariants would establish neither global uniqueness of the full finite record nor identification under physical vector optics.

A polynomial representation makes the remaining branch structure explicit. Let \((t,d,L,r)=(n_3,d,L_2,n_2)\) and \(W=rt(2L-d)+3t-3r\). Then

\[
\begin{aligned}
&2P_3d^2(t-1)^2-(t^2-t+1)=0,\\
&2J_{23}dtL^2+2L-3d(t+1)=0,\\
&[2P_2L^3t^3+3(t^2-1)](r-1)^2-Lt^3(r^2-r+1)=0,\\
&2J_{12}LW^2t+2Wt^2-3Lrt^3-3r^2(Lt^3-3t^2+3)=0.
\end{aligned}
\tag{29}
\]

Their total degrees are \(4,4,8,8\), giving a coarse Bezout bound of 1024 for a generic finite isolated complex fiber. This is neither a practical root count nor a finite bound on an exceptional continuum. Positivity and native-domain filters remain essential after clearing denominators.

The identities and branch inequalities were independently symbolically audited. This integration did not launch new numerical searches. Detailed definitions and proof are in [cubic_scalar_inverse.md](theory/cubic_scalar_inverse.md).

## 9. A complete outer approximation can be tightened to the exact model

Let \(\mathcal T_3\) denote the full degree-three Taylor polynomial, including unknown beam angles and source offsets; it is distinct from the single invariant \(P_3\) above. On a parameter box, suppose a uniform componentwise remainder bound is certified:

\[
F=\mathcal T_3+R,\qquad |R|\le\tau.
\]

For \(0\le\lambda\le1\), define

\[
F_\lambda=\mathcal T_3+\lambda R,\qquad
\mathcal C_\lambda=
\{\theta:\ |F_\lambda(\theta)-y|
\le\eta+(1-\lambda)\tau\}.
\tag{30}
\]

**Conditional enclosure theorem.** For a fixed certified box and bound \(\tau\), these tubes are nested:
\(\mathcal C_{\lambda_2}\subseteq\mathcal C_{\lambda_1}\) if \(\lambda_2\ge\lambda_1\).
Every exact-model compatible point remains feasible at the same parameter vector throughout, and \(\mathcal C_1\) is exactly the original observation condition.

Indeed,
\[
|F_{\lambda_1}-y|
\le |F_{\lambda_2}-y|+
(\lambda_2-\lambda_1)|R|
\le\eta+(1-\lambda_1)\tau.
\]
This proves the nesting and the endpoint preservation directly. Begin with a **complete remainder-thickened polynomial cover**, then tighten or exclude regions with verified inequalities. A collection of nominal centered-cubic roots is not such a cover when the source/beam coordinates are unknown.

Both the exact model and the full unknown-beam cubic polynomial remain affine in \((d,g,p_x,p_y)\). Hence every \(F_\lambda\) retains the four-variable geometry LP when \(\tau\) is a boxwise constant bound. The polygon and positive-coefficient circuit reduction applies where the source-offset coefficient is certified positive; otherwise retain the unnormalized LP.

One route to a computable remainder jointly scales \((e=\sin\alpha,\beta,p)\) by a scalar \(z\), leaving the other unknowns free over the box. The exact trace \(G(z)\) is odd on coherent branches. If those branches are certified analytic on the whole complex disk \(|z|\le\rho>1\), with \(|G_{hk}(z)|\le M_{hk}\), Cauchy bounds give

\[
|F_{hk}-(\mathcal T_3)_{hk}|
\le \frac{M_{hk}\rho^{-5}}{1-\rho^{-2}}.
\tag{31}
\]

After odd degree \(2q+1\), the numerator becomes \(M_{hk}\rho^{-(2q+3)}\). An alternative is a validated fifth-derivative bound over the entire real scaling path, divided by \(120\). Branch analyticity on the whole disk or path is a substantial condition; sampled endpoint validity does not imply it. If this route fails near a critical branch, the box remains unresolved. A direct interval enclosure of \(F-\mathcal T_3\) on an endpoint-safe box is a looser fallback.

Near the local-rank witness, noiseless truncation error \(O(\varepsilon^5)\) and hardware sensitivity \(O(\varepsilon^3)\) suggest, and under uniform local inverse bounds yield, a hardware correction \(O(\varepsilon^2)\). Measurement perturbations instead amplify as \(O(\eta/\varepsilon^3)\). These asymptotic statements provide no usable noise threshold without explicit constants.

Projected Newton or Krawczyk validation can certify only the selected eighteen equations or projections to which it is applied. Every one of the 400 observation constraints and all branch/bound conditions still require independent enforcement. The hierarchy is sound under its stated enclosures; practical global convergence and useful precision have not been established.

## 10. Role of a harmonic initializer

The per-axis response has the exact reflection form \(x=H_x(\cos\gamma)\) and \(y=H_y(\sin\gamma)\). A tensor-Chebyshev dictionary ties reflected sidebands and has 56 real columns per axis at degree five, with six nonlinear rotor coordinates. Its derivative was implemented and spot-checked.

This is an optional initializer. Free coefficients absorb individual rotor reversals and permutations; exact optical completion is required. A whole-torus analytic-tail certificate requires more than sampled physical validity. Explicit measurement, tail, derivative, rotor-box and nullspace error charges prevent treating optical derivatives as measurements. Current generic tail bounds require too many free coefficients for 200 samples at useful precision. See [harmonic_invariants.md](newtheory/harmonic_invariants.md).

## 11. Physical vector optics: exact graph and off-axis local identifiability

This section uses coupled vector refraction throughout. It establishes a separate local theorem for the same eighteen kinds of native coordinates and the ideal clock, under the physical surface geometry defined here. It does not reuse the canonical third-harmonic determinant. The supplied independently audited derivation is preserved in [physical_vector_inverse.md](theory/physical_vector_inverse.md).

### Exact surfaces, branches and conditional geometry

Write the source offset as \(b=p_x+ip_y\), and let \(q\) be the complex transverse component of the normalized incident direction \((\tan\beta_x,\tan\beta_y,1)\). Its axial component is \(\sqrt{1-|q|^2}>0\). At zero tilt, the derivative from \(\beta_x+i\beta_y\) to \(q\) is the identity when angles are radians.

Set \(u_j=\tan(a_j)(\cos\gamma_j,\sin\gamma_j)\), with \(\gamma_j=2\pi N_jt+\phi_j\). The flat entrance planes are \(z=6,\ 9+g,\ 12+2g\); the corresponding exit planes are
\[
z=9+u_1\cdot p,\quad
z=12+g+u_2\cdot p,\quad
z=15+2g+u_3\cdot p.
\]
The screen is \(z=15+2g+d\). Exit normals are \((-u_j,1)/\sqrt{1+|u_j|^2}\). These planes fix both axial vertex thickness 3 and the sign convention for wedges.

At one flat entrance, let \(X\in\mathbb R^2\) be the incoming transverse direction cosine. Introduce \(H>0\), outgoing direction \((B,v)\), and
\[
H^2+|X|^2=n^2,\qquad
B-X+u(v-H)=0,\qquad |B|^2+v^2=1,
\]
\[
v>0,\qquad P=H-u\cdot X>0,\qquad R=v-u\cdot B>0.
\tag{V3}
\]
The tangential equations express vector optical-momentum conservation, and the inequalities select the forward transmitted nongrazing branch. The standard vector-refraction principle is described in Ken Moore's [Ansys/Zemax technical reference](https://optics.ansys.com/hc/en-us/articles/42661810210707-What-is-a-ray); the inverse construction here is not attributed to that reference.

Let \(W=vX-HB\), \(V=vX-(u\cdot X)B\). Exact propagation to the next flat plane, whose axial separation from the exit vertex is \(\ell\), obeys
\[
vP\,p_{\rm next}=vP\,p+W(u\cdot p)+3V+\ell PB.
\tag{V4}
\]
Indeed the axial internal traversal is \(h=H(3+u\cdot p)/P\), and
\[
p_{\rm next}=p+(3+u\cdot p)X/P+(3+\ell-h)B/v.
\]
Multiplying and collecting terms gives (V4). For physical sequential traversal also require positive internal and external propagation, including \(3+u\cdot p>0\) and \(3+\ell-h>0\). The small-wedge witnesses below satisfy these inequalities strictly. These are declared physical-model constraints, not silent additions to the earlier canonical verifier.

Using auxiliary variables for dot products, \(W,V,P\), the graph has polynomial equations of degree at most three; branch signs select roots without square-root evaluations in the graph. Initial propagation is \(v_0p_{\rm entry}=v_0b+6q\), with \(|q|^2+v_0^2=1,\ v_0>0\).

Directions do not depend on source position or \(g,d\). Composing (V4) with \(\ell=(g,g,d)\) therefore gives, exactly at arbitrary admissible wedges and beam angles,
\[
F_{\rm vec}(t)=A(t)b+B_0(t)+gC(t)+dD(t),
\tag{V5}
\]
where \(A\) is a real \(2\times2\) matrix and the remaining coefficients are real two-vectors depending on the fourteen optical coordinates. Fixed optics and componentwise error strips thus give a four-variable LP in \((p_x,p_y,g,d)\), including the native box and affine surface-order inequalities. Euclidean residual bounds instead give second-order-cone constraints. Vector coupling preserves the four-dimensional affine reduction, but the canonical scalar-offset polygon formulas require a new derivation before reuse.

### A centered exact gauge and the first physical discrepancy

At \(b=q=0\), the first exit point is its on-axis vertex \((0,0,9)\), and its deflection is
\[
\Delta_1=\arcsin(n_1\sin a_1)-a_1.
\]
Every downstream ray depends on \(n_1,a_1\) only through \(\Delta_1\). The level set of \(\Delta_1\) is therefore an exact one-dimensional gauge, with tangent
\[
\frac{da_1}{dn_1}
=-\frac{\sin a_1}
 {n_1\cos a_1-\sqrt{1-n_1^2\sin^2a_1}}.
\tag{V6}
\]
Full eighteen-parameter rank cannot hold at this centered physical witness. This obstruction is stronger than simply lacking the canonical pure third harmonics.

For one centered prism, put \(r=\sin a\), \(s=r(\cos\gamma,\sin\gamma)\), \(k=n-1\), and \(\mathcal B=k(n^2-n+1)/2\). Its slopes expand as
\[
\text{canonical: }ks+\mathcal B(s_x^3,s_y^3)+O(r^5),
\qquad
\text{vector: }ks+\mathcal B|s|^2s+O(r^5).
\]
The canonical complex slope contains an artificial term
\((\mathcal B r^3/4)e^{-3i\gamma}\); the leading maximum slope-vector discrepancy is \(\mathcal B r^3/2\). This has the same order as the canonical cubic identification signal. Small relative error in total deflection consequently does not establish validity of cubic hardware inference in physical optics.

### A reproducible vector recurrence

The following recurrence derives the physical coefficient identities. Use complex \(s_j=\sin(a_j)e^{i\gamma_j}\), \(k_j=n_j-1\), and grade \(b,q,s\) jointly as degree one. Write incoming direction and position as \(x_1+x_3+O(5)\), \(r_1+r_3+O(5)\). Initialize
\[
x_1=q,\quad x_3=0,\quad
r_1=b+6q,\quad r_3=3|q|^2q.
\]
For one prism, abbreviate \(n=n_j,\ k=n-1,\ s=s_j\), and take \(\ell=g,g,d\). Then
\[
y_1=x_1+ks,
\]
\[
y_3=x_3+\frac{k}{2n}|x_1|^2s+\frac k2|s|^2x_1
       +\frac k2s^2\overline{x_1}+\frac{kn}{2}|s|^2s,
\]
\[
r_{1,\rm next}=r_1+3x_1/n+\ell y_1,
\]
\[
\begin{aligned}
r_{3,\rm next}={}&r_3+3x_3/n+\ell y_3+
 \frac{3|x_1|^2x_1}{2n^3}+\frac{\ell}{2}|y_1|^2y_1\\
&-k(x_1/n+s)\operatorname{Re}
   [\overline{s}(r_1+3x_1/n)] .
\end{aligned}
\tag{V7}
\]
Replace \(x_1,x_3\) by \(y_1,y_3\) after each prism. Expansion of (V3)-(V4) gives (V7); the \(|y_1|^2y_1/2\) term converts a direction cosine to an air slope. Exact affinity in \(b\) means that the \(b\)-dependent terms of this joint-cubic recurrence also give the leading quadratic wedge coefficients at a fixed nonzero offset. The offset need not shrink with the wedges.

### Offset-excited formal hardware inversion

Set \(q=0\), retain an unknown fixed \(b\ne0\), and write
\(h_j=\sin(a_j)e^{i\phi_j}\), so \(s_j=h_je^{2\pi iN_jt}\).
The leading self-second and positive mixed-sum coefficients of the complex trace are
\[
S_j=-\frac{\overline b\,k_jh_j^2}{2},\qquad
M_{ij}=-\frac{\overline b\,k_ik_jh_ih_j}{2n_j},
\quad i<j.
\tag{V8}
\]
They follow from the quadratic source correction
\[
-\sum_jk_j\operatorname{Re}(\overline{s_j}b)
 \left(s_j+\frac1{n_j}\sum_{i<j}k_is_i\right).
\]
These identities concern leading homogeneous coefficients. Exact torus coefficients have higher wedge-order corrections, and finite-amplitude measured coefficients must not be substituted without an error analysis.

The ratios
\[
R_{ij}=\frac{M_{ij}^2}{S_iS_j}=\frac{k_ik_j}{n_j^2}
\]
eliminate phase and offset. Put
\[
r=R_{13}/R_{23},\quad u=\sqrt{R_{12}/r},\quad
k_2=\frac{u}{1-u},\quad n_2=\frac1{1-u},\quad k_1=rk_2.
\]
Then \(c=R_{23}/k_2=k_3/(1+k_3)^2\). Its derivative is
\((1-k_3)/(1+k_3)^3>0\) on \(k_3\in[.3,.8]\), so the native branch is uniquely
\[
k_3=\frac{1-2c-\sqrt{1-4c}}{2c},\qquad n_3=1+k_3.
\tag{V9}
\]
The other root is reciprocal and outside the native interval. This inverse is smooth in the interior with nonzero \(b,h_j\).

At leading order DC gives \(b\). With the indices known, (V8) gives wedge magnitudes; fundamentals give phases and the positive gains
\[
K_1=k_1(d+2g+3/n_2+3/n_3),\quad
K_2=k_2(d+g+3/n_3),\quad K_3=k_3d.
\tag{V10}
\]
Thus \(d=K_3/k_3\), \(g=K_2/k_2-d-3/n_3\); \(K_1\) supplies redundancy. The construction is a smooth inverse for formal leading coefficients, not a global finite-wedge optical inverse.

For one prism, the excitation is visible exactly: with
\(D=\tan[\arcsin(n\sin a)-a]\) and \(\chi=(\tan a)D/2\),
\[
Z=(1-\chi)b+\ell D e^{i\gamma}
       -\chi\overline b\,e^{2i\gamma}.
\]
The source offset itself supplies the second harmonic that disappears at the centered gauge.

### The unknown tilt supplies a separate complex channel

At zero tilt, exact affinity gives
\(Z=Z_0+A b+B\overline b\).
Under common rotor rotation the three terms have torus weights \(1,0,2\). Therefore the complex negative-fundamental coefficient at \(-e_j\) vanishes identically at \(q=0\) for every other parameter value. Its leading tilt variation is
\[
C_{-e_j}
=-\frac{k_j}{2n_j}bq\,\overline{h_j}
 +\text{higher wedge orders}.
\tag{V11}
\]
For nonzero \(b,h_j\), the multiplier of complex \(q\) is nonzero and identifies both real beam-angle directions. The sign and factor were independently checked from (V7) and direct intersection geometry. This complex channel must not be confused with the conjugate symmetry of an individual real coordinate. Its finite-sample use requires the following argument.

### Exact ideal-200-sample local-rank theorem

Choose witness frequencies \(N=(1,5,25)/20\) Hz. For all \(|m|_1\le2\), the 25 nodes
\[
z_m=\exp[2\pi i(m_1+5m_2+25m_3)/400]
\tag{V12}
\]
are distinct: label differences have \(\ell_1\) norm at most four, whereas a nonzero integer base-five relation has norm at least six; the labels lie between \(-50\) and \(50\), so no further modulo-400 collision occurs. Add \(kz_m^k\) at the six signed fundamental nodes. The resulting 31 columns have a nonsingular first-31-row confluent Vandermonde minor. Equivalently, an annihilating polynomial of degree at most 30 would have 31 zeros counted with multiplicity. A fixed realified left inverse on the 200 samples therefore exists.

This extractor is a proof device fixed at the witness. It is not an estimator supplied with the unknown speeds.

Fix any interior hardware \((n_1,n_2,n_3,g,d)\), an interior nonzero source offset, \(q=0\), and nonzero complex amplitudes \(A_j\) whose phases lie in the native interior. Set
\[
h_j=\varepsilon A_j/K_j,\qquad \varepsilon>0.
\]
For example \(b=1\), \(A_j=1\), and the interior hardware \(h_*\) of section 7 satisfy the witness conditions for sufficiently small \(\varepsilon\). The eighteen real coordinates are six amplitude components, three speeds, five hardware variables, two source components and two beam components.

Scale the nine amplitude/speed columns by \(\varepsilon^{-1}\), five hardware columns at fixed \(A\) by \(\varepsilon^{-2}\), two source columns by one and two beam columns by \(\varepsilon^{-1}\). Before scaling the beam columns, compensate the zero-wedge drift by
\[
\delta b=-B_{\rm beam}\delta q,\qquad
B_{\rm beam}=6+2g+d+3\sum_j1/n_j.
\]
This is an invertible triangular column change.

After the fixed extractor, the limiting negative-fundamental block has beam rank two by (V11), the DC source block is the identity, and fundamentals/confluent fundamentals supply nine amplitude/speed directions. The quadratic hardware block has rank five: at fixed \(A,b\), its normalized coefficients are
\[
\widehat S_j=-\frac{\overline b\,k_jA_j^2}{2K_j^2},\qquad
\widehat M_{ij}=
-\frac{\overline b\,k_ik_jA_iA_j}{2n_jK_iK_j}.
\]
Their ratios recover the indices by (V9), after which
\[
K_j=|A_j|\sqrt{\frac{|b|k_j}{2|\widehat S_j|}}
\]
and (V10) recover \(d,g\). This smooth left inverse proves rank five. Hardware DC components are removed with the source identity block, whose unscaled quadratic contribution vanishes in the limit. Beam positive-fundamental components can be removed with the fundamental block. Neither crossblock reduces rank.

The transfer to exact finite wedges uses uniform \(C^1\) expansions on a compact strict branch near the witness. At \(q=0\), angular output is odd in wedges, with retained degree one and next degree three; affine source transport is even, with retained degree two and next degree four. Hardware differentiation at fixed \(A\) removes the leading fundamental dependence. All corresponding remainders vanish under the stated column scaling. Differentiated quadratic frequency terms are \(O(\varepsilon^2)\), hence vanish after \(\varepsilon^{-1}\) scaling; repeated quadratic nodes are unnecessary. Compensated beam derivatives begin at order \(\varepsilon\), with a vanishing scaled remainder.

The limiting transformed Jacobian therefore has rank \(2+2+9+5=18\). Continuity proves:

**Physical-vector theorem.** For all sufficiently small positive \(\varepsilon\), the exact coupled-vector map to the 400 real sampled outputs has rank eighteen at this off-axis family, with all eighteen parameters unknown. It is locally injective there. A nonzero analytic minor gives generic local rank on the connected analytic branch containing the witness. No numerical wedge threshold, global uniqueness or usable noise guarantee follows.

### Excitation, finite readout and the limits of stability

Hardware-separating channels scale as \(|b|\varepsilon^2\); the beam channels scale as \(|b|\varepsilon\) times a beam perturbation. These powers alone are not useful numerical conditioning bounds: native units, gain normalization and the finite dictionary's smallest singular value also matter.

Although exact torus quadratic coefficients have \(O(\varepsilon^4)\) corrections, the 31-column finite extractor can mix unrepresented cubic angular terms into quadratic rows. Its safe normalized hardware remainder is therefore only \(O(\varepsilon)\), unless extra cancellation or a richer dictionary is proved. Exact torus selection rules do not automatically exclude temporal aliases. After degree-two terms are accounted for, fundamental extraction has relative \(O(\varepsilon^2)\) higher-order contamination; a fundamental-only fit can absorb quadratic terms and needs its own analysis.

An exact near-centered lower bound follows along the first-prism gauge (V6). The source-transport matrices from (V4) satisfy
\[
A_j=I+\frac{W_ju_j^T}{v_jP_j}=I+O(\varepsilon^2).
\]
On a compact guarded gauge tube, their product has gauge derivative \(O(\varepsilon^2)\), while the centered angular output is exactly constant. Exact source affinity yields
\[
F_{\rm vec}(g_s,b)-F_{\rm vec}(g_0,b)
 =[A(g_s)-A(g_0)]b,
\qquad
\|F_{\rm vec}(g_s,b)-F_{\rm vec}(g_0,b)\|
\le C_G|b|\varepsilon^2|s|.
\tag{V13}
\]
Here \(g_s\) denotes a parameterized gauge path, not the air gap. In a specified observation norm, bounded error consequently creates an indistinguishability scale proportional to \(\eta/(|b|\varepsilon^2)\), capped by the available gauge segment. A numerical minimax bound requires the norm, gauge parameterization and constant \(C_G\). This exact gauge eliminates the centered cubic angular remainder; a generic amplitude-compensated variation would not do so.

The remaining physical publication work is quantitative finite-angle and noise certification, a constructive globally complete inference method, and treatment of weak or vanishing excitation. All-output residual and branch checks remain necessary after any projected local correction.

## 12. Evidence status and the remaining mathematical bottleneck

| Status | Result | Scope |
|---|---|---|
| Derived exactly: canonical | Paired-prism law, affine geometry, halfplane/circuit elimination, ideal-clock polynomial lift | Original independent-axis equations and strict branches retained |
| Proved locally: canonical | Rank eighteen on its small-wedge finite-record family | Does not transfer its harmonic channels to vector optics |
| Independently checked exactly | Rational determinant (20), beam identity (21), cubic invariant identities and scalar reconstruction | Specific algebraic certificates, not recovery statistics |
| Established model limitation | Vector rotational equivariance and Fourier selection rule (V2) | The canonical third-harmonic rank construction does not transfer to physical vector optics |
| Derived exactly: physical vector | Root-free Snell/position graph, four-variable affine geometry, centered first-prism gauge | Declared coupled-vector surfaces and branch/traversal conditions |
| Independently audited: physical vector | Cubic recurrence, offset ratio inverse and negative-fundamental beam coefficient | Leading formal coefficients; finite readout requires remainder control |
| Proved locally: physical vector | Rank eighteen for the off-axis small-wedge ideal-200 family | Unknown offset and tilt retained; no numerical threshold, global uniqueness or noise guarantee |
| Reported; full proof pending integration | Global uniqueness of the five formal canonical cubic invariants | Does not imply full-record or physical vector-model uniqueness |
| Derived conditionally | Nested remainder tubes, interval/dual exclusions and full-cover accuracy bounds | Require validated uniform enclosures and complete coverage |
| Open | Practical complete initial cover and refinement across weak, singular and critical regions | Needed for a broadly usable certified inverse |

Previously completed small checks support specific derivations: twelve symbolic optical/lift identities passed; polygon and LP extrema agreed within \(4.44\times10^{-16}\); a harmonic Jacobian check reached \(2.04\times10^{-8}\) relative error; exact-rational controls confirmed the necessity of dual-defect corrections; a separate interval minor check certified a local chart under its own timestamp and arithmetic assumptions. They are distinct from the exact ideal-clock theorem and from broad recovery evidence.

For either model, the unsolved global bottleneck is a practically manageable **complete initial cover over the full native prior**, including unknown beam coordinates, followed by refinement of every weak or singular surviving branch against the appropriate exact equations. The canonical scalar reconstruction and physical offset-excited inverse supply different structured components. The new physical local theorem respects (V1)-(V2) and resolves the absence of a vector rank witness, but leaves quantitative finite-angle readout, excitation-dependent conditioning, global coverage and exact residual enforcement open. Useful certified bounds and structural proofs take priority over isolated case repairs.

No uniformly accurate inverse over the complete prior is proved. Preserved compatible-pair evidence at the earlier assistant-selected \(\eta=10^{-8}\) benchmark applies to the canonical model; it is not numerical evidence about vector optics. The physical centered gauge independently rules out full-prior unique recovery at that stratum, and (V13) explains poor stability near it. An inference method must return remaining uncertainty when the supplied observations and excitation cannot support the requested native accuracy.

The original Dropbox research and observations remain unchanged. Completed empirical outputs and arithmetic audits are preserved in [EVIDENCE_REPORT.md](EVIDENCE_REPORT.md); no new recovery experiments were run for this integration. The linked supporting notes are workspace companions to this standalone report. [SCOPE_CONFIRMATION.txt](SCOPE_CONFIRMATION.txt) records the revised work boundary.
