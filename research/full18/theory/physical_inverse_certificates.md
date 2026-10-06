# Data-derived rotor enclosure and an exact 17+1 physical inverse

## Status and scope

This appendix develops a sufficient, conditional certification framework for the exact physical-vector inverse. **These conditions have not been evaluated successfully for an actual observation record.** They do not constitute an established useful finite-noise recovery guarantee.

The [oblique physical construction](physical_oblique_inverse.md) has a rank-seventeen first-order record and one remaining beam-orientation fiber. A quadratic channel breaks that fiber at the audited witness. The next useful result is an exact strong/weak inverse with an explicit scalar Schur-complement margin, preceded by a record-derived rotor enclosure. Neither stage assumes that most physical parameters or rotor frequencies are known.

Every condition below is allowed to fail. A failed condition leaves its original-prior region **unresolved**; it does not authorize discarding that region or replacing it with a favorable fitted neighborhood. Complete global recovery requires coverage of every compatible region, including centered, zero-amplitude, repeated-frequency and near-critical strata.

The formulas and audit qualifications are transcribed from the independently audited source memo supplied for this report. This was a documentation-only integration: no new numerical tests, symbolic calculations, parameter sweeps or recovery experiments were run. Independent mathematical and symbolic audit is not external peer review or a proof-assistant certificate. The document remains a research draft requiring expert review.

Throughout, \(F\in\mathbb R^{400}\) denotes the exact two-coordinate record on \(t_k=k/20\), \(k=0,\ldots,199\). Measurement error is the nonnegative parameter \(\epsilon\), with \(|F_m-y_m|\le\epsilon\). For unequal componentwise allowances use \(\boldsymbol\epsilon=(\epsilon_m)\), so \(|F-y|\le\boldsymbol\epsilon\); the uniform case is \(\boldsymbol\epsilon=\epsilon\mathbf1\). Angular derivatives use radians unless native degree conversion is stated; length parameters retain their native model units. All eighteen parameters, physical branches and original bounds remain part of the inverse problem.

**Independent scales and uniformity.** The dimensionless small-wedge scale is \(\kappa\); \(\tau\) denotes a certified model-remainder bound. Neither is measurement error. Reducing \(\epsilon\) does not remove finite-angle contamination. With the observed record held fixed, feature boxes, retained domains and uncertainty constants generally depend on the selected allowance. Their \(\epsilon\)-dependence is suppressed in later formulas for readability; claims over an error interval require uniform certification throughout that interval. Feature rows are fixed within each certificate. Generic finite counts for exact formal coefficients or exact-data fibers do not bound the generally continuous compatible set at positive error.

The [global inverse completeness theorem](global_inverse_completeness.md) and [explicit six-root compiler](explicit_six_root_compiler.md) establish exact global set completeness in principle for their finite exact input encodings and specified sample times. They retain singular, disconnected and positive-dimensional possibilities. This appendix supplies possible quantitative local certificates; neither it nor the companions supplies a practical complete solver or a useful numerical noise threshold. Measured-time admissibility does not establish admissibility between samples.

The native box has speeds in \([-3.5,3.5]\) Hz; signed wedges and initial phases in \([-18,18]\) degrees; indices in \([1.3,1.8]\); workpiece distance \(d\in[50,200]\); common gap \(g\in[2,15]\); beam angles in \([-25,25]\) degrees; and each source coordinate in \([-5,5]\). Source distance is \(6\), and each prism's reference axial thickness is \(3\).

## 1. Seven-tone enclosure using all 200 observations

On a uniformly guarded prior region, suppose the exact physical record has a certified first-order enclosure

\[
F_h(t_k)=B_h+\sum_{i=1}^{3}
\left[C_{hi}\cos(2\pi\nu_i t_k)+D_{hi}\sin(2\pi\nu_i t_k)\right]+R_{hk},
\qquad |R_{hk}|\leq\tau_{hk},
\]

for \(h\in\{x,y\}\) and \(k=0,\ldots,199\). The unknown \(\nu_i\) are positive frequency magnitudes; their signs and prism assignments are not presumed. The \(C,D\) are general oblique ellipse coefficients. A wedge-degree-one Taylor bound suffices; its safe remainder is \(O(\kappa^2)\), without normal incidence or a known beam.

Set

\[
\delta=\max_{h,k}(\epsilon_{hk}+\tau_{hk}),
\]

including any explicitly bounded clock or evaluator discrepancy in the allowance. Build a stacked real Hankel design \(H_y\in\mathbb R^{386\times7}\). Its rows are

\[
(y_{h,k},y_{h,k+1},\ldots,y_{h,k+6}),
\qquad h\in\{x,y\},\quad k=0,\ldots,192.
\]

Let \(v_y\) contain the corresponding entries \(y_{h,k+7}\). Thus the design and target jointly use all 200 samples. A true seven-tone record obeys

\[
H_0c=-v_0,
\qquad p(z)=z^7+\sum_{j=0}^{6}c_jz^j,
\]

whose roots are \(1\) and \(\exp(\pm2\pi i\nu_i/20)\).

Choose a **verified** left inverse \(D\) of \(H_y\), so \(DH_y=I_7\), and define

\[
\gamma=\|D\|_{\infty\leftarrow\infty},
\qquad \widehat c=-Dv_y.
\]

If

\[
q_H=7\gamma\delta<1,
\]

every true \(H_0\) compatible with this first-order enclosure has full column rank, and its recurrence vector satisfies

\[
\boxed{
\|c-\widehat c\|_\infty
\leq
\Delta_c:=\frac{\gamma\delta(1+\|\widehat c\|_1)}{1-7\gamma\delta}.
}
\tag{1}
\]

**Proof.** Write \(H_0=H_y+E\), \(v_0=v_y+e\), with \(\|E\|_\infty\leq7\delta\), \(\|e\|_\infty\leq\delta\). Multiplication by \(D\) gives

\[
(I+DE)c=\widehat c-De,
\qquad
(I+DE)(c-\widehat c)=-D(e+E\widehat c).
\]

The Neumann bound proves (1) and the full-column-rank claim.

A floating pseudoinverse is only a proposal. If a proposed \(\widetilde D\) has certified defect

\[
\|I-\widetilde D H_y\|\leq\rho<1,
\]

use the corrected mathematical left inverse

\[
D=(\widetilde D H_y)^{-1}\widetilde D,
\qquad
\gamma\leq\frac{\|\widetilde D\|}{1-\rho},
\]

or carry the defect explicitly. **The corrected \(D\) must also define the center \(\widehat c=-Dv_y\).** If a floating approximation to that center is retained, add its certified coefficient error to \(\Delta_c\) before applying the root test. All arithmetic enclosures must round outward.

Stacking the two real axes matters. A circular complex trace may contain only one signed exponential in \(x+iy\), while its real coordinates carry both members of the conjugate pair. This construction therefore does not require a spurious negative-complex-frequency coefficient. Rank can still fail when DC is zero, a rotor amplitude vanishes or nodes collide; such cases require their own charts or remain unresolved.

The constant root \(1\) may be imposed as an additional exact constraint. A differenced six-tone chart removes DC but doubles worst-case noise and attenuates frequencies near zero. It is an optional separate chart, not a reason to disregard rank failure.

## 2. Certified roots, signed speeds and complete labeling

Set \(\widehat p(z)=z^7+\sum_{j=0}^{6}\widehat c_jz^j\). Candidate disjoint disks or contours must satisfy

\[
|\widehat p(z)|>\Delta_c\sum_{j=0}^{6}|z|^j
\]

on their boundaries, with certified root counts summing to seven. Rouché's theorem then encloses every compatible first-order spectrum. Intersect these enclosures with the unit circle, exact root \(1\), conjugate-pair structure and original native speed arc. No compatible rotor node may be discarded.

Arguments of the three conjugate pairs enclose frequency magnitudes. Retain every pairing and label assignment consistent with those intervals and the native box. A certified interval trigonometric design then encloses the ellipse matrices \(E_i=[C_i,D_i]\) in the two real screen coordinates. This coefficient reconstruction must charge frequency-interval uncertainty; using only midpoint frequencies is insufficient.

For notation, let \(q\) be incident transverse direction cosine, \(z_0=\sqrt{1-|q|^2}>0\), \(t=q/z_0\), \(b\) the source offset, and \(B\) the zero-wedge screen baseline. Define

\[
h_i=\frac{\sqrt{n_i^2-|q|^2}}{z_0},\qquad
D_i=d+(3-i)g,\qquad
L_i=D_i+3\sum_{j>i}h_j^{-1},\qquad
W_i=D_i+3\sum_{j>i}h_j^{-3}.
\]

Write \(e_i=\sin a_i\) for the signed wedge sine, and \(R(\phi_i)\) for the two-dimensional phase rotation. For the physical first-order transfer matrix

\[
M_i=(h_i-1)\left[
L_i I+(W_i+L_i/h_i)tt^T-tB^T/h_i
\right],
\]

a certified \(\det M_i>0\) gives, under a fit using the positive frequency magnitude,

\[
\operatorname{sign}N_i=\operatorname{sign}\det E_i.
\]

Indeed,

\[
E_i=e_i M_iR(\phi_i)\operatorname{diag}(1,\operatorname{sign}N_i),
\qquad
\det E_i=e_i^2\det M_i\operatorname{sign}N_i.
\]

An interval determinant for \(E_i\) excluding zero recovers direction independently of wedge sign and phase. If either determinant cannot be signed, retain both directions. A useful explicit physical check is

\[
\det M_i=(h_i-1)^2L_i
\left[L_i+(W_i+L_i/h_i)|t|^2-(t\cdot B)/h_i\right].
\]

The structural shape equations and observed coefficient intervals constrain hardware and the two-component beam slope. The explicit first-order shape inverse and consistency curve must form a **complete outer set**, retaining all admissible assignments and disconnected branches. Proof-witness frequencies are not initialization data.

This seven-tone stage requires fewer resolved modes than full cubic sideband recovery, but it still needs a genuine uniform first-order remainder on every region it claims to cover. If a branch margin or tail bound cannot be certified, use exact optical exclusion/subdivision or return that region unresolved. A tail bound around one good fit does not establish global coverage.

## 3. Exact strong/weak formulation

Suppose a data-derived branch admits a physical chart

\[
\theta=\Theta(z,\psi),\qquad z\in\mathbb R^{17},
\]

where \(\psi\) is the remaining beam-orientation parameter. The chart must be one-to-one, preserve native units and bounds, and stay on a strict optical branch. Coordinate singularities, including vanishing beam magnitude, require other charts.

Choose seventeen fixed real finite-record features and one scalar quadratic feature:

\[
f(z,\psi)=P_fF(\Theta(z,\psi)),
\qquad
g(z,\psi)=P_gF(\Theta(z,\psi)).
\]

The rows are obtained from the data-derived rotor chart and then held fixed. The raw first-order coefficient and frequency observations may be compressed to seventeen independent directions by a verified selection. Include frequency-tangent channels: frequencies are not treated as known or fixed.

Every evaluation of \(f,g\) uses the **exact physical forward map**. Expansions construct the chart and propose preconditioners; they do not replace the exact equations in this theorem.

For the observed record define

\[
a_0=P_fy,\quad c_0=P_gy,
\qquad r_a=|P_f|\boldsymbol\epsilon,\quad r_c=|P_g|\boldsymbol\epsilon,
\]

\[
\mathcal A=a_0+[-r_a,r_a],
\qquad \mathcal C=[c_0-r_c,c_0+r_c].
\]

The rectangular feature box is an outer approximation to the correlated feature-noise image. Keeping those correlations can sharpen the result but is not needed for soundness. Its dependence on the error allowance propagates to the retained domains \(I,\mathcal Z\), the margin \(\mu\), and the uniform factors \(\lambda_j,\overline U,\overline v\) below. Values certified for one allowance cannot be reused for a larger family without proving the corresponding uniform hypotheses.

## 4. The uniform strong inverse requirement

On a scalar interval \(I\) and strong-variable domain \(\mathcal Z\), require that for **every** \((\psi,a)\in I\times\mathcal A\), the exact equations

\[
f(z,\psi)=a
\]

have exactly one solution \(z=\zeta(\psi,a)\in\mathcal Z\), remaining in the physical chart. This is an existence-and-uniqueness condition over a whole family, not just a nonsingular fitted Jacobian.

A sufficient implementation is uniform contraction. With fixed preconditioner \(C_f\), certify

\[
\sup_{\mathcal Z\times I}\|I-C_fJ_{f,z}\|\leq q_f<1
\]

and the self-map condition

\[
z-C_f[f(z,\psi)-a]\in\mathcal Z
\quad
\text{for all }z\in\mathcal Z,\ \psi\in I,\ a\in\mathcal A.
\]

For compact convex \(\mathcal Z\), Banach's theorem supplies existence and uniqueness. An interval-Newton/Krawczyk theorem with the same uniform parameter coverage, or a structural global strong inverse, may replace this sufficient test. Strong-variable scaling is allowed, but the map back to native coordinates must be retained.

Solving at one \(\psi\), or separately at interval endpoints, is insufficient. If one strong chart cannot cover the interval, subdivide the **data-derived** branch and retain a verified cover. An arbitrarily tiny truth-containing chart is not an assumption.

## 5. Exact scalar Schur complement and bracket

Define

\[
G(\psi,a)=g(\zeta(\psi,a),\psi),
\quad J=J_{f,z},
\quad w=g_zJ^{-1},
\quad s=g_\psi-g_zJ^{-1}f_\psi.
\]

Implicit differentiation gives

\[
D_\psi\zeta=-J^{-1}f_\psi,
\qquad D_a\zeta=J^{-1},
\qquad \partial_\psi G=s,
\qquad D_aG=w.
\tag{2}
\]

Require constant sign and \(|s|\geq\mu>0\) throughout the strong solution family. For positive sign, impose the uniform endpoint bracket

\[
\sup_{a\in\mathcal A}G(\psi_{\rm left},a)\leq\inf\mathcal C,
\qquad
\inf_{a\in\mathcal A}G(\psi_{\rm right},a)\geq\sup\mathcal C.
\tag{3}
\]

For negative sign, the explicit conditions are
\[
\inf_{a\in\mathcal A}G(\psi_{\rm left},a)\ge\sup\mathcal C,
\qquad
\sup_{a\in\mathcal A}G(\psi_{\rm right},a)\le\inf\mathcal C.
\]
Equations (2)-(3), with this negative-sign alternative when appropriate, ensure exactly one solution of \(G(\psi,a)=c\) for every \((a,c)\in\mathcal A\times\mathcal C\). Strict endpoint inequalities provide an interior-root margin.

This reduces the exact eighteen-feature inverse to a scalar monotone root while solving all eighteen original parameters jointly. Bisection or interval Newton may enclose the roots. If only some targets are bracketed, one can still enclose or exclude intervals through \(G(I,\mathcal A)-\mathcal C\), but cannot claim existence for every target. The uniform bracket is needed for the finite perturbation result below.

## 6. Finite-noise propagation in native coordinates

For two feature targets \((a,c),(a',c')\in\mathcal A\times\mathcal C\), connect them by a straight segment. Convexity and uniform bracketing keep its entire solution path in the certified family. With \(\lambda_j=\sup|w_j|\),

\[
|\Delta\psi|\leq
\frac{|\Delta c|+\sum_j\lambda_j|\Delta a_j|}{\mu}.
\tag{4}
\]

The exact physical-coordinate differential is

\[
d\theta=U\,da+v\,d\psi,
\qquad U=\Theta_zJ^{-1},
\qquad v=\Theta_\psi-\Theta_zJ^{-1}f_\psi.
\tag{5}
\]

Here \(v\) is the **actual native tangent** to the constant-strong-feature fiber. Set \(\overline U_{ij}=\sup|U_{ij}|\) and \(\overline v_i=\sup|v_i|\) over the certified family. Then

\[
\boxed{
|\Delta\theta_i|
\leq\sum_j\overline U_{ij}|\Delta a_j|
+\frac{\overline v_i}{\mu}
\left[|\Delta c|+\sum_j\lambda_j|\Delta a_j|\right].
}
\tag{6}
\]

For the compatible feature-tube diameter, use \(|\Delta a_j|\leq2r_{a,j}\) and \(|\Delta c|\leq2r_c\). For error relative to the exact center-feature solution, use \(r_a,r_c\), provided that solution exists under the preceding certificate. A center-feature solution need not be compatible with the full original record.

The pointwise reparameterization-invariant weak-conditioning quantity is \(|s|/\|v\|\) in a specified native norm, not \(|s|\) alone. Reparameterizing \(\psi\) scales \(s\) and \(v\) by the same local factor. The uniform bound \(\mu/\sup\|v\|\) is valid, but need not remain numerically identical under nonlinear reparameterization because its extrema may occur at different points.

A useful hardware-precision claim must report \(\overline v_i/\mu\) and \(\overline U\), with angle and length units. The row \(w\) also matters: noise in strong features changes scalar closure. Omitting \(w\) would incorrectly treat jointly unknown nuisance parameters as calibrated.

## 7. The 5/12/1 hierarchy and matching lower-bound powers

At the audited oblique family, a native-conditioned adapted chart has five strong directions, twelve directions of size \(\kappa\), and one final direction of size \(\kappa^2\). A bounded inverse for the normalized first-order shape map is needed; a rational nonzero determinant alone supplies no useful constant.

The exact Schur complement can be sought in the form

\[
s=\kappa^2s_2+O(\kappa^3),
\]

with nonzero quadratic derivative \(s_2\). Uniform bounds on \(J^{-1}\), chart derivatives and the remainder are required on the **actual retained branch**. Wedge exponents alone do not certify them. Subject to these bounds, twelve adapted modes have errors of order \(\epsilon/\kappa\), and one has error of order \(\epsilon/\kappa^2\). These involve independent measurement and geometric scales: shrinking \(\kappa\) at fixed positive \(\epsilon\) worsens the bounds and can invalidate the certificate hypotheses. Native coordinates mix these modes through \(v\); no individual named physical parameter is assigned the weak rate, nor are native wedges or offsets individually guaranteed the strong rate.

A matching deterministic lower bound requires a curve holding all seventeen first-order observables fixed, together with certified endpoint separation. One choice is a monotone native coordinate \(\theta_j(s)=s\). Two endpoints separated by \(\Delta s\) then have native maximum-coordinate separation at least \(|\Delta s|\). Certify, over the entire curve segment,

\[
\|DF(\theta(s))\theta'(s)\|_\infty\leq C\kappa^2
\]

over **all 400 original real observations**, using the full optical derivative or a differentiated Taylor remainder along the leading-order fiber. A bound on only the scalar feature or selected projections is insufficient.

For a certified available interval of length \(r\), the two-endpoint midpoint-record argument gives minimax native error at least

\[
\frac12\min\left(r,\frac{2\epsilon}{C\kappa^2}\right).
\]

Here \(\epsilon\) is a common componentwise error allowance for this lower-bound statement. Alternatively, use a certified secant bound \(\|\theta(s)-\theta(s')\|_\infty\geq c_{\rm sec}|s-s'|\) and retain \(c_{\rm sec}\) in the lower bound. Unit tangent norm alone is insufficient: arc length need not lower-bound endpoint separation. Chart conditioning enters \(C\) and any secant factor. Upper and lower bounds match **powers**, not automatically numerical constants.

## 8. Exact-model correction and every residual

For noiseless data, a successful strong inverse and scalar bracket give at most one solution of the selected eighteen features on that branch. Verify all **400 scalar observations**, physical branch and surface-order constraints, and original bounds before declaring it an inverse of the original record.

For noisy data, the result is a native parameter tube, not a unique system. Intersect it with all 400 observation strips. Exact affine geometry elimination remains available to tighten the tube. A feature-space overapproximation may include systems that fail the original record; these must not be reported as compatible.

For a practical global claim, every original-prior region must be excluded, covered by a certified branch, or returned unresolved. Multiple surviving branches can contain distant hardware even if each is individually well conditioned. The scalar theorem supplies an exact-model correction mechanism conditional on its premises; it does not itself replace global coverage. Exact ambiguity-complete global coverage is established in principle by the companion theorem and explicit compiler under their finite exact encoding and sampling assumptions; making that construction computationally useful remains open.

## 9. Minimal decisive certification steps

Without parameter sweeps, a future decisive validation would establish:

1. A certified first-order remainder on a named prior region, the observed stacked-Hankel condition \(q_H<1\), and seven certified root enclosures.
2. Full ellipse coefficient intervals, direction signs where certified, and a complete first-order shape/fiber enclosure.
3. A uniform strong inverse over an entire retained scalar interval, rather than at one point.
4. A nonzero native-normalized Schur margin, endpoint bracket, and explicit coordinate uncertainty factors from (6).
5. All 400 exact residual checks, with every other prior branch covered, excluded, or openly unresolved.

These are remaining requirements, **not completed tests or authorization for a new campaign**. Failure identifies insufficient harmonic signal, loose remainders, weak excitation, scalar turning points or unresolved global ambiguity. No condition can be repaired by assuming the true frequency or a favorable hardware neighborhood in advance.

## 10. Normalized quadratic observables and the audited witness

The oblique construction uses

\[
\mathcal I_3=C_{+2N_3}/U_{+N_3}^2
\]

along a first-order fiber. Division by a small fundamental must be included in the noise budget.

One option is a fixed nonzero reference \(U_0\) from the certified data-derived branch and scalar feature \(\operatorname{Re}(C_{+2N_3}/U_0^2)\), or another fixed real quadrature. This remains a linear record functional with an exact noise row norm. Include the fundamental \(U\) in the strong features. Along their exact constant-feature fiber, \(U\) stays fixed and the invariant derivative transfers. Variations in measured \(U\) are handled by the strong-feature term \(w\) in (4).

If noisy ratios are used directly and

\[
|U-\widehat U|\leq r_U<|\widehat U|,
\qquad |C-\widehat C|\leq r_C,
\]

a valid finite complex ratio bound is

\[
\left|\frac{C}{U^2}-\frac{\widehat C}{\widehat U^2}\right|
\leq
\frac{r_C}{(|\widehat U|-r_U)^2}
+\frac{|\widehat C|r_U(2|\widehat U|+r_U)}
{(|\widehat U|-r_U)^2|\widehat U|^2}.
\]

The source proof reports a limiting beam-orientation fiber tangent with native infinity norm approximately \(981.79\) at its rational witness. The raw self-second-harmonic directional gain after native normalization is approximately \(1.88\kappa^2\) per native infinity unit. These are witness-specific directional values, not smallest-singular-value bounds or uniform robustness constants. They show why the tangent factor in (5)–(6) cannot be omitted. The native norm mixes degrees, refractive-index units and native lengths, consistent with the requested coordinate convention rather than a dimensionless physical metric.

The physical matrix audit also establishes a positive orientation bound over the stated geometric/index/source box: its transverse eigenfactor \(L_i\) is at least \(50\), while the parallel factor is at least

\[
50-\frac{125}{287}>49.5.
\]

Under that audited convention, \(\det M_i>0\) is structural. Signed rotation still requires the observed ellipse's interval determinant to exclude zero.

## 11. What is established, conditional and open

| Status | Statement |
|---|---|
| Derived and independently audited | Hankel perturbation enclosure, root-count sufficiency, exact Schur identities, finite native uncertainty bound, normalized-ratio bound and the qualifications above. |
| Conditional certificate | Complete data-derived branch enclosure; uniform remainders and arithmetic bounds; strong inverse; Schur sign and bracket; native uncertainty factors; full-record residual compatibility. |
| Established in principle in companion notes | Ambiguity-complete exact global inverse for finite exact encodings and specified sample times, with an explicit branch-preserving compiler; no practical runtime claim. |
| Not established for an actual record | Successful useful finite-noise recovery using this full certificate chain. |
| Open global/publication requirements | Practical complete coverage of all original-prior branches, refinement near weak or singular strata, useful quantitative constants, and physical experimental validation. |

The oblique local-rank theorem supplies the mathematical direction for these certificates. It does not by itself discharge their data-dependent premises. Universal global uniqueness is false on the original prior because of the centered gauge; exact set completeness retains that ambiguity rather than removing it.

The formerly missing related witness sources are available through the [proof-check README](proof_checks/README.md) and [manifest](proof_checks/manifest.json). All seven saved files have verified SHA256 values matching the supplied hashes. They cover the named oblique physical-vector witness identities and audit counterparts, plus one separately scoped canonical identity; they do not certify this entire data-dependent chain for an actual record. Packaging verified byte identity. No script was executed, imported, compiled or mathematically retested during this documentation update. Their historical `epsilon` wedge notation remains in the unchanged sources; the documentation distinguishes wedge scale \(\kappa\) from measurement error \(\epsilon\).
