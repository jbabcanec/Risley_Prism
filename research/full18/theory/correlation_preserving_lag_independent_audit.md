# Independent audit: correlation-preserving lag certificate

## Verdict and scope

The proposed certificate is mathematically valid as a **conditional sufficient theorem**, with the rank, chart, and remainder qualifications below. It makes a real structural reduction: the degree-two projected optical contribution is a sum of nine conjugate-pair terms with explicitly differentiable finite-harmonic factors, and exact geometry affinity removes every pure geometry Hessian. It does **not** establish the decisive numerical inequality on the existing native box. The projected second-parameter derivatives of the exact higher-order tail remain a genuine, presently unevaluated obligation.

This audit uses mathematical reading and derivation only. No new numerical evaluation, replay, refinement, parameter search, or test was run. The target remains the same κ=0.1 witness, original samples, frozen charts, and native ±0.001 box. The earlier unsuccessful lag-majorant test is neither altered nor contradicted.

Primary source definitions: `quantitative_algebraic_correction.md`, `native_chord_corrected_record_envelope.md`, `algebraic_defect_correction.md`, and the exact paired position transfer in `oblique_inverse.md`. The proposed new theorem was supplied in the parent task; its draft was not yet present when the audit was begun.

## 1. Dimensions, indices, and the frozen center identity

Write ℓ=11 and K=200−7ℓ=123. With axis-major stacking,

    H_r[(axis,k),j] = r_axis,k+jℓ,
    v_r[(axis,k)]   = r_axis,k+7ℓ,
    0≤k≤K−1, 0≤j≤6.

Thus H_r is 246×7, v_r has length 246, R is 7×6, and L0 is 6×246. The largest H index is 188 and the largest next-lag index is 199; no sample outside the original record is introduced. Partition L0=(Lx,Ly), with each block 6×123. Other stacking conventions require the corresponding fixed permutation.

The six-column projected lag matrix and forcing are

    J(θ)=L0 H_s(θ) R,
    u(θ)=L0[v_s(θ)+H_s(θ)cbar],
    s(θ)=y−R1(θ),   cbar=−1/7·1.

Assume L0 is the exact frozen left inverse satisfying J(θc)=I. This can mean the exactly defined corrected inverse `(P H_sc R)^−1 P`, as in the existing chart. It must not mean merely a floating proposal with a small observed defect. If a proposal is used without exact correction, its center defect has to be included in the certificate.

For G(θ)=L0 H_R1(θ) R, linearity of record assembly gives exactly

    J(θ)=I−[G(θ)−G(θc)].

This identity needs no exact recurrence for the corrected records. A bound sup ||G−Gc||∞≤μ<1 proves invertibility of the projected 6×6 matrix and consequently full column rank of H_s R. It does not by itself certify the six non-DC roots, rotor assignments, inverse reconstruction, or the full chord map.

L0, R, and the center must remain frozen when differentiating. Recomputing L0 with θ would create extra derivative terms and change the theorem.

## 2. Exact geometry affinity is a genuine reduction

Let x consist of the fourteen non-geometry native coordinates and let

    u=(b_x,b_y,g,d).

For a given sample, every refracted direction is independent of these four variables. For one prism define

    A_i=I+(v_i−w_i)m_i^T/(1−m_i^T v_i).

The exact position transfer from the existing model becomes

    p_next=A_i p+3 A_i v_i+ℓ_i w_i,

where ℓ_1=ℓ_2=g and ℓ_3=d. Composition from p_in=b+6t gives

    F=A3 A2 A1 b
      +6 A3 A2 A1 t
      +3[A3 A2 A1 v1+A3 A2 v2+A3 v3]
      +g[A3 A2 w1+A3 w2]
      +d w3.

All displayed A_i,v_i,w_i depend on x, not u. Hence F is exactly affine in u, including finite thickness and tilted-surface position effects. Taylor extraction in wedge scale, subtraction of F1 or F2, lag assembly, and multiplication by frozen matrices preserve this affinity. Therefore

    G(x,u)=C(x)+Σ_(a=1)^4 u_a B_a(x)

is exact, and every ∂²G/∂u_a∂u_b vanishes identically. This removes ten symmetric geometry-geometry Hessian blocks, rather than merely placing small bounds on them. The 14 optical-coordinate Hessian blocks and 4×14 mixed blocks remain.

This statement uses the actual native source/gap/distance variables. It need not retain the same zero-block form under an arbitrary nonlinear or mixed-coordinate reparameterization. The unchanged diagonal native scaling is compatible with it.

## 3. Quadratic harmonic support and the correct rank statement

Let e_i=sin(π a_i/180), φ_i=π φ_i^native/180, and let ε_i denote the ith unit multi-index. The homogeneous quadratic wedge term P2=F2−F1 has formal support

    {0} ∪ {±2ε_i : i=1,2,3}
        ∪ {±ε_i±ε_j : 1≤i<j≤3}.

There are 18 nonzero signed multi-indices, grouped into 9 conjugate pairs, and one DC term. The self-product DC contributions are combined into that one term. The full F2 support also includes the six first-degree modes; this explains the familiar count 25 for degree at most two.

These are **formal support counts**. Some coefficients may vanish, and distinct multi-indices may yield the same sampled node for special speeds. The factorization below still holds in those cases, but “18 distinct nonzero observed tones” would require additional hypotheses.

For a real two-axis record use vector Fourier coefficients C_m∈C²:

    P2,k=C_0+2 Re Σ_(m∈M+) C_m ζ_m^k,
    ζ_m=exp[(2π i/20) m·N],
    C_−m=conjugate(C_m).

This conjugacy applies componentwise to the two real axes. It is generally false for a single scalar coefficient sequence obtained by identifying the entire paired record with x+i y. The latter representation requires separate treatment of the signed coefficients.

For each m,

    v(ζ)=(1,ζ,...,ζ^(K−1))^T,
    w(ζ)=(1,ζ^ℓ,...,ζ^(6ℓ))^T,
    U(ζ)=[Lx v(ζ), Ly v(ζ)],
    q(ζ)^T=w(ζ)^T R.

Then the exact quadratic projected contribution is

    G2=2 Re Σ_(m∈M+) [U(ζ_m) C_m] q(ζ_m)^T.

The DC part vanishes because 1^T R=0. Each **complex** outer product has rank at most one. Its real conjugate-pair contribution `2 Re(A q^T)` has real rank at most two, not generally one. The total matrix need not have small rank; it can have full rank six. The useful compression is the nine structured terms and their shared parameter dependence.

The coefficients factor as

    C_m=e_i e_j exp(i m·φ) D_m(h,t,u),

with i=j for a self harmonic. Here h=(h1,h2,h3) and t=(t_x,t_y), so “five shape coordinates” means all three effective indices and both beam slopes. D_m is affine in u and rational in these five shape coordinates on the guarded chart. This rationality can be obtained from the order-two wavevector/slope jet circuit: after common homogeneous wavevector rescaling, its zero-wedge square roots are h_i and 1, and their Taylor coefficients are rational. It is not an assumption that all exact finite-wedge optics are rational in h,t.

## 4. Shape charts and derivative conversion cannot be omitted

In the native index/beam-angle chart,

    t_j=tan(π β_j/180),
    h_i=sqrt[n_i²+(n_i²−1)|t|²].

Thus h and t are convenient coefficient coordinates, not independent native input coordinates. A change in a beam angle changes t and all h_i. If z=(h,t), the second native derivative of a coefficient includes both terms

    ∂pq D = Σ_ab D_za zb · z_a,p z_b,q
            +Σ_a D_za · z_a,pq.

The amplitude sine derivatives and all degree/radian factors also remain. Treating h as fixed while taking a native beam derivative, or omitting the second derivative of the coordinate transformation, gives the wrong native Hessian.

A rectangular enclosure in (h,t) may be used to bound coefficient derivatives, provided it contains the image of the original native box and avoids every denominator zero. It is only an enclosure device; it does not change the target region or supply a new prior box. Coefficient bounds can become looser if native h/t dependence is discarded.

## 5. Explicit projected harmonic derivatives

Use ω=(2π/20)m·N and write U(ω),q(ω) to avoid ambiguity between complex-ζ and real-frequency derivatives. Define D_k=diag(0,...,K−1) and D_ℓ=diag(0,ℓ,...,6ℓ). Then

    U_ω=i[Lx D_k v,Ly D_k v],
    U_ωω=−[Lx D_k² v,Ly D_k² v],
    (q^T)_ω=i(D_ℓ w)^T R,
    (q^T)_ωω=−(D_ℓ² w)^T R.

Therefore ∂N_i U=(2π/20)m_i U_ω and ∂N_iN_j U=(2π/20)²m_i m_j U_ωω, with identical rules for q. The coefficients C_m are independent of N once phases are separately factored as above. There are no hidden 200-sample optical Hessians in these factors.

For T=U C q^T, two arbitrary parameter derivatives satisfy the exact nine-term product rule

    T_ij=U_ij C q^T+U C_ij q^T+U C q_ij^T
         +U_i C_j q^T+U_j C_i q^T
         +U_i C q_j^T+U_j C q_i^T
         +U C_i q_j^T+U C_j q_i^T.

Absent dependencies make many terms zero. For example, two speed derivatives act only on U and q, while material/beam derivatives act on C. Summing these signed complex expressions, then taking their real part, **before** entrywise absolute values is the desired projected calculation. Bounding every U,C,q factor separately by scalar norms is valid but can lose the cancellation that motivates the construction.

This gives a finite algebraic derivative compiler for G2. It makes the quadratic Hessian explicit, but does not automatically provide tight upper bounds on the rational coefficient jets over a whole box.

## 6. The 64-sign linear bound is an exact identity

Let B_i=∂i G(θc) be exact real 6×6 matrices, and let |δ_i|≤r_i in the same coordinates used by these derivatives. Then

    α_linear
      =sup_δ ||Σ_i δ_i B_i||∞
      =max_row a max_(σ∈{−1,+1}^6)
         Σ_i r_i |Σ_col b σ_b (B_i)_ab|.

Indeed the row 1-norm is the maximum of its signed linear functionals. The finite row/sign maxima commute with the supremum over the parameter box, and that box's linear support function is Σ_i r_i|coefficient_i|. No minimax relaxation is involved.

Only 64 column-sign vectors per row are needed, regardless of the number of uncertain parameters. In fact ±σ produce the same value, so 32 representatives suffice if desired. This is an identity for the center-linear term, not for the nonlinear map or an uncertain interval matrix with unspecified dependence.

The identity is no larger than the usual componentwise triangle bound. It can be strictly smaller, but strict or substantial improvement is not automatic. If center derivatives are known only through certified intervals, their uncertainty must still be propagated; binary midpoint derivatives cannot be substituted as exact coefficients. Also do not multiply by a native radius twice if B_i were already taken in scaled coordinates.

## 7. Affine-geometry Taylor certificate

Set ξ=x−xc, η=u−uc, and

    A(x)=C(x)+Σ_a uc_a B_a(x).

There is the exact decomposition

    G(x,u)−Gc
      =DA(xc)ξ+Σ_a B_a(xc)η_a
       +[A(xc+ξ)−A(xc)−DA(xc)ξ]
       +Σ_a η_a[B_a(xc+ξ)−B_a(xc)].

Suppose |ξ_i|≤r_i and |η_a|≤r_a. All suprema below are over the optical-coordinate box, including each center-to-point segment. Entrywise bounds give

    E=1/2 Σ_(i,j optical) r_i r_j sup |∂ij A|
      +Σ_(a geometry,i optical) r_a r_i sup |∂i B_a|.

Then |nonlinear remainder|≤E entrywise, and

    sup_box ||G−Gc||∞ ≤ α_linear+||E||∞ = μ.

Hence μ<1 certifies the lag matrix everywhere on the unchanged box, with ||J(θ)^−1||∞≤1/(1−μ). The factor 1/2 applies to the full ordered optical double sum. The mixed sum has coefficient one. There is no missing geometry-geometry term and no justification for deleting the mixed sum.

Using separate G2 and G3 enclosures is valid; keeping their center-linear derivatives combined before the sign optimization can preserve additional cancellation. Summing separate norm bounds discards that cancellation but remains sufficient.

## 8. The augmented right-hand side and filter sign

Define the projected augmented remainder array

    T_R=[G, E_R],
    E_R=L0[v_R1+H_R1 cbar].

For each quadratic harmonic its right factor is

    [w(ζ)^T R, ζ^(7ℓ)+w(ζ)^T cbar].

This is a 1×7 row. Its DC value is zero in every component, since 1^T R=0 and 1+1^T cbar=0. The analogous augmented corrected array is [J,u], whose increment is the negative of the increment in T_R.

Put a_c=−u_c and c_c=cbar+R a_c. Contracting the harmonic right row with (a_c,1)^T gives exactly

    ζ^(7ℓ)+Σ_(j=0)^6 c_c,j ζ^(jℓ)
      =p_c(ζ^ℓ).

Thus the lag variable of p_c is ζ^ℓ, not ζ. Define the exact filtered remainder

    K(θ)=L0[v_R1(θ)+H_R1(θ)c_c].

From J a=−u one obtains

    J(θ)[a(θ)−a_c]=K(θ)−K(θc),
    a(θ)−a_c=J(θ)^−1[K(θ)−K(θc)].

The sign on the remainder increment is **positive**. This identity holds even when the raw center recurrence residual is nonzero: only its L0 projection needs to vanish, as it does by the definition of a_c. It avoids separately bounding δJ·a_c and δu before their common-filter cancellation.

Consequently, any certified ν≥sup ||K−Kc||∞ gives

    sup ||a−a_c||∞≤ν/(1−μ).

This alone does not certify root disks or coefficient recovery. Furthermore p_c and a_c are frozen in this calculation; differentiating a parameter-dependent fitted filter would be a different expression. Noise, if introduced, contributes separate record-linear terms and cannot be treated as following the eighteen-parameter correlations.

## 9. What is still genuinely unknown: the projected exact tail

The exact decomposition is R1=P2+R2, hence G=G2+G3 with

    G3(θ)=∫_0^1 [(1−λ)²/2]
               L0 H_(∂λ³ F(λe;other θ)) R dλ.

The optical model and every native derivative used here must be smooth on the entire joint parameter/λ domain, with the required transmitted roots and denominators guarded. Under those conditions differentiation under the integral gives

    ∂ij G3=∫_0^1 [(1−λ)²/2]
                L0 H_(∂ij∂λ³ F(λe;other θ)) R dλ.

Every derivative includes the λe dependence and the native coordinate conversions. The integrand is still affine in geometry, so pure geometry Hessians remain zero. But the optical-optical second derivatives and the mixed derivatives of its geometry coefficients do not vanish.

These are order-three λ jets with order-two parameter jets, or equivalent exact projected-tail bounds. They are not supplied by the previously saved first-parameter jets. Small tail values or a small center derivative do not bound a tail Hessian on a finite box. Taking absolute values sample-by-sample before the frozen projection can again lose the shared phase dependence; writing the projected integral is an exact identity, not itself a tight enclosure method.

The integral weight has total mass 1/6, so a uniform bound on the projected derivative integrand can be multiplied by 1/6. This does not rescue an unproved integrand bound. Likewise R2=O(κ³) is an asymptotic statement whose parameter derivatives and multiplication by L0 require explicit scaling: native wedge derivatives can lower wedge powers, and L0 has the familiar inverse-signal scaling. It provides no numerical finite-box margin by itself.

## 10. Final diagnosis

The proposal is more than a relabeling of an arbitrary 18-dimensional Hessian. It analytically eliminates pure geometry curvature, replaces the entire quadratic optical contribution by nine explicit harmonic factors, and gives an exact finite-sign formula for the linear part. The augmented filter also removes an avoidable independent forcing bound.

It does not eliminate all new second-derivative work. In particular, the previously first-jet-only calculation cannot supply the exact-tail Taylor remainder required here. If the new theorem is described as removing the additional Hessian obligation completely, that claim is false. If it is described as identifying a much smaller, correlation-preserving projected-tail obligation after explicitly resolving the dominant quadratic structure, that claim is correct.

No value of μ, ν, or a nonlinear contraction factor is certified by this audit. No native error guarantee, positive noise allowance, full chord self-map, or all-branch inverse claim follows until the new quantitative inequalities and all subsequent domain/reconstruction obligations are actually established.

## 11. Review of the subsequently available draft

The draft `correlation_preserving_lag_certificate.md` became available before this audit concluded and was read in full. Its stated complex-rank-one/real-rank-two distinction, augmented-filter sign, six- and seven-column sign counts, affine position-block recurrence, and degrees 188/199 agree with the derivations above. Its numerical threshold discussion is expressly conditional on a separately verified center bound and does not assert a new passed certificate.

The draft receives a conditional mathematical pass. Recommended clarifications are to display the native h/t conversion and its second-derivative chain rule, and to state explicitly that the center derivative coefficients in the finite-sign formula must be exact or rigorously enclosed. Neither clarification changes the proposed theorem. The exact-tail curvature obligation remains open.

## 12. Supplement: explicit cubic and higher-tail route

The subsequently added Section 9 of `correlation_preserving_lag_certificate.md` has been reviewed mathematically. No numerical evaluation or test was performed. The route is valid, subject to the dependence and derivative qualifications below.

### Taylor coefficient and integral constants

For a projected geometry column Φ_j(λ,x), let K_(p,j)=∂λ^p Φ_j(0,x)/p!. Taylor's formula at λ=0 gives exactly

    G_(3,j)=K_(3,j)
       +(1/6)∫_0^1 (1−λ)^3 ∂λ^4 Φ_j(λ,x) dλ.

The weight has total mass 1/24. A uniform entrywise bound on D_x^α∂λ^4 Φ_j therefore contributes M_(j,α,4)/24, for |α|≤2. This is a fourth-λ, second-optical mixed jet, potentially total derivative order six; it does not require a full order-six tensor over all eighteen native coordinates. Center-geometry columns should be combined before magnitude bounds when possible.

### Cubic support and plane slope

The cubic formal support consists of six multi-indices with |m|₁=1 and 38 with |m|₁=3. The latter count is 6 of type (±3,0,0), 24 of type (±2,±1,0), and 8 of type (±1,±1,±1). Thus there are 44 formal signed modes and 22 conjugate pairs. There is no cubic DC term. Coefficients may vanish or sampled nodes may collide, so this is again a formal support count.

The exact plane slope expansion is

    λe_i v_i / sqrt(1−λ²e_i²)
       =λe_i v_i+(λ³e_i³/2)v_i+O(λ⁵).

The cubic correction is essential. Collect every cubic monomial contributing to the same mode before bounding it. Each complex projected mode remains an outer product of rank at most one; the real conjugate-pair contribution has rank at most two.

The filtered fundamental-mode cancellation is **pointwise**: its value vanishes when the trial lag node is a root of the frozen p_c. It does not generally cancel frequency derivatives. For z=e^(iℓω),

    d/dω p_c(z)=iℓ z p_c′(z),
    d²/dω² p_c(z)=−ℓ²[z p_c′(z)+z² p_c″(z)].

These terms can be nonzero where p_c(z)=0 and must remain in the derivative bounds.

### Finiteness and uniformity

Strict lower bounds for every relevant square-root argument and denominator, together with the tangent-chart cosine guards and |λe_i|<1, make the exact circuit analytic on a neighborhood of the compact native-box/λ domain. The required mixed derivatives are therefore finite. Surface-order guards additionally retain physical legitimacy; compactness without exclusion of poles or branch boundaries would not suffice.

The guard constants alone do not give a sufficiently small curvature bound. They do, however, allow a finite explicit derivative majorant to be propagated through the circuit. If a naive interval for a denominator crosses zero despite a separately verified strict guard, that guard must be incorporated into the denominator enclosure before taking reciprocals.

The normalized jet recurrences in Section 9 are correct. For multi-indices, the square-root convolution must mean

    Σ_(β≤ν, β≠0, β≠ν) b_β b_(ν−β),

not strict componentwise inequalities. For example, both splits of ν=(1,1) into (1,0)+(0,1) must be present. The analogous inverse sum includes β=ν but excludes β=0. Derivative extraction multiplies the λ-four, optical-α normalized coefficient by 4!·α!, with α!=∏_i α_i!.

The index set 0≤λ-order≤4 and total optical order≤2 is downward closed, so the truncated recurrences compute all needed coefficients without higher optical orders. With fourteen optical coordinates it has 5×120 scalar coefficient slots per scalar circuit node. This is a finite constructive route, not an unknown Hessian oracle.

For a **uniform** bound over the whole domain, these derivative coefficient functions must be enclosed for every expansion base point (λ,x) in that domain. Jets computed only at a single numerical center do not provide that uniform bound by themselves.

### Limits of claimed correlation retention

Using common formal jet monomials and postponing absolute values until after projection is not sufficient if each sample's coefficient or base-point dependence has already been replaced by an independent interval. Common monomial labels do not reconstruct that lost dependence.

To retain the intended correlation, keep the derivative coefficients as joint functions of the common base point (λ,x) until the projected sum is assembled and bounded, or use a validated common Taylor-model representation with all remainders propagated. A signed arithmetic circuit is an exact representation, but ordinary node-by-node interval evaluation of that circuit can still lose dependence. It has no automatic sharpness advantage merely because its final output is projected.

Thus the normalized recurrences prove a finite computable majorant exists under the guards. They do not, without a specified joint enclosure scheme, prove that the majorant retains the useful cancellations or is substantially sharper than a raw sample bound.

### Fallback bound and structural verdict

If every real record coordinate of D_x^α∂λ^4 Π_j is bounded by B, then ||H||∞≤7B and

    ||D_x^α∂λ^4 Φ_j||∞
       ≤7||L0||∞||R||∞ B.

The factor 7 is correct. No extra factor of two is needed for the axes: both axes are already included in the row norm of L0. For the frozen filtered forcing, a corresponding simple bound is ||L0||∞(1+||c_c||₁)B; the augmented operator needs its own right-operator norm.

Applying this fallback only to the order-four-and-higher tail preserves the explicit quadratic and cubic calculations, although the resulting certificate can still be too loose. Section 9 therefore closes the gap of specifying a **finite analytic construction** for the remaining curvature bounds. It does not close the gap of proving those bounds fit the available lag margin on the unchanged native box. The distinction between finite, explicitly computable, correlation-preserving, and quantitatively sufficient must be maintained.

### Zero-wedge coefficient denominators: confirmed

The final draft's additional statement about forward quadratic/cubic coefficient denominators is correct in its scaled (h,t) chart. Let ρ=|t|², s_i=λe_i v_i, and c_i=sqrt(1−|s_i|²). The scaled direction recurrence can be written

    H_i=sqrt(h_i²+ρ−|X_in|²),
    I_i=−X_in·s_i+H_i c_i,
    E_i=sqrt(I_i²−h_i²+1),
    X_out=X_in+(I_i−E_i)s_i,
    Z_out=H_i+(E_i−I_i)c_i.

At zero wedge, X_in=t, H_i=I_i=h_i, and c_i=E_i=Z_out=1. The plane-intersection denominator is also one. The normalized formal inverse/square-root recurrences therefore introduce only rational constants and powers of the positive h_i into the unnormalized forward P2/P3 coefficients. The inverse-reconstruction denominators U, B_perp, f1, and h_i−1 are absent. Native coordinate conversion retains its own tangent/cosine and h_i guards.

The draft now includes the shared-base-point caveat, value-only filtered cancellation qualification, and correct multi-index convolution convention reported in this supplement. With those clarifications, the cubic/higher-tail construction passes this mathematical review. Its numerical sufficiency remains unestablished.
