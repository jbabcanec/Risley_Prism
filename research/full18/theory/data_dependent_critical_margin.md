# Data-dependent critical-normal certificates

## Scope

All eighteen original parameters remain unknown, the 200 observations and original priors are unchanged, and epsilon is a symbolic bound on each screen coordinate. No record provenance is assumed. The unit-direction notation and weak physical closure W are those of global_boundary_continuation.md. In particular H,P,Z have its explicit positive uniform bounds, while the outgoing normal R=Z-u dot X_out can vanish on W.

This note proves a direct, data-only guard for the last prism and a finite low-degree upstream certificate. It does not claim that either guard succeeds for every record. Critical weak records provably prevent such a universal guarantee.

## 1. Critical refraction has a directly observable final-plane consequence

At any prism put v=X_out/Z, let p_exit be its exit point, let p_next be the next flat-plane point, and write ell=g,g,d as appropriate. Define

B=ell-u dot p_exit,
kappa=R/Z=1-u dot v.

Weak sequential traversal gives

0<=B<=3+ell.                                         (1)

Indeed, the internal axial travel is h=3+u dot p_exit>=0, and B=3+ell-h. The outgoing intersection equation gives the exact identity

ell-u dot p_next=kappa B.                            (2)

At R=0, (2) becomes u dot p_next=ell. This is the rank-collapse hyperplane of the source transport; it is valid without inverting that singular transport.

For the final prism p_next=F_k, ell=d, and 50<=d<=200. Put

u_*=tan(pi/10),
A_(k,l)(epsilon)=|y_(k,l)|+epsilon,
M_k(epsilon)=u_* sqrt(A_(k,x)^2+A_(k,y)^2).

Every compatible weak or strict system satisfies |F_(k,l)|<=A_(k,l) and |u_3(k)|<=u_*. Consequently

kappa_3(k)>=max(0,[50-M_k(epsilon)]/53).              (3)

Proof. From (1)-(2), kappa>=(d-u dot F_k)/(3+d). The right side is at least (d-M_k)/(3+d). Since M_k>=0, this last expression increases with d, so its minimum over [50,200] occurs at d=50. Also kappa>=0 on W. This proves (3).

Thus a small-enough observed screen radius excludes the entire final-prism critical boundary, with no fitting, spectral remainder, initial CAD, or prior knowledge of any parameter. At epsilon=0, the sufficient radius condition is |y_k|_2<50/u_* for each sample. The number 50 and the divisor 53 come from the actual distance lower bound and three-unit prism thickness, not an adjustable localization assumption.

Let z_3*>0 be the globally certified unit axial lower bound from global_boundary_continuation.md. If

m_3(epsilon)=min_k [50-M_k(epsilon)]/53>0,

then every compatible system satisfies

R_3(k)>=z_3* m_3(epsilon)>0 for all k.                (4)

Equation (4) is a critical-root margin. Its numerical usefulness can be limited by a conservative z_3*, but (3) directly gives a potentially stronger bound on the source-transport denominator R_3/Z_3.

The original phase/speed bounds sharpen the first two sample tests. The instantaneous rotor angle belongs to [-alpha_k,alpha_k] with alpha_k=pi/10+7pi k/20, while signed wedge supplies the opposite direction as well. For k=0,1 use alpha_0=pi/10 and alpha_1=9pi/20. Writing A=A_(k,x), B=A_(k,y), replace M_k by the smaller exact support bound

M_k^sharp=u_* sqrt(A^2+B^2) if B<=A tan(alpha_k),
M_k^sharp=u_*[A cos(alpha_k)+B sin(alpha_k)] otherwise.

This maximizes A|u_x|+B|u_y| over the allowed signed rotor sector, so it also includes the entire measurement-error square. For k>=2, the signed angular sectors already cover every direction and the disk support M_k is exact for this instantaneous relaxation. The same proofs apply with M_k^sharp. Formula (7) below retains the simpler disk bound; the sharpened first-two-sample epsilon tests instead use the displayed piecewise algebraic support expression.

## 2. Exact symbolic threshold and a polynomial product certificate

Fix a proposed ratio margin mu with

0<=mu<50/53, C_mu=50-53mu>0.                        (5)

The sign condition C_mu>0 is essential; it cannot be discarded before squaring. At sample k, the explicitly algebraic inequality

C_mu^2>u_*^2[(|y_(k,x)|+epsilon)^2
             +(|y_(k,y)|+epsilon)^2]                (6)

excludes every compatible system with R_3(k)<=mu Z_3(k).

For a=|y_(k,x)|, b=|y_(k,y)| and S=C_mu/u_*, if a^2+b^2<S^2, the exact admissible interval for this strict exclusion is

0<=epsilon<epsilon_(k,mu),
epsilon_(k,mu)=[sqrt(2S^2-(a-b)^2)-(a+b)]/2.          (7)

The endpoint in (7) is positive under the stated condition. Taking the minimum over all 200 samples gives a uniform ratio guard. If a^2+b^2>=S^2 at any sample, this particular all-sample certificate has no nonnegative epsilon interval. Equality in (6) does not exclude the proposed closed near-critical set.

There is a short polynomial certificate behind (6). Under R<=mu Z, identity (2) gives

Z(u dot F-C_mu)
 =(1-mu)Z(d-50)+mu Z(3+d-B)+(mu Z-R)B>=0.            (8)

Every factor on the right has its required sign from (1), the original distance bound, Z>0, and (5). Hence u dot F>=C_mu>0. On the other hand, with A_l=|y_l|+epsilon,

u_*^2(A_x^2+A_y^2)-C_mu^2
 =u_*^2 sum_l (A_l-F_l)(A_l+F_l)
  +(u_*^2-|u|^2)|F|^2
  +(u_x F_y-u_y F_x)^2
  +(u dot F-C_mu)(u dot F+C_mu).                     (9)

Every summand is nonnegative, contradicting (6). Equations (8)-(9) are directly checkable low-degree polynomial/product identities. They explain the certificate rather than merely appeal to general branch-and-bound or an unspecified positivity solver.

The same guard bounds the final source-transport determinant. The one-prism formula det A=H R/(Z P) gives

det A_3 >= [h_*/(h_*+u_*)] m_3,
h_*=sqrt((13/10)^2-1),                              (10)

because H/P=H/(H-u dot X_in)>=h_*/(h_*+u_*). This permits well-defined bounded backward transport through prism 3 throughout the compatible set. No claim about the determinants of prisms 1 and 2 follows from (10) alone.

## 3. Sixteen explicit upstream exposure inequalities

For stage j=1 or 2, suppose downstream ratios have already been certified positive throughout a set D of optical points containing every compatible system. For a fixed sample, the downstream affine map is

F=T_down p_j+c_down(w), w=(g,d),                     (11)

where p_j is the next-flat point immediately after prism j, T_down is the product of downstream source-transport matrices, and c_down is affine in w. Source offset b is absent. All downstream matrices are invertible under the certified guards.

Set

l=u_j^T T_down^(-1),
xi_j(w;y_k)=ell_j(w)+l c_down(w)-l y_k.

If F_k=y_k+e, with e in [-epsilon,epsilon]^2, then (2) gives

kappa_j B_j=xi_j(w;y_k)-l e.                        (12)

Therefore

kappa_j>=min_(w in vertices([2,15]x[50,200]))
          [xi_j(w;y_k)-epsilon ||l||_1]/[3+ell_j(w)]. (13)

The minimum is exactly at the four distance-box vertices: a linear-fractional function with a strictly positive affine denominator is a denominator-weighted average of its vertex values. Formula (13) discards none of the original offsets; their contribution has canceled from the critical-plane row itself.

For a target mu>=0, avoid absolute values by checking the following sixteen inequalities for each sample:

ell_j(w_v)-u_j dot p_j(x,w_v,y_k+epsilon sigma)
   >mu[3+ell_j(w_v)],                               (14)

where w_v runs through the four distance-box vertices and sigma through {(-1,-1),(-1,1),(1,-1),(1,1)}. Here p_j is obtained by backward transport through downstream prisms only. If (14) holds for every optical x in D, then every compatible system has kappa_j>mu. The proof follows by affine interpolation in w,e and (1)-(2). Weak >= versions with a specified positive mu also give a usable nonzero guard.

This certificate is independent of the current prism's inverse transport: it remains meaningful when R_j=0. For j=2 it needs only the final-prism guard, and for j=1 it needs guards for prisms 2 and 3. This permits a backward certification order 3,2,1. Failure to certify (14) does not establish the existence of a near-critical system.

### Low-degree form and exact verification

Using independent local unit-ray graph variables X_in,X_out,H,Z,R,u, the exact reverse spatial step is

p_prev=[K p_next-ell V-3X_in R]/(H R),
K=H R I-(Z X_in-H X_out)u^T,
V=H X_out-X_in(u dot X_out).                         (15)

All denominators are positive under the downstream guards. K,V have degree at most three in these local graph variables. Starting from y_k+epsilon sigma, one reverse step has numerator degree at most four and denominator degree two. Two steps have numerator degree at most seven and denominator degree four. Thus after positive denominator clearing, (14) has degree at most five for j=2 and at most eight for j=1. The degrees refer to the local polynomial ray graph, not to dense elimination into the eighteen native-chart variables. Physical graph equalities and guards must accompany these variables.

One can verify (14) on an explicitly specified optical superset D by supplying polynomial identities of the form

P-delta = sum_S sigma_S product_(i in S) G_i + sum_a h_a E_a,

where P is a cleared numerator in (14), delta>0, E_a=0 are exact ray-graph equalities, G_i>=0 are the stated prior/guard inequalities, and every sigma_S is an explicitly displayed sum of polynomial squares. Direct coefficient comparison and signs prove the certificate; no initial CAD is needed to check it. An instantaneous optical superset can omit cross-time rotor consistency safely, provided it still contains every compatible trace. This may make the certificate too conservative.

No bounded search degree or guaranteed easy construction of such identities is asserted. The concrete reduction is to sixteen known degree<=8 inequalities per upstream stage/sample, with positive denominators and no offset variables. The remaining bottleneck is proving their universal sign over the chosen optical domain. A failed low-degree search is inconclusive.

The exact positive-circuit compiler in reduced_algebraic_inverse_addendum.md can tighten the distance domain or exclude optical strata on which this exposure test fails. It must still use all original mixed-strict flags, shared noise errors, and cross-sample optics. Its pointwise determinant identities alone do not certify their signs uniformly over D.

## 4. Retention certificates and an unavoidable obstruction

Suppose an algebraically encoded weak system theta_0 in W is verified against the original full prior and all 200 samples, has R_j(k)=0, and has residual

r_0=||Fbar(theta_0)-y||_infinity.

For every epsilon>r_0 and every desired positive normal threshold rho, the hierarchical wedge strictification in global_boundary_continuation.md supplies actual strict physical systems with residual<epsilon and R_j(k)<rho. Its univariate sign isolation makes the witness conversion effective. Thus this weak point is a rigorous retention certificate: the record has no uniform positive critical margin at that noise budget. At epsilon=r_0, physical feasibility at the endpoint requires a separate check.

Such weak critical systems occur within the original prior; explicit audited examples are in full_prior_spectral_cover.md and critical_filtered_remainder.md. Taking y to be their exact weak record proves that no procedure can guarantee a positive critical margin for every record and every positive epsilon. This is an information-theoretic obstruction, not a failure of the proposed certificates. It does not assume that an arbitrary supplied record has that provenance.

For a specified record the sound outcomes are therefore:

- Exclude near-critical optics using (6), or verified upstream inequalities (14), obtaining explicit margins.
- Retain them with a verified full-record weak critical witness and the strictification argument. A direct strict physical witness satisfying a specified near-critical threshold retains that particular target set; one such witness alone does not rule out every positive uniform margin. Only a weak R=0 witness with the stated strictification/noise slack, or another verified sequence with R tending to zero, establishes that stronger conclusion.
- Leave them unresolved if neither certificate is available.

The minimal unresolved test is whether the full-record noise tube meets the compact critical boundary image. Positive circuits remove affine geometry, and (14) supplies special low-degree separating inequalities, but the remaining universal/existential optical test can still be hard. No assumption about a small spectral tail, a successful local fit, or record provenance removes that burden.

## Proof-check status

A targeted symbolic check verified identities (8), (9), and (15) as formal polynomial/rational identities. An independent mathematical audit confirmed the ratio guard, necessary sign before squaring, exact disk-based epsilon threshold, four-vertex/sixteen-corner upstream argument, and the conservative local-graph degree bounds. The weak-boundary retention step uses the separately audited strictification theorem. No numerical campaign or record-specific success is claimed.
