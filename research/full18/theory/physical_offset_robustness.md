# Physical-vector off-axis robustness: quadratic index recovery and finite-record certification

This extends the canonical-only memo to the newly audited physical-vector construction. All assertions refer to the stated physical propagation convention, a strict transmitted branch, zero incident transverse direction q at the reference, and fixed nonzero transverse offset b=p_x+i p_y. The finite-record construction is local around a labeled, separated rotor chart. It is not a global full-prior inverse.

## 1. Structural coefficients and exact inverse

Write k_i=n_i-1, h_i=sin(alpha_i) exp(i phi_i), a_i=K_i h_i, with

K1=k1(d+2g+3/n2+3/n3), K2=k2(d+g+3/n3), K3=k3 d.

At a_i=epsilon A_i, the leading complex second-harmonic and mixed-sum coefficients are

C_2ei = -conj(b) k_i h_i^2/2,
C_ei+ej = -conj(b) k_i k_j h_i h_j/(2 n_j), i<j.

These are leading quadratic coefficients, not exact finite-wedge Fourier coefficients. Define their leading coefficient ratios

Rij=Mij^2/(Si Sj)=k_i k_j/n_j^2.

The phase cancels and these ideal ratios are positive real. Using magnitudes gives the same positive ratio; imaginary-part consistency is an additional model check, not an extra degree of freedom.

Let r=R13/R23=k1/k2, u=sqrt(R12/r)=k2/n2. Then

n2=1/(1-u), k1=r k2,
c=R23/k2=k3/(1+k3)^2.

The last map is strictly increasing for 0<k3<1, including the full index interval k3 in [0.3,0.8]. Its inverse is unique there; a stable expression is k3=2c/[1-2c+sqrt(1-4c)]. All recovered indices must pass original bounds and every coefficient-domain filter.

Recover gains from the raw leading coefficients and fundamental amplitudes:

K_i=|a_i| sqrt(|b| k_i/[2|C_2ei|]),
L_i=K_i/k_i,
d=L3, g=L2-L3-3/n3.

L1 supplies an independent consistency equation. Phase and sign branches remain constrained by the original wedge/phase prior; magnitude formulas do not authorize dropping them.

## 2. Audited index differential bounds

Let ell_ij=d log Rij. Direct differentiation gives

D = d log u = (ell12-ell13+ell23)/2,
dn2=n2 k2 D,

dn1=k1[(1-n2/2)ell13+(-1+n2/2)ell23+(n2/2)ell12],

dn3=[k3 n3/(2-n3)] [(1-n2/2)ell23+(n2/2)ell13-(n2/2)ell12].

All three formulas were independently derived from the explicit inverse above. In particular, if every |ell_ij|<=delta_R, the native index box implies

|dn1| <= 1.08 delta_R,
|dn2| <= 2.16 delta_R,
|dn3| <= 13.68 delta_R.

These are conservative differential bounds. The last factor exposes real conditioning loss as n3 approaches 2. It must not be hidden inside an undifferentiated generic condition number.

Useful relative-index bounds are

|d log k1| <= 1.35 delta_R,
|d log k2| <= 2.70 delta_R,
|d log k3| <= 17.10 delta_R.

If coefficient errors obey |Chat-C|<=r_C |C| with r_C<1, then

|log(|Chat|/|C|)| <= -log(1-r_C).

Hence each finite ratio log error is bounded by

|Delta log Rij| <= 2[-log(1-r_Mij)] + [-log(1-r_Si)] + [-log(1-r_Sj)].

For a common r this is 4[-log(1-r)]. This denominator-aware bound correctly diverges when a coefficient uncertainty disk reaches zero. Dividing tiny coefficients without this check is not robust inversion.

The differential bounds become finite endpoint bounds by integrating along a certified inverse path that remains in the stated index box. Do not infer a global endpoint Lipschitz result merely from bounds on the inverse derivative. Alternatively, and more safely for hard-noise data, propagate positive intervals through r,u,n2,k1,c,k3 and intersect all original bounds. This monotone triangular interval inverse is a finite-noise result and automatically handles invalid ratios.

## 3. Distance and gap uncertainty

Let E_i=|d log|a_i|| + (1/2)|d log|b|| + (1/2)|d log|C_2ei||. Then

|d log Li| <= E_i + (1/2)|d log k_i|.

In particular,

|dd| <= d(E3+8.55 delta_R),
|dg| <= L2(E2+1.35 delta_R) + d(E3+8.55 delta_R) + (3/n3^2)|dn3|.

Keeping these local expressions is preferable to large prior-wide constants. If E_i<=E, bounds d<=200 and L2<=200+15+3/1.3 give the conservative transparent estimates

|dd| <= 200 E + 1710 delta_R,
|dg| <= 417.308 E + 2016.04 delta_R.

The last coefficient uses the sharper direct bound (3/n3^2)|dn3| <= (38/3) delta_R. Rounding has been upward. Native lengths retain their original units; no millimetre assumption is made.

## 4. All eighteen coordinates: physical sensitivity hierarchy

Use adapted coordinates z=(three real leading amplitudes, two source baselines, three speeds, three phases, two beam coordinates, five hardware coordinates). At q=0, compensate beam perturbations by delta p=-B0 delta q with B0=6+2g+d+3 sum(1/n_i). The leading negative-fundamental beam channel is

D_q C_-ei = -epsilon k_i b conjugate(A_i/K_i)/(2n_i) + O(epsilon^3).

Since b and A_i are nonzero, one such complex channel supplies two independent real beam directions. Together with the nine fundamental directions, two source baselines, and five quadratic hardware directions, the audited structural construction supplies all eighteen directions.

For fixed b!=0, a natural scaling is

S=diag(1 repeated 5, epsilon repeated 6, |b|epsilon repeated 2, |b|epsilon^2 repeated 5).

Thus the local adapted-coordinate rates are

hardware: eta/(|b|epsilon^2),
beam: eta/(|b|epsilon),
phase/speed: eta/epsilon,
leading amplitudes/source baselines: eta,

with all finite-record and optical constants retained. In wedge exponent terminology there are five strong, eight first-order, and five second-order directions, but the beam and hardware constants also depend on offset excitation.

These are adapted directions, not claims that each native wedge or source offset has the strong rate. Actual alpha_i=arcsin(a_i/K_i) inherits hardware uncertainty; its leading compensation contribution is O(eta/(|b|epsilon)). Actual source p inherits B0 times beam uncertainty and, away from q=0, hardware uncertainty through B0. Native degree/radian conversions must also be explicit.

A local chart must keep its offset uncertainty away from zero, all wedge amplitudes away from zero, and its beam excursion sufficiently small for the structural remainder test. A typical small-noise requirement is eta <= c |b|epsilon^2, so both hardware errors and beam errors stay in their respective local chart widths. This does not justify a uniform bound as b tends to zero.

## 5. Finite 200-sample extraction with cubic-aware leakage control

Use the existing base-seven witness N=(1,7,49)/20 Hz on t_k=k/20. All 63 nodes indexed by |m|_1<=3 are distinct, and adding the six repeated fundamental nodes gives the previously proved 69-column real confluent dictionary. This sampling statement is independent of the optics model and therefore transfers to the physical-vector chart.

The smaller base-five 31-column dictionary suffices for the quadratic limiting rank argument but not for claiming that all cubic finite-record contamination is removed. The cubic-aware base-seven dictionary avoids that gap without additional data.

Construct an explicitly verified left inverse and keep the l1 norm of each selected row. It extracts the required fundamental, negative-fundamental, source, and quadratic channels while annihilating other degree<=3 modes at the reference frequencies. Distinct nodes establish algebraic rank, not a useful numerical stability constant. The 200 actual sample times and frequency intervals, including modulo-20 aliases, must be used in the certificate.

At q=0, physical source-affinity and rotational parity give

F(e,b,0)=F_odd(e)+A_even(e)b.

The wedge-degree-three truncation therefore has value remainder O(|b|epsilon^4+epsilon^5). Cubic-aware quadratic extraction has hardware-normalized truncation error

O(epsilon^2 + epsilon^3/|b|).

By contrast, the smaller 31-column extraction can leak O(epsilon^3) centered angular terms into quadratic rows and gives only O(epsilon/|b|) without an additional leakage proof.

For derivatives, a forward remainder alone is insufficient. Expand in wedge degree while keeping the offset and beam dependence explicit; do not reuse an expansion that treats the fixed offset b as small. At q=0, D_q F has an even centered-angular part and an odd source-linear part. Truncating wedge degree at three gives D_q remainder O(epsilon^4+|b|epsilon^5), or O(epsilon^3/|b|+epsilon^4) after beam scaling. Hardware derivatives retain the value remainder order when taken at fixed leading amplitudes. Uniform box bounds still require explicit derivative enclosures and strict optical margins.

## 6. Unknown rotor and beam coordinates

The same six confluent fundamental columns address unknown frequencies locally. If deltaN is a candidate frequency error and T is the observation duration, hardware-normalized leakage terms include

- O((T deltaN)^2/(|b|epsilon)) from fundamental curvature after value-and-first-derivative annihilation;
- O(T deltaN) from shifts of quadratic modes;
- O(epsilon T deltaN/|b|) from shifts of cubic modes.

These charges supplement measurement error and optical remainder. A schematic quadratic hardware budget is therefore

C_D [eta/(|b|epsilon^2) + epsilon^2 + epsilon^3/|b| + (T deltaN)^2/(|b|epsilon) + T deltaN + epsilon T deltaN/|b|],

plus the explicit relative amplitude/offset terms and any unmodeled beam contribution. C_D is the actual certified finite-record row-norm constant, not an assumed order-one constant.

No unconditional frequency-initializer bias is established. A fundamental-only fit can absorb quadratic sidelobes and suffer O(|b|epsilon) frequency/relative-amplitude bias. An O(epsilon^2) statement requires a fitted model that includes the quadratic nuisance dictionary and a verified local fitting condition. Even then it is local in rotor labels.

The displayed coefficient inverse is derived at q=0. Unknown q must be estimated jointly through its negative-fundamental channels and propagated through the complete coefficient map, or enclosed and corrected. Simply applying q=0 ratios to an unknown-beam record does not solve full18. The projected exact-model certificate below is the clean way to handle all these interactions.

## 7. A constructive exact-model certificate

Let P be eighteen fixed real functionals selected from the cubic-aware demixer, and F_[3] the appropriate wedge-degree-three optical expansion, including its source and beam derivatives. At chart center zc set

T0=P D F_[3](zc) S^-1, L=T0^-1 P.

The inverse T0 is constructed from the fundamental tangent block, the explicit negative-fundamental beam block, the index inverse above, and the gain/distance/gap reconstruction. Its nondegeneracy is the structural theorem; its norm must be computed and certified.

On a convex native-valid adapted chart Z, certify

q_* = sup_Z ||L D F_vec(z) S^-1-I||_infinity < 1.

Decompose this into explicit polynomial coefficient variation and a differentiated optical remainder. This condition is stronger and more informative than checking a nonzero determinant at a point, and is directly usable for uncertainty contraction.

Every pair of points in Z compatible with the same full record and componentwise allowance eta then satisfies

||S(z-z')||_infinity <= 2||L||_(infinity<-infinity) eta/(1-q_*).

Proof: integrate the derivative along the segment and apply the Neumann inequality. For heterogeneous allowances use the row sums sum_m |L_jm| eta_m. All 400 original residual constraints must still be enforced. The theorem proves chart injectivity and conditional uncertainty, not existence or global label/branch uniqueness.

## 8. Lower bounds and what remains to prove

At centered incidence b=0 the exact first-prism deflection gauge remains a continuum. The earlier mixed-derivative argument already gives a rigorous O(1/|b|) instability lower bound near that stratum. A sharper matching lower bound follows if one certifies that the exact source transport along the gauge has derivative O(epsilon^2): then two systems separated by delta in n1 have forward separation at most C_G |b|epsilon^2 delta, so minimax error is at least

(1/2) min(r_gauge, 2eta/[C_G |b|epsilon^2]).

This sharper transport condition must be established from the physical affine ray map; it is not inferred from the quadratic index formulas alone. For general amplitude-compensated hardware paths a safe derivative order is O(|b|epsilon^2+epsilon^3), giving five-dimensional lower rates matching epsilon^-2 for fixed nonzero b.

A decisive minimal validation consists of one symbolic check of the coefficient inverse/differentials, one certified cubic-aware finite-clock demixer, one optical derivative remainder bound, and one q_*<1 on a specified nonzero-offset chart at supplied measurement precision. No parameter sweep is required. A complete global inverse still needs coverage of all labels and all surviving native regions, including the exact centered gauge and critical/zero-wedge strata.

## 9. Strengthened exact gauge lower bound: now audited

The physical-optics worker has confirmed the missing transport condition. For an exact prism, the source transport matrix has form

A_j=I+W_j u_j^T/(Z_j P_j), W_j=Z_j X_j-H_j B_j.

On the centered small-wedge family, X_j,B_j=O(epsilon), W_j=O(epsilon), u_j=O(epsilon), Z_j=1+O(epsilon^2), and H_j=n_j+O(epsilon^2), with P_j bounded away from zero on the guarded tube. Thus each A_j=I+O(epsilon^2), and so does their product. Along the exact first-deflection gauge, d alpha1/d n1=O(epsilon), yielding D_s A=O(epsilon^2).

Therefore the sharpened minimax lower bound in Section 8 is established under the stated uniform optical guards, not merely conjectured. It matches the |b|epsilon^2 upper information scale in at least the exact first-prism gauge direction. Five-dimensional amplitude-compensated hardware lower bounds retain denominator |b|epsilon^2+epsilon^3, which matches the upper wedge exponent for fixed nonzero b.

## 10. Do not assume the needed rotor or parameter chart is known

The witness left inverse is a proof of local identifiability, not a rotor estimator. The projected certificate of Section 7 is an a posteriori stopping criterion, not permission to initialize from a tiny box containing the truth. A complete method must derive every candidate rotor/parameter chart from the original record and prior, or explicitly return unresolved alternatives.

Here is one rigorous data-first sufficient condition that obtains rotor boxes without prior knowledge of their locations. It is mathematically explicit, but deliberately not advertised as an efficient universal method.

Suppose a uniform wedge-cubic remainder bound is certified on the entire current prior region, not merely near a fitted point. For the complex observation y_k=x_k+i y_k, combine this tail with measurement error to obtain |v_k-y_k|<=delta for every compatible cubic signal v. Its harmonic support is at most the 63 modes |m|_1<=3.

Construct the 63 by 63 observed Hankel matrix H_y=(y_(r+c)) for r,c=0,...,62 and rhs b_y=(y_(r+63)) for r=0,...,62. Only samples 0 through 125 are needed. Let chat solve H_y chat=-b_y, and set gamma=||H_y^-1||_infinity. If

63 delta gamma < 1,

then every possible true H_v in this noise box is invertible. Therefore every compatible cubic signal has exactly 63 distinct nonzero modes, rather than an unobserved lower-order Prony representation. Its monic annihilating polynomial p(z)=z^63+sum_(j=0)^62 c_j z^j has coefficient enclosure

||c-chat||_infinity <= gamma delta (1+||chat||_1)/(1-63 delta gamma).

This follows directly from H_v c=-b_v and the inverse perturbation identity; it does not assume a statistical noise model.

If disjoint candidate root disks satisfy a certified Rouche inequality

|phat(z)| > Delta_c sum_(j=0)^62 |z|^j

on every boundary, and their root counts sum to 63, every compatible cubic spectrum lies in their union with the certified counts. Map intersections with the unit circle to phase/frequency arcs. Retain every triple of possible fundamental roots satisfying the original signed-speed arc and whose products z1^m1 z2^m2 z3^m3, |m|_1<=3, can realize the entire enclosed spectrum. This is a finite, complete labeling test; permutations and sign alternatives are retained until optical coefficient constraints exclude them. The fundamental native arc lies strictly inside the sampling alias interval and is inverted uniquely once a root assignment is fixed.

The remaining 74 samples, all original paired residuals, physical guards, and structured coefficient identities must still be enforced. Root estimates alone are not hardware solutions.

This condition is deliberately strong. If modes vanish, collide, or are weak compared with delta, the rank63 test cannot pass. A smaller-rank fit is not allowed to discard weak unresolved modes without a separate completeness argument. If the uniform cubic-tail bound fails over a prior region, that region remains unresolved or is treated by exact optical exclusion; a locally estimated tail cannot silently replace it. Thus the procedure is genuinely data-checkable and all-frequency-unknown, but its failure reports lack of certification rather than manufacturing a frequency chart.

A publishable practical algorithm would exploit physical harmonic sparsity, sparse recurrence bounds, directed order constraints, and interval branch contraction to reduce this conservative full-dictionary cost. Those are next algorithmic steps, not results established by the witness theorem.

## 11. Directed difference modes encode physical prism order

The audited offset-dependent quadratic correction is

- sum_j k_j (s_j dot b) [sum_(i<j) k_i s_i/n_j+s_j],

where s dot b=Re(conj(s)b). For i<j its mixed term is

-(k_i k_j/[2n_j]) [b s_i conj(s_j)+conj(b)s_i s_j].

Therefore, with the complex paired record x+i y,

D_(i-j) = -b k_i k_j h_i conj(h_j)/(2n_j),
D_(j-i) = 0

at leading quadratic order. The corresponding positive-sum M_ij has the same magnitude, and

D_(i-j)/M_ij = (b/conj(b)) (conj(h_j)/h_j).

Equivalently, the division-free relation is

conj(b) a_j D_(i-j) - b conj(a_j) M_ij = 0.

This relation does not require knowing K_j or the sign of the wedge; K_j is positive real and cancels. It combines physical-order direction, phase consistency and sum/difference mode amplitude without forming a noisy ratio. Using only one real screen coordinate would lose the distinction between the two complex-directed frequencies; the paired record is essential.

For a candidate signed fundamental labeling, draw edge i->j when the observed directed difference mode can be certified as the nonzero one and the reverse is compatible with its bounded contamination. Three certified edges must form a transitive tournament, whose topological order is the unique physical prism order. A certified cycle rejects that candidate labeling/model region. Weak or ambiguous edges retain the remaining partial orders; no unsupported choice is made.

Unknown beam tilt and higher optical orders can populate the nominally zero reverse mode. Let rho_D be its extraction uncertainty from noise, tails and frequency intervals, and let U_rev(Q) be an independently enclosed reverse-mode contribution over the current beam/parameter set Q under the hypothesis that this direction is reversed. A sufficient exclusion of that hypothesis is

max(0, |Dhat_forward|-rho_D) > U_rev(Q).

A forward/reverse amplitude contrast by itself is insufficient without U_rev. The equality of leading |D| and |M| and the division-free phase identity provide extra interval consistency checks. U_rev may be bounded by beam derivatives times a verified beam enclosure plus higher-order remainder. If the unknown beam box is too broad, the order test simply remains unresolved; it does not assume a small or known beam.

This gives a physical, data-checkable pruning rule for the all-frequency-unknown labeling procedure in Section 10. It is not a standalone global label theorem: candidate spectra, coefficient enclosures, source/beam uncertainty and full residual constraints must be kept together.

Dictionary dimension convention: the 69 real sine/cosine/confluent temporal basis functions are applied separately to x and y. Equivalently one uses 69 complex exponential/confluent columns on the complex record. Realification of the latter has 138 real nuisance columns and 400 real output rows. This does not mean the physical model has 138 unknowns; only eighteen structured parameter directions are selected. The 63-complex-mode Hankel argument in Section 10 uses the complex-record convention consistently.
