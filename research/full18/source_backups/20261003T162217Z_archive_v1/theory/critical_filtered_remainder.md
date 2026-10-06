# Filtered finite-angle remainder: a sharp critical-boundary obstruction

## Result

Directly filtering the exact optical remainder is strictly stronger than propagating one global absolute Taylor tail. A boundary-safe frequency-dependent upper bound is available:

`||A_v(F-F^[m])||_infinity <= omega_inst(d_L(v))/2`,

where `A_v` is a real normalized annihilator of the degree-`m` formal signal, its degree is `L`, `d_L(v)=tan(18 degrees) max_(i,1<=r<=L)|z_i^r-1|`, and `omega_inst` is a certified instantaneous optical modulus in the norm `max_i ||u_i-u'_i||_2`. The whole-prior optical modulus has order `delta^(1/8)`.

This note proves that the exponent cannot be improved uniformly by the standard finite-order Prony filters. There are strictly physical original-200 records, with three nonzero distinct speeds and distinct low-order harmonic nodes, for which

`||A_v(F-F^[m])||_infinity = Theta(h^(1/8))`,

as the largest speed is of order `h`. This holds for every fixed full-lattice Taylor/Prony order whose annihilator fits the original record, including `m=1,2,3,4`. All eighteen variables remain unknown to the inverse; the explicit parameter family is a proof witness only.

Consequently no uniform filtered-tail bound `O(h^beta)` with `beta>1/8` can hold on the original unguarded prior for these filters. Raising the standard finite-order annihilator degree does not turn the global critical-boundary remainder into a high-order smooth small-frequency remainder. This is not an impossibility theorem for an exact nonlinear inverse or for more informative data-dependent boundary exclusion.

## 1. Why direct filtering gives a valid upper bound

The full-prior intrinsic denominator guards, uniform finite-order Taylor expansion and exact annihilators are in [full_prior_spectral_cover.md](full_prior_spectral_cover.md). A normalized real annihilator coefficient vector has l1 norm one and coefficient sum zero, since it has the constant node one as a root. Its positive and negative coefficient masses are therefore each one half.

For any original physical record, the instantaneous slopes at two positions of a length-`L+1` window differ by at most

`d_L(v)=a_* max_(i,1<=r<=L)|z_i^r-1|`,

in `max_i ||u_i-u'_i||_2`. The nonwedge parameters remain the same. The difference of the two convex averages of screen points is bounded by their diameter, proving

`||A_v F||_infinity <= omega_inst(d_L(v))/2`.            (1)

Since `A_v F^[m]=0`, the left side is exactly the filtered remainder. The pointwise optical modulus is derived by finite two-endpoint propagation through the exact square-root/reciprocal circuit. Internal glass roots and position denominators have unconditional positive lower bounds; three potentially critical exit roots each contribute a square-root modulus. This gives a finite explicit bound of form `omega_inst(delta)<=C delta^(1/8)`. No intermediate configuration between the two endpoints needs to be physical.

Equation (1) exactly removes a constant trace and goes to zero with the rotor frequencies. It is stronger than an absolute Taylor bound for some data. The question is whether a much higher small-frequency power can hold without excluding critical configurations. The following construction answers no for the standard annihilators.

## 2. An explicit triple-critical algebraic limit inside all native bounds

Use unit directions, the notation of Section 1 of `full_prior_spectral_cover.md`, and the fixed parameters

`t_x=t_y=23/50`, `n_1=n_2=n_3=9/5`,

`b=(0,0)`, `g=2`, `d=200`, `|u_1|=a=8/25`.

Let

`z_0=[1+2(23/50)^2]^(-1/2)`, `q_0=(q,q)=(23/50)z_0(1,1)`,

`nu=n^2-1`, `H_1=sqrt(n^2-|q_0|^2)`, `D_1=1+a^2`,

`c=[H_1-sqrt(D_1 nu)]/(q a)`.

Define the positive phase `phi_c` by the algebraic sine and cosine

`cos(phi_c)=[c+sqrt(2-c^2)]/2`,

`sin(phi_c)=[c-sqrt(2-c^2)]/2`,

and set

`U_1=a(cos(phi_c),sin(phi_c))`.

One has `0<phi_c<18 degrees` and `E_1=0` exactly. For display only, `phi_c` is about `10.6045 degrees`. The beam angles are `atan(23/50)<25 degrees`, and `atan(a)<18 degrees`.

For stages 2 and 3 choose an x-directed slope `U_j=(a_j,0)` that makes the exit critical at the incoming direction `(q_(j-1),z_(j-1))`. The correct three-dimensional formula is

`a_j=z_(j-1)^2 / [H_j q_(j-1),x + sqrt(nu) sqrt(q_(j-1),x^2+z_(j-1)^2)]`,  (2)

where `H_j=sqrt(n^2-|q_(j-1)|^2)`. The y component is nonzero here, so the coplanar formula with denominator `Hq+sqrt(nu)` must not be substituted.

Indeed, for `U=(a,0)`,

`E^2=z^2-2H q_x a+(q_x^2-nu)a^2`.

Rationalizing its positive root gives (2). At this critical root,

`q_out,x=sqrt(1-q_y^2)/sqrt(1+a^2)`,

`q_out,y=q_y`, `z_out=a q_out,x`.

The exact rational-interval certificate [critical_filtered_tail_witness.py](proof_checks/critical_filtered_tail_witness.py) proves the following strict bounds at the algebraic limiting point:

- all wedge amplitudes are below `tan(18 degrees)`;
- `0.032<a_2<0.033` and `0.00030<a_3<0.00032`;
- every `H_j`, `P_j`, and `z_j` is positive;
- every internal traversal numerator is positive;
- the three external flight margins exceed `0.60`, `1.76`, and `199.98`;
- the screen x coordinate lies in `(645433,645434)` and y in `(343242,343243)`.

The limiting point has `E_1=E_2=E_3=0` and is excluded from the physical set. Its role is to generate the strictly physical sequence below. Every inequality in the certificate uses rational interval arithmetic; decimal displays are not used as proof inequalities.

## 3. The positive three-root cascade

First keep `U_2,U_3` fixed and decrease the first phase:

`U_1(s)=a(cos(phi_c-s),sin(phi_c-s))`, `s>0`.

Use superscript zero implicitly for every limiting quantity in the formulas below. Put `P_j=H_j-U_j dot q_(j-1)` and `D_j=1+|U_j|^2`. All components of `q_0,U_1,q_1,U_2,q_2,U_3` relevant below are positive, except the stated zero y components.

Because

`U_1'(0)=(U_1,y,-U_1,x)`,

one obtains

`E_1(s)^2=A_1 s+O(s^2)`,

`A_1=2P_1 q (U_1,x-U_1,y)>0`.                         (3)

The transmitted direction identity gives

`q_1(s)=q_1-U_1 E_1(s)/D_1+O(s)`.

Consequently

`E_2(s)^2=A_2 E_1(s)+O(s)`,

`A_2=(2P_2/D_1)[(q_1 dot U_1)/H_2+U_2 dot U_1]>0`.  (4)

Similarly,

`q_2(s)=q_2-U_2 E_2(s)/D_2+O(E_1(s))`,

`E_3(s)^2=A_3 E_2(s)+O(E_1(s))`,

`A_3=(2P_3/D_2)[(q_2 dot U_2)/H_3+U_3 dot U_2]>0`.  (5)

These statements follow by differentiating the smooth radicand with respect to its incoming direction, not by differentiating a square root at zero. They imply

`E_1(s) ~ A_1^(1/2) s^(1/2)`,

`E_2(s) ~ A_2^(1/2) A_1^(1/4) s^(1/4)`,

`E_3(s) ~ K s^(1/8)`,

`K=A_3^(1/2) A_2^(1/4) A_1^(1/8)>0`.                  (6)

The positive leading coefficients make all three exit radicands strictly positive for every sufficiently small `s>0`. The remaining branch, traversal and native-bound inequalities stay satisfied by the strict margins above. Thus these are actual physical systems.

Let `p_exit,3` be the third glass exit position and `B_3=d-U_3 dot p_exit,3>0`. This position is independent of the third outgoing root when the incoming quantities are fixed. The exact derivative of the final x coordinate with respect to `E_3`, at the limit and holding those incoming quantities fixed, is

`-B_3(U_3,x z_3+q_3,x)/(D_3 z_3^2)<0`.

All changes of incoming third-prism quantities are of order `E_2=O(s^(1/4))`. Hence the exact final screen coordinate has the Puiseux expansion

`F_x(s)=F_x(0)-C s^(1/8)+O(s^(1/4))`,

`C=[B_3(U_3,x z_3+q_3,x)/(D_3 z_3^2)]K>0`.           (7)

The rational-interval check also verifies positivity of all three `A_j` and of `C`. For context only, `A_1≈0.3098`, `A_2≈0.5444`, `A_3≈0.05719`. Equation (7) is an asymptotic result; no evaluated moderate-size range for its asymptotic domination is asserted.

## 4. Put the cascade on the original 200 timestamps

Let `h>0` be small and choose phase increments per sample and initial phases

`omega_1=-h`, `omega_2=h^2`, `omega_3=h^3`,

`phi_1=phi_c-h`, `phi_2=phi_3=0`.

The signed speeds in the original hertz units are

`N_i=(10/pi) omega_i`.

At the original times `t_k=k/20`, the first phase is exactly `phi_c-h(k+1)`. The other two slopes rotate by `k h^2` and `k h^3`. Their changes are of higher order than the positive leading radicands in (3)-(5), so every sample is strictly physical for sufficiently small `h>0`. There are only 200 samples, so one common positive cutoff works for all of them.

Direct substitution in the same finite radical expansions, or the instantaneous modulus for the higher-order perturbations, gives uniformly for `k=0,...,199`

`F_(h),x,k = F_x(0)-C h^(1/8)(k+1)^(1/8)+O(h^(1/4))`.  (8)

All three signed speeds are nonzero and distinct for small positive `h`. All wedge amplitudes remain fixed and nonzero. There is no exact zero-speed or zero-wedge explanation in this witness family.

For any fixed `m`, all formal nodes indexed by `Lambda_m={ell:|ell|_1<=m}` are distinct and nonaliased for sufficiently small `h`. Indeed a difference of two exponents has phase

`h(-d_1+d_2 h+d_3 h^2)`

with nonzero integer vector `d` and `|d|_1<=2m`. The first nonzero coefficient in this polynomial dominates for sufficiently small `h`; its magnitude is also below `2pi`. This proves distinctness without assigning or estimating any true frequency in an inverse procedure.

### Optional exact algebraic input encoding

The obstruction is not restricted to transcendental records. Choose the native half-angle speed charts instead as `v=(-r,r^2,r^3)`, with positive rational `r` tending to zero, and take initial first phase `phi_c-2 atan(r)`. Then the phase increments are `(-2 atan(r),2 atan(r^2),2 atan(r^3))`. All chart parameters, rotor sine/cosine values at the 200 integer sample steps, and exact screen records are algebraic. The first displacement is `2 atan(r)(k+1)`, so (8) holds with leading coefficient `2^(1/8)C` and `r` in place of `h`. The remaining phase increments are of orders `r^2` and `r^3`; the positivity and node-distinctness proofs are unchanged. Thus a sequence of finitely algebraically encoded records has the same sharp exponent.

## 5. Every fixed standard Prony order retains the fractional tail

Let

`p_(m,h)(T)=product_(ell in Lambda_m) [T-exp(i ell dot omega(h))]`,

and let `L=L_m=|Lambda_m|`. Conjugate symmetry makes its coefficients real. Let

`A_(m,h)=p_(m,h)(S)/||coeff(p_(m,h))||_1`.

For sufficiently small `h`, it is the standard monic full-support annihilator of the degree-`m` formal signal. Its coefficients satisfy

`p_(m,h)(T) -> (T-1)^L`,

`||coeff(p_(m,h))||_1 -> 2^L`.

It annihilates `F_h^[m]` exactly and kills the constant term in (8) exactly. Therefore at the first admissible window, provided `L<=199`,

`[A_(m,h)(F_h-F_h^[m])]_x,0`

`= -(C/2^L) h^(1/8) Delta^L[x^(1/8)]_(x=1) + o(h^(1/8))`.  (9)

The coefficient is nonzero. For `alpha=1/8`, the elementary repeated fundamental-theorem-of-calculus identity gives

`Delta^L[x^alpha]_(x=1)`

`= integral_[0,1]^L alpha(alpha-1)...(alpha-L+1)`

`                   (1+s_1+...+s_L)^(alpha-L) ds`.

The integrand has a constant nonzero sign. In particular its absolute value is bounded below by

`|alpha(alpha-1)...(alpha-L+1)| (L+1)^(alpha-L)>0`.

The x-coordinate expression gives the lower bound. For the upper bound on both screen coordinates, apply (1): `d_L(v)=O(h)` for fixed `L`, and the global modulus is `O(h^(1/8))`. Equivalently, the remaining finite windows and both coordinates admit the same finite Puiseux expansion structure. Therefore (9) proves

`||A_(m,h)(F_h-F_h^[m])||_infinity = Theta(h^(1/8))`.   (10)

The implicit constants depend on the fixed order and proof family. This is not a numerical noise threshold.

For the original 200 observations, the standard full three-frequency lattice orders `m=1,2,3,4` have `L=7,25,63,129`. Each satisfies (10). Any adaptive choice among these finitely many orders still has a positive lower constant after taking their minimum, so cannot obtain a uniformly higher power by switching degree. Orders with `L>=200` have no full annihilation window in this experiment.

This last conclusion concerns these prescribed exponential annihilators. It does not claim that an arbitrary record-dependent linear filter, an exact nonlinear solver, or a model explicitly incorporating critical fractional-power terms must behave the same way.

## 6. Consequences for candidate-cover claims

The three-variable spectral cover and filtered bound remain sound. They may be useful after actual data-dependent exclusion of critical-root neighborhoods, or on record-derived regions with sharper constants. However:

- Uniform optical smoothness cannot be inferred from bounded original wedges or from distinct harmonic nodes.
- Distinct nonzero speeds alone do not make the exact finite-angle filtered remainder a high-order smooth function of the frequency scale.
- A finite change among the standard Taylor/Prony degrees does not remove the nested critical-root contribution on the full original prior.
- A data-derived candidate-cover theorem that promises high-order spectral accuracy must account explicitly for the boundary cascade or certify its exclusion. Supplying a critical-root guard without such a certificate changes the theorem's scope.

This completes the bounded filtered-tail investigation: an exact all-prior bound exists, and its worst-case frequency exponent is sharp for the standard filters. Practical complete optical localization still needs extra exact geometric exclusion or a validated boundary-aware feature model. No global completeness or uniqueness claim follows from this obstruction alone.

## Audit status

An independent mathematical audit checked the exact critical point, all cascade signs and factors, the distinct-speed original-200 construction, the normalized-filter limit, the nonzero finite-difference coefficient, and the finite adaptive-order conclusion. The exact rational-interval witness script was independently inspected and run successfully. This is a mathematical and symbolic audit, not a proof-assistant formalization; no parameter sweep or practical finite-noise threshold is claimed.

Local packaging note (2026-10-03): the audit and successful proof-check execution above are reported in the supplied audited memo. The exact supplied [proof source](proof_checks/critical_filtered_tail_witness.py) is now bundled locally (3,790 bytes; SHA-256 `95a75e0fe690dfdae4180ba9ce9137e009393465295c4bffe593a5b0c25808ec`), and its hash was verified during integration. The companion [full-prior spectral memo](full_prior_spectral_cover.md) is also bundled. The mathematical proof check was not rerun during this integration; local packaging or syntax checks do not replace the supplied independent mathematical audit.
