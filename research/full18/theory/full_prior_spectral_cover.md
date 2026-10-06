# Full-prior finite-angle remainders and a collision-safe spectral cover

## Verdict and scope

This addendum establishes three structural results for the original passive 200-sample, all-eighteen-unknown full-vector model:

1. The axial and plane-intersection denominators have explicit positive lower bounds on the whole admitted original prior, including limits where an exit critical-refraction root vanishes. A positive critical-root guard is not needed for continuity or for finite endpoint Taylor bounds.
2. Every fixed wedge Taylor order has a constructive, uniform finite-angle remainder bound on that whole prior. This follows from endpoint algebraic identities, not from incorrectly bounding derivatives across a critical root.
3. The first-order remainder gives a complete unknown-frequency outer cover defined by 772 degree-at-most-six polynomial inequalities in only three variables. The cover retains all zero amplitudes, zero speeds, frequency collisions, signs, and assignments. Its exterior has an explicit residual gap.

These do not close useful all-eighteen-parameter localization. A single exact algebraic physical limiting family proves that every global first-order absolute tail bound exceeds 17,600 native screen units. The same example forces the global absolute tails for Taylor orders one through four all to exceed 17,000. Accordingly, a global-tail-only spectral localization is provably vacuous for a broad class of bounded-amplitude records. No assertion of literature novelty is made.

The meaningful new distinction is between existence of a rigorous whole-prior tail and usefulness of that tail. The latter cannot be inferred from bounded native wedges or from existence of a formal 17+1 inverse.

## 1. Intrinsic denominator margins without a critical-root guard

Use unit external directions. At a single prism let the incoming direction be `(q,z)`, with `|q|^2+z^2=1` and `z>0`. Write the exit plane slope as `u`, with

`|u| <= a_* = tan(18 degrees)`.

Let

`H = sqrt(n^2-|q|^2)`,
`D = 1+|u|^2`,
`P = H-u dot q`,
`E = sqrt(P^2-D(n^2-1))`.

In this section permit the limiting transmitted branch `E>=0`. The actual physical predicate still requires `E>0`. The output direction `(r,w)` obeys

`h = (P-E)/D = (n^2-1)/(P+E)`,
`r = q+u h`,
`w = H-h`,
`w-u dot r = E`.

Since `n>=1.3`,

`P >= p_* := sqrt(1.3^2-1)-tan(18 degrees) > 0.505`.

In particular `h>0`. Also `P^2-E^2=D(n^2-1)` and `P>=sqrt(D(n^2-1))`, so

`0<h <= sqrt((n^2-1)/D) <= sqrt(n^2-1)`.

It follows directly that

`w >= sqrt(z^2+n^2-1)-sqrt(n^2-1) > 0`.                 (1)

There is also a useful geometric proof of (1). From `E>=0`,

`u dot q <= w-h|u|^2`.

The unit-norm identities therefore give

`z^2-w^2 = 2h(u dot q)+h^2|u|^2 <= 2hw-h^2|u|^2 <= 2hw`.

Thus `z^2<=2Hw-w^2`, whose lower root is (1).

Set

`zeta_0 = [1+2 tan^2(25 degrees)]^(-1/2)`,
`zeta_j = sqrt(zeta_(j-1)^2+1.8^2-1)-sqrt(1.8^2-1)`, for `j=1,2,3`.

The right side of (1) increases with `z` and decreases with `n`. Consequently every weakly transmitted three-prism trace in the original prior satisfies

`H_j >= sqrt(1.3^2-1)`, `P_j>=p_*`, and `z_j>=zeta_j>0`.  (2)

These bounds can be small; their significance here is that they are explicit and unconditional on critical-root separation. They apply to every sampled rotor orientation. The positive denominators in the scaled six-root compiler follow after multiplying by its bounded positive scale factors.

The exact position recursion

`p_exit = p+q(3+u dot p)/P`,
`p_next = p_exit+(r/w)(ell-u dot p_exit)`

therefore extends continuously to the compact weak optical domain, with the original bounded geometry variables. It is legitimate to use this extension for outer bounds even when an extended point fails traversal. It is not legitimate to report that point as a physical witness.

For a completely explicit crude bound, put `A_0=5+6 tan(25 degrees)` and define

`A_exit,j = A_(j-1)+(3+sqrt(2) a_* A_(j-1))/p_*`,
`A_j = A_exit,j + [ell_max,j+sqrt(2) a_* A_exit,j]/zeta_j`,

where `ell_max,1=ell_max,2=15` and `ell_max,3=200`. Then `|p_j|_infinity<=A_j`. Finiteness does not depend on traversal margins.

Neither (1) nor (2) bounds derivatives of `E` at `E=0`. A derivative-based LP exclusion certificate still needs its stated derivative conditions or another modulus; continuity alone is not differentiability.

## 2. A whole-prior endpoint Taylor-remainder compiler

Keep the nonwedge parameters in their complete original compact prior. Treat the six components of the instantaneous slope vectors `u_1,u_2,u_3` as wedge variables, and put

`a = max_j |u_j| <= a_*`.

At zero wedges all exit radicands are bounded below by `zeta_0^2`; the local Taylor expansion is well-defined uniformly in the remaining unknown parameters. Let `F^[m]` denote the degree-at-most-`m` Taylor polynomial in these six wedge variables. It is the formal optical Taylor polynomial, not a measured extra experiment.

### Endpoint square-root lemma

For `A>0`, `A+d>=0`, and every integer `m>=0`,

`|sqrt(A+d)-sum_(j=0)^m binom(1/2,j) A^(1/2-j) d^j|`

`<= c_m |d|^(m+1)/A^(m+1/2)`,

where

`c_m = binom(2m,m)/4^m`.                               (3)

For `d>=0`, Taylor's integral remainder gives the smaller usual coefficient `|binom(1/2,m+1)|`. For `-A<=d<0`, set `lambda=-d/A`. All nonconstant coefficients of `sqrt(1-lambda)` are negative. Dividing its tail by `lambda^(m+1)` and using `0<=lambda<=1` bounds it by the tail at one, which equals `c_m`. The endpoint `d=-A` is included by continuity.

Formula (3) stays finite when the endpoint root is zero. In particular, for `m=1` the exact identity is

`sqrt(A+d)-sqrt(A)-d/(2sqrt(A))`

`= -d^2/[2sqrt(A)(sqrt(A+d)+sqrt(A))^2]`.

No positive lower bound on `sqrt(A+d)` occurs.

### Endpoint reciprocal lemma

For `A>=A_*>0` and `A+d>=d_*>0`,

`1/(A+d) = sum_(j=0)^m (-d)^j/A^(j+1)`

`            + (-d)^(m+1)/[A^(m+1)(A+d)]`.             (4)

Again only the base value and the actual endpoint need be controlled.

### Uniform finite-order theorem

For each fixed `m` there is a constructively bounded finite constant `C_m`, depending only on the original nonfrequency prior, such that every physically admissible original record satisfies

`||F-F^[m]||_infinity <= C_m a^(m+1)`.                 (5)

The same statement holds on the weak optical extension. No positive lower bound on any exit critical root is imposed.

Proof and construction. Traverse the finite six-root arithmetic circuit. Carry its homogeneous jet of degree at most `m`, certified bounds on each homogeneous coefficient, an endpoint magnitude bound, and a remainder constant. Initial components have exact jets. Sums and products are treated by finite convolution and bounding omitted products with `a<=a_*`.

For a square root, write its radicand as its zero-wedge value plus `d`. Induction first gives an endpoint estimate `|d|<=L a`; apply (3), replace powers of `d` by their computed jets, and bound their omitted finite products. Every zero-wedge radicand has a proved positive lower bound. For reciprocals apply (4), using the unconditional endpoint denominator bounds in (2), and handle finite jet products in the same way. The position recursion closes the induction. All constants use only finite arithmetic, square roots and outward bounds over original compact nonwedge intervals, so rational upper bounds are computable.

This construction does not require the line segment from zero wedge to the endpoint to remain physical. It uses algebraic endpoint identities, whose hypotheses are checked only at the zero-wedge base and the admitted endpoint. It does not assert that the resulting constants are moderate or that increasing `m` makes the bound decrease on the full wedge prior.

One may instead carry the jets in the signed wedge sines used in `oblique_inverse.md`. The additional normal factors `sqrt(1-|s_j|^2)` have endpoint lower bound `cos(18 degrees)>0`; the same proof applies. The formal coefficient functions through degree two agree under the change `tan(alpha)=sin(alpha)+O(sin^3(alpha))`. Amplitude reconstruction must use the chosen coordinate consistently.

## 3. A complete degree-six unknown-frequency cover

Let `tau_1` be any certified uniform absolute bound for the first-order remainder on the region under consideration. For a whole-prior statement it must cover the whole original prior. Let

`r_y(theta)=||F(theta)-y||_infinity`,
`v_i=tan(pi N_i/20)`,
`z_i=(1+i v_i)/(1-i v_i)=exp(2 pi i N_i/20)`.

The native speed chart is bijective on `N_i in [-3.5,3.5]`, and `|v_i|<1`. The two real first-order channels are sums of a constant and the six exponentials with nodes `z_i,conjugate(z_i)`. Some coefficients may vanish and nodes may coincide.

Define

`Q_v(T)=(T-1) product_(i=1)^3 [(1+v_i^2)T^2-2(1-v_i^2)T+(1+v_i^2)]`

`       =sum_(j=0)^7 Q_j(v) T^j`.                      (6)

Every `Q_j` is a real polynomial of total degree at most six. The polynomial annihilates the first-order signal for every frequency triple, including repeated and zero nodes. Multiplicity in (6) is harmless: an annihilator need not be minimal.

Because `|v_i|<1`, its coefficients have alternating signs. Evaluating at `T=-1` gives the particularly useful exact identity

`sum_(j=0)^7 |Q_j(v)| = -Q_v(-1) =128`.               (7)

Thus normalization by 128 introduces no unknown or small divisor.

For the fixed observed record define

`S_y(v) = (1/128) max_(h=x,y; 0<=k<=192)`

`                       |sum_(j=0)^7 Q_j(v)y_(h,k+j)|`. (8)

### Spectral exclusion theorem

Every physical system satisfies

`r_y(theta) >= S_y(v(theta))-tau_1`.                  (9)

Indeed (6) annihilates `F^[1]`; apply its coefficient vector to `y-F^[1]=(y-F)+(F-F^[1])` and use (7). This uses every observation window in the actual original 200 samples. There is no true-frequency input or fitted-frequency substitution.

For arbitrary `epsilon>=0` and explicit `gamma>0`, put

`Omega_(epsilon,gamma)={v in the full speed chart:`

`                              S_y(v)<=epsilon+tau_1+gamma}`. (10)

Every `epsilon`-compatible physical system has its speed triple in (10). At every frequency triple outside (10), every physical system has

`r_y(theta)>epsilon+gamma`.                            (11)

The cover (10) is given by exactly 772 scalar polynomial inequalities of degree at most six, in three real variables, besides the three speed bounds. For algebraic input record/error/tail/gap values it therefore has a finite exact three-variable semialgebraic decomposition. That decomposition is an all-branches frequency cover, not an eighteen-variable generic interval-search restatement. It neither bounds the number of useful separated regions nor localizes the other fifteen unknown native coordinates.

Its dependence on `v_i^2` and its symmetry in the three factors deliberately retain both speed signs and every prism assignment. The separate ellipse orientation and physical-order constraints must resolve them when possible. Zero wedges, zero DC, zero speeds and collisions are not silently discarded because a Hankel rank test fails.

### Stronger treatment of exact collision strata

The single polynomial (6) keeps collisions soundly, but repeated factors can weaken its score. They can be removed without assuming a generic spectrum. Partition the speed chart according to which `v_i` vanish and which nonzero squares `v_i^2` coincide. These are the 15 set partitions of `{0,1,2,3}`, with the block of `0` representing zero speeds. For each nonzero block select one representative and use its quadratic factor only once. With `q` nonzero blocks, the resulting polynomial has degree `2q+1`, coefficient l1 norm exactly `2^(2q+1)`, and polynomial dependence of degree at most `2q` on the selected `v_i`.

The normalized score on that exact stratum uses every window `k=0,...,K-L-1`, where `L=2q+1`, and satisfies the same exclusion theorem. The union over all 15 explicitly represented strata gives another complete outer cover, with stronger tests on the zero/collision strata. A stratum is never discarded because its annihilator order drops. At all zero speeds the normalized polynomial is `(T-1)/2`; the exact finite-angle trace itself is constant, so this stratum has filtered tail zero and is excluded whenever an observed consecutive difference exceeds `2 epsilon`.

### Filtered remainders can evade the absolute-tail obstruction

The relevant error in (9) is actually the filtered tail, not necessarily its absolute norm. For any normalized annihilator `A_v` above, a proved bound

`||A_v(F-F^[1])||_infinity <= b(v)`

replaces `tau_1` in (9) without any other change. Because `A_v F^[1]=0`, this is equivalently a bound on `A_v F` itself. Such a bound must be established; the annihilation identity alone does not provide it.

One concrete alternative comes from a certified global instantaneous optical modulus. Let `omega(delta)` bound the screen difference between two weakly admitted instantaneous tilt configurations with the same nonwedge variables and tilt-vector distance `max_i ||u_i-u_i' ||_2` at most `delta`. The guards in Section 1 and the three possibly critical exit square roots give a constructive modulus of form `omega(delta)<=C delta^(1/8)` on the compact instantaneous prior; internal glass roots have positive uniform margins. The detailed two-endpoint circuit propagation is developed in `global_boundary_continuation.md`.

For an annihilator of degree `L`, put

`d_L(v)=a_* max_(1<=i<=3,1<=r<=L) |z_i^r-1|`.

Its normalized real coefficients sum to zero and have total positive and negative masses both equal to one half. Taking the difference of the corresponding convex averages of the screen points in a window proves

`||A_v F||_infinity <= omega(d_L(v))/2`.                (12)

Consequently `b(v)=min(tau_1,omega(d_L(v))/2)` is valid wherever both bounds apply. This restores the correct zero-frequency limit and does not require a positive critical-root margin. Its usefulness depends on actual verified constants, not merely the exponent. The lower-tail witness below has all speeds zero and is annihilated exactly; it therefore does not disprove a sharper filtered-tail approach.

### General finite-order version

A degree-`m` wedge Taylor signal has nodes

`z^ell`, `ell in Lambda_m={ell in Z^3:|ell|_1<=m}`.

The product over all these formal nodes is an annihilator even at collisions. Its degree is

`L_m=sum_(j=0)^3 2^j binom(3,j) binom(m,j)`

`    =(4m^3+6m^2+8m+3)/3`.

Thus `L_1=7`, `L_2=25`, `L_3=63`, `L_4=129`, and `L_5=231`. Normalize this real annihilator by its coefficient l1 norm, which is nonzero because it is monic. The corresponding window score again satisfies (9), with `tau_m` in place of `tau_1`. For `m>=5` this full-support annihilator has no window within 200 observations. This does not rule out more structured nonlinear methods at higher order; it rules out simply increasing this full-lattice Prony annihilator order on the original record.

## 4. A rigorous large-tail obstruction inside the original prior

The following one-parameter physical limiting family demonstrates why the global bound in (5) is not automatically useful.

Use coplanar positive deflections, zero speeds and phases, beam slope `t_x=2/5,t_y=0`, indices `n_1=n_2=n_3=9/5`, source offsets zero, `g=2`, `d=200`, and first wedge slope `u_1=8/25`. These satisfy every original native bound. In particular `atan(2/5)<22.5 degrees<25 degrees` and `8/25<tan(18 degrees)`.

For a positive incoming coplanar unit direction `(q,z)`, write `H=sqrt(n^2-q^2)`. The positive critical exit slope is

`u_c(q,z)=z^2/[qH+sqrt(n^2-1)]`.                      (13)

It is obtained by solving `E^2=z^2-2Hq u-(H^2-1)u^2=0`. At (13), the limiting outgoing direction is

`q_out=1/sqrt(1+u_c^2)`, `z_out=u_c/sqrt(1+u_c^2)`.

First propagate through the strictly transmitted first prism. Choose `u_2=u_c(q_1,z_1)` and then `u_3=u_c(q_2,z_2)`. At this limiting point the second and third exit critical roots are zero. It is not a physical solution and is never reported as one.

The following conservative open rational bounds are rigorously verified by exact rational interval arithmetic in `proof_checks/full_prior_tail_witness.py`:

- `0.185<u_2<0.187` and `0.011<u_3<0.012`, both below the original 18-degree wedge bound;
- all internal traversal numerators are positive;
- the three external flight margins exceed `0.95`, `0.88`, and `199.85`, respectively;
- `17821<F_x<17822`;
- `196<F_x^[1]<197` for slope-coordinate first order;
- `192<F_x^[1]<193` for the sine-coordinate first order of the existing oblique construction.

To obtain strictly physical systems, keep the first prism unchanged and set, successively,

`u_2(s)=(1-s)u_c(q_1,z_1)`,
`u_3(s)=(1-s)u_c(q_2(s),z_2(s))`, with `s>0`.

For sufficiently small `s`, both exit radicands are strictly positive, all other optical and traversal inequalities remain strict by the displayed margins, and every original bound remains valid. The exact record and its formal Taylor polynomials converge to the limiting values. This is a genuine physical approaching sequence, not a use of a forbidden boundary root as data.

It follows that any uniform absolute first-order remainder bound on the whole original prior must satisfy

`tau_1>17600`.                                        (14)

For a bound of the form `C_1 a^2` in slope coordinates, this also forces

`C_1>17600/(8/25)^2=171875`.                           (15)

This lower bound on a universal constant comes from a far-from-small-wedge portion of the prior; applying it unchanged near tiny wedges can be extremely conservative.

The same script propagates exact Taylor jets of orders one through four in the radial variable `lambda`, with `u_i(lambda)=lambda u_i`. This is exactly evaluation of the multivariate degree-`m` Taylor polynomial at the specified wedge vector. The certified values are enclosed in the following unit rational intervals:

- degree 1: `196<F_x^[1]<197`;
- degree 2: `240<F_x^[2]<241`;
- degree 3: `278<F_x^[3]<279`;
- degree 4: `308<F_x^[4]<309`.

The same certificate propagates the corresponding sine-coordinate jets and verifies the same lower tail bound for all four degrees. Thus each uniform whole-prior absolute tail `tau_m`, for `m=1,2,3,4`, exceeds 17,000 in either of these coordinate conventions. The computation is one explicitly specified algebraic proof witness. Every certification inequality uses exact rational arithmetic and rational square-root bounds; floating-point values are printed only for readability. It is not a numerical parameter campaign.

### Exact vacuity consequence

For every normalized annihilator coefficient vector, its action on `y` has magnitude at most `||y||_infinity`. Hence if

`||y||_infinity<=epsilon+tau_m`,                       (16)

the corresponding global-tail spectral candidate set is the entire speed prior. In particular the degree-six cover cannot localize any frequency for records with `||y||_infinity<=17600` when its tail is a sound whole-prior first-order bound.

The existing seven-tone Hankel sufficient condition also necessarily fails in this situation. If `D H_y=I_7`, `gamma_H=||D||_infinity`, and `Y=||y||_infinity>0`, then

`1<=gamma_H ||H_y||_infinity<=7 gamma_H Y`.

Therefore `7 gamma_H delta>=delta/Y>=1` whenever `delta>=Y`. If `Y=0`, the design cannot have the required rank. This is a failure of that specific global-tail sufficient condition, not an impossibility theorem for the exact inverse.

More generally, using only the necessary uniform surrogate band `||F^[1](theta)-y||<=epsilon+tau_1` cannot remove any frequency triple under (16): axial zero-wedge systems with zero offsets have `F^[1]=0` for every speed triple. A parameter-dependent bound that vanishes at zero wedges can be stronger; this argument does not forbid such a bound, exact optical profiling, or direct semialgebraic exclusion.

## 5. The strongest conditional all-eighteen-parameter use

The spectral theorem can remove an explicitly certified portion of the complete speed prior with a known exterior gap. On its retained cells, validated demixing may provide coefficient regions for the formal fundamental, second-harmonic and DC coefficients, after charging the tail and frequency variation. No rank assumption is needed for retaining a cell; rank and separation are needed only for a particular inversion or demixing chart.

A rigorous full optical candidate theorem must then have one of the following checkable forms:

1. Every retained spectral cell has a finite cover of all solutions of the corresponding coefficient-enclosure constraints and original parameter bounds, including every singular coefficient fiber and every physically admissible boundary-approaching portion; or
2. Each portion not so covered has a proved exact residual lower bound, for example by the geometry-profiled certificate or the critical-root-continuous certificate developed separately.

For a regular demixing cell, the already audited last-prism polynomial constraints and strong/weak inverse supply concrete candidate mechanisms. They must use the leading baseline with its quadratic DC uncertainty; they must retain uncertain coefficient regions rather than replace them by fitted points. Saturation denominators, zero amplitudes, assignments and exceptional positive-dimensional fibers remain explicit.

A cover satisfying these conditions, together with (11), is globally complete. The remaining requirements are substantive and cannot be inferred from this note's polynomial frequency cover. Calling the entire unresolved full prior one exceptional candidate is formally a cover but provides no useful localization claim.

## 6. What changed and what remains open

New relative to the earlier addenda:

- Critical-root separation is unnecessary for global axial/position denominator bounds and finite endpoint Taylor remainders.
- The complete first-order speed outer set can be constructed from degree-six constraints in three variables, with an explicit normalized annihilator and exterior gap, without a seven-column rank or root-separation premise.
- An exact physical limiting family quantifies why the single global-tail implementation cannot supply a uniformly useful original-prior inverse; increasing the standard full-lattice Prony order through every order that fits the original 200 observations does not remove this obstruction when certification uses one whole-prior absolute tail. Sharper filtered remainders are not ruled out.

Still not proved: a quantitatively useful finite all-eighteen-parameter candidate cover for arbitrary exact-compatible original records. Advancing further requires a genuinely sharper data- or region-dependent contamination theorem, structural exact exclusion of the large-tail branches, or an exact global geometric reduction. That additional structure must be proved or certified; it cannot be supplied as a free boundary guard or a hidden neighborhood around the truth.

## Follow-on filtered-tail result

`critical_filtered_remainder.md` develops the direct filtered bound further. A separate exact triple-critical proof family shows that its global frequency exponent `1/8` is sharp for every standard finite-order Prony annihilator usable on the original record, even when all three speeds and all finite-order harmonic nodes are distinct. This is an asymptotic boundary obstruction, not a practical noise threshold or a proof against data-dependent exclusion.

## Audit status

An independent audit checked the intrinsic guards, endpoint square-root lemma and Taylor induction, degree-six normalized annihilator, 772 inequalities, 15 collision strata, filtered half-mass bound, and the large-tail witness. The exact rational-interval witness script was inspected and run successfully. The instantaneous distance norm and collision-stratum window ranges were clarified during audit. This is a mathematical/symbolic audit, not formal proof-assistant verification.
