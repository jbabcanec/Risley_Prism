# Explicit six-root compiler: independently verified addendum

This addendum replaces the unspecified fixed-size quantifier-elimination precomputation in global_inverse_completeness.md with an explicit finite Boolean compiler. The final global decomposition is still enormous and no practical K=200 solver is claimed.

## 1. Scaled ray directions

Set t=(tan beta_x,tan beta_y), Q=1+|t|^2, X_0=t, Z_0=1, L_0=1. Every external direction is scaled by the same factor sqrt(Q), so its squared norm is Q. The physical unit direction is the scaled vector divided by sqrt(Q); this common scaling leaves all ray slopes unchanged.

At stage j, suppose the previous scaled direction is (X_(j-1),Z_(j-1))/L_(j-1), with L_(j-1)>0. For the instantaneous exit tilt u_j set

D_j=1+|u_j|^2,
H_j=sqrt(n_j^2 Q L_(j-1)^2-|X_(j-1)|^2),
P_j=H_j-u_j dot X_(j-1),
E_j=sqrt(P_j^2-D_j(n_j^2-1)Q L_(j-1)^2),
Z_j=H_j D_j-P_j+E_j,
X_j=X_(j-1)D_j+u_j(P_j-E_j),
L_j=L_(j-1)D_j.                                      (1)

The square roots are positive. Require both radicands strictly positive, P_j>0, and Z_j>0. The transmitted normal component of the new scaled direction is E_j/L_(j-1)>0. To verify (1), the tangential condition is B-X+u(Z-H)=0. Writing P=H-u dot X and D=1+|u|^2 gives the normal-output equation

R^2=P^2-D(n^2-1)Q,
R=D Z-H D+P.

Choosing R>0 gives (1) after clearing the previous positive denominator. The six roots occur in the ordered tower H_1,E_1,H_2,E_2,H_3,E_3; each radicand uses only earlier roots.

## 2. Positions and strict traversal

Start with p_0=b+6t. For ell_j=g,g,d define

p_exit=p+X_(j-1)(3+u_j dot p)/P_j,
p_next=p_exit+(X_j/Z_j)(ell_j-u_j dot p_exit).          (2)

These are exact plane intersections. Require 3+u_j dot p>0 and ell_j-u_j dot p_exit>0. The first condition, together with P_j,H_j>0, gives positive internal axial travel; the second gives positive external axial travel. All position denominators can be formed from products of positive P_j and Z_j. Thus clearing them preserves every sign. Do not rationalize by conjugate norms: a conjugate factor can vanish even when the physical denominator is positive. No such norm division is used here.

Equations (1)-(2), the six positive-radicand guards, and these strict inequalities reproduce the unique branch in the full polynomial ray graph. No reflective, backward, grazing, or total-internal-reflection roots are admitted.

## 3. Verified weighted degree bounds

Assign every instantaneous input component t,b,u_j,n_j,g,d and measured value/noise bound weight one; Q has weight two. Assign H_j weight 2j and E_j weight 2j+1. Induction gives

weight(L_j)<=2j,
weight(X_j),weight(Z_j)<=2j+2,
weight(P_j)<=2j+1.

The defining radicands have weight at most twice their assigned root weight. Therefore substituting r^2=R during tower reduction never increases weighted degree.

Write p=A/T with T>0 and weights a,b satisfying a>=b; initially a=1,b=0. Forming the exit point gives

A_exit=P_j A+X_(j-1)(3T+u_j dot A),
T_exit=P_j T.

Hence weight(A_exit)<=a+2j+1 and weight(T_exit)<=b+2j+1. The next-flat point can be represented by

A_next=Z_j A_exit+X_j(ell_j T_exit-u_j dot A_exit),
T_next=Z_j T_exit,

so a_next<=a+4j+4 and b_next<=b+4j+3. After three prisms this yields

weight(A_3)<=37, weight(T_3)<=33.                     (3)

Every interval observation test, after positive denominator clearing, has weighted degree at most37: e.g. A_(3,l)-(y_l+eta_l)T_3<=0. All branch and traversal guards have lower degree. A per-sample squared Euclidean residual has weighted degree at most74.

## 4. Explicit radical-sign elimination

Suppose a current tower level is r=sqrt(R)>0 and all earlier radicands satisfy their domain guards. Reduce a polynomial expression using r^2=R to E=A+B r, where A,B use only earlier roots. Let C=A^2-B^2 R. The sign of E is determined by the three signs of A,B,C:

- B=0: sign(E)=sign(A).
- A=0: sign(E)=sign(B).
- A and B have the same nonzero sign: E has that sign.
- A and B have opposite signs: sign(E)=sign(A) sign(C).

The last case follows by multiplying A+B r by the nonzero conjugate factor of the known opposite sign, or directly comparing |A| with |B|sqrt(R). These cases cover zero, strict, and weak tests exactly, including C=0 cancellation.

Recursively apply this rule to all six levels, retaining the positive-radicand guards themselves. An expression of weighted degree d yields expressions of degree at most2d one level down; after six levels every terminal ordinary polynomial has degree at most64d. The domain guards must not be discarded: these sign identities are only used where the relevant positive roots exist.

Thus every interval/noise/branch predicate has an explicit Boolean sign-circuit representation using ordinary instantaneous-input polynomials of degree at most

37*64=2368.                                          (4)

For per-sample Euclidean balls the bound is74*64=4736.

There are at most3^6=729 terminal sign-polynomial tests per original atom when one keeps a shared sign-circuit representation. This is a bound on terminal polynomial tests, not a claim that naïvely expanded textual Boolean syntax has only729 occurrences. There are at most24 primitive atoms for the interval model: eighteen branch/traversal tests, four observation-band tests, and two nonnegative noise-bound tests. Hence24*729=17496 is a safe bound on terminal polynomial occurrences before deduplication, excluding the separate prior-box predicate. One may also compile every radicand-domain guard independently with the same conservative bound.

All compiler operations are explicit: polynomial addition/multiplication, reduction by six monic quadratic relations, and the finite sign table. No data sweep, genericity assumption, algebraic root choice without signs, or general auxiliary-variable QE is required for this preprocessing.

## 5. Consequences and remaining limits

Substitute the exact rational rotor iterates from global_inverse_completeness.md into these polynomials. Their denominators are positive, so each sign remains equivalent after clearing denominators. This gives a fully specified exact finite-record formula in eighteen shared native-chart coordinates, with sample polynomial degree O(K) and integer coefficient heights O(K+tau) for rational data, up to the now-explicit fixed compilation constants.

More explicitly, an instantaneous polynomial of ordinary degree D becomes a chart polynomial of degree at most D(6k+7) after multiplying by the positive product of the three rotor denominators, each to power D. To see this, a monomial with total tilt degree m has base degree at most D-m; its substituted tilt numerators contribute at most m(2k+3), and unused denominator powers contribute at most(3D-m)(2k+2). The sum is D(6k+7). Thus max degree over K consecutive samples is at most D(6K+1). For D=2368 and K=200 this conservative degree bound is2,843,968.

The final complete real-algebraic decomposition still has fixed-dimension but astronomical worst-case complexity. Neither (4) nor the sign-count bound makes degree2368 polynomials in eighteen variables tractable. The result improves implementability and proof specificity of the compiler, not the practical solver status.

## 6. Independent check of the alternate25-auxiliary cubic graph

The scaled direction convention also gives a polynomial graph with25 auxiliary variables: Q plus eight variables per prism (H, B_x,B_y,Z,p_next,x,p_next,y,P,alpha). Use Q=1+|t|^2 and p_0=b+6t. At each prism impose

H^2+|X|^2=n^2 Q,
|B|^2+Z^2=Q,
B-X+u(Z-H)=0,
P=H-u dot X,
alpha=u dot p,
ZP p_next=ZP p+ZX alpha-HB alpha+3ZX-3(u dot X)B+ell P B.

All scalar equations have degree at most three, including n^2Q. Require H,Z,P>0, Z-u dot B>0, 3+alpha>0, and (3+ell)P-H(3+alpha)>0. The spatial equation follows by eliminating internal axial height h=H(3+alpha)/P from the exact two-segment displacement. This confirms the count and branch equivalence; the original26-auxiliary unit-direction graph is also correct.
