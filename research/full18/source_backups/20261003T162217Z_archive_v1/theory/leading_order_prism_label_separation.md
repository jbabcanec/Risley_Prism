# Generic physical-order separation in the axial-offset leading inverse

## Scope

This note concerns the **formal leading fundamental and quadratic coefficients** in vector_inverse.md at axial incidence q=0 and nonzero source offset b. It proves that these coefficients generically determine physical prism order, in addition to determining hardware conditional on that order. Wrong physical labelings are not generic symmetries of even this leading model.

It does not identify exact finite-wedge Fourier coefficients from 200 samples and does not establish generic uniqueness of the exact full18 map. The finite200 transfer still requires controlled coefficient extraction, finite-angle correction, and exclusion of remote branches. The exact domain contraction theorem is separate.

## 1. Modal quantities independent of trial physical order

Label three resolved modal nodes by i=1,2,3, temporarily using their true physical order for the proof. Put k_i=n_i-1, so k_i is in [0.3,0.8]. Let h_i be the nonzero complex wedge sine/phase amplitude. The leading data are

A_i=k_i L_i h_i,
S_i=-(conjugate(b)/2)k_i h_i^2,
M_ij=-(conjugate(b)/2)k_i k_j h_i h_j/n_j, i&lt;j,

where

L_1=d+2g+3/n_2+3/n_3,
L_2=d+g+3/n_3,
L_3=d.                                               (1)

The modal ratio and positive amplitude quantities are

R_ij=M_ij^2/(S_i S_j)=k_i k_j/n_max(i,j)^2,
D_i=|A_i| sqrt(|b|/(2|S_i|))=sqrt(k_i)L_i.             (2)

These are computable from the formal coefficient list without assigning physical labels.

For every trial physical order sigma, the ratio inversion from vector_inverse.md gives at most one physical triple k_i' in [0.3,0.8]. Specifically,

r=R_sigma1,sigma3 / R_sigma2,sigma3,
u=sqrt(R_sigma1,sigma2 / r),
k_sigma2'=u/(1-u), k_sigma1'=r k_sigma2',
R_sigma2,sigma3/k_sigma2'=k_sigma3'/(1+k_sigma3')^2.

The last scalar map is strictly increasing on [0.3,0.8], so its allowed solution is unique. Invalid signs, ratios, or index bounds reject the order. No quadratic conjugate outside the original index interval is retained as physical.

For a trial whose index solution is physical, its inferred positive levers satisfy

L_i'=D_i/sqrt(k_i')=s_i L_i,
s_i=sqrt(k_i/k_i')&gt;0.                               (3)

For fixed true indices and a fixed trial order, s_i depends only on those indices, not on g,d, wedge amplitudes, phases, or b.

## 2. The spare lever identity

The last two trial levers set

d'=L_sigma3',
g'=L_sigma2'-L_sigma3'-3/n_sigma3'.

The first lever is compatible only if

L_sigma1'-2L_sigma2'+L_sigma3'
  =3/n_sigma2'-3/n_sigma3'.                           (4)

The source, distance and gap bounds still have to hold, including g'&gt;0. Substituting (1) and (3), (4) is an affine equation in the true variables (d,g).

Let v_i equal 1 for the first or third position in the trial order and -2 for its middle position. Its two affine coefficients are

coefficient of d: sum_i v_i s_i,
coefficient of g: 2v_1s_1+v_2s_2.                    (5)

If both coefficients vanish, the vector (v_i s_i) is proportional to (1,-2,1). Since every s_i is positive and precisely one v_i is negative, the trial middle modal index must be the true middle index 2. The only possibilities are therefore the correct order (1,2,3) and its reversal (3,2,1).

For the reversal, (5) also implies s_1=s_2=s_3. Equivalently k_i'=lambda k_i for one lambda&gt;0. Comparing the three ratio equations gives

lambda/(1+lambda k_1)=1/(1+k_2),
lambda/(1+lambda k_1)=1/(1+k_3),
lambda/(1+lambda k_2)=1/(1+k_3).

The first two imply k_2=k_3. The third then implies lambda=1, and the first implies k_1=k_2. Thus all three indices are equal, and every s_i=1.

But with equal indices the reversal is physically impossible:

g'=L_2-L_1-3/n=-g-6/n&lt;0.                            (6)

It violates the original positive gap even before the lower bound g&gt;=2 is enforced.

Hence every wrong order that can be physically admissible has at least one nonzero affine coefficient in (5).

## 3. Generic uniqueness of the leading physical assignment

For any fixed true index triple, each wrong physical order is either impossible for every (g,d), or satisfies one proper affine-line constraint in the (g,d) plane. There are at most five wrong orders. Outside the union of these at most five lines, the only physical order compatible with the entire formal leading coefficient list is the true order.

The exceptional set can be smaller because each line must additionally satisfy the true and inferred original distance/gap boxes, all index bounds, and any wedge/phase consistency constraints. No claim is made that every such line actually contains an alias.

Once order and indices are known, (2) determines L_i; (1) determines g,d; and A_i/(k_iL_i) recovers each complex h_i. The original disjoint positive/negative wedge phase sectors select its unique signed wedge and phase whenever it belongs to the native prior. The leading DC gives b. Modal speeds are already uniquely represented by their resolved sampled nodes within the width-seven-Hz native interval, which is shorter than the twenty-Hz sampling alias period.

Thus, conditional on resolved distinct modal nodes and access to the formal coefficients, generic physical label ambiguity is absent. This excludes a potential permutation explanation for generic degree in the leading model. It does not exclude finite-wedge aliases or an alternative nonaxial system producing the same exact record.

## Audit and provenance

The coefficient formulas and index inversion are the audited results in vector_inverse.md. The affine-line label separation proof is derived above. No numerical parameter search, new optical approximation, literature novelty assertion, or change to the full problem's unknowns is used. Its restriction to formal leading coefficients is essential. An independent audit passes the proof; see Section 9 of physical_domain_contraction_audit.md.
