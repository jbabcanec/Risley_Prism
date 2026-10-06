# Reduced algebraic compiler and exact affine-fiber inverse

## Scope and status

This addendum gives three exact structural improvements for the original full18, three-prism, 200-sample vector-Snell inverse. It retains the original native bounds, sampled-time physical branch, all signed rotors, singular fibers, and symbolic coordinatewise noise epsilon. It does not alter the audited existing memos.

1. A bijective change of the three refractive-index coordinates removes one square root at every sample without increasing dimension. A combined spatial recurrence reduces the root-free instantaneous degree bound from 2368 to 896, with at most 243 terminal sign tests per primitive atom instead of 729.
2. The source-offset-to-screen map is globally invertible at every physically admissible optical point. One exact screen sample therefore eliminates both source offsets without a generic-rank assumption. The remaining exact-data geometry problem has only two variables and Helly order three; its rank-zero, rank-one, and rank-two distance fibers have explicit complete descriptions.
3. For positive symbolic noise, an exact minimal-positive-circuit compiler eliminates the four affine geometry variables and retains every strict endpoint. It gives an explicit maximum-of-ratios formula for the optical-pointwise consistency threshold. This replaces unspecified small linear feasibility elimination with determinants and signs. Positive noise still requires four geometric degrees of freedom in the anchor representation.

These are genuine exact representation reductions, not a complete tractability theorem. They do not supply a small all-branches optical candidate cover, a positive exterior residual gap, or uniform conditioning near critical optical boundaries.

## 1. An index chart with five roots

Use the original rational angular/rotor chart, and let t=(s_x,s_y), Q=1+|t|^2. Replace the three independent coordinates (n_1,n_2,n_3) by

h=sqrt(n_1^2 Q-|t|^2)>0,
c_2=(n_2^2-1)Q,
c_3=(n_3^2-1)Q.                                      (1)

All other optical and geometric coordinates remain. There are still fourteen optical coordinates and eighteen total coordinates. The exact index priors become

(13/10)^2 Q-|t|^2 <= h^2 <= (9/5)^2 Q-|t|^2, h>0,
((13/10)^2-1)Q <= c_j <= ((9/5)^2-1)Q, j=2,3.       (2)

This is a coupled semialgebraic prior, not a product box. In particular h>1 and c_j>0 throughout it. The inverse is

n_1=sqrt((h^2+|t|^2)/Q),
n_j=sqrt(1+c_j/Q), j=2,3.                           (3)

Thus (1) is a bijection of the entire original index/source prior with (2). There is no new unknown, fixed hidden index, omitted endpoint, or choice of sign. Native index ranges cannot be obtained by projecting h or c alone: one must project/optimize the algebraic expressions (3), or retain n as an auxiliary output linked by (3).

Set X_0=t, Z_0=L_0=1 and D_j=1+|u_j|^2. For the first prism set H_1=h; for the others use

H_j=sqrt(Z_(j-1)^2+c_j L_(j-1)^2), j=2,3.

Set c_1=h^2-1 only as an expression, not another independent coordinate. For every j define

P_j=H_j-u_j dot X_(j-1),
E_j=sqrt(P_j^2-D_j c_j L_(j-1)^2),
Z_j=H_j D_j-P_j+E_j,
X_j=X_(j-1)D_j+u_j(P_j-E_j),
L_j=L_(j-1)D_j.                                     (4)

In the first-prism expression L_0=1. Require E_j>0, P_j>0 and Z_j>0, and retain all traversal inequalities. Positive H_j is selected at the indicated square root. Its radicand is automatically positive for j=2,3 under (2), but it is harmless to keep the explicit positive-radicand guard.

Proof of equivalence. The existing scaled direction invariant is |X_j|^2+Z_j^2=Q L_j^2. Consequently

n_j^2 Q L_(j-1)^2-|X_(j-1)|^2
 = Z_(j-1)^2+(n_j^2-1)Q L_(j-1)^2.

For j=1 the same expression is exactly h^2. Also (n_1^2-1)Q=h^2-1. Substitution into the existing six-root recurrence proves (4), and induction preserves the direction invariant. The only sequential roots are now

E_1, H_2, E_2, H_3, E_3.

Conversely, reconstruct the unique positive indices by (3). The same equations become the original physical recurrence with its original positive-root, forward-normal and forward-axial tests. Hence no branch is added or removed.

## 2. Combined spatial recurrence and degree bound 896

The two-stage intersection formula can be combined before polynomial numerator propagation. If p=A/T, T>0, then one prism maps it to A_next/T_next, where

T_next=Z_j P_j T,
A_next=Z_j P_j A
       +(Z_j X_(j-1)-H_j X_j)(u_j dot A)
       +3[Z_j X_(j-1)-(u_j dot X_(j-1))X_j]T
       +ell_j P_j X_j T,                             (5)

with ell_1=ell_2=g, ell_3=d. Initially A=b+6t and T=1. All vectors in (5) have two components.

For verification, the exit-point formula gives

p_exit=p+X_(j-1)(3+u_j dot p)/P_j.

The remaining outgoing axial distance has numerator

ell_j P_j-H_j(u_j dot p)-3(u_j dot X_(j-1)).

Multiplying the final two-segment position by Z_j P_j gives (5). This cancellation is exact; it does not divide by a conjugate or by a potentially vanishing coefficient.

Assign weight one to h,t,b,u_j,g,d,y,epsilon and weight two to c_2,c_3. The fixed constants have weight zero. The five root weights are respectively 2,3,4,5,6. Induction in (4) gives

weight(L_j)<=2j,
weight(X_j),weight(Z_j)<=2j+1,
weight(H_j)<=2j-1,
weight(P_j),weight(E_j)<=2j.                          (6)

Every radicand has weight at most twice its root weight. The first root follows from P_1=h-u_1 dot t and c_1=h^2-1. The later bounds follow from H_j^2=Z_(j-1)^2+c_j L_(j-1)^2.

Suppose weight(A)<=a and weight(T)<=b with a=b+1. In (5), the first three terms have weight at most a+4j+1, and the last at most b+4j+2=a+4j+1. Therefore numerator and denominator budgets both increase by 4j+1, preserving a=b+1. Starting with a=1,b=0 gives after three prisms

weight(A_3)<=28, weight(T_3)<=27.                    (7)

Every interval-observation atom A_(3,l)-(y_l +/- epsilon)T_3 has weight at most 28. All branch/traversal atoms have no greater weight. For example the two traversal tests are

3T+u_j dot A>0,
ell_j P_j T-H_j(u_j dot A)-3(u_j dot X_(j-1))T>0,

with the already positive denominators cleared.

Apply the exact radical-sign table in explicit_six_root_compiler.md to the five roots. At one level E=A+B sqrt(R), the three tests A,B,A^2-B^2R determine its sign, with all positive-radicand guards retained. Root reduction never increases weight, and eliminating one level at most doubles it. Thus the root-free instantaneous ordinary degree is at most

28*2^5=896.                                         (8)

There are at most 3^5=243 terminal sign-polynomial tests per primitive atom in the shared sign-circuit representation. Keeping the old conservative allowance of 24 primitive atoms gives 5832 terminal occurrences before deduplication. This is a circuit-leaf count, not a promise about fully expanded Boolean syntax. Per-sample squared Euclidean residuals have weight at most 56 and terminal degree at most 1792.

After substituting the same rational rotor formula as the existing compiler, a terminal instantaneous degree-D polynomial has sample-k chart degree at most D(6k+7). For K=200, (8) gives the conservative maximum 896*1201=1,076,096. The bound remains very large. The point is a proved reduction, not an assertion that dense expansion or final CAD is practical. The new index prior has fixed degree two and does not change the O(K) substituted-degree conclusion. Exact finite input and coefficient-height caveats are those of the existing completeness theorem.

## 3. Globally invertible source transport

For this section use actual ray slopes, independently of the scaled numerator notation. At one prism let

a=X_in/H,
v=X_out/Z,
P=H-u dot X_in>0,
R=Z-u dot X_out>0,
H>0, Z>0.

The incoming-to-exit and exit-to-next-flat intersection identities are

(I-a u^T)p_exit=p+3a,
p_next=(I-v u^T)p_exit+ell v.

The first matrix is invertible since 1-u dot a=P/H>0. Hence

p_next=A p+3A a+ell v,
A=(I-v u^T)(I-a u^T)^(-1).                          (9)

The two-by-two matrix determinant lemma gives

det A=(1-u dot v)/(1-u dot a)=H R/(Z P)>0.           (10)

Both factors in (9) are therefore invertible. An explicit inverse transport is

p=A^(-1)(p_next-ell v)-3a,
A^(-1)=I+(v-a)u^T/(1-u dot v).                       (11)

At sample k the three-prism composition is

F_k(x,b,w)=T_k(x)b+C_k(x)w+d_k(x),
w=(g,d),
T_k=A_(3,k)A_(2,k)A_(1,k),                          (12)

det T_k=product_j H_(j,k)R_(j,k)/(Z_(j,k)P_(j,k))>0.

The initial b-to-first-flat propagation is a translation by 6t and introduces no additional matrix. Formula (12) is the exact affine geometry decomposition with the additional global theorem det T_k>0.

No positive lower bound for det T_k on the open full physical prior is asserted. It can tend to zero as an outgoing-normal/critical boundary is approached, making (11) poorly conditioned. Exact invertibility at every physical point remains true. At a weak compactification boundary with R=0, this elimination cannot replace the unreduced graph; that boundary must be retained separately if used in a global closure argument.

## 4. Exact-data anchor elimination and all distance-rank strata

Assume epsilon=0 and fix any measured anchor k0. Then

b=T_(k0)^(-1)[y_(k0)-C_(k0)w-d_(k0)].                (13)

This holds at every physically admissible optical point, without choosing a nonzero minor or making a genericity assumption. Substitution into all other sample equations gives a system

D(x)w=z(x),                                         (14)

with 2(K-1) scalar rows, where sample block k is

D_k=C_k-T_k T_(k0)^(-1)C_(k0),
z_k=y_k-d_k-T_k T_(k0)^(-1)(y_(k0)-d_(k0)).

The exact g,d bounds, original source square |b_i|<=5 after (13), and every original strict traversal condition become weak or strict affine inequalities in w. None may be dropped. Denominators in (11) are physical positive quantities, so they can be cleared without branch-sign ambiguity.

Thus the exact geometry fiber is a mixed-strict polygon, possibly empty, lower-dimensional, or open along some faces, cut by (14). Finite Helly in R^2 gives an exact at-most-three-row feasibility characterization when equalities are represented by paired weak halfspaces. There are O(K^3) constant-size tests, improving the generic four-geometry-variable O(K^5) count for this exact-data case. A pulled-back row depends on its own sample and the common anchor, so a three-row test can involve up to four distinct sampled ray traces. This is a constant sample count, not a claim that only three traces occur.

There is also a more direct complete distance-rank atlas:

- Rank two: choose any two independent scalar rows of D. They determine a unique w by a two-by-two solve. Retain it iff every row of (14), every source/distance bound, and every strict traversal row is satisfied.
- Rank one: choose any nonzero scalar row. Require every remaining equation to be consistent with that row, equivalently rank[D|z]=rank D=1. The solutions form one affine line; intersect it with every original inequality. This gives an interval, point, ray, or empty set before bounded distance priors; the bounded prior makes any nonempty final interval bounded, with exact strict/weak endpoint flags.
- Rank zero: require every entry of D and z to vanish. Retain the full polygon given by the remaining affine constraints.

Selecting the first available pivot row or pair in a fixed lexicographic order gives disjoint symbolic branches if desired. Enumerating all choices instead gives a redundant but complete cover. All rank-deficient distance gauges survive. Once w is known or represented, recover b using (13). This does not reduce the optical dimension or prove that the remaining optical fiber is small.

## 5. Positive noise: exact anchor residual and a sound two-distance relaxation

For epsilon>0, equality (13) with the reported y_(k0) alone is not exact. Introduce an anchor error e in [-epsilon,epsilon]^2, with F_(k0)=y_(k0)+e. Then

b=T_(k0)^(-1)[y_(k0)+e-C_(k0)w-d_(k0)].             (15)

All remaining sample, prior and traversal constraints are affine in (w,e), four scalar variables. This is a bijective reparameterization of the original noisy affine geometry fiber, and its exact Helly order remains five. The error at the anchor is one shared vector across every transformed row; it must not be replaced by independent errors per row. A five-row test in these anchored coordinates can involve six sampled traces including the anchor. The unanchored circuit compiler in Section6 avoids that extra shared trace: each original geometry row involves at most one sampled trace, so a five-row circuit uses at most five.

A useful sound outer relaxation retains only w. After (15), put each weak row in the form

a_i^T w+v_i^T e<=b_i(epsilon).

Every feasible w necessarily satisfies

a_i^T w<=b_i(epsilon)+epsilon ||v_i||_1.             (16)

For a strict row the same necessary inequality is strict. Keeping it strict is valid because a_i^T w < b_i-v_i^Te <= b_i+epsilon||v_i||_1. Weakening it further is also a sound outer relaxation, but may leave boundary-only false candidates. Include exact distance bounds, the transformed source-bound rows, all traversal rows, and all observation rows. At epsilon=0 this is the exact two-distance problem. For epsilon>0 it can be strictly larger than the projection because the individual best values of e may disagree.

An infeasibility certificate for (16) excludes the optical point in the exact noisy problem. Its minimal positive circuits have support at most three. This is a lower-dimensional screening certificate; feasibility does not certify a physical noisy system and does not replace (15).

## 6. Explicit mixed-strict circuit elimination for symbolic noise

This section applies either to the original four-variable geometry representation or to a lower-dimensional affine fiber such as Section4. Let q in R^d, with d=4 for the full noisy problem, and collect every geometry-box, observation, and traversal row as

a_i(x)^T q <= beta_i(x,y)+epsilon delta_i,

marking the traversal rows as strict (<). Take delta_i=1 for every coordinatewise observation row and delta_i=0 for the geometry box and traversal rows. Optical branch predicates are handled separately. Positive denominator clearing may instead multiply both beta and delta by the same positive optical factor; all arguments still apply with delta_i>=0.

Define the nonnegative dependence cone

C_x={lambda>=0: sum_i lambda_i a_i(x)=0}.

A minimal positive circuit is a nonzero lambda in C_x whose support is inclusion-minimal. Normalize its entries to sum to one if convenient. Such a support has r<=d+1 rows, their row rank is r-1, and the positive null vector is unique up to scale. Conversely any support with these properties defines a circuit. In rank-degenerate cases smaller supports, including a single zero-normal row, are essential and must be included.

For a circuit define

B_lambda=sum_i lambda_i beta_i,
D_lambda=sum_i lambda_i delta_i>=0,
s_lambda=true iff some positive lambda_i belongs to a strict row.

### Mixed-strict feasibility theorem

The entire affine system is feasible if and only if every minimal positive circuit obeys

B_lambda+epsilon D_lambda >=0 if s_lambda=false,
B_lambda+epsilon D_lambda >0  if s_lambda=true.       (17)

Necessity follows by multiplying every row by its nonnegative multiplier and summing; a positively weighted strict row gives a strict sum.

For sufficiency first weaken all rows. Farkas' alternative says this closed system is feasible iff lambda^T b>=0 for every lambda in C_x, where b=beta+epsilon delta. The cone is pointed and generated by its extreme rays, exactly the minimal positive circuits, so the weak parts of (17) suffice.

Now let P be the nonempty weak feasible polyhedron. If P contains a point satisfying each strict row separately with strict slack, averaging these finitely many points gives one point satisfying all strict rows. If no strictly feasible point exists, some marked row i is equality throughout P. Linear-programming duality for minimizing a_i^Tq over P then gives mu>=0 with A^Tmu=-a_i and -b^Tmu=b_i. Equivalently lambda=mu+e_i belongs to C_x, has lambda_i>0 and b^Tlambda=0. Decompose lambda into its nonnegative circuit rays. Their b-pairings are nonnegative by weak feasibility, so every used circuit has zero pairing; at least one used circuit contains marked row i. That contradicts the strict condition in (17). This proves sufficiency. The finite LP optimum is attained on the nonempty polyhedron; in the present application the geometry box also makes P compact.

### Determinant compiler

Enumerate supports I of size r<=d+1. For each candidate rank r-1 choose r-1 independent coordinate columns. Signed (r-1)-by-(r-1) cofactors give a null vector for those columns. Verify that it annihilates all d columns and that all r entries have the same nonzero sign; orient them positive. This certifies exactly a positive circuit. For r=1 the condition is simply a_i=0 and lambda_i=1. Retain lower-rank cases through their smaller supports rather than dividing by a vanishing minor.

On each determinant-sign branch, B_lambda and D_lambda are sums of a cofactor times one row right-hand-side. Their degree in the affine-row coefficients is at most d+1 (at most five for the original geometry). No generic real QE is needed for the affine elimination. There are O(M^(d+1)) support tests for M=O(K) rows, and only constantly many coordinate-column choices. The actual optical coefficients remain the exact five-root/rational functions; this coefficient-degree statement is not a low total native-chart degree claim.

### Exact optical-pointwise noise threshold

Circuits with D_lambda=0 impose permanent feasibility conditions B_lambda>=0, made strict when s_lambda=true. If any of these fail, that optical point has no compatible geometry at any noise budget.

Otherwise every circuit with D_lambda>0 imposes

epsilon>=rho_lambda or epsilon>rho_lambda,
rho_lambda=-B_lambda/D_lambda,

with strictness exactly s_lambda. Together with epsilon>=0 define

rho=max(0, max_(D_lambda>0) rho_lambda).              (18)

If there are no positive-D circuits, interpret the inner maximum as -infinity. The compatible noise budgets at this fixed optical point are [rho,infinity) unless at least one strict positive-D circuit attains rho, in which case they are (rho,infinity). Permanent strict zero-D conditions have already been tested. This covers rho=0 and negative circuit ratios correctly. A strict circuit with a smaller threshold does not open the endpoint.

Equations (17)-(18) provide an exact finite maximum-of-ratios representation for the profiled consistency threshold and an exact endpoint flag. They also explain why replacing strict traversal by closure feasibility can give a wrong answer precisely at the first compatible noise level. Computing the global threshold still requires minimizing over all optical points and handling optical boundaries; (18) does not solve that nonlinear task.

## 7. What is improved and what remains

The five-root chart and degree896 compiler apply to the exact symbolic-noise all18 inverse, not only to a local small-wedge approximation. The globally nonsingular source map removes both offsets from every exact-data fiber without excluding rank-deficient hardware strata. The positive-circuit formula is an explicit affine quantifier elimination with all strict cases, including singular coefficient matrices and open threshold endpoints.

The new index chart is coupled; inverse source transport can become ill-conditioned at critical outgoing-normal boundaries; positive noise retains two anchor-error degrees of freedom; and minimal-circuit counts can still be large. None of these results proves a quantitatively useful all-branches optical cover or a manageable final decomposition at K=200. A claim that the missing practical inverse theorem is now closed would be false. These reductions can be combined with certified exterior exclusion, weak-boundary analysis, and rigorously bounded harmonic candidate generation, retaining every unresolved optical stratum in the complete backend.

## 8. Audit and bounded proof check

An independent mathematical audit verified the five-root chart, sharpened degree896 recurrence, source determinant/inverse identities, exact/noisy anchor distinction, and mixed-strict circuit/threshold theorem. A single symbolic proof check, with fully formal two-dimensional directions and plane slopes, verified the determinant identity, inverse-matrix identity, and equality of the combined spatial numerator with the two-segment intersection formula. No parameter sweep, reconstruction campaign, or practical runtime claim was used. This is a mathematical/symbolic audit, not proof-assistant verification or a literature-novelty claim.
