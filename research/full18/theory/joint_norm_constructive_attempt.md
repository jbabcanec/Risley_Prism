# A joint norm circuit for three prisms: bounded constructive attempt

## Stopping verdict

The one-rotor implicit relation does generalize to an exact, compact, jointly constrained norm circuit for the full three-prism model. A 32-dimensional algebra, a common-kernel test, and at most 33 scalar norm evaluations encode both observed coordinates on the same algebraic ray branch. A one-dimensional kernel also reconstructs that branch by linear algebra.

This **does not meet the requested success criterion** of materially reducing the remaining global optical solve. The matrix coefficients still depend nonlinearly on all fourteen optical coordinates. The original four affine geometry coordinates, or their already established profile, also remain. No data-only polynomial eigenvalue problem for the three speeds, no smaller physical-invariant inverse, and no tractable all-branches recovery follows from this construction.

This is the stopping point of one bounded constructive attempt. It is not a claim that a different nonlinear inverse is impossible, and it does not start another sequence of obstructions.

## 1. The exact rank-32 coefficient algebra

Use the five-root chart and combined spatial numerator from `reduced_algebraic_inverse_addendum.md`. Its optical coordinates are the three speeds, three signed wedge amplitudes, three phases, two beam slopes, and the three index coordinates h,c_2,c_3. The original indices reconstruct bijectively from this chart and its coupled prior. Thus there are still fourteen optical coordinates.

At each original time k/20, write the five successive roots as

r_1=E_1, r_2=H_2, r_3=E_2, r_4=H_3, r_5=E_3,

with triangular equations

r_i^2 = R_i(r_1,...,r_(i-1)).                       (1)

Each R_i is a polynomial in earlier roots and the instantaneous optical inputs. The rotor slopes are exact Laurent functions of their sampled nodes:

u_(j,x)(k) = (a_j/2)[alpha_j z_j^k+alpha_j^(-1)z_j^(-k)],
u_(j,y)(k) = (a_j/2i)[alpha_j z_j^k-alpha_j^(-1)z_j^(-k)],

where a_j is the signed wedge tangent, alpha_j its phase exponential, and z_j its sampled speed exponential. Their real physical charts and all native bounds are unchanged. In particular, D_j=1+a_j^2 is independent of k.

Over the coefficient ring B, form

mathcal A = B[r_1,...,r_5]/(r_i^2-R_i : i=1,...,5).

This is free of rank 32 over B, with basis

e_epsilon = r_1^(epsilon_1)...r_5^(epsilon_5),
epsilon in {0,1}^5.                                (2)

The freeness is exact even at degenerate specializations: repeatedly dividing by a monic quadratic gives a free rank-two extension at each step. No nonzero-discriminant or generic-branch premise is used.

For a in mathcal A, let M_a be its multiplication matrix in this basis. All M_a commute, and M_(ab)=M_a M_b. They can be built without expanding a large resultant. At one extension mathcal A_i=mathcal A_(i-1) plus r_i mathcal A_(i-1), write a=a_0+a_1 r_i. Then

M_a = [[M_(a_0), M_(a_1) M_(R_i)],
       [M_(a_1), M_(a_0)]].                         (3)

This is a recursive block construction. It is not a tensor product of five independent scalar two-by-two companions: later R_i depend on earlier roots. The dependence is retained in M_(R_i).

The field norm, when that terminology applies, and more generally the algebra determinant are computed by

N(a)=det M_a.

Successive quadratic norms give the same quantity: at the last extension replace a_0+a_1r_i by a_0^2-a_1^2 R_i, then continue downward. This remains an algebraic determinant identity at nonreduced fibers.

## 2. Why the two outputs need a joint test

Let the exact position recurrence give the final screen coordinates

F_x=A_x/T,    F_y=A_y/T,

where A_x,A_y,T belong to mathcal A. On the unique physical ray, T>0 by the original branch and traversal guards. For measured values x,y, put

g_x=A_x-xT,    g_y=A_y-yT.                           (4)

The two separate scalar equations N(g_x)=N(g_y)=0 are insufficient: one algebraic branch can fit x and a different branch can fit y. They do not certify a common ray.

At a fixed numerical specialization of the shared parameters, work over C. Then

ker M_(g_x) intersection ker M_(g_y) != {0}          (5)

if and only if there is a common complex tower point satisfying g_x=g_y=0.

This holds for nonreduced fibers as well. If the ideal (g_x,g_y) is the whole algebra, a linear combination of its two multiplication operators is the identity, so the common kernel is zero. Otherwise choose a maximal local component where both elements lie in the maximal ideal. A nonzero vector in that component's socle is annihilated by the maximal ideal, hence by both elements. Conversely a proper ideal has a common point over C. Therefore the stacked 64-by-32 matrix in (5) exactly encodes common algebraic-branch compatibility.

There is an equivalent compact determinant test. Define

D(lambda)=det[M_(g_x)+lambda M_(g_y)].              (6)

This has degree at most 32. In a simultaneous triangularization, or in the decomposition into local algebras, it factors as

D(lambda)=product_p [g_x(p)+lambda g_y(p)]^(m_p),

where p runs over the complex tower points and m_p is their local multiplicity. A product of polynomials over C is identically zero exactly when one factor is identically zero. Thus

D(lambda) identically zero
    if and only if some common tower point has g_x=g_y=0. (7)

Equivalently, all at most 33 coefficients vanish. Alternatively, evaluate (6) at any 33 distinct fixed rational lambda values and require every result to vanish. This avoids treating the two outputs as unrelated norms and avoids enumerating all maximal minors of the stacked matrix.

## 3. Denominators and physical branches cannot be discarded

A point with T=0 and A_x=A_y=0 satisfies (4) for every observed pair. Such a point is not a screen trace. The raw common-kernel or norm test must therefore be localized at T before it is used as a necessary-and-sufficient algebraic screen test.

This localization has a finite-dimensional linear-algebra implementation at every fixed shared-parameter trial. Let M_T act on the 32-dimensional specialized algebra. Its stable image

V_T = image(M_T^32)                                 (8)

is the direct sum of the local components where T(p) is nonzero. On components where T(p)=0, multiplication by T is nilpotent, and power 32 kills them. On the other components it is invertible. Thus V_T represents the localized algebra mathcal A[T^(-1)] and has dimension r<=32. Its algebra identity is the appropriate idempotent component of 1, which need not be the original full-algebra vector 1.

Restrict the multiplication matrices to V_T and put

U_x=(M_T|_(V_T))^(-1)(M_(A_x)|_(V_T)),
U_y=(M_T|_(V_T))^(-1)(M_(A_y)|_(V_T)).               (9)

Now the common kernel of U_x-xI and U_y-yI is nonzero exactly when some common complex tower point has T!=0 and the stated screen coordinates. The determinant of

(U_x-xI)+lambda(U_y-yI)

has degree at most r, so the same at-most-33-evaluation test applies. If V_T is zero-dimensional, there is no allowed algebraic point; the empty determinant is 1, correctly rejecting compatibility.

No unphysical root is thereby accepted as a physical solution. Positive H and E roots, all forward-normal and axial conditions, every strict traversal inequality, and the original bounds still have to be checked at a recovered common point. Localization only removes denominator-zero components. As the shared parameters vary, the dimension and basis of V_T can change; the restricted and inverse matrices are rational on rank/pivot strata, not one globally polynomial fixed-size matrix family. Any global symbolic use has to retain these strata rather than silently assume one invertible matrix throughout the prior.

There is also an exact ambient-matrix version requiring no chosen localization basis. Put G=[M_(g_x);M_(g_y)]. A common point surviving T!=0 exists exactly when

rank([G;M_T^32]) > rank(G).                          (9a)

Indeed, this rank increase is equivalent to M_T^32 acting nontrivially on ker G. The common kernel is invariant under multiplication. Its local summands with T(p)=0 are killed by that power, while multiplication is invertible on its surviving local summands. This is a polynomial rank-stratum criterion at every specialization, still followed by the original physical-point checks.

## 4. A conditional branch-reconstruction step

Suppose the localized common kernel has dimension one and v is a nonzero vector spanning it. Every root multiplication matrix preserves that line, because it commutes with U_x and U_y. Therefore

M_(r_i)v=lambda_i v.

For any nonzero coordinate v_j,

lambda_i=(M_(r_i)v)_j/v_j.                          (10)

The lambda_i satisfy the triangular root equations and the two observations. They recover the common algebraic ray point, which can then be checked against all the physical inequalities. For real trial coefficients and real observations, a one-dimensional complex kernel has a real spanning vector and the recovered lambda_i are real. Physical admissibility still needs its explicit sign checks.

Kernel dimension greater than one is not a rejection criterion. It can mean several common algebraic points or multiplicity at one point. The commuting operators restricted to that common kernel retain the joint-point information; one must preserve or resolve it rather than discard the parameter trial. No generic one-dimensional-kernel assertion is made on the full prior.

Equation (10) reconstructs latent ray variables **conditional on the shared parameter trial**. It is not a reconstruction of those shared physical parameters from data. Indeed, for a physical trial, the direct five-root forward trace is already simpler than building these matrices.

## 5. Why this does not yet extract the three speeds

For each of the 200 measured times, the construction yields a compact exact algebraic compatibility condition. It keeps its coefficients constrained by the physical parameters; no arbitrary free 25-slot analogue is substituted for the full optical family.

However, the coefficient matrices in (3), (6), and (9) still contain:

- all three rotor nodes, phases, and signed amplitudes;
- the two beam slopes and three index-chart coordinates;
- the four affine geometry coordinates through the spatial numerator, unless the previously proved geometry elimination is used.

Their dependence is nonlinear and shared across samples. The observed x_k,y_k enter affinely in a per-sample pencil, but the unknown speeds are not eigenvalues of a known data matrix. The pencil itself remains unknown until the other physical coordinates are supplied or solved. Calling it a polynomial-eigenvalue problem does not eliminate those coefficients.

Nor can the sample-specific kernel vectors be propagated by an assumed fixed finite linear shift matrix. The separately audited shift-module obstruction shows why such a step would introduce a false model. Retaining all these vectors explicitly instead adds latent variables and their nonlinear parameter constraints; it does not produce the requested smaller global solve.

There are two available eliminations already established in earlier work:

1. Exact-data source anchoring removes the two source offsets and leaves two distance variables.
2. Exact affine-fiber profiling eliminates the geometry block while retaining its strict and rank-degenerate cases.

The joint norm circuit supplies no further proved elimination of the fourteen optical variables. Constraining its coefficients to the physical family is necessary for correctness, but doing that alone just retains the original nonlinear inverse in another representation.

## 6. What has and has not been obtained

Proved in this attempt:

- A recursively factorized rank-32 algebra for the exact five-root three-prism ray lift.
- A common-branch two-coordinate algebraic compatibility test, including nonreduced specializations.
- An at-most-33-evaluation joint norm encoding.
- A denominator-localization construction and conditional one-dimensional-kernel ray reconstruction.

Not obtained:

- A data-derived elimination order reducing the shared optical unknowns.
- A known-coefficient polynomial pencil whose roots give the three speeds.
- A small identifiable nonlinear coefficient space with a proved reconstruction map from the 200 samples.
- A practical complete full18 inverse or a new global conditioning guarantee.

The honest answer is therefore **no constructive breakthrough in the missing inverse from this pass**. The exact constrained norm representation exists, but the essential recovery step remains unsolved. No numerical campaign, generic CAD reformulation, or published-novelty claim is used to fill that gap.

## Audit status

An independent mathematical audit passed the rank-32 algebra, nonreduced-fiber common-kernel criterion, joint norm equivalence, denominator localization, ambient rank alternative, and conditional root reconstruction. It also verified the rank/pivot-stratum qualification and the stopping verdict: this is not a reduction of the shared optical inverse. The separate report is `joint_norm_independent_audit.md`. No numerical campaign or proof-assistant formalization was used.
