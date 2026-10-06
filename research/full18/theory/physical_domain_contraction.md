# Reverse-order contraction and the global finite-fiber theorem

## Result and scope

The original strict physical full18 prior is contractible. The same is true of its interior and of the exact compact weak closure. A constructive contraction scales the signed wedge slopes to zero in reverse physical order, prism 3, then 2, then 1; it then contracts the remaining box coordinates at zero wedge.

Consequently the existing full18 rank witness applies generically throughout the whole physical interior, not merely on an unspecified component containing the witness. Outside a compact observation exceptional set of dimension at most 17, every exact 200-sample physical fiber is finite. Every positive-dimensional physical exact ambiguity is nongeneric.

This does **not** prove generic global uniqueness, low algebraic degree, rational reconstruction, or an efficient full18 inverse. A separate dominant-collision exclusion remains necessary. Section 5 identifies that finite200 obstruction precisely; it is not replaced by a continuous-trace argument.

All original bounds, signed rotors, indices, distances, beam angles, source offsets, and all 200 times t=k/20 are retained. No numerical campaign or new calibration assumption is used.

## 1. Exact convexity of a one-prism physical slice

Fix one sample and the incoming external unit direction (X,z), z>0, transverse entrance position p, refractive index n>1, and vertex-to-next-flat distance ell>0. Write

H=sqrt(n^2-|X|^2), nu=n^2-1, P(u)=H-u dot X,
D(u)=1+|u|^2, Delta(u)=P(u)^2-nu D(u).

On the transmitted branch the strict no-critical-refraction and incoming-normal conditions are equivalently

f(u):=H-u dot X-sqrt(nu)*sqrt(1+|u|^2)>0.             (1)

Indeed, f>0 implies P>0 and Delta>0; conversely P>0 and Delta>0 imply (1). The function sqrt(1+|u|^2) is convex, so f is concave. Also

f(0)=sqrt(nu+z^2)-sqrt(nu)>0.                        (2)

Thus if u is physically allowed, every lambda u with 0<=lambda<=1 is allowed for direction refraction. More explicitly,

f(lambda u)>=(1-lambda)f(0)+lambda f(u)>0.           (3)

No assertion that Delta itself is concave is required.

The exit position and two traversal conditions are

p_exit=p+X(3+u dot p)/P(u),
A(u)=3+u dot p>0,
Bflight(u)=ell-u dot p_exit>0.

Since P(u)>0, the second condition is equivalent to the affine inequality

C(u):=ell H-u dot [H p+(ell+3)X]>0.                 (4)

Both A and C are affine, and A(0)=3, C(0)=ell H>0. Hence they remain strictly positive along the whole contraction segment.

For completeness, the transmitted axial component also stays positive. Put R=sqrt(Delta), c=nu/[P+R]. Then

Z=H-c,
0<c<=sqrt(nu/D)<=sqrt(nu),
Z>=sqrt(nu+z^2)-sqrt(nu)>0.                        (5)

The bound follows from (P-R)(P+R)=D nu and P-R<=sqrt(D nu). This is the same exact guard as in global_boundary_continuation.md. It applies to the nonnegative-root weak branch as well.

Therefore the complete physically ordered admissible slope set for a fixed incoming state is the intersection of the convex superlevel set (1) and the two affine halfspaces (4) and A>0. It contains the flat slope 0 and is convex. Its intersection with a fixed rotor radial line contains the segment from any allowed slope to zero.

The weak version replaces > by >=. A weakly allowed u contracts to a strictly allowed current prism for every lambda<1, because the values at zero in (2) and (4) are strictly positive.

## 2. The global reverse-order homotopy

Use the original bijective algebraic chart from global_inverse_completeness.md. Its three signed wedge coordinates are r_j=tan(a_j); each instantaneous slope is u_j(k)=r_j e_j(k), where e_j(k) is a unit rotor direction. The other fifteen coordinates are unchanged in the first part of the homotopy.

First replace r_3 by lambda r_3, lambda decreasing from 1 to 0. Every incoming state at prism 3 is fixed. Section 1 proves all optical and traversal constraints at all 200 samples simultaneously. The screen is the next flat plane, with ell=d>0.

Next keep r_3=0 and replace r_2 by lambda r_2. The incoming state at prism 2 is fixed. Section 1 preserves propagation to the third entrance plane, with ell=g>0. The already flat third prism is always physically valid: a flat glass slab of thickness 3 preserves the incoming external direction, has positive internal axial travel 3, and has external axial flight d>0 to the flat screen. Its transverse entrance position may change without creating an obstruction.

Finally keep r_2=r_3=0 and replace r_1 by lambda r_1. Section 1 preserves the first prism and its flight g to the second entrance. Both later flat slabs and their positive fixed gaps remain valid.

This operation uses one common scalar lambda per prism for all 200 sampled slopes. Scaling a signed r_j toward zero stays in its original wedge interval; no speed or phase changes and no rotor relabeling occur. In native angular coordinates the same continuous path is a_j(lambda)=atan(lambda tan(a_j)), with the specified degree conversion.

Concatenating these three paths continuously gives a strong deformation retraction onto the zero-wedge set

Z0={r_1=r_2=r_3=0, all other fifteen coordinates in their original prior intervals}. (6)

Every point of Z0 is fixed throughout this retraction. All three zero-wedge elements remain actual index-dependent glass slabs; none is deleted.

At zero wedge every allowed index, beam direction, offset, speed, phase, gap, and screen distance is physical. The remaining fifteen-coordinate set is exactly a product box, and its ordinary straight contraction to a chosen interior flat point stays physical. This proves:

- P, the strict physical set including allowed prior faces, is contractible;
- O=P intersect int(prior), the full eighteen-dimensional physical interior, is contractible;
- W, the compact weak physical set/closure from global_boundary_continuation.md, is contractible by the same construction with weak inequalities.

For O use open prior intervals throughout; zero wedge is in the interior of its signed interval. For W, the positive axial bound (5) keeps the weak ray trace well-defined, even at critical starting points. The homotopy does not assert that every intermediate point of a weak contraction is already fully strict: an earlier, not-yet-contracted prism can remain critical until its turn.

This proof is specific to positive index contrast, positive gap/distances, flat entrances, flat screen, and reverse physical order. Simultaneously shrinking all wedges is unnecessary and is not claimed to preserve the constraints.

## 3. Generic full rank on the entire physical interior

The exact map

F:O -> R^400

is real analytic and semialgebraic in the algebraic chart. Strict roots and the positive denominators give analyticity; the exact 200-time rotor chart and root equations give semialgebraicity. The native-angle chart is an analytic diffeomorphism, so full rank transfers to native coordinates.

The existing vector_inverse.md witness supplies an 18-by-18 coordinate minor m of DF that is nonzero at an actual admissible small-wedge physical point. Fix that one minor. Section 2 proves O connected. The identity theorem for real-analytic functions therefore implies that m cannot vanish on an open subset of O.

Because its zero set is semialgebraic, empty interior implies

dim {theta in O:m(theta)=0}<=17.                  (7)

In particular the entire rank-deficient locus {rank DF<18} has dimension at most 17. There are no hidden full-dimensional physical components on which the original rank witness has no force. This is the specific component gap resolved by the contraction theorem.

The result remains generic rank, not everywhere rank. Zero-wedge fibers, the centered axial first-prism gauge, and other singular configurations are retained in the exceptional set.

## 4. A full-prior generic finite-fiber consequence

Let Zm={theta in O:m(theta)=0}, and use the already proved compact continuous extension Fbar:W->R^400. Define the compact parameter exceptional set

B=(W\O) union closure_W(Zm),
E=Fbar(B).                                         (8)

The boundary dimension theorem in global_boundary_continuation.md, (7), and semialgebraic closure give dim B<=17. Thus E is compact semialgebraic with dim E<=17.

For every y in Fbar(W)\E, every weak preimage belongs to O and has m!=0. The selected eighteen output coordinates are a local analytic diffeomorphism at each preimage. Therefore the full-record fiber is locally discrete. It is also a closed subset of compact W. A compact discrete fiber is finite: an infinite compact fiber would have an accumulation point in that same fiber, contradicting local isolation there.

Hence

Fbar^(-1)(y) is a finite set of strictly physical interior systems
for every y in Fbar(W)\E.                           (9)

The observation image has dimension exactly 18 because of the rank witness. Thus the excluded observation set really is lower-dimensional in the model image. The bad parameter preimage is lower-dimensional as well: on O\Zm the selected-output inverse-function charts show that the inverse image of an at-most-17-dimensional observation set has dimension at most 17; Zm and the boundary already do.

This is an unconditional whole-prior generic finite-fiber statement, with a deliberately enlarged exceptional set based on one fixed minor. No concrete value for the number of solutions is obtained. A finite fiber can have multiple physically distinct elements. No useful bound on their number or the cost of finding them follows from (9).

## 5. The exact remaining dominant-collision lemma

The following identifies a concrete finite200 proof target for generic physical uniqueness. It does not assert that the target has been proved.

Set U=O\Zm and J(theta)=DF(theta). Consider the off-diagonal collision set

C={(theta,theta') in U x U: theta!=theta', F(theta)=F(theta')}.

At every point of C, each J has rank 18. The collision Jacobian is the 400-by-36 matrix

[J(theta), -J(theta')].                             (10)

If (10) has rank at least 19 at every collision outside a set of dimension at most 17, then dim C<=17: the rank-19 points locally lie in the zero set of nineteen independent equations in 36 variables, while the remaining exceptional set already has that dimension bound. Projecting C to either parameter factor preserves the at-most-17 bound, and together with Section 4 this proves generic physical uniqueness.

Conversely, a dominant family of generic physical aliases forces an eighteen-dimensional stratum of collisions on which

rank[J(theta), -J(theta')]=18,
image J(theta)=image J(theta').                      (11)

Here is the argument without a generic-map shortcut. On any collision stratum, a tangent vector with dtheta=0 must satisfy J(theta') dtheta'=0, hence dtheta'=0. The first projection therefore has injective differential. Any stratum whose first projection has dimension 18 has dimension exactly 18 and projects locally diffeomorphically; it is a local branch theta'=T(theta). Differentiating F(T(theta))=F(theta) makes the two eighteen-dimensional Jacobian images equal. Distinct regular points with the same tangent observation sheet are therefore the only possible source of a full-dimensional generic alias family.

A sufficient sharp remaining task is to prove that the finite200 set

{theta!=theta', F(theta)=F(theta'), rank[J(theta),-J(theta')]=18}

has dimension at most 17 after the original physical guards. It is a model-specific tangent-sheet coincidence question, not the already settled rank18 question at one parameter. Neither domain contractibility nor the local-rank witness answers it. A rank19 computation at an arbitrary pair that is not on the collision set does not answer it either.

This formulation uses exactly the original 400 observed coordinates at k/20. It makes no claim that equality of those values implies equality of continuous traces. Infinite-time Fourier, branch-divisor, or torus-identifiability arguments need a separate finite200 separation theorem before they can establish this lemma.

## 6. Why no low-degree or rational inverse is concluded

On a connected analytic semialgebraic graph, the Zariski closure of the graph is irreducible, and the rank18 result makes its projection to the Zariski observation image generically finite. This defines an algebraic inverse degree after the graph/field chart is specified. It does not bound that degree by a small number.

The physical fiber count and this complex algebraic degree are different quantities. Positive root selection, native index positivity, phase/speed intervals, and all traversal inequalities can reject algebraic conjugates. Even a unique physical branch need not be rational in the observations: the elementary positive map x->x^2 has one positive inverse branch but inverse sqrt(y).

Likewise, the five-root instantaneous tower bounds forward branch multiplicity conditional on shared parameters; it does not bound the number of shared-parameter inverse solutions. The exponential shifted-root result in exact_shift_module_obstruction.md blocks interpreting it as one fixed finite linear state. None of the three facts supplies the missing bound on the degree of the global optical inverse.

The progress here is therefore global control of the physical domain and generic finiteness over the full prior, plus a precise finite-record collision condition that must still be excluded to obtain generic uniqueness. The useful full18 reconstruction and low-degree elimination problem remain open in this pass.

## Sources and audit

The optical formulas, full18 rank witness, and exact closure are the audited corpus results identified above. The contraction and its one-prism convexity proof are derived here. Standard semialgebraic dimension/stratification facts are as in the primary mathematical background already used in that corpus: Saugata Basu, Algorithms in Real Algebraic Geometry: A Survey, https://www.math.purdue.edu/~sbasu/raag_survey2011.pdf, and Michel Coste, An Introduction to Semialgebraic Geometry, author publication list https://perso.univ-rennes1.fr/michel.coste/articles.html. Neither is being cited as an optical contraction or global uniqueness theorem.

An independent mathematical audit is in physical_domain_contraction_audit.md. No literature-novelty assertion, numerical sweep, or proof-assistant formalization is claimed.
