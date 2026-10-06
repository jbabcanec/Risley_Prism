# Independent audit: contractibility of the original physical domain

## Verdict and scope

**Pass.** For the original full eighteen-parameter, three-prism vector-Snell model, the fixed samples t_k=k/20, k=0,...,199, the original signed parameter intervals, and the sequential-traversal inequalities in the existing exact graph, the strict physical domain P, its prior-interior part O=P∩int(Pi), and the weak compactification W are all contractible.

The proof is exact and uses no small-wedge approximation or numerical campaign. It works by setting the signed wedge slopes to zero in reverse physical order 3,2,1, followed by contraction of the remaining fifteen box coordinates. Combined with the prior full18 rank witness, it removes the previously unresolved possibility of a separate open physical component on which every full-rank minor vanishes. It does **not** establish global injectivity, degree one, a one-sheet covering, or a useful conditioning constant.

Sources audited: `vector_inverse.md`, especially Sections 1–2 and 7; `global_boundary_continuation.md`, Sections 1–3; and the algebraic chart and exact sequential graph in `global_inverse_completeness.md`, Sections 1–2. The full18 witness is taken from its existing proof; the new audit concerns the global propagation of that witness through the contraction argument.

## 1. One-prism feasible slopes form a convex set

Fix an incoming external direction (X,z), with |X|²+z²=1 and z>0, an entrance position p, an index n>1, and an exit-vertex-to-next-flat separation ell>0. Put

H=√(n²−|X|²)=√(n²−1+z²),  nu=n²−1,
Pn(u)=H−u·X,  D(u)=1+|u|²,
Delta(u)=Pn(u)²−D(u)nu.

At this step the incoming state, n, and ell are held fixed. The forward incoming-normal and strict no-critical-refraction conditions are exactly

f(u):=H−u·X−√nu √(1+|u|²)>0.                         (1)

Indeed, (1) is equivalent to Pn>0 and Delta>0. The weak counterpart f(u)≥0 is equivalent to Pn>0 and Delta≥0 because nu>0. On the original weak branch, the existing bound Pn≥h_*>0 makes this equivalence applicable even though W was initially defined using Delta≥0.

The map u↦√(1+|u|²) is convex. Hence f is concave, its strict and weak superlevel sets are convex, and

f(0)=H−√nu=√(nu+z²)−√nu>0.                         (2)

The first traversal condition is

A(u)=3+u·p>0.                                      (3)

For the second, the entrance-to-exit intersection is

p_exit=p+X(3+u·p)/Pn.

A direct cancellation gives

Pn(ell−u·p_exit)
  =ell H−u·[Hp+(ell+3)X]
  =:Q(u).                                         (4)

Since Pn>0, the external forward-flight condition is precisely Q(u)>0. Both A and Q are affine functions of u, and

A(0)=3>0,  Q(0)=ell H>0.                           (5)

Thus the complete strict one-prism feasible-slope set is the intersection of the convex set f>0 and the two affine halfspaces A>0, Q>0. The corresponding weak set uses ≥0. In particular, if u is strictly feasible, every lambda u, 0≤lambda≤1, is strictly feasible. If u is weakly feasible, lambda u is weakly feasible for 0≤lambda≤1 and strictly feasible for every 0≤lambda<1. Explicitly,

f(lambda u)≥(1−lambda)f(0)+lambda f(u),
A(lambda u)=(1−lambda)3+lambda A(u),
Q(lambda u)=(1−lambda)ell H+lambda Q(u).

This is stronger than a merely infinitesimal inward derivative: the entire segment to zero is admissible for the fixed incoming state.

## 2. Positive outgoing axial direction is preserved, including at criticality

For any u satisfying the weak inequalities in Section 1, set

R=√Delta,  c=(Pn−R)/D,  B=X+c u,  Z=H−c.

Then |B|²+Z²=1 and Z−u·B=R. Moreover, 0≤R≤Pn, and

(Pn−R)²≤(Pn−R)(Pn+R)=D nu,

so 0<c≤√(nu/D)≤√nu. Therefore

Z≥√(nu+z²)−√nu>0.                                (6)

This bound remains strict when R=0. No division by the critical-normal quantity R is used. Combined with the original bounded beam and index intervals, it gives the uniform positive axial bounds already established in `global_boundary_continuation.md`.

Consequently each one-prism contraction remains on the selected transmitted branch. It does not pass through axial grazing or an undefined intersection denominator.

## 3. Why reverse physical order makes the global contraction valid

Write r_j=tan(a_j) in the algebraic chart, and write the instantaneous slope at any sample as

u_j(k)=r_j e_j(k),

where e_j(k) is the unit rotor direction determined by that prism's phase and speed. Replacing r_j by lambda r_j scales every sampled u_j(k) by the same lambda. All original signed wedge bounds are preserved; in native coordinates the change is a_j↦arctan(lambda tan(a_j)), with the appropriate degree conversion. Prior-interior wedge values remain in their interior intervals.

First scale r_3 to zero, keeping every other coordinate fixed. Prism 3's incoming direction and entrance position are fixed because prisms 1 and 2 have not changed. Section 1 applies simultaneously to all 200 samples.

Next scale r_2 to zero. Prism 2's incoming state remains fixed. Prism 3 is now flat. A flat prism accepts every incoming unit direction with positive axial component on this branch: Delta=z²>0, Pn=H>0, A=3>0, and its external-flight margin is ell>0. Its output external direction equals its input direction, although its index can change its internal lateral transport. Its admissibility is independent of the transverse entrance position. Thus moving prism 2's downstream ray cannot invalidate prism 3.

Finally scale r_1 to zero. Prisms 2 and 3 are flat, so exactly the same argument applies.

The dependence of transverse positions on earlier wedges creates no missing constraint here: during the contraction of the current prism its incoming position is fixed, and all downstream admissibility tests have become position-independent. This is the reason forward-order scaling would not be justified by the same argument.

The argument uses precisely the original graph's positive internal axial travel and positive exit-to-next-flat travel. Extra requirements such as finite apertures, collision avoidance between complete physical solids, or additional surface-intersection predicates are not part of this theorem and would need separate treatment if added to the model.

## 4. Continuous deformation and contraction of P, O, and W

A convenient explicit three-stage homotopy uses s∈[0,1] and the continuous clamp function [x]_0^1=max(0,min(1,x)):

r_1(s)=[3−3s]_0^1 r_1,
r_2(s)=[2−3s]_0^1 r_2,
r_3(s)=[1−3s]_0^1 r_3.                            (7)

All other coordinates are held fixed. Section 3 proves that (7) preserves P, preserves O, and preserves W. For W the earlier prisms may remain critical during initial stages; Section 2 keeps their incoming states well-defined, while the currently scaled prism satisfies the weak or strict inequalities from Section 1. No differentiability of sqrt(Delta) at a critical point is needed. The parameter homotopy itself is continuous and semialgebraic in chart coordinates.

Let Z_P={r_1=r_2=r_3=0}∩Pi and Z_O={r_1=r_2=r_3=0}∩int(Pi). Every point of either zero-wedge box is strictly physical: all three slabs have positive thickness 3, both gaps are positive, the screen separation is positive, and the incident axial direction is positive throughout the original beam box. Equation (7) fixes these zero-wedge points, so it is a strong deformation retraction of P and W onto Z_P, and of O onto Z_O.

Both zero-wedge sets are products of fifteen intervals, closed for Z_P and open for Z_O. Choose one interior basepoint, for example indices n_j=3/2, gap g=17/2, distance d=125, and all remaining coordinates zero. Straight-line interpolation of the fifteen chart coordinates to this basepoint stays in the relevant box and keeps all wedges zero. Concatenating it with (7) gives a contraction of P, O, and W to that same point. The contraction fixes the basepoint throughout.

The contraction is not a replacement for the separate hierarchical strictification proof that W=closure(O): the reverse-order path can leave an upstream critical constraint unchanged for an initial time interval. Density and boundary-dimension conclusions below use the already proved equality W=closure(O).

## 5. Consequence for the global analytic critical locus

Let F:O→R^400 be the exact sampled map. It is real analytic and semialgebraic in the algebraic chart: all radicals have strictly positive arguments on O, the relevant denominators are positive, and the sampled rotor expressions are rational functions with positive denominators.

By the prior full18 witness, there is an interior physical point theta_* and a fixed set I of eighteen original scalar output coordinates such that

m(theta)=det D(F_I)(theta),  m(theta_*)≠0.           (8)

The chart change preserves full rank. Because O is nonempty, open, and connected by Section 4, the real-analytic identity theorem implies that m cannot vanish on any nonempty open subset of O. Its zero set is semialgebraic, so it has dimension at most seventeen. Therefore both

K_I={theta∈O:m(theta)=0},
C={theta∈O:rank DF(theta)<18}

have dimension at most seventeen, since C⊆K_I. Full rank holds on an open dense subset of the entire O, rather than merely on the analytic component containing the witness. Taking relative closure in W preserves these semialgebraic dimension bounds.

This conclusion does not assert that every minor is nonzero generically. It identifies one fixed minor that is nontrivial globally and gives a convenient exceptional superset of the full-map critical locus. Producing an explicit named output selector I would require extracting one from the existing witness proof; existential choice is enough for the theorem above.

## 6. Generic finiteness of the complete physical exact fiber

Use the existing continuous semialgebraic extension Fbar:W→R^400 and W=closure(O). Define

E=Fbar((W\O)∪closure_W(C)).                        (9)

This is a compact semialgebraic set of dimension at most seventeen: the boundary W\O has that bound, Section 5 bounds closure_W(C), and semialgebraic images do not increase dimension. One may replace C by the larger fixed-minor locus K_I to obtain a possibly larger valid exceptional set.

If y∉E, the weak exact fiber Fbar^{-1}(y) is compact, lies wholly in O, and consists only of full-rank points. At each point some eighteen output coordinates give an invertible Jacobian, so the inverse function theorem makes that point isolated in the full-record fiber. A compact discrete fiber is finite. Because no point lies on W\O, this is exactly the entire physical fiber in P as well. The fiber can be empty.

This is a genuine generic statement relative to the eighteen-dimensional attainable model image. The rank witness makes dim Fbar(W)=18. The good records are relatively open, and are dense in Fbar(W): every nonempty open parameter subset contains a full-rank point; its local image has dimension eighteen and cannot lie in the dimension-at-most-seventeen E. Continuity and W=closure(O) then give density of F(O)\E in Fbar(W).

If one wants a parameter-generic formulation, F^{-1}(E) has dimension at most seventeen on O: on the full-rank locus this follows locally by the immersion charts, and the remaining critical locus already has dimension at most seventeen. Adding W\O retains the same bound.

## 7. Claims that do not follow

- Contractibility of the parameter domain does not force F or a selected square projection F_I to be injective.
- Removing critical and boundary values can leave several sheets; removing their preimages can disconnect even a contractible parameter domain.
- Signed degree one does not rule out additional oppositely oriented sheets.
- Generic finite full-record fibers can have more than one physical parameter tuple.
- The contraction passes through zero-wedge degeneracies, so it is not a contraction through full-rank systems and supplies no uniform inverse-Jacobian bound.
- W contractible does not mean the graph is differentiable at critical-refraction boundary points.
- None of these exact topological statements certifies useful finite-angle conditioning or a tractable all-branches numerical algorithm.

The proposed theorem is valid with these scope distinctions intact.

## 8. Follow-up audit of the finite200 collision target and algebraic degree

The final `physical_domain_contraction.md`, especially its Sections 5–6, passes the substantive audit.

For U=O\Zm, every Jacobian J has rank eighteen. Stratify the off-diagonal finite200 collision set into smooth semialgebraic strata. A tangent vector (v,v') satisfies J(theta)v=J(theta')v'. Thus either factor projection has injective differential: v=0 forces v'=0, and conversely. Every stratum consequently has dimension at most eighteen. A stratum with eighteen-dimensional first projection is locally the graph of a map T between open parameter regions. Differentiating F(T(theta))=F(theta) gives equality of the two Jacobian images and rank[J(theta),−J(theta')]=18 there.

Conversely, wherever the collision Jacobian has rank at least nineteen, selecting nineteen independent equation differentials and applying the implicit function theorem bounds the local collision-set dimension by 36−19=17. Therefore the rank-eighteen tangent-sheet coincidence set is exactly the remaining possible source of a dominant off-diagonal collision family. A dimension-at-most-seventeen bound for that set, together with the already excluded parameter/boundary exceptional set, suffices for generic physical uniqueness. The manuscript properly leaves that bound unproved and keeps the collision equation itself indispensable; no off-collision rank test or infinite-time identity is substituted.

The algebraic-degree distinction is also correct. More precisely, use the graph of the real-analytic semialgebraic map F over the connected open domain O. If a product of two polynomials vanishes on this graph, the product of their analytic pullbacks vanishes on connected O; the identity theorem makes one pullback vanish identically. Its vanishing ideal is therefore prime, and its Zariski closure is irreducible (the same argument works with complex coefficients). Semialgebraicity gives graph dimension eighteen; the rank witness gives observation-image dimension eighteen. The dominant projection between these irreducible eighteen-dimensional Zariski closures is generically finite. This supplies a well-defined generic algebraic degree in a specified chart/field formulation, with no small-degree bound and no equality asserted with the number of physically admissible real roots. Physical uniqueness does not imply rational invertibility.

Minor wording correction requested: equation (5) of `physical_domain_contraction.md` prints c<√(nu/D) and then says the bound also applies on the weak branch. At R=0 equality holds. Use c≤√(nu/D) for the statement covering W, or explicitly separate the strict and weak versions. The positive axial conclusion is unaffected.

## 9. Independent audit of leading-order physical-label separation

`leading_order_prism_label_separation.md` passes, with its stated restriction to the axial-offset formal leading fundamental/quadratic coefficient model.

The modal ratios determine at most one physical index triple for each trial order because k/(1+k)² is strictly increasing for k∈[0.3,0.8]. For any physical trial triple, s_i=√(k_i/k_i') is positive and depends only on the true indices and trial order. The spare lever identity is affine in (d,g), with coefficients sum_i(v_i s_i) and 2v_1s_1+v_2s_2. Simultaneous vanishing forces (v_i s_i) to be proportional to (1,−2,1); positivity of all s_i forces the same middle mode. Thus only the correct order or full reversal can have an identically zero variable part.

For reversal, these coefficient equations force all s_i equal, hence k_i'=lambda k_i. Comparing the three ratio equations gives k_2=k_3, then lambda=1, then k_1=k_2. All indices are therefore equal. In this sole exceptional case the reconstructed gap is g'=−g−6/n<0, so reversal is impossible in the original physical box. Every potentially physical wrong order consequently obeys one proper affine-line condition in the true (g,d) plane. The union of at most five such lines is a valid exceptional superset; the note correctly does not assert that every point on a line yields an alias.

The concluding reconstruction statements also pass under the declared conditions: nonzero wedge amplitudes, nonzero source offset, resolved distinct modal nodes, and access to the individual formal coefficients. The signed-wedge/phase sectors in the original ±18-degree prior are disjoint away from zero, giving a unique signed representation of each recovered h_i. The speed interval has width seven Hz, below the twenty-Hz sampled-node alias period. None of this supplies extraction of exact finite-wedge coefficients from 200 samples, separation of colliding formal modes, exclusion of remote finite-wedge solutions, or exclusion of an alternative nonaxial system. The note expressly preserves those limitations.
