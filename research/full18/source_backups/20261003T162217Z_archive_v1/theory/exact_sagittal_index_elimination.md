# Exact sagittal elimination of the last material index

## Result and scope

This is an exact finite-wedge optical reduction for the original three-prism, full18 inverse. It uses only the original paired screen samples. No ray directions, rotor speeds, beam coordinates, index values, continuous-time derivatives, or additional screen planes are assumed observed.

Let the thirteen retained optical coordinates be the first two indices, all nine wedge/phase/speed coordinates, and the two incident-beam coordinates. Conditional on those thirteen coordinates, a five-sample sagittal determinant generates a univariate polynomial of degree at most 64 for the remaining squared index G=n_3^2. Every exact physical solution is among its real roots. For each retained root, all four geometry variables remain subject to an exact affine solve/feasibility problem and the full record is checked. On charts where the determinant is not identically zero, this reduces the existing fourteen-optical-variable exact inverse to thirteen optical variables with finite last-index reconstruction.

The nonzero-chart condition is genuine, not an assumed generic rank: a single explicit interior finite-wedge witness has a rigorously nonzero five-row derivative determinant, with all 200 physical branch/traversal tests satisfied. An independent audit verifies the identity, norm construction, degree, and derivative argument. The accompanying fixed-point interval proof check certifies the witness.

This does not yet solve the remaining thirteen-variable optical inverse, prove global uniqueness, or establish a useful full-prior runtime. Exceptional zero-determinant strata must be retained. For positive observation noise, finite-index reconstruction becomes a necessary interval/sublevel test; it cannot in general remain a finite list.

## 1. The exact last-prism sagittal identity

Use actual external unit directions. At the third entrance, let the incoming transverse direction cosine be q in R^2 and the transverse position be p. The internal optical momentum is (q,H), where

H=sqrt(G-|q|^2)&gt;0,  G=n_3^2.

Let u be the last tilted-plane slope, B and Z the outgoing transverse and axial unit-direction components, and L=3+d the entrance-to-screen axial separation. Tangential Snell and internal intersection give

B=q+(H-Z)u,
P=H-u·q&gt;0,
a=(3+u·p)/P,
y-p=a q+(L-aH)B/Z.

Here a is the internal transverse propagation multiplier; the physical axial internal length is aH. Set

s=a+(L-aH)/Z,
K=(L-aH)(H/Z-1).

Then

y-p=s q+K u,  Hs-K=L.

For planar determinant [v,w]=v_x w_y-v_y w_x, this implies the root-free-in-the-exit-direction identity

**[y-p, q+H u]=(3+d)[q,u].**                                      (1)

There is no division by [q,u]. The identity remains valid when incoming transverse direction and wedge slope are parallel, including q=0 or u=0, although such rows can lose information. It does not require measuring B or Z and does not require solving the final transmitted square root.

If [q,u] is nonzero, the data determine s=[y-p,u]/[q,u] and K=[q,y-p]/[q,u]. Then (1) is Hs=3+d+K. This gives the useful explicit relation

G=|q|^2+(3+d+K)^2/s^2,                                      (2)

where s is nonzero. The determinant construction below avoids every such division and retains singular rows.

## 2. Upstream affine geometry and five linear rows

Fix the thirteen retained optical coordinates x. Trace only prisms 1 and 2. Their incoming-to-third-entrance direction q_k is independent of source position, g and d, and their position has the exact form

p_k=T_k b+c_k g+e_k,  b=(b_x,b_y).                              (3)

All coefficients in (3) depend only on x and the known timestamp. This is the usual exact affine propagation restricted to the two upstream prisms. It does not use a small-angle approximation.

At each timestamp define

alpha_k=|q_k|^2,
H_k(G)=sqrt(G-alpha_k)&gt;0,
v_k(G)=q_k+H_k(G)u_{3,k},
delta_k=[q_k,u_{3,k}].

Every candidate G in the original interval [169/100,81/25] has G-alpha_k&gt;0 because |q_k|&lt;1 on the external forward branch. Equation (1) gives

F_k(w,G)=A_k(G)w+z_k(G)=0,  w=(b_x,b_y,g,d),                   (4)

with the explicit augmented row

[A_k | z_k]=
[-[T_k e_x,v_k], -[T_k e_y,v_k], -[c_k,v_k], -delta_k,
 [y_k-e_k,v_k]-3 delta_k].                                    (5)

Every entry is affine in its own H_k. The d column is independent of H_k. All measured screen coordinates enter exactly as supplied; there is no extracted DC or spectral coefficient.

Choose five timestamps I and form the 5-by-5 augmented determinant

Delta_I(G)=det [A_I(G) | z_I(G)].                              (6)

Every compatible exact solution has Delta_I(G)=0. This necessary equation does not itself certify Snell transmission or the other screen-coordinate condition.

## 3. A degree-at-most-64 univariate candidate polynomial

In (6), each determinant term uses the root-free d column once. Thus Delta_I is a polynomial in the five H variables of total degree at most four. Its coefficients consist of the known data and the trial upstream optical quantities.

Before forming a norm, group equal alpha values. If the five alpha values have r distinct values a_1,...,a_r, substitute a single shared radical h_j=sqrt(G-a_j) wherever the value repeats. Then reduce powers with h_j^2=G-a_j and define

P_I(G)=product over sigma in {−1,+1}^r of
 Delta_I(sigma_1 h_1,...,sigma_r h_r).                         (7)

This is a polynomial in G. It can be constructed by successive sign-pair multiplication and reduction, without retaining square roots in the result.

The linear factors G-a_j have independent square classes in R(G): valuation at G=a_j detects the exponent of each factor. Consequently adjoining the r distinct square roots gives a field of degree 2^r, and (7) is its field norm. Therefore

P_I is identically zero if and only if Delta_I is identically zero.

Grouping is essential. Independently flipping duplicate radicals can create impossible conjugate branches and a false identically-zero product, even for a nonzero physical radical function.

Give G weight two and each h weight one. Delta_I has weight at most four. The product has weight at most 4·2^r, and radical reduction preserves weight. Hence

**deg_G P_I &lt;= 4·2^(r−1) &lt;= 64.**                            (8)

The coefficient field contains the trial upstream ray quantities and observed positions. This is a conditional polynomial, not a data-only speed pencil. For exact computational root isolation its coefficients require the same exact/algebraic input conventions as the existing inverse.

On a nonzero chart, every compatible physical last index belongs to the at-most-64 distinct real roots of P_I in its original squared-index interval. Roots from negative-radical conjugates can be extraneous. Evaluate the original positive-root equations and physical graph at every candidate; clearing the norm never authorizes keeping an unphysical branch.

## 4. Exact reconstruction algorithm and singular geometry

For each trial x in the original thirteen-coordinate optical domain:

1. Trace the first two direction stages at the original timestamps, checking the position-independent physical branch conditions. Compute q_k,T_k,c_k,e_k and carry the affine upstream traversal inequalities into the geometry solve in step 4.
2. Select a five-row chart I. Group repeated alpha values and form P_I. If it is nonzero, isolate all its real roots G in [169/100,81/25]. If it vanishes identically, try another chart or retain the exceptional optical stratum in the existing exact inverse; do not reject the trial.
3. For each root take n_3=sqrt(G)&gt;0, use the positive H_k, and reject only on violated original constraints. Recompute the exact final Snell direction rather than treating (1) as sufficient refraction information.
4. Solve the full exact affine geometry problem for b_x,b_y,g,d with every original source/distance bound, observation equation and strict traversal inequality. At rank four a nonsingular pivot determines geometry uniquely. At deficient rank retain the full affine geometry fiber; the earlier anchor/rank atlas or mixed-strict circuit elimination applies unchanged.
5. Validate all 200 paired screen samples with the exact physical model. A sagittal/norm candidate is not yet a physical inverse solution.

This covers all exact physical solutions whose retained optical point has at least one nonzero chart. A finite index candidate list does not imply a finite geometry fiber or a finite set of full18 systems, because x is still variable and genuine singular fibers remain possible.

## 5. A certified physical nonzero chart

Use the original times t_k=k/20. Choose

n_1=n_2=n_3=3/2,
tan(a_1)=tan(a_2)=tan(a_3)=1/10,
phi_1=phi_2=phi_3=0,
incident slopes t=(1/3,1/10),
b=(1,2), g=10, d=100.

Take the per-sample complex rotor multipliers

z_1=(15+8i)/17, z_2=(4+3i)/5, z_3=(3+4i)/5.

The signed speeds are N_j=20 arg(z_j)/(2 pi). Each lies strictly inside the original ±3.5 Hz interval. Both beam-angle coordinates, every wedge angle, source coordinate, index and distance lie in the original bounds. Generate y from this exact algebraic physical system only for the proof witness; the actual reconstruction algorithm never uses true hidden parameter values.

At fixed x and fixed y, differentiate the first five residuals (4) in (b_x,b_y,g,d,G). The first four derivative columns are A_I, and

partial_G F_k=[y_k-p_k,u_k]/(2H_k).                            (9)

The script proof_checks/sagittal_index_witness.py evaluates these expressions using fixed-point rational intervals at denominator 2^240. Every primitive arithmetic operation rounds outward using integer arithmetic; square roots are enclosed using integer square root. It certifies

-12477326382140449029055049739176464031438116561319332031617789789 / 2^240
 &lt;= det D_(b_x,b_y,g,d,G)(F_0,...,F_4) &lt;=
-12477326382140449029055049739176464031438116561319332031617789653 / 2^240.

The interval is strictly negative, approximately −7.061916467402616·10^−9. All 200 sampled transmitted/axial/intersection/ordering guards have lower bounds above 0.824. The first five alpha_k intervals are pairwise disjoint.

To connect this determinant to (6), fix the true geometry w*. Add the linear combination A_I(G)w* to the augmented final column. It becomes F_I(w*,G), which vanishes at G*=9/4. Differentiating the determinant therefore gives

Delta_I'(G*)=det[A_I(G*) | partial_G F_I(w*,G*)] !=0.           (10)

Thus the physical radical determinant has a simple zero and is not identically zero. The first four columns have rank four. The inverse-function theorem gives a genuine local exact reconstruction of all five variables (b_x,b_y,g,d,G) from five sagittal equations with the other thirteen optics fixed.

Continuity supplies an admissible open nonzero chart around the witness. The independently audited physical_domain_contraction.md establishes that the entire physical interior O is connected. The fixed five-row derivative determinant J(theta), evaluated at the exact record y=F(theta), is analytic and semialgebraic throughout O: upstream roots, axial components and transmitted denominators remain strictly positive. Therefore its zero set Z has dimension at most 17 on all of O.

More precisely, let W be the compact weak closure and Fbar its continuous semialgebraic observation extension. Set

B=(W\O) union closure_W(Z),  E=Fbar(B).

The existing boundary dimension theorem gives dim B&lt;=17, so E is a compact semialgebraic observation exceptional set with dim E&lt;=17. For every exact record y in Fbar(W) outside E, every physical preimage has J!=0. Hence the same fixed first-five-sample index determinant is nonzero at every compatible thirteen-coordinate optical point. The degree-64 index reconstruction therefore covers every physical solution for generic records throughout the full prior, not just one witness component.

This is a generic theorem, not an effective certification that an arbitrary supplied record avoids E. Exceptional records and zero-chart strata must still be retained by a complete arbitrary-data inverse. This is a single narrow proof check, not a parameter campaign or a practical accuracy experiment.

## 6. Positive noise: useful necessary bounds, not finite candidates

For coordinatewise screen errors at most epsilon, write the exact output y_k+eta_k with |eta_(k,l)|&lt;=epsilon. Equation (1) implies the necessary bound

|F_k(w,G;y)| &lt;= epsilon ||v_k(G)||_1.                         (11)

This bound is also exactly the possible range of the one sagittal projection of that sample's error square, but it does not impose the other optical/position constraints.

Let C_k(G) be the cofactors of the final column in (6). Since the same four geometry columns are unchanged by output perturbation, any compatible noisy solution satisfies

|Delta_I(G;y)| &lt;= epsilon sum_(k in I) |C_k(G)| ||v_k(G)||_1.  (12)

This is a geometry-free one-dimensional necessary sublevel condition. It can screen intervals of G or combine several charts. Feasibility still requires the original noisy affine geometry constraints and every physical branch guard.

### Exact scalar interval construction for the noisy tube

For any fixed noise budget epsilon, the necessary set in (12) can itself be computed as a finite union of closed intervals and points using scalar polynomial root isolation of degree at most 64.

A final-column cofactor C_k is a four-by-four determinant containing the root-free d column, so it has total radical degree at most three in the other four samples. Its grouped field norm has degree at most 3 times 2^3=24 in G. Each component v_(k,l)=q_(k,l)+u_(k,l) sqrt(G-alpha_k) has the linear norm

q_(k,l)^2-u_(k,l)^2(G-alpha_k).

Omit terms that vanish identically. The real roots of the five cofactor norms and ten component norms partition the squared-index prior at at most 5 times 24 plus 10=130 candidate interior points. Extraneous conjugate roots merely refine this partition. On each resulting open interval, the physical signs of every nonzero cofactor and vector component are fixed.

After fixing those signs, the right-hand side of (12) is one explicit radical polynomial of total radical degree at most four. The two boundary functions Delta_I minus that expression and minus Delta_I minus that expression therefore have grouped norms of degree at most 64. Isolate their roots on each sign interval and test the original positive-root signs, keeping the appropriate endpoints. Identically-zero boundary functions are treated as equality throughout their interval, never as a reason to delete it.

This constructs the one-dimensional necessary tube exactly, with no multi-variable geometry solve. It can be intersected over several five-row charts. It remains a screening test; passing it does not certify the full noisy inverse.

For an interior exact solution, continuity allows an open interval of nearby G at every positive error budget, even while the other thirteen optics and geometry remain fixed. Therefore no general finite-index bound such as 64 extends to positive-noise feasible vectors. Claiming such a finite list would be false. The exact-data elimination is nevertheless a new optical reduction; the noisy determinant tube is an additional sound screening tool.

## 7. What this closes and leaves open

Closed: an exact, finite-wedge, observation-level identity eliminates the last refractive index by a univariate polynomial of bounded degree on a verified physical chart. This goes beyond the existing four-variable geometry reduction and retains unknown rotors, beam, source, gaps and screen distance. The proof uses no artificial third harmonic or hidden ray-direction observation.

Open: a useful all-prior search over the remaining thirteen optical coordinates, control of exceptional determinant-zero strata without restoring a large solve, finite-noise conditioning, and global uniqueness. The reduction should be combined with other constructive attacks rather than represented as the completed full18 reverse solver.
