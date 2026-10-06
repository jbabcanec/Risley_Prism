# Optical-core reductions and the information missing at first order

This note integrates audited structural leads into the general model. It uses no recovery experiments. Native order, all18 freedoms and the original passive record are retained. Statements concern the strict mathematical independent-axis model.

## 1. Eliminate explicit glass indices without losing their spatial effect

Use the paired-prism notation of algebraic_record.md: incoming unit transverse direction X, effective face sine s and cosine c>0, inside-glass vertical component H>0, outgoing unit air components (a,v), with v>0. The exact direction identities imply

\[
c a+s v=cX+sH,
\quad\text{hence}\quad sH=c(a-X)+sv.
\tag{1}
\]

This is unsquared tangential conservation. With H^2+X^2=n^2 it gives

\[
n^2s^2=[c(a-X)+sv]^2+s^2X^2.
\tag{2}
\]

For s nonzero one can solve (2) for n^2, but the unsquared sign and physical branch conditions must remain. In particular require H>0, a^2+v^2=1, v>0 and R=-sa+cv>0. Squaring (1) alone admits a wrong sign.

There is an exact division-free elimination valid even at s=0. For every occurrence j of prism i across both axes and all samples, retain H_ij>0 and X_ij. Replace the explicit common n_i by

\[
H_{ij}^2+X_{ij}^2=H_{i0}^2+X_{i0}^2,
\qquad1.3^2\le H_{i0}^2+X_{i0}^2\le1.8^2.
\tag{3}
\]

Then reconstruct n_i as the positive square root of the common value. This is equivalent to the original index constraints, since the allowed index is positive. It is a structural elimination of three named variables, not a reduction of the intrinsic hardware dimension: the common H reference values still carry that information.

At s=0, (1) only says a=X. The plate's index still changes the spatial displacement 3X/H. Therefore retain the full position transfer

\[
vD p_{next}=R(Hp+3X)+\ell aD,
\qquad D=cH-sX>0,
\tag{4}
\]

and never erase H from the spatial model. This gives a lower-degree common-index constraint in latent-ray coordinates without pretending those rays are measured.

## 2. Why a first-order optical model cannot identify all18

Scale the wedges as alpha_i=epsilon b_i on a regular compact chart, with b_i bounded and all other hardware in a compact smooth domain. At first order in the wedges, each axis is a constant plus one real sinusoidal response per rotor:

\[
x_k=b_x+\sum_i A_{xi}\cos\gamma_{ik},\qquad
y_k=b_y+\sum_i A_{yi}\sin\gamma_{ik}.
\tag{5}
\]

The coefficients depend on the unknown source, glass and geometry. Both axes share each speed and phase. Thus the whole first-order family factors through at most14 real quantities: two baselines, three speeds, three phases and six real amplitudes. The Jacobian of that first-order family has rank at most14, regardless of the number of sampled times.

The exact family differs from it by O(epsilon^2). Differentiation in the scaled coordinates (b_i and all unscaled remaining parameters) preserves that order on a regular compact chart. At least four singular values of the exact Jacobian in those scaled coordinates are therefore O(epsilon^2), by the rank bound and the singular-value perturbation inequality.

The scaling qualification matters. Native derivatives with respect to alpha_i divide the scaled derivative by epsilon; the same O(epsilon^2) conclusion cannot be transferred unqualified to native coordinates. Equation (5) identifies missing higher-order information, rather than proving a finite-noise lower bound or global nonuniqueness by itself.

## 3. Stronger cubic weakness at centered normal incidence

Now fix beta_x=beta_y=p_x=p_y=0, while keeping all prism indices, distances and gaps unknown. Simultaneously changing every signed wedge to its negative changes each transverse direction and position to its negative, while preserving all positive longitudinal components. This follows directly from (2) and (4): Q,a,K,T change sign; H,c,R,v,D,L do not. Consequently the exact response is odd in the joint wedge vector; even total wedge degrees vanish.

At first order, in radians,

\[
z_1(t)=\sum_{i=1}^3 e_i K_i e^{i\gamma_i(t)},
\qquad e_i=\sin\alpha_i,
\qquad
K_i=(n_i-1)\left[d+(3-i)g+3\sum_{j>i}\frac1{n_j}\right].
\tag{6}
\]

To derive K_i, a small wedge creates air deflection (n_i-1)alpha_i. It then crosses the remaining (3-i) air gaps, the final distance d, and each downstream flat glass thickness with slope divided by n_j. Its own thickness adds no first-order displacement because the normal incoming ray reaches its exit at transverse position zero.

Let h=(n1,n2,n3,g,d). For any perturbation delta h, keep speeds and phases fixed and choose

\[
\delta e_i=-e_i\,\delta\log K_i,
\qquad\delta\alpha_i=-\tan\alpha_i\,\delta\log K_i.
\tag{7}
\]

Then delta(e_i K_i)=0 exactly, so all three complex first-order amplitudes are unchanged. The five independent hardware perturbations in h produce five independent native tangent directions; their wedge corrections are O(epsilon) when alpha=epsilon b. Odd symmetry makes the remaining exact directional response O(epsilon^3) on a regular compact chart. The angle compensation in (7) uses radians.

These are five concrete native directions whose information first appears through cubic optical interactions. This is a structural conditioning statement on the centered/normal stratum, not a generic global-uniqueness obstruction. It suggests a principled next elimination: normalize informative cubic coefficients by the first-order amplitudes, then solve for h and restore wedges. Obtaining those coefficients from the actual noisy finite record, handling unknown beam/source coordinates, and proving a global inverse are separate requirements.

The cubic recurrence, exact rational determinant, unknown beam/source block and finite-record remainder argument are now integrated in [full18_local_rank.md](full18_local_rank.md). Together they prove local rank eighteen for a sufficiently small nonzero wedge family on the ideal 200-sample clock. No explicit wedge threshold, global uniqueness or finite-noise accuracy guarantee follows. The five formal cubic hardware invariants also admit the scalar reconstruction in [cubic_scalar_inverse.md](cubic_scalar_inverse.md), whose exact-model completion remains a separate requirement.
