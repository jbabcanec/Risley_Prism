# Full eighteen-parameter local identifiability from the finite 200-position record

This note integrates the audited cubic construction, exact rational certificate and finite-sample argument. It concerns the exact original independent-axis optical equations at ideal timestamps t_k=k/20. It is a local identifiability theorem with all eighteen parameters free, not a global inverse or a finite-noise accuracy theorem. No recovery experiments are used in the proof.

**Physical-model limitation.** This theorem does not establish identifiability for true three-dimensional vector optics. At a centered axial source with rotationally symmetric reference geometry, vector rotational equivariance forbids the pure third-harmonic and all-plus cubic torus modes used here. The canonical theorem remains a theorem about its declared independent-axis equations. The selection rule, temporal-alias qualifications and resulting publication scope are stated near the beginning of [REPORT.md](../REPORT.md).

## 1. Theorem and precise scope

Let

\[
h_*=(n_1,n_2,n_3,g,d)=(3/2,4/3,5/3,3,100),
\quad N_*=(1,7,49)/20,
\quad\phi_*=0,
\]

and set both source angles and positions to zero at the witness. Define the positive first-order gains K_i(h) below. Fix any three nonzero real amplitudes A_i, for example A_i=1. Parameterize wedge sines by

\[
e_i=\sin\alpha_i=\varepsilon A_i/K_i(h).
\tag{1}
\]

**Theorem.** There exists epsilon0>0 such that for every 0<epsilon<epsilon0, the strict physical forward map to all 200 x/y positions has Jacobian rank18 at this witness family. All native parameters, including the four source coordinates, are allowed to vary in taking the Jacobian. Each such point has a neighborhood on which the full record map is injective.

The witness lies in the interior of the native bounds for sufficiently small epsilon, and strict physical branches hold by continuity from the flat, normal-incidence state. Setting source coordinates to zero locates the witness; it does not supply them to the inverse or remove their four derivative columns.

No explicit epsilon0, useful singular-value lower bound, guaranteed convergence basin, or observation allowance is established here. The result does not prove global uniqueness, uniqueness throughout the native box, or coverage of every physical connected component. The clock is exact k/20, not silently substituted for the stored binary64 timestamp array.

## 2. Derive the cubic optical recurrence

Angles in this section are radians. Let s denote one effective face sine. At centered normal incidence, simultaneous reversal of all wedge signs reverses transverse directions and positions. The response is therefore odd in the wedge sines.

Expand the incoming unit horizontal direction and entry position as

\[
X=x_1+x_3+O(s^5),\qquad p=p_1+p_3+O(s^5),
\]

where subscripts denote homogeneous degree in all small variables. Put kappa=n-1. Expanding the exact paired-prism equations gives outgoing direction components

\[
b_1=x_1+\kappa s,
\]
\[
b_3=x_3+\kappa x_1s^2+\frac{\kappa}{2n}x_1^2s
 +\frac{n\kappa}{2}s^3.
\tag{2}
\]

The positive longitudinal component implies outgoing slope b_1+b_3+b_1^3/2+O(s^5). The exact spatial multiplier has expansion

\[
L=1+L_2+O(s^4),\qquad
L_2=-\kappa s^2-\frac\kappa n x_1s.
\tag{3}
\]

Using K=3(X/H)L in the paired transfer gives

\[
q_1=p_1+\frac{3x_1}{n}+\ell b_1,
\]
\[
\begin{aligned}
q_3={}&p_3+\frac{3x_3}{n}+\ell b_3+\frac\ell2b_1^3
 +\frac{3x_1^3}{2n^3}\\
&-\kappa s^2p_1-\frac\kappa n x_1sp_1
 -\frac{3\kappa}{n}x_1s^2-\frac{3\kappa}{n^2}x_1^2s.
\end{aligned}
\tag{4}
\]

Initialize x1=x3=p1=p3=0 and apply (2)-(4) for the three ordered prisms with ell=(g,g,d), replacing the incoming tuple by the outgoing tuple after each prism. This is a symbolic Taylor recurrence derived from the exact model. It does not treat a cubic truncation as exact measured data.

The final first and third homogeneous position terms are

\[
p_1=\sum_iK_is_i,
\qquad
K_i=(n_i-1)\left[d+(3-i)g+3\sum_{j>i}\frac1{n_j}\right],
\tag{5}
\]
\[
p_3=\sum_{|m|=3}c_m(h)s^m.
\tag{6}
\]

The gains are positive on the native domain. Normal incidence has the same scalar coefficients on both axes; substitute s_i=e_i cos gamma_i on x and s_i=e_i sin gamma_i on y.

## 3. Five cubic invariants distinguish the hidden hardware

Keeping first-order amplitudes e_iK_i fixed removes the leading wedge/index/geometry ambiguity. Define

\[
I_m(h)=\frac{c_m(h)}{\prod_iK_i(h)^{m_i}},
\qquad
m\in\{300,030,003,210,021\}.
\tag{7}
\]

These are rational functions of h, explicitly defined by the finite recurrence. With (1), the corresponding cubic amplitudes are epsilon^3 A^m I_m(h). Their derivative with respect to h therefore separates hardware if the five-by-five derivative of I is nonsingular.

**Exact rational certificate.** At h*, the gains are

\[
(K_1,K_2,K_3)=(2201/40,524/15,200/3),
\]

and, with row order(300,030,003,210,021) and column order(n1,n2,n3,g,d),

\[
\det D_hI(h_*)=
-\frac{2342246801318629443}
{280149096378993020072641029913075712000}\ne0.
\tag{8}
\]

[cubic_rank_check.py](cubic_rank_check.py) constructs the recurrence, differentiates it and evaluates this determinant using exact rational arithmetic. Its complete matrix is saved in [cubic_rank_check.json](cubic_rank_check.json). This value was also independently reproduced from the exact paired formulas in the external audit. It is one mathematical certificate of nonvanishing, not a hardware recovery count or an approximate floating determinant test.

Equation(8) proves local invertibility of these five normalized cubic quantities near h*. It does not prove their global injectivity or assert that they can be recovered stably from arbitrary noisy positions.

## 4. Unknown beam angles and source offsets remain identifiable directions

To obtain beam derivatives, initialize the same homogeneous recurrence with

\[
x_1=\beta,\quad x_3=-\beta^3/6,
\quad p_1=p+6\beta,\quad p_3=2\beta^3.
\tag{9}
\]

At zero wedges the output is p+B0 beta to first order, where

\[
B_0=d+2g+6+3\sum_i\frac1{n_i}.
\tag{10}
\]

Let q_p and q_beta be the coefficients of p s3^2 and beta s3^2 in the final cubic polynomial. The recurrence gives

\[
q_p=1-n_3,
\qquad
B_0q_p-q_\beta=-\frac d2(n_3-1)(3n_3+1).
\tag{11}
\]

For verification, write S=B0-d. At only s3 nonzero, the source derivative multiplier is 1-(n3-1)s3^2, while the beam derivative coefficient is

\[
q_\beta=-(n_3-1)S
 +d\left[(n_3-1)+\frac32(n_3-1)^2\right].
\]

Substitution proves (11) directly. At h*, its value is -200. It is nonzero throughout the native n3,d intervals.

DC and the 2N3 harmonic therefore distinguish source angle from source offset on each axis at leading relevant orders. Specifically, use an ordinary offset column and the compensated direction delta beta=1, delta p=-B0. The compensated direction removes the constant leading response but retains the nonzero quadratic coefficient q_beta-B0 q_p. Squared cosine and sine supply opposite nonzero 2N3 factors; neither axis loses rank.

## 5. The finite200 record separates the required modes

For the native speed witness N*, every degree-at-most-three carrier is

\[
z_m^k,
\qquad z_m=\exp\left(2\pi i\frac{m_1+7m_2+49m_3}{400}\right),
\quad |m|_1\le3.
\tag{12}
\]

There are63 distinct nodes. Their numerators lie in[-147,147], so equality modulo400 is equality as integers. In a difference vector each coordinate lies in[-6,6]. A nonzero third digit has49 magnitude exceeding42+6; with third digit zero, a nonzero second digit has7 magnitude exceeding6. Thus no two nodes agree.

Unknown speeds additionally require k z_m^k at the six positive/negative fundamental nodes. These give69 confluent columns. The first69 time rows are independent: if a row vector annihilates them, its polynomial P of degree at most68 vanishes at all63 nodes, with derivative zero at the six repeated nodes. It has69 zeros counted with multiplicity and must be zero. All200 rows therefore have column rank69.

Fixed real linear functionals of the200 position samples can consequently separate the first-order amplitude/phase/frequency directions, five chosen cubic channels, and DC plus 2N3 on each axis in this limiting temporal space. For the cubic channels, the frequencies3Ni, 2N1+N2, and 2N2+N3 isolate the selected monomials up to known nonzero trigonometric factors. No supplied optical derivatives or independent torus measurements are assumed.

The linear functionals are used to prove a Jacobian rank at the witness. They are not claimed to provide a stable global initializer with unknown rotors. Higher exact optical orders are handled by the following analytic remainder argument, rather than set to zero.

## 6. Assemble the full18 proof

Use local coordinates(A1,A2,A3,N1,N2,N3,phi1,phi2,phi3,h,source4), with e_i defined by(1). For every fixed epsilon>0 this is an invertible local change from the native coordinates: Ki>0 and cos alpha_i>0. Phase/angle unit conversions are nonzero constant factors.

At zero source angles/positions, use these Jacobian column transformations:

* Scale the nine amplitude/phase/speed columns by epsilon^-1.
* Scale the five hardware columns at fixed A by epsilon^-3.
* On each source pair retain one ordinary offset column and scale the compensated beam column(delta beta=1,delta p=-B0) by epsilon^-2.

Apply the fixed sample functionals from section5. In the limit epsilon->0, the resulting eighteen-by-eighteen projected Jacobian is block triangular after arranging odd and even temporal channels:

* The nine fundamental columns are independent: each nonzero Ai gives independent amplitude and phase coefficients, and its speed gives the independent t times carrier.
* The five cubic hardware columns have determinant(8), multiplied by nonzero Ai and trigonometric factors. Hardware contributions to fundamental channels are allowed upper off-diagonal terms; they do not spoil triangularity.
* Each beam/source block is nonsingular by(11). Ordinary offsets supply DC; compensated beam variations supply the quadratic channel. Contributions of compensated columns to DC are harmless off-diagonal entries.

At centered normal incidence the exact optical response is odd in the wedges. Its next term after cubic is O(epsilon^5), and its hardware derivatives have the same order on a regular compact chart. Beam/source derivatives are even in wedges; after the quadratic term their next term is O(epsilon^4). Fundamental derivative remainders vanish after their stated scaling. Because the record is finite and the strict branch is analytic near the flat state, these expansions and differentiated remainders are valid uniformly over the200 sample times in a sufficiently small parameter neighborhood.

The limiting projected determinant is nonzero. Continuity makes the exact projected determinant nonzero for all sufficiently small positive epsilon. Hence the exact full400-by18 observation Jacobian has rank18. A nonsingular eighteen-row minor exists, and the inverse function theorem applied to those outputs gives a locally injective full-record map. This proves the theorem.

All18 directions were retained. In particular, cubic hardware compensation does not fix wedges, and evaluating the witness at normal centered incidence does not declare the unknown beam coordinates known.

## 7. Generic and noise consequences, with limits

Analyticity and a nonzero minor imply generic rank18 in the connected analytic interior component containing the witness. Fix one nonsingular eighteen-row minor at one sufficiently small positive-epsilon witness. Its determinant is analytic and not identically zero on that component, so the rank-deficient set is contained in its zero set and has dimension at most17. This is a component-qualified statement, not a proof about every component or every native system. Attached native parameter faces are treated separately below.

For the same exact clock t_k=k/20, use an injective eighteen-dimensional semialgebraic chart: bounded sine coordinates for the three rotor step angles pi*Ni/10, three wedge angles, three initial phases and two source angles, together with the three indices and four geometry coordinates. Every relevant native angle arc lies strictly between minus and plus pi/2, so these coordinate changes have nonzero derivatives and their cosine branches are positive. Retain separate phase and speed coordinates when a wedge is zero; quotienting out those fibers would change the full-hardware problem. In this chart, the exact forward graph and its critical set are semialgebraic by the positive-root construction in [algebraic_record.md](algebraic_record.md). The rank-deficient set therefore has a critical-value image of dimension at most17.

**Exact-record generic finiteness.** Let S be the eighteen-dimensional image of this component. Outside the critical-value set, every point of an exact fiber F(theta)=y is regular and hence isolated, by a nonsingular eighteen-output minor. The fiber is semialgebraic, and a zero-dimensional semialgebraic set is finite. Thus, outside an exceptional semialgebraic subset of S of dimension at most17 (or its closure), there are finitely many compatible exact systems in this component. They need not be unique. Here "generic records" is relative to S, not to the ambient four-hundred-dimensional observation space. In particular, zero-wedge speed/phase continua belong to exceptional fibers. This argument does not assert coverage of all other physical components.

Boundary language requires care. Images of included native parameter faces have dimension at most17. At excluded strict optical boundaries the forward map may be undefined, so one must not write F(boundary) without an extension. Instead use the finite-output projection of the frontier of the component's forward graph, namely its closure minus the graph itself. This semialgebraic frontier has dimension at most17, as does its projection. These boundary values, together with native-face and critical values, can be included in the exceptional set. Removing them is useful when discussing persistence of regular branches; it is not needed merely to prove finite regular fibers. No fixed positive margin from critical transmission is silently imposed.

The finiteness statement is specifically for exact equality, or eta=0. For eta>0, an interior hardware point satisfying every observation strip with strict residual slack has an open neighborhood of compatible hardware and therefore infinitely many compatible systems. Neither finite-noise accuracy nor global uniqueness follows. The exact-clock rank witness and algebraization used here share one timestamp contract; a separate minor evaluated on stored binary64 timestamps would not automatically establish this ideal-clock corollary.

The same construction explains weak conditioning. Vary h while holding first-order amplitudes fixed, with exact native compensation

\[
\delta\alpha_i=-\tan\alpha_i\,\delta\log K_i,
\tag{13}
\]

in radians, and keep speeds/phases fixed. The five independent native directions then have output derivative O(epsilon^3). Thus the proof of rank does not provide good accuracy at finite input precision. Compensated beam directions have leading order epsilon^2.

Still missing are an explicit wedge threshold, useful finite-noise inverse constants, a constructive global entrance mechanism, and exclusion of distant compatible hardware. Those are separate from the finite200 local-rank theorem and remain the targets of the general method in REPORT.md.
