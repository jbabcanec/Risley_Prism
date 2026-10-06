# Physical-vector inverse theory for three rotating prisms

## Status, provenance and scope

This appendix integrates the supplied, independently audited mathematical memo. It does not report a new numerical verification or an empirical campaign. The derivations establish a constructive small-wedge, off-axis witness for full rank of the **exact physical-vector, 200-sample, 18-parameter forward map**. They also supply an exact root-free forward graph and a conditional four-dimensional affine geometry reduction.

They do **not** establish global uniqueness of the exact finite-wedge inverse, a numerical small-wedge threshold, or uniform conditioning throughout the prior box.

**Notation and error dependence.** The nonnegative measurement-error parameter is \(\epsilon\), with componentwise condition \(|F_m-y_m|\le\epsilon\); unequal allowances may be denoted \(\epsilon_m\). The independent, dimensionless wedge scale is \(\kappa\), and a certified approximation remainder is \(\tau\). Reducing measurement error does not remove finite-angle contamination. Certificate domains and constants may depend on \(\epsilon\); reusing them across an error interval requires uniform certification. Generic finite exact-data or formal-coefficient fibers must not be confused with generally continuous positive-error compatible sets.

The companion [global inverse completeness theorem](global_inverse_completeness.md) and [explicit six-root compiler](explicit_six_root_compiler.md) now establish exact global set completeness in principle under their finite exact input encodings and sampling assumptions, including exceptional and positive-dimensional fibers. They supply neither global uniqueness nor a practical complete solver. Optical admissibility there is imposed at measured times; between-sample admissibility is separate.

The independent-axis canonical model studied in the other appendices is a different, mathematically valid model. Its pure-third-harmonic hardware inverse is not a theorem about coupled vector Snell optics. The physical construction here instead uses offset-excited second harmonics and complex negative fundamentals.

The vector Snell law is standard. A primary technical reference is Ken Moore, Ansys/Zemax, [“What is a ray?”](https://optics.ansys.com/hc/en-us/articles/42661810210707-What-is-a-ray), section “Refraction, reflection, and diffraction.” The inverse constructions and proofs below are derived research results; they are not attributed to that reference.

## 1. Exact physical model and conventions

Identify a transverse vector in \(\mathbb R^2\) with its complex coordinate \(x+iy\). Dot products and absolute values retain their real Euclidean meanings. The source is \((p_x,p_y,0)\), with complex transverse position

\[
b=p_x+i p_y.
\]

The incident unit direction is the normalization of

\[
(\tan\beta_x,\tan\beta_y,1).
\]

Write its transverse direction cosine as \(q\), and its positive axial component as \(\sqrt{1-|q|^2}\). At \(\beta_x=\beta_y=0\), the derivative from \(\beta_x+i\beta_y\) to \(q\) is the identity. Angles in the equations are in radians; the stated prior limits may be expressed in degrees.

For prism \(j\), define

\[
\gamma_j(t)=2\pi N_jt+\phi_j,
\qquad
u_j=\tan a_j\,(\cos\gamma_j,\sin\gamma_j).
\]

The flat entrance planes are

\[
z=6,\qquad z=9+g,\qquad z=12+2g.
\]

At transverse position \(p\), the corresponding exit planes are

\[
z=9+u_1\mathbin\cdot p,\qquad
z=12+g+u_2\mathbin\cdot p,\qquad
z=15+2g+u_3\mathbin\cdot p.
\]

Each prism therefore has axial vertex thickness \(3\). The screen is at \(z=15+2g+d\). The forward-facing exit unit normal is

\[
\frac{(-u_j,1)}{\sqrt{1+|u_j|^2}}.
\]

The external index is one; the glass indices are \(n_1,n_2,n_3\). The hardware box is

\[
n_j\in[1.3,1.8],\qquad g\in[2,15],\qquad d\in[50,200].
\]

The remaining stated bounds are \(\pm18^\circ\) for wedge and phase angles, \(\pm3.5\,\mathrm{Hz}\) for speeds, \(\pm25^\circ\) for beam angles, and \(\pm5\) for each source coordinate. Lengths use the original model's consistent native length unit.

The rank witness uses positive, sufficiently small interior wedges, zero nominal beam tilt and a fixed nonzero source offset inside its box. All equations use the forward transmitted branch, exclude total internal reflection and grazing incidence, and require positive axial direction. If physical sequential surface traversal is required, positive propagation lengths and the condition that each next flat entrance or screen follows the current exit must also be enforced. The small-wedge witness satisfies these strict conditions.

## 2. Exact root-free local graph

At a flat prism entrance, let \(X=(X_1,X_2)\) be the incoming external transverse direction cosine. Introduce \(H>0\) and \(G=n^2\), with

\[
H^2+|X|^2=G.
\]

The internal unit direction is \((X,H)/n\). Write the outgoing external unit direction as \((B,Z)\), where \(B\in\mathbb R^2\). The exact exit conditions are

\[
B-X+u(Z-H)=0,\qquad |B|^2+Z^2=1,\qquad Z>0.
\]

Introduce

\[
P=H-u\mathbin\cdot X>0,
\qquad
R=Z-u\mathbin\cdot B>0.
\]

The tangential equations preserve optical momentum along both exit-plane tangents. The inequality \(R>0\) selects the transmitted normal branch; together with \(H>0\) and \(Z>0\), it removes unwanted algebraic roots.

For transverse entrance position \(p\), set

\[
W=ZX-HB,\qquad V=ZX-(u\mathbin\cdot X)B.
\]

Let \(\ell\) be the axial gap from the exit vertex to the next flat plane. The exact position equation is

\[
ZP\,p_{\mathrm{next}}
=ZP\,p+W(u\mathbin\cdot p)+3V+\ell P B. \tag{1}
\]

To verify (1), the axial internal traversal is

\[
\tau_{\mathrm{int}}=\frac{H(3+u\mathbin\cdot p)}{P}.
\]

Direct propagation gives

\[
p_{\mathrm{next}}
=p+\frac{(3+u\mathbin\cdot p)X}{P}
 +\frac{(3+\ell-\tau_{\mathrm{int}})B}{Z},
\]

which rearranges to (1). This calculation also identifies the positive-traversal inequalities needed when the surfaces must be encountered in the stated physical order.

After declaring dot products and \(W,V,P\) as auxiliary variables, the graph uses polynomial equations of degree at most three. For example, \(ZPp_{\mathrm{next}}\) and \(\ell PB\) are cubic; \(W(u\mathbin\cdot p)\) is at most cubic with \(W\) auxiliary. The graph itself requires no square-root evaluations. The stated positive inequalities select the intended roots.

Initial propagation has the same form:

\[
Z_0p_{\mathrm{entry}}=Z_0b+6X_0,
\qquad |X_0|^2+Z_0^2=1,
\qquad Z_0>0.
\]

### Conditional affine geometry reduction

For fixed optical parameters and directions, (1) is affine in \(p\) and \(\ell\). Directions are independent of positions, \(g\) and \(d\). Apply \(\ell=(g,g,d)\), with initial position \(b+6q/\sqrt{1-|q|^2}\). The exact final map is therefore

\[
F(t)=A(t)b+\mathcal B(t)+gC(t)+dD(t). \tag{2}
\]

Here \(A\) is a real \(2\times2\) matrix, while \(\mathcal B,C,D\) are transverse two-vectors. The coefficients depend only on fourteen optical parameters: the three indices, wedges, phases and speeds, and the two beam angles. The four remaining geometry variables are \((p_x,p_y,g,d)\).

Consequently, fixed-optics, componentwise interval observations yield a four-dimensional linear feasibility problem, including the geometry box and any affine surface-order inequalities. Certified source projection and linear-programming elimination survive vector coupling. The earlier independent-axis normalized polygon formula cannot be transferred without deriving its coupled-vector counterpart. Euclidean residual bounds instead produce second-order-cone constraints.

## 3. Centered obstruction and canonical discrepancy

For an axial ray through the first prism's on-axis vertex, the exit point is fixed at \((0,0,9)\), and the deflection is

\[
\delta_1=\arcsin(n_1\sin a_1)-a_1.
\]

Its entire output ray depends on \((n_1,a_1)\) only through \(\delta_1\). All downstream optics preserve this exact one-dimensional level-set gauge. A tangent satisfies

\[
\frac{da_1}{dn_1}
=-\frac{\sin a_1}
 {n_1\cos a_1-\sqrt{1-n_1^2\sin^2 a_1}}.
\]

Thus full rank eighteen cannot hold at that centered physical witness.

At a centered axial source, simultaneous rotation of all rotors rotates the complex screen coordinate:

\[
Z(\gamma+t\mathbf 1)=e^{it}Z(\gamma).
\]

A complex torus Fourier coefficient indexed by \(m\in\mathbb Z^3\) therefore vanishes unless \(\sum_jm_j=1\). Pure third harmonics vanish exactly. Mixed phases such as \(2\gamma_i-\gamma_j\) and \(\gamma_i+\gamma_j-\gamma_k\) are allowed. This is a torus selection rule; temporal frequency collisions require separate treatment.

For one prism, put \(r=\sin a\), \(s=r(\cos\gamma,\sin\gamma)\), \(k=n-1\), and

\[
\mathcal C=\frac{k(n^2-n+1)}{2}.
\]

The centered slopes are

\[
\begin{aligned}
\text{canonical:}\quad &ks+\mathcal C(s_x^3,s_y^3)+O(r^5),\\
\text{physical:}\quad &ks+\mathcal C|s|^2s+O(r^5).
\end{aligned}
\]

The canonical complex slope contains the artificial third-harmonic term

\[
\frac{\mathcal C r^3}{4}e^{-3i\gamma}.
\]

The leading maximum slope-vector discrepancy is \(\mathcal C r^3/2\). It is of the same order as the canonical cubic identification information. A relative \(O(a^2)\) error in total deflection therefore does not justify transferring canonical cubic hardware inference to physical optics.

## 4. Reproducible cubic vector recurrence

Let

\[
s_j=\sin a_j\,e^{i\gamma_j},\qquad k_j=n_j-1.
\]

For this formal recurrence, grade \(b,q,s\) as degree one. Write the incoming transverse direction as \(x_1+x_3+O(5)\) and the entrance position as \(r_1+r_3+O(5)\). Initially,

\[
x_1=q,\qquad x_3=0,\qquad
r_1=b+6q,\qquad r_3=3|q|^2q.
\]

The last term comes from exact source-to-first-plane propagation. It does not affect the off-axis coefficients or linear beam derivatives used below, but is required for the complete cubic recurrence.

At prism \(j\), abbreviate \(n=n_j\), \(k=n-1\), \(s=s_j\), and take \(\ell=g\) for \(j=1,2\), or \(\ell=d\) for \(j=3\). Then

\[
\begin{aligned}
y_1={}&x_1+ks,\\
y_3={}&x_3+\frac{k}{2n}|x_1|^2s
 +\frac{k}{2}|s|^2x_1
 +\frac{k}{2}s^2\overline{x_1}
 +\frac{kn}{2}|s|^2s,\\
r_{1,\mathrm{next}}={}&r_1+\frac{3x_1}{n}+\ell y_1,\\
r_{3,\mathrm{next}}={}&r_3+\frac{3x_3}{n}+\ell y_3
 +\frac{3|x_1|^2x_1}{2n^3}
 +\frac{\ell}{2}|y_1|^2y_1\\
&-k\left(\frac{x_1}{n}+s\right)
 \operatorname{Re}\!\left[\overline{s}
 \left(r_1+\frac{3x_1}{n}\right)\right].
\end{aligned} \tag{3}
\]

Set \(x_{1,\mathrm{next}}=y_1\) and \(x_{3,\mathrm{next}}=y_3\). Equation (3) follows by expanding the exact graph of Section 2. The direction formula uses direction cosines; the term \(|y_1|^2y_1/2\) converts outgoing direction cosine to air slope.

The exact position map is affine in \(b\). Therefore the \(b\)-dependent terms extracted from this joint-cubic recurrence also give the full leading wedge-order coefficients for fixed \(b\); they do not require the source offset to shrink with the wedges.

## 5. Offset-excited hardware inverse

Set \(q=0\), retaining an unknown, fixed, nonzero \(b\). Define

\[
h_j=\sin a_j\,e^{i\phi_j},\qquad
s_j=h_j e^{2\pi iN_jt}.
\]

At leading quadratic wedge order, the offset-dependent screen correction is

\[
-\sum_j k_j(s_j\mathbin\cdot b)
 \left(\frac{P_{j-1}}{n_j}+s_j\right),
\qquad P_{j-1}=\sum_{i<j}k_is_i,
\]

where \(s\mathbin\cdot b=\operatorname{Re}(\overline{s}b)\). The self second-harmonic and positive mixed-sum coefficients are consequently

\[
S_j=-\frac{\overline b\,k_jh_j^2}{2}, \tag{4}
\]

\[
M_{ij}=-\frac{\overline b\,k_ik_jh_ih_j}{2n_j},
\qquad i<j. \tag{5}
\]

These are **leading homogeneous coefficient identities**, not exact finite-wedge identities for measured Fourier coefficients. Their next corrections in exact torus coefficients have wedge degree four.

Form the phase- and offset-free ratios

\[
R_{ij}=\frac{M_{ij}^2}{S_iS_j}=\frac{k_ik_j}{n_j^2}.
\]

With

\[
r=\frac{R_{13}}{R_{23}},\qquad
u=\sqrt{\frac{R_{12}}{r}},
\]

the positive physical branch gives

\[
k_2=\frac{u}{1-u},\qquad
n_2=\frac{1}{1-u},\qquad
k_1=rk_2.
\]

Set \(c=R_{23}/k_2\). It determines \(k_3\) through

\[
c=\frac{k_3}{(1+k_3)^2}.
\]

This function is strictly increasing for \(k_3\in[0.3,0.8]\), so the physical branch is

\[
k_3=\frac{1-2c-\sqrt{1-4c}}{2c}.
\]

The other root is its reciprocal and lies outside this interval. The inverse is smooth in the interior physical box.

The leading fundamental gains are

\[
\begin{aligned}
K_1&=k_1\left(d+2g+\frac3{n_2}+\frac3{n_3}\right),\\
K_2&=k_2\left(d+g+\frac3{n_3}\right),\\
K_3&=k_3d.
\end{aligned} \tag{6}
\]

The leading DC term gives \(b\). Equations (4), the recovered indices and nonzero \(b\) give wedge magnitudes; the fundamentals give phases and gains on the chosen local wedge/phase branch. Then

\[
d=\frac{K_3}{k_3},\qquad
g=\frac{K_2}{k_2}-d-\frac3{n_3}.
\]

The value of \(K_1\) supplies redundancy.

An exact one-prism illustration explains the excitation. For axial incidence at offset \(b\), screen distance \(\ell\) from the exit vertex, and

\[
D=\tan[\arcsin(n\sin a)-a],\qquad
\chi=\frac{(\tan a)D}{2},
\]

the complex screen position is

\[
Z=(1-\chi)b+\ell D e^{i\gamma}
 -\chi\overline b\,e^{2i\gamma}.
\]

Unknown nonzero offset thus supplies genuine second-harmonic index/wedge information. That information vanishes at \(b=0\).

## 6. Separating both unknown beam angles

At \(q=0\), the exact vector map is affine in the source offset:

\[
Z=Z_0+A b+B\overline b.
\]

Under simultaneous rotor rotation, the torus weights of \(Z_0,A,B\) are respectively \(1,0,2\). Every complex negative-fundamental coefficient indexed by \(-e_j\) therefore vanishes identically at \(q=0\), for every value of the other sixteen parameters. This statement concerns \(Z=x+iy\); the Fourier symmetry of a single real coordinate does not identify this channel.

Its leading tilt derivative is

\[
C_{-e_j}
=-\frac{k_j}{2n_j}\,bq\,\overline{h_j}
 +\text{higher wedge orders}. \tag{7}
\]

For nonzero \(b\) and \(h_j\), (7) is a nonzero complex-linear map of \(q\), and supplies both real beam-angle directions. It follows from the last term of (3), and also from direct first-order intersection geometry. The supplied independent audits agreed on its sign and factor.

Exact torus functionals are not presumed accessible from the finite measurements. The next section uses a finite, fixed linear extractor solely to prove Jacobian rank.

## 7. Exact 200-sample full-rank theorem

### Explicit finite design

Use the ideal sampling clock

\[
t_k=\frac{k}{20},\qquad k=0,\ldots,199,
\]

and witness frequencies

\[
(N_1,N_2,N_3)=\frac{(1,5,25)}{20}\ \mathrm{Hz}.
\]

For the twenty-five integer labels \(m\in\mathbb Z^3\) with \(|m|_1\le2\), set

\[
z_m=\exp\!\left[\frac{2\pi i(m_1+5m_2+25m_3)}{400}\right].
\]

These nodes are distinct. The difference of two labels has \(\ell^1\) norm at most four, while any nonzero integer base-five relation \(v_1+5v_2+25v_3=0\) has \(\ell^1\) norm at least six. The integer exponents lie between \(-50\) and \(50\), so no further modulo-400 alias occurs.

Form a \(200\times31\) complex dictionary with the twenty-five columns \(z_m^k\) and the six confluent columns \(kz_m^k\) for \(m=\pm e_j\). Its first thirty-one rows form a nonsingular confluent Vandermonde matrix. A fixed left inverse consequently exists, and its realification can be applied to the two measured screen coordinates.

This extractor is fixed at the witness and is a proof device for a Jacobian. It is not an estimator that assumes the unknown frequencies have already been recovered. The theorem here uses the stated ideal clock; transfer to a different stored sampling clock requires its own argument.

### Coordinates and column scaling

Fix any interior hardware vector \(\mathfrak h=(n_1,n_2,n_3,g,d)\), any fixed nonzero \(b\) in its allowed box, \(q=0\), and three nonzero complex amplitudes \(A_j\). Parameterize

\[
h_j=\sin a_j\,e^{i\phi_j}
=\frac{\kappa A_j}{K_j(\mathfrak h)}.
\]

Choose the phases inside their allowed interval and \(\kappa>0\) sufficiently small. For every nonzero such \(\kappa\), this is a smooth local change of variables from positive wedge magnitudes and phases to complex amplitudes, retaining the five hardware coordinates.

The eighteen real coordinates are six amplitude components, three frequencies, five hardware coordinates, two source-offset components and two beam components. Scale their Jacobian columns respectively by

\[
\kappa^{-1},\qquad
\kappa^{-1},\qquad
\kappa^{-2},\qquad
1,\qquad
\kappa^{-1}.
\]

Before scaling the beam columns, compensate the zero-wedge beam drift with

\[
\delta b=-B_0\,\delta q,
\qquad
B_0=6+2g+d+3\sum_{j=1}^3\frac1{n_j}.
\]

This is an invertible triangular column transformation. The coefficient \(B_0\) has units of length, consistent with the compensation.

### Limiting rank and crossblocks

The fixed extractor yields the following limiting blocks:

1. **Negative fundamentals:** only the two beam columns survive. Equation (7) gives a nonzero complex multiplier and hence real rank two.
2. **DC:** the source-offset block is the two-dimensional identity.
3. **Fundamentals and confluent fundamentals:** six amplitude directions and three frequency directions are independent, because all \(A_j\) are nonzero.
4. **Quadratic self/sum coefficients:** the five hardware columns are independent.

For the last claim, hold \(A,b\) fixed. The normalized quadratic coefficients are

\[
S_j=-\frac{\overline b\,k_jA_j^2}{2K_j^2},
\qquad
M_{ij}=-\frac{\overline b\,k_ik_jA_iA_j}
 {2n_jK_iK_j}.
\]

Their ratios recover all three indices as in Section 5. On the positive-gain branch,

\[
K_j=|A_j|\sqrt{\frac{|b|k_j}{2|S_j|}}.
\]

Equation (6) then recovers \(d,g\). This is a smooth left inverse of the five-dimensional hardware map, proving that its differential has rank five.

Hardware columns may retain a limiting DC component. Eliminate it using the source-offset identity block; the unscaled offset columns have vanishing quadratic block in this limit. Beam columns may retain positive-fundamental components, which can be removed using the fundamental blocks. Neither crossblock reduces rank.

### Transfer of differentiated remainders

At \(q=0\), angular output is odd in the wedges and the affine source correction is even. The retained angular term has degree one, followed by degree three; the retained source-dependent term has degree two, followed by degree four. Their \(C^1\) remainders vanish after the stated column scalings.

A differentiated quadratic frequency term is \(O(\kappa^2)\), and vanishes after frequency-column scaling by \(\kappa^{-1}\). Repeated quadratic nodes are therefore unnecessary. After the baseline compensation, beam derivatives start at \(O(\kappa)\), followed by higher wedge orders; their \(\kappa^{-1}\)-scaled remainder vanishes. Their leading negative-fundamental block is nonzero by (7).

The limiting transformed Jacobian has rank

\[
2+2+9+5=18.
\]

Continuity therefore implies that, for every sufficiently small positive \(\kappa\), the **exact physically coupled, 400-real-output sampled Jacobian has rank eighteen**.

This proves local identifiability on an explicit family of admissible off-axis witnesses, with all eighteen parameters unknown. A nonzero analytic minor also establishes generic rank on the connected analytic branch containing the witness. It does not prove global uniqueness, rank at every parameter point, or a usable numerical threshold for \(\kappa\).

## 8. Conditioning, excitation and inference limits

- The centered axial configuration has an exact first-prism gauge. A practical stability statement must quantify excitation away from it.
- At fixed nonzero \(b\), the hardware-separating channels scale as \(|b|\kappa^2\). Beam-identifying negative fundamentals scale as \(|b|\kappa\) times the beam perturbation.
- These orders improve on the canonical cubic mechanism, but do not establish useful numerical conditioning. Gain normalization, native length units, parameter scaling and the finite dictionary's smallest singular value remain relevant.
- The explicit base-five design certifies rank; it is not claimed to optimize sampling or conditioning.
- Exact torus quadratic coefficients have \(O(\kappa^4)\) corrections. The finite thirty-one-column extractor can leak unrepresented cubic angular terms into its quadratic rows. The safe normalized hardware remainder is consequently \(O(\kappa)\), unless an additional cancellation or richer dictionary is proved.
- Fundamental extraction after accounting for degree-two modes has relative higher-order contamination \(O(\kappa^2)\). A naive fundamental-only fit may absorb quadratic sidelobes. There is no universal frequency-estimation bias claim without a specified estimator and design.
- Noise amplification near \(b=0\) or another weak-excitation configuration is unavoidable. Quantitative guarantees require explicit remainder bounds and a lower bound for the appropriately scaled Jacobian or design singular values.
- The formal ratio inverse supplies an initializer and a rank certificate, not an exact finite-wedge global inverse.
- The conditional four-dimensional geometry reduction of Section 2 is exact at arbitrary admissible wedges and beam angles; it is not asymptotic.

### Exact near-centered first-gauge bound

There is a sharper excitation bound along the exact first-prism centered gauge. At \(q=0\), with every wedge \(O(\kappa)\), the exact individual source-transport matrix from (1) is

\[
A_j=I+\frac{W_ju_j^T}{Z_jP_j}=I+O(\kappa^2).
\]

Indeed, \(X,B,W,u=O(\kappa)\), while \(Z,P\) remain bounded away from zero on a compact admissible tube. Their product also equals \(I+O(\kappa^2)\).

Parameterize a guarded first-prism deflection-level-set gauge by \(s\). Its tangent satisfies \(da_1/dn_1=O(\kappa)\). The gauge derivative of the total source-transport matrix is therefore \(O(\kappa^2)\), uniformly on the compact tube.

The centered output is exactly constant on this gauge. Write \(g_s\) for its optical configuration, distinguished from the scalar gap \(g\). Exact affinity in \(b\) gives

\[
F(g_s,b)-F(g_0,b)=[A(g_s)-A(g_0)]b,
\]

and hence

\[
\|F(g_s,b)-F(g_0,b)\|
\le C_G|b|\kappa^2|s|.
\]

For bounded observation error \(\epsilon\), this yields an unavoidable local indistinguishability scale proportional to

\[
\frac{\epsilon}{|b|\kappa^2},
\]

capped by the available gauge segment. A numerical minimax statement must specify the constant, the observation and parameter norms, and the gauge parameterization. Unlike a general amplitude-compensated hardware variation, this exact gauge eliminates the centered \(O(\kappa^3)\) angular remainder identically.

## 9. Audit status and remaining publication work

The supplied independent audits agreed on the required vector-recurrence coefficients, the centered gauge, the offset-excited ratio inverse, the negative-fundamental beam coefficient, the thirty-one-column finite design and the scaled full-rank argument. This appendix preserves those mathematical results and their limitations; no new numerical experiment was performed for this integration.

Related symbolic sources are now supplied with the [proof-check README](proof_checks/README.md) and [manifest](proof_checks/manifest.json). The manifest records matching supplied and saved SHA256 hashes for all seven files: three oblique physical-vector checks and their independent counterparts, plus a separately scoped canonical check. Their presence does not turn every derivation in this appendix into an executable proof. Packaging verified byte identity; no scripts were executed, imported, compiled or mathematically retested during this update. Their original `epsilon` wedge notation remains unchanged in the source, while this appendix uses \(\kappa\).

The exact forward graph and conditional affine geometry reduction hold on their explicitly guarded physical branches. The harmonic inverse is a formal leading-order result. The finite-record theorem establishes exact local rank for sufficiently small positive wedges on an off-axis witness family and generic rank on its analytic component. The companion global theorem and explicit compiler establish ambiguity-complete exact inference in principle for their finite exact encodings and sample times. Quantitative finite-angle error bounds, useful conditioning and practical treatment of weak or singular branches remain publication tasks. Universal global uniqueness on the original box is false because of the exact centered gauge; the complete inverse must retain those ambiguities.
