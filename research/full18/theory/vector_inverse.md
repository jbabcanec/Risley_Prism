# Physical-vector inverse theory for three rotating prisms

## Status and scope

This memo establishes a constructive small-wedge, off-axis witness for full rank of the **exact physical-vector 200-sample, 18-parameter forward map**. It also gives an exact root-free forward graph and conditional four-dimensional affine geometry reduction. It does not establish global uniqueness of the exact finite-wedge inverse, a numerical small-wedge threshold, or uniform conditioning over the full prior box.

The previously developed independent-axis canonical model remains a mathematically valid, different model. Its pure-third-harmonic hardware inverse must not be presented as a theorem about vector Snell optics. The physical construction below replaces those channels with offset-excited second harmonics and complex negative fundamentals.

The vector Snell law used here is standard; an official primary technical reference is Ken Moore, Ansys/Zemax, [What is a ray?](https://optics.ansys.com/hc/en-us/articles/42661810210707-What-is-a-ray), section “Refraction, reflection, and diffraction.” The inverse constructions and proofs below are derived here, not attributed to that source.

## 1. Exact physical model

Use transverse vectors in R² or the corresponding complex coordinate x+iy. The source is (p_x,p_y,0), with complex transverse position b=p_x+i p_y. Incident direction is the normalization of

(tan beta_x, tan beta_y, 1).

Its transverse direction cosine is q and its positive axial component is sqrt(1-|q|²). At beta=0, the derivative from beta_x+i beta_y to q is the identity.

For prism j, let gamma_j(t)=2 pi N_j t+phi_j, and define

u_j=tan(a_j)(cos gamma_j, sin gamma_j).

The flat entrance planes are z=6, 9+g, 12+2g. The exit planes are

z=9+u_1 dot p,
z=12+g+u_2 dot p,
z=15+2g+u_3 dot p.

Thus each prism has axial vertex thickness 3. The screen is z=15+2g+d. The forward-facing exit unit normal is (-u_j,1)/sqrt(1+|u_j|²). The external index is one; the three glass indices are n_j.

The stated hardware box is n_j in [1.3,1.8], g in [2,15], d in [50,200]. The other stated bounds include wedge/phase angles within ±18 degrees, speeds within ±3.5 Hz, beam angles within ±25 degrees and source coordinates within ±5. The rank witness uses interior positive, sufficiently small wedges, zero nominal beam tilt, and a fixed nonzero source offset inside its box.

All equations use the forward transmitted branch, exclude total internal reflection and grazing incidence, and require forward axial direction. If physical sequential surface traversal is required, also enforce positive propagation lengths and that the next flat entrance/screen occurs after the current exit. The small-wedge witness satisfies these strict conditions.

## 2. Exact root-free local graph

At a flat prism entrance, let X=(X_1,X_2) be the incoming external transverse direction cosine. Introduce H>0 and G=n² with

H²+|X|²=G.

The internal unit direction is (X,H)/n. Let the outgoing external unit direction be (B,Z), with B in R². The exact exit conditions are

B-X+u(Z-H)=0,
|B|²+Z²=1,
Z>0.

Introduce

P=H-u dot X>0,
R=Z-u dot B>0.

The tangential equations preserve optical momentum along both exit-plane tangents; R>0 selects the transmitted normal branch. Together with H>0 and Z>0 these remove the unwanted algebraic roots.

For transverse entrance position p, define

W=ZX-HB,
V=ZX-(u dot X)B.

Let ell be the axial gap from the exit vertex to the next flat plane. The exact position equation is

ZP p_next = ZP p + W(u dot p) + 3V + ell P B.       (1)

To verify it, the axial internal traversal is

h=H(3+u dot p)/P.

Then

p_next=p+(3+u dot p)X/P+(3+ell-h)B/Z,

which rearranges to (1). This verification also shows which positive-traversal inequalities must accompany the graph if surfaces must be encountered in the stated physical order.

Declare dot products and W,V,P as auxiliary variables. The graph can then be written with polynomial equations of degree at most three: for example, ZP p_next is cubic, W(u dot p) is cubic when W is auxiliary, and ell P B is cubic. No square-root evaluations are needed in the graph itself; positive inequalities select the intended roots. Initial propagation can likewise be represented by Z_0 p_entry=Z_0 b+6 X_0, with |X_0|²+Z_0²=1 and Z_0>0.

### Conditional affine geometry reduction

For fixed directions and optical parameters, (1) is affine in p and ell. Directions are independent of positions, g and d. Applying ell=(g,g,d), and initial p_entry=b+6q/sqrt(1-|q|²), therefore gives the exact final map

F(t)=A(t)b+B_0(t)+g C(t)+d D(t).                  (2)

Here A is a 2-by-2 real matrix; the remaining terms are two-vectors. Their coefficients depend only on the fourteen optical parameters: three indices, wedges, phases and speeds, plus two beam angles. The four remaining geometry variables are p_x,p_y,g,d.

Thus fixed-optics, componentwise interval data produce a four-dimensional linear feasibility problem, augmented by the geometry box and any affine surface-order inequalities. Certified source projection/LP elimination survives vector coupling. The earlier independent-axis normalized-polygon formula must not be copied without rederiving its coupled-vector counterpart. Euclidean residual bounds give second-order-cone rather than linear inequalities.

## 3. Centered obstruction and canonical discrepancy

For an axial ray through the first prism's on-axis vertex, the exit point is fixed at (0,0,9) and its output deflection is

delta_1=asin(n_1 sin a_1)-a_1.

Its entire output ray depends on n_1 and a_1 only through delta_1. All downstream optics consequently preserve an exact one-dimensional level-set gauge. A tangent is

da_1/dn_1=-sin a_1/[n_1 cos a_1-sqrt(1-n_1² sin² a_1)].

Full18 rank cannot hold at that centered physical witness.

More generally, at a centered axial source, simultaneous rotation of every rotor rotates the screen vector:

Z(gamma+t 1)=exp(it)Z(gamma).

Hence a complex torus Fourier coefficient indexed by m vanishes unless sum(m_j)=1. Pure third harmonics vanish exactly, while 2 gamma_i-gamma_j and gamma_i+gamma_j-gamma_k are allowed.

For a single prism, set r=sin a, s=r(cos gamma,sin gamma), k=n-1 and B=k(n²-n+1)/2. The centered single-prism slopes are

canonical: k s+B(s_x³,s_y³)+O(r⁵),
physical:  k s+B|s|²s+O(r⁵).

The canonical complex slope has the artificial third-harmonic term (B r³/4)exp(-3i gamma). The leading maximum slope-vector discrepancy is B r³/2. It is of the same order as the canonical cubic identification information, so a relative O(a²) error in total deflection does not justify cubic hardware inference for physical optics.

## 4. Reproducible cubic vector recurrence

This recurrence supplies the needed physical coefficients without numerical ray-tracing experiments. Write complex rotor s_j=sin(a_j)exp(i gamma_j), k_j=n_j-1. Grade b,q,s as degree one for this formal recurrence. Denote incoming transverse direction by x_1+x_3+O(5), and entrance position by r_1+r_3+O(5).

Initially,

x_1=q, x_3=0,
r_1=b+6q, r_3=3|q|²q.

The last term comes from exact source-to-first-plane propagation. It has no effect on the off-axis coefficients or linear beam derivatives used below, but is needed for a complete cubic recurrence.

At prism j, abbreviate n=n_j, k=n-1, s=s_j, and use ell=g for j=1,2 and ell=d for j=3. Then

y_1=x_1+k s,

y_3=x_3+[k/(2n)]|x_1|²s+[k/2]|s|²x_1
          +[k/2]s² conjugate(x_1)+[kn/2]|s|²s,

r_1_next=r_1+3x_1/n+ell y_1,

r_3_next=r_3+3x_3/n+ell y_3
          +3|x_1|²x_1/(2n³)+(ell/2)|y_1|²y_1
          -k(x_1/n+s) Re[conjugate(s)(r_1+3x_1/n)].      (3)

Set x_1_next=y_1 and x_3_next=y_3. Equations (3) follow by expanding the exact graph in Section 2. The direction formula uses direction cosines; the |y_1|²y_1/2 term converts outgoing direction cosine to air slope.

Because the exact position map is affine in b, the b-dependent terms extracted from this joint-cubic recurrence are also the full leading wedge-order coefficients for fixed b, rather than requiring b to shrink with the wedges.

## 5. Offset-excited hardware inverse

Set q=0, but retain an unknown nonzero fixed source offset b. Let h_j=sin(a_j)exp(i phi_j), so s_j=h_j exp(2 pi i N_j t). At leading quadratic wedge order, the offset-dependent screen correction is

- sum_j k_j(s_j dot b)(P_{j-1}/n_j+s_j),
P_{j-1}=sum_{i<j} k_i s_i,

where s dot b=Re(conjugate(s)b).

Thus the self second-harmonic and positive mixed-sum coefficients are

S_j=-conjugate(b) k_j h_j²/2,                    (4)
M_ij=-conjugate(b) k_i k_j h_i h_j/(2n_j), i<j. (5)

These are leading homogeneous coefficients, not exact finite-wedge coefficient identities. In exact torus coefficients their next corrections have wedge degree four.

The phase- and offset-free ratios are

R_ij=M_ij²/(S_i S_j)=k_i k_j/n_j².

Define r=R_13/R_23 and u=sqrt(R_12/r). Then

k_2=u/(1-u), n_2=1/(1-u), k_1=r k_2.

Let c=R_23/k_2. It determines k_3 from c=k_3/(1+k_3)². On k_3 in [0.3,0.8] this is strictly increasing, so the physical branch is

k_3=[1-2c-sqrt(1-4c)]/(2c).

The other root is its reciprocal and lies outside this interval. The inverse is smooth in the interior physical box.

The leading fundamental gains are

K_1=k_1(d+2g+3/n_2+3/n_3),
K_2=k_2(d+g+3/n_3),
K_3=k_3 d.                                      (6)

DC gives b at leading order. Equations (4), the recovered indices and nonzero b recover wedge magnitudes; fundamentals recover phases and gains. Then d=K_3/k_3 and g=K_2/k_2-d-3/n_3. K_1 provides redundancy.

An exact one-prism illustration explains the excitation. For axial incidence at offset b, screen distance ell from the exit vertex, D=tan[asin(n sin a)-a] and h=(tan a)D/2,

Z=(1-h)b+ell D exp(i gamma)-h conjugate(b)exp(2i gamma).

Unknown nonzero b therefore supplies genuine second-harmonic index/wedge information that vanishes at b=0.

## 6. Separating the two unknown beam angles

At q=0 the exact full-vector map is affine in b:

Z=Z_0+A b+B conjugate(b).

Simultaneous rotor rotation assigns torus weights 1,0,2 to Z_0,A,B respectively. Consequently every complex negative-fundamental coefficient indexed by -e_j vanishes identically at q=0, for every other parameter value. This assertion concerns the complex coordinate Z=x+iy; scalar real-coordinate Fourier symmetry alone does not identify this channel.

Its leading tilt derivative is

C_{-e_j}= -[k_j/(2n_j)] b q conjugate(h_j)+higher wedge orders. (7)

For b and h_j nonzero, (7) is a nonzero complex-linear map of q, hence supplies both beam-angle directions. It follows directly from the last term of (3), and also from direct first-order intersection geometry. Independent derivations checked the sign and factor.

Exact torus functionals are not assumed accessible from finitely many samples. The next section replaces that assumption with a finite fixed linear extractor used solely for the rank proof.

## 7. Exact 200-sample full18 local-rank theorem

### Explicit design

Use t_k=k/20, k=0,...,199, and witness frequencies

(N_1,N_2,N_3)=(1,5,25)/20 Hz.

For the 25 integer labels |m|_1<=2, set

z_m=exp[2 pi i(m_1+5m_2+25m_3)/400].

These nodes are distinct. A difference of two labels has l1 norm at most four, whereas a nonzero integer base-five relation u_1+5u_2+25u_3=0 has l1 norm at least six. Exponents range from -50 to 50, so no additional modulo-400 alias occurs.

Form a 200-by-31 complex dictionary from the 25 columns z_m^k and the six confluent columns k z_m^k for m=+/-e_j. The first 31 rows are a nonsingular confluent Vandermonde matrix. Hence a fixed left inverse exists. Its realification can be applied to the observed pair of screen coordinates.

The extractor is fixed at this witness and used to certify a Jacobian. It is not a proposed estimator that assumes the unknown frequencies have already been recovered.

### Coordinates and scaling

Fix any interior hardware h=(n_1,n_2,n_3,g,d), any nonzero b in its allowed box, q=0, and three nonzero complex A_j. Parameterize

h_j=sin(a_j)exp(i phi_j)=epsilon A_j/K_j(h).

Choose phases in their allowed interior and sufficiently small epsilon>0. This is a smooth local change of variables from nonzero wedge magnitudes/phases to complex amplitudes A_j, with five hardware coordinates unchanged.

The eighteen real coordinates are six amplitude components, three frequencies, five hardware parameters, two source-offset components and two beam components.

Scale the corresponding Jacobian columns by epsilon^-1, epsilon^-1, epsilon^-2, one and epsilon^-1, respectively. Before beam scaling, compensate zero-wedge beam drift by adding source-offset variation

delta b=-B_0 delta q,
B_0=6+2g+d+3 sum_j(1/n_j).

B_0 has units of length, so this compensation is dimensionally consistent. It is an invertible triangular column transformation.

### Limiting rank

The fixed extractor produces the following limiting blocks:

1. Negative fundamentals: only the two beam columns survive, with a nonzero complex multiplier from (7).
2. DC: the source-offset block is the identity.
3. Fundamentals and confluent fundamentals: six amplitude directions and three frequency directions are independent. Nonzero A_j are essential.
4. Quadratic self/sum coefficients: the five hardware columns are independent.

For the last assertion, hold A,b fixed. The normalized coefficients are

S_j=-conjugate(b) k_j A_j²/(2K_j²),
M_ij=-conjugate(b) k_i k_j A_i A_j/(2n_j K_i K_j).

Their ratios recover the three indices as in Section 5. Then

K_j=|A_j| sqrt[|b| k_j/(2|S_j|)]

on the positive-gain branch, and (6) recovers d and g. This is a smooth left inverse of the five-dimensional hardware map, proving its differential has rank five.

Hardware columns may have a nonzero limiting DC component. Eliminate it using the source-offset identity block; unscaled offset columns have vanishing quadratic block in this limit. Beam columns may have positive-fundamental components, removable using the fundamental blocks. These crossblocks do not reduce rank.

### Remainder transfer

At q=0, angular output is odd in the wedges and the affine source correction is even. The retained angular term is degree one with next term degree three; the retained source-dependent term is degree two with next term degree four. Their C1 remainders vanish after the stated column scalings. A differentiated quadratic frequency term is O(epsilon²), so after epsilon^-1 frequency-column scaling it vanishes; repeated quadratic nodes are unnecessary.

After baseline compensation, beam derivatives start at O(epsilon), followed by higher wedge orders. Their epsilon^-1-scaled remainder vanishes. The beam negative-fundamental block is nonzero by (7).

Therefore the limiting transformed Jacobian has rank 2+2+9+5=18. For all sufficiently small positive epsilon, continuity implies that the **exact**, physically coupled, 400-real-output sampled Jacobian has rank eighteen.

This establishes local identifiability at an explicit family of admissible off-axis witnesses. A nonzero analytic minor also gives generic rank on the connected analytic branch containing such a witness. It does not prove global uniqueness, uniform rank at every parameter point, or a specific usable numerical epsilon threshold.

## 8. Conditioning, excitation and inference limits

- The centered axial configuration has an exact first-prism gauge. Any practical stability statement must quantify excitation away from it.
- With fixed nonzero b, the new hardware-separating channels scale as |b| epsilon². The beam-identifying negative fundamentals scale as |b| epsilon times beam perturbation.
- These orders improve on the canonical cubic mechanism but do not alone prove useful numerical conditioning. Gain normalization, length units, parameter-box scaling and the finite dictionary's smallest singular value matter.
- The explicit base-five design is a rank certificate, not a claim of optimal sampling or good conditioning.
- Exact torus quadratic coefficients have O(epsilon⁴) corrections. The finite 31-column extractor can leak unrepresented cubic angular terms into its quadratic rows. Its safe normalized hardware remainder is therefore O(epsilon), unless an additional cancellation or richer dictionary is proved.
- Fundamental extraction after accounting for degree-two modes has O(epsilon²) relative higher-order contamination. A naive fundamental-only fit can instead absorb quadratic sidelobes; no universal frequency-estimation bias follows without specifying the estimator and design.
- Near b=0 or other weak-excitation configurations, noise amplification is unavoidable. A quantitative theorem requires explicit remainder bounds and a lower bound on the appropriately scaled Jacobian/design singular values.
- The formal ratio inverse is not an exact finite-wedge global inverse. It can provide a principled initializer and a rank certificate for subsequent exact-model inference.
- The exact four-dimensional conditional geometry reduction in Section 2 remains valid at arbitrary admissible wedges and beam angles; unlike the harmonic inverse, it is not asymptotic.

## 9. Audit summary

Independent checks agreed on the vector recurrence's required coefficients, centered gauge, offset-excited ratio inverse, negative-fundamental beam coefficient, 31-column finite design, and scaled full18 rank argument. No empirical sweep or ray-tracing experiment is required for these proofs. Remaining publication work is quantitative certification of finite-angle error/conditioning and careful separation of local rank, formal hardware inversion and global exact-data uniqueness.

### Exact near-centered first-gauge lower bound

There is a sharper excitation statement along the exact first-prism centered gauge. For q=0 and all wedges O(epsilon), the exact individual source-transport matrix from (1) is

A_j=I+W_j u_j^T/(Z_j P_j)=I+O(epsilon²),

because X,B,W,u are O(epsilon), while Z and P stay bounded away from zero. Their product also equals I+O(epsilon²). Along a guarded first-prism deflection-level-set gauge parameterized by s, da_1/dn_1=O(epsilon), so the gauge derivative of the total source-transport matrix is O(epsilon²), uniformly on a compact admissible tube.

The centered output is exactly constant on this gauge. By exact affinity in b,

F(g_s,b)-F(g_0,b)=[A(g_s)-A(g_0)]b,

and hence

||F(g_s,b)-F(g_0,b)|| <= C_G |b| epsilon² |s|.

Under bounded observation error eta, this supplies an unavoidable local indistinguishability scale proportional to eta/(|b| epsilon²), capped by the available gauge segment. The constant and norm must be specified for a numerical minimax statement. Unlike a general amplitude-compensated hardware variation, this exact gauge eliminates the centered O(epsilon³) angular remainder identically.
