# Explicit branch-preserving six-root compiler

## Status, provenance and scope

This appendix integrates the supplied, independently audited compiler memo. It specifies a finite Boolean compiler for the exact, coupled three-prism vector-Snell model. It replaces the unspecified fixed-size quantifier-elimination precomputation in the [global constructive inverse theorem](global_inverse_completeness.md) with explicit polynomial operations and a radical-sign table. No radical expansion, quantifier elimination, cylindrical algebraic decomposition, numerical test or new symbolic audit was executed for this documentation integration.

The construction preserves all eighteen unknown parameters, signed wedges and speeds, the original bounds, and the original passive observations at \(t_k=k/20\), \(k=0,\ldots,199\). It is an exact finite-angle construction, not a low-order optical approximation. Every optical and traversal guard below is imposed at the measured times. Physical admissibility between samples is a separate obligation.

Measurement error is the symbolic nonnegative parameter \(\epsilon\). The componentwise observation condition is \(|F_{hk}-y_{hk}|\leq\epsilon\); unequal symbolic allowances \(\epsilon_{hk}\) are also permitted. The compiler does not assign a test noise level or establish a useful noise threshold. Wedge angles are denoted \(a_j\), not \(\epsilon\).

For exact algorithmic decisions, numerical input must have a finite exact encoding, such as rational data and noise bounds or the algebraic encodings specified in the companion theorem. A symbolic error parameter may be retained in the polynomial predicates and in a parametric decomposition. Complexity bounds stated specifically for eighteen shared variables refer to a fixed encoded error value; treating \(\epsilon\) as an additional free variable changes the fixed dimension to nineteen. Arbitrary real values available only through approximation oracles do not support unconditional exact sign decisions.

The compiler is constructive, but its output bounds are enormous. Neither an expanded compiler nor a practical complete solver for the 200-sample problem is supplied. The research remains a draft requiring expert review; the reported independent audit is not external peer review or a proof-assistant certificate.

## 1. Scaled ray directions and the six-root tower

Let the incident transverse slope be

\[
t=(\tan\beta_x,\tan\beta_y),\qquad Q=1+|t|^2,
\qquad X_0=t,\quad Z_0=1,\quad L_0=1.
\]

Here angles in trigonometric expressions are in radians, with the native degree conversion understood when required. The initial physical unit direction is \((t,1)/\sqrt Q\). Every external direction is represented using the same scale \(\sqrt Q\): at stage \(j-1\), the scaled direction is

\[
\frac{(X_{j-1},Z_{j-1})}{L_{j-1}},
\qquad L_{j-1}>0,
\]

and its squared norm is \(Q\). Dividing this vector by \(\sqrt Q\) gives the physical unit direction. The common scaling leaves all propagation slopes unchanged.

For the instantaneous exit-plane tilt vector \(u_j\in\mathbb R^2\), define

\[
\begin{aligned}
D_j&=1+|u_j|^2,\\
H_j&=\sqrt{n_j^2Q L_{j-1}^2-|X_{j-1}|^2},\\
P_j&=H_j-u_j\cdot X_{j-1},\\
E_j&=\sqrt{P_j^2-D_j(n_j^2-1)Q L_{j-1}^2},\\
Z_j&=H_jD_j-P_j+E_j,\\
X_j&=X_{j-1}D_j+u_j(P_j-E_j),\\
L_j&=L_{j-1}D_j.
\end{aligned} \tag{1}
\]

Both square roots are the positive roots. At every stage require

\[
n_j^2Q L_{j-1}^2-|X_{j-1}|^2>0,
\quad
P_j^2-D_j(n_j^2-1)Q L_{j-1}^2>0,
\quad P_j>0,
\quad Z_j>0. \tag{2}
\]

The six roots form the ordered tower

\[
H_1,E_1,H_2,E_2,H_3,E_3.
\]

Each radicand uses only earlier roots and ordinary inputs. Since \(D_j>0\), induction gives \(L_j>0\).

To verify the refraction recurrence, first suppress the previous denominator. Tangential optical-momentum conservation is

\[
B-X+u(Z-H)=0.
\]

With \(P=H-u\cdot X\), \(D=1+|u|^2\), the transmitted normal numerator satisfies

\[
R^2=P^2-D(n^2-1)Q,
\qquad R=DZ-HD+P.
\]

Selecting \(R>0\) and clearing the previous positive denominator gives (1). In the denominator representation,

\[
\frac{Z_j-u_j\cdot X_j}{L_j}
=\frac{E_j}{L_{j-1}}>0. \tag{3}
\]

Here \(E_j/L_{j-1}\) is the outgoing scaled direction's unnormalized normal residual, obtained by taking its dot product with \((-u_j,1)\). The physical unit direction's component along the unit normal is \(E_j/[L_{j-1}\sqrt{Q D_j}]\). Both quantities have the same positive sign. The guards select transmission and forward axial propagation, excluding total internal reflection and grazing incidence.

## 2. Exact positions, positive denominators and traversal order

Start with the entrance position

\[
p_0=b+6t,
\qquad b=(p_x,p_y).
\]

For prism \(j\), take \(\ell_j=g,g,d\), respectively. From its entrance position \(p\), compute

\[
\begin{aligned}
p_{\rm exit}
&=p+X_{j-1}\frac{3+u_j\cdot p}{P_j},\\
p_{\rm next}
&=p_{\rm exit}
 +\frac{X_j}{Z_j}\bigl(\ell_j-u_j\cdot p_{\rm exit}\bigr).
\end{aligned} \tag{4}
\]

These are the exact tilted-exit-plane intersection and subsequent propagation to the next flat plane. The reference axial prism thickness is \(3\). Enforce

\[
3+u_j\cdot p>0,
\qquad
\ell_j-u_j\cdot p_{\rm exit}>0. \tag{5}
\]

The first inequality, together with \(H_j,P_j>0\), gives positive internal axial travel. The second gives positive external axial travel. Thus the stated exit and next flat plane are encountered in their required physical order.

All position denominators can be products of the positive quantities \(P_j\) and \(Z_j\). Clearing these denominators preserves strict, weak and equality tests. Do not rationalize them using conjugate norms: a conjugate factor can vanish even when the physical denominator is positive. No division by such a norm is part of this construction.

Equations (1), (4), and guards (2), (5) reproduce the unique transmitted branch of the full polynomial ray graph. They admit no reflective, backward, grazing or total-internal-reflection roots. If an alternative mathematical model deliberately omits sequential traversal, it must explicitly omit (5) and retain that convention throughout; the theorem cannot silently switch between the two domains.

## 3. Weighted degree bounds

Give each instantaneous input component \(t,b,u_j,n_j,g,d\), each measured value and each error allowance weight one. Thus \(Q\) has weight two. Assign root weights

\[
\operatorname{wt}(H_j)=2j,
\qquad \operatorname{wt}(E_j)=2j+1.
\]

Induction in (1) gives

\[
\begin{aligned}
\operatorname{wt}(L_j)&\leq2j,\\
\operatorname{wt}(X_j),\operatorname{wt}(Z_j)&\leq2j+2,\\
\operatorname{wt}(P_j)&\leq2j+1.
\end{aligned} \tag{6}
\]

Each defining radicand has weight at most twice its root's assigned weight. Reduction by \(r^2=R\) therefore never increases weighted degree.

For positions, write \(p=A/T\) with \(T>0\), and degree bounds \(a=\operatorname{wt}(A)\), \(b_T=\operatorname{wt}(T)\), where \(a\geq b_T\). The symbol \(b_T\) is a degree bound, not the source position. Initially \(a=1\), \(b_T=0\). Equation (4) admits the numerator and denominator representation

\[
\begin{aligned}
A_{\rm exit}
 &=P_jA+X_{j-1}(3T+u_j\cdot A),\\
T_{\rm exit}&=P_jT,\\
A_{\rm next}
 &=Z_j A_{\rm exit}
 +X_j(\ell_jT_{\rm exit}-u_j\cdot A_{\rm exit}),\\
T_{\rm next}&=Z_jT_{\rm exit}.
\end{aligned} \tag{7}
\]

Consequently,

\[
\begin{aligned}
\operatorname{wt}(A_{\rm exit})&\leq a+2j+1,\\
\operatorname{wt}(T_{\rm exit})&\leq b_T+2j+1,\\
a_{\rm next}&\leq a+4j+4,\\
(b_T)_{\rm next}&\leq b_T+4j+3.
\end{aligned}
\]

After three prisms,

\[
\boxed{\operatorname{wt}(A_3)\leq37,
\qquad \operatorname{wt}(T_3)\leq33.} \tag{8}
\]

Every componentwise observation-strip test has weighted degree at most \(37\) after positive-denominator clearing. For example,

\[
A_{3,h}-(y_h+\epsilon_h)T_3\leq0
\]

is the upper strip. The lower strip is treated identically. All branch and traversal guards have lower degree. A separate squared Euclidean residual at one sample,

\[
|A_3-yT_3|^2-\epsilon^2T_3^2\leq0,
\qquad \epsilon\geq0,
\]

has weighted degree at most \(74\). This per-sample ball does not assert an analogous compilation bound for a global noise constraint coupling all samples.

## 4. Explicit radical-sign elimination

At a tower level \(r=\sqrt R>0\), retain all earlier radicand-domain guards. Reduce the expression using \(r^2=R\) to

\[
E=A+Br,
\qquad C=A^2-B^2R,
\]

where \(A,B,C\) involve only earlier roots. Its sign follows from this finite table:

| Case | Exact sign of \(E\) |
| --- | --- |
| \(B=0\) | \(\operatorname{sgn}(A)\) |
| \(A=0\), \(B\ne0\) | \(\operatorname{sgn}(B)\) |
| \(A,B\) have the same nonzero sign | Their common sign |
| \(A,B\) have opposite signs | \(\operatorname{sgn}(A)\operatorname{sgn}(C)\) |

The first row includes \(A=B=0\). In the opposite-sign case, comparing \(|A|\) with \(|B|\sqrt R\), or multiplying by the conjugate factor whose sign is then known, proves the last row. In particular \(C=0\) correctly detects cancellation. These rules preserve zero, strict and weak tests; replacing them with squaring alone would not.

Apply the rule recursively to the six levels, including the positive-radicand guards themselves in the final conjunction. The identities are valid only on those guarded domains. A failed or zero radicand is not an admissible branch recovered by a different sign choice.

An expression of weighted degree \(d\) produces expressions of degree at most \(2d\) one level down. After all six levels, terminal ordinary input polynomials have degree at most \(64d\). Therefore

\[
\boxed{D_{\rm interval}\leq37\cdot64=2368,}
\qquad
\boxed{D_{\rm per\text{-}sample\ ball}\leq74\cdot64=4736.} \tag{9}
\]

At most \(3^6=729\) terminal polynomial sign tests are needed per original atom in a shared sign-circuit representation. This is not a bound of \(729\) on occurrences in a naively expanded textual Boolean formula: repeated conditions should remain shared, and expansion may duplicate them.

For the interval model there are at most \(24\) primitive atoms: eighteen branch/traversal tests, four output-strip tests, and two nonnegative per-axis error-bound tests. The branch/traversal count is six per prism: two positive radicands, \(P_j>0\), \(Z_j>0\), and the two inequalities in (5). With one common \(\epsilon\), one of the two error-bound atoms is redundant. A safe bound before deduplication is

\[
\boxed{24\cdot729=17\,496}
\]

terminal sign-polynomial occurrences, excluding the separate prior-box predicate. Radicand guards are themselves compiled with the same conservative procedure. Their presence is essential, even if some become redundant on a particular prior region.

Every preprocessing operation is explicit: polynomial addition and multiplication, reduction by six monic quadratic relations, and use of the finite sign table. This route needs neither auxiliary-variable quantifier elimination nor a genericity assumption. It specifies a compiler; it does not claim that the full compiler output has been generated here.

## 5. Exact rotor substitution and the finite-record formula

Use the bijective rational chart from the [global theorem](global_inverse_completeness.md):

\[
r_j=\tan(\pi a_j/180),\quad
f_j=\tan(\pi\phi_j/360),\quad
v_j=\tan(\pi N_j/20),
\]

where the native wedge \(a_j\) and phase \(\phi_j\) are in degrees and speed \(N_j\) is in hertz. At sample index \(k\), the exact complex exit-plane slope is

\[
u_{j,x}(k)+i u_{j,y}(k)
=r_j\frac{(1+if_j)^2(1+iv_j)^{2k}}
{(1+f_j^2)(1+v_j^2)^k}. \tag{10}
\]

Every denominator is strictly positive for every real chart point. Its degree is \(2k+2\); the numerator degree is at most \(2k+3\). The imaginary-unit notation simply specifies two real polynomials.

Substituting (10) into the terminal sign polynomials and clearing positive denominators therefore preserves every Boolean connective and every strict, weak or equality sign. Conjoining the resulting predicates at all sample indices, together with the original chart prior box, gives an explicit formula for every compatible system. It includes singular, disconnected and positive-dimensional fibers; the compiler does not assume nonzero wedges, distinct frequencies, local rank or a unique inverse.

For an instantaneous polynomial of ordinary degree \(D\), multiply by the positive product of all three rotor denominators, each raised to power \(D\). If a monomial has total tilt degree \(m\), then its substituted degree is bounded by

\[
(D-m)+m(2k+3)+(3D-m)(2k+2)=D(6k+7). \tag{11}
\]

For \(K\) consecutive samples indexed from zero, the maximum is thus

\[
\boxed{D(6K+1).} \tag{12}
\]

At \(D=2368\), \(K=200\), this conservative bound is

\[
\boxed{2\,843\,968.}
\]

For rational data of bit size \(\tau\), integer coefficient heights after denominator clearing are \(O(K+\tau)\), with constants depending on the fixed compiler. The exact input and algebraic-endpoint qualifications in the global theorem remain necessary. For a general rational clock converted to an integer lattice, the analogous degree depends on the largest lattice index, not merely on the bit length of that index.

The resulting complete real-algebraic decomposition has fixed dimension but astronomical worst-case complexity. Degree \(2368\) instantaneous polynomials and the terminal-count bound do not make the eighteen-variable inverse tractable. The explicit compiler improves proof specificity and gives an implementable sequence of operations in principle; it is not a completed practical implementation or a numerical runtime bound.

## 6. Alternate 25-auxiliary cubic graph

The same scaled-direction convention provides a root-free single-sample graph with \(25\) auxiliary variables: \(Q\), plus eight per prism,

\[
(H,B_x,B_y,Z,p_{{\rm next},x},p_{{\rm next},y},P,\alpha).
\]

Here the scaled incoming and outgoing direction vectors have squared norm \(Q\); this is a direct graph representation rather than the denominator bookkeeping of (1). Start with

\[
Q=1+|t|^2,\qquad X_0=t,\quad Z_0=1,
\qquad p_0=b+6t.
\]

At each prism, impose

\[
\begin{aligned}
H^2+|X|^2&=n^2Q,\\
|B|^2+Z^2&=Q,\\
B-X+u(Z-H)&=0,\\
P&=H-u\cdot X,\\
\alpha&=u\cdot p,\\
ZPp_{\rm next}
&=ZPp+ZX\alpha-HB\alpha+3ZX
 -3(u\cdot X)B+\ell PB.
\end{aligned} \tag{13}
\]

The direction and position equations in (13) are vector equations where appropriate. Every scalar equation has ordinary total degree at most three, including \(n^2Q\), \(ZX\alpha\), and \((u\cdot X)B\).

Preserve the strict inequalities

\[
H>0,\quad Z>0,\quad P>0,\quad Z-u\cdot B>0,
\quad3+\alpha>0,
\quad(3+\ell)P-H(3+\alpha)>0. \tag{14}
\]

The internal axial travel is \(h=H(3+\alpha)/P\). Eliminating \(h\) from the exact two-segment displacement yields the final equation in (13). The last two inequalities in (14) enforce positive internal and subsequent external travel, respectively. Thus this graph has the same unique transmitted physical lift as the original 26-auxiliary unit-direction graph in the companion theorem. Both counts and representations are valid; they should not be conflated.

## 7. What the compiler establishes and what remains open

The integrated derivation establishes an explicit six-root optical compiler with branch-preserving sign elimination, exact traversal constraints, conservative degree and sign-count bounds, and exact substitution of every sampled rotor orientation. Combined with complete real-algebraic decomposition under the stated finite input model, it supplies a terminating ambiguity-complete inverse in principle. No practical runtime, successful global computation or favorable numerical conditioning follows from those statements.

In particular, this appendix does not establish global uniqueness, a useful symbolic-error threshold for all eighteen native coordinates, or recovery within the \(0.001\) native target for an actual record. Those are compatible-set and conditioning questions, not consequences of compiler correctness. The companion global theorem supplies the exact conditional coordinate-diameter criterion; quantitative data-conditioned bounds require the separate symbolic-\(\epsilon\) analysis. All unresolved physical branches remain in the compatible-set description.

Independent audit reported agreement with the six-root recurrence, positive position-denominator construction, weighted degree bounds, sign table and alternate cubic graph. This appendix records that supplied audit status. It does not report a new executed derivation check, a physical experiment or a completed all-branches computation.
