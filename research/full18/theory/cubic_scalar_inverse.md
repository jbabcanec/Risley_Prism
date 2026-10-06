# A scalar inverse for five formal cubic hardware invariants

2026-10-02. This note derives a triangular inverse from the supplied,
independently audited invariant identities. No new symbolic computation,
numerical experiment, or validation run is claimed here.

The result concerns five **formal cubic invariants**, labelled in physical
prism order. It does not identify those invariants with a finite-amplitude
optical record, establish their extraction from 200 positions, or recover
the other thirteen native coordinates. The original exact full18 inverse
remains the target of a subsequent exact-model correction and certification
stage.

These are invariants of the canonical independent-axis model. Vector
rotational equivariance forbids the pure third-harmonic torus channels used
by this canonical construction at a centered axial source. No physical
three-dimensional identification claim follows; see the model-scope
warning in [REPORT.md](../REPORT.md). A separate strict-monotonicity and
connectedness proof of uniqueness for this formal canonical map has been
reported, but is not yet incorporated in this note. The correspondence
proof below remains independent of that pending integration.

## 1. Hardware domain and the formal invariant map

The five hardware coordinates considered here are

\[
 (n_1,n_2,n_3,d,g)\in[1.3,1.8]^3\times[50,200]\times[2,15].
\]

Distances use the original native units; each prism has the fixed thickness
3. Define

\[
 F(n)=1+\frac{n}{(n-1)^2},\qquad
 \delta(n)=3\left(\frac1n-\frac1{n^3}\right),
\]

\[
 \begin{aligned}
 L_3&=d,\\
 L_2&=d+g+\frac3{n_3},\\
 L_1&=d+2g+\frac3{n_2}+\frac3{n_3},\\
 M_3&=L_3,\\
 M_2&=L_2-\delta(n_3),\\
 M_1&=L_1-\delta(n_2)-\delta(n_3).
 \end{aligned}
\]

Let the formal invariant inputs be

\[
 P_1=I_{300},\quad P_2=I_{030},\quad P_3=I_{003},\quad
 J_{12}=I_{210},\quad J_{23}=I_{021}.
\]

Their defining identities for this note are

\[
 2P_iL_i^2=F(n_i)-\frac{L_i-M_i}{L_i},\qquad i=1,2,3,       \tag{1}
\]

\[
 J_{23}=\frac{3d(n_3+1)-2L_2}{2dn_3L_2^2},                 \tag{2}
\]

\[
 J_{12}=\frac{3n_2M_2+3L_2-2L_1}{2n_2L_2L_1^2}.          \tag{3}
\]

The normalization and formal expansion underlying the symbols I are part of
the supplied identities. In particular, this note does not reinterpret I as
an uncorrected Fourier coefficient, a time derivative, or a polynomial fit
to the observed positions. Labels 1, 2, 3 refer to physical order, not a
sorting of speeds or spectral magnitudes.

## 2. Positivity and the single-valued material inverse

For n>1,

\[
 F'(n)=-\frac{n+1}{(n-1)^3}<0.
\]

Writing s=n-1>0, the equation F(n)=T becomes
`(T-1)s^2-s-1=0`. For T>1 its unique positive root gives

\[
 F^{-1}(T)=1+\frac{1+\sqrt{4T-3}}{2(T-1)}.                 \tag{4}
\]

The other quadratic root has s<0 and is outside the physical chart. The
native material interval is equivalent to

\[
 n\in[1.3,1.8]\quad\Longleftrightarrow\quad
 T\in\left[\frac{61}{16},\frac{139}{9}\right].             \tag{5}
\]

All five invariants in (1)--(3) are strictly positive on the native hardware
box. First, F(n)>1, while each subtracted quantity in (1) is nonnegative and
less than 6/L_i, with L_i>=50. Hence each P_i>0.

For (2), L_2=d+g+3/n_3<d+18<3d/2, whereas n_3+1>2. Thus
`3d(n_3+1)-2L_2>0`. Every denominator in (2) is positive.

For (3), cancellation in M_2 gives

\[
 M_2=d+g+\frac3{n_3^3}>0.
\]

Also

\[
 3L_2-2L_1=d-g+\frac3{n_3}-\frac6{n_2}>0,
\]

because d-g>=35 and n_2>1. Its numerator is consequently positive, as is its
denominator. Nonpositive exact input values for any P_i or J are therefore
incompatible with this native formal map. For interval-valued inputs, a
sign-indeterminate interval must be handled as an interval constraint, not
discarded on the sign of its midpoint.

## 3. Reduction to one scalar variable

Assume positive exact inputs `(P_1,P_2,P_3,J_{12},J_{23})`. Let

\[
 t=n_3\in[1.3,1.8].
\]

Every expression below is retained only when its listed domain filters hold.

**Last-prism distance.** Equation (1) with i=3 gives the unique positive
distance

\[
 d(t)=\sqrt{\frac{F(t)}{2P_3}}.                             \tag{6}
\]

Require `50<=d(t)<=200`.

**Middle effective distance.** Equation (2) is the quadratic

\[
 2J_{23}dtL_2^2+2L_2-3d(t+1)=0.                            \tag{7}
\]

Its leading coefficient is positive and its constant coefficient is
negative, so its two real roots have opposite signs. The unique positive
root, in rationalized form, is

\[
 L_2(t)=\frac{3d(t+1)}{1+\sqrt{1+6J_{23}d^2t(t+1)}}.        \tag{8}
\]

The radicand exceeds one and the denominator is positive. Formula (8)
avoids subtracting two nearly equal positive quantities; it is the same
root as the ordinary quadratic formula, not another branch choice.

**Gap and middle material.** Recover

\[
 g(t)=L_2-d-\frac3t,                                      \tag{9}
\]

and require `2<=g(t)<=15`. Then set

\[
 T_2(t)=2P_2L_2^2+\frac{\delta(t)}{L_2},\qquad
 n_2(t)=F^{-1}(T_2(t)).                                   \tag{10}
\]

Require `T_2(t)` to lie in (5). This both validates (4) and enforces the
entire original bound on n_2; it is not enough merely to require a real
square root.

**First effective distance.** Put

\[
 V(t)=2L_2-d+\frac3{n_2}-\frac3t.                          \tag{11}
\]

Substituting (9) proves exactly

\[
 V=d+2g+\frac3{n_2}+\frac3t=L_1>0.                        \tag{12}
\]

Thus V is not an independent geometry variable or an approximation to L_1.

**Scalar closure.** The remaining mixed invariant (3) is equivalent to

\[
 \boxed{\;
 \Psi(t)=2J_{12}n_2L_2V^2+2V-3L_2
                    -3n_2\bigl(L_2-\delta(t)\bigr)=0.
 \;}                                                     \tag{13}
\]

The equivalence uses only multiplication by the strictly positive
denominator `2n_2L_2V^2`. There is no squaring or discarded sign in (13).

**First material.** For each retained root of (13), define

\[
 T_1(t)=2P_1V^2+\frac{\delta(n_2)+\delta(t)}{V},\qquad
 n_1(t)=F^{-1}(T_1(t)).                                   \tag{14}
\]

Require T_1 to lie in (5). This is the remaining n_1 hardware filter.

The full admissibility filter is therefore: t in its closed native interval;
the positive branches in (6), (8), (10), and (14); d and g in their original
intervals; T_2 and T_1 in (5); and the exact scalar equation (13). All
denominators are then positive. The endpoints of the native intervals are
allowed. The radical-domain and material-range filters must remain visible
when implementing scalar root isolation.

## 4. Exact correspondence theorem for these five formal invariants

**Theorem.** Fix positive exact invariant inputs. Admissible roots t of
(13), filtered as above, are in bijection with native hardware vectors
`(n_1,n_2,n_3,d,g)` realizing all five identities (1)--(3).

**Proof, hardware to scalar root.** For a realizing hardware vector take
t=n_3. Equation (1) at i=3 and d>0 force (6). The positive-root uniqueness
of (7) forces (8). The definition of L_2 forces (9). Strict monotonicity of
F on n>1 forces the unique n_2 in (10). The definitions then give (12), so
(3) becomes (13). Finally (1) at i=1 forces (14). Every native filter is
satisfied by the starting hardware.

**Proof, scalar root to hardware.** Starting with an admissible t, formulas
(6), (8), and (9) satisfy the P_3 and J_{23} identities and the definition of
L_2. Formula (10) satisfies the P_2 identity. Identity (12) gives the correct
L_1, and (13) gives J_{12} after division by its positive denominator.
Formula (14) gives P_1. All recovered hardware lies in its original domain
by the filters. Each stage has only one allowed positive/material branch,
so a fixed t gives only one hardware vector. Distinct t values give distinct
hardware because their third refractive indices differ. These two
constructions are inverse to each other. QED.

This theorem does **not** say that Psi is monotone or that it has only one
admissible root. Several roots, multiple roots, or an exceptional continuum
are not excluded by the triangular formulas alone. A single Newton solve
would not establish completeness. P_1 enters the final n_1 reconstruction
and its filter; it does not enter the scalar closure equation itself.

## 5. Polynomial branch formulation

There is also a four-variable polynomial representation useful for exact
branch isolation. Set

\[
 (t,d,L,r)=(n_3,d,L_2,n_2),\qquad
 W=rt(2L-d)+3t-3r.
\]

On the physical branch, W=rtV. The scalar construction is equivalent to
the following equations together with the same domain and hardware filters:

\[
 2P_3d^2(t-1)^2-(t^2-t+1)=0,                              \tag{15}
\]

\[
 2J_{23}dtL^2+2L-3d(t+1)=0,                               \tag{16}
\]

\[
 \bigl[2P_2L^3t^3+3(t^2-1)\bigr](r-1)^2
                       -Lt^3(r^2-r+1)=0,                \tag{17}
\]

\[
 2J_{12}LW^2t+2Wt^2-3Lrt^3
                       -3r^2(Lt^3-3t^2+3)=0.            \tag{18}
\]

For (15), substitute `F(t)=(t^2-t+1)/(t-1)^2` into (6). For (17), use
`F(r)=(r^2-r+1)/(r-1)^2` in (10) and clear the positive denominator Lt^3.
For (18), substitute V=W/(rt) in (13), use
`delta(t)=3(t^2-1)/t^3`, and multiply by rt^3. Equation (16) is already (7).

The positive d branch, L>0, native t,r>1, recovered gap bound, V>0, and
final n_1 filter are essential. They remove extraneous branches that the
polynomial system considered over all complex or real coordinates permits.
Within these filters every cleared denominator is positive and the
polynomial equations are equivalent to the corresponding rational ones.

After expanding W, the total degrees are 4, 4, 8, and 8. On a generic
zero-dimensional complex fiber, the product-of-degrees Bezout bound is

\[
 4\cdot4\cdot8\cdot8=1024,
\]

counting multiplicity before real and physical filtering. This is a coarse
algebraic bound, not a count of actual native hardware solutions and not a
proof that every fiber is zero dimensional. Special invariant values may
have degeneracies or positive-dimensional components; those possibilities
must be detected rather than ruled out by the degree product.

For rational or algebraic invariant inputs, certified real polynomial
branch isolation is one possible route to enclosing all solutions. A scalar
implementation must likewise partition the admissible t domain, enclose
every root and retain multiple-root possibilities. Domain boundaries and
material-range filters cannot be replaced by a single favorable local
branch. If the invariant inputs are intervals, the problem is a family of
equations or inequalities over those intervals, not the equation obtained
by substituting their midpoint values.

## 6. What this contributes to the exact full18 problem

The result replaces a simultaneous five-hardware-variable inversion of
these formal cubic identities by a one-scalar branch problem, followed by
explicit formulas. It retains the complete native material, distance and
gap ranges within that formal map. It does not prove that finite-amplitude
measurements supply these five invariants without bias.

Three further steps are required before using it as a certified inverse of
the original passive record:

1. **Bound higher-order contamination and readout error.** Exact Snell optics
   has higher-order terms. Finite-amplitude cubic fits, finite-window Fourier
   coefficients and formal cubic coefficients are distinct objects. Their
   discrepancy, measurement error, rotor uncertainty, normalization error,
   and any dependence on other unknown source coordinates must be enclosed.
   The formal map's physical labels must also be related to actual prism
   order. No small-wedge or known-source assumption may silently replace
   the native full18 domain.
2. **Enclose every admissible scalar branch.** With bounded invariant inputs,
   propagate all compatible branches through (6)--(14), including multiple
   roots and domain boundaries. The displayed formulas alone give no
   uniform error modulus near a multiple root and no global uniqueness
   result. Positive-dimensional exceptional fibers require explicit
   treatment.
3. **Correct and certify against the exact ray graph.** The resulting
   hardware branches can supply seeds or boxes to the companion exact
   sparse cubic ray-graph formulation. That stage must retain its auxiliary
   ray states, exact interface and propagation equations, and strict physical
   branch predicates. Release all eighteen unknown physical coordinates and
   check all 200 original timed x,y measurements within their supplied hard
   error bands. A formal cubic match alone does not satisfy those exact
   constraints. Local correction of one branch also does not exclude other
   exact-model branches; a global cover or other complete exclusion argument
   is still needed.

The formal scalar inverse is therefore an explicit algebraic component for
constructing the method. Its correspondence theorem is complete for the
five supplied formal invariants, while its transfer to finite-noise exact
optical recovery remains a separate mathematical obligation.
