# Exact bounded-noise elimination of the four geometry variables

This construction uses the original ordered, independent-axis three-prism model, one passive screen, and its 200 time-labelled two-dimensional positions. All eighteen hardware coordinates remain unknown. It adds no parked states, second screen, calibrated glass, approximate truth, or narrower parameter prior.

Fix the timestamp contract first: the identities hold for any declared timestamps, including exact `k/20`, or the exact binary64 values supplied as data. A proof for one contract does not silently cover the other. Below, `m=400` scalar observations are interleaved by axis and satisfy a hard componentwise error bound `eta >= 0`.

Write

\[
q=(N_1,N_2,N_3,a_{x1},a_{x2},a_{x3},a_{y1},a_{y2},a_{y3},n_1,n_2,n_3,b_{ax},b_{ay}),
\quad \ell=(d,g,p_x,p_y).
\]

The nonlinear vector has fourteen coordinates. The native affine box is

\[
\mathcal L=[50,200]\times[2,15]\times[-5,5]^2.
\]

The results below exactly eliminate the *unknown* affine variables for each possible `q`. Fixing `q` when evaluating this elimination is not an assertion that those fourteen physical parameters are known. They still range over their entire native domain in the inverse problem.

## 1. Exact optical identity and a positive source coefficient

At one timestamp and one axis, let `X` be the incoming unit transverse direction at a prism's flat entry face. Put

\[
H=\sqrt{n^2-X^2},\qquad
s=\sin(a_x)\cos\gamma\quad\text{or}\quad\sin(a_x)\sin\gamma,
\qquad c=\sqrt{1-s^2},
\]

where `gamma = 2 pi N t + a_y` with phases converted from degrees. Define

\[
Q=Xc+Hs,\quad R=\sqrt{1-Q^2},\quad
X'=Qc-Rs,\quad Z'=Rc+Qs,\quad P=Hc-Xs.
\]

On the strict transmitted forward branch, `R>0` and `Z'>0`. The native prior also gives `P>0` independently: `|X|<=1`, `n>=1.3`, and `|s|<=sin(18 degrees)` imply

\[
P\ge\sqrt{1.3^2-1}\cos18^\circ-\sin18^\circ>0.48.
\]

A rational justification of the final comparison is available from `sqrt(.69)>.83`, `sin18<.3091`, and `cos18>.951`: their resulting lower bound is `.48023>.48`. The trigonometric bounds follow from `sin18=(sqrt(5)-1)/4`, `sqrt(5)<2.2364`, and `1-.3091^2>.951^2`.

Let `p` be the transverse position at the flat entry plane. With thickness 3, the actual exit-face coordinate is

\[
p_e=\frac{c(Hp+3X)}P.
\]

Indeed, the internal slope is `X/H`, the exit-plane slope is `s/c`, and their line-plane intersection gives this formula. Since

\[
1-\frac{s}{c}\frac{X'}{Z'}=\frac{R}{cZ'},
\]

propagation from that face to the next nominal parallel plane, at nominal gap `a`, is exactly

\[
p_{next}=L p+K+a T,
\quad L=\frac{HR}{Z'P}>0,
\quad K=\frac{3XR}{Z'P},
\quad T=\frac{X'}{Z'}.
\tag{1}
\]

No paraxial approximation occurs here. Apply (1) in physical order with gaps `g,g,d` and initial position `p_axis+6 T_0`. At each scalar observation,

\[
F_i(q,\ell)=b_i(q)+D_i(q)d+G_i(q)g+\kappa_i(q)p_{axis(i)},
\tag{2}
\]

where

\[
\begin{aligned}
\kappa&=L_3L_2L_1>0,\\
b&=6T_0\kappa+K_3+L_3K_2+L_3L_2K_1,\\
G&=L_3L_2T_1+L_3T_2,\qquad D=T_3.
\end{aligned}
\tag{3}
\]

Thus `F(q,l)=b(q)+A(q)l`, with x rows `[D,G,kappa,0]` and y rows `[D,G,0,kappa]`. The coefficients use only `q` and the timestamp. The positive source coefficient is an exact structural fact. It has **no uniform positive lower bound asserted over the full native physical domain**: approaching critical transmission can make `R`, hence `kappa`, small.

## 2. Exact source elimination leaves a two-dimensional polygon

For a fixed admissible `q`, normalize each row by its positive `kappa_i`. Define

\[
z_i=(y_i-b_i)/\kappa_i,\quad
\alpha_i=D_i/\kappa_i,\quad\beta_i=G_i/\kappa_i,\quad
e_i=\eta/\kappa_i.
\]

The observation constraint is precisely

\[
L_i(d,g)\le p_{axis(i)}\le U_i(d,g),
\quad
L_i=z_i-\alpha_i d-\beta_i g-e_i,
\quad U_i=z_i-\alpha_i d-\beta_i g+e_i.
\tag{4}
\]

For each axis, append the native source interval as a virtual lower endpoint `L_0=-5` and upper endpoint `U_0=5`. Define the polygon

\[
\mathcal P(q)=\{(d,g)\in[50,200]\times[2,15]:
L_i(d,g)\le U_j(d,g)\text{ for every }i,j\text{ on each axis}\}.
\tag{5}
\]

**Theorem 1 (exact bounded-noise elimination).** There exists `l` in the complete native affine box with `|F(q,l)-y|<=eta` if and only if `P(q)` is nonempty. For every point of `P(q)`, all compatible source positions are given exactly by

\[
p_x\in[\max_{i\in x\cup\{0\}}L_i,\ \min_{j\in x\cup\{0\}}U_j],
\quad
p_y\in[\max_{i\in y\cup\{0\}}L_i,\ \min_{j\in y\cup\{0\}}U_j].
\tag{6}
\]

*Proof.* Each row of (4) is equivalent to its original observation strip because its divisor is positive. A finite family of closed intervals has nonempty intersection exactly when its largest lower endpoint does not exceed its smallest upper endpoint. This is equivalent to every lower endpoint being at most every upper endpoint. The native bounds are already included. Equations (5)-(6) therefore give both directions and reconstruct every possible source position. No rank condition on `A` is needed. □

For two actual observations `i,j` on the same axis, their halfplane is

\[
(\alpha_j-\alpha_i)d+(\beta_j-\beta_i)g
\le z_j-z_i+e_i+e_j.
\tag{7}
\]

There are at most `2*201^2+4 = 80,806` halfplanes, including the four native geometry bounds. Many are tautologies or redundant. This is a finite explicit elimination; an implementation can materialize the halfplanes and clip a polygon, or add violated constraints lazily. Given `(d,g)`, the strongest source-intersection violation is found in `O(m)` by taking maxima and minima, so a lazy separation oracle need not enumerate all pairs.

The native source bounds must not be dropped. Conversely, no four-sample determinant needs to be assumed nonzero: zero wedges, collisions, and loss of affine rank are still handled by interval intersections and a possibly higher-dimensional polygon.

## 3. Equivalent exact four-variable LP and sparse Farkas certificates

Let `r=y-b(q)`, and let `L,U` denote the four native affine bound vectors. Form

\[
B(q)=\begin{bmatrix}A(q)\\-A(q)\\I_4\\-I_4\end{bmatrix},
\qquad h(q)=\begin{bmatrix}r+\eta\mathbf1\\-r+\eta\mathbf1\\U\\-L\end{bmatrix}.
\tag{8}
\]

Compatibility is exactly `B l <= h`, a four-variable feasibility LP with 808 inequalities. By the alternative theorem for linear inequalities, it is infeasible exactly when

\[
\lambda\ge0,\quad B(q)^T\lambda=0,\quad\lambda^T h(q)<0.
\tag{9}
\]

**Theorem 2 (a sparse exact infeasibility witness exists).** Whenever (8) is infeasible, a witness (9) exists using at most five of its 808 inequalities.

*Proof.* Normalize a nonzero Farkas witness by `sum(lambda)=1`. Minimize `lambda^T h` over `lambda>=0`, `B^T lambda=0`, `sum(lambda)=1`. This is a nonempty compact polytope, and its minimum is negative. A minimizing extreme point exists. Its positive columns in the five-row equality matrix `[B^T;1^T]` are linearly independent, or a nonzero feasible two-sided perturbation would contradict extremality. Its support therefore has at most five entries. □

This makes a rejection certificate small even when finding it uses all 400 observations. A floating LP status alone is not such a certificate; signs, annihilation, and strict negativity must be independently enclosed or checked exactly.

In the generic rank-four five-row case, signed `4x4` cofactors give a null vector. It is a positive circuit only when its nonzero entries have the required common sign. Lower-rank circuits require smaller minors or direct exact linear algebra. Checking only generic nonzero five-row determinants would miss degenerate cases.

For an unbounded affine vector, a related necessary noisy exterior-minor test uses five selected observation rows `I`: if `w^T A_I=0`, then

\[
|w^T(y_I-b_I)|\le\eta\|w\|_1.
\tag{10}
\]

This is useful as a cheap rejection constraint but does not by itself enforce the native affine box. Equations (8)-(9), or Theorem 1, do enforce it.

## 4. A scalar exact elimination and a cancellation-preserving dual

Write the native affine box as `l=c+v`, `|v_j|<=r_j`, and put `f_c(q)=b(q)+A(q)c`. Define

\[
\delta(q)=\min_{|v|\le r}\|y-f_c(q)-A(q)v\|_\infty.
\tag{11}
\]

Compatibility at `q` is precisely `delta(q)<=eta`. Its exact dual formula is

\[
\delta(q)=\max_{\|w\|_1\le1}
\left[w^T(y-f_c(q))-\sum_{j=1}^{4}r_j|(A(q)^Tw)_j|\right].
\tag{12}
\]

*Proof.* Use `||z||_infinity=max_{||w||_1<=1} w^T z` in (11). The affine box and the dual norm ball are compact convex sets, and the bilinear expression permits the elementary linear-programming min-max interchange. Minimizing `-w^T A v` over each interval `[-r_j,r_j]` gives `-r_j|(A^T w)_j|`. □

Equivalently, `y-f_c` belongs to the zonotope

\[
A(q)\operatorname{diag}(r)[-1,1]^4+\eta[-1,1]^{400}
\]

if and only if every `w` satisfies

\[
w^T(y-f_c(q))\le\eta\|w\|_1+\sum_jr_j|(A(q)^Tw)_j|.
\tag{13}
\]

This explicitly retains both hard measurement error and every affine bound. It also removes the need to numerically prove an exact nullspace relation: **every fixed vector `w` gives a valid necessary inequality**, even when it is only an approximate LP dual or approximate left-null vector. Its nonzero defect `A^T w` is paid for by the known affine radii.

## 5. Certified exclusion of an entire fourteen-dimensional box

Let `Q` be a box of nonlinear parameters on which interval calculations certify the required strict optical branches. For any fixed exactly represented rational vector `w`, obtain rigorous enclosures of the combined expressions

\[
u(q)=w^T(y-f_c(q)),\qquad a_j(q)=w^T A_j(q),\qquad q\in Q.
\]

**Theorem 3 (uniform nonlinear-box exclusion).** If

\[
\underline u(Q)-\sum_jr_j\max\{|\underline a_j(Q)|,|\overline a_j(Q)|\}
-\eta\|w\|_1>0,
\tag{14}
\]

then no `q` in `Q` and no affine hardware in the original native box explain the observations.

*Proof.* A compatible pair would satisfy (13). The left-hand side minus the right-hand side of (13) is at least the strictly positive quantity (14) for every `q` in `Q`, a contradiction. □

To preserve useful cancellation, enclose `w^T f_c` and `w^T A_j` as combined expressions where possible. Bounding them first by sums of absolute sensitivities can be much weaker. A numerical LP, spectral method, optimizer, or learned proposal may choose `w`; none supplies acceptance evidence. Only the final independently checked strict inequality matters. `w` can be rounded to rationals before checking because exact nullspace membership is not required.

An alternative outer-LP construction is also sound. Suppose

\[
f_c(Q)\subseteq f_0\pm e,\qquad A(Q)\subseteq M\pm E,\qquad E,e\ge0.
\]

Every compatible `v` must belong to the polytope

\[
|v|\le r,\qquad
|Mv-(y-f_0)|\le\eta\mathbf1+e+Er.
\tag{15}
\]

Indeed, subtract and add `f_0+Mv` and bound `(A-M)v` by `E|v|<=Er`. Thus certified infeasibility of (15) rejects the whole `Q`. Certified coordinate extrema of (15) can contract the affine box for that `Q`; floating LP extrema alone are not certified outer bounds. Replacing `r` by previously certified contracted radii can sharpen subsequent passes.

The direct dual test (14) and the outer-LP test (15) do not divide by `kappa`. They therefore remain useful when source normalization is ill-conditioned, provided the optical interval evaluation itself succeeds. A failed guard enclosure is inconclusive; it is not a physical rejection.

## 6. Finite termination that is actually justified

On a compact set of nonlinear hardware with uniformly positive certified optical margins, `A` and `f_c` are continuous, hence so is `delta`. Suppose every point in that set obeys `delta(q)>=eta+gamma` for a fixed `gamma>0`. If the interval evaluations in (14) converge as box widths go to zero, a fair exhaustive subdivision eventually produces a finite cover of rejected boxes.

To see this, choose a dual maximizer at each point using (12). Its strict positive separation survives in a neighborhood by continuity. Compactness gives a finite subcover; a sufficiently fine subdivision lies within such neighborhoods, and convergent interval bounds eventually verify the strict margins.

This statement **does not** prove tractable running time, resolve boxes touching `delta=eta`, supply a positive global separation gap, cover an open physical domain all the way to critical boundaries, or establish that a retained box contains a solution. Those require separate arguments. See the companion global-search and finite-bound notes for their contracts. The exact elimination reduces the nonlinear search from eighteen to fourteen variables; it does not complete that search by itself.

## 7. Implementation and audit status

`elimination_benchmark.py` is a standalone implementation of (3)-(7) and a comparison with the equivalent four-variable LP. It reads only the saved observation-derived adverse candidate, its successful gain-continuation result, two observation-derived compatible profiles, and their actual observation record. It retains all native affine bounds and uses centering/scaling only as changes of numerical coordinates. It writes no original research files and imports no Dropbox project code.

The benchmark reports feasibility agreement, conditional `(d,g)` extrema, source reconstruction, residuals, and timings. Its floating polygon and LP answers are numerical diagnostics, not outward or exact-rational certificates. A production certified implementation must check the final inequalities or dual with exact/interval arithmetic.

The parent agent executed the benchmark in the current authorized research context; the saved report is `elimination_benchmark.json`. Its four cases are the earlier incompatible observation-derived candidate, the successful gain-continuation candidate, and two observation-derived profiles of the same actual record. The following results were inspected from that report:

| Conditional optical input | Polygon feasible | Four-variable LP feasible | Largest `(d,g)` extremum difference | Polygon time | LP time |
|---|---:|---:|---:|---:|---:|
| Earlier blind candidate | no | no | — | 0.020 s | 0.012 s |
| Gain-continuation candidate | yes | yes | 0 | 1.091 s | 0.035 s |
| Profile 0 | yes | yes | 0 | 0.652 s | 0.013 s |
| Profile 1 | yes | yes | `4.44e-16` | 0.597 s | 0.018 s |

The materialized polygon uses 80,404 halfplanes after removing the 402 zero-normal tautologies. **This naive polygon implementation is slower than the original four-variable LP**, even though it makes the elimination explicit. The benchmark supports the algebraic equivalence and a useful alternative representation, not a speed advantage. A lazy two-dimensional separation method remains a possible implementation improvement, not a demonstrated result.

For the gain-continuation optical input, the conditional geometry ranges are `d in [190.00109959606723,190.00110039923203]` and `g in [2.9994763808396505,2.9994770166361002]` in the floating benchmark. These are conditional on that particular fourteen-vector. They are not global inverse bounds: other compatible optical parameters in the same observation record have different geometry ranges.

Some floating LP extrema reconstruct residuals as large as `1.000000082740371e-8` against an allowance `1e-8`, an excess of about `8.27e-16`. This is precisely why raw numerical LP answers are not declared accepted hard-error certificates. The polygon, LP, and coefficient evaluations in this small benchmark do not implement the outward fourteen-box contractor of Theorem 3.

No optical fit was run for this benchmark and no hidden truth vector was read. Input and script hashes are retained in its JSON report. The exact identities and proofs above are mathematical derivations; the numerical comparison is separate supporting evidence, not their formal verification.
