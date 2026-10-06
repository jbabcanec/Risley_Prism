# Exact algebraic optics do not give a finite linear shift lift

## Verdict and scope

This bounded attack does **not** produce the missing elegant full18 inverse. It does prove a precise obstruction to an appealing proposed shortcut: the fixed quadratic-tower size of one vector-Snell ray trace does not become a fixed-size Prony, matrix-pencil, or finite-dimensional linear shift state.

An entirely physical, uniformly noncritical one-rotating-prism subfamily of the original model has the following properties.

1. Each screen sample is only quadratic over the instantaneous rotor function field, but its first K shifted samples generate a function-field extension of degree exactly 2^K.
2. On the actual original 200 consecutive timestamps, its scalar x-coordinate has a nonsingular 100-by-100 Hankel matrix for all but finitely many allowed initial phases. Thus its exact finite record generically requires linear-recurrence order at least 100, despite having just one active rotor.
3. Nevertheless, that same trace obeys an exact implicit algebraic-exponential relation with at most 25 coefficient slots. The field-degree and Hankel results therefore do not rule out nonlinear algebraic recovery, and do not constitute a computational complexity lower bound.

The mechanism is moving branch points, not a small-wedge approximation or a critical-boundary pathology. The full18 problem still needs an independently proved small nonlinear elimination/reconstruction. None is supplied here.

## 1. An admissible exact subfamily

Use the original native bounds and times k/20. Set the first two wedges to zero and retain their actual glass slabs. Take

- all three indices n_j = 3/2;
- beam slope t = (1/3, 0) and source offset b = (0, 0);
- gap g = 2 and workpiece distance d = 100;
- last wedge slope amplitude a = 1/10.

These are proof parameters, not quantities furnished to an inverse algorithm. All eighteen coordinates remain unknown in the original problem. The zero wedges are legal prior points; a later continuity argument also gives fully active nearby examples.

Write the incident external direction as (q, 0, z_0), where

q = 1/sqrt(10),    z_0 = 3/sqrt(10),    H = sqrt(43/20).

A zero-wedge slab restores the incident external direction after its flat exit and contributes internal transverse displacement 3q/H. The entrance x-coordinate of the third prism is consequently

p = (6 + 2g)/3 + 6q/H = 10/3 + 6 sqrt(2/43).

Its entrance y-coordinate is zero. Thus neither zero-wedge slab has been deleted or replaced by a zero-thickness element.

Let w denote the final rotor phase exponential and put

u = a(w + w^(-1))/2,
u_y = a(w - w^(-1))/(2i),
D = 1 + a^2,
P = H - qu,
A = Hp + 3q,
B_0 = dP - uA,
C = HD - P = Ha^2 + qu,
J = Dq + uP,
R(w) = P^2 - D(5/4),
E(w) = sqrt(R(w)).

On |w| = 1, the physical square root is positive. The last exit position is (A/P, 0), its remaining axial flight is B = B_0/P = d - uA/P, and the exact final x-coordinate is

f(w) = A/P + (B_0/P) (J - uE)/(C + E).                 (1)

This follows directly from h = (P-E)/D, outgoing direction r_x = q+uh and Z = H-h, and the exact paired position transfer. In particular, it includes the tilted-plane intersection.

### Uniform physical margins

The elementary bounds H > 7/5, q < 1/3, p < 5 and |u| <= 1/10 imply

P > 41/30,
R >= (H-aq)^2 - D(5/4) > 2179/3600 > 0.

Also A/P < 7, so B > 99.3. The internal traversal numerator 3+up exceeds 2.5. The outgoing axial numerator has the explicit margin C+E > 43/60: E > sqrt(2179)/60 > 3/4 while C > -1/30. Thus every rotor orientation is strictly transmitted, forward axial, and sequentially traversed. The first two slabs have internal axial traversal 3 and positive external gap 2. There is a whole-circle physical margin, not merely admissibility at a selected phase.

The beam angle atan(1/3) is below 25 degrees, and the last wedge angle atan(1/10) is below 18 degrees. All remaining chosen native values satisfy their original bounds.

Fix the sampled rotor node

z = (99 + 20i)/101 = exp(2i atan(1/10)).

Its signed speed is N = (20/pi) atan(1/10), within [-3.5,3.5] Hz. This z has infinite multiplicative order: if z were a root of unity, z+z^(-1) = 198/101 would be a rational algebraic integer, hence an integer, a contradiction. For every allowed initial phase phi in [-18,18] degrees, the physical record is exactly

x_k = f(z^k w_0),    w_0 = exp(i pi phi/180),    k = 0,...,199.  (2)

## 2. The observed coordinate exposes the ray radical

The dependence in (1) is genuinely Möbius in E. Its relevant derivative is

d[(J-uE)/(C+E)]/dE = -D(q+Hu)/(C+E)^2.

Neither B_0 nor q+Hu is the zero rational function. On the physical circle they are positive: B_0 = PB > 0, and q+Hu >= q-aH > 0 using q > 3/10 and H < 3/2.

More explicitly, rearranging (1) gives

E = [B_0 J - C(Pf-A)] / [(Pf-A) + uB_0].              (3)

The denominator is nonzero as a rational function; on the physical circle it equals B_0 D(q+Hu)/(C+E) > 0. Therefore, with F = C(w),

F(f) = F(E).

There is no hidden cancellation that makes the measured x-coordinate rational in the rotor. The same statement holds after every shift w -> z^k w.

## 3. Shifted roots have independent square classes

Put b_* = aq/2 and c_* = D(5/4). The radicand factors as

R(w) = [H-b_*(w+w^(-1))-sqrt(c_*)]
       [H-b_*(w+w^(-1))+sqrt(c_*)].                  (4)

The physical margin implies H-aq > sqrt(c_*). Both equations

w+w^(-1) = (H +/- sqrt(c_*))/b_*

therefore have two distinct positive reciprocal real roots, and their right-hand sides are distinct and greater than two. Altogether R has four distinct nonzero positive real simple zeros rho_1,...,rho_4. Its poles at zero and infinity have even order two.

The zeros of R(z^k w) are z^(-k)rho_j. These sets are disjoint for distinct integers k. Indeed, an equality z^(-k)rho_j = z^(-l)rho_h forces rho_j = rho_h by absolute values, then z^(l-k) = 1, so k=l.

For any nonempty finite set I of shifts, the product

product_(k in I) R(z^k w)

has odd valuation at a simple zero belonging to any one of its factors. A square in C(w) has even valuation everywhere, so this product is not a square. Thus the shifted radicands are independent in F*/F*^2. The elementary multiquadratic extension theorem gives

[F(E(w), E(zw), ..., E(z^(K-1)w)) : F] = 2^K.         (5)

One can also see the theorem by successively using the independent sign-change characters of the square roots. Formula (3) consequently gives

[F(f(w), f(zw), ..., f(z^(K-1)w)) : F] = 2^K.         (6)

### What the degree statement means

Equation (6) is a statement about functions of the initial rotor variable w. It is **not** the claim that specializing to an arbitrary numeric phase gives numeric sample values of number-field degree 2^K. Specialization can change degree, and over the numeric constant field C every number is already present. The actual finite-record consequence is proved separately below.

The proof does identify why a proposed exact shift module grows. After rationalizing (1), each shifted f_k is a nonzero rational coefficient times E_k plus a rational term. The sign-change characters imply that

1, f_0, ..., f_(K-1)

are linearly independent over F. Therefore an F-vector-space lift containing those observables and 1 has dimension at least K+1. If it is also a unital multiplicatively closed subalgebra of the compositum, its dimension is at least 2^K: products of the independent roots give the basis characters. The latter is the relevant obstruction to retaining all multiplicative ray-state relations in one fixed algebraic quotient.

A finite-dimensional shift-invariant lift of algebraic functions over F cannot contain f and all its shifts. A finite quadratic tower for one sample is not a shift-invariant tower; the next sample introduces new branch divisors. The original three-prism recurrence is a quadratic tower, not automatically a globally multiquadratic Galois extension. The one-prism subfamily already suffices for the obstruction, without making that stronger assertion about the full tower.

## 4. A numerical-record consequence: maximal Hankel rank

Define the symbolic Hankel matrix

H_n(w) = [f(z^(i+j)w)]_(0 <= i,j < n).

Its determinant is not the zero algebraic function, for every n >= 1. For n=1, f is nonzero because it is not rational over F. For the induction, the newest sample f_(2n-2) occurs only in the bottom-right entry. Expanding in that entry gives

det H_n = f_(2n-2) det H_(n-1) + V,

where V belongs to F(f_0,...,f_(2n-3)). The induction coefficient is nonzero, and (6) says the new f_(2n-2) does not belong to that earlier field. The displayed expression therefore cannot vanish.

The positive physical root in (1) is holomorphic on an annulus containing the unit circle. To justify this rather than assume it, R is nonzero and positive real on that circle, so its winding number there is zero and it has a holomorphic square root on a sufficiently thin annulus. All the physical rational denominators stay nonzero on a possibly thinner annulus. Every finite set of rotated copies has the same property.

Consequently det H_100 is a holomorphic nonzero function near the closed initial-phase arc [-18,18] degrees. Its zeros on that compact arc are isolated and finite. At every other allowed phase, the **actual numerical** matrix

[x_(i+j)]_(0 <= i,j < 100)

is nonsingular. It uses samples 0 through 198 of the specified 200. Any exact constant-coefficient recurrence valid across that record must then have order at least 100. In particular, a modest fixed list of tones, a small ordinary matrix pencil, or a low-rank Hankel surrogate is not an exact representation of this benign one-rotor trace.

This is an exact rank statement, not a numerical-conditioning guarantee. A nonsingular Hankel matrix can have extremely small singular values, especially in a small-wedge optical model.

For the infinite exact sequence there is no finite-order constant-coefficient recurrence at any initial phase. If such a recurrence existed, its corresponding finite linear combination of shifted f would vanish along the dense irrational-rotation orbit of w_0. Continuity on the circle would make that an identity there, and analyticity would make it a functional identity, contradicting the shift independence above. This uses whole-circle physical admissibility, which was verified in Section 1.

### Fully active nearby systems

Choose an interior allowed initial phase where det H_100 is nonzero. The first two zero-wedge rotor speeds may already be chosen distinct, nonzero, and different from the third speed, since those inactive slopes do not affect the proof record. All 200 physical traces and their determinant depend continuously on the native coordinates near this strictly physical system. Giving the first two wedges sufficiently small nonzero values preserves the determinant and every strict physical inequality. One can also move g slightly above 2 to enter the interior of its prior. Hence maximal finite-record Hankel rank occurs in an open set inside the original prior with all three wedges and all three distinct speeds nonzero. The obstruction is not confined to an unobservable zero-wedge stratum. This continuity argument concerns the fixed 200-sample rank statement, not infinite-sequence nonrecurrence throughout that neighborhood.

## 5. Nonlinear algebraic-exponential structure survives

The negative linear-lift result must not be mistaken for a negative nonlinear-inversion theorem. The same one-prism record has a quite small implicit relation.

For a formal observed x-coordinate X, define

L(X,u) = PX-A+uB_0,
M(X,u) = B_0 J-C(PX-A).

Equation (3) and E^2=R give

M(X,u)^2 - R(u)L(X,u)^2 = 0.                         (7)

This polynomial has degree at most two in X and at most four in u. Here is an explicit degree check, not an assumed sparse cancellation. Write B_0=b_0-b_1u, where b_0=dH and b_1=A+dq. Then

L = -b_1u^2 + (b_0-qX)u + HX-A,
M = b_1q u^3 + [-b_0q-b_1H+q^2X]u^2 + lower terms,
R = q^2u^2-2Hqu+constant.

The coefficients of u^6 and u^5 in M^2 and RL^2 agree, giving degree at most four. The coefficient of X^2 is P^2(C^2-R), and

C^2-R = D[q^2+H^2a^2-1+2Hqu],

so its u-degree is at most three. Substituting u=a(w+w^(-1))/2 gives the support bound

sum_(ell=-4)^4 c_(0,ell) w^ell
 + X sum_(ell=-4)^4 c_(1,ell) w^ell
 + X^2 sum_(ell=-3)^3 c_(2,ell) w^ell = 0.             (8)

There are at most 9+9+7=25 coefficient slots. The coefficients depend only on the fixed optical and geometry parameters. Unknown initial phase merely multiplies c_(b,ell) by w_0^ell. Therefore every sampled record (2) satisfies

sum_(b,ell) d_(b,ell) x_k^b z^(ell k) = 0,            (9)

with a nonzero coefficient vector. This is an exact algebraic-exponential relation, with no Taylor truncation. Its nontriviality also follows from the genuine quadratic field extension established in Section 2.

Thus the 200-by-25 feature matrix with entries x_k^b z^(ell k) has a nonzero right kernel at the true speed. This gives a speed-only polynomial rank-defect condition for this special subfamily after clearing Laurent powers. It is only a necessary candidate condition: collapsed trial nodes, unphysical polynomial roots, and coefficient vectors not arising from the optical family can give spurious candidates. No finite-candidate, unique-reconstruction, or practical full18 result follows without additional work.

This explicit relation explains the limit of the obstruction. Exponential growth of a joint algebraic extension does not prevent a compact arithmetic circuit or a useful nonlinear implicit relation. It prevents replacing the exact optics by the particular fixed finite **linear** shift representation.

## 6. Consequence for the full18 proposal

The existing five-root per-sample compiler provides finite implicit polynomial relations for the full exact optical map. Merely treating all coefficients of a large Laurent-polynomial dictionary as independent unknowns does not yield a frequency extractor. If the dictionary has more than 200 columns, its 200-row feature matrix has a right kernel for every trial speed triple. A rank-defect test is then vacuous unless it additionally enforces a smaller support or the nonlinear optical relations between coefficients. This is a dimension count for that construction, not a proof that every possible sparse relation is large.

No bound showing a useful, identifiable, full three-rotor implicit support within the available 200 observations is proved here. Nor is a low-dimensional reconstruction of its optical coefficient constraints. Those are exactly the missing positive ingredients; one cannot substitute the single-sample algebraic degree or the existence of resultants for them.

The answer to this bounded attack is therefore:

- **No** fixed-size exact Prony/matrix-pencil or algebraic ray-root shift module follows from the finite instantaneous quadratic lift. A rigorous, physically interior counterexample explains the growth and its actual 200-sample effect.
- **Open here** whether a specially structured nonlinear algebraic-exponential elimination solves the full18 inverse efficiently. The 25-slot one-prism relation shows why the linear obstruction must not be extended to that question.

This advances the diagnosis of the missing elegant inverse, not its construction. It does not establish a lower bound for all exact algorithms, does not challenge the existing complete algebraic fallback, and makes no published-novelty claim. No simulation, reconstruction sweep, or numerical campaign was used.

## Audit status

An independent mathematical audit passed the physical subfamily and margins, branch-divisor independence, function-field versus actual finite-record rank distinction, finite exceptional phase set, nearby fully active finite-record extension, and the exact 25-slot nonlinear relation. It corrected and verified the native-degree conversion in (2). The separate report is `exact_shift_module_independent_audit.md`. This is a mathematical audit, not proof-assistant formalization, a numerically evaluated determinant, or a literature-novelty assessment.
