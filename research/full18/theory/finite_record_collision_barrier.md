# Exact off-axis finite-record collision barrier

> Integration notation: the observation allowance \(\eta\) in this supplied memo is the report's symbolic measurement-error bound \(\epsilon\). The symbol \(A\) is a native screen-length amplitude normalization; it is neither observation error nor a wedge angle. The source reference “vector_inverse.md” resolves here to [physical_vector_inverse.md](physical_vector_inverse.md), specifically its exact physical model, offset-excited hardware inverse, and scaled finite-record rank theorem. This addendum was supplied as independently audited; saving it does not constitute a new mathematical or numerical verification.

## Result and limits

This addendum constructs two different, strictly physical three-prism systems inside the original full18 prior whose **exact** paired records agree through fourth order in a slow-time variable. All six wedges are positive, the source offset is nonzero, every rotor speed is nonzero, and each system has distinct nonaliased fundamental and degree-two composite modes. For sufficiently small positive wedge scale, both original 400-by-18 sampled Jacobians have full column rank.

Their native rotor speeds differ by more than 0.003 Hz, yet their exact 200-sample records can differ by at most a constant times A(h T)^5, where A is a native screen-length amplitude scale, h is the interlacing speed scale, and T is the observation-window half-width. The fifth power is not an uncharged first-order optical approximation: the wedges and phases are corrected by an exact implicit-function construction so that the complete nonlinear optical jets cancel. No optical Taylor-tail floor is added.

The physical frequencies, phase margins, divided-difference constants, and nonresonance checks below are explicit. A positive admissible amplitude cutoff and the nonlinear coefficient are defined by finite, checkable exact optimization/contraction certificates. They have not been numerically evaluated. Therefore this is a proved small-amplitude family and a constructive conditional quantitative bound, not a claimed numerical noise limit for an arbitrary supplied record or a practically sized wedge.

The exponent five is familiar from three-source superresolution. The new point here is its realization by the exact finite-thickness vector-Snell map, with nonzero source excitation and ultimately full18 rank, while respecting the original native bounds and the actual 200 timestamps.

## 1. A deterministic two-system lemma

Write F(theta) in R^400 for the original paired record and N(theta) in R^3 for the native signed speeds. If two strictly admissible systems theta+ and theta- satisfy

~~~text
||F(theta+) - F(theta-)||_infinity <= 2 eta,
~~~

then their midpoint record is compatible with both under componentwise observation allowance eta. Every estimator of the three speeds has worst-case native infinity-norm error at least

~~~text
(1/2) ||N(theta+) - N(theta-)||_infinity.                 (1)
~~~

This is the triangle inequality at the common midpoint observation. No stochastic noise or square-root-of-200 benefit is assumed. Consequently a speed-coordinate lower bound is already a lower bound against recovering all eighteen native coordinates to the same target. The estimator is not told the nuisance parameters used to construct the two systems; selecting an admissible subclass for a lower bound does not turn them into known calibration inputs in the original problem.

For a particular observed record y, the obstruction applies only if y lies in both exact observation boxes. Existence of a bad midpoint record does not prove that every record is ambiguous.

## 2. Fixed allowed geometry and complex wedge coordinates

Use the exact vector model of [physical_vector_inverse.md](physical_vector_inverse.md). Fix, solely to construct a lower-bound subclass,

~~~text
n1 = n2 = n3 = 3/2,   g = 5,   d = 100,
p_x = 1, p_y = 0,    beta_x = beta_y = 0.
~~~

These values are strictly inside the original bounds. In particular the source is genuinely off axis. The leading circular gains at this hardware are

~~~text
K1 = 57,   K2 = 107/2,   K3 = 50.
~~~

Represent a wedge by its complex sine vector z = sin(a) exp(i gamma). The actual plane slope is z/sqrt(1-|z|^2), so the exact optical map is real analytic in Re z and Im z around zero, on a strict physical branch. Let Z(z1,z2,z3) be the complex screen coordinate for the fixed geometry above. Its expansion starts

~~~text
Z(z1,z2,z3) = 1 + K1 z1 + K2 z2 + K3 z3 + O(||z||^2). (2)
~~~

The source-dependent quadratic terms are retained in Z throughout this proof. Equation (2) is used only to identify an invertible derivative for the exact correction equations.

There is a simple useful uniform guard. If |z_j| <= 111/50000, then |u_j| < 1/400. For every choice of their three arguments the transmitted branch is strict: recursively the incoming external transverse direction has norm below 1/50, the internal H and normal P exceed 1.49, the outgoing normal and axial factors exceed 0.99, and the output transverse norm increases by at most (3/2)/400 at each prism. The entrance positions remain of norm below 2. In particular internal axial traversal lies between 2.97 and 3.03; the next entrance gaps exceed 4.97, and the last screen gap exceeds 99.97.

Here is an elementary way to check those deliberately loose bounds. Assuming incoming |X| <= 1/50 and |u| <= 1/400,

~~~text
H = sqrt(9/4-|X|^2),  P = H-u dot X > 1.49,
R_normal^2 = 1-9/4 + P^2/(1+|u|^2) > 0.99^2.
~~~

The positive transmitted formula gives Z_axial > 0, H > Z_axial, and |B| <= |X|+(3/2)|u|. Starting from X=0 gives |B| <= 9/800 < 1/50. Hence Z_axial=sqrt(1-|B|^2) > 0.99. With entrance-position norm <= 2, the exact internal traversal H(3+u dot p)/P is in (2.97,3.03). An intermediate transverse displacement is less than 3.03(1/50)/1.49+5.03(1/50)/0.99 < 0.15, so the first entrance norm 1 stays below 2 through both subsequent entrances. These bounds prove the induction and the original strict surface order. No critical branch is approached in this construction.

## 3. Six explicit interlaced nodes with no finite-record alias

For l=0,...,5 set

~~~text
r_l = l + 1/2 + l^2/1000.
~~~

Thus

~~~text
r = (0.5, 1.501, 2.504, 3.509, 4.516, 5.525).
~~~

System + uses l=(0,2,4), assigned to physical prisms (1,2,3), and system - uses l=(1,3,5), in that order. Their speeds are N_l=h r_l, where h > 0 will be specified.

For either triple, any nonzero integer u with ||u||_1 <= 4 has u dot r != 0. A short exact proof avoids numerical spectral tests:

- For the even triple, multiplying a relation by 250 gives

  ~~~text
  125(u1+5u2+9u3)+(u2+4u3)=0.
  ~~~

  The second term has magnitude <= 16 < 125, so both integer terms vanish. The resulting integer relations are multiples of (11,-4,1), whose l1 norm is 16.

- For the odd triple, multiplying by 1000 gives

  ~~~text
  500(3u1+7u2+11u3)+(u1+9u2+25u3)=0.
  ~~~

  The second term has magnitude <= 100 < 500. Both vanish; the primitive relation is (19,-16,5), of l1 norm 40.

For 0 < h <= 0.003, every difference of degree-two composite frequencies is smaller than 4h max r_l <= 0.0663 Hz in magnitude. It cannot be a nonzero multiple of the 20 Hz sampling rate. Hence all 25 nodes

~~~text
exp(2 pi i (m dot N)/20),   ||m||_1 <= 2,
~~~

are distinct for each system. Appending the six confluent columns k z_m^k at the positive/negative fundamental nodes gives exactly the nonsingular 31-column finite-record design from [physical_vector_inverse.md](physical_vector_inverse.md). The first 31 of the original 200 timestamps suffice for its algebraic rank; all 200 are retained for the actual residual bound. No frequency is assumed observed or supplied to the inverse estimator.

## 4. Divided-difference weights and the exact correction

Define positive rational weights

~~~text
D = product_(j=1)^5 (r_j-r_0),
w_l = D / |product_(m!=l)(r_l-r_m)|.
~~~

Their exact values are

~~~text
w0 = 1,
w1 = 2505/503,
w2 = 5025020/506521,
w3 = 119405/12084,
w4 = 2515027515/511079689,
w5 = 6012113393/6132956268.
~~~

Every w_l lies strictly between 0.9 and 11. Also

~~~text
B0 := D/5! = product_(j=1)^5 (1+j/1000)
   = 25377130631853/25000000000000 < 1.02.             (3)
~~~

The usual divided-difference identities give

~~~text
sum_l (-1)^l w_l r_l^m = 0,  m=0,...,4,
sum_l (-1)^l w_l r_l^5 = -D.                           (4)
~~~

Let t_c=199/40, the midpoint of the actual sample window [0,199/20]. Put r_c=(r_0+r_5)/2 and fix

~~~text
chi = 2 pi h r_c t_c,   s = 2 pi h (t-t_c).
~~~

Introduce complex amplitudes c_l close to w_l, with c_0=1 held fixed. At time t define the actual wedge sine vectors in the two systems by

~~~text
z_j^+(t) = (A/K_j) c_(2j-2) exp(i chi) exp(i r_(2j-2) s),
z_j^-(t) = (A/K_j) c_(2j-1) exp(i chi) exp(i r_(2j-1) s),
~~~

for j=1,2,3. In original native parameters these are

~~~text
sin(a_j^+/-) = A |c_l| / K_j,
phi_j^+/- = arg(c_l) + 2 pi h (r_c-r_l)t_c,            (5)
~~~

with the phase in (5) converted to degrees. Their speeds are exactly h r_l.

Let G_+(A,c;s), G_-(A,c;s) be the exact complex outputs obtained by putting these z vectors into Z. The quotient

~~~text
H(A,c;s) = [G_+(A,c;s)-G_-(A,c;s)]/A                  (6)
~~~

has a removable singularity at A=0 because the two zero-wedge outputs both equal 1. It extends real analytically. At A=0,

~~~text
H(0,c;s) = exp(i chi) sum_l (-1)^l c_l exp(i r_l s).
~~~

Impose the five complex exact equations

~~~text
J_m(A,c) := i^(-m) exp(-i chi) partial_s^m H(A,c;0)=0,
m=0,...,4.                                            (7)
~~~

These are ten real equations in Re c1,...,Re c5,Im c1,...,Im c5. At A=0 their solution is c=w by (4). Their derivative is the realification of the complex matrix

~~~text
M_(m,l) = (-1)^l r_l^m,  m=0,...,4, l=1,...,5.
~~~

M is a five-node Vandermonde times nonzero column signs and is invertible. In particular this is a square invertible derivative, not a dimension-count argument. The real analytic implicit-function theorem gives unique nearby exact amplitudes

~~~text
c_l(A) = w_l + O(A),   c0(A)=1,                        (8)
~~~

for which every equation (7) holds. The complex corrections in (8) change both the physical wedges and phases. Holding phases fixed would leave only five real degrees of freedom and would not prove this off-axis cancellation.

The corrected output difference therefore has **exactly zero** slow-time derivatives through order four. No unknown or omitted optical remainder is being set to zero approximately.

## 5. A finite, checkable implicit-function certificate

The preceding existence can be made quantitative without a recovery campaign. Fix h, let v be the ten real correction coordinates, v0 the values from w, rho=1/20, and

~~~text
V = {v: ||v-v0||_infinity <= rho},   A_pre=1/100.
~~~

For v in V, Re c_l > 17/20 and |c_l| < 111/10. Hence A <= A_pre implies the uniform physical guard of Section 2. Let B be the realification of M, and use the following finite constants, computed from the exact extended equations (7):

~~~text
C0 >= sup_(0 <= A <= A_pre) ||B^-1 partial_A J(A,v0)||_infinity,
C1 >= sup_(0 <= A <= A_pre,v in V)
                   ||B^-1 partial_A D_v J(A,v)||_(infinity<-infinity).
~~~

Set

~~~text
A0 = min(A_pre, 1/(2 C1), rho/(2 C0)),                 (9)
~~~

where a zero denominator removes that restriction. The constants are finite because the entire compact domain has strict optical/traversal margins. For 0 <= A <= A0,

~~~text
||I-B^-1 D_v J(A,v)|| <= 1/2,
||B^-1 J(A,v0)|| <= rho/2.
~~~

Thus v -> v-B^-1 J(A,v) maps V into itself and contracts by at most 1/2. This verifies existence and uniqueness of the correction over the entire stated amplitude interval, including its endpoints. Its derivative is exactly

~~~text
v'(A) = -(D_v J)^-1 partial_A J.                     (10)
~~~

For rational h the s=0 trigonometric constants exp(i chi) are algebraic. The derivatives in (7) are algebraic functions of A and v on their selected real branches; the removable quotient at A=0 is supplied by its exact limit. Consequently finite outward algebraic upper bounds C0,C1 can be certified by real-algebraic optimization or interval evaluation of the differentiated six-root graph. Formulas (9)-(10) specify precisely what must be certified. Merely quoting the implicit-function theorem does not supply a useful numerical A0.

At h=0.003 the uncorrected phase term in (5) has magnitude at most

~~~text
1079973/80000 degrees = 13.4996625 degrees.
~~~

Since |Im c_l| <= 1/20 and Re c_l > 17/20, |arg c_l| < atan(1/17) < 4 degrees. Every corrected phase is therefore strictly inside (-18,18) degrees. Positive corrected wedge amplitudes, the source box, speeds and all hardware bounds are simultaneously preserved.

## 6. Exact finite-record error bound with no optical-tail floor

Write mathcal H(A,s)=H(A,c(A);s). Its first five s-jets vanish for every A. For real nodes, the integral formula for divided differences, applied to exp(i r s), gives

~~~text
|mathcal H(0,s)| <= B0 |s|^5.                         (11)
~~~

One proof is to write the fifth divided difference as the integral of the fifth derivative over the five-simplex, of volume 1/5!. The fifth derivative has modulus |s|^5. The sign and coefficient of the divided difference are -D by (4), and exp(i chi) has unit modulus. This proof works for complex exponentials without using an invalid complex-valued mean-value theorem.

Let s0=2 pi h t_c and certify

~~~text
C >= (1/5!) sup_(0 <= A <= A0, |s| <= s0)
                    |partial_A partial_s^5 mathcal H(A,s)|.       (12)
~~~

The derivative in (12) includes (10); it does not hold corrected nuisance amplitudes fixed. C is finite. Subtract mathcal H(0,s) from mathcal H(A,s), use the fundamental theorem of calculus in A, then the fifth-order integral remainder in s. Since all lower s-jets of both terms vanish, this yields

~~~text
|mathcal H(A,s)-mathcal H(0,s)| <= A C |s|^5.
~~~

Hence the complete nonlinear physical record satisfies

~~~text
||F(theta+(A))-F(theta-(A))||_infinity
  <= A (B0+A C) (2 pi h t_c)^5.                       (13)
~~~

The real componentwise norm is at most the complex modulus used in deriving (13). All 200 sample pairs are bounded, and the same bound actually holds at every time in the full sampled interval. The A^2 contribution in (13) is multiplied by the same fifth power of the observation window. It is not an additive O(A^2) optical-floor term.

The constant C is checkable in the same exact framework. The common denominator of all r_l is 1000; substitution v_s=tan(s/2000) makes exp(i r_l s) rational in v_s. For rational h, the endpoint tan(s0/2000) is algebraic. Equation (10), the exact correction graph, and the branch-preserving six-root formulas therefore give a compact semialgebraic optimization predicate for (12). This is an existence-of-certificate result, not a claim that high-order algebraic optimization is inexpensive.

Define

~~~text
A1 = min(A0, (2-B0)/C),                               (14)
~~~

omitting the second restriction when C=0. Since B0 < 1.02, A1 > 0, and for 0 < A <= A1,

~~~text
record separation <= 2 A (2 pi h t_c)^5.              (15)
~~~

## 7. Both systems can have full18 rank

This statement uses the already proved scaled full-rank argument in [physical_vector_inverse.md](physical_vector_inverse.md), not a bare assertion of genericity.

That argument requires: nonzero offset b; zero nominal beam tilt; nonzero limiting complex fundamental amplitudes; interior hardware; and 25 distinct degree-two sampled nodes together with their six fundamental confluent columns. It does not require the particular base-five frequencies used there as one explicit witness. The proof uses those frequencies only to establish the stated finite Vandermonde condition.

All hypotheses hold here for every fixed h in (0,0.003]. Source b=1 is nonzero. The hardware is interior. The corrected wedges and phases in (8) have nonzero limiting amplitudes w_l exp(i phi_l)/K_j. Section 3 proves the exact finite-clock node and confluent conditions independently for both triples. The same normalized quadratic ratios, negative-fundamental beam channel, and hardware inverse then give rank eighteen in the limiting scaled Jacobian.

The O(A) changes in the limiting amplitude chart produced by (8) do not alter that limit. Thus there is A_rank(h) > 0 such that BOTH exact full18 Jacobians have rank eighteen whenever 0 < A <= A_rank(h). This conclusion does not freeze the fourteen optical or four geometry unknowns when differentiating: it concerns their original joint sampled Jacobian.

A numerical value for A_rank is not inferred from distinctness alone. A checkable value can be obtained from the exact selected scaled minor along each corrected algebraic amplitude branch, by bounding its nonzero limit away from zero and its finite-A remainder, or by exact sign isolation. The limit proof guarantees a positive interval, but does not say it is practically large. Set

~~~text
A_star(h)=min(A1,A_rank(h)).                           (16)
~~~

No connected-component/generic-density argument is needed, and there is no centered first-prism gauge in these systems.

## 8. Explicit 0.001-native-speed obstruction

Choose h=3/1000 Hz. The two triples, in physical prism order, are

~~~text
N+ = (0.001500, 0.007512, 0.013548) Hz,
N- = (0.004503, 0.010527, 0.016575) Hz.
~~~

Their differences are (0.003003,0.003015,0.003027) Hz, so (1) gives worst-case speed error at least 0.0015135 Hz whenever their observation boxes overlap. Every speed is strictly positive and all three speeds in either system are distinct.

On the actual 200-sample clock,

~~~text
2 pi h t_c = 597 pi/20000,
(597 pi/20000)^5 < 73/10000000.                       (17)
~~~

For example (17) follows by substituting the rational upper bound pi < 22/7; no numerical optical experiment is involved. For every

~~~text
0 < A <= A_star(3/1000),
eta >= (73/10000000) A,                              (18)
~~~

equations (15)-(17) show that the exact midpoint record is compatible with both full-rank, off-axis physical systems. No deterministic estimator can guarantee 0.001 native accuracy for all eighteen coordinates on that record, even if only speed accuracy were demanded.

Here A is a screen-length amplitude normalization, not the observation allowance, an angle in degrees, or a measured signal norm. The cutoff A_star has not been evaluated. Accordingly (18) is a rigorous explicitly scaled conditional obstruction, not a numerically established statement for any selected practical wedge or supplied dataset.

## 9. A full-rank, fixed-positive-amplitude modulus lower bound

The construction also explains a global-to-local distinction that a Jacobian LP constant cannot capture. First let h vary in [0,h0], h0 <= 0.003. The common angle chi(h), jet equations, and physical guards vary analytically. The contraction and remainder constants in Sections 5-6 can be bounded uniformly on this compact h interval, including h=0. Hence there is a positive common amplitude cutoff giving the exact corrections and bound (15) for every h in [0,h0].

Choose one fixed positive A strictly below that common cutoff and small enough that both original Jacobians have full rank at h=h0, as guaranteed by Section 7. The uniform implicit-function construction gives c(A,h) real analytic in h on an open interval containing [0,h0]: invertible D_vJ gives local analytic continuations, and uniqueness makes them agree. All original native parameter maps, including the nonzero-amplitude arg and arcsine maps, are analytic there.

For each system choose one original full18 Jacobian minor that is nonzero at h0. Composed with its corrected parameter family, this minor is a real analytic, nonidentically-zero scalar function of h. Its zero at h=0, if any, has finite order. Therefore there is h1 > 0, with h1 <= h0, such that BOTH minors are nonzero throughout 0 < h <= h1. This proves full18 rank on that entire punctured family at the ONE fixed positive amplitude A. It does not assume a uniform bound from the frequency-dependent proof in Section 7.

Consider only this restricted two-branch family. All wedges are bounded below by a positive multiple of fixed A, the source offset is exactly 1, and every system has full18 rank. For any eta > 0 choose

~~~text
h = min(h1, [eta/(A (2 pi t_c)^5)]^(1/5)).
~~~

The midpoint record overlaps both observation boxes and their largest native speed difference is (1009/1000)h. Thus the deterministic worst-case speed risk over this restricted family obeys

~~~text
risk(eta) >= (1009/2000)
             min(h1, [eta/(A (2 pi t_c)^5)]^(1/5)).   (19)
~~~

This is a nonzero-wedge, off-axis, full-rank modulus barrier at fixed amplitude. The limiting configuration h=0 has all speeds zero; there the sampled Jacobian has rank at most four, because all non-speed derivatives are constant two-vector sequences and all speed derivatives are two-vector sequences proportional to time. No uniform inverse Lipschitz or singular-value constant is asserted on the punctured family. The admissible fixed A and h1 remain positive but unevaluated quantities; selecting nonzero analytic minors establishes their existence, not useful sizes.

The full original prior already has stronger constant minimax floors from other gauges. Equation (19) is useful because it persists after excluding zero wedges, zero source excitation, and singular individual systems in this particular subclass, and identifies an independent finite-time spectral mechanism. It does not replace the record-specific envelope theorem or claim the fifth power is the worst possible singular exponent of the full optical inverse.

## 10. Consequences for a complete inverse and novelty boundary

1. Distinct/nonaliased modes and full local rank do not by themselves certify 0.001 recovery. The exact finite record can contain two separated regular branches.
2. A global candidate cover must retain both systems whenever the record allowance meets their separation scale. Frequency extraction or a local 17+1 correction on just one branch would be incomplete.
3. The example is far from refraction/traversal boundaries. Better critical-boundary compactification does not remove this separate slow-spectrum obstruction.
4. The uncertainty comes from an allowed joint change of wedges and phases as well as frequencies. An oracle fixing those nuisance parameters changes the experiment and can remove this construction.
5. A quantitative practical conclusion still needs the certified constants and a supplied record. The construction neither supplies a useful universal inverse upper constant nor claims the current dataset lies near these branches.
6. Taking more timestamps inside the same 9.95-second window cannot remove this deterministic two-system obstruction: equation (13) bounds the entire continuous time interval, so its midpoint function is within the same allowance of both systems at every extra time. A longer observation window changes the bound.

The standard source-superresolution comparison is Batenkov, Goldman and Yomdin, *Super-resolution of near-colliding point sources*, [arXiv:1904.09186](https://arxiv.org/abs/1904.09186) (published in Information and Inference, [DOI:10.1093/imaiai/iaaa005](https://doi.org/10.1093/imaiai/iaaa005)). The elementary divided-difference cancellation and its fifth power belong to that established phenomenon. The exact nonlinear optical jet correction, native-prior realization, original-clock nonresonance certificate and compatibility with the full18 physical rank proof are the additional assertions proved here.
