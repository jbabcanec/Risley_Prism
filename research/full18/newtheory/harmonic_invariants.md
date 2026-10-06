# Finite-record harmonic construction for the original full18 inverse

2026-10-02. New derivations and a small checked implementation. This is a
constructive frontend and a conditional certification bridge, not a theorem
that all eighteen parameters are recoverable from every native 200-point
record. All eighteen physical unknowns and their original bounds remain in
the final inverse. No new measurements, known hardware, or reduced noise
allowance are introduced.

## 1. Exact structure and the assumption that must remain visible

Write `gamma_i(t)=2*pi*N_i*t+phi_i`, where phi is the initial phase in radians,
and `A_i=sin(ax_i)`. In the canonical independent-axis optics the effective
exit-face sine is exactly `A_i*cos(gamma_i)` on x and `A_i*sin(gamma_i)` on y.
This follows from the existing normal construction: its normalization is
`sqrt(1+tan(ax_i)^2)=1/cos(ax_i)` on the native wedge interval. Thus the exact
position response has the form

    x(t) = H_x(cos(gamma_1), cos(gamma_2), cos(gamma_3)),
    y(t) = H_y(sin(gamma_1), sin(gamma_2), sin(gamma_3)).             (1)

The unknown signed wedges, glasses, physical order, geometry, and source
coordinates are all inside H_x,H_y. The two H functions have common optical
hardware but different source angle and position. They are not independent
physical functions even though the surrogate below relaxes that coupling.

**Scope condition.** At a measured time (1) needs only its actual physical
branch. A global torus Fourier series or cube Chebyshev series with a uniform
tail requires an appropriate real and complex extension of H. Strict Snell
and grazing margins at 200 timestamps do not establish such an extension.
Native [1.3,1.8] glass and +/-25-degree source bounds cannot silently be
replaced by the smaller R5 family. The constructions below explicitly say
where a whole-torus certificate is required.

## 2. Reflection-tied dictionary: fewer unknowns, not missing sidebands

If H has an absolutely convergent Chebyshev expansion, set

    H_x(u) = sum_(h in N^3) a_h product_i T_(h_i)(u_i),
    H_y(u) = sum_(h in N^3) b_h product_i T_(h_i)(u_i),              (2)

with real coefficients. Substitution into (1) gives exactly

    Phi_x[k,h] = product_i cos(h_i*gamma_i(t_k)),
    Phi_y[k,h] = product_i cos(h_i*(gamma_i(t_k)-pi/2)),
    x = Phi_x a,  y = Phi_y b.                                   (3)

For a finite cutoff |h|_1<=H, there are binomial(H+3,3) real columns per axis:

| H | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| columns | 20 | 35 | 56 | 84 | 120 | 165 | 220 |

Each column contains every reflected signed Fourier sideband belonging to h,
with their exact equal amplitudes and phase relations. Degree five has 231
signed lattice lines but only 56 real orbit coefficients. Consequently the
unstructured 231-line-versus-200-row obstruction is not an obstruction to
this structured fit. There is no assertion that its 56 columns are always
independent or well conditioned: collisions, slow rotors, aliasing and the
finite trajectory can make them dependent or nearly so. H=9 already exceeds
the 200-row per-axis rank limit, before considering such conditioning.

For fixed six rotor coordinates `(N_1,N_2,N_3,phi_1,phi_2,phi_3)`, fit a and b
by separate real linear least squares or by hard-band linear programs. Only
six nonlinear coordinates remain in this relaxed frontend. A degree-3 then
degree-5 continuation supplies rotor proposals to the exact full18 solver;
it is different from merely weighting a projection of the old full model.

The exact full-column-rank residual Jacobian is especially inexpensive. For
one axis, let `c=Phi^+ y`, `r=Phi*c-y`, `P=Phi*Phi^+`. Then

    dr/dq = (I-P) Phi_q c - (Phi^+)^T Phi_q^T r.                  (4)

The second term is necessary at a nonzero residual. Derivatives of Phi are
products with one cosine replaced by `-h_j*sin(h_j*gamma_j)` and multiplied
by `2*pi*t_k` for N_j or `pi/180` for a native degree phase. No division by a
cosine is needed. At rank changes the usual smooth full-rank formula does
not apply; the implementation explicitly marks a truncated-rank derivative.

**Exact remaining symmetries of this relaxation.** Reversing any one rotor,
`(N_j,phi_j)->(-N_j,-phi_j)`, leaves Phi_x unchanged and multiplies each
Phi_y column by `(-1)^h_j`. Free b coefficients absorb that factor. Permuting
the three rotor labels also merely permutes columns. Therefore the free
dictionary alone cannot recover individual signed orientations or physical
order, regardless of residual size. All eight sign branches and all six
physical orders may still need exact optical completion. Phase bounds alone
do not remove this reversal because +/-18 degrees is symmetric.

For comparison with complex-line phase arguments, put
`z=x+i*y`, `s(h)=number of nonzero h_i`, and `h=|m|`. The torus coefficient is

    z_m = exp(i*m.phi) * 2^(-s(h)) *
          [a_h + i*exp(-i*pi*sum(m)/2)*b_h].                    (5)

If |m|_1 is odd, sum(m) is odd and the bracket is real, so its phase equals
`m.phi modulo pi` whenever it is nonzero. Several m that coincide modulo
the 20-Hz sampling rate contribute a sum to one observed line; that sum need
not satisfy any one term's phase rule. Equation (5) is a torus identity,
not an identity for a finite-window FFT coefficient with leakage.

## 3. A quantitative bridge from positions to the existing optical jets

Suppose, for a declared hardware/rotor box, the exact per-axis torus response
extends holomorphically to `|Im(gamma_i)|<=sigma` and is bounded by M there.
Let `x=exp(-sigma)`. Fourier contour shifting and the reflection relation give

    |a_h|, |b_h| <= C_h = 2^s(h) M x^|h|.                       (6)

The function-tail bound for total degree H is

    tau_0(H) = M sum_(l>H) (4*l^2+2)*x^l.                      (7)

This is the same signed-shell count as the old Fourier bound: coefficient
tying reduces the number of free unknowns, not the optical remainder.

For a desired Taylor coefficient alpha at the flat normalized input, define

    w_alpha[h] = product_i [u^alpha_i] T_(h_i)(u).

These are exact integers, not numerical differentiated observations. For
`n>0`, `n>=r`, and n-r even, the univariate coefficient is

    [u^r]T_n(u) = (-1)^((n-r)/2) * n * 2^(r-1)
                   * ((n+r)/2-1)! / (((n-r)/2)! * r!).          (8)

It is zero otherwise; T_0=1 is handled separately. The Chebyshev ODE gives
`(r+2)(r+1)c_(r+2)=-(n^2-r^2)c_r`; starting with |c_0|<=1 and |c_1|<=n proves
`|[u^r]T_n|<=n^r/r!` wherever the coefficient is nonzero. Therefore a valid
omitted Taylor-jet allowance, with r=|alpha|_1, is

    tau_alpha(H) = M/alpha! *
                  sum_(l>H) (4*l^2+2)*l^r*x^l.                 (9)

Each sum is an explicit rational function of x: apply `(x*d/dx)^r` to the
closed geometric-shell tail in (7). Uniform small function error alone
does not justify setting this derivative-weighted allowance to zero.

At fixed true rotors, let `ell_alpha=w_alpha^T Phi^+`. Assume Phi has full
column rank, or verify the weaker row-reproduction condition
`w_alpha^T Phi^+ Phi=w_alpha^T`. Under that assumption, for hard per-coordinate
observation error eta, the finite readout obeys

    |ell_alpha*y_observed - true_jet_alpha|
        <= (eta+tau_0)*||ell_alpha||_1 + tau_alpha.              (10)

This follows by substituting `y=Phi*c+tail+e` and bounding the exact linear
functional of the two per-sample error vectors. There is no independent
noise or root-K averaging assumption. Formula (10), followed by interval
evaluation of U's cubic-order polynomial or Y's coupled quartic inverse,
is an actual bridge to those notes' input quantities. It does not presume
those coefficients were directly measured.

At a rank-deficient dictionary, such as an exact rotor collision, do not
assert (10) without that row-reproduction check. Define the row defect

    n_alpha^T = w_alpha^T - ell_alpha*Phi.

The missing term is `n_alpha^T*c`; if the coefficient bounds C_h in (6) hold,
add the following allowance to the right side of (10):

    nu_alpha = sum_h C_h*|n_alpha[h]|.                         (10a)

The resulting inequality is valid at every rank and for any chosen numerical
linear readout ell_alpha, including a truncated pseudoinverse. It also gives
a direct way to charge a validated nonzero row-reproduction rounding error.
Absent coefficient bounds or row reproduction, no finite noise-only jet
guarantee follows. For a rank-one, twenty-column dictionary with all rotors
stationary, the new check constructs coefficients of magnitude at most one
that attain a jet error 8.8 despite zero measurement noise; (10a) gives exactly
8.8. Stationary or colliding rotors are retained as valid hypotheses rather
than silently discarded to recover the full-rank condition.

Unknown rotors are handled by a box, not by pretending their estimated
values are exact. Let Phi_0 be a center dictionary and assume simultaneous
coordinate radii delta_N and delta_phi (the latter in radians). Real cosine
product differentiation gives the safe entrywise bound

    |Phi_*[k,h]-Phi_0[k,h]|
      <= sum_i h_i*(2*pi*|t_k|*delta_N_i+delta_phi_i).

Using (6), define delta_k as the sum of this bound times C_h over retained h.
Then add `sum_k |ell_alpha[k]|*delta_k` to (10), together with the center
dictionary's nullspace charge (10a) whenever row reproduction is not proved.
The inequality holds for every truth in that rotor box. Finite-precision implementations must also
enclose matrix inversion, dot products and the analytic constants.

The structured route reduces the retained coefficient count from 231 free
signed lines to 56 real columns at degree five. That reduction alone does
not establish a useful tail or jet-accuracy budget. No actual native-wide
sigma,M or useful resulting eta has yet been established. Original C requires whole-torus branch margins;
R5 proofs do not cover all native competitors. The key remaining calculation
is a boxwise analytic remainder with useful constants and a certified
lower-rank bound for this specific structured dictionary.

Indeed the generic shell majorant (7) is still impractical at illustrative
strip widths. For the hypothetical target `eta/M=1e-8`, direct evaluation of
its closed geometric-sum formula gives:

| assumed sigma | first H with tau_0/M <= 1e-8 | real columns per axis | tau_0(H)/M | tau_0(H-1)/M |
|---|---:|---:|---:|---:|
| 0.5 | 57 | 34,220 | 9.1788742e-9 | 1.4629970e-8 |
| 1.0 | 26 | 3,654 | 9.0650477e-9 | 2.2890243e-8 |

At degree eight, the last degree with fewer than 200 free columns, the same
normalized bounds are 13.04846 and .07280957, respectively. These are checked
floating-point illustrations of the analytic majorant, not interval proofs
or asserted optical strip widths. They show that coefficient tying has not
made this generic tail bound sufficient at 200 samples; the derivative-weighted
jet requirement is stricter still. This is a failure of this sufficient bound
and free-coefficient readout route, not an impossibility theorem for the
18-dimensional physical optical family. Useful certification needs sharper
hardware-dependent tails, coupled physical constraints, or another route.

## 4. A safe frequency-box rejection test with the same 200 observations

At fixed rotor proposal, define

    d_inf(y,range(Phi)) = min_c ||y-Phi*c||_infinity.

It is a linear program. Its exact dual is

    max_w w^T y, subject to Phi^T w=0 and ||w||_1<=1.            (11)

If (7) is certified for every proposed competitor and the primal lower
bound exceeds eta+tau_0, that rotor hypothesis is impossible. This is an
exclusion test; a passing relaxed coefficient fit is not a physical system.

For an entire rotor box centered at Phi_0, use delta_k above. A dual vector
with `Phi_0^T w=0` proves exclusion when

    |w^T y| > (eta+tau_0)*||w||_1 + sum_k |w_k|*delta_k.        (12)

More generally, without exact annihilation, add
`sum_h C_h*|w^T Phi_0[:,h]|` to the right side. This latter form naturally
absorbs a validated nonzero rounding enclosure for Phi_0^T w. Rows from both
axes may be combined, preserving separate coefficient blocks and the same
six rotor variables. Subsequent splitting of rotor boxes retains collisions
and arbitrarily small speeds; no fixed spectral cutoff is required.

The program (11) and inequality (12) specify an implementable pruning rule
for a global rotor search. Without a remainder valid for every competitor,
a low residual threshold from this surrogate is only an initializer rule.
Choosing the tail from the fitted record's own residual would not validate
the exclusion of other physical systems.

## 5. Close frequencies: clustered moments instead of deleting lines

For a cluster with frequencies `f_j=f_c+delta_j`, |delta_j|<=Delta, center the
sample times at t_c and write tau=t-t_c. Absorb the center-time phase into
the complex coefficient d_j. For any nonnegative integer R,

    sum_j d_j exp(2*pi*i*f_j*t)
      = exp(2*pi*i*f_c*tau) sum_(r=0)^R mu_r*tau^r + remainder,
    mu_r = (2*pi*i)^r/r! * sum_j d_j*delta_j^r,
    |remainder| <= sum_j |d_j| *
           (2*pi*Delta*max|tau|)^(R+1)/(R+1)!.                 (13)

The bound uses the integral Taylor remainder along real delta_j*tau; the
complex exponential has unit modulus there, so no extra exponential factor
is needed. The 200-point grid has max|tau|=4.975 s. For a .01-Hz half-width
cluster, `2*pi*Delta*max|tau|<.313`; R=5 gives a relative remainder below
1.31e-6 times the cluster absolute-amplitude sum. This is an illustrative
numerical bound, not a detector-noise guarantee.

The stable basis is polynomial times one carrier, rather than nearly equal
Vandermonde columns. Moment envelopes are
`|mu_r|<=(2*pi*Delta)^r sum|d_j|/r!`; retaining these inequalities prevents
arbitrary polynomial interpolation. At Delta=0, the moments above degree
zero vanish exactly. This cluster treatment is compatible with slow rotors
and collisions, but it does not separate exactly coincident contributions:
hardware coupling still has to identify them. If centered phases are used
with the native time-zero phase prior, preserve the relation
`phi_center=phi_0+2*pi*N*t_c`; do not apply the +/-18-degree prior at t_c.

## 6. Why an exact finite constant-coefficient annihilator cannot be the cure

For a finite Fourier truncation with distinct sampled nodes
`z_m=exp(2*pi*i*(m.N)/20)`, the shift polynomial

    p(S)=product_m (S-z_m)

annihilates the truncated signal. For the exact optics it leaves the omitted
harmonics. If p has coefficients h_j, each filtered observation has hard
noise at most `eta*sum|h_j|`; its optical remainder is bounded by

    sum_(omitted m) |c_m|*|p(z_m)| <= tau_0*sum|h_j|.          (14)

Coefficient normalization cannot remove the ratio between true rejection
margin and this uncertainty. Products can have large coefficient sums;
using the data-adapted dual (11) is usually preferable to explicitly forming
a high-degree annihilator. Merge exactly aliased nodes algebraically; do
not assume individual higher harmonics remain below the Nyquist frequency.

There is a simple exact obstruction to a universal finite annihilator.
Take a valid native configuration with only the last wedge active, zero
source angle and position, refractive index n>1, and A nonzero but small.
The outgoing tangent as a function of `s=A*cos(gamma)` is

    t(s) = s*(n*sqrt(1-s^2)-sqrt(1-n^2*s^2)) /
           (sqrt(1-s^2)*sqrt(1-n^2*s^2)+n*s^2).                (15)

The screen response is d*t(s). At s=1/n the coefficient of the local radical
sqrt(1-n^2*s^2) in t is nonzero (its derivative with respect to that radical,
holding the other analytic factors at the branch point, is -n^2). Hence this
function is not a finite Laurent polynomial in exp(i*gamma) and has infinitely
many nonzero harmonics. Choose an allowed speed with N/20 irrational. If a
nonzero finite polynomial p(S) annihilated its whole sampled sequence, density
of the sampled phases and continuity would give a torus identity, forcing
`p(exp(2*pi*i*m*N/20))*c_m=0` for every m. Infinitely many nonzero c_m provide
infinitely many distinct roots, impossible for a nonzero polynomial.

This argument concerns a true recurrence for the sequence, not merely a
fit to 200 values. Every scalar 200-point record has a nonzero length-101
complex annihilator vector because its associated matrix has 100 rows and
101 columns. Such a finite interpolation identity alone cannot certify
optical structure or extrapolation. The source note H's variable-coefficient
*Fourier-index* recurrence is a different and valid statement; it should not
be confused with a fixed constant-coefficient recurrence across timestamps.

## 7. Deliverables and verification

- `harmonic_dictionary.py`: observed-data-ready dictionary, its six native
  derivatives, variable projection, exact Chebyshev-to-Taylor weights and
  per-jet hard-noise gains. No project import or source modification.
- `check_harmonic_dictionary.py` and `harmonic_dictionary_checks.json`:
  all derivative checks passed on nonzero residuals at H=3 and H=5; worst
  relative error 2.04e-8. Exact integer polynomial coefficients were checked
  against an independent polynomial representation through degree twelve.
  The rotor-reversal symmetry was numerically checked to 8.85e-14.
  The rank-deficient nullspace charge and the illustrative tail budgets above
  are included in the same reproducible check report.
- Those checks establish implementation consistency, not a whole-prior
  tail certificate, reliable frequency initialization, or full18 recovery.

Read-only source references: `paper/P2_THEORY_2026_09_28/B_large_sieve_coefficients.md`
(unstructured finite-window fitting); `C_analyticity_tail.md` (explicit
conditional strip and shell tails); `H_holonomic_structure.md` (different
Fourier-index recurrence); `K_joint_rotor_coefficients.md` (local joint
rotor/coefficient refinement); `U_three_prism_order.md` and
`Y_global_quartic_shape.md` (optical jet invariants and their restricted
families); `DIARY.md`, entries 2026-09-30 and 2026-10-01 (the unresolved
finite-record bridge and the distinction from controlled measurements).
