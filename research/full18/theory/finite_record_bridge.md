# An exact finite-sample bridge for the cubic local-rank construction

This note independently verifies the temporal linear-algebra part of the incoming cubic-invariant construction. It is a proof, not a recovery run. It uses ideal timestamps k/20 and does not establish a finite-noise or global inverse by itself.

## 1. A native rotor choice with no degree-three alias collisions

Choose N=(1,7,49)/20 Hz=(.05,.35,2.45), all inside the original native box. For a Fourier multi-index m with |m|_1<=3, its sampled carrier is

\[
\exp(2\pi i(m\cdot N)t_k)=z_m^k,
\qquad z_m=\exp\left(2\pi i\frac{m_1+7m_2+49m_3}{400}\right).
\]

There are63 indices:1+6+18+38. Their integer numerators lie in[-147,147]. Thus equality modulo400 is ordinary equality. To prove uniqueness, subtract two index vectors. Each coordinate difference lies in[-6,6]. If the third difference is nonzero, the absolute49-term exceeds the maximum42+6 contribution of the lower terms. If it is zero and the second difference is nonzero, the7-term exceeds the possible first-coordinate difference6. Hence all three differences must vanish.

This proves distinctness of every degree-at-most-three sampled node, including reflected sidebands. It does not claim a useful numerical separation or conditioning bound for estimating them from noise.

## 2. Frequency derivatives fit within the same200 rows

Unknown speeds produce additional temporal columns k z_m^k at the six fundamental nodes m=+e_i and m=-e_i. Treat these six nodes as multiplicity2 and the other57 as multiplicity1. The total number of columns is69.

The first69 rows k=0,...,68 form a nonsingular confluent Vandermonde matrix. To verify this without a determinant formula, suppose a row coefficient vector b_k annihilates every column and put P(z)=sum_(k=0)^68 b_k z^k. At every simple node P(z_m)=0. At each repeated fundamental node both P(z_m)=0 and z_m P'(z_m)=0; z_m is nonzero. Therefore P has69 roots counted with multiplicity, while its degree is at most68. It must vanish identically. The square matrix is nonsingular, so all69 columns are independent on the full200-row record too.

This argument supplies an exact coefficient readout on the truncated temporal space: invert the first69-row matrix, or use any exact left inverse on200 rows. It uses the actual finite positions as its input, rather than assuming temporal derivatives or torus coefficients were measured. Real sine/cosine versions follow by an invertible change of basis between conjugate carrier pairs.

## 3. How this enters a full18 local-rank argument

The centered-normal first-order response has three amplitudes, three phases and three speeds, giving9 directions. The proposed normalized cubic construction addresses five glass/gap/distance directions after compensating wedges to keep first-order amplitudes fixed. Two baseline source directions remain order1; two baseline-compensated beam directions first appear at quadratic wedge order. The dimensions9+5+2+2=18 match the native unknowns.

For a rigorously derived Jacobian family J(epsilon), use the temporal readout above and the declared parameter column scaling. If the resulting18-by-18 minor has an analytic limit M0 with det(M0) nonzero, continuity proves a nonzero actual finite-record minor for all sufficiently small positive epsilon. Higher optical orders are not assumed absent: after the stated scaling their remainders must tend to zero, as established by an analytic Taylor argument on a regular compact branch.

The temporal proof in sections 1-2 is complete. The exact cubic recurrence, normalized five-variable determinant, beam/source block and remainder scaling are now audited together in [full18_local_rank.md](full18_local_rank.md). That combined argument proves full eighteen-parameter local rank for a sufficiently small nonzero wedge family on this ideal clock. It establishes neither global uniqueness nor a practical noise guarantee.

The coefficient readout and column scaling can amplify errors strongly as wedges shrink. A separate explicit finite-noise bound and the global competitor exclusion of REPORT.md are still required.
