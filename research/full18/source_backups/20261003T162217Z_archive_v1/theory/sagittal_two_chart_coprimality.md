# Single-witness certificate: two sagittal norms have exactly one common root

## Certified outcome

At exactly the rational-rotor finite-wedge witness in `exact_sagittal_index_elimination.md`, take the two five-row charts

- I=(0,1,2,3,4),
- J=(0,1,2,3,5).

Their grouped sagittal norm polynomials P_I(G) and P_J(G) both have degree exactly 64. Over the exact real coefficient field of this witness,

**gcd(P_I,P_J)=G−9/4, up to a nonzero constant.**

This is a certificate using outward-rounded integer interval arithmetic. It does not interpret rounded coefficients as exact; it does not use a numerical parameter campaign or a hidden-truth estimator. It strengthens the earlier nonzero-chart witness and removes the degree-dropping-specialization concern for these two norm polynomials.

The proof program is `proof_checks/sagittal_two_chart_coprimality.py`. Run:

    python proof_checks/sagittal_two_chart_coprimality.py 1024

Its exact integer endpoints and every Euclidean leading-coefficient enclosure are saved in `proof_checks/sagittal_two_chart_coprimality_certificate.json`. The denominator of every saved endpoint is 2^1024. The matching execution transcript is `proof_checks/sagittal_two_chart_coprimality_1024.log`. A higher-precision execution at 4096 bits gave the same conclusion; its transcript is also retained. These are repeated arithmetic checks of the same two polynomials at one witness, not distinct parameter tests.

## 1. Exact witness and affine-column transformation

All witness parameters are those already fixed in the source theorem: indices 3/2, slope magnitudes 1/10, initial rotor phases zero, incident slopes (1/3,1/10), source (1,2), g=10, d=100, and rational per-sample rotors ((15+8i)/17,(4+3i)/5,(3+4i)/5). The new program generates samples 0 through 5 with the exact finite-wedge model represented by rigorous dyadic intervals. The earlier witness certification establishes the full 200-sample physical guards. The new program separately verifies every guard needed for its six samples and pairwise distinctness of their six alpha_k=|q_k|^2; thus no radical grouping ambiguity arises in either chart.

Let h_k=sqrt(G−alpha_k), H_k*=sqrt(9/4−alpha_k), and let p_k* be the incoming-third-entrance position at the witness. Add A(G)w* to the augmented determinant's last column. This leaves its determinant unchanged. The exact sagittal identity at the witness gives

F_k(w*,G)=[y_k−p_k*,u_k](h_k−H_k*).

The program therefore constructs the equivalent row whose first four entries are the original geometry coefficients, and whose last entry is r_k(h_k−H_k*), where r_k=[y_k−p_k*,u_k]. This transformation is an exact identity, not an adjustment to rounded data. It avoids needless numerical cancellation in the known root. It is used only in the proof witness, where w* is known; it is not proposed as a reconstruction step using hidden geometry.

## 2. Exact norm and exact deflation

The polynomial variable in the program is X=G−9/4. Hence

h_k^2=X+beta_k,  beta_k=9/4−alpha_k.

Each determinant is expanded by the Leibniz formula into 31 possible radical masks; the product of all five radicals is absent because the d column is root-free. Polynomial arithmetic in the quotient algebra uses

h_m h_n = h_(m XOR n) product_(j in m AND n)(X+beta_j).

At each radical elimination, write the current expression as E+h_j O and replace it by

E^2−(X+beta_j)O^2.

This is exactly multiplication by the sign conjugate. After all five eliminations, only a scalar polynomial of degree at most 64 remains. Nonzero dyadic scalings are allowed between eliminations; their effects are recorded. The final scaled polynomials are

Ptilde_I(X)=2^539 P_I(X+9/4),
Ptilde_J(X)=2^508 P_J(X+9/4).

Their leading coefficients are certified within these exact rational intervals:

31162515572236/10^56 &lt;= lc(Ptilde_I) &lt;= 31162515572237/10^56,
579335623826654/10^56 &lt;= lc(Ptilde_J) &lt;= 579335623826655/10^56.

Both intervals are strictly positive, proving full degree64 rather than merely presuming it from the upper bound.

The physical all-positive radical branch vanishes exactly at X=0, so each exact norm has zero constant coefficient. Consequently Q_I=Ptilde_I/X and Q_J=Ptilde_J/X are exact degree63 polynomials. Dropping the constant coefficient in the translated coordinate is precisely exact algebraic deflation; it is equivalent to synthetic division by G−9/4 before translating. The program checks that the computed constant-coefficient enclosures contain zero, but containment alone is not the justification for deflation: the exact optical identity is.

Both deflated constant coefficients are also certified nonzero (approximately −0.00681564549 and +0.00002442901118 after scaling). In particular, the common root is simple in both norm polynomials.

## 3. Certified polynomial Euclidean chain

Starting from Q_I,Q_J, perform ordinary polynomial long division using interval coefficients. Each divisor leading-coefficient interval excludes zero. During a long-division step the highest coefficient is mathematically canceled, so that coefficient is discarded exactly; all remaining coefficients are outward enclosures of the exact remainder. Only positive power-of-two factors (with possibly negative exponents) are applied between divisions. There is no tolerance-based trimming, no rounded-coefficient gcd, and no discarded remainder whose interval merely contains zero.

The 63 successive exact remainder degrees are

62,61,...,1,0.

Every leading-coefficient interval excludes zero. After the recorded rescalings, the final constant C satisfies the exact rational bounds

6519538297218639/10^16 &lt;= C &lt;= 6519538297218640/10^16.

The tighter saved dyadic interval has width at most 2^(−242). Therefore C is strictly positive. By the Euclidean algorithm, gcd(Q_I,Q_J)=1, equivalently their resultant is nonzero. An explicit giant resultant is unnecessary: the certified nonzero-remainder chain proves exactly the same coprimality statement.

The final remainder's displayed value is normalization-dependent and is not claimed to equal the unscaled resultant. Combining coprimality with exact deflation proves gcd(P_I,P_J)=G−9/4.

## 4. Why the arithmetic is rigorous

All enclosed values are represented by integer endpoint pairs divided by a fixed S=2^BITS. Addition and subtraction are exact endpoint operations. Multiplication takes the min/max of all four integer endpoint products and rounds outward to the dyadic grid. Division is used only when the divisor interval excludes zero; each endpoint ratio is individually rounded outward using integer division. Positive square roots use integer square roots, with the upper endpoint incremented whenever necessary. Rotors and starting witness values use exact fractions before enclosure. Displayed decimal values are never used in arithmetic or branch decisions.

All coefficient operations and all polynomial-division pivots are consequently enclosed, despite dependencies among the exact optical radicals and observed values. Dependency can only enlarge the enclosures. The proof succeeds with enough margin that a 1024-bit run suffices. The certificate contains actual integer endpoints, so decimal rendering does not enter the conclusion.

## 5. What the specialization can establish

This certificate itself proves a statement about the exact two witness polynomials. A generic conditional-recovery claim additionally needs a correctly specified coefficient graph/field, rather than an unjustified specialization assertion.

On a connected real-analytic physical parameter domain where the same retained-optics constructions and fixed two charts are defined, pull back both polynomials' coefficients along y=F(theta). The true g(theta)=n_3(theta)^2 is a common root. The degree64 leading coefficients and the degree-one subresultant coefficients are analytic functions there. The certified witness has both leading coefficients nonzero and gcd exactly degree one, so its degree-one subresultant is nonzero. Thus these functions do not vanish identically. On any irreducible analytic coefficient graph containing this witness, outside their proper vanishing loci, the two polynomials have gcd exactly degree one. If that subresultant is S_1(G)=s_1 G+s_0, then

G=−s_0/s_1

on the physical graph wherever s_1 is nonzero. The coefficients s_0,s_1 are polynomial expressions in the norm coefficients, so this is rational conditional recovery over their coefficient field. Full degree64 at the witness is important for transferring this fixed-degree subresultant statement.

The exact global coefficient-graph formulation and exceptional-record transfer should be assessed alongside the existing domain and boundary results. The certificate does not on its own show that every arbitrary supplied retained-optics/data point lies on this generic graph, certify that a record avoids the exceptional locus, or remove exceptional strata. It does not solve the remaining thirteen optical variables or establish a practical full-prior reconstruction runtime. Positive noise still produces feasibility tubes rather than a single exact index value.
