# Frozen full18 accuracy sweep

All ten cases remain in every denominator. All errors compare physical prism positions without relabeling. Noise uses one fixed bounded pattern per level; this is empirical recovery evidence, not a uniform guarantee.

| Observation allowance | All18 errors <= .001 | Strict-model compatible fits | Complete |
|---|---:|---:|---|
| 0 | 7/10 | numerical noiseless test | True |
| 1e-06 | 7/10 | 7/10 | True |
| 0.0001 | 6/10 | 7/10 | True |
| 0.001 | 2/10 | 7/10 | True |
| 0.01 | 0/10 | 8/10 | True |

| Case | Noiseless maximum error | eta 1e-6 | eta 1e-4 | eta 1e-3 | eta 1e-2 |
|---|---:|---:|---:|---:|---:|
| random_00 | 3.7425618e-12 (ax1) | 9.5811883e-06 (d_W) | 0.00095813302 (d_W) | 0.0095821733 (d_W) | 0.095831488 (d_W) |
| random_01 | 8.7070351e-12 (ax2) | 1.2361982e-05 (d_W) | 0.0012362186 (d_W) | 0.012363567 (d_W) | 0.12377331 (d_W) |
| random_02 | 19.978212 (ax3) | 14.14719 (ax2) | 14.147409 (ax2) | 18.326793 (ay2) | 14.172185 (ax2) |
| random_03 | 15.930955 (ay2) | 15.946216 (ay2) | 15.946176 (ay2) | 15.945777 (ay2) | 0.0018110806 (bm_px) |
| random_04 | 9.8792086e-12 (ax2) | 2.5306867e-06 (d_W) | 0.0002531932 (d_W) | 0.0025330985 (d_W) | 0.025305451 (d_W) |
| random_05 | 1.0217605e-11 (d_W) | 4.7070076e-07 (bm_py) | 2.4858362e-05 (ax3) | 0.00039390194 (bm_py) | 0.0024863063 (ax3) |
| random_06 | 6.434675e-11 (d_W) | 5.8416493e-07 (ax2) | 5.8413726e-05 (ax2) | 0.00058403444 (ax2) | 0.005845479 (ax2) |
| random_07 | 33.990569 (ay2) | 30.075472 (ay2) | 30.07541 (ay2) | 33.99057 (ay2) | 7.7623967 (d_W) |
| random_08 | 6.5739414e-11 (d_W) | 4.0460034e-06 (d_W) | 0.00040465639 (d_W) | 0.0040473304 (d_W) | 0.041014805 (d_W) |
| random_09 | 2.5451641e-11 (d_W) | 1.2241674e-06 (ax2) | 0.00012236491 (ax2) | 0.0012236236 (ax2) | 0.012234155 (ax2) |

Coordinate units: Hz for N; degrees for ax, ay and beam angles; dimensionless glass indices; native unspecified position units for distances, gap and source positions.
Noiseless means no added noise on binary64 legacy-generated samples; it does not assert exact strict-model feasibility at eta=0.
Full coordinate errors, residuals, physical checks and candidate provenance are in FULL18_ACCURACY.csv and ACCURACY_SUMMARY.json.
