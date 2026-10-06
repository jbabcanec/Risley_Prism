# Related work and publication validation gap

Research draft for expert review. This targeted comparison does not establish priority or first-ever novelty. Independent symbolic checks within this project are not external peer review.

Third-order Risley optics predates this work. Yajun Li derived a nonparaxial thick-prism model and its third-order expansion for beam steering in 2011. That paper's inverse problem concerns directing a beam to a requested direction; it should not be equated automatically with blind recovery of eighteen physical parameters from a sampled screen trace. [Li, 2011](https://doi.org/10.1364/AO.50.000679)

Numerical calibration of wedge angles, refractive indices and installation parameters is also established. Li et al. used genetic-algorithm identification and reported experimental pointing improvements in 2017. Its full text was not completely accessible in this comparison, so no exhaustive fixed-versus-estimated parameter list is claimed. Yuan et al.'s 2023 telescope paper explicitly calibrates six errors: two encoder zeros, two wedge angles and two refractive indices, using recorded gimbal directions and prism encoder readings. This supplies a concrete calibration comparator with additional measurement information. [Li et al., 2017](https://doi.org/10.1364/AO.56.007358); [Yuan et al., 2023, Section 2 and Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC10051434/)

Fourier-based, target-free calibration of prism main-section zeros already appears in Li and Zhou's dispersion-image method. Three-prism forward/inverse modeling also has substantial precedent: Li, Liu and Sun treated blind zones and inverse steering by ray-tracing refinement; Qin et al. developed nonparaxial closed-form scan-pattern relations. These are relevant foundations, not evidence that blind physical identifiability has already been settled for the present observation map. [Li and Zhou, 2021](https://doi.org/10.1364/AO.440678); [Li et al., 2017](https://doi.org/10.1364/OE.25.007677); [Qin et al., 2024](https://doi.org/10.1016/j.optcom.2024.130915)

The closest audited calibration comparator is Brazeal, Wilkinson and Hochmair's Livox Mid-40 observation model. It uses exact three-dimensional vector Snell refraction, two identical prisms with shared wedge/index parameters, azimuth/zenith observations and an extended Kalman filter. Its thirteen model parameters are reduced by fixing air index, wedge angle and one tilt. It discusses wedge/index/air correlation without supplying the invariance proof developed for the distinct centered configuration considered here. The Livox model is not merely paraxial; the present three-prism screen-position theorem neither refutes it nor directly establishes identifiability for its sensor and observation model. [Brazeal et al., 2021](https://doi.org/10.3390/s21144722)

The general numerical machinery also has established origins. Elimination of linear variables from separable nonlinear least squares is classical variable projection; confluent Prony systems and their local accuracy have an existing theory. The present constrained geometry elimination and finite-record arguments should credit these foundations while identifying their specific optical hypotheses and contributions. [Golub and Pereyra, 1973](https://doi.org/10.1137/0710036); [Batenkov and Yomdin, 2013](https://arxiv.org/abs/1106.1137)

The defensible proposed contribution is the combination of blind finite-record physical eighteen-parameter local identifiability, an explicit leading inverse curve and quadratic ambiguity breaker, exact conditional geometry elimination, excitation obstructions, and conditional native-coordinate uncertainty certificates. Formal leading coefficients, exact finite-angle inference, local rank, global uniqueness and useful conditioning remain separate claims.

No physical experiment or successfully evaluated useful finite-noise certificate has yet been established for this project. Publication requires expert mathematical review, stronger literature comparison, explicit remainder and conditioning bounds, complete treatment or disclosure of surviving branches, and validation of all original observation constraints. Experimental validation remains a separate future requirement; this revision authorizes no empirical campaign.

## Primary-source access and claim provenance

| Source | Verified access and limitation |
| --- | --- |
| Li 2011, AO 50, 679–686 | Optica publisher abstract inspected; supports nonparaxial thick-prism and third-order scope. Full text was not retrieved in this lane. |
| Li et al. 2017, AO 56, 7358–7366 | Optica publisher abstract inspected; wedge/index/installation identification is explicit. Full parameter list remains unverified. |
| Yuan et al. 2023, Micromachines 14, 569 | Publisher-deposited full text at PMC inspected, especially Section 2, equations (9)–(10), Table 1. The paper's title concerns a Fizeau telescope; the calibration is a section within it. |
| Li and Zhou 2021, AO 60, 10437–10447 | Optica publisher abstract inspected; target-free zero calibration uses dispersion images and Fourier characteristics. |
| Li et al. 2017, OE 25, 7677–7688 | Publisher search-index abstract and author abstract in PubMed verified. Direct publisher full-text rendering was unavailable. |
| Qin et al. 2024, Optics Communications 570, 130915 | Author institution's [publication record and abstract](https://research.buaa.edu.cn/en/publications/closed-form-analytical-solution-and-scan-pattern-shaping-theory-o/) inspected; publisher full text unavailable. |
| Brazeal et al. 2021, Sensors 21, 4722 | [NOAA primary repository metadata/abstract](https://repository.library.noaa.gov/view/noaa/32976) verified in this lane. Detailed parameter comparison uses the parent research lane's supplied full-PDF audit; direct PDF/MDPI fetching failed here. |
| Golub and Pereyra 1973, SINUM 10, 413–432 | SIAM publisher abstract inspected; its linear/nonlinear separation is explicit. |
| Batenkov and Yomdin 2013, SIAM J. Applied Mathematics 73, 134–154 | Author-deposited arXiv abstract and publication metadata inspected. |

Access notes distinguish fresh verification from the supplied parent audit. Links point to primary publication or author/institutional repositories; search snippets are not treated as a substitute for unavailable detailed methods.
