# Independent certificate-engine review

**Passed on 2026-10-03.** The bounded executable review finished with exit code 0. The exact machine-readable evidence is [independent_review.json](independent_review.json); the initial mathematical and provenance inspection is [REVIEW_CONTRACT.md](REVIEW_CONTRACT.md).

This pass concerns a forward/domain/observation box certificate engine and one explicitly constructed synthetic full-vector fixture. It is not a practical complete inverse, a hardware-data result, or a formal proof of the Python program.

## What was reviewed

The engine implements the coupled-vector Snell model, not the independent-axis forward model. The eighteen input chart coordinates are all retained. The exact clock is k/20 for k=0,...,199. The full shared geometry is represented affinely in (b_x,b_y,g,d,constant); no geometry is fitted independently to individual rows.

Source review checked:
- Outward dyadic endpoint rounding uses exact rational and integer arithmetic. All four multiplication/division endpoint combinations are used; zero-containing divisors are rejected.
- Square roots use integer-square-root floor/ceiling comparisons. The kernel rejects an interval with negative lower endpoint.
- Pi uses exact alternating-series Machin bounds; trigonometric prior endpoints use polynomial evaluations with explicit Taylor remainder bounds.
- The sampled rotor identity uses rational phase/speed charts, preserving the signed rotor and fixed clock.
- The forward recurrence, positive branch selection, tilted-plane traversal, and affine geometry agree with the audited vector model.
- Prior containment and disjointness are conservative. A box crossing an unresolved prior endpoint cannot receive the whole-box retained verdict.
- Where Delta straddles zero, the engine explicitly evaluates its nonnegative-root subset and marks whole-box physical validity false. Such enclosures concern potentially physical parameters. A later contradiction can still exclude that subset; this is not an assertion that negative-radicand parameters have a real output.
- Retained requires strictly positive physical lower bounds and, when observations are supplied, containment of every output enclosure in its observation band. Overlap alone cannot establish compatibility.
- Excluded requires a proved strict-guard failure, prior disjointness, or disjoint observation interval. Uncertain denominators produce unresolved.

Original Dropbox fmodel.py and physical_jet.py were inspected read-only and found to trace axes independently. They were neither imported nor executed. The previously saved canonical 200-record was not used.

## Independent executable evidence

The [independent reference](independent_reference.py) uses normalized three-dimensional exit normals, vector Snell refraction, and direct three-dimensional ray/plane intersections. Its rotor values are formed from exact rational complex products. It does not use the engine's affine position recurrence. It evaluated only the existing fixture's one generating point; it created no second dataset.

The reference uses the same separately reviewed interval kernel at 224-bit precision, while the engine certificates use 80-bit precision. Thus formula independence is stronger than arithmetic-library independence; the shared kernel is part of the explicitly reviewed trusted code.

[verify_spotcheck.py](verify_spotcheck.py) passed:

| Check | Result |
| --- | --- |
| Deterministic rational arithmetic, root, domain and constant checks | 515 passed; no random cases or floating arithmetic |
| Independent direct-vector screen coordinate enclosures | All 400 contained in engine point enclosures |
| Reference coordinates inside the fixture's stated point-rounding error | All 400 |
| Independent exact affine support evaluations over shared geometry | 1,600 passed: 1,200 traversal rows and 400 screen rows |
| Small supplied box physical lower bounds | All 1,200 traversal and 2,400 optical bounds strictly positive |
| Small supplied box outputs inside observation allowance | All 400 |
| Data-only final-prism margins checked using the radical formula for tan(18 degrees) | All 200 |
| Far-distance certificate replay | Passed |
| Tampered witness and changed input | Both rejected |

The independent reference's largest coordinate width is exactly

1427/13479973333575319897333507543509815336818572211270286240551805124608.

The certified all-sample final-prism ratio lower bound is exactly

57039989462345798531330865/64073068439575346259427328.

This data-only margin is conditional on original-prior full-vector compatibility. It does not establish that any compatible physical system exists.

The original spot-check also verifies rejection of binary-float inputs, a stored floating clock endpoint of 9.950000000000001, and explicitly canonical provenance. These input labels are caller assertions and do not authenticate unknown real-world data provenance.

## The four saved cases

Every case uses the same [synthetic full-vector record](spotcheck/synthetic_fullvector_record.json).

- **Supplied small box: retained.** Every one of the eighteen coordinates has positive width, with chart halfwidth 1/100000000. All 200 samples are strictly physical and every output enclosure lies in the stated allowance. The box was supplied around the construction, not recovered.
- **Far-distance box: excluded.** The first x-output enclosure is disjoint from its observation band. This excludes the entire specified box.
- **Nontransmitted box: excluded.** The first prism's first-sample exit-radicand upper bound is negative.
- **Full original-prior outer box: unresolved.** The first outgoing-axial interval overlaps zero. This is an enclosure limitation, not proof that an actual admissible ray has zero axial component. The original-prior compatible set has not been localized.

The synthetic observation allowance is exactly

33070397787796324767/2417851639229258349412352.

It was derived from the supplied small box's forward enclosure to validate the implementation. It is not an inferred sensor-error bound, a favorable noise campaign, or a demonstrated recovery threshold.

## Reproduction and identity

Using the existing WSL Python, from Windows:

    wsl.exe -d Ubuntu -- python3 -B /mnt/c/Users/josep/Documents/Codex/2026-10-02/task/full18_research/certificate_engine/verify_spotcheck.py

This reads the saved fixture/certificates and rewrites only independent_review.json. It does not execute the original project or generate observations.

Frozen source identities checked before and after the review:

- engine.py: 7cbce954039aa9cb93e00cedb6ae2b4d4a760de8eb46edc8f41a76c5637eba6b
- interval.py: 3ddc0eaa3430028211d41962fcb2ce6d22c5c70dc24f89c3fae83245b8c90cbe
- independent_reference.py: 61abcf7120446b9fb94b20e04dc2a57bde2b6eb363ce33d4a2b88ca1c795546c
- verify_spotcheck.py: 408edb1eebcb0ae75dc78e6c0bb2cf5f8e11268cd8127826a4347eaf47cbaca0
- spotcheck.py: 72647f53580595994171b35471efcac5f1785357f1e55455042384261a5e5803
- synthetic_fullvector_record.json: 1c83fb2ce416e7e52971e0feb0c7e2f70d91e5d92088d516a3b3ce8aad99de9c

## Remaining bottleneck

This engine performs interval enclosure of complete affine geometry fibers but does not solve their joint LP feasibility problem. It has no adaptive global cover, all-sheet continuation, exact exceptional-stratum fallback, certificate of global uniqueness, or all18 native-coordinate recovery. Interval dependency can leave broad boxes unresolved.

The next implementation work is sparse exact geometry/dual separation and a replayable complete optical cover. A qualifying actual full-vector 200-record, its coordinate units, and a justified bounded-error interpretation are still needed before record-specific recovery claims.

