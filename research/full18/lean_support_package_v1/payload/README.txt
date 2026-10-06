Risley Lean supporting sources and existing verification evidence, package v1

This package preserves two previously compiled Lean sources and their successful
verification evidence. Packaging performed file reads, byte copies, inventory
checks and SHA256 comparisons only. No Lean compiler, check script, numerical
test, proof search, dependency installation or update was run to make it.

Recorded proof scope: 19 theorems, comprising 11 in RisleySupport.lean and 8 in
CriticalMargin.lean. The saved successful compiler logs report, for every theorem,
exactly [propext, Classical.choice, Quot.sound]. Existing verification records and
independent audits report no admitted goals, sorryAx dependencies or custom axioms.
These are earlier recorded results, not a new compilation or formal audit.

RisleySupport contains source-transport determinant/factorization/inverse lemmas,
a spatial recurrence identity, critical/external-flight algebraic sign lemmas,
and a square-root difference bound. The sign lemmas check the stated algebraic
derivative expressions; they do not formalize differentiation of the optical map.
CriticalMargin proves the conditional final-prism observation-margin implication
from explicit real-coordinate intersection, traversal, slope and observation
hypotheses. It does not derive all those hypotheses from the full optical model.

No full three-prism compiler, complete inverse, boundary compactification,
corrector/contraction theorem, native accuracy certificate, or concrete 200-sample
record is formalized by this package. The later chord and interval-domain work
adds no Lean coverage. Standard logical axioms are distinct from optical premises.

Layout:
- formalization/: exact source bytes, original READMEs, successful compiler logs,
  verification JSON, independent audit notes, and unexecuted original check scripts.
- dependency-environment/: exact lakefile.toml, lean-toolchain and lake-manifest.json
  snapshots from the existing environment used by the original direct-compiler
  scripts. The lock records mathlib and its transitive dependency revisions.
- MANIFEST.json: file sizes, SHA256 hashes, theorem names and recorded axiom scope.
  Its file inventory excludes MANIFEST.json itself to avoid a circular self-hash.

Frozen toolchain and dependency identity:
- Lean 4.33.0, Windows x86_64, release commit
  d8b18978322de05a8f3dba51ef03cf5461676c17.
- mathlib v4.33.0, commit db584cd6d46c92f209a44c0f1c829460d327499d.

Reproduction boundaries:
The dependency snapshot is metadata, not a bundled Lean installation or mathlib
checkout/build cache. Its default Lake target is the original environment's
Risley library, whose unrelated source is deliberately absent. This archive is
not a standalone default-Lake-build project. Restore the recorded compiler and
locked dependencies in an appropriate separate environment before a future check.
The original check.ps1 and check-critical-margin.ps1 show the exact historical
direct compiler calls and LEAN_PATH construction; their absolute local paths
must be reviewed/adapted for another machine. Neither script was run here.
The original READMEs also retain their historical workspace-relative paths.

Excluded deliberately: .olean objects, compiler/dependency binaries, build caches,
unrelated optical/project sources, and superseded failed-draft logs. Original
READMEs and verification JSON mention some excluded objects or failed drafts;
those historical references are preserved, and do not assert archive membership.
Saved logs contain the historical local compiler/workspace paths as provenance.

The companion risley_lean_support_sources.txt concatenates every package text
file with labeled byte-count/hash boundaries. Bytes between each BEGIN and END
boundary are the complete corresponding file bytes, with a separate delimiter
newline before END. The ZIP is the authoritative individually recoverable copy.

This package preserves the originals and records existing evidence. It makes no
new proof, physical validation, global uniqueness or practical recovery claim.
