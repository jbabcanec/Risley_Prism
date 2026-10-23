#!/usr/bin/env python3
"""Produce the final analysis-ready dataset by dropping schema-incomplete rows.

Drops any case with a rung missing `min_cycles`. In practice that is exactly
the handful inherited from tag atlas2, which predates the diagnostic fields --
so the output has a uniform schema and needs no conditional handling.

KEPT DELIBERATELY: rows where x18 is null. Those are cases where solve18
produced no vector at all (the 'no generators detectable' failure). They carry
their `cert` field and are the hardest failures in the set -- dropping them
would bias the failure taxonomy toward the easy cases.

NOT A COMPLETENESS CRITERION: T_req / sigma_N / min_amp appear on FAILED rungs
only, by construction. ~80% of rungs lack them because nothing failed there.
Filter on presence when doing certificate analysis; never treat absence as a
defect.

The input is moved to results/_prefilter/ rather than deleted, because
aggregate.py globs results/sweep_*.jsonl and would otherwise read both files
and double-count every case.

  python3 finalize_dataset.py
"""
import json, os, shutil, sys

DEFAULT_IN  = "experiments/results/sweep_adaptive_merged_0000000.jsonl"
DEFAULT_OUT = "experiments/results/sweep_adaptive_ready_0000000.jsonl"


def main(src=DEFAULT_IN, dst=DEFAULT_OUT):
    if not os.path.exists(src):
        sys.exit(f"missing {src} -- run merge_results.py first")

    n = kept = dropped = x18_null = rungs = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            try:
                r = json.loads(line)
            except Exception:
                continue
            n += 1
            trail = r.get("trail") or []
            if any("min_cycles" not in row for row in trail) or not trail:
                dropped += 1
                continue
            kept += 1
            rungs += len(trail)
            if any(row.get("x18") is None for row in trail):
                x18_null += 1
            fout.write(json.dumps(r) + "\n")

    hold = os.path.join(os.path.dirname(src), "_prefilter")
    os.makedirs(hold, exist_ok=True)
    shutil.move(src, os.path.join(hold, os.path.basename(src)))

    mb = os.path.getsize(dst) / 1e6
    print(f"read            : {n:,}")
    print(f"kept            : {kept:,}")
    print(f"dropped         : {dropped:,}   (rung missing min_cycles)")
    print(f"rungs kept      : {rungs:,}  ({rungs/max(kept,1):.2f} per case)")
    print(f"  of which with x18 == null : {x18_null:,}  "
          f"(total front-end failures, kept on purpose)")
    print(f"\nREADY -> {dst}  ({mb:.0f} MB)")
    print(f"superset moved  -> {hold}/  (out of aggregate.py's glob)")
    print(f"\nNow run:  python experiments/aggregate.py")


if __name__ == "__main__":
    main(*(sys.argv[1:3] or []))
