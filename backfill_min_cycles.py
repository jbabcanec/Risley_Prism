#!/usr/bin/env python3
"""Add min_cycles to any row missing it, so the field is universal.

WHY: rows from tags predating the diagnostic fields (atlas2) carry x18 but no
min_cycles. Unlike T_req/sigma_N/min_amp -- which exist only on FAILED rungs
and are genuinely absent from most successful cases by design -- min_cycles is
written on every row by current sweep.py, and is purely derived:

    min_cycles = min(|N_1|,|N_2|,|N_3|) * T

The speeds come from case_at(case, seed0), so this needs no re-solving. After
this runs, min_cycles is present on every rung in the dataset and the only
conditional fields are the three that are conditional by construction.

  python3 backfill_min_cycles.py experiments/results/sweep_adaptive_merged_0000000.jsonl
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from risley_lattice import case_at


def main(path):
    tmp = path + ".tmp"
    speeds = {}                      # (case, seed0) -> min|N_i|
    n = patched = rungs = skipped = 0

    with open(path) as fin, open(tmp, "w") as fout:
        for line in fin:
            try:
                r = json.loads(line)
            except Exception:
                fout.write(line)     # pass through anything unparseable
                skipped += 1
                continue
            n += 1
            need = [row for row in (r.get("trail") or [])
                    if "min_cycles" not in row and row.get("T") is not None]
            if need:
                key = (r["case"], r.get("seed0", 424242))
                if key not in speeds:
                    tc = case_at(key[0], key[1])
                    speeds[key] = float(np.min(np.abs(tc[:3])))
                s = speeds[key]
                for row in need:
                    row["min_cycles"] = s * float(row["T"])
                    rungs += 1
                patched += 1
            fout.write(json.dumps(r) + "\n")

    os.replace(tmp, path)
    print(f"rows read      : {n:,}")
    print(f"cases patched  : {patched:,}  ({rungs:,} rungs)")
    if skipped:
        print(f"unparseable    : {skipped:,} (passed through unchanged)")
    print(f"\n{path} updated in place")
    if patched == 0:
        print("nothing needed patching -- min_cycles already universal")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "experiments/results/sweep_adaptive_merged_0000000.jsonl")
