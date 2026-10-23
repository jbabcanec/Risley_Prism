#!/usr/bin/env python3
"""Merge the campaign's tags into one deduplicated set that aggregate.py reads.

WHY THIS EXISTS
  Results are spread over several tags (atlas2..atlas5) in nested directories,
  and aggregate.py does a FLAT glob of experiments/results/sweep_*.jsonl with
  no dedup. Two consequences it cannot handle alone:

    * nesting -- the tarball unpacks to results/<tag>/, which its glob misses.
    * duplicates -- atlas4 ran before the resume feature existed, so it may
      have recomputed cases atlas2 already had. load() appends every row, so
      a duplicated case would be counted twice and skew the recovery rates.

SCHEMA PRIORITY (best first)
  atlas5, atlas4, atlas3   x18 + min_cycles + T_req + sigma_N + min_amp
  atlas2                   x18 only  (predates the diagnostic fields)
  atlas                    no x18    -- REJECTED, this is why you re-ran

  Files are read in that order and the first row seen for a given
  (exp, case, snr) wins, so a case present in both atlas4 and atlas2 keeps the
  richer atlas4 row. Memory stays modest: only the key set is held, not rows.

USAGE
  tar xzf risley_results.tar.gz
  python3 merge_results.py
  python experiments/aggregate.py
  python experiments/aggregate.py --figs
"""
import argparse, glob, json, os, shutil, sys
from collections import Counter

TAG_PRIORITY = ["atlas5", "atlas4", "atlas3", "atlas2"]


def tag_of(path):
    nm = os.path.basename(path)
    if not (nm.startswith("sweep_") and nm.endswith(".jsonl")):
        return None
    parts = nm[len("sweep_"):-len(".jsonl")].split("_")
    return "_".join(parts[1:-1]) if len(parts) >= 3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results",
                    help="where the tarball was unpacked (default: results)")
    ap.add_argument("--dst", default="experiments/results",
                    help="where aggregate.py looks (default: experiments/results)")
    ap.add_argument("--tags", default=",".join(TAG_PRIORITY),
                    help="comma-separated, best schema first")
    a = ap.parse_args()

    order = [t.strip() for t in a.tags.split(",") if t.strip()]
    rank = {t: i for i, t in enumerate(order)}

    files = [p for p in glob.glob(os.path.join(a.src, "**", "sweep_*.jsonl"),
                                  recursive=True) if tag_of(p) in rank]
    if not files:
        sys.exit(f"no sweep_*.jsonl under {a.src!r} for tags {order}. "
                 f"Did you unpack the tarball?")
    files.sort(key=lambda p: (rank[tag_of(p)], p))

    os.makedirs(a.dst, exist_ok=True)
    stale = [p for p in glob.glob(os.path.join(a.dst, "sweep_*.jsonl"))]
    if stale:
        hold = os.path.join(a.dst, "_stale")
        os.makedirs(hold, exist_ok=True)
        for p in stale:
            shutil.move(p, os.path.join(hold, os.path.basename(p)))
        print(f"moved {len(stale)} pre-existing sweep_*.jsonl -> {hold}/ "
              f"(they would otherwise be double-counted)")

    seen, out, kept = set(), {}, Counter()
    dup = Counter(); nox18 = Counter(); partial = Counter()

    for p in files:
        tag = tag_of(p)
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    key = (r["exp"], r["case"], r.get("snr"))
                except Exception:
                    continue
                trail = r.get("trail") or []
                if trail and "x18" not in trail[0]:
                    nox18[tag] += 1
                    continue
                if key in seen:
                    dup[tag] += 1
                    continue
                seen.add(key)
                kept[tag] += 1
                if trail and "min_cycles" not in trail[0]:
                    partial[tag] += 1
                exp = r["exp"]
                if exp not in out:
                    out[exp] = open(
                        os.path.join(a.dst, f"sweep_{exp}_merged_0000000.jsonl"),
                        "w")
                out[exp].write(line if line.endswith("\n") else line + "\n")

    for f in out.values():
        f.close()

    print(f"\nread {len(files):,} files")
    print(f"{'tag':<10}{'kept':>12}{'duplicate':>12}{'no-x18':>10}"
          f"{'x18-only':>10}")
    for t in order:
        if kept[t] or dup[t] or nox18[t]:
            print(f"{t:<10}{kept[t]:>12,}{dup[t]:>12,}{nox18[t]:>10,}"
                  f"{partial[t]:>10,}")
    print(f"{'TOTAL':<10}{sum(kept.values()):>12,}{sum(dup.values()):>12,}"
          f"{sum(nox18.values()):>10,}{sum(partial.values()):>10,}")
    print(f"\n-> {len(out)} merged file(s) in {a.dst}/")
    for exp in out:
        print(f"     sweep_{exp}_merged_0000000.jsonl")
    if sum(partial.values()):
        print(f"\nNOTE: {sum(partial.values()):,} cases carry x18 but lack "
              f"min_cycles/T_req/sigma_N/min_amp\n      (atlas2 predates those "
              f"fields). min_cycles is recomputable from the case index;\n"
              f"      T_req/sigma_N/min_amp are not -- they need the solve redone.")
    print("\nNow run:  python experiments/aggregate.py")


if __name__ == "__main__":
    main()
