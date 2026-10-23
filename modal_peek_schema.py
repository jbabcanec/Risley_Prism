"""
Show which fields ACTUALLY exist in each tag's rows. Reads one row per tag,
so it returns in seconds.

  modal run modal_peek_schema.py

Answers, per tag, without guessing:
  x18          the recovered 18-parameter vector
  min_cycles   min|N_i| * T                      (added Sep 6)
  T_req        certified observation-time floor  (added Sep 6, failures only)
  sigma_N      Fisher uncertainty on generators  (added Sep 6, failures only)
  min_amp      weakest prism amplitude           (added Sep 6, failures only)

NOTE: T_req/sigma_N/min_amp are only written on FAILED rows, so their absence
in a sampled row is not proof the tag lacks them -- this samples several rows
per tag and reports whether any row carries them.
"""
import modal

app = modal.App("risley-peek-schema")
image = modal.Image.debian_slim(python_version="3.11")
volume = modal.Volume.from_name("risley-results", create_if_missing=False)

FIELDS = ["x18", "min_cycles", "T_req", "sigma_N", "min_amp"]


@app.function(image=image, volumes={"/results": volume},
              timeout=1800, cpu=2.0, memory=4096)
def peek(rows_per_tag: int = 400) -> str:
    import os, json, collections

    per_tag = collections.defaultdict(list)      # tag -> file paths
    for dirpath, _d, filenames in os.walk("/results"):
        for nm in filenames:
            if nm.startswith("sweep_") and nm.endswith(".jsonl"):
                parts = nm[6:-6].split("_")
                if len(parts) >= 3:
                    per_tag["_".join(parts[1:-1])].append(
                        os.path.join(dirpath, nm))

    out = []
    out.append("COUNTS ARE OF TRAIL ENTRIES (T-rungs), NOT CASES.")
    out.append("Each case attempts 1+ T values, so a field's count exceeds the")
    out.append("case count. T_req/sigma_N/min_amp appear on FAILED rungs only.")
    out.append("")
    hdr = (f"{'tag':<10}{'cases':>7}{'solved':>8}{'rungs':>7}{'r/case':>8}"
           + "".join(f"{f:>12}" for f in FIELDS))
    out.append(hdr)
    out.append("-" * len(hdr))

    verdict = {}
    for tag in sorted(per_tag):
        seen = {f: 0 for f in FIELDS}
        n = solved = failed = rungs = 0
        rec_keys, trail_keys = set(), set()
        for p in sorted(per_tag[tag])[:40]:
            if n >= rows_per_tag:
                break
            try:
                fh = open(p)
            except OSError:
                continue
            with fh as f:
                for line in f:
                    if n >= rows_per_tag:
                        break
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    n += 1
                    rec_keys |= set(r.keys())
                    tr = r.get("trail") or []
                    if r.get("T_solved") is not None:
                        solved += 1
                    else:
                        failed += 1
                    for row in tr:
                        rungs += 1
                        trail_keys |= set(row.keys())
                        for fld in FIELDS:
                            if fld in row:
                                seen[fld] += 1
        if not n:
            continue
        out.append(f"{tag:<10}{n:>7}{solved:>8}{rungs:>7}"
                   f"{rungs/max(n,1):>8.2f}" + "".join(
            f"{(str(seen[f]) if seen[f] else '--'):>12}" for f in FIELDS))
        verdict[tag] = (seen, n, failed, sorted(rec_keys), sorted(trail_keys))

    out.append("\nper-tag detail (fields present in ANY sampled row):")
    for tag, (seen, n, failed, rk, tk) in verdict.items():
        has = [f for f in FIELDS if seen[f]]
        miss = [f for f in FIELDS if not seen[f]]
        out.append(f"\n  {tag}  ({n} rows sampled, {failed} failed)")
        out.append(f"    has     : {', '.join(has) or 'NONE'}")
        out.append(f"    missing : {', '.join(miss) or 'none'}")
        if failed == 0 and {"T_req", "sigma_N", "min_amp"} & set(miss):
            out.append(f"    (no failed rows sampled -- T_req/sigma_N/min_amp "
                       f"are only written on failures, so 'missing' here is "
                       f"inconclusive for those three)")
        out.append(f"    trail[0] keys: {', '.join(tk)}")

    out.append("\nRESUME IMPACT: sweep.py counts a case as done only if its "
               "row has x18.")
    for tag, (seen, n, failed, rk, tk) in verdict.items():
        state = ("SKIPPED by resume" if seen["x18"]
                 else "RECOMPUTED by resume (no x18)")
        out.append(f"  {tag:<10} -> {state}")
    return "\n".join(out)


@app.local_entrypoint()
def main(rows_per_tag: int = 400):
    print(peek.remote(rows_per_tag))
