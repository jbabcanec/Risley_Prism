#!/usr/bin/env python3
"""Measure CPU-seconds per case from sweep JSONL files.

Usage:
  modal volume get risley-results sweep_adaptive_atlas4_0000000.jsonl ./a4.jsonl
  python3 cost_check.py a4.jsonl [more.jsonl ...]

Prints sec/case split by solved-vs-failed so you can see exactly where the
compute goes, and projects the cost of the remaining campaign.
"""
import json, sys

RATE_PER_CPU_SEC = 0.0575 / 3600.0   # ~$0.0575 per CPU-hour (cpu+mem observed)

def main(paths):
    solved_s, failed_s = [], []
    calls = 0
    for p in paths:
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = r.get('trail', [])
            s = sum(x.get('secs', 0) for x in t)
            calls += len(t)
            (solved_s if r.get('T_solved') is not None else failed_s).append(s)

    n = len(solved_s) + len(failed_s)
    if not n:
        print("no rows found"); return
    tot = sum(solved_s) + sum(failed_s)

    def stat(v, label):
        if not v:
            print(f"  {label:<8} 0 cases"); return
        v = sorted(v)
        print(f"  {label:<8} {len(v):>6} cases  "
              f"mean {sum(v)/len(v):>7.1f}s  med {v[len(v)//2]:>7.1f}s  "
              f"p90 {v[int(len(v)*.9)]:>7.1f}s  total {sum(v):>9.0f}s")

    print(f"\nfile(s): {', '.join(paths)}")
    print(f"cases {n}   solve18 calls {calls} ({calls/n:.2f}/case)   "
          f"failure rate {100*len(failed_s)/n:.1f}%")
    stat(solved_s, "solved")
    stat(failed_s, "FAILED")
    print(f"\n  overall  {tot/n:.1f} CPU-sec/case")
    if failed_s and solved_s:
        share = 100 * sum(failed_s) / tot
        print(f"  failures are {len(failed_s)/n*100:.0f}% of cases "
              f"but {share:.0f}% of the compute")
    per_case = tot / n * RATE_PER_CPU_SEC
    print(f"\n  cost  ${per_case*1000:.2f} per 1,000 cases"
          f"   ->  ${per_case*1e6:,.0f} for 1,000,000")

if __name__ == '__main__':
    main(sys.argv[1:] or ['sample.jsonl'])
