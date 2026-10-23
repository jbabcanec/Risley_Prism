"""
Count UNIQUE cases banked so far, across every tag, without downloading.

Answers directly: how many unique cases do we actually have, how much overlap
is there between atlas4 and atlas5, and how many more shards to reach a target.

  modal run modal_count_cases.py
  modal run modal_count_cases.py --target 500000

Cheap (~2 CPU-min) and read-only. Safe to run while a campaign is going.
"""
import modal

app = modal.App("risley-count-cases")
image = modal.Image.debian_slim(python_version="3.11")
volume = modal.Volume.from_name("risley-results", create_if_missing=False)

PRE_X18_TAGS = {"atlas"}


_re_case = None
_re_snr = None


def _read_one(arg):
    """Read one JSONL and return (tag, rows, nox18, {(case,snr)}).

    Runs in a thread pool: Volume reads are network I/O and release the GIL,
    so concurrency is the whole game here. Reading ~20k files serially took
    tens of minutes; 32-way it is a couple of ninutes.
    """
    import re
    global _re_case, _re_snr
    if _re_case is None:
        _re_case = re.compile(rb'"case":\s*(\d+)')
        _re_snr  = re.compile(rb'"snr":\s*(null|-?[\d.]+)')
    path, tag = arg
    keys = set()
    n = nox = 0
    try:
        fh = open(path, "rb")
    except OSError:
        return tag, 0, 0, keys
    with fh as f:
        for line in f:
            n += 1
            if b'"x18": [' not in line:
                nox += 1
                continue
            mc = _re_case.search(line)
            if not mc:
                continue
            ms = _re_snr.search(line)
            keys.add((int(mc.group(1)), ms.group(1) if ms else b"null"))
    return tag, n, nox, keys


@app.function(image=image, volumes={"/results": volume},
              timeout=3600, cpu=8.0, memory=16384)
def count(target: int = 500000, exp_filter: str = "adaptive",
          skip: str = "atlas", threads: int = 32) -> str:
    import os, collections
    from concurrent.futures import ThreadPoolExecutor

    skip_tags = {t.strip() for t in skip.split(",") if t.strip()}

    work = []
    files = collections.Counter()
    skipped = collections.Counter()
    for dirpath, _d, filenames in os.walk("/results"):
        for nm in filenames:
            if not (nm.startswith("sweep_") and nm.endswith(".jsonl")):
                continue
            parts = nm[6:-6].split("_")
            if len(parts) < 3:
                continue
            exp, tag = parts[0], "_".join(parts[1:-1])
            if exp != exp_filter:
                continue
            if tag in skip_tags:
                skipped[tag] += 1        # e.g. 'atlas': 35k files, no x18
                continue
            files[tag] += 1
            work.append((os.path.join(dirpath, nm), tag))

    if skipped:
        print("skipping (no x18, would dominate I/O): "
              + ", ".join(f"{t}:{n:,}" for t, n in skipped.items()), flush=True)
    print(f"reading {len(work):,} files with {threads} threads...", flush=True)

    keys_by_tag = collections.defaultdict(set)
    rows  = collections.Counter()
    nox18 = collections.Counter()
    done = 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for tag, n, nox, keys in ex.map(_read_one, work):
            rows[tag] += n
            nox18[tag] += nox
            keys_by_tag[tag] |= keys
            done += 1
            if done % 2000 == 0:
                print(f"  {done:,}/{len(work):,} files", flush=True)

    out = [f"\nexperiment: {exp_filter}",
           f"{'tag':<10}{'files':>9}{'rows':>12}{'no-x18':>10}{'unique':>12}"]
    for tag in sorted(files):
        out.append(f"{tag:<10}{files[tag]:>9,}{rows[tag]:>12,}"
                   f"{nox18[tag]:>10,}{len(keys_by_tag[tag]):>12,}")

    usable = {t: k for t, k in keys_by_tag.items() if t not in PRE_X18_TAGS}
    union = set().union(*usable.values()) if usable else set()
    out.append(f"\nUNIQUE cases with x18 (all tags except {PRE_X18_TAGS}): "
               f"{len(union):,}")

    # pairwise overlap -- how much did atlas5 re-tread vs atlas4?
    tags = sorted(usable)
    if len(tags) > 1:
        out.append("\noverlap between tags (shared cases):")
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                n = len(usable[tags[i]] & usable[tags[j]])
                if n:
                    out.append(f"  {tags[i]} & {tags[j]}: {n:,}")

    idx = sorted(c for c, _s in union)
    if idx:
        below = sum(1 for c in idx if c < target)
        out.append(f"\ncoverage:")
        out.append(f"  highest case index reached : {idx[-1]:,}")
        out.append(f"  unique cases below {target:,} : {below:,}")
        out.append(f"  contiguous from 0 up to    : "
                   f"{_frontier(set(c for c, _ in union)):,}")
        need = max(0, target - len(union))
        out.append(f"\nto reach {target:,} unique: {need:,} more cases "
                   f"= {(need + 199) // 200:,} shards")
        if need:
            hrs = (need + 199) // 200 / 58.0
            out.append(f"  approx {hrs:.0f} h and ${hrs * 15.7:,.0f} "
                       f"at atlas5's observed rate")
    return "\n".join(out)


def _frontier(cases):
    """Largest N such that every index 0..N-1 is present."""
    n = 0
    while n in cases:
        n += 1
    return n


@app.local_entrypoint()
def main(target: int = 500000, exp: str = "adaptive",
         skip: str = "atlas", threads: int = 32):
    print(count.remote(target, exp, skip, threads))
