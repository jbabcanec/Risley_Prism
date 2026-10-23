"""
modal_sweep_continuous.py -- fully cloud-side campaign runner.

ALWAYS use --detach:
  modal run --detach modal_sweep_continuous.py --tag atlas5

WHAT CHANGED AFTER THE atlas4 STALL (2026-09-07)
------------------------------------------------
atlas4 died of its own success: every finished shard wrote 2 files into one
flat directory, so /results grew past 52,000 entries. Volume commits reconcile
against that directory, so each commit got slower as the run progressed, until
commits began failing outright -- at which point a container would compute for
25 minutes and then throw all 50 cases away. Modal's own web UI 500s trying to
list that directory, which is the same wall from the outside.

Three fixes:

  1. FEWER, BIGGER FILES.  SUB_SIZE 25 -> 250. One file per 250 cases means
     ~4,000 files for a 1M campaign instead of 40,000, and 1,000 containers
     instead of 20,000 -- so 20x fewer commits in total.

  2. INCREMENTAL COMMITS.  A background thread commits every COMMIT_EVERY
     seconds while the workers run, instead of a single commit at the very
     end. A container that is preempted or times out now loses at most ~15
     minutes of work rather than its entire shard. This was the expensive
     failure mode in BOTH atlas3 and atlas4.

  3. PER-TAG OUTPUT DIRECTORY.  Results go to /results/<tag>/ instead of the
     shared flat root, so a new campaign never inherits an old one's file
     count. RESUME_DIRS still points at the old root, so every case already
     computed under a previous tag is skipped, not recomputed.

Re-running is always safe -- completed cases are skipped automatically.

After the campaign, pack and download in one file (52k individual downloads
is not viable):
  modal run modal_pack_results.py --tag atlas5
  modal volume get risley-results atlas5_results.tar.gz ./atlas5_results.tar.gz
"""

import modal

app = modal.App("risley-sweep-continuous")

worker_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.24", "scipy>=1.10")
    .add_local_dir("risley_lattice",     remote_path="/repo/risley_lattice")
    .add_local_dir("experiments",        remote_path="/repo/experiments")
    .add_local_dir("reverse_problem_v2", remote_path="/repo/reverse_problem_v2")
)

volume = modal.Volume.from_name("risley-results", create_if_missing=True)

N_WORKERS    = 2      # parallel workers per container == cpu count
SUB_SIZE     = 50     # cases per worker == cases per output file
COMMIT_EVERY = 900    # seconds between incremental volume commits

# CPU COUNT: deliberately 2, not 4.  This campaign has now tested three
# container sizes on Modal's spot capacity:
#   8 CPU -> heavy preemption (bigger containers are harder to place)
#   4 CPU -> atlas3, 110/112 calls failed
#   2 CPU -> atlas4, ran 24 h straight without a placement problem
# 2 CPU is the only shape with demonstrated survival, so it stays. Going to
# 4 would buy ~7% on the memory-to-CPU ratio and is not worth re-testing a
# configuration that already failed twice.
#
# SUB_SIZE: drives BOTH cases-per-file and worker lifetime, so it is the real
# safety knob -- and it MUST leave margin against SHARD_TIMEOUT.
#
# MEASURED COST PER CASE, corrected twice:
#   ~60 CPU-s   WRONG. Taken from atlas4 throughput without separating real
#               compute from fast resume-skips.
#   >108 CPU-s  ACTUAL. atlas5 shard 960400 ran the full 3 h timeout without
#               finishing 100 cases/worker (12 commit checkpoints, then
#               cancellation at exactly 10,800 s).
#
# At ~108 CPU-s/case, SUB_SIZE=100 needs 10,800 s against a 10,800 s timeout:
# a 1.0x margin, not the 1.8x claimed. That is why ~26% of atlas5's shards
# time out. SUB_SIZE=50 needs ~5,400 s -> genuine 2x margin.
#
#   100 -> 10,000 files/1M but 3.0 h per worker == the timeout   (atlas5 bug)
#    50 -> 20,000 files/1M and 1.5 h per worker, 2x margin   <-- chosen
#    25 -> 40,000 files/1M (the flat-directory count that killed atlas4,
#          though per-tag output makes this less dangerous than it was)
#
# Timeouts are survivable now -- incremental commits mean a killed shard keeps
# its work and the retry resumes -- but each one still burns a container
# startup and a partial re-run, so margin is worth having.
SHARD_TIMEOUT = 10800   # 3 hours: ~2x margin over the ~1.5 h expected

def _discover_resume_dirs():
    """Every place a previously-computed case might live: the flat root plus
    every tag directory modal_reorg_volume.py created under _archive/.

    Discovered rather than hardcoded -- an earlier version listed only
    atlas3/atlas4 and would have silently recomputed atlas2's ~44,000 cases.
    Including ALL tags is safe because _load_done() only counts a row as done
    if it carries x18, so pre-x18 rows (e.g. the original 'atlas' run) are
    correctly rejected and recomputed."""
    import os
    dirs = ["/results"]
    arch = "/results/_archive"
    if os.path.isdir(arch):
        try:
            dirs += sorted(
                os.path.join(arch, d) for d in os.listdir(arch)
                if os.path.isdir(os.path.join(arch, d)))
        except OSError:
            pass
    return dirs


def _run_one_subshard(sp: dict) -> int:
    """One worker: SUB_SIZE cases into its own resume-safe file."""
    import sys, os, argparse
    sys.path.insert(0, "/repo")
    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })
    import experiments.sweep as sw

    sw.OUTDIR = sp["outdir"]
    sw.RESUME_DIRS = sp["resume_dirs"]
    os.makedirs(sw.OUTDIR, exist_ok=True)

    args = argparse.Namespace(
        exp=sp["exp"], start=sp["start"], count=sp["count"], tag=sp["tag"],
        seed0=sp["seed0"], T=sp["T"], Tmax=sp["Tmax"], bisect=sp["bisect"],
        P=sp["P"], snr=sp["snr"], bounds=sp["bounds"],
    )
    sw.run(args)
    return sp["start"]


@app.function(
    image=worker_image,
    volumes={"/results": volume},
    timeout=SHARD_TIMEOUT,
    cpu=float(N_WORKERS),
    memory=2048,
    retries=2,
)
def run_shard(params: dict) -> str:
    import os, threading
    from concurrent.futures import ThreadPoolExecutor

    base    = params["start"]
    total   = params["count"]
    tag     = params["tag"]
    outdir  = f"/results/{tag}"
    os.makedirs(outdir, exist_ok=True)

    resume_dirs = params.get("resume_dirs") or _discover_resume_dirs()
    print(f"[resume] scanning {len(resume_dirs)} dirs: "
          f"{', '.join(resume_dirs)}", flush=True)

    sub_params, remaining, s = [], total, base
    while remaining > 0:
        c = min(SUB_SIZE, remaining)
        sub_params.append({
            "exp":    params["exp"],
            "start":  s,
            "count":  c,
            "tag":    tag,
            "outdir": outdir,
            # old flat layout: skip cases already computed under any prior tag
            "resume_dirs": resume_dirs,
            "seed0":  424242,
            "T":      params.get("T",      [10.0, 20.0, 40.0, 80.0]),
            "Tmax":   params.get("Tmax",   320.0),
            "bisect": params.get("bisect", 5),
            "P":      params.get("P",      3),
            "snr":    params.get("snr",    [60, 50, 40, 30, 20]),
            "bounds": params.get("bounds", False),
        })
        s         += c
        remaining -= c

    # --- incremental commits: preemption/timeout must not discard the shard ---
    stop = threading.Event()

    def committer():
        n = 0
        while not stop.wait(COMMIT_EVERY):
            try:
                volume.commit()
                n += 1
                print(f"[commit] checkpoint {n} for shard {base}", flush=True)
            except Exception as e:
                print(f"[commit] FAILED: {type(e).__name__}: {e}", flush=True)

    t = threading.Thread(target=committer, daemon=True)
    t.start()

    try:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            list(ex.map(_run_one_subshard, sub_params))
    finally:
        stop.set()
        t.join(timeout=30)
        volume.commit()          # final flush, even if a worker raised

    return f"shard {base}-{base + total - 1} OK"


@app.local_entrypoint()
def main(
    exp:    str  = "adaptive",
    tag:    str  = "atlas5",
    total:  int  = 1_000_000,
    prisms: int  = 3,
    bounds: bool = False,
):
    shard_size = N_WORKERS * SUB_SIZE          # 1,000 cases per container

    T_by_exp = {
        "adaptive": [10.0, 20.0, 40.0, 80.0],
        "fixedT":   [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0],
        "mint":     [10.0],
        "speedsP":  [10.0, 20.0, 40.0, 80.0],
        "noise":    [10.0, 20.0, 40.0, 80.0],
    }

    param_list = [
        {
            "exp":    exp,
            "start":  s,
            "count":  min(shard_size, total - s),
            "tag":    tag,
            "T":      T_by_exp.get(exp, [10.0, 20.0, 40.0, 80.0]),
            "P":      prisms,
            "bounds": bounds,
        }
        for s in range(0, total, shard_size)
    ]

    n = len(param_list)
    print(f"exp={exp}  tag={tag}  total={total:,}")
    print(f"  {n:,} shards x {shard_size} cases "
          f"({N_WORKERS} workers x {SUB_SIZE})")
    print(f"  -> {total // SUB_SIZE:,} output files in /results/{tag}/")
    print(f"  -> commit checkpoint every {COMMIT_EVERY // 60} min; "
          f"shard timeout {SHARD_TIMEOUT // 3600} h")
    print(f"  -> already-computed cases (any prior tag) are skipped\n")

    handles = [run_shard.spawn(p) for p in param_list]
    print(f"{len(handles):,} shards submitted. You can close your terminal.")
    print("  Monitor: https://modal.com/apps/riley-prisms")
