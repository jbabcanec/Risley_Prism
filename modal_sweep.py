"""
modal_sweep.py — Modal wrapper for the Risley large-scale campaign.
Place at repo root. Run from repo root.

Quick start:
  modal run modal_sweep.py --test                          # smoke-test: 200 cases (~2 min)
  modal run modal_sweep.py                                 # E1: 1,000,000 cases
  modal run modal_sweep.py --exp fixedT --tag scaling      # E3: scaling laws
  modal run modal_sweep.py --exp mint   --tag mint         # E4: prescriptions
  modal run modal_sweep.py --exp noise  --tag noise        # E6: noise phase diagram
  modal run modal_sweep.py --exp speedsP --prisms 2 --tag fp2   # E5: P=2 frontier
  modal run modal_sweep.py --exp speedsP --prisms 4 --tag fp4   # E5: P=4 frontier
  modal run modal_sweep.py --exp speedsP --prisms 5 --tag fp5   # E5: P=5 frontier
  modal run modal_sweep.py --exp speedsP --prisms 6 --tag fp6   # E5: P=6 frontier

Download results and aggregate:
  modal volume get risley-results /results ./experiments/results
  python experiments/aggregate.py
  python experiments/aggregate.py --figs
"""

import modal

app = modal.App("risley-sweep")

# Bake the solver code directly into the image (current Modal API — no Mount needed)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.24", "scipy>=1.10")
    .add_local_dir("risley_lattice",     remote_path="/repo/risley_lattice")
    .add_local_dir("experiments",        remote_path="/repo/experiments")
    .add_local_dir("reverse_problem_v2", remote_path="/repo/reverse_problem_v2")
)

# Persistent volume — results survive between runs and modal sessions
volume = modal.Volume.from_name("risley-results", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/results": volume},
    timeout=1800,         # 30-min ceiling; adaptive ladder at T=80s can be slow
    cpu=1.0,
    retries=1,            # auto-retry once on transient Modal failure
)
def run_shard(params: dict) -> str:
    import sys, os, argparse
    sys.path.insert(0, "/repo")
    # Single-thread BLAS — multithreaded eig/svd is not bitwise reproducible
    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })

    import experiments.sweep as sw
    sw.OUTDIR = "/results"          # redirect all JSONL output to the volume

    args = argparse.Namespace(
        exp=params["exp"],
        start=params["start"],
        count=params["count"],
        tag=params["tag"],
        seed0=424242,
        T=params.get("T", [10.0, 20.0, 40.0, 80.0]),
        Tmax=params.get("Tmax", 320.0),
        bisect=params.get("bisect", 5),
        P=params.get("P", 3),
        snr=params.get("snr", [60, 50, 40, 30, 20]),
        bounds=params.get("bounds", False),
    )
    sw.run(args)
    volume.commit()
    return f"shard {params['start']}–{params['start'] + params['count'] - 1} OK"


@app.local_entrypoint()
def main(
    exp: str = "adaptive",
    tag: str = "atlas",
    total: int = 1_000_000,
    shard_size: int = 25,
    prisms: int = 3,
    bounds: bool = False,
    test: bool = False,
):
    """
    --test            run 2 shards (200 cases) to verify setup before committing
    --exp             adaptive | fixedT | mint | speedsP | noise
    --tag             label embedded in output filenames
    --total           total cases to run (default 1,000,000 for E1 atlas)
    --shard_size      cases per Modal container (default 100)
    --prisms          prism count for speedsP experiment (2–6)
    --bounds          certify each success (adds ~37 forward evals; needed for E2)
    """
    if test:
        total, shard_size = 50, 25

    # Default T-ladders per experiment type
    T_by_exp = {
        "adaptive": [10.0, 20.0, 40.0, 80.0],
        "fixedT":   [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0],
        "mint":     [10.0],
        "speedsP":  [10.0, 20.0, 40.0, 80.0],
        "noise":    [10.0, 20.0, 40.0, 80.0],
    }

    param_list = [
        {
            "exp": exp,
            "start": s,
            "count": shard_size,
            "tag": tag,
            "T": T_by_exp.get(exp, [10.0, 20.0, 40.0, 80.0]),
            "P": prisms,
            "bounds": bounds,
        }
        for s in range(0, total, shard_size)
    ]

    print(f"Launching {len(param_list)} shards × {shard_size} cases = {total} total  "
          f"[exp={exp}, tag={tag}]")

    results = list(run_shard.map(param_list, return_exceptions=True))

    ok = sum(1 for r in results if not isinstance(r, Exception))
    bad = [
        (param_list[i]["start"], r)
        for i, r in enumerate(results)
        if isinstance(r, Exception)
    ]

    print(f"\nDone: {ok}/{len(param_list)} shards succeeded")
    if bad:
        print(f"Failed shards (first 10):")
        for s, e in bad[:10]:
            print(f"  start={s}: {e}")
