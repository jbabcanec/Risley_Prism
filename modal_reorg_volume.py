"""
Move the old flat result files out of /results into /results/_archive/<tag>/.

WHY: /results holds 52,000+ files. Volume commits reconcile against that
directory, which is what made atlas4 slow down and then start failing; Modal's
own file browser 500s on it too. Nothing is deleted -- files are moved, and
the new runner still resumes from _archive/, so no case is ever recomputed.

Run this ONCE, before launching the next campaign:

  modal run modal_reorg_volume.py                 # dry run, shows the plan
  modal run modal_reorg_volume.py --apply         # actually move

Safe to re-run: already-archived files are left alone.
"""
import modal

app = modal.App("risley-reorg-volume")
image = modal.Image.debian_slim(python_version="3.11")
volume = modal.Volume.from_name("risley-results", create_if_missing=False)


def _tag_of(name: str):
    """sweep_<exp>_<tag>_<start>.jsonl -> tag  (None if not a result file)."""
    if not (name.startswith("sweep_") and name.endswith(".jsonl")):
        return None
    parts = name[len("sweep_"):-len(".jsonl")].split("_")
    if len(parts) < 3:
        return None
    return "_".join(parts[1:-1]) or None


@app.function(image=image, volumes={"/results": volume},
              timeout=7200, cpu=2.0, memory=4096)
def reorg(apply: bool = False) -> str:
    import os, collections

    root = "/results"
    names = [n for n in os.listdir(root) if os.path.isfile(os.path.join(root, n))]
    buckets = collections.defaultdict(list)
    for n in names:
        t = _tag_of(n)
        if t:
            buckets[t].append(n)

    plan = ", ".join(f"{t}:{len(v):,}" for t, v in sorted(buckets.items()))
    total = sum(len(v) for v in buckets.values())
    head = (f"/results holds {len(names):,} files; "
            f"{total:,} are result files.\n  by tag -> {plan}\n")

    if not apply:
        return head + "\nDRY RUN -- nothing moved. Re-run with --apply"

    moved = 0
    for tag, files in sorted(buckets.items()):
        dest = os.path.join(root, "_archive", tag)
        os.makedirs(dest, exist_ok=True)
        for i, n in enumerate(files):
            src, dst = os.path.join(root, n), os.path.join(dest, n)
            if os.path.exists(dst):
                continue
            try:
                os.replace(src, dst)
                moved += 1
            except OSError as e:
                print(f"  skip {n}: {e}", flush=True)
            if moved and moved % 2000 == 0:
                print(f"  moved {moved:,}/{total:,} -- checkpointing", flush=True)
                volume.commit()
        print(f"  {tag}: done ({len(files):,} files)", flush=True)

    volume.commit()
    left = len([n for n in os.listdir(root)
                if os.path.isfile(os.path.join(root, n))])
    return head + f"\nmoved {moved:,} files; {left:,} loose files remain in /results"


@app.local_entrypoint()
def main(apply: bool = False):
    print(reorg.remote(apply))
