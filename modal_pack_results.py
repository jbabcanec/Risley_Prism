"""
Pack results into ONE tarball inside Modal, so you download 1 file instead of
50,000+.

Results now live in several directories: /results/<tag>/ for current runs and
/results/_archive/<tag>/ for anything modal_reorg_volume.py tidied away. This
walks all of them, so a campaign split across tags comes back as one dataset.

  modal run modal_pack_results.py                      # every x18-bearing tag
  modal run modal_pack_results.py --tags atlas4,atlas5
  modal run modal_pack_results.py --tags all           # includes pre-x18 atlas

  modal volume get risley-results risley_results.tar.gz ./risley_results.tar.gz
  tar xzf risley_results.tar.gz
  python3 cost_check.py results/*/*.jsonl
"""
import modal

app = modal.App("risley-pack-results")
image = modal.Image.debian_slim(python_version="3.11")
volume = modal.Volume.from_name("risley-results", create_if_missing=False)

# the original 'atlas' run predates x18, so it is excluded unless --tags all
PRE_X18_TAGS = {"atlas"}


def _parse(name):
    """sweep_<exp>_<tag>_<start>.jsonl -> (exp, tag) or None."""
    if not (name.startswith("sweep_") and name.endswith(".jsonl")):
        return None
    parts = name[len("sweep_"):-len(".jsonl")].split("_")
    if len(parts) < 3:
        return None
    return parts[0], "_".join(parts[1:-1])


@app.function(image=image, volumes={"/results": volume},
              timeout=7200, cpu=2.0, memory=4096)
def pack(tags: str = "") -> str:
    import os, tarfile, collections

    want = {t.strip() for t in tags.split(",") if t.strip()}
    take_all = "all" in want

    found = collections.defaultdict(list)          # tag -> [(abs, arcname)]
    for dirpath, _dirnames, filenames in os.walk("/results"):
        for nm in filenames:
            pt = _parse(nm)
            if not pt:
                continue
            _exp, tag = pt
            if want and not take_all and tag not in want:
                continue
            if not want and tag in PRE_X18_TAGS:
                continue                            # default: skip pre-x18
            found[tag].append((os.path.join(dirpath, nm),
                               f"results/{tag}/{nm}"))

    if not found:
        return f"no files matched tags={tags or '(default: all x18 tags)'}"

    out = "/results/risley_results.tar.gz"
    if os.path.exists(out):
        os.remove(out)

    rows = with_x18 = 0
    n = 0
    total = sum(len(v) for v in found.values())
    with tarfile.open(out, "w:gz") as tf:
        for tag in sorted(found):
            for src, arc in found[tag]:
                tf.add(src, arcname=arc)
                n += 1
                if n % 5000 == 0:
                    print(f"  packed {n:,}/{total:,}", flush=True)
                try:
                    with open(src) as fh:
                        for line in fh:
                            rows += 1
                            if '"x18"' in line:
                                with_x18 += 1
                except Exception:
                    pass
            print(f"  {tag}: {len(found[tag]):,} files", flush=True)

    volume.commit()
    mb = os.path.getsize(out) / 1e6
    return (f"\n{total:,} files from {len(found)} tag(s)\n"
            f"{rows:,} case-records, {with_x18:,} carry x18\n"
            f"-> risley_results.tar.gz ({mb:.1f} MB)")


@app.local_entrypoint()
def main(tags: str = ""):
    print(pack.remote(tags))
    print("\nNow run:\n  modal volume get risley-results "
          "risley_results.tar.gz ./risley_results.tar.gz")
