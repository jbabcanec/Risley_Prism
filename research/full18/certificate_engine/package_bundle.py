"""Package the certificate engine plus its verified local documentation closure.

No research model or observation generator is executed by this packaging step.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFEST = HERE / "MANIFEST.json"
TARGET = ROOT / "risley_certificate_engine_v1.zip"
ARCHIVE = ROOT / "risley_certificate_engine_v1_archive.zip"
INITIAL_HASH = "4971e782e783dd50efa75452b19a6eab4cd2304c1b44724fdc6f16cb2f954b3d"
REPORT_HASH = "4b035cf63e77a4659503247b3e421bf763218a0df57dcd1f2f4065ea94a6e719"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(ROOT/"REPORT.md") == REPORT_HASH
    audit = json.loads((HERE/"adaptive_independent_review.json").read_text())
    assert audit["status"] == "passed"
    for name, expected in audit["source_sha256"].items():
        assert sha(HERE/name) == expected
    assert sha(HERE/"verify_adaptive.py") == audit["independent_checker_sha256"]
    assert sha(HERE/"spotcheck"/"synthetic_fullvector_record.json") == audit["fixture_sha256"]
    if not ARCHIVE.exists():
        assert sha(TARGET) == INITIAL_HASH
        shutil.copyfile(TARGET, ARCHIVE)
    assert sha(ARCHIVE) == INITIAL_HASH
    old = json.loads(MANIFEST.read_text())
    selected = {(ROOT/item["path"]).resolve() for item in old["files"]}
    selected.update(p.resolve() for p in HERE.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts and p != MANIFEST)
    selected.discard(MANIFEST.resolve())
    queue = list(selected)
    while queue:
        path = queue.pop()
        assert path.is_relative_to(ROOT) and path.is_file(), path
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        # Bracket multiplication in TeX and code is not a Markdown file link.
        text = re.sub(r"```.*?```|~~~.*?~~~", "", text, flags=re.S)
        text = re.sub(r"\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$", "", text, flags=re.S)
        text = re.sub(r"`[^`\n]*`|(?<!\\)\$[^\n$]*\$", "", text)
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            link = link.strip().strip("<>").split("#", 1)[0]
            if not link or "://" in link or link.startswith(("mailto:","sandbox:")):
                continue
            dependency = (path.parent/link).resolve()
            assert dependency.is_relative_to(ROOT) and dependency.is_file(), (path,link)
            if dependency not in selected and dependency != MANIFEST.resolve():
                selected.add(dependency)
                queue.append(dependency)
    files = [{"path":p.relative_to(ROOT).as_posix(), "bytes":p.stat().st_size,"sha256":sha(p)}
             for p in sorted(selected,key=lambda p:p.relative_to(ROOT).as_posix())]
    manifest = {"artifact":"bounded full-vector certificate engine, implementation revision2",
                "status":"independently reviewed and checked", "dataset_count":1,
                "dataset_kind":"same saved synthetic fullvector proof spot-check",
                "scope":"shared affine LP, sparse exact exclusions and coverage-preserving bounded optical subdivision; no complete inverse",
                "published_report_version":9,"published_report_sha256":REPORT_HASH,
                "file_count":len(files),"files":files}
    MANIFEST.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    with zipfile.ZipFile(TARGET,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as bundle:
        for item in files:
            bundle.write(ROOT/item["path"], item["path"])
        bundle.write(MANIFEST,"certificate_engine/MANIFEST.json")
    with zipfile.ZipFile(TARGET) as bundle:
        assert len(bundle.namelist()) == len(files)+1
        assert len(set(bundle.namelist())) == len(files)+1
        for item in files:
            data = bundle.read(item["path"])
            assert len(data) == item["bytes"]
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
        assert bundle.read("certificate_engine/MANIFEST.json") == MANIFEST.read_bytes()
    assert sha(ROOT/"REPORT.md") == REPORT_HASH
    print(json.dumps({"zip":str(TARGET),"bytes":TARGET.stat().st_size,"sha256":sha(TARGET),
                      "verified_entries":len(files)+1,"prior_zip_sha256":sha(ARCHIVE),
                      "report_unchanged":True},indent=2))


if __name__ == "__main__":
    main()
