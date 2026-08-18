from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

from v23_download_frozen_images import main as hydrate_legacy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/v4_1_yolo"
REPORTS = ROOT / "reports"
SPLIT_FREEZE = ROOT / "data/v4_0_split_freeze.json"
ANNOT_FREEZE = ROOT / "data/v4_0_annotation_freeze.json"

NEW = {
    "POS_6610209": ("train", "https://s0.geograph.org.uk/geophotos/06/61/02/6610209_b56889e7.jpg", "091a4fccce53d7f35d2f2aaffadb32d0849f24e16f5d481e8376806300bb235d"),
    "POS_5952661": ("train", "https://s0.geograph.org.uk/geophotos/05/95/26/5952661_bd4ace05.jpg", "53a396cb1b42a510d5529c794acd31275a10f50d5bd05a321941bfab151edbc4"),
    "POS_7480474": ("train", "https://s0.geograph.org.uk/geophotos/07/48/04/7480474_1541bbba.jpg", "99b62979b21c9fe0d190a1eecce27b3290a1bc6e36bc88c460de0c54a9be52d9"),
    "POS_1352733": ("train", "https://s0.geograph.org.uk/geophotos/01/35/27/1352733_5d58f3d3.jpg", "39e7b071e6ec49d60bccb63eab300af768f3639c19863cd72fda755a7a58c549"),
    "POS_354803": ("val", "https://s0.geograph.org.uk/photos/35/48/354803_ffaaee49.jpg", "165648a66f22712dd34f36ed5ea8ee5916ac083f737fc1ec2f3af2f9a8f3442b"),
    "POS_543992": ("val", "https://s0.geograph.org.uk/photos/54/39/543992_e2a6dc83.jpg", "81ed550af775a8c1108f3528770463d0fa6ac187b9087922f58a225822940f8e"),
}
FINAL_IDS = {"POS_8091164", "POS_8239540"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_exact(url: str, expected: str, out: Path) -> None:
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"GridSight-UK-v4.1/1.0 (+https://github.com/xm2325/gridsight_uk)","Accept":"image/jpeg,image/*;q=0.8,*/*;q=0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            got = hashlib.sha256(data).hexdigest()
            if got != expected:
                raise RuntimeError(f"SHA mismatch: expected {expected}, got {got}")
            out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(data); return
        except Exception as exc:
            last = exc
            if attempt < 4: time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(f"Failed {url}: {last!r}")


def insulator_only(src: Path, dst: Path) -> int:
    rows = []
    for line in src.read_text().splitlines():
        if not line.strip(): continue
        parts = line.split()
        if int(float(parts[0])) == 2:
            rows.append("0 " + " ".join(parts[1:]))
    if not rows:
        raise RuntimeError(f"No insulator labels in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True); dst.write_text("\n".join(rows) + "\n")
    return len(rows)


def main():
    split = json.loads(SPLIT_FREEZE.read_text()); annot = json.loads(ANNOT_FREEZE.read_text())
    assert split["frozen_before_v4_model_inference"] is True
    assert annot["created_before_v4_model_inference"] is True
    assert {x["record_id"] for x in split["final_holdout_sources"]} == FINAL_IDS

    # Hydrate only legacy closed-cycle development data. Old final holdout is not touched by this function.
    hydrate_legacy()
    if OUT.exists(): shutil.rmtree(OUT)
    for s in ("train", "val"):
        (OUT / f"images/{s}").mkdir(parents=True, exist_ok=True)
        (OUT / f"labels/{s}").mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads((ROOT / "data/image_sources.json").read_text())
    legacy_rows = []
    for item in source_manifest["images"]:
        rid = item["record_id"]
        src_img = ROOT / "data/images" / item["split"] / item["filename"]
        src_lab = ROOT / "data/labels" / item["split"] / f"{rid}.txt"
        if not src_img.exists() or not src_lab.exists(): raise FileNotFoundError(rid)
        shutil.copy2(src_img, OUT / "images/train" / f"{rid}.jpg")
        n = insulator_only(src_lab, OUT / "labels/train" / f"{rid}.txt")
        legacy_rows.append({"record_id":rid,"n_insulators":n,"runtime_image_sha256":sha256(src_img),"role":"train_legacy"})

    new_rows = []
    for rid, (role, url, expected_sha) in NEW.items():
        image = OUT / f"images/{role}/{rid}.jpg"
        fetch_exact(url, expected_sha, image)
        src_lab = ROOT / f"data/v4_0_labels/{role}/{rid}.txt"
        if not src_lab.exists(): raise FileNotFoundError(src_lab)
        n = insulator_only(src_lab, OUT / f"labels/{role}/{rid}.txt")
        new_rows.append({"record_id":rid,"role":role,"n_insulators":n,"image_sha256":sha256(image),"source_url":url})

    # The final holdout labels may be versioned in the repository, but no final image may enter this runtime dataset.
    for rid in FINAL_IDS:
        if (OUT / f"images/final/{rid}.jpg").exists() or (ROOT / f"data/final_holdout/images/{rid}.jpg").exists():
            raise RuntimeError(f"v4 final holdout image unexpectedly visible to training job: {rid}")

    yaml = f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: insulator\n"
    (OUT / "data.yaml").write_text(yaml)
    report = {
        "status":"PASS","evidence_type":"v4.1-one-class-insulator-dataset",
        "split_release_sha256":split["split_release_sha256"],
        "legacy":legacy_rows,"new":new_rows,
        "counts":{
            "train_sources":len(legacy_rows)+sum(x["role"]=="train" for x in new_rows),
            "train_insulators":sum(x["n_insulators"] for x in legacy_rows)+sum(x["n_insulators"] for x in new_rows if x["role"]=="train"),
            "val_sources":sum(x["role"]=="val" for x in new_rows),
            "val_insulators":sum(x["n_insulators"] for x in new_rows if x["role"]=="val")
        },
        "final_holdout_visible_to_model":False,
        "final_holdout_ids":sorted(FINAL_IDS)
    }
    REPORTS.mkdir(exist_ok=True); (REPORTS / "v4_1_dataset_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
