"""Download a deterministic, group-separated TTPLA mirror subset with provenance."""
import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_json(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def fetch_rows(cfg, split):
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        query = urllib.parse.urlencode({"dataset": cfg["dataset_id"], "config": cfg["dataset_config"],
                                        "split": split, "offset": offset, "length": 100})
        page = get_json(cfg["dataset_api"] + "?" + query)
        total = page["num_rows_total"]
        rows.extend(page["rows"])
        offset += len(page["rows"])
        if not page["rows"]:
            break
    if len(rows) != total:
        raise RuntimeError(f"Incomplete {split} metadata: {len(rows)}/{total}")
    return rows


def group(row):
    return row["row"]["file_name"].split("_", 1)[0]


def categories(row):
    return row["row"]["annotations"]["category_name"]


def choose(rows, split, kind, wanted, owners):
    if kind == "positive":
        candidates = [r for r in rows if "tower_lattice" in categories(r)]
    else:
        candidates = [r for r in rows if "tower_lattice" not in categories(r)
                      and ({"tower_wooden", "tower_tucohy"} & set(categories(r)))]
    candidates.sort(key=lambda r: (r["row"]["file_name"], r["row_idx"]))
    selected = []
    # First pass maximises distinct source groups within a split.
    seen_here = set()
    for row in candidates:
        g = group(row)
        if owners.get(g, split) != split or g in seen_here:
            continue
        owners[g] = split
        seen_here.add(g)
        selected.append(row)
        if len(selected) == wanted:
            return selected
    # Second pass fills from already claimed groups, still never crossing splits.
    used = {(r["row_idx"], r["row"]["file_name"]) for r in selected}
    for row in candidates:
        g = group(row)
        key = (row["row_idx"], row["row"]["file_name"])
        if owners.get(g) != split or key in used:
            continue
        selected.append(row)
        if len(selected) == wanted:
            return selected
    raise RuntimeError(f"Only {len(selected)}/{wanted} {split} {kind} rows satisfy group separation")


def yolo_lines(row):
    data = row["row"]
    w, h = data["width"], data["height"]
    ann = data["annotations"]
    lines, corrections = [], []
    for name, segmentation in zip(ann["category_name"], ann["segmentation"]):
        if name != "tower_lattice":
            continue
        if len(segmentation) < 6 or len(segmentation) % 2:
            raise ValueError("Invalid lattice polygon")
        values = []
        for i, value in enumerate(segmentation):
            span = w if i % 2 == 0 else h
            raw = float(value)
            # The mirror contains at least one publisher coordinate 1.19 px above
            # the canvas. Preserve the raw polygon in the manifest and only clamp
            # sub-2-pixel serialization noise; reject larger geometry errors.
            if raw < -2.0 or raw > span + 2.0:
                raise ValueError(f'Polygon outside image in {data["file_name"]}: {raw} vs [0,{span}]')
            clipped = min(float(span), max(0.0, raw))
            if clipped != raw:
                corrections.append({"coordinate_index": i, "raw": raw, "clipped": clipped,
                                    "axis": "x" if i % 2 == 0 else "y"})
            norm = clipped / span
            values.append(norm)
        lines.append("0 " + " ".join(f"{x:.10f}" for x in values))
    return lines, corrections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ttpla_steelwork_demo_v1.json")
    args = ap.parse_args()
    cfg_path = ROOT / args.config
    cfg = json.loads(cfg_path.read_text())
    out = ROOT / cfg["dataset_output"]
    if out.exists():
        raise FileExistsError(f"Existing dataset: {out}; audit rather than overwrite")
    all_rows = {split: fetch_rows(cfg, split) for split in ["train", "val", "test"]}
    owners, chosen = {}, []
    for split in cfg["selection"]["order"]:
        counts = cfg["selection"]["counts"][split]
        for kind in ["positive", "hard_negative"]:
            for row in choose(all_rows[split], split, kind, counts[kind], owners):
                chosen.append((split, kind, row))
    out.mkdir(parents=True)
    records = []
    for split, kind, wrapped in chosen:
        row = wrapped["row"]
        image_url = row["image"]["src"]
        suffix = Path(row["file_name"]).suffix.lower() or ".jpg"
        image_path = out / "images" / split / f'{wrapped["row_idx"]:04d}_{Path(row["file_name"]).stem}{suffix}'
        image_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(image_url, image_path)
        label_path = out / "labels" / split / (image_path.stem + ".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines, corrections = yolo_lines(wrapped)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        records.append({
            "split": split, "selection_kind": kind, "source_group": group(wrapped),
            "dataset_row_index": wrapped["row_idx"], "image_id": row["image_id"],
            "file_name": row["file_name"], "width": row["width"], "height": row["height"],
            "image_file": str(image_path.relative_to(out)), "image_sha256": sha(image_path),
            "image_bytes": image_path.stat().st_size, "download_url": image_url,
            "rows_api": cfg["dataset_api"], "mirror_page": cfg["mirror_page"],
            "source_categories": row["annotations"]["category_name"],
            "source_annotations": row["annotations"], "label_file": str(label_path.relative_to(out)),
            "label_sha256": sha(label_path), "lattice_instances": len(lines),
            "boundary_corrections": corrections,
        })
    split_groups = {s: {r["source_group"] for r in records if r["split"] == s} for s in ["train", "val", "test"]}
    assert not (split_groups["train"] & split_groups["val"] | split_groups["train"] & split_groups["test"] |
                split_groups["val"] & split_groups["test"])
    (out / "dataset.yaml").write_text(f'path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: lattice_tower_steel_structure\n')
    manifest = {
        "status": "COMPLETE_SMALL_MIRROR_SUBSET",
        "config": cfg, "config_sha256": sha(cfg_path),
        "selection_is_full_ttpla": False, "publisher_split_preserved": True,
        "source_group_overlap": False, "uk_images": 0,
        "boundary_policy": "Preserve raw polygons; clamp only coordinates within 2 px of the canvas and record every correction; reject larger errors.",
        "records": records,
        "summary": {s: dict(Counter(r["selection_kind"] for r in records if r["split"] == s))
                    for s in ["train", "val", "test"]},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest_sha256": sha(out / "manifest.json"), "summary": manifest["summary"],
                      "bytes": sum(r["image_bytes"] for r in records)}, indent=2))


if __name__ == "__main__":
    main()
