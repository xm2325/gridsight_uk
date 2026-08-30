"""Prepare an expanded asset-grouped insulator specialist dataset.

All UK images here are consumed development assets. The untouched v3 cohort is
hash-pinned only and its pixels and boxes are never read for training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from PIL import Image

from prepare_uk_insulator_adaptation_v1 import NEGATIVES as PILOT_NEGATIVES
from prepare_uk_insulator_adaptation_v1 import POSITIVES as PILOT_POSITIVES

ROOT = Path(__file__).resolve().parents[1]
EPRI = ROOT / "data/external/epri_components_v1"
PILOT = ROOT / "data/external/uk_distribution_pilot_v1"
LOC_V1 = ROOT / "data/external/uk_insulator_localisation_v1"
LOC_V2 = ROOT / "data/external/uk_insulator_localisation_v2"
DEV_V2 = ROOT / "data/external/uk_insulator_development_v2"
ACCEPTANCE = ROOT / "data/external/uk_insulator_localisation_v3"
OUT = ROOT / "data/external/uk_insulator_adaptation_v2"

PINS = {
    EPRI / "manifest.json": "56e0517fcbf864f6c60aa1e2b0869cf9061138a32eb2b2acd40ad37efcb8cffa",
    PILOT / "manifest.json": "fcb4c41d8379bc3c4eab0e7e6c7a099af1d2ec6b533bcc5953d15816ed59d171",
    LOC_V1 / "manifest.json": "bb2e53667faa1d07e4d8be9a46f6aa73e9f6a95efecfe1ec0f3d3f879b2ea1b2",
    LOC_V2 / "manifest.json": "2fde93a4332e4499cb047a4a684808c798b2e13c387375fcb6ef98395697ffdf",
    DEV_V2 / "manifest.json": "e08f12a5738ad5f860f4c7165d7a4ffbba798c3e33a4569fdf9e5b4a8f020950",
    ACCEPTANCE / "manifest.json": "d74f206e506c9c61303cdf20c092c44c107332cc3931ccf0f6a8079e68ac50ac",
}

DEV_GROUPS = {
    "pilot_6414068", "pilot_6337870", "pilot_5722811", "pilot_4279330",
    "naddle_201106", "middle_rigg_201506", "pole_transformer_201305",
    "tredis_201311", "acol_200706",
}


def sha(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stable_key(seed, value):
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def yolo_lines(boxes, width, height):
    return ["0 " + " ".join(f"{value:.10f}" for value in (
        (x0 + x1) / 2 / width, (y0 + y1) / 2 / height,
        (x1 - x0) / width, (y1 - y0) / height)) for x0, y0, x1, y1 in boxes]


def square_crop(box, width, height, scale, minimum=192):
    x0, y0, x1, y1 = box
    # A square must fit inside both source dimensions.  Using the larger source
    # dimension here can produce a negative origin for portrait/landscape images.
    side = min(width, height, max(minimum, round(max(x1 - x0, y1 - y0) * scale)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = max(0, min(width - side, round(cx - side / 2)))
    top = max(0, min(height - side, round(cy - side / 2)))
    right, bottom = min(width, left + side), min(height, top + side)
    left, top = max(0, right - side), max(0, bottom - side)
    return [left, top, right, bottom]


def crop_labels(boxes, crop):
    x0, y0, x1, y1 = crop
    selected = []
    for box in boxes:
        bx0, by0, bx1, by1 = box
        intersects = max(x0, bx0) < min(x1, bx1) and max(y0, by0) < min(y1, by1)
        contained = x0 <= bx0 and y0 <= by0 and bx1 <= x1 and by1 <= y1
        if intersects and not contained:
            return None
        if contained:
            selected.append([bx0 - x0, by0 - y0, bx1 - x0, by1 - y0])
    return selected


def negative_crops(width, height, side=320):
    if width <= side and height <= side:
        return []
    side = min(side, width, height)
    windows = {(0, 0, side, side), (width - side, 0, width, side),
               (0, height - side, side, height), (width - side, height - side, width, height),
               ((width - side) // 2, (height - side) // 2,
                (width + side) // 2, (height + side) // 2)}
    return [list(row) for row in sorted(windows)]


def uk_sources():
    pilot = json.loads((PILOT / "manifest.json").read_text())
    by_id = {row["geograph_id"]: row for row in pilot["images"]}
    rows = []
    for photo_id, value in PILOT_POSITIVES.items():
        source = by_id[photo_id]
        rows.append({"source": "pilot", "photo_id": photo_id, "asset_group": f"pilot_{photo_id}",
                     "image_path": PILOT / source["image_file"], "image_sha256": source["sha256"],
                     "width": source["width"], "height": source["height"], "boxes": value["boxes"],
                     "reference_status": "analyst visible-object box; not expert reviewed"})
    for photo_id in PILOT_NEGATIVES:
        source = by_id[photo_id]
        rows.append({"source": "pilot", "photo_id": photo_id, "asset_group": f"pilot_{photo_id}",
                     "image_path": PILOT / source["image_file"], "image_sha256": source["sha256"],
                     "width": source["width"], "height": source["height"], "boxes": [],
                     "reference_status": "analyst no-target development decision"})
    for name, directory in (("localisation_v1", LOC_V1), ("localisation_v2", LOC_V2)):
        manifest = json.loads((directory / "manifest.json").read_text())
        for source in manifest["records"]:
            if source["role"] not in {"prospective_test", "hard_negative"}:
                continue
            rows.append({"source": name, "photo_id": source["photo_id"],
                         "asset_group": source["asset_group"], "image_path": ROOT / source["image_file"],
                         "image_sha256": source["image_sha256"], "width": source["width"],
                         "height": source["height"], "boxes": source["boxes"],
                         "reference_status": source["reference_status"]})
    development = json.loads((DEV_V2 / "manifest.json").read_text())
    for source in development["records"]:
        rows.append({"source": "development_v2", "photo_id": source["photo_id"],
                     "asset_group": source["asset_group"], "image_path": ROOT / source["image_file"],
                     "image_sha256": source["image_sha256"], "width": source["width"],
                     "height": source["height"], "boxes": source["boxes"],
                     "reference_status": source["reference_status"]})
    for row in rows:
        row["split"] = "dev" if row["asset_group"] in DEV_GROUPS else "train"
    return rows


def verify_definitions():
    for path, expected in PINS.items():
        if sha(path) != expected:
            raise ValueError(f"Pinned manifest changed: {path}")
    sources = uk_sources()
    for row in sources:
        if sha(row["image_path"]) != row["image_sha256"]:
            raise ValueError(f"UK image hash mismatch: {row['source']} {row['photo_id']}")
        for box in row["boxes"]:
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= row["width"] and 0 <= y0 < y1 <= row["height"]):
                raise ValueError(f"Invalid UK box: {row['photo_id']} {box}")
    groups = {split: {row["asset_group"] for row in sources if row["split"] == split}
              for split in ("train", "dev")}
    if groups["train"] & groups["dev"]:
        raise ValueError("UK development asset-group leakage")
    acceptance = json.loads((ACCEPTANCE / "manifest.json").read_text())
    accepted = [row for row in acceptance["records"] if row["role"] != "excluded"]
    source_hashes = {row["image_sha256"] for row in sources}
    source_ids = {row["photo_id"] for row in sources}
    source_groups = {row["asset_group"] for row in sources}
    if (source_hashes & {row["image_sha256"] for row in accepted} or
            source_ids & {row["photo_id"] for row in accepted} or
            source_groups & {row["asset_group"] for row in accepted}):
        raise ValueError("Training sources cross the untouched v3 boundary")
    return sources, groups


def link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def add_sample(source, split, name, boxes, records, origin, origin_id, origin_group,
               origin_sha, reference_status, crop=None):
    if sha(source) != origin_sha:
        raise ValueError(f"Source image hash mismatch: {origin_id}")
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        if crop:
            adjusted = crop_labels(boxes, crop)
            if adjusted is None:
                return False
            x0, y0, x1, y1 = crop
            image = image.crop(crop)
            image_path = OUT / "images" / split / f"{name}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path, quality=95)
        else:
            adjusted = boxes
            image_path = OUT / "images" / split / f"{name}.jpg"
            link_or_copy(source, image_path)
        width, height = image.size
    if not adjusted and boxes and crop:
        raise ValueError(f"Positive crop lost every target: {origin_id} {crop}")
    for box in adjusted:
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"Prepared box outside image: {origin_id} {box}")
    label_path = OUT / "labels" / split / f"{name}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(yolo_lines(adjusted, width, height)) + ("\n" if adjusted else ""))
    records.append({"sample_id": name, "split": split, "origin": origin, "origin_id": origin_id,
                    "origin_group": origin_group, "origin_sha256": origin_sha,
                    "image_file": str(image_path.relative_to(OUT)), "image_sha256": sha(image_path),
                    "label_file": str(label_path.relative_to(OUT)), "label_sha256": sha(label_path),
                    "width": width, "height": height, "boxes": adjusted, "crop_xyxy": crop,
                    "reference_status": reference_status})
    return True


def main(verify_only=False):
    sources, groups = verify_definitions()
    summary = {split: {"images": sum(row["split"] == split for row in sources),
                       "asset_groups": len(groups[split]),
                       "boxes": sum(len(row["boxes"]) for row in sources if row["split"] == split),
                       "negative_images": sum(row["split"] == split and not row["boxes"] for row in sources)}
               for split in ("train", "dev")}
    if verify_only:
        print(json.dumps({"status": "DEFINITIONS_VERIFIED", "uk": summary,
                          "acceptance_images_read_for_training": False}, indent=2))
        return
    if OUT.exists():
        raise FileExistsError(f"Existing prepared dataset: {OUT}")
    records = []
    epri = json.loads((EPRI / "manifest.json").read_text())
    for split, positive_n, negative_n in (("train", 160, 20), ("dev", 50, 10)):
        rows = [row for row in epri["images"] if row["split"] == split]
        positives = [row for row in rows if any(ref["class_name"] == "insulator" for ref in row["references"])]
        negatives = [row for row in rows if not any(ref["class_name"] == "insulator" for ref in row["references"])]
        selected = (sorted(positives, key=lambda row: stable_key(101, row["image_id"]))[:positive_n] +
                    sorted(negatives, key=lambda row: stable_key(103, row["image_id"]))[:negative_n])
        for row in selected:
            boxes = [ref["box"] for ref in row["references"] if ref["class_name"] == "insulator"]
            add_sample(EPRI / row["image_file"], split, f"epri_{row['image_id']}", boxes, records,
                       "EPRI", row["image_id"], row["circuit"], row["sha256"], "publisher polygon")
    for row in sources:
        prefix = f"uk_{row['source']}_{row['photo_id']}"
        add_sample(row["image_path"], row["split"], prefix + "_full", row["boxes"], records,
                   "UK_DEVELOPMENT", row["photo_id"], row["asset_group"], row["image_sha256"],
                   row["reference_status"])
        seen = set()
        if row["boxes"]:
            for box_index, box in enumerate(row["boxes"]):
                for scale in (4.0, 8.0):
                    crop = square_crop(box, row["width"], row["height"], scale)
                    if tuple(crop) in seen:
                        continue
                    seen.add(tuple(crop))
                    add_sample(row["image_path"], row["split"],
                               f"{prefix}_object{box_index + 1}_s{int(scale)}", row["boxes"], records,
                               "UK_DEVELOPMENT", row["photo_id"], row["asset_group"], row["image_sha256"],
                               row["reference_status"], crop)
        else:
            for index, crop in enumerate(negative_crops(row["width"], row["height"]), 1):
                add_sample(row["image_path"], row["split"], f"{prefix}_negative{index}", [], records,
                           "UK_DEVELOPMENT", row["photo_id"], row["asset_group"], row["image_sha256"],
                           row["reference_status"], crop)
    for location, path in (("local", OUT),
                           ("roihu", Path("/scratch/project_2012997/keen_ai") / OUT.relative_to(ROOT))):
        (OUT / f"dataset_{location}.yaml").write_text(
            f"path: {path}\ntrain: images/train\nval: images/dev\nnames:\n  0: insulator\n")
    manifest = {
        "version": "uk-insulator-adaptation-v2", "selection_frozen_before_training": True,
        "source_manifest_sha256": {str(path.relative_to(ROOT)): expected for path, expected in PINS.items()},
        "untouched_acceptance_manifest_sha256": PINS[ACCEPTANCE / "manifest.json"],
        "acceptance_images_read_for_training": False,
        "uk_reference_status": "analyst visible-object boxes or consumed no-target decisions; not expert reviewed",
        "uk_asset_groups": {split: sorted(groups[split]) for split in groups},
        "uk_asset_group_overlap": False, "uk_independent_source_summary": summary,
        "records": records,
        "counts": {split: {"samples": sum(row["split"] == split for row in records),
                           "boxes": sum(len(row["boxes"]) for row in records if row["split"] == split),
                           "uk_samples": sum(row["split"] == split and row["origin"] == "UK_DEVELOPMENT" for row in records),
                           "uk_asset_groups": len(groups[split])}
                   for split in ("train", "dev")},
        "claim_boundary": "Expanded development adaptation only. Instance crops are repeated views, not independent assets; v3 acceptance is untouched.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest_sha256": sha(OUT / "manifest.json"), "counts": manifest["counts"],
                      "uk": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-definitions", action="store_true")
    main(parser.parse_args().verify_definitions)
