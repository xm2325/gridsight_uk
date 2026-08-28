#!/usr/bin/env python3
"""Freeze 100 real, annotated InsPLAD validation images before GPU inference.

Sampling balances filename capture prefixes to reduce repeated views. A prefix
is not a verified physical asset ID. No selection depends on model predictions.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SEED = "gridsight-insplad100-20260827-v1"
TARGET_CATEGORIES = {7: "polymer insulator", 8: "glass insulator"}
ARCHIVE_SHA = "09bab48de770c548c6e834cbabe3735314eb04064ac853babd873061dc520948"


def hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def rank(value, seed=SEED):
    return hash_bytes(f"{seed}|{value}".encode())


def capture_group(filename):
    if PurePosixPath(filename).name != filename or "_DJI_" not in filename:
        raise ValueError(f"Unexpected source image name: {filename}")
    return filename.split("_DJI_", 1)[0]


def ordered_candidates(images, seed=SEED):
    groups = collections.defaultdict(list)
    for image in images:
        groups[capture_group(image["file_name"])].append(image)
    keys = sorted(groups, key=lambda group: rank(group, seed))
    for group in keys:
        groups[group].sort(key=lambda image: rank(image["file_name"], seed))
    for index in range(max(map(len, groups.values()), default=0)):
        for group in keys:
            if index < len(groups[group]):
                yield groups[group][index]


def target_references(annotations):
    references = []
    for annotation in annotations:
        # Exact IDs: shackles containing the word "insulator" are NOT insulators.
        if annotation["category_id"] not in TARGET_CATEGORIES:
            continue
        if annotation.get("iscrowd", 0):
            raise ValueError("This diagnostic evaluator does not implement crowd annotations")
        x, y, width, height = map(float, annotation["bbox"])
        if width <= 0 or height <= 0:
            raise ValueError("Invalid COCO box extent")
        references.append({"annotation_id": annotation["id"], "category_id": annotation["category_id"],
                           "category": TARGET_CATEGORIES[annotation["category_id"]],
                           "box": [x, y, x + width, y + height]})
    return references


def verify_dataset(directory):
    from PIL import Image
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["images"]
    if manifest["dataset"] != "InsPLAD-det" or len(rows) != 100:
        raise ValueError("Require the frozen 100-image InsPLAD dataset")
    if len({r["image_id"] for r in rows}) != 100 or len({r["sha256"] for r in rows}) != 100:
        raise ValueError("Duplicate image identities or bytes")
    annotation_bytes = (directory / "annotations.json").read_bytes()
    if hash_bytes(annotation_bytes) != manifest["annotation_sha256"]:
        raise ValueError("COCO annotation hash mismatch")
    coco = json.loads(annotation_bytes)
    by_image = collections.defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    for row in rows:
        path = (directory / row["image_file"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Image escaped dataset directory")
        if hash_bytes(path.read_bytes()) != row["sha256"]:
            raise ValueError(f"Image hash mismatch: {row['image_id']}")
        with Image.open(path) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError("Image dimensions mismatch")
        if target_references(by_image[row["image_id"]]) != row["references"]:
            raise ValueError("Manifest references differ from source COCO annotations")
    return manifest


def main():
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/external/insplad_cache/InsPLAD-det.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "data/external/insplad100")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    directory = args.output
    if args.verify_only:
        manifest = verify_dataset(directory)
        print(json.dumps({"status": "VERIFIED_100", **manifest["summary"]}), flush=True)
        return
    if (directory / "manifest.json").exists():
        raise FileExistsError("Manifest already frozen; use --verify-only, do not resample")
    provenance_path = args.archive.with_suffix(".source.json")
    provenance = json.loads(provenance_path.read_text())
    if provenance["sha256"] != ARCHIVE_SHA or args.archive.stat().st_size != provenance["bytes"]:
        raise ValueError("Not the previously verified official detection archive")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "images").mkdir(exist_ok=True)
    annotations_member = "annotations/instances_val.json"
    with zipfile.ZipFile(args.archive) as archive:
        original_annotations = archive.read(annotations_member)
        source = json.loads(original_annotations)
        categories = {c["id"]: c["name"] for c in source["categories"]}
        if any(categories.get(key) != value for key, value in TARGET_CATEGORIES.items()):
            raise ValueError("Unexpected source category mapping")
        by_image = collections.defaultdict(list)
        for annotation in source["annotations"]:
            by_image[annotation["image_id"]].append(annotation)
        rows, selected_images, hashes, excluded_duplicates = [], [], set(), []
        for image in ordered_candidates(source["images"]):
            member = "val/" + image["file_name"]
            data = archive.read(member)  # zipfile verifies each selected member's CRC.
            sha = hash_bytes(data)
            if sha in hashes:
                excluded_duplicates.append(image["id"])
                continue
            with Image.open(io.BytesIO(data)) as pixels:
                if pixels.size != (image["width"], image["height"]):
                    raise ValueError("COCO dimensions disagree with image bytes")
                pixels.verify()
            relative = "images/" + image["file_name"]
            path = directory / relative
            with path.open("xb") as output:
                output.write(data)
            hashes.add(sha)
            selected_images.append(image)
            rows.append({"image_id": image["id"], "file_name": image["file_name"], "image_file": relative,
                         "archive_member": member, "sha256": sha, "bytes": len(data),
                         "width": image["width"], "height": image["height"],
                         "capture_prefix": capture_group(image["file_name"]),
                         "references": target_references(by_image[image["id"]])})
            if len(rows) == 100:
                break
    if len(rows) != 100:
        raise RuntimeError("Not enough unique valid images")
    selected_ids = {r["image_id"] for r in rows}
    subset = {**source, "images": selected_images,
              "annotations": [a for a in source["annotations"] if a["image_id"] in selected_ids]}
    annotation_bytes = (json.dumps(subset, indent=2) + "\n").encode()
    (directory / "annotations.json").write_bytes(annotation_bytes)
    (directory / "source_instances_val.json").write_bytes(original_annotations)
    (directory / "source_archive.json").write_text(json.dumps(provenance, indent=2) + "\n")
    summary = {"n_images": len(rows), "capture_prefix_count": len({r["capture_prefix"] for r in rows}),
               "images_with_insulators": sum(bool(r["references"]) for r in rows),
               "images_without_annotated_insulators": sum(not r["references"] for r in rows),
               "insulator_instances": sum(len(r["references"]) for r in rows),
               "target_counts": dict(collections.Counter(ref["category"] for r in rows for ref in r["references"])),
               "image_bytes": sum(r["bytes"] for r in rows)}
    manifest = {"dataset": "InsPLAD-det", "version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_page": provenance["source_page"], "license": provenance["license_from_publisher"],
                "scope": "100 public validation images; Brazilian UAV inspection, NOT UK generalisation or defect evaluation",
                "selection": {"seed": SEED, "pool": annotations_member, "pool_size": len(source["images"]),
                              "method": "SHA256 order within filename-prefix groups, round-robin across hash-ordered groups; exact byte duplicates skipped",
                              "group_semantics": "filename capture prefix, NOT verified unique tower/asset identity",
                              "excluded_exact_duplicate_ids": excluded_duplicates,
                              "uses_model_predictions": False},
                "source_annotation_sha256": hash_bytes(original_annotations),
                "annotation_sha256": hash_bytes(annotation_bytes), "target_categories": TARGET_CATEGORIES,
                "summary": summary, "images": rows}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_dataset(directory)
    print(json.dumps({"status": "FROZEN_100", **summary}), flush=True)


if __name__ == "__main__":
    main()
