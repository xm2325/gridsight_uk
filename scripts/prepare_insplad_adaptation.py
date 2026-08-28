#!/usr/bin/env python3
"""Prepare a new group-disjoint supervised dataset from the cached official ZIP."""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from prepare_insplad100 import capture_group, hash_bytes, target_references, TARGET_CATEGORIES
from roihu_demo_ablation import digest
from roihu_benchmark100 import write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/insplad_adapt_protocol.json"
DEFAULT_DATASET = ROOT / "data/external/insplad_adaptation_v1"


def family(filename):
    group = capture_group(filename)
    result = group.split("-", 1)[0]
    if not result or not result.isalnum():
        raise ValueError(f"Unsafe filename family: {filename}")
    return result


def rank(value, seed):
    return hash_bytes(f"{seed}|{value}".encode())


def split_families(training_images, validation_images, protocol):
    training = {family(row["file_name"]) for row in training_images}
    excluded = {family(row["file_name"]) for row in validation_images}
    ordered = sorted(training - excluded, key=lambda group: rank("family|" + group, protocol["seed"]))
    h, d = protocol["holdout_families"], protocol["development_families"]
    if len(ordered) <= h + d:
        raise ValueError("Not enough disjoint families")
    return {"holdout": ordered[:h], "dev": ordered[h:h + d], "train": ordered[h + d:]}, sorted(excluded)


def ordered_images(images, groups, seed):
    pool = collections.defaultdict(list)
    for row in images:
        if family(row["file_name"]) in groups:
            pool[family(row["file_name"])].append(row)
    keys = sorted(pool, key=lambda group: rank("sample-family|" + group, seed))
    for group in keys:
        pool[group].sort(key=lambda row: rank("image|" + row["file_name"], seed))
    for index in range(max(map(len, pool.values()), default=0)):
        for group in keys:
            if index < len(pool[group]):
                yield pool[group][index]


def yolo_labels(references, width, height):
    lines = []
    for reference in references:
        x1, y1, x2, y2 = reference["box"]
        if (not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or
                not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height):
            raise ValueError("Reference box lies outside original image")
        values = [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                  (x2 - x1) / width, (y2 - y1) / height]
        lines.append("0 " + " ".join(f"{value:.10f}" for value in values))
    return "\n".join(lines) + ("\n" if lines else "")


def choose_overfit(rows):
    chosen = []
    for category in (8, 7):
        match = next((r for r in rows if r["split"] == "train" and r["references"] and
                      {a["category_id"] for a in r["references"]} == {category}), None)
        if match is None:
            raise ValueError("Need a training example for each insulator material")
        chosen.append(match["image_id"])
    if len(set(chosen)) != 2:
        raise ValueError("Overfit examples must be distinct")
    return chosen


def verify_dataset(directory, protocol=None):
    from PIL import Image
    directory = directory.resolve()
    protocol = protocol or json.loads(PROTOCOL.read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Adaptation protocol differs from frozen data")
    rows = manifest["images"]
    if collections.Counter(row["split"] for row in rows) != protocol["split_sizes"]:
        raise ValueError("Incomplete adaptation dataset")
    if len({row["sha256"] for row in rows}) != len(rows) or len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate identities or image bytes across splits")
    groups = {role: {row["family"] for row in rows if row["split"] == role}
              for role in protocol["split_sizes"]}
    for a in groups:
        if groups[a] & set(manifest["excluded_validation_families"]):
            raise ValueError("Previously observed validation family entered new dataset")
        for b in groups:
            if a != b and groups[a] & groups[b]:
                raise ValueError("Filename family leakage across splits")
    for filename, expected in manifest["source_annotation_sha256"].items():
        if digest(directory / filename) != expected:
            raise ValueError("Source annotations changed")
    source = json.loads((directory / "source_instances_train.json").read_text())
    by_image = collections.defaultdict(list)
    for annotation in source["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    for row in rows:
        if row["family"] != family(row["file_name"]) or row["family"] not in manifest["families"][row["split"]]:
            raise ValueError("Image assigned to wrong family/split")
        image_path = (directory / row["image_file"]).resolve()
        label_path = (directory / row["label_file"]).resolve()
        if not image_path.is_relative_to(directory) or not label_path.is_relative_to(directory):
            raise ValueError("Unsafe image or label path")
        if digest(image_path) != row["sha256"]:
            raise ValueError("Image bytes changed")
        with Image.open(image_path) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError("Image dimensions changed")
        references = target_references(by_image[row["image_id"]])
        if references != row["references"] or label_path.read_text() != yolo_labels(references, row["width"], row["height"]):
            raise ValueError("YOLO label conversion differs from original COCO reference")
    if choose_overfit(rows) != manifest["overfit_image_ids"]:
        raise ValueError("Overfit examples changed")
    return manifest


def main():
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/external/insplad_cache/InsPLAD-det.zip")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    directory = args.output.resolve()
    if args.verify_only:
        result = verify_dataset(directory, protocol)
        print(json.dumps({"status": "VERIFIED_GROUP_DISJOINT_DATASET", "summary": result["summary"]}))
        return
    if directory.exists():
        raise FileExistsError("Never overwrite a prepared or partial dataset; inspect it first")
    sha = hashlib.sha256()
    with args.archive.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    if sha.hexdigest() != protocol["source_archive_sha256"]:
        raise ValueError("Cached source archive no longer matches its verified bytes")
    with zipfile.ZipFile(args.archive) as archive:
        train_bytes = archive.read("annotations/instances_train.json")
        val_bytes = archive.read("annotations/instances_val.json")
        source, validation = json.loads(train_bytes), json.loads(val_bytes)
        mapping = {row["id"]: row["name"] for row in source["categories"]}
        if any(mapping.get(key) != value for key, value in TARGET_CATEGORIES.items()):
            raise ValueError("Source class IDs changed")
        groups, excluded = split_families(source["images"], validation["images"], protocol)
        directory.mkdir(parents=True)
        (directory / "source_instances_train.json").write_bytes(train_bytes)
        (directory / "source_instances_val.json").write_bytes(val_bytes)
        by_image = collections.defaultdict(list)
        for annotation in source["annotations"]:
            by_image[annotation["image_id"]].append(annotation)
        previous = json.loads((ROOT / "data/external/insplad100/manifest.json").read_text())
        used_hashes = {row["sha256"] for row in previous["images"]}
        rows, rejected_duplicates, missing_members = [], [], []
        available_members = set(archive.namelist())
        for role, count in protocol["split_sizes"].items():
            (directory / "images" / role).mkdir(parents=True)
            (directory / "labels" / role).mkdir(parents=True)
            selected = 0
            for row in ordered_images(source["images"], groups[role], protocol["seed"]):
                if "train/" + row["file_name"] not in available_members:
                    missing_members.append(row["file_name"])
                    continue
                data = archive.read("train/" + row["file_name"])
                image_sha = hash_bytes(data)
                if image_sha in used_hashes:
                    rejected_duplicates.append(row["file_name"])
                    continue
                with Image.open(io.BytesIO(data)) as image:
                    if image.size != (row["width"], row["height"]):
                        raise ValueError("Source dimensions mismatch")
                    image.verify()
                references = target_references(by_image[row["id"]])
                label = yolo_labels(references, row["width"], row["height"])
                image_file = f"images/{role}/{row['file_name']}"
                label_file = f"labels/{role}/{Path(row['file_name']).stem}.txt"
                (directory / image_file).write_bytes(data)
                (directory / label_file).write_text(label)
                used_hashes.add(image_sha)
                rows.append({"image_id": row["id"], "source_split": "official_train", "split": role,
                             "file_name": row["file_name"], "image_file": image_file, "label_file": label_file,
                             "family": family(row["file_name"]), "capture_prefix": capture_group(row["file_name"]),
                             "width": row["width"], "height": row["height"], "sha256": image_sha,
                             "bytes": len(data), "references": references})
                selected += 1
                if selected == count:
                    break
            if selected != count:
                raise ValueError(f"Insufficient samples in {role}")
        summary = {}
        for role in protocol["split_sizes"]:
            selected = [row for row in rows if row["split"] == role]
            summary[role] = {"images": len(selected), "families": len({r["family"] for r in selected}),
                             "positive_images": sum(bool(r["references"]) for r in selected),
                             "reference_count": sum(len(r["references"]) for r in selected),
                             "by_material": dict(collections.Counter(a["category"] for r in selected for a in r["references"]))}
        overfit_ids = choose_overfit(rows)
        manifest = {"dataset": "InsPLAD-det supervised adaptation v1", "created_at": datetime.now(timezone.utc).isoformat(),
                    "protocol": protocol, "protocol_sha256": digest(PROTOCOL),
                    "source_url": "https://data.mendeley.com/datasets/5n3fjgvfyz/1", "license": "CC BY-NC 3.0",
                    "source_archive_sha256": sha.hexdigest(), "families": groups,
                    "excluded_validation_families": excluded, "overfit_image_ids": overfit_ids,
                    "source_annotation_sha256": {"source_instances_train.json": hash_bytes(train_bytes), "source_instances_val.json": hash_bytes(val_bytes)},
                    "images": rows, "summary": summary, "rejected_duplicate_filenames": rejected_duplicates,
                    "source_images_missing_from_archive": missing_members,
                    "claim_scope": protocol["scope"]}
        common_yaml = f"path: {json.dumps(str(directory))}\nnames:\n  0: insulator\n"
        (directory / "train.yaml").write_text(common_yaml + "train: images/train\nval: images/dev\n")
        (directory / "overfit_images.txt").write_text("\n".join(str(directory / r["image_file"]) for r in rows if r["image_id"] in overfit_ids) + "\n")
        (directory / "overfit.yaml").write_text(common_yaml + "train: overfit_images.txt\nval: overfit_images.txt\n")
        write_json(directory / "manifest.json", manifest)
    verified = verify_dataset(directory, protocol)
    print(json.dumps({"status": "PREPARED_AND_VERIFIED", "manifest_sha256": digest(directory / "manifest.json"),
                      "summary": verified["summary"], "overfit_image_ids": overfit_ids}), flush=True)


if __name__ == "__main__":
    main()
