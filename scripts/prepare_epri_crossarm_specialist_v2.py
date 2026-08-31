#!/usr/bin/env python3
"""Build and verify a deterministic one-class EPRI crossarm dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/epri_crossarm_specialist_v2.json"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def link_exact(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError(f"Existing symlink changed: {target}")
        return
    if target.exists():
        if sha(target) != sha(source):
            raise ValueError(f"Existing derived image changed: {target}")
        return
    target.symlink_to(os.path.relpath(source, target.parent))


def label_text(record) -> str:
    lines = []
    width, height = record["width"], record["height"]
    for reference in record["references"]:
        if reference["class_name"] != "crossarm":
            continue
        x0, y0, x1, y1 = reference["box"]
        lines.append(f"0 {(x0+x1)/(2*width):.9f} {(y0+y1)/(2*height):.9f} {(x1-x0)/width:.9f} {(y1-y0)/height:.9f}")
    return "\n".join(lines) + ("\n" if lines else "")


def expected_rows(config, source_manifest):
    result = []
    repeats = config["sampling"]["positive_repeat_copies"]
    for record in source_manifest["images"]:
        has_crossarm = any(r["class_name"] == "crossarm" for r in record["references"])
        copies = 1 + (repeats if record["split"] == "train" and has_crossarm else 0)
        for copy_index in range(copies):
            suffix = "" if copy_index == 0 else f"__repeat{copy_index}"
            result.append((record, copy_index, f"{record['image_id']}{suffix}"))
    return result


def build() -> dict:
    config = json.loads(CONFIG.read_text())
    source = ROOT / config["source_dataset"]
    target = ROOT / config["derived_dataset"]
    if sha(source / "manifest.json") != config["source_manifest_sha256"]:
        raise ValueError("Frozen EPRI source manifest changed")
    manifest = json.loads((source / "manifest.json").read_text())
    rows = expected_rows(config, manifest)
    derived = []
    for record, copy_index, derived_id in rows:
        split = record["split"]
        source_image = source / record["image_file"]
        image = target / "images" / split / f"{derived_id}.jpg"
        label = target / "labels" / split / f"{derived_id}.txt"
        if sha(source_image) != record["sha256"]:
            raise ValueError(f"Source image hash changed: {record['image_id']}")
        link_exact(source_image, image)
        text = label_text(record)
        label.parent.mkdir(parents=True, exist_ok=True)
        if label.exists() and label.read_text() != text:
            raise ValueError(f"Existing derived label changed: {label}")
        label.write_text(text)
        derived.append({
            "derived_id": derived_id,
            "source_image_id": record["image_id"],
            "split": split,
            "circuit": record["circuit"],
            "repeat_index": copy_index,
            "independent_source_image": copy_index == 0,
            "has_crossarm": bool(text),
            "crossarm_instances": text.count("\n"),
            "image_file": str(image.relative_to(target)),
            "image_sha256": record["sha256"],
            "label_file": str(label.relative_to(target)),
            "label_sha256": sha(label),
        })
    for split in ("train", "dev", "eval"):
        listing = target / f"{split}.txt"
        listing.write_text("\n".join(str(target / r["image_file"]) for r in derived if r["split"] == split) + "\n")
    yaml = target / "train_roihu.yaml"
    yaml.write_text(f"path: {target}\ntrain: train.txt\nval: dev.txt\nnames:\n  0: crossarm\n")
    output = {
        "protocol_id": config["protocol_id"],
        "status": "VERIFIED_DERIVED_TRAINING_VIEW",
        "source_manifest_sha256": config["source_manifest_sha256"],
        "class_names": ["crossarm"],
        "repetition_is_independent_data": False,
        "records": derived,
    }
    write_json(target / "manifest.json", output)
    verify()
    return output


def verify() -> dict:
    config = json.loads(CONFIG.read_text())
    source = ROOT / config["source_dataset"]
    target = ROOT / config["derived_dataset"]
    if sha(source / "manifest.json") != config["source_manifest_sha256"]:
        raise ValueError("Frozen EPRI source manifest changed")
    source_manifest = json.loads((source / "manifest.json").read_text())
    manifest = json.loads((target / "manifest.json").read_text())
    expected = expected_rows(config, source_manifest)
    if len(manifest["records"]) != len(expected):
        raise ValueError("Derived record count changed")
    by_source = {r["image_id"]: r for r in source_manifest["images"]}
    seen = set()
    counts = {split: {"samples": 0, "unique": set(), "positive_samples": 0, "negative_samples": 0}
              for split in ("train", "dev", "eval")}
    for row in manifest["records"]:
        if row["derived_id"] in seen:
            raise ValueError("Duplicate derived ID")
        seen.add(row["derived_id"])
        source_row = by_source[row["source_image_id"]]
        if row["split"] != source_row["split"] or row["circuit"] != source_row["circuit"]:
            raise ValueError("Derived split or circuit changed")
        image, label = target / row["image_file"], target / row["label_file"]
        if sha(image) != source_row["sha256"] or sha(label) != row["label_sha256"]:
            raise ValueError("Derived image or label hash changed")
        if label.read_text() != label_text(source_row):
            raise ValueError("Derived label no longer matches publisher crossarm boxes")
        count = counts[row["split"]]
        count["samples"] += 1
        count["unique"].add(row["source_image_id"])
        count["positive_samples" if row["has_crossarm"] else "negative_samples"] += 1
    if {r["circuit"] for r in manifest["records"] if r["split"] == "train"} & {r["circuit"] for r in manifest["records"] if r["split"] != "train"}:
        raise ValueError("Circuit leakage into training")
    summary = {split: {**{k: v for k, v in count.items() if k != "unique"}, "unique": len(count["unique"])}
               for split, count in counts.items()}
    expected_summary = {
        "train": {"samples": 476, "unique": 320, "positive_samples": 234, "negative_samples": 242},
        "dev": {"samples": 80, "unique": 80, "positive_samples": 28, "negative_samples": 52},
        "eval": {"samples": 100, "unique": 100, "positive_samples": 46, "negative_samples": 54},
    }
    if summary != expected_summary:
        raise ValueError(f"Derived summary changed: {summary}")
    return {"status": "VERIFIED", "summary": summary, "manifest_sha256": sha(target / "manifest.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify() if args.verify_only else build(), indent=2))
