#!/usr/bin/env python3
"""Explain stored detection errors; no model execution or parameter selection."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from prepare_insplad100 import verify_dataset
from roihu_demo_ablation import digest, iou

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["n640", "n1280", "m1280", "m1280_tiles"]
KEY = "conf_0.05_iou_0.50"


def match(predictions, references):
    unused = set(range(len(references)))
    matched, false_positives = [], []
    for prediction in sorted(predictions, key=lambda p: p["score"], reverse=True):
        best = max(unused, key=lambda j: iou(prediction["box"], references[j]["box"]), default=None)
        if best is not None and iou(prediction["box"], references[best]["box"]) >= 0.5:
            unused.remove(best)
            matched.append((prediction, best))
        else:
            false_positives.append(prediction)
    return matched, false_positives, unused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/external/insplad100")
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads((run / "results.json").read_text())
    if report["status"] != "COMPLETED_100_IMAGE_DIAGNOSTIC":
        raise ValueError("Require complete inference")
    manifest = verify_dataset(args.dataset)
    if digest(args.dataset / "manifest.json") != report["dataset_manifest_sha256"]:
        raise ValueError("Dataset changed after inference")
    rows = {row["image_id"]: row for row in manifest["images"]}
    results = {(r["image_id"], r["arm"]): r for r in report["results"]}
    all_matches, details, audit = {}, {}, {}
    for arm in ARMS:
        matched_ids = set()
        type_tp, type_total = collections.Counter(), collections.Counter()
        negative_images_with_fp = negative_image_fp = 0
        overlap_types = collections.Counter()
        for row in rows.values():
            result = results[row["image_id"], arm]
            stored = json.loads((run / result["prediction_file"]).read_text())
            predictions = [p for p in stored["merged_predictions"] if p["class_id"] == 2 and p["score"] >= 0.05]
            matched, false, missing = match(predictions, row["references"])
            m = result["metrics"][KEY]
            if (len(matched), len(false), len(missing)) != (m["tp"], m["fp"], m["fn"]):
                raise ValueError("Detailed matching disagrees with recorded metrics")
            type_total.update(ref["category"] for ref in row["references"])
            for _, index in matched:
                reference = row["references"][index]
                type_tp[reference["category"]] += 1
                matched_ids.add((row["image_id"], reference["annotation_id"]))
            if not row["references"]:
                negative_images_with_fp += bool(false)
                negative_image_fp += len(false)
            for prediction in false:
                best_iou = max((iou(prediction["box"], ref["box"]) for ref in row["references"]), default=0.0)
                label = "overlap_ge_0.5_but_reference_already_matched" if best_iou >= 0.5 else (
                    "partial_overlap_0.1_to_0.5" if best_iou >= 0.1 else "overlap_below_0.1_or_no_reference")
                overlap_types[label] += 1
            details[row["image_id"], arm] = (matched, false, missing)
        all_matches[arm] = matched_ids
        audit[arm] = {"recall_by_ground_truth_type_not_material_classification": {
            name: {"tp": type_tp[name], "total": total, "recall": type_tp[name] / total}
            for name, total in type_total.items()},
            "negative_images_with_fp": negative_images_with_fp,
            "negative_image_fp_count": negative_image_fp,
            "fp_overlap_description_not_confirmed_error_cause": dict(overlap_types)}
    tile_changes = {"gained_reference_ids": sorted(all_matches["m1280_tiles"] - all_matches["m1280"]),
                    "lost_reference_ids": sorted(all_matches["m1280"] - all_matches["m1280_tiles"])}
    first_positive = next(row["image_id"] for row in rows.values() if row["references"])
    highest_added_fp = max((rid for rid in rows if rid != first_positive), key=lambda rid:
                          results[rid, "m1280_tiles"]["metrics"][KEY]["fp"] - results[rid, "m1280"]["metrics"][KEY]["fp"])
    all_missed = next(rid for rid, row in rows.items() if row["references"] and rid not in (first_positive, highest_added_fp)
                      and all(results[rid, arm]["metrics"][KEY]["tp"] == 0 for arm in ARMS))
    cases = [(first_positive, "First annotated sample"), (highest_added_fp, "Largest added FP count with tiles"),
             (all_missed, "Missed by all four models")]
    output = run / "report"
    output.mkdir(exist_ok=True)
    data = {"operating_point": KEY, "arms": audit, "tiling_changes": tile_changes,
            "case_selection": [{"image_id": rid, "file_name": rows[rid]["file_name"], "reason": reason}
                               for rid, reason in cases],
            "case_scope": "Illustrative examples: first positive by fixed order, plus post-hoc error examples. All 100 images remain in the report."}
    (output / "error_audit.json").write_text(json.dumps(data, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    fig, axes = plt.subplots(3, 4, figsize=(16, 8.8))
    for row_index, (rid, reason) in enumerate(cases):
        row = rows[rid]
        with Image.open(args.dataset / row["image_file"]) as source:
            image = source.convert("RGB")
        for column, arm in enumerate(ARMS):
            axis = axes[row_index, column]
            axis.imshow(image)
            matched, false, missing = details[rid, arm]
            for reference in row["references"]:
                x1, y1, x2, y2 = reference["box"]
                axis.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="#2563eb", linewidth=1.7, linestyle="--"))
            for prediction, color in [(p, "#009e73") for p, _ in matched] + [(p, "#d55e00") for p in false]:
                x1, y1, x2, y2 = prediction["box"]
                axis.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=1.7))
            axis.set_xticks([]); axis.set_yticks([])
            axis.set_title(f"{arm}  |  TP {len(matched)}  FP {len(false)}  FN {len(missing)}", fontsize=10)
            if column == 0:
                axis.set_ylabel(f"{reason}\nImage {rid}", fontsize=9)
    fig.suptitle("Real predictions | blue dashed: reference · green: matched · orange: unmatched", fontsize=14)
    fig.text(0.05, 0.02, "InsPLAD · score >= 0.05 · IoU >= 0.50 · illustrative success and failure cases; complete 100-image results in HTML", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output / "case_comparison.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
