#!/usr/bin/env python3
"""Derive abstaining, unscored pole-top assembly search regions."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pole_top_development_v2.json"


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path):
    return json.loads(Path(path).read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def unknown(reason):
    return {"status": "unknown", "label": "pole-top assembly search region", "xyxy": None,
            "score": None, "derived": True, "physical_component_verified": False,
            "reference_truth": False, "reason": reason}


def derive_region(poles, crossarms, width, height, rules):
    candidates = []
    for pole_index, pole in enumerate(poles):
        px0, py0, px1, py1 = pole["box"]
        pole_width, pole_height = px1 - px0, py1 - py0
        if pole_width <= 0 or pole_height / pole_width < rules["minimum_pole_height_over_width"]:
            continue
        pole_centre_x = (px0 + px1) / 2
        endpoints = [(pole_centre_x, py0), (pole_centre_x, py1)]
        for arm_index, arm in enumerate(crossarms):
            ax0, ay0, ax1, ay1 = arm["box"]
            arm_width, arm_height = ax1 - ax0, ay1 - ay0
            if arm_width <= 0 or arm_height <= 0:
                continue
            arm_centre = ((ax0 + ax1) / 2, (ay0 + ay1) / 2)
            if not (ax0 - rules["crossarm_horizontal_padding_fraction"] * arm_width <= pole_centre_x <=
                    ax1 + rules["crossarm_horizontal_padding_fraction"] * arm_width):
                continue
            if (arm_centre[1] - py0) / pole_height > rules["maximum_crossarm_centre_relative_to_pole_top"]:
                continue
            distances = [math.dist(point, arm_centre) for point in endpoints]
            endpoint_index = min(range(2), key=distances.__getitem__)
            near, far = distances[endpoint_index], distances[1 - endpoint_index]
            if near > rules["maximum_near_endpoint_distance_over_pole_height"] * pole_height:
                continue
            if (far < rules["minimum_far_to_near_distance_ratio"] * near or
                    far - near < rules["minimum_far_minus_near_over_pole_height"] * pole_height):
                continue
            candidates.append({"rank": near / pole_height, "pole_index": pole_index,
                               "crossarm_index": arm_index, "endpoint_index": endpoint_index,
                               "arm_centre": arm_centre, "pole_width": pole_width,
                               "pole_height": pole_height, "arm_width": arm_width,
                               "arm_height": arm_height})
    if not candidates:
        return unknown("no unique guarded crossarm near an upright pole endpoint")
    candidates.sort(key=lambda row: (row["rank"], row["pole_index"], row["crossarm_index"]))
    selected = candidates[0]
    if len(candidates) > 1:
        margin = candidates[1]["rank"] - selected["rank"]
        if margin < rules["ambiguity_margin_over_pole_height"]:
            return unknown("multiple pole/crossarm endpoint associations are geometrically ambiguous")
    side = max(rules["region_pole_width_multiplier"] * selected["pole_width"],
               rules["region_crossarm_extent_multiplier"] * selected["arm_width"],
               rules["region_crossarm_extent_multiplier"] * selected["arm_height"],
               rules["region_minimum_image_fraction"] * min(width, height))
    cx, cy = selected["arm_centre"]
    xyxy = [max(0.0, cx - side / 2), max(0.0, cy - side / 2),
            min(float(width), cx + side / 2), min(float(height), cy + side / 2)]
    return {"status": "geometry_candidate", "label": "pole-top assembly search region",
            "xyxy": xyxy, "score": None, "derived": True,
            "physical_component_verified": False, "reference_truth": False,
            "source_pole_prediction_index": selected["pole_index"],
            "source_crossarm_prediction_index": selected["crossarm_index"],
            "selected_axis_endpoint": "upper" if selected["endpoint_index"] == 0 else "lower",
            "reason": "guarded crossarm centred near the unambiguous endpoint of one upright pole"}


def build(output=None):
    config = load(CONFIG)
    source = ROOT / config["source_dataset"]
    input_run = ROOT / config["input_run"]
    if sha(source / "manifest.json") != config["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    if sha(input_run / "results.json") != config["input_result_sha256"]:
        raise ValueError("Pinned crossarm guardrail result changed")
    input_result = load(input_run / "results.json")
    if input_result["uk_v3_accessed"] or input_result["uk_ground_truth_used"]:
        raise ValueError("Input evidence boundary changed")
    metadata = {row["image_id"]: row for row in load(source / "manifest.json")["images"]}
    output = ROOT / config["run"] if output is None else Path(output)
    if output.exists():
        raise FileExistsError("Output exists; inspect rather than overwrite")
    output.mkdir(parents=True)
    (output / "code").mkdir()
    snapshots = {}
    for relative in ("configs/pole_top_development_v2.json", "scripts/pole_top_development_v2.py"):
        path = ROOT / relative
        shutil.copy2(path, output / "code" / path.name)
        snapshots[relative] = sha(path)
    records = []
    for item in input_result["uk_records"]:
        path = input_run / item["record_file"]
        if sha(path) != item["record_sha256"]:
            raise ValueError(f"Input record changed: {path}")
        source_record = load(path)
        meta = metadata[source_record["image_id"]]
        region = derive_region(source_record["input_poles"], source_record["guarded_predictions"],
                               meta["width"], meta["height"], config["geometry"])
        payload = {"image_id": source_record["image_id"], "image_sha256": meta["sha256"],
                   "input_record_sha256": item["record_sha256"], "pole_top": region,
                   "ground_truth_status": "NONE", "accuracy": None,
                   "score_is_probability": False, "reference_truth": False}
        target = output / "uk" / f"{source_record['image_id']}.json"
        write(target, payload)
        records.append({"image_id": source_record["image_id"],
                        "record_file": str(target.relative_to(output)),
                        "record_sha256": sha(target), "status": region["status"]})
    result = {"status": "COMPLETE_DEVELOPMENT_POLE_TOP_GEOMETRY",
              "completed_at": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": sha(CONFIG), "source_snapshots": snapshots,
              "input_result_sha256": config["input_result_sha256"],
              "model_inference": False, "gradient_steps": 0,
              "uk_development": {"images": len(records),
                                 "geometry_candidates": sum(row["status"] == "geometry_candidate" for row in records),
                                 "abstentions": sum(row["status"] == "unknown" for row in records),
                                 "ground_truth_status": "NONE", "accuracy": None},
              "records": records, "uk_v3_accessed": False, "uk_ground_truth_used": False,
              "scores_are_probabilities": False, "claim_boundary": config["claim_boundary"]}
    write(output / "results.json", result)
    return result


if __name__ == "__main__":
    result = build()
    print(json.dumps({"status": result["status"], "uk_development": result["uk_development"],
                      "results_sha256": sha(ROOT / load(CONFIG)["run"] / "results.json")}, indent=2))
