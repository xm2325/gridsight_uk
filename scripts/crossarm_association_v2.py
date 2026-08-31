#!/usr/bin/env python3
"""Apply a frozen upright-pole morphology guardrail to pinned crossarm outputs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from keen_component_metrics import match_image

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/crossarm_association_v2.json"


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


def upright_pole_associated(candidate, poles, rules):
    x0, y0, x1, y1 = candidate["box"]
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        return False
    candidate_centre_x = (x0 + x1) / 2
    candidate_centre_y = (y0 + y1) / 2
    for pole in poles:
        px0, py0, px1, py1 = pole["box"]
        pole_width, pole_height = px1 - px0, py1 - py0
        if pole_width <= 0 or pole_height <= 0:
            continue
        if pole_height / pole_width < rules["minimum_pole_height_over_width"]:
            continue
        pole_centre_x = (px0 + px1) / 2
        if not (x0 - rules["pole_centre_horizontal_candidate_padding"] * width <= pole_centre_x <=
                x1 + rules["pole_centre_horizontal_candidate_padding"] * width):
            continue
        relative_y = (candidate_centre_y - py0) / pole_height
        if not (rules["candidate_centre_min_relative_to_pole_top"] <= relative_y <=
                rules["candidate_centre_max_relative_to_pole_top"]):
            continue
        if width < rules["minimum_candidate_width_over_pole_width"] * pole_width:
            continue
        if height > rules["maximum_candidate_height_over_pole_height"] * pole_height:
            continue
        if (abs(candidate_centre_x - pole_centre_x) / pole_width >
                rules["maximum_candidate_centre_offset_over_pole_width"]):
            continue
        return True
    return False


def guard(predictions, poles, rules, threshold):
    return [row for row in predictions
            if row["score"] >= threshold and upright_pole_associated(row, poles, rules)]


def summarize_epri(records, rules, prompt_index, threshold):
    tp = fp = fn = 0
    for record in records:
        predictions = guard(record["prompt_predictions"][prompt_index], record["poles"], rules, threshold)
        metric = match_image(predictions, record["references"], 0.5, threshold)
        tp += metric["tp"]
        fp += metric["fp"]
        fn += metric["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "iou": 0.5, "threshold": threshold}


def verify_input(config):
    run = ROOT / config["input_run"]
    if sha(run / "results.json") != config["input_result_sha256"]:
        raise ValueError("Pinned input result changed")
    if sha(run / "frozen_choices.json") != config["input_frozen_choices_sha256"]:
        raise ValueError("Pinned input choices changed")
    result = load(run / "results.json")
    if result["uk_v3_accessed"] or result["uk_ground_truth_used"]:
        raise ValueError("Input evidence boundary changed")
    if result["selected"]["prompt_index"] != config["input_prompt_index"]:
        raise ValueError("Pinned prompt changed")
    epri_records = []
    for item in result["epri_records"]:
        path = run / item["record_file"]
        if sha(path) != item["record_sha256"]:
            raise ValueError(f"EPRI record changed: {path}")
        epri_records.append(load(path))
    uk_records = []
    for item in result["uk_records"]:
        path = run / item["record_file"]
        if sha(path) != item["record_sha256"]:
            raise ValueError(f"UK record changed: {path}")
        record = load(path)
        if (record["ground_truth_status"] != "NONE" or record["scores_are_probabilities"]
                or record["reference_truth"]):
            raise ValueError("UK truth/probability boundary changed")
        uk_records.append(record)
    if len(epri_records) != 80 or len(uk_records) != 27:
        raise ValueError("Pinned cohort size changed")
    return run, result, epri_records, uk_records


def build(output=None):
    config = load(CONFIG)
    run, input_result, epri_records, uk_records = verify_input(config)
    output = ROOT / config["run"] if output is None else Path(output)
    if output.exists():
        raise FileExistsError("Output exists; inspect rather than overwrite")
    output.mkdir(parents=True)
    (output / "code").mkdir()
    snapshots = {}
    for relative in ("configs/crossarm_association_v2.json", "scripts/crossarm_association_v2.py"):
        source = ROOT / relative
        shutil.copy2(source, output / "code" / source.name)
        snapshots[relative] = sha(source)
    rules = config["rules"]
    prompt_index = config["input_prompt_index"]
    threshold = config["input_raw_threshold"]
    epri_metrics = summarize_epri(epri_records, rules, prompt_index, threshold)
    records, before_total, after_total = [], 0, 0
    for source in uk_records:
        before = [row for row in source["associated_predictions"] if row["score"] >= threshold]
        after = guard(source["nms_predictions"], source["poles"], rules, threshold)
        before_total += len(before)
        after_total += len(after)
        payload = {
            "image_id": source["image_id"], "image_sha256": source["image_sha256"],
            "input_record_sha256": sha(run / "uk" / f"{source['image_id']}.json"),
            "input_poles": source["poles"], "before_predictions": before,
            "guarded_predictions": after, "ground_truth_status": "NONE",
            "accuracy": None, "scores_are_probabilities": False, "reference_truth": False,
        }
        target = output / "uk" / f"{source['image_id']}.json"
        write(target, payload)
        records.append({"image_id": source["image_id"],
                        "record_file": str(target.relative_to(output)),
                        "record_sha256": sha(target), "before_count": len(before),
                        "after_count": len(after)})
    report = {
        "status": "COMPLETE_DEVELOPMENT_MORPHOLOGY_GUARDRAIL",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(CONFIG), "source_snapshots": snapshots,
        "input_result_sha256": config["input_result_sha256"],
        "model_inference": False, "gradient_steps": 0,
        "epri_fixed_rule_metrics": epri_metrics,
        "input_epri_selected_metrics": input_result["selected"],
        "uk_development": {"images": 27, "before_proposals": before_total,
                           "after_proposals": after_total,
                           "images_with_after_proposals": sum(row["after_count"] > 0 for row in records),
                           "ground_truth_status": "NONE", "accuracy": None},
        "uk_records": records, "uk_v3_accessed": False, "uk_ground_truth_used": False,
        "scores_are_probabilities": False, "claim_boundary": config["claim_boundary"],
    }
    write(output / "results.json", report)
    write(output / "verification.json", {
        "status": "VERIFIED_PINNED_POSTPROCESSING_INPUTS",
        "input_result_sha256": config["input_result_sha256"],
        "input_frozen_choices_sha256": config["input_frozen_choices_sha256"],
        "verified_epri_records": 80, "verified_uk_records": 27,
        "uk_v3_accessed": False, "uk_ground_truth_used": False,
        "model_inference": False, "scores_presented_as_probabilities": False,
    })
    return report


if __name__ == "__main__":
    result = build()
    print(json.dumps({"status": result["status"], "epri": result["epri_fixed_rule_metrics"],
                      "uk_development": result["uk_development"],
                      "results_sha256": sha(ROOT / load(CONFIG)["run"] / "results.json")}, indent=2))
