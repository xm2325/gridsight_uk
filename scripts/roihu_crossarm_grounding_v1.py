#!/usr/bin/env python3
"""Select a grounded crossarm proposal arm on EPRI truth, then audit UK development."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

from keen_component_metrics import match_image
from roihu_demo_ablation import nms
from insplad_adapt_common import start_runtime

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/crossarm_grounding_v1.json"


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path):
    return json.loads(Path(path).read_text())


def write(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def suppress_poles(predictions, score_threshold, containment_threshold):
    kept = []
    for prediction in sorted((p for p in predictions if p["class_id"] == 0 and p["score"] >= score_threshold),
                             key=lambda row: row["score"], reverse=True):
        if any(intersection(prediction["box"], other["box"]) /
               max(1e-12, min(area(prediction["box"]), area(other["box"]))) >= containment_threshold
               for other in kept):
            continue
        kept.append(prediction)
    return kept


def pole_associated(candidate, poles, rules):
    x0, y0, x1, y1 = candidate["box"]
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        return False
    for pole in poles:
        px0, py0, px1, py1 = pole["box"]
        pole_width, pole_height = px1 - px0, py1 - py0
        if pole_width <= 0 or pole_height <= 0:
            continue
        pole_centre_x = (px0 + px1) / 2
        candidate_centre_y = (y0 + y1) / 2
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
        return True
    return False


def variant_predictions(predictions, variant, poles, rules):
    if variant == "raw":
        return predictions
    if variant == "pole_associated":
        return [prediction for prediction in predictions if pole_associated(prediction, poles, rules)]
    raise ValueError(f"Unknown variant: {variant}")


def select_arm(records, prompts, variants, thresholds, rules):
    rows = []
    for prompt_index, _ in enumerate(prompts):
        for variant in variants:
            for threshold in thresholds:
                tp = fp = fn = 0
                for record in records:
                    predictions = variant_predictions(record["predictions"][prompt_index], variant,
                                                      record["poles"], rules)
                    metric = match_image(predictions, record["references"], 0.5, threshold)
                    tp += metric["tp"]
                    fp += metric["fp"]
                    fn += metric["fn"]
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                rows.append({"prompt_index": prompt_index, "prompt": prompts[prompt_index],
                             "variant": variant, "threshold": threshold,
                             "tp": tp, "fp": fp, "fn": fn,
                             "precision": precision, "recall": recall, "f1": f1})
    selected = max(rows, key=lambda row: (row["f1"], row["precision"], row["threshold"],
                                          row["variant"] == "pole_associated", -row["prompt_index"]))
    return {"candidates": rows, "selected": selected,
            "rule": "maximum EPRI circuit-4 F1; ties use precision, higher threshold, pole association, prompt order"}


def verify_release(directory: Path, expected: str):
    if sha(directory / "model.safetensors") != expected:
        raise ValueError("Grounding model hash changed")
    manifest = load(directory / "verified_manifest.json")
    for row in manifest["files"]:
        if sha(directory / row["file"]) != row["sha256"]:
            raise ValueError(f"Grounding release file changed: {row['file']}")


def baseline_record(run: Path, split: str, image_id: str):
    summary = load(run / split / "supervised.json")
    row = next(record for record in summary["records"] if record["image_id"] == image_id)
    path = run / split / row["prediction_file"]
    if sha(path) != row["prediction_sha256"]:
        raise ValueError(f"Baseline prediction changed: {image_id}")
    return load(path)


def preflight():
    config = load(CONFIG)
    epri = ROOT / config["epri_dataset"]
    baseline = ROOT / config["component_baseline_run"]
    uk = ROOT / config["uk_development"]["dataset"]
    if sha(epri / "manifest.json") != config["epri_manifest_sha256"]:
        raise ValueError("EPRI manifest changed")
    if sha(baseline / "results.json") != config["component_baseline_result_sha256"]:
        raise ValueError("Component baseline result changed")
    if sha(uk / "manifest.json") != config["uk_development"]["manifest_sha256"]:
        raise ValueError("UK development manifest changed")
    verify_release(ROOT / config["grounding_model"], config["grounding_model_sha256"])
    epri_manifest = load(epri / "manifest.json")
    epri_rows = [row for row in epri_manifest["images"] if row["split"] == config["epri_selection_split"]]
    uk_manifest = load(uk / "manifest.json")
    if (len(epri_rows), {row["circuit"] for row in epri_rows}) != (80, {4}):
        raise ValueError("EPRI selection boundary changed")
    if len(uk_manifest["images"]) != 27 or any(row["ground_truth_status"] != "NONE" for row in uk_manifest["images"]):
        raise ValueError("UK development role changed")
    for row in [*epri_rows, *uk_manifest["images"]]:
        root = epri if "circuit" in row else uk
        if sha(root / row["image_file"]) != row["sha256"]:
            raise ValueError(f"Source image changed: {row.get('image_id')}")
    if (ROOT / config["run"]).exists():
        raise FileExistsError("Run exists; inspect rather than submit a duplicate")
    return config, epri_rows, uk_manifest["images"]


def infer_prompt(model, processor, image, prompt, config, raw_target):
    import numpy as np
    import torch
    inputs = processor(images=image, text=prompt, size=config["grounding_size"], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        raw = model(**inputs)
    logits = raw.logits[0].float().cpu().numpy()
    boxes = raw.pred_boxes[0].float().cpu().numpy()
    scores = raw.logits[0].sigmoid().amax(-1).float().cpu().numpy()
    raw_target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_target, token_logits=logits, boxes_cxcywh=boxes,
                        input_ids=inputs["input_ids"].cpu().numpy())
    predictions = []
    for query_index, (box, score) in enumerate(zip(boxes, scores)):
        if score < config["raw_score_floor"]:
            continue
        cx, cy, width, height = map(float, box)
        xyxy = [max(0.0, (cx - width / 2) * image.width),
                max(0.0, (cy - height / 2) * image.height),
                min(float(image.width), (cx + width / 2) * image.width),
                min(float(image.height), (cy + height / 2) * image.height)]
        if xyxy[0] >= xyxy[2] or xyxy[1] >= xyxy[3]:
            continue
        predictions.append({"class_id": 1, "score": float(score), "box": xyxy,
                            "query_index": query_index, "source": "grounding_dino",
                            "calibrated_probability": False, "reference_truth": False})
    return predictions, nms(predictions, config["nms_iou"]), sha(raw_target)


def main(check_only=False):
    config, epri_rows, uk_rows = preflight()
    if check_only:
        print(json.dumps({"event": "INPUT_CHECK_PASSED", "epri_selection_images": len(epri_rows),
                          "uk_development_images": len(uk_rows), "uk_v3_accessed": False,
                          "output_exists": False}))
        return
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "gputest":
        raise RuntimeError("Requires Roihu gputest; no local model fallback")
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    output = ROOT / config["run"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "code").mkdir()
    snapshots = {}
    for relative in ("configs/crossarm_grounding_v1.json", "scripts/roihu_crossarm_grounding_v1.py",
                     "scripts/crossarm_grounding_v1.sbatch", "scripts/keen_component_metrics.py",
                     "scripts/roihu_demo_ablation.py"):
        source = ROOT / relative
        shutil.copy2(source, output / "code" / source.name)
        snapshots[relative] = sha(source)
    report = {"status": "EPRI_SELECTION_INFERENCE", "started_at": datetime.now(timezone.utc).isoformat(),
              "runtime": {**start_runtime(), "transformers": transformers.__version__},
              "config": config, "protocol_sha256": sha(CONFIG), "source_snapshots": snapshots,
              "epri_records": [], "uk_records": [], "gradient_steps": 0,
              "uk_ground_truth_used": False, "uk_v3_accessed": False, "performance_metrics": None}
    write(output / "results.json", report)
    started = time.perf_counter()
    try:
        processor = AutoProcessor.from_pretrained(ROOT / config["grounding_model"], local_files_only=True,
                                                  trust_remote_code=False)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            ROOT / config["grounding_model"], local_files_only=True, trust_remote_code=False,
            use_safetensors=True, disable_custom_kernels=True).to("cuda").eval()
        selection_records = []
        epri_root = ROOT / config["epri_dataset"]
        baseline_root = ROOT / config["component_baseline_run"]
        rules = config["selection"]["pole_association"]
        for number, row in enumerate(epri_rows, 1):
            with Image.open(epri_root / row["image_file"]) as source:
                image = source.convert("RGB")
            baseline = baseline_record(baseline_root, "dev", row["image_id"])
            poles = suppress_poles(baseline["predictions"], config["selection"]["pole_score_threshold"],
                                   config["selection"]["pole_min_area_containment_threshold"])
            prompt_predictions, prompt_records = [], []
            for prompt_index, prompt in enumerate(config["prompts"]):
                raw_path = output / "epri_raw" / row["image_id"] / f"prompt_{prompt_index}.npz"
                raw, merged, raw_sha = infer_prompt(model, processor, image, prompt, config, raw_path)
                prompt_predictions.append(merged)
                prompt_records.append({"prompt_index": prompt_index, "raw_count": len(raw),
                                       "nms_count": len(merged),
                                       "raw_file": str(raw_path.relative_to(output)), "raw_sha256": raw_sha})
            record_path = output / "epri" / f"{row['image_id']}.json"
            payload = {"image_id": row["image_id"], "image_sha256": row["sha256"],
                       "prompt_predictions": prompt_predictions, "prompt_records": prompt_records,
                       "poles": poles, "references": [{**reference, "class_id": 1}
                                                       for reference in row["references"]
                                                       if reference["class_name"] == "crossarm"],
                       "reference_source": "publisher EPRI polygon boxes"}
            write(record_path, payload)
            selection_records.append({"predictions": prompt_predictions, "poles": poles,
                                      "references": payload["references"]})
            report["epri_records"].append({"image_id": row["image_id"],
                                           "record_file": str(record_path.relative_to(output)),
                                           "record_sha256": sha(record_path)})
            if number % 20 == 0:
                print(json.dumps({"event": "EPRI_PROGRESS", "count": number, "total": len(epri_rows)}), flush=True)
        selection = select_arm(selection_records, config["prompts"], config["selection"]["variants"],
                               config["selection"]["score_candidates"], rules)
        frozen = {"protocol_sha256": sha(CONFIG), "selected": selection["selected"],
                  "selection_rule": selection["rule"], "selection_candidates": selection["candidates"],
                  "selection_source": "EPRI circuit 4 publisher truth only",
                  "uk_development_used_for_selection": False, "uk_v3_accessed": False}
        write(output / "frozen_choices.json", frozen)
        report["frozen_choices_sha256"] = sha(output / "frozen_choices.json")
        report["status"] = "FROZEN_BEFORE_UK_DEVELOPMENT"
        write(output / "results.json", report)
        selected = frozen["selected"]
        uk_root = ROOT / config["uk_development"]["dataset"]
        for number, row in enumerate(uk_rows, 1):
            with Image.open(uk_root / row["image_file"]) as source:
                image = source.convert("RGB")
            baseline = baseline_record(baseline_root, "uk", row["image_id"])
            poles = suppress_poles(baseline["predictions"], config["selection"]["pole_score_threshold"],
                                   config["selection"]["pole_min_area_containment_threshold"])
            raw_path = output / "uk_raw" / row["image_id"] / "selected_prompt.npz"
            raw, merged, raw_sha = infer_prompt(model, processor, image, selected["prompt"], config, raw_path)
            filtered = variant_predictions(merged, selected["variant"], poles, rules)
            displayed = [prediction for prediction in filtered if prediction["score"] >= selected["threshold"]]
            record_path = output / "uk" / f"{row['image_id']}.json"
            payload = {"image_id": row["image_id"], "image_sha256": row["sha256"],
                       "selected_prompt": selected["prompt"], "selected_variant": selected["variant"],
                       "selected_threshold": selected["threshold"], "raw_predictions": raw,
                       "nms_predictions": merged, "associated_predictions": filtered,
                       "display_predictions": displayed, "poles": poles,
                       "raw_file": str(raw_path.relative_to(output)), "raw_sha256": raw_sha,
                       "ground_truth_status": "NONE", "performance_metrics": None,
                       "scores_are_probabilities": False, "reference_truth": False}
            write(record_path, payload)
            report["uk_records"].append({"image_id": row["image_id"],
                                         "record_file": str(record_path.relative_to(output)),
                                         "record_sha256": sha(record_path),
                                         "display_count": len(displayed)})
            print(json.dumps({"event": "UK_DEVELOPMENT_PROGRESS", "count": number,
                              "total": len(uk_rows), "proposals": len(displayed)}), flush=True)
        report.update(status="COMPLETE_EPRI_SELECTED_UK_DEVELOPMENT_DIAGNOSTIC",
                      completed_at=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.perf_counter() - started,
                      selected=selected,
                      uk_proposal_count=sum(row["display_count"] for row in report["uk_records"]),
                      uk_images_with_proposals=sum(row["display_count"] > 0 for row in report["uk_records"]),
                      performance_metrics={"epri_selection": selected, "uk_development": None},
                      claim_boundary=config["claim_boundary"])
        write(output / "results.json", report)
        print(json.dumps({"event": "COMPLETE", "result_sha256": sha(output / "results.json"),
                          "selected": selected, "uk_proposals": report["uk_proposal_count"],
                          "uk_images_with_proposals": report["uk_images_with_proposals"]}), flush=True)
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write(output / "results.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    main(args.check_inputs)
