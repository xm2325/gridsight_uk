#!/usr/bin/env python3
"""Train one bounded EPRI crossarm specialist and freeze transfer inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from keen_component_metrics import match_image, summarize, validate_predictions
from prepare_epri_crossarm_specialist_v2 import build as build_dataset, verify as verify_dataset
from roihu_demo_ablation import infer, nms
from insplad_adapt_common import start_runtime

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/epri_crossarm_specialist_v2.json"


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


def crossarm_references(record):
    return [{**reference, "class_id": 0} for reference in record["references"]
            if reference["class_name"] == "crossarm"]


def select_threshold(records, candidates, iou=0.5):
    rows = []
    for threshold in candidates:
        tp = fp = fn = 0
        for record in records:
            metric = match_image(record["predictions"], record["references"], iou, threshold)
            tp += metric["tp"]
            fp += metric["fp"]
            fn += metric["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall, "f1": f1})
    selected = max(rows, key=lambda row: (row["f1"], row["threshold"]))
    return {"candidates": rows, "selected": selected,
            "rule": "maximum development F1 at IoU 0.5; ties choose higher threshold"}


def verify_uk(config):
    source = ROOT / config["uk_acceptance"]["dataset"]
    if sha(source / "manifest.json") != config["uk_acceptance"]["manifest_sha256"]:
        raise ValueError("Frozen UK v3 manifest changed")
    manifest = load(source / "manifest.json")
    if (not manifest["selection_frozen_before_v2_adapted_model_inference"] or
            manifest["model_inference_performed_before_freeze"]):
        raise ValueError("UK v3 acceptance boundary is not frozen")
    rows = [row for row in manifest["records"] if row["role"] != "excluded"]
    if (len(rows), len({row["asset_group"] for row in rows})) != (9, 9):
        raise ValueError("UK v3 image/group counts changed")
    for row in rows:
        if sha(ROOT / row["image_file"]) != row["image_sha256"]:
            raise ValueError(f"UK source image changed: {row['record_id']}")
    return source, rows


def preflight(build=False):
    config = load(CONFIG)
    source = ROOT / config["source_dataset"]
    if sha(source / "manifest.json") != config["source_manifest_sha256"]:
        raise ValueError("Frozen EPRI source manifest changed")
    if sha(ROOT / config["initial_checkpoint"]) != config["initial_checkpoint_sha256"]:
        raise ValueError("Initial component checkpoint changed")
    if sha(ROOT / config["baseline_result"]) != config["baseline_result_sha256"]:
        raise ValueError("Pinned baseline result changed")
    if build:
        build_dataset()
    dataset = verify_dataset()
    uk_source, uk_rows = verify_uk(config)
    if (ROOT / config["run"]).exists():
        raise FileExistsError("Run exists; inspect rather than submit a duplicate")
    return config, load(source / "manifest.json"), dataset, uk_source, uk_rows


def predict(model, records, image_root, output, split, config, labelled):
    import torch
    from PIL import Image
    target = output / split / "predictions"
    target.mkdir(parents=True, exist_ok=False)
    floor = config["inference"]["raw_score_floor"]
    settings = {"imgsz": config["inference"]["imgsz"], "tiled": False}
    infer(model, Image.new("RGB", (1280, 960)), settings, 0, floor)
    summaries = []
    metric_records = []
    for index, row in enumerate(records):
        image_id = row["image_id"] if labelled else row["record_id"]
        image_file = image_root / row["image_file"] if labelled else ROOT / row["image_file"]
        image_hash = row["sha256"] if labelled else row["image_sha256"]
        if sha(image_file) != image_hash:
            raise ValueError(f"Inference image changed: {image_id}")
        with Image.open(image_file) as photo:
            rgb = photo.convert("RGB")
            width, height = rgb.size
        torch.cuda.synchronize()
        started = time.perf_counter()
        raw, merged, regions = infer(model, rgb, settings, 0, floor)
        torch.cuda.synchronize()
        validate_predictions(raw, width, height, 1)
        if merged != nms(raw, config["inference"]["nms_iou"]):
            raise ValueError("Frozen NMS does not reproduce")
        payload = {"image_id": image_id, "image_sha256": image_hash,
                   "class_names": ["crossarm"], "raw_predictions": raw,
                   "predictions": merged, "raw_score_floor": floor,
                   "nms_iou": config["inference"]["nms_iou"],
                   "manual_roi_used": False, "full_frame_only": True,
                   "reference_boxes_accessed_or_written": False if not labelled else True}
        prediction_file = target / f"{image_id}.json"
        write(prediction_file, payload)
        references = crossarm_references(row) if labelled else []
        summaries.append({"image_id": image_id,
                          "prediction_file": str(prediction_file.relative_to(output)),
                          "prediction_sha256": sha(prediction_file),
                          "raw_count": len(raw), "nms_count": len(merged),
                          "elapsed_seconds": time.perf_counter() - started})
        if labelled:
            metric_records.append({"image_id": image_id, "predictions": merged,
                                   "references": references})
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(json.dumps({"event": "INFERRED", "split": split,
                              "count": index + 1, "total": len(records)}), flush=True)
    thresholds = config["development_selection"]["score_candidates"]
    result = {"split": split, "records": summaries,
              "verified_prediction_files": len(summaries),
              "has_reference_truth": labelled,
              "reference_boxes_available_to_model": False}
    if labelled:
        result["metric_records"] = metric_records
        result["summary"] = summarize(metric_records, ["crossarm"], thresholds,
                                      config["frozen_evaluation"]["ap_ious"])
    write(output / split / "results.json", result)
    return result


def main(check_only=False):
    config, source_manifest, dataset_receipt, uk_source, uk_rows = preflight(build=True)
    if check_only:
        print(json.dumps({"event": "INPUT_CHECK_PASSED", "dataset": dataset_receipt,
                          "uk_images": len(uk_rows), "output_exists": False}))
        return
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "gputest":
        raise RuntimeError("Model execution requires Roihu gputest; there is no local fallback")
    import torch
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPETrainer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    output = ROOT / config["run"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "code").mkdir()
    snapshots = {}
    for relative in ["configs/epri_crossarm_specialist_v2.json",
                     "scripts/prepare_epri_crossarm_specialist_v2.py",
                     "scripts/roihu_epri_crossarm_specialist_v2.py",
                     "scripts/epri_crossarm_specialist_v2.sbatch",
                     "scripts/keen_component_metrics.py", "scripts/roihu_demo_ablation.py"]:
        path = ROOT / relative
        shutil.copy2(path, output / "code" / path.name)
        snapshots[relative] = sha(path)
    report = {"status": "TRAINING", "started_at": datetime.now(timezone.utc).isoformat(),
              "runtime": start_runtime(), "config": config, "protocol_sha256": sha(CONFIG),
              "source_snapshots": snapshots, "dataset_receipt": dataset_receipt,
              "uk_reference_boxes_accessed_for_selection_or_inference": False}
    write(output / "results.json", report)
    started = time.perf_counter()
    try:
        h = config["training"]
        model = YOLOE(config["detection_yaml"]).load(str(ROOT / config["initial_checkpoint"]))

        def on_start(trainer):
            expected = [str(ROOT / config["derived_dataset"] / "images" / "train" / f"{row['derived_id']}.jpg")
                        for row in load(ROOT / config["derived_dataset"] / "manifest.json")["records"]
                        if row["split"] == "train"]
            if sorted(trainer.train_loader.dataset.im_files) != sorted(expected):
                raise ValueError("Training loader differs from frozen repeated sample list")
            if trainer.model.names != {0: "crossarm"}:
                raise ValueError(f"One-class mapping changed: {trainer.model.names}")
            report["training_setup"] = {"training_samples": len(expected),
                                        "unique_source_images": 320,
                                        "repeats_are_independent_data": False,
                                        "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
                                        "batches_per_epoch": len(trainer.train_loader)}
            write(output / "results.json", report)

        def on_batch(trainer):
            if trainer.loss is not None and not torch.isfinite(trainer.loss).all():
                raise ValueError("Non-finite training loss")

        def on_epoch(trainer):
            report["training_progress"] = {"completed_epochs": trainer.epoch + 1,
                                            "elapsed_seconds": time.perf_counter() - started,
                                            "dev_metrics": {k: float(v) for k, v in trainer.metrics.items()}}
            write(output / "results.json", report)

        model.add_callback("on_train_start", on_start)
        model.add_callback("on_train_batch_end", on_batch)
        model.add_callback("on_fit_epoch_end", on_epoch)
        model.train(data=str(ROOT / config["derived_dataset"] / "train_roihu.yaml"),
                    trainer=YOLOEPETrainer,
                    **{key: h[key] for key in ("epochs", "imgsz", "batch", "nbs", "optimizer", "lr0", "lrf",
                                                "weight_decay", "warmup_epochs", "seed", "workers", "amp", "freeze",
                                                "cache", "mosaic", "mixup", "copy_paste", "translate", "scale",
                                                "fliplr", "hsv_h", "hsv_s", "hsv_v")},
                    device=0, deterministic=True, compile=False, project=str(output), name="training",
                    exist_ok=False, pretrained=True, patience=h["epochs"] + 1, plots=False,
                    verbose=False, save=True, save_period=-1, cos_lr=False, close_mosaic=0,
                    val=True, degrees=0.0, shear=0.0, perspective=0.0, flipud=0.0)
        if report.get("training_progress", {}).get("completed_epochs") != h["epochs"]:
            raise ValueError("Fixed training budget did not complete")
        checkpoint = Path(model.trainer.best)
        if not checkpoint.exists():
            raise FileNotFoundError("Development-selected checkpoint absent")
        report["selected_checkpoint"] = str(checkpoint)
        report["selected_checkpoint_sha256"] = sha(checkpoint)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        specialist = YOLOE(str(checkpoint)).to("cuda:0")
        if specialist.names != {0: "crossarm"}:
            raise ValueError("Reloaded specialist class map changed")
        dev_rows = [row for row in source_manifest["images"] if row["split"] == "dev"]
        dev = predict(specialist, dev_rows, ROOT / config["source_dataset"], output, "dev", config, True)
        threshold = select_threshold(dev["metric_records"], config["development_selection"]["score_candidates"])
        frozen = {"protocol_sha256": sha(CONFIG), "checkpoint_sha256": sha(checkpoint),
                  "development_threshold_selection": threshold,
                  "selected_score_threshold": threshold["selected"]["threshold"],
                  "pole_min_area_containment_threshold": config["development_selection"]["pole_min_area_containment_threshold"],
                  "evaluation_used_for_selection": False, "uk_used_for_selection": False,
                  "uk_reference_boxes_accessed": False}
        write(output / "frozen_choices.json", frozen)
        report["frozen_choices_sha256"] = sha(output / "frozen_choices.json")
        report["status"] = "FROZEN_BEFORE_EVALUATION_AND_UK"
        write(output / "results.json", report)

        eval_rows = [row for row in source_manifest["images"] if row["split"] == "eval"]
        evaluation = predict(specialist, eval_rows, ROOT / config["source_dataset"], output, "eval", config, True)
        uk = predict(specialist, uk_rows, uk_source, output, "uk", config, False)
        selected = frozen["selected_score_threshold"]
        eval_selected = select_threshold(evaluation["metric_records"], [selected])["selected"]
        uk_count = 0
        for row in uk["records"]:
            payload = load(output / row["prediction_file"])
            uk_count += sum(prediction["score"] >= selected for prediction in payload["predictions"])
        report.update(status="COMPLETE_FROZEN_CROSSARM_SPECIALIST",
                      completed_at=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.perf_counter() - started,
                      selected_score_threshold=selected,
                      development_selected_metrics=threshold["selected"],
                      evaluation_selected_metrics=eval_selected,
                      uk_proposal_count_at_selected_threshold=uk_count,
                      uk_images_with_proposals=sum(any(p["score"] >= selected for p in load(output / row["prediction_file"])["predictions"])
                                                   for row in uk["records"]),
                      claims={"uk_accuracy": False, "steel_composition": False,
                              "physical_pole_top": False, "calibrated_probability": False})
        write(output / "results.json", report)
        print(json.dumps({"event": "COMPLETE", "result_sha256": sha(output / "results.json"),
                          "selected_threshold": selected,
                          "dev": report["development_selected_metrics"],
                          "eval": report["evaluation_selected_metrics"],
                          "uk_proposals": uk_count}), flush=True)
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write(output / "results.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    main(args.check_inputs)
