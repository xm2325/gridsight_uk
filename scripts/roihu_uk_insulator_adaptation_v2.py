#!/usr/bin/env python3
"""One-shot UK insulator specialist v2 adaptation and frozen v3 evaluation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from roihu_uk_insulator_localisation_v1 import infer_regions, match_counts, nms, tile_windows

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def totals_for(run_root, prediction_rows, records, scores, ious):
    metrics = {}
    by_id = {row["record_id"]: row for row in records}
    for score in scores:
        metrics[str(score)] = {}
        for threshold in ious:
            totals = {"tp": 0, "fp": 0, "fn": 0}
            per_image = []
            for row in prediction_rows:
                payload = load(run_root / row["prediction_file"])
                predictions = [p for p in payload["full_plus_tiles"] if p["raw_score"] >= score]
                counts = match_counts(predictions, by_id[row["record_id"]]["boxes"], threshold)
                for key in totals:
                    totals[key] += counts[key]
                per_image.append({"record_id": row["record_id"], "role": by_id[row["record_id"]]["role"],
                                  **{key: counts[key] for key in totals}})
            tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
            metrics[str(score)][str(threshold)] = {
                **totals,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                "per_image": per_image,
            }
    return metrics


def evaluate_model(model, model_name, selected_class, records, target, cfg, torch):
    rows = []
    for record in records:
        from PIL import Image
        image_path = ROOT / record["image_file"]
        if sha(image_path) != record["image_sha256"]:
            raise ValueError(f"Acceptance image hash mismatch: {record['record_id']}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        full = [(0, 0, image.width, image.height)]
        tiles = tile_windows(image.width, image.height, cfg["tile_size"], cfg["tile_overlap"])
        all_full, selected_full, t1 = infer_regions(
            model, image, full, cfg["full_imgsz"], cfg["raw_score_floor"],
            cfg["max_det_per_region"], model_name, selected_class, 0, torch)
        all_tiles, selected_tiles, t2 = infer_regions(
            model, image, tiles, cfg["tile_imgsz"], cfg["raw_score_floor"],
            cfg["max_det_per_region"], model_name, selected_class, 0, torch)
        full_predictions = nms(selected_full, cfg["nms_iou"])
        combined = nms(selected_full + selected_tiles, cfg["nms_iou"])
        path = target / model_name / f"{record['record_id']}.json"
        write(path, {
            "record_id": record["record_id"], "role": record["role"],
            "image_sha256": record["image_sha256"], "references": record["boxes"],
            "reference_status": record["reference_status"], "model": model_name,
            "selected_class_id": selected_class, "tile_windows": tiles,
            "raw_region_predictions": {"full": all_full, "tiles": all_tiles},
            "full": full_predictions, "full_plus_tiles": combined,
            "inference_seconds": {"full": t1, "tiles": t2},
            "warning": "Raw uncalibrated proposals; scores are not probabilities and references are not expert truth."
        })
        rows.append({"record_id": record["record_id"], "role": record["role"],
                     "prediction_file": str(path.relative_to(target.parent)), "sha256": sha(path)})
        print(json.dumps({"event": "ACCEPTANCE_INFERENCE", "model": model_name,
                          "record_id": record["record_id"], "full": len(full_predictions),
                          "full_plus_tiles": len(combined)}), flush=True)
    return rows


def main(config_name):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "gputest":
        raise RuntimeError("Requires Roihu gputest; no local model fallback")
    import torch
    import ultralytics
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPETrainer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config_path = ROOT / config_name
    cfg = load(config_path)
    dataset = ROOT / cfg["dataset"]
    dataset_manifest_path = dataset / "manifest.json"
    acceptance = ROOT / cfg["acceptance_dataset"]
    acceptance_manifest_path = acceptance / "manifest.json"
    baseline_path = ROOT / cfg["baseline_checkpoint"]
    pins = [(dataset_manifest_path, cfg["dataset_manifest_sha256"]),
            (acceptance_manifest_path, cfg["acceptance_manifest_sha256"]),
            (baseline_path, cfg["baseline_checkpoint_sha256"])]
    for path, expected in pins:
        if sha(path) != expected:
            raise ValueError(f"Pinned input changed: {path}")
    dataset_manifest = load(dataset_manifest_path)
    if dataset_manifest["acceptance_images_read_for_training"]:
        raise ValueError("Dataset manifest permits acceptance leakage")
    acceptance_manifest = load(acceptance_manifest_path)
    if (not acceptance_manifest["selection_frozen_before_v2_adapted_model_inference"] or
            acceptance_manifest["model_inference_performed_before_freeze"]):
        raise ValueError("Acceptance boundary is not prospective")
    records = [row for row in acceptance_manifest["records"] if row["role"] in {"prospective_test", "hard_negative"}]
    expected = cfg["acceptance_counts"]
    if (len(records), sum(len(row["boxes"]) for row in records), len({row["asset_group"] for row in records}),
            sum(row["role"] == "hard_negative" for row in records)) != (
            expected["images"], expected["positive_boxes"], expected["asset_groups"], expected["hard_negatives"]):
        raise ValueError("Frozen acceptance counts changed")

    out = ROOT / cfg["run"]
    if out.exists():
        raise FileExistsError("Existing run: inspect it instead of submitting a duplicate")
    out.mkdir(parents=True)
    (out / "code").mkdir()
    snapshots = {}
    snapshot_paths = [config_path, Path(__file__), ROOT / "scripts/uk_insulator_adaptation_v2.sbatch",
                      ROOT / "scripts/prepare_uk_insulator_adaptation_v2.py",
                      ROOT / "scripts/acquire_uk_insulator_localisation_v3.py",
                      ROOT / "scripts/acquire_uk_insulator_development_v2.py",
                      ROOT / "scripts/roihu_uk_insulator_localisation_v1.py"]
    for path in snapshot_paths:
        shutil.copy2(path, out / "code" / path.name)
        snapshots[str(path.relative_to(ROOT))] = sha(path)
    result = {
        "status": "TRAINING", "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": os.environ.get("GRIDSIGHT_SUBMISSION_COMMIT") or os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "protocol": cfg, "protocol_sha256": sha(config_path), "source_snapshots": snapshots,
        "dataset_manifest_sha256": sha(dataset_manifest_path),
        "acceptance_manifest_sha256": sha(acceptance_manifest_path),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "ultralytics": ultralytics.__version__,
                    "gpu": torch.cuda.get_device_name(0), "slurm_job_id": os.environ["SLURM_JOB_ID"]},
        "training_progress": {}, "acceptance_predictions": {}, "claim_boundary": cfg["claim_boundary"]
    }
    write(out / "results.json", result)
    started = time.perf_counter()
    try:
        torch.set_num_threads(8)
        torch.manual_seed(cfg["training"]["seed"])
        model = YOLOE(cfg["detection_yaml"]).load(str(baseline_path))
        training = cfg["training"]
        expected_names = {0: "insulator"}

        def on_start(trainer):
            expected_train = {str(dataset / row["image_file"]) for row in dataset_manifest["records"] if row["split"] == "train"}
            expected_dev = {str(dataset / row["image_file"]) for row in dataset_manifest["records"] if row["split"] == "dev"}
            actual_train = set(trainer.train_loader.dataset.im_files)
            actual_dev = set(trainer.test_loader.dataset.im_files)
            if actual_train != expected_train or actual_dev != expected_dev or trainer.model.names != expected_names:
                raise ValueError("Trainer data boundary or one-class mapping changed")
            result["training_setup"] = {"training_images": len(actual_train), "development_images": len(actual_dev),
                                         "class_names": trainer.model.names, "args": vars(trainer.args),
                                         "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)}
            write(out / "results.json", result)

        def on_batch(trainer):
            if trainer.loss is not None and not torch.isfinite(trainer.loss).all():
                raise ValueError("Non-finite training loss")

        def on_epoch(trainer):
            result["training_progress"] = {
                "completed_epochs": trainer.epoch + 1,
                "losses": trainer.loss_items.detach().cpu().tolist() if trainer.loss_items is not None else [],
                "development_metrics": {key: float(value) for key, value in trainer.metrics.items()},
                "elapsed_seconds": time.perf_counter() - started,
            }
            write(out / "results.json", result)
            print(json.dumps({"event": "TRAINING_PROGRESS", **result["training_progress"]}), flush=True)

        model.add_callback("on_train_start", on_start)
        model.add_callback("on_train_batch_end", on_batch)
        model.add_callback("on_fit_epoch_end", on_epoch)
        keys = ("epochs", "imgsz", "batch", "nbs", "optimizer", "lr0", "lrf", "weight_decay",
                "warmup_epochs", "seed", "workers", "amp", "freeze", "cache", "mosaic", "mixup",
                "copy_paste", "translate", "scale", "fliplr", "hsv_h", "hsv_s", "hsv_v")
        model.train(data=str(dataset / "dataset_roihu.yaml"), trainer=YOLOEPETrainer,
                    **{key: training[key] for key in keys}, device=0, deterministic=True, compile=False,
                    project=str(out), name="training", exist_ok=False, pretrained=True,
                    patience=training["epochs"] + 1, plots=False, verbose=False, save=True, save_period=-1,
                    cos_lr=False, close_mosaic=0, val=True, degrees=0.0, shear=0.0, perspective=0.0, flipud=0.0)
        if result["training_progress"].get("completed_epochs") != training["epochs"]:
            raise ValueError("Fixed training budget did not complete")
        checkpoint = Path(model.trainer.best)
        if not checkpoint.exists():
            raise FileNotFoundError("No development-selected checkpoint")
        checkpoint_hash = sha(checkpoint)
        result.update(status="CHECKPOINT_FROZEN_BEFORE_ACCEPTANCE", selected_checkpoint=str(checkpoint.relative_to(out)),
                      selected_checkpoint_sha256=checkpoint_hash, training_seconds=time.perf_counter() - started,
                      acceptance_used_for_training_or_selection=False)
        choices = {"protocol_sha256": sha(config_path), "dataset_manifest_sha256": sha(dataset_manifest_path),
                   "acceptance_manifest_sha256": sha(acceptance_manifest_path),
                   "baseline_checkpoint_sha256": sha(baseline_path), "selected_checkpoint_sha256": checkpoint_hash,
                   "evaluation": cfg["evaluation"], "acceptance_used_for_selection": False}
        write(out / "frozen_choices_before_acceptance.json", choices)
        result["frozen_choices_sha256"] = sha(out / "frozen_choices_before_acceptance.json")
        write(out / "results.json", result)
        with (out / "acceptance_evaluation_receipt.json").open("x") as receipt:
            json.dump({"status": "STARTED", "frozen_choices_sha256": result["frozen_choices_sha256"]}, receipt, indent=2)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        evaluation = cfg["evaluation"]
        target = out / "predictions"
        baseline = YOLOE(str(baseline_path)).to("cuda:0")
        baseline_rows = evaluate_model(baseline, "baseline_v1_adapted", cfg["baseline_insulator_class_id"],
                                       records, target, evaluation, torch)
        result["acceptance_predictions"]["baseline_v1_adapted"] = baseline_rows
        write(out / "results.json", result)
        del baseline
        gc.collect()
        torch.cuda.empty_cache()
        adapted = YOLOE(str(checkpoint)).to("cuda:0")
        if adapted.names != expected_names:
            raise ValueError(f"Adapted checkpoint class mapping changed: {adapted.names}")
        adapted_rows = evaluate_model(adapted, "adapted_specialist", 0, records, target, evaluation, torch)
        result["acceptance_predictions"]["adapted_specialist"] = adapted_rows
        result["metrics"] = {
            "baseline_v1_adapted_full_plus_tiles": totals_for(out, baseline_rows, records, evaluation["operating_scores"], evaluation["evaluation_ious"]),
            "adapted_specialist_full_plus_tiles": totals_for(out, adapted_rows, records, evaluation["operating_scores"], evaluation["evaluation_ious"]),
        }
        # Supplementary full-frame metrics are derived without changing any operating point.
        for name, rows in (("baseline_v1_adapted_full", baseline_rows), ("adapted_specialist_full", adapted_rows)):
            copied = []
            for row in rows:
                payload = load(out / row["prediction_file"])
                temporary = out / "metrics_inputs" / name / f"{row['record_id']}.json"
                payload["full_plus_tiles"] = payload["full"]
                write(temporary, payload)
                copied.append({**row, "prediction_file": str(temporary.relative_to(out))})
            result["metrics"][name] = totals_for(out, copied, records, evaluation["operating_scores"], evaluation["evaluation_ious"])
        result.update(status="COMPLETE", completed_at=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.perf_counter() - started,
                      integrity={"acceptance_reference_boxes_used_for_inference": False,
                                 "acceptance_used_for_training_or_checkpoint_selection": False,
                                 "thresholds_selected_from_acceptance_results": False,
                                 "acceptance_inference_passes_per_checkpoint": 1,
                                 "outputs_are_calibrated_probabilities": False,
                                 "reference_boxes_are_expert_truth": False})
        write(out / "results.json", result)
        write(out / "acceptance_evaluation_receipt.json", {"status": "COMPLETE",
              "frozen_choices_sha256": result["frozen_choices_sha256"], "results_sha256": sha(out / "results.json")})
        print(json.dumps({"event": "COMPLETE", "metrics": result["metrics"]}, indent=2), flush=True)
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started)
        write(out / "results.json", result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/uk_insulator_adaptation_v2.json")
    main(parser.parse_args().config)
