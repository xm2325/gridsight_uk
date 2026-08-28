#!/usr/bin/env python3
"""One fixed-budget distribution-component run, with immutable raw predictions."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from prepare_keen_components import ROOT, OUT as DATA, digest, write_json, verify
from keen_component_metrics import validate_predictions, summarize, match_image, geometric_confusion
from roihu_demo_ablation import infer, nms
from insplad_adapt_common import start_runtime

PROTOCOL = ROOT / "configs/keen_components_v1.json"
RUN = ROOT / "runs/keen_components/epri_components_v1_20260827"


def verify_predictions(output, rows, records, names, floor):
    by_id = {r["image_id"]: r for r in rows}
    seen = set()
    if len(records) != len(rows):
        raise ValueError("Prediction count differs from the frozen input list")
    for record in records:
        key = record["image_id"]
        if key in seen or key not in by_id:
            raise ValueError("Duplicate or unexpected prediction record")
        seen.add(key)
        row = by_id[key]
        path = output / record["prediction_file"]
        if digest(path) != record["prediction_sha256"]:
            raise ValueError("Raw prediction file changed")
        p = json.loads(path.read_text())
        if p["image_sha256"] != row["sha256"] or p["image_id"] != key or p["arm"] != record["arm"]:
            raise ValueError("Prediction source or arm mismatch")
        validate_predictions(p["raw_predictions"], row["width"], row["height"], len(names))
        if p["predictions"] != nms(p["raw_predictions"], .5):
            raise ValueError("Class-aware NMS does not reproduce")
        if any(x["score"] < floor for x in p["predictions"]):
            raise ValueError("Prediction below the frozen inference threshold")
        if record.get("metrics_025") is not None:
            expected = match_image(p["predictions"], row["references"], .5, .25)
            if expected != record["metrics_025"]:
                raise ValueError("Stored per-image metrics do not reproduce")
    return len(records)


def predict_rows(model, rows, source, output, arm, config, labelled=True):
    import torch
    from PIL import Image
    target = output / "predictions" / arm
    target.mkdir(parents=True, exist_ok=False)
    arm_cfg = dict(imgsz=config["inference"]["imgsz"], tiled=False)
    floor = config["inference"]["confidence_floor"]
    infer(model, Image.new("RGB", (1280, 960)), arm_cfg, 0, floor)
    records, metric_records = [], []
    for index, row in enumerate(rows):
        path = source / row["image_file"]
        if digest(path) != row["sha256"]:
            raise ValueError("Original image changed immediately before inference")
        with Image.open(path) as photo:
            image = photo.convert("RGB")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        raw, merged, regions = infer(model, image, arm_cfg, 0, floor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        validate_predictions(raw, row["width"], row["height"], len(config["classes"]))
        name = f"predictions/{arm}/{row['image_id']}.json"
        payload = {"image_id": row["image_id"], "image_sha256": row["sha256"], "arm": arm,
                   "raw_predictions": raw, "predictions": merged, "class_names": config["classes"],
                   "confidence_floor": floor, "postprocess": "class-aware NMS IoU=0.5; no class collapse",
                   "manual_roi_used": False, "material_status": "not classified"}
        write_json(output / name, payload)
        metrics = match_image(merged, row["references"], .5, .25) if labelled else None
        records.append({"image_id": row["image_id"], "arm": arm, "image_sha256": row["sha256"],
                        "prediction_file": name, "prediction_sha256": digest(output/name),
                        "elapsed_seconds": elapsed, "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
                        "regions": regions, "metrics_025": metrics})
        if labelled:
            metric_records.append({"image_id": row["image_id"], "predictions": merged, "references": row["references"]})
        if (index+1) % 25 == 0 or index+1 == len(rows):
            print(json.dumps({"event": "INFERRED", "arm": arm, "output": str(output), "count": index+1,
                              "total": len(rows)}), flush=True)
    verification_count = verify_predictions(output, rows, records, config["classes"], floor)
    summary = summarize(metric_records, config["classes"], config["evaluation"]["score_thresholds"],
                        config["evaluation"]["ap_ious"]) if labelled else None
    confusion = geometric_confusion(metric_records, config["classes"]) if labelled else None
    report = {"arm": arm, "records": records, "summary": summary, "geometric_confusion": confusion,
              "verified_prediction_count": verification_count, "has_ground_truth": labelled}
    write_json(output / f"{arm}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    config = json.loads(PROTOCOL.read_text())
    manifest = verify()
    if digest(DATA/"manifest.json") != config["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the frozen fingerprint")
    if digest(DATA/"selection_plan.json") != config["selection_plan_sha256"]:
        raise ValueError("Selection plan differs from frozen experiment")
    for split, size in config["split_sizes"].items():
        if sum(r["split"] == split for r in manifest["images"]) != size:
            raise ValueError("Incorrect source split size")
    if config["classes"] != manifest["classes"]:
        raise ValueError("Class mapping differs from publisher-derived labels")
    if digest(ROOT/config["model_checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("Original model weights changed")
    uk_dir = ROOT/config["uk_pilot"]
    uk = json.loads((uk_dir/"manifest.json").read_text())
    if digest(uk_dir/"manifest.json") != config["uk_manifest_sha256"]:
        raise ValueError("Qualitative pilot manifest differs from the frozen fingerprint")
    for row in uk["images"]:
        if row["ground_truth_status"] != "NONE" or digest(uk_dir/row["image_file"]) != row["sha256"]:
            raise ValueError("UK qualitative source checksum or role changed")
    if args.check_inputs:
        print(json.dumps({"event": "INPUT_CHECK_PASSED", "epri_images": len(manifest["images"]), "uk_images": len(uk["images"])}))
        return
    lock = ROOT / "runs/keen_components" / f"evaluation-{digest(DATA/'manifest.json')}.json"
    if lock.exists() or RUN.exists():
        raise FileExistsError("Run or evaluation receipt already exists; inspect checkpoints rather than submit a duplicate experiment")
    runtime = start_runtime()
    import torch
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPETrainer
    RUN.mkdir(parents=True, exist_ok=False)
    code = RUN/"code"
    code.mkdir()
    snapshots = {}
    for rel in ["scripts/roihu_keen_components.py", "scripts/keen_component_metrics.py",
                "scripts/prepare_keen_components.py", "scripts/roihu_demo_ablation.py",
                "scripts/insplad_adapt_common.py", "scripts/roihu_keen_components.sbatch", "configs/keen_components_v1.json"]:
        p = ROOT/rel
        shutil.copyfile(p, code/p.name)
        snapshots[rel] = digest(p)
    report = {"status": "TRAINING", "started_at": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
              "config": config, "protocol_sha256": digest(PROTOCOL), "source_snapshots": snapshots,
              "dataset_manifest_sha256": digest(DATA/"manifest.json"), "uk_manifest_sha256": digest(uk_dir/"manifest.json"),
              "evaluations": {}}
    write_json(RUN/"dataset_manifest.json", manifest)
    write_json(RUN/"uk_manifest.json", uk)
    write_json(RUN/"results.json", report)
    print(json.dumps({"event": "RUN_STARTED", "path": str(RUN), "slurm_job_id": runtime["slurm_job_id"]}), flush=True)
    started = time.perf_counter()
    try:
        h = config["training"]
        model = YOLOE(config["detection_yaml"]).load(str(ROOT/config["model_checkpoint"]))
        expected_names = dict(enumerate(config["classes"]))

        def on_start(trainer):
            expected = {str(DATA/r["image_file"]) for r in manifest["images"] if r["split"] == "train"}
            actual = set(trainer.train_loader.dataset.im_files)
            if actual != expected or trainer.model.names != expected_names:
                raise ValueError("Trainer loaded incorrect images or class labels")
            expected_dev = {str(DATA/r["image_file"]) for r in manifest["images"] if r["split"] == "dev"}
            if set(trainer.test_loader.dataset.im_files) != expected_dev:
                raise ValueError("Validation loader crosses the development boundary")
            report["training_setup"] = {"args": vars(trainer.args), "class_names": trainer.model.names,
                "training_images": sorted(actual), "validation_images": sorted(expected_dev),
                "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
                "batches_per_epoch": len(trainer.train_loader), "accumulate": trainer.accumulate}
            write_json(RUN/"results.json", report)

        def on_batch(trainer):
            if trainer.loss is not None and not torch.isfinite(trainer.loss).all():
                raise ValueError("Non-finite training loss")

        def on_epoch(trainer):
            report["training_progress"] = {"completed_epochs": trainer.epoch+1,
                "losses": trainer.loss_items.detach().cpu().tolist() if trainer.loss_items is not None else [],
                "dev_metrics": {key: float(value) for key, value in trainer.metrics.items()},
                "elapsed_seconds": time.perf_counter()-started}
            write_json(RUN/"results.json", report)
            print(json.dumps({"event": "TRAINING_PROGRESS", **report["training_progress"]}), flush=True)

        model.add_callback("on_train_start", on_start)
        model.add_callback("on_train_batch_end", on_batch)
        model.add_callback("on_fit_epoch_end", on_epoch)
        model.train(data=str(DATA/"train_roihu.yaml"), trainer=YOLOEPETrainer,
                    **{key:h[key] for key in ("epochs","imgsz","batch","nbs","optimizer","lr0","lrf","weight_decay",
                        "warmup_epochs","seed","workers","amp","freeze","cache","mosaic","mixup","copy_paste",
                        "translate","scale","fliplr","hsv_h","hsv_s","hsv_v")},
                    device=0, deterministic=True, compile=False, project=str(RUN), name="training", exist_ok=False,
                    pretrained=True, patience=h["epochs"]+1, plots=False, verbose=False, save=True, save_period=-1,
                    cos_lr=False, close_mosaic=0, val=True, degrees=0., shear=0., perspective=0., flipud=0.)
        if report.get("training_progress",{}).get("completed_epochs") != h["epochs"]:
            raise ValueError("Fixed epoch budget did not complete")
        checkpoint = Path(model.trainer.best)
        if not checkpoint.exists():
            raise FileNotFoundError("No development-selected checkpoint")
        report.update(selected_checkpoint=str(checkpoint), selected_checkpoint_sha256=digest(checkpoint),
                      training_seconds=time.perf_counter()-started)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        choices = {"protocol_sha256": digest(PROTOCOL), "dataset_manifest_sha256": report["dataset_manifest_sha256"],
                   "uk_manifest_sha256": report["uk_manifest_sha256"], "checkpoint_sha256": digest(checkpoint),
                   "baseline_checkpoint_sha256": config["checkpoint_sha256"], "prompts": config["baseline_prompts"],
                   "evaluation": config["evaluation"], "eval_used_for_selection": False, "uk_used_for_selection": False}
        write_json(RUN/"frozen_choices.json", choices)
        report["frozen_choices_sha256"] = digest(RUN/"frozen_choices.json")
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("x") as f:
            json.dump({"status":"EVALUATION_STARTED", "run": str(RUN), "choices_sha256":report["frozen_choices_sha256"]},f,indent=2)
        report["status"] = "EVALUATING_FROZEN_MODELS"
        write_json(RUN/"results.json", report)
        for arm in ["open_vocabulary", "supervised"]:
            if arm == "open_vocabulary":
                model = YOLOE(str(ROOT/config["model_checkpoint"])).to("cuda:0")
                prompts = config["baseline_prompts"]
                model.set_classes(prompts, model.get_text_pe(prompts))
            else:
                model = YOLOE(str(checkpoint)).to("cuda:0")
                if model.names != expected_names:
                    raise ValueError("Reloaded trained checkpoint has wrong class names")
            for split in ["dev", "eval", "uk"]:
                rows = uk["images"] if split == "uk" else [r for r in manifest["images"] if r["split"] == split]
                result = predict_rows(model, rows, uk_dir if split == "uk" else DATA,
                                      RUN/split, arm, config, labelled=split != "uk")
                report["evaluations"].setdefault(split,{})[arm] = {key:value for key,value in result.items() if key != "records"}
                write_json(RUN/"results.json", report)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status="COMPLETED_MULTICOMPONENT_TRAINING_AND_FROZEN_EVALUATION", elapsed_seconds=time.perf_counter()-started)
        write_json(RUN/"results.json", report)
        write_json(lock,{"status":"COMPLETED", "run":str(RUN), "results_sha256":digest(RUN/"results.json"),
                         "choices_sha256":report["frozen_choices_sha256"], "prediction_records":sum(
                             arm["verified_prediction_count"] for split in report["evaluations"].values() for arm in split.values())})
        print(json.dumps({"event":"COMPLETED", "path":str(RUN), "elapsed_seconds":report["elapsed_seconds"],
                          "eval":report["evaluations"]["eval"]}), flush=True)
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_json(RUN/"results.json", report)
        raise


if __name__ == "__main__":
    main()
