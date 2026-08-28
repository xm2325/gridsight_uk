"""Shared, auditable inference for prompt and supervised adaptation experiments."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from prepare_insplad_adaptation import PROTOCOL, ROOT
from roihu_benchmark100 import operating_metrics, summarize, write_json
from roihu_demo_ablation import digest, infer, nms


def load_protocol():
    protocol = json.loads(PROTOCOL.read_text())
    if digest(ROOT / protocol["previous_run"] / "results.json") != protocol["previous_results_sha256"]:
        raise ValueError("Previous experiment changed; preserve the baseline")
    return protocol


def start_runtime(device=0):
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runtime/ultralytics_config"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    import torch
    import ultralytics
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Requires a Slurm CUDA allocation; never run models on a login node")
    if not 0 <= device < torch.cuda.device_count():
        raise ValueError("Invalid visible CUDA device")
    torch.cuda.set_device(device)
    torch.manual_seed(17)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    # Keep telemetry integrations off; the config lives in this project, not HOME.
    settings = ultralytics.settings
    settings.update({key: False for key in ("wandb", "mlflow", "clearml", "comet", "neptune", "dvc", "tensorboard") if key in settings})
    os.chdir(ROOT)
    config = json.loads((ROOT / "configs/roihu_benchmark_weights.json").read_text())
    hashes = {}
    for asset in config["assets"]:
        path = ROOT / asset["name"] if asset["name"].endswith(".ts") else ROOT / "weights" / asset["name"]
        hashes[asset["name"]] = digest(path)
        if hashes[asset["name"]] != asset["digest"].removeprefix("sha256:"):
            raise ValueError(f"Original pretrained weight hash mismatch: {path}")
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "ultralytics": importlib.metadata.version("ultralytics"), "gpu": torch.cuda.get_device_name(device),
            "slurm_job_id": os.environ["SLURM_JOB_ID"], "original_weights_sha256": hashes}


def create_run(stage, protocol, runtime):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "runs/insplad_adaptation" / stage / run_id
    output.mkdir(parents=True, exist_ok=False)
    code = output / "code"
    code.mkdir()
    snapshots = {}
    for relative in ("scripts/insplad_adapt_common.py", "scripts/roihu_insplad_train.py",
                     "scripts/roihu_prompt_ablation.py", "scripts/prepare_insplad_adaptation.py",
                     "scripts/roihu_adaptation.sbatch", "configs/insplad_adapt_protocol.json",
                     "configs/insplad_adapt_control_v1.json"):
        source = ROOT / relative
        if source.is_file():
            shutil.copyfile(source, code / source.name)
            snapshots[source.name] = digest(source)
    report = {"status": "RUNNING", "stage": stage, "run_id": run_id, "runtime": runtime,
              "protocol": protocol, "protocol_sha256": digest(PROTOCOL),
              "common_script_sha256": digest(__file__), "code_snapshot_sha256": snapshots, "results": []}
    write_json(output / "results.json", report)
    print(json.dumps({"event": "RUN_START", "stage": stage, "output": str(output), "run_id": run_id}), flush=True)
    return output, report


def collapse_targets(predictions, target_ids, threshold=0.5):
    selected = [{**p, "source_class_id": p["class_id"], "class_id": 0}
                for p in predictions if p["class_id"] in target_ids]
    return nms(selected, threshold)


def make_record(row, arm, predictions, elapsed, peak, prediction_file, protocol, **extra):
    return {"arm": arm, "image_id": row["image_id"], "file_name": row["file_name"],
            "image_sha256": row["sha256"], "reference_count": len(row["references"]),
            "regions": 1, "elapsed_seconds": elapsed, "peak_allocated_cuda_bytes": peak,
            "prediction_file": prediction_file,
            "metrics": operating_metrics(predictions, [ref["box"] for ref in row["references"]], protocol), **extra}


def run_predictions(model, rows, dataset, output, arm, target_ids, protocol, device=0, on_progress=None):
    from PIL import Image
    import torch
    folder = output / "predictions" / arm
    folder.mkdir(parents=True, exist_ok=False)
    config = dict(imgsz=protocol["inference_imgsz"], tiled=False)
    warmup = Image.new("RGB", (rows[0]["width"], rows[0]["height"]))
    infer(model, warmup, config, device, protocol["inference_confidence"])
    torch.cuda.synchronize(device)
    records = []
    for index, row in enumerate(rows):
        with Image.open(dataset / row["image_file"]) as source:
            image = source.convert("RGB")
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        raw, merged, _ = infer(model, image, config, device, protocol["inference_confidence"])
        predictions = collapse_targets(merged, target_ids, protocol["nms_iou"])
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        path = f"predictions/{arm}/{row['image_id']}.json"
        write_json(output / path, {"image_id": row["image_id"], "arm": arm, "raw_predictions": raw,
                                  "classwise_merged_predictions": merged, "target_predictions": predictions,
                                  "target_ids": target_ids, "image_sha256": row["sha256"]})
        record = make_record(row, arm, predictions, elapsed, torch.cuda.max_memory_allocated(device), path, protocol,
                             reused=False)
        records.append(record)
        if on_progress:
            on_progress(record)
        if (index + 1) % 20 == 0 or index + 1 == len(rows):
            print(json.dumps({"event": "INFERENCE_PROGRESS", "arm": arm, "images_done": index + 1, "images_total": len(rows)}), flush=True)
    return records


def reuse_diagnostic_baseline(manifest, dataset, output, protocol):
    source = ROOT / protocol["previous_run"]
    previous = json.loads((source / "results.json").read_text())
    if previous["dataset_manifest_sha256"] != digest(dataset / "manifest.json"):
        raise ValueError("Diagnostic images changed since baseline")
    by_id = {row["image_id"]: row for row in manifest["images"]}
    (output / "predictions/long_multi").mkdir(parents=True)
    records = []
    for row in previous["results"]:
        if row["arm"] != "m1280":
            continue
        stored = json.loads((source / row["prediction_file"]).read_text())
        targets = collapse_targets(stored["merged_predictions"], [2], protocol["nms_iou"])
        path = f"predictions/long_multi/{row['image_id']}.json"
        write_json(output / path, {"image_id": row["image_id"], "arm": "long_multi", "target_predictions": targets,
                                  "raw_predictions": stored["raw_predictions"], "classwise_merged_predictions": stored["merged_predictions"],
                                  "target_ids": [2], "image_sha256": row["image_sha256"],
                                  "reused_from": str(source / row["prediction_file"]),
                                  "source_prediction_sha256": digest(source / row["prediction_file"])})
        record = make_record(by_id[row["image_id"]], "long_multi", targets, row["elapsed_seconds"],
                             row["peak_allocated_cuda_bytes"], path, protocol, reused=True)
        if record["metrics"] != row["metrics"]:
            raise ValueError("Reused baseline metrics do not match original")
        records.append(record)
    if len(records) != 100:
        raise ValueError("Missing diagnostic baseline images")
    return records


def select_prompt(records, protocol):
    arms = list(protocol["prompt_arms"])
    summary = summarize(records, arms, protocol)
    if any(summary[arm]["n_images"] != 100 for arm in arms):
        raise ValueError("All prompt arms must include all 100 diagnostic images")
    winner = max(arms, key=lambda arm: summary[arm]["metrics"][protocol["primary_operating_point"]]["f1"])
    return winner, summary


def overfit_gate(records, protocol):
    gate = protocol["overfit"]["gate"]
    key = f"conf_{gate['confidence']:.2f}_iou_{gate['iou']:.2f}"
    summary = summarize(records, ["overfit"], protocol)["overfit"]
    metrics = summary["metrics"][key]
    passed = (len(records) == 2 and metrics["tp"] > 0 and metrics["recall"] >= gate["required_recall"] and
              metrics["precision"] >= gate["required_precision"])
    return {"passed": passed, "operating_point": key, "metrics": metrics,
            "scope": "training-set reconstruction only; not generalisation"}


def verify_records(records, rows, output, protocol):
    by_id = {row["image_id"]: row for row in rows}
    seen = set()
    for record in records:
        identity = record["arm"], record["image_id"]
        if identity in seen:
            raise ValueError("Duplicate image/arm prediction")
        seen.add(identity)
        row = by_id[record["image_id"]]
        if record["image_sha256"] != row["sha256"]:
            raise ValueError("Prediction image hash mismatch")
        path = (output / record["prediction_file"]).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError("Prediction path escaped run directory")
        stored = json.loads(path.read_text())
        if (stored["arm"], stored["image_id"]) != identity or stored["image_sha256"] != row["sha256"]:
            raise ValueError("Prediction identity does not match record")
        merged = nms(stored["raw_predictions"], protocol["nms_iou"])
        if merged != stored["classwise_merged_predictions"]:
            raise ValueError("Classwise NMS differs from stored result")
        targets = collapse_targets(merged, stored["target_ids"], protocol["nms_iou"])
        if targets != stored["target_predictions"]:
            raise ValueError("Target collapse/NMS differs from stored result")
        expected = operating_metrics(targets, [ref["box"] for ref in row["references"]], protocol)
        if expected != record["metrics"]:
            raise ValueError("Saved metrics differ from raw predictions")
    return len(seen)
