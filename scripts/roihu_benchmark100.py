#!/usr/bin/env python3
"""Run the preregistered four-arm comparison on the frozen InsPLAD100 subset."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from prepare_insplad100 import SEED, verify_dataset
from roihu_demo_ablation import ARMS, PROMPTS, counts_to_metrics, digest, evaluate, infer

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/insplad100_protocol.json"


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def metric_key(confidence, iou_threshold):
    return f"conf_{confidence:.2f}_iou_{iou_threshold:.2f}"


def operating_metrics(predictions, reference_boxes, protocol):
    return {metric_key(conf, threshold): evaluate(
        [p for p in predictions if p["score"] >= conf], reference_boxes, threshold)
        for conf in protocol["operating_confidences"] for threshold in protocol["iou_thresholds"]}


def summarize(results, arms, protocol):
    output = {}
    for arm in arms:
        rows = [row for row in results if row["arm"] == arm]
        metrics = {}
        for conf in protocol["operating_confidences"]:
            for threshold in protocol["iou_thresholds"]:
                key = metric_key(conf, threshold)
                pooled = counts_to_metrics(*(sum(row["metrics"][key][name] for row in rows)
                                             for name in ("tp", "fp", "fn")))
                pooled["false_positives_per_image"] = pooled["fp"] / len(rows) if rows else 0.0
                metrics[key] = pooled
        elapsed = sum(row["elapsed_seconds"] for row in rows)
        output[arm] = {"n_images": len(rows), "metrics": metrics,
                       "inference_seconds": elapsed, "mean_seconds_per_image": elapsed / len(rows) if rows else 0.0,
                       "total_regions": sum(row["regions"] for row in rows),
                       "peak_allocated_cuda_bytes": max((row["peak_allocated_cuda_bytes"] for row in rows), default=0)}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/external/insplad100")
    parser.add_argument("--limit", type=int, default=100, help="Values below 100 mark the run as smoke-only")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 100 or len(set(args.arms)) != len(args.arms):
        parser.error("Require limit 1..100 and distinct arms")
    dataset = args.dataset.resolve()
    manifest = verify_dataset(dataset)
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if manifest["selection"]["seed"] != SEED or protocol["selection_seed"] != SEED:
        raise RuntimeError("Dataset selection differs from preregistration")
    if protocol["prompts"] != PROMPTS or protocol["arms"] != list(ARMS):
        raise RuntimeError("Runner and frozen ablation protocol disagree")
    if args.check_inputs:
        print(json.dumps({"status": "INPUTS_VERIFIED", **manifest["summary"]}))
        return
    import torch
    from PIL import Image
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLOE
    if not os.getenv("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("GPU inference requires a Slurm CUDA allocation, never a login node")
    if not 0 <= args.device < torch.cuda.device_count():
        raise RuntimeError("Invalid visible GPU index")
    torch.cuda.set_device(args.device)
    torch.manual_seed(17)
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))))
    torch.backends.cudnn.benchmark = False
    os.chdir(ROOT)
    weights_config = json.loads((ROOT / "configs/roihu_benchmark_weights.json").read_text())
    weight_hashes = {}
    for asset in weights_config["assets"]:
        path = ROOT / asset["name"] if asset["name"].endswith(".ts") else ROOT / "weights" / asset["name"]
        actual = digest(path)
        if actual != asset["digest"].removeprefix("sha256:"):
            raise RuntimeError(f"Weight hash mismatch: {path}")
        weight_hashes[asset["name"]] = actual
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "runs/insplad100" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    rows = manifest["images"][:args.limit]
    print(json.dumps({"event": "RUN_START", "run_id": run_id, "output": str(output),
                      "images": len(rows), "arms": args.arms}), flush=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    report = {"status": "RUNNING", "run_id": run_id, "protocol": protocol,
              "arms": {name: ARMS[name] for name in args.arms}, "n_requested_images": len(rows),
              "dataset_summary": manifest["summary"], "dataset_manifest_sha256": digest(dataset / "manifest.json"),
              "protocol_sha256": digest(PROTOCOL_PATH), "git_commit": commit,
              "script_sha256": digest(__file__), "common_inference_sha256": digest(ROOT / "scripts/roihu_demo_ablation.py"),
              "weights_sha256": weight_hashes, "results": [],
              "runtime": {"python": platform.python_version(), "architecture": platform.machine(),
                          "torch": torch.__version__, "cuda_build": torch.version.cuda,
                          "ultralytics": importlib.metadata.version("ultralytics"),
                          "torchvision": importlib.metadata.version("torchvision"),
                          "gpu": torch.cuda.get_device_name(args.device), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                          "timing": "Synchronized single-pass wall time including crop/tiling, prediction, transfers and NMS; excludes image disk decode, loading, warm-up and output writing. Not p95."}}
    write_json(output / "results.json", report)
    write_json(output / "dataset_manifest.json", manifest)
    model, active_checkpoint = None, None
    started_run = time.perf_counter()
    try:
        for arm_name in args.arms:
            arm = ARMS[arm_name]
            if arm["checkpoint"] != active_checkpoint:
                model = None
                gc.collect()
                torch.cuda.empty_cache()
                model = YOLOE(str(ROOT / "weights" / arm["checkpoint"])).to(f"cuda:{args.device}")
                model.set_classes(PROMPTS)
                active_checkpoint = arm["checkpoint"]
            warmup = Image.new("RGB", (rows[0]["width"], rows[0]["height"]))
            infer(model, warmup, arm, args.device, protocol["inference_confidence"])
            torch.cuda.synchronize(args.device)
            print(json.dumps({"event": "ARM_START", "arm": arm_name}), flush=True)
            for index, row in enumerate(rows):
                with Image.open(dataset / row["image_file"]) as source:
                    image = source.convert("RGB")
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
                started = time.perf_counter()
                raw, merged, regions = infer(model, image, arm, args.device, protocol["inference_confidence"])
                torch.cuda.synchronize(args.device)
                elapsed = time.perf_counter() - started
                insulators = [p for p in merged if p["class_id"] == protocol["scored_model_class_id"]]
                references = [ref["box"] for ref in row["references"]]
                prediction_file = f"predictions/{row['image_id']}_{arm_name}.json"
                write_json(output / prediction_file, {"image_id": row["image_id"], "arm": arm_name,
                    "raw_predictions": raw, "merged_predictions": merged})
                result = {"arm": arm_name, "image_id": row["image_id"], "file_name": row["file_name"],
                          "capture_prefix": row["capture_prefix"], "image_sha256": row["sha256"],
                          "reference_count": len(references), "regions": regions,
                          "elapsed_seconds": elapsed,
                          "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(args.device),
                          "metrics": operating_metrics(insulators, references, protocol),
                          "prediction_file": prediction_file}
                report["results"].append(result)
                write_json(output / "results.json", report)
                if (index + 1) % 10 == 0 or index + 1 == len(rows):
                    print(json.dumps({"event": "PROGRESS", "arm": arm_name, "images_done": index + 1,
                                      "images_total": len(rows), "last_seconds": round(elapsed, 3)}), flush=True)
        report["summary"] = summarize(report["results"], args.arms, protocol)
        if len(report["results"]) != len(rows) * len(args.arms):
            raise RuntimeError("Incomplete comparison")
        report["status"] = "COMPLETED_100_IMAGE_DIAGNOSTIC" if len(rows) == 100 and args.arms == list(ARMS) else "COMPLETED_SMOKE_OR_PARTIAL_ARMS"
    except Exception as error:
        report["status"] = "FAILED_PARTIAL_RESULTS_NOT_COMPLETE"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["run_elapsed_seconds"] = time.perf_counter() - started_run
        write_json(output / "results.json", report)
    print(json.dumps({"event": "RUN_COMPLETE", "status": report["status"], "output": str(output),
                      "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
