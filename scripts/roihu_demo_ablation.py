#!/usr/bin/env python3
"""Development-only GPU ablation; never imports or runs the frozen evaluators.

Uses exact Commons originals in a NEW cache, without modifying legacy inputs.
No training, manual inference ROIs, material classification, or website deployment.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "data/image_sources.json"
CACHE = ROOT / "data/images/roihu_originals"
OUTPUT_ROOT = ROOT / "runs/roihu_demo"
# Explicit allowlist: no source is admitted merely because it has a label file.
SOURCE_ROLES = {
    "POS_5442616": "previously used validation; development diagnostic only",
    "POS_2326530": "adaptive development showcase; NOT an independent test",
}
PROMPTS = [
    "steel lattice transmission tower structure",
    "crossarm of an electricity transmission tower",
    "insulator string on an electricity transmission tower",
    "earth wire peak at the top of an electricity transmission tower",
]
ARMS = {
    "n640": {"checkpoint": "yoloe-26n-seg.pt", "imgsz": 640, "tiled": False},
    "n1280": {"checkpoint": "yoloe-26n-seg.pt", "imgsz": 1280, "tiled": False},
    "m1280": {"checkpoint": "yoloe-26m-seg.pt", "imgsz": 1280, "tiled": False},
    "m1280_tiles": {"checkpoint": "yoloe-26m-seg.pt", "imgsz": 1280, "tiled": True},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def select_sources(ids):
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Choose one or more distinct development source IDs")
    forbidden = set(ids) - SOURCE_ROLES.keys()
    if forbidden:
        raise ValueError(f"Not on development allowlist (holdouts forbidden): {sorted(forbidden)}")
    rows = {r["record_id"]: r for r in json.loads(SOURCE_MANIFEST.read_text())["images"]}
    return [rows[rid] for rid in ids]


def verify_source(path, row):
    from PIL import Image

    if digest(path) != row["expected_sha256"]:
        raise ValueError(f"Original-byte hash mismatch: {row['record_id']}")
    with Image.open(path) as image:
        if image.size != (row["expected_width_px"], row["expected_height_px"]):
            raise ValueError(f"Original dimensions mismatch: {row['record_id']}")


def prepare_sources(rows, download=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = CACHE / row["filename"]
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"{path}: run --prepare-only first")
            request = urllib.request.Request(row["url"], headers={"User-Agent": "GridSight-UK/roihu-development"})
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != row["expected_sha256"]:
                raise ValueError(f"Download is not the frozen Commons original: {row['record_id']}")
            temporary = path.with_suffix(".download")
            temporary.write_bytes(data)
            verify_source(temporary, row)
            temporary.replace(path)
        verify_source(path, row)


def axis_starts(length, tile, overlap):
    if length <= 0 or tile <= 0 or not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("Require positive dimensions and 0 <= overlap < 1")
    if length <= tile:
        return [0]
    step = max(1, round(tile * (1 - overlap)))
    return sorted(set(range(0, length - tile + 1, step)) | {length - tile})


def windows(width, height, tile=1280, overlap=0.25):
    return [(x, y, min(x + tile, width), min(y + tile, height))
            for y in axis_starts(height, tile, overlap)
            for x in axis_starts(width, tile, overlap)]


def offset_box(box, x, y, width, height):
    return [max(0.0, min(float(width), box[0] + x)),
            max(0.0, min(float(height), box[1] + y)),
            max(0.0, min(float(width), box[2] + x)),
            max(0.0, min(float(height), box[3] + y))]


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def nms(predictions, threshold=0.5):
    kept = []
    for prediction in sorted(predictions, key=lambda p: p["score"], reverse=True):
        if not any(prediction["class_id"] == other["class_id"] and
                   iou(prediction["box"], other["box"]) > threshold for other in kept):
            kept.append(prediction)
    return kept


def counts_to_metrics(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def evaluate(predictions, references, threshold):
    unused = set(range(len(references)))
    tp = 0
    for prediction in sorted(predictions, key=lambda p: p["score"], reverse=True):
        best = max(unused, key=lambda j: iou(prediction["box"], references[j]), default=None)
        if best is not None and iou(prediction["box"], references[best]) >= threshold:
            tp += 1
            unused.remove(best)
    return counts_to_metrics(tp, len(predictions) - tp, len(unused))


def insulator_references(row):
    path = ROOT / "data/labels" / row["split"] / f"{row['record_id']}.txt"
    boxes = []
    width, height = row["expected_width_px"], row["expected_height_px"]
    for line in path.read_text().splitlines():
        cls, x, y, w, h = map(float, line.split())
        if cls == 2:
            boxes.append([(x - w / 2) * width, (y - h / 2) * height,
                          (x + w / 2) * width, (y + h / 2) * height])
    return boxes, digest(path)


def infer(model, image, arm, device, confidence):
    # Full-frame context is always retained; no ground-truth or manual ROI is used.
    regions = [(0, 0, image.width, image.height)]
    if arm["tiled"]:
        regions += [r for r in windows(image.width, image.height) if r != regions[0]]
    predictions = []
    for index, (x1, y1, x2, y2) in enumerate(regions):
        result = model.predict(image.crop((x1, y1, x2, y2)), imgsz=arm["imgsz"],
                               conf=confidence, iou=0.5, agnostic_nms=False,
                               device=device, half=False, max_det=300, verbose=False)[0]
        if result.boxes is not None:
            for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                       result.boxes.conf.cpu().tolist(),
                                       result.boxes.cls.cpu().tolist()):
                predictions.append({"class_id": int(cls), "score": float(score),
                                    "box": offset_box(box, x1, y1, image.width, image.height),
                                    "region": index})
    merged = nms(predictions)
    return predictions, merged, len(regions)


def overlay(image, predictions, path, arm_name):
    from PIL import ImageDraw, ImageFont

    canvas = image.copy()
    canvas.thumbnail((1800, 1800))
    sx, sy = canvas.width / image.width, canvas.height / image.height
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, canvas.width, 32), fill="black")
    draw.text((6, 8), f"{arm_name} | model boxes | scores NOT probabilities | development only", fill="white", font=font)
    for prediction in predictions:
        b = prediction["box"]
        box = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
        draw.rectangle(box, outline="#20c997", width=2)
        draw.text((box[0], max(34, box[1] - 13)), f"insulator s={prediction['score']:.3f}", fill="#20c997", font=font)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=list(SOURCE_ROLES))
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--prepare-only", action="store_true", help="Download/verify original pixels only; no models")
    parser.add_argument("--check-inputs", action="store_true", help="Check local inputs only; no network/models")
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA GPU index; no silent CPU fallback")
    parser.add_argument("--conf", type=float, default=0.05, help="Fixed development score threshold, shared by all arms")
    args = parser.parse_args()
    if not math.isfinite(args.conf) or not 0 < args.conf <= 1:
        parser.error("--conf must be in (0, 1]")
    if len(args.arms) != len(set(args.arms)):
        parser.error("--arms must be distinct; repeated arms would inflate pooled counts")
    rows = select_sources(args.sources)
    prepare_sources(rows, download=args.prepare_only)
    if args.prepare_only or args.check_inputs:
        print(json.dumps({"status": "INPUTS_VERIFIED", "sources": [r["record_id"] for r in rows]}))
        return

    # No PyTorch import is needed for data checks or unit tests.
    import torch
    from PIL import Image
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLOE

    if not torch.cuda.is_available() or not 0 <= args.device < torch.cuda.device_count():
        raise RuntimeError("A valid CUDA allocation is required. Use Slurm; do not run inference on login nodes.")
    torch.cuda.set_device(args.device)
    torch.manual_seed(17)
    os.chdir(ROOT)  # YOLOE text-encoder caches are relative to the working directory.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUTPUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    report = {
        "status": "RUNNING", "run_id": run_id, "git_commit": commit,
        "script_sha256": digest(__file__), "source_manifest_sha256": digest(SOURCE_MANIFEST),
        "claim_scope": f"{len(rows)} previously consumed development source(s); NOT independent holdout performance.",
        "reference_scope": "Existing component references; not independently adjudicated inspection ground truth.",
        "protocol": {"arms": {name: ARMS[name] for name in args.arms}, "prompts": PROMPTS,
                     "confidence": args.conf, "nms_iou": 0.5, "tile_size": 1280, "overlap": 0.25,
                     "scored_class_id": 2, "half": False, "reference_boxes_used_for_inference": False,
                     "frozen_holdouts_used": False, "temperature_or_score_calibration": False},
        "runtime": {"python": platform.python_version(), "architecture": platform.machine(),
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    "ultralytics": importlib.metadata.version("ultralytics"),
                    "torchvision": importlib.metadata.version("torchvision"),
                    "numpy": importlib.metadata.version("numpy"),
                    "pillow": importlib.metadata.version("Pillow"),
                    "gpu": torch.cuda.get_device_name(args.device), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                    "timing": "Warm-up excluded; synchronized wall time for inference, tiling, CPU transfer and NMS. Not a p95 benchmark."},
        "checkpoints": {}, "results": [],
    }
    write_json(out / "results.json", report)
    model = None
    active_checkpoint = None
    try:
        encoder = ROOT / "mobileclip2_b.ts"
        if not encoder.is_file():
            raise FileNotFoundError(f"Pre-stage the official YOLOE text encoder at {encoder}; see ROIHU_DEMO.md")
        report["text_encoder_sha256"] = digest(encoder)
        for arm_name in args.arms:
            arm = ARMS[arm_name]
            name = arm["checkpoint"]
            if name != active_checkpoint:
                model = None
                gc.collect()
                torch.cuda.empty_cache()
                weight = ROOT / "weights" / name
                if not weight.is_file():
                    raise FileNotFoundError(f"Pre-stage the official checkpoint at {weight}; see ROIHU_DEMO.md")
                model = YOLOE(str(weight)).to(f"cuda:{args.device}")
                model.set_classes(PROMPTS)
                active_checkpoint = name
                report["checkpoints"][name] = digest(weight)
            warmup = Image.new("RGB", (arm["imgsz"], arm["imgsz"]))
            model.predict(warmup, imgsz=arm["imgsz"], device=args.device, half=False,
                          conf=args.conf, iou=0.5, agnostic_nms=False, verbose=False)
            for row in rows:
                rid = row["record_id"]
                with Image.open(CACHE / row["filename"]) as source:
                    image = source.convert("RGB")
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
                started = time.perf_counter()
                raw, merged, n_regions = infer(model, image, arm, args.device, args.conf)
                torch.cuda.synchronize(args.device)
                elapsed = time.perf_counter() - started
                # References enter only AFTER inference, never as crops/prompts/filters.
                references, label_hash = insulator_references(row)
                insulators = [p for p in merged if p["class_id"] == 2]
                result = {"arm": arm_name, "source_id": rid, "source_role": SOURCE_ROLES[rid],
                          "source_url": row["url"], "source_page": f"https://www.geograph.org.uk/photo/{rid[4:]}",
                          "source_sha256": row["expected_sha256"], "labels_sha256": label_hash,
                          "dimensions": [image.width, image.height], "regions": n_regions,
                          "elapsed_seconds": elapsed,
                          "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(args.device),
                          "metrics": {str(t): evaluate(insulators, references, t) for t in (0.3, 0.5)},
                          "raw_predictions": raw, "merged_predictions": merged}
                report["results"].append(result)
                overlay(image, insulators, out / f"{rid}_{arm_name}.png", arm_name)
                write_json(out / "results.json", report)
                print(json.dumps({k: result[k] for k in ("arm", "source_id", "metrics", "elapsed_seconds")}), flush=True)
        report["pooled_development_metrics"] = {}
        for arm_name in args.arms:
            report["pooled_development_metrics"][arm_name] = {}
            for threshold in ("0.3", "0.5"):
                metrics = [r["metrics"][threshold] for r in report["results"] if r["arm"] == arm_name]
                report["pooled_development_metrics"][arm_name][threshold] = counts_to_metrics(
                    *(sum(m[k] for m in metrics) for k in ("tp", "fp", "fn")))
        report["status"] = "COMPLETED_DEVELOPMENT_ONLY"
    except Exception as exc:
        report["status"] = "FAILED_PARTIAL_RESULTS_NOT_COMPLETE"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(out / "results.json", report)
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
