"""Frozen UK small-insulator localisation technique comparison; Roihu CUDA only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

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
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n"); temporary.replace(path)


def iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    intersection = ix * iy
    union = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)
    return intersection / union if union > 0 else 0.


def axis_starts(length, tile, overlap):
    if length <= 0 or tile <= 0 or not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("Require positive dimensions and 0 <= overlap < 1")
    if length <= tile:
        return [0]
    step = max(1, round(tile * (1 - overlap)))
    return sorted(set(range(0, length - tile + 1, step)) | {length - tile})


def tile_windows(width, height, tile, overlap):
    return [(x, y, min(x + tile, width), min(y + tile, height))
            for y in axis_starts(height, tile, overlap)
            for x in axis_starts(width, tile, overlap)]


def nms(predictions, threshold):
    kept = []
    for prediction in sorted(predictions, key=lambda p: p["raw_score"], reverse=True):
        if not any(iou(prediction["xyxy"], other["xyxy"]) > threshold for other in kept):
            kept.append(prediction)
    return kept


def fixed_priority_fusion(epri, mpid, threshold):
    """Keep EPRI first, then non-overlapping MPID proposals; never compare scores across models."""
    kept = list(sorted(epri, key=lambda p: p["raw_score"], reverse=True))
    for prediction in sorted(mpid, key=lambda p: p["raw_score"], reverse=True):
        if not any(iou(prediction["xyxy"], other["xyxy"]) > threshold for other in kept):
            kept.append(prediction)
    return kept


def match_counts(predictions, references, threshold):
    candidates = sorted(((iou(p["xyxy"], r), pi, ri)
                         for pi, p in enumerate(predictions) for ri, r in enumerate(references)), reverse=True)
    used_p, used_r, matches = set(), set(), []
    for overlap, pi, ri in candidates:
        if overlap < threshold:
            break
        if pi not in used_p and ri not in used_r:
            used_p.add(pi); used_r.add(ri)
            matches.append({"prediction_index": pi, "reference_index": ri, "iou": overlap})
    tp, fp, fn = len(matches), len(predictions)-len(matches), len(references)-len(matches)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp/(tp+fp) if tp+fp else 0., "recall": tp/(tp+fn) if tp+fn else 0.,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., "matches": matches}


def infer_regions(model, image, regions, imgsz, raw_floor, max_det, model_source,
                  selected_class, device, torch):
    all_boxes, selected, elapsed = [], [], 0.
    for region_index, (x0, y0, x1, y1) in enumerate(regions):
        torch.cuda.synchronize(); tick = time.perf_counter()
        result = model.predict(image.crop((x0, y0, x1, y1)), imgsz=imgsz, conf=raw_floor,
                               iou=0.5, agnostic_nms=False, device=device, half=False,
                               max_det=max_det, verbose=False)[0]
        torch.cuda.synchronize(); elapsed += time.perf_counter() - tick
        if result.boxes is None:
            continue
        for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                   result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            class_id = int(cls)
            row = {"source_model": model_source, "source_class_id": class_id,
                   "source_class_name": str(model.names[class_id]), "raw_score": float(score),
                   "xyxy": [max(0., min(float(image.width), float(box[0])+x0)),
                            max(0., min(float(image.height), float(box[1])+y0)),
                            max(0., min(float(image.width), float(box[2])+x0)),
                            max(0., min(float(image.height), float(box[3])+y0))],
                   "region_index": region_index, "region_xyxy": [x0, y0, x1, y1],
                   "calibrated_probability": False, "reference_truth": False}
            all_boxes.append(row)
            if selected_class is None or class_id == selected_class:
                selected.append(row)
    return all_boxes, selected, elapsed


def main(config_name):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "gputest":
        raise RuntimeError("Requires Roihu gputest; no local model fallback")
    import torch
    import ultralytics
    from PIL import Image
    from ultralytics import YOLOE
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cfg_path = ROOT / config_name; cfg = load(cfg_path)
    source = ROOT / cfg["source_dataset"]; manifest_path = source / "manifest.json"
    pins = [(manifest_path, cfg["source_manifest_sha256"]),
            (ROOT/cfg["epri_checkpoint"], cfg["epri_checkpoint_sha256"]),
            (ROOT/cfg["mpid_checkpoint"], cfg["mpid_checkpoint_sha256"])]
    for path, expected in pins:
        if sha(path) != expected:
            raise ValueError(f"Hash mismatch: {path}")
    manifest = load(manifest_path)
    if not manifest["selection_frozen_before_model_inference"] or manifest["model_inference_performed_before_freeze"]:
        raise ValueError("Acceptance boundary is not prospective")
    records = [r for r in manifest["records"] if r["role"] in {"prospective_test", "hard_negative"}]
    if len(records) != 8 or sum(len(r["boxes"]) for r in records) != 40:
        raise ValueError("Frozen acceptance counts changed")
    for record in records:
        if sha(ROOT/record["image_file"]) != record["image_sha256"]:
            raise ValueError(f"Image hash mismatch: {record['record_id']}")
    out = ROOT / cfg["run"]
    out.mkdir(parents=True, exist_ok=False); (out/"code").mkdir(); (out/"predictions").mkdir()
    for path in (cfg_path, Path(__file__), ROOT/"scripts/uk_insulator_localisation_v1.sbatch",
                 ROOT/"scripts/acquire_uk_insulator_localisation_v1.py"):
        shutil.copy2(path, out/"code"/path.name)
    result = {
        "status": "RUNNING", "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "protocol": cfg, "protocol_sha256": sha(cfg_path), "source_manifest_sha256": sha(manifest_path),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "ultralytics": ultralytics.__version__,
                    "gpu": torch.cuda.get_device_name(0), "slurm_job_id": os.environ["SLURM_JOB_ID"]},
        "checkpoints": {"epri": {"path": cfg["epri_checkpoint"], "sha256": sha(ROOT/cfg["epri_checkpoint"])},
                        "mpid": {"path": cfg["mpid_checkpoint"], "sha256": sha(ROOT/cfg["mpid_checkpoint"])}},
        "records": [], "claim_boundary": cfg["claim_boundary"]}
    write(out/"results.json", result)
    started = time.perf_counter()
    try:
        torch.set_num_threads(8); torch.manual_seed(47); torch.cuda.set_device(0)
        epri = YOLOE(str(ROOT/cfg["epri_checkpoint"])).to("cuda:0")
        mpid = YOLOE(str(ROOT/cfg["mpid_checkpoint"])).to("cuda:0")
        if int(cfg["epri_insulator_class_id"]) not in epri.names:
            raise ValueError("Pinned EPRI insulator class is absent")
        if len(mpid.names) != 3:
            raise ValueError(f"Pinned MPID model has unexpected classes: {mpid.names}")
        for record in records:
            image = Image.open(ROOT/record["image_file"]).convert("RGB")
            full = [(0, 0, image.width, image.height)]
            tiles = tile_windows(image.width, image.height, cfg["tile_size"], cfg["tile_overlap"])
            all_epri_full, epri_full, t1 = infer_regions(
                epri, image, full, cfg["full_imgsz"], cfg["raw_score_floor"],
                cfg["max_det_per_region"], "epri", cfg["epri_insulator_class_id"], 0, torch)
            all_epri_tiles, epri_tiles, t2 = infer_regions(
                epri, image, tiles, cfg["tile_imgsz"], cfg["raw_score_floor"],
                cfg["max_det_per_region"], "epri", cfg["epri_insulator_class_id"], 0, torch)
            all_mpid_full, mpid_full, t3 = infer_regions(
                mpid, image, full, cfg["full_imgsz"], cfg["raw_score_floor"],
                cfg["max_det_per_region"], "mpid", None, 0, torch)
            all_mpid_tiles, mpid_tiles, t4 = infer_regions(
                mpid, image, tiles, cfg["tile_imgsz"], cfg["raw_score_floor"],
                cfg["max_det_per_region"], "mpid", None, 0, torch)
            arms = {
                "epri_full": nms(epri_full, cfg["nms_iou"]),
                "epri_full_plus_tiles": nms(epri_full + epri_tiles, cfg["nms_iou"]),
                "mpid_full_plus_tiles": nms(mpid_full + mpid_tiles, cfg["nms_iou"]),
            }
            primary = cfg["primary_operating_score"]
            arms["proposal_fusion"] = fixed_priority_fusion(
                [p for p in arms["epri_full_plus_tiles"] if p["raw_score"] >= primary],
                [p for p in arms["mpid_full_plus_tiles"] if p["raw_score"] >= primary], cfg["nms_iou"])
            payload = {"record_id": record["record_id"], "role": record["role"],
                       "image_sha256": record["image_sha256"], "references": record["boxes"],
                       "reference_status": record["reference_status"], "tile_windows": tiles,
                       "raw_region_predictions": {"epri_full": all_epri_full, "epri_tiles": all_epri_tiles,
                                                  "mpid_full": all_mpid_full, "mpid_tiles": all_mpid_tiles},
                       "arms": arms, "inference_seconds": {"epri_full": t1, "epri_tiles": t2,
                                                            "mpid_full": t3, "mpid_tiles": t4},
                       "warning": "Raw uncalibrated model proposals; scores are not probabilities and references are not expert truth."}
            target = out/"predictions"/f"{record['record_id']}.json"; write(target, payload)
            result["records"].append({"record_id": record["record_id"], "role": record["role"],
                                      "prediction_file": str(target.relative_to(out)), "sha256": sha(target)})
            write(out/"results.json", result)
            print(json.dumps({"record_id": record["record_id"], "references": len(record["boxes"]),
                              "primary_counts": {name: len([p for p in rows if p["raw_score"] >= primary])
                                                 for name, rows in arms.items()}}), flush=True)
        # Recompute aggregate metrics from the immutable per-image prediction files.
        metrics = {}
        for arm in cfg["arms"]:
            metrics[arm] = {}
            for score in cfg["operating_scores"]:
                metrics[arm][str(score)] = {}
                for threshold in cfg["evaluation_ious"]:
                    totals = {"tp": 0, "fp": 0, "fn": 0}; per_image = []
                    for row in result["records"]:
                        payload = load(out/row["prediction_file"])
                        if arm == "proposal_fusion":
                            e = [p for p in payload["arms"]["epri_full_plus_tiles"] if p["raw_score"] >= score]
                            m = [p for p in payload["arms"]["mpid_full_plus_tiles"] if p["raw_score"] >= score]
                            predictions = fixed_priority_fusion(e, m, cfg["nms_iou"])
                        else:
                            predictions = [p for p in payload["arms"][arm] if p["raw_score"] >= score]
                        counts = match_counts(predictions, payload["references"], threshold)
                        for key in totals: totals[key] += counts[key]
                        per_image.append({"record_id": row["record_id"], "role": row["role"],
                                          **{k: counts[k] for k in ("tp", "fp", "fn")}})
                    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
                    metrics[arm][str(score)][str(threshold)] = {
                        **totals, "precision": tp/(tp+fp) if tp+fp else 0.,
                        "recall": tp/(tp+fn) if tp+fn else 0.,
                        "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., "per_image": per_image}
        result.update(status="COMPLETE", completed_at=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.perf_counter()-started, metrics=metrics,
                      integrity={"reference_boxes_used_for_inference": False,
                                 "training_or_parameter_updates": 0,
                                 "thresholds_selected_from_acceptance_results": False,
                                 "mpid_material_classes_scored_as_uk_material_truth": False,
                                 "cross_model_scores_compared_in_fusion": False})
        write(out/"results.json", result)
        print(json.dumps({"status": result["status"], "metrics": metrics}, indent=2), flush=True)
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter()-started)
        write(out/"results.json", result); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/uk_insulator_localisation_prospective_v1.json")
    main(parser.parse_args().config)
