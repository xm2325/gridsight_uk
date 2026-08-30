#!/usr/bin/env python3
"""Frozen UK multi-component inference for an evidence-separated Keen-style overlay."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from material_head_v2_common import decide_v2
from roihu_demo_ablation import nms as box_nms

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


def extent(box, width, height, padding):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * padding, (y1 - y0) * padding
    return [max(0, math.floor(x0 - dx)), max(0, math.floor(y0 - dy)),
            min(width, math.ceil(x1 + dx)), min(height, math.ceil(y1 + dy))]


def box_pole_top_region(poles, crossarms, width, height):
    """Derive an unscored pole-axis endpoint ROI or abstain.

    The result is deliberately a search region, never a physical pole-top label.
    """
    unknown = {"status": "unknown", "label": "pole-top search region", "xyxy": None,
               "score": None, "derived": True, "physical_component_verified": False}
    candidates = []
    for pole_index, pole in enumerate(poles):
        x0, y0, x1, y1 = pole["xyxy"]
        pole_width, pole_height = x1 - x0, y1 - y0
        if pole_width <= 0 or pole_height < 1.8 * pole_width:
            continue
        endpoints = [((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1)]
        for crossarm_index, crossarm in enumerate(crossarms):
            ax0, ay0, ax1, ay1 = crossarm["xyxy"]
            centre = ((ax0 + ax1) / 2, (ay0 + ay1) / 2)
            distances = [math.dist(endpoint, centre) for endpoint in endpoints]
            chosen = min(range(2), key=distances.__getitem__)
            near, far = distances[chosen], distances[1 - chosen]
            if near > .45 * pole_height or far < max(1.5 * near, near + .15 * pole_height):
                continue
            point = endpoints[chosen]
            if min(point[0], point[1], width - point[0], height - point[1]) <= 2:
                continue
            candidates.append((near / pole_height, pole_index, crossarm_index, chosen,
                               point, pole_width, pole_height, crossarm))
    if not candidates:
        return {**unknown, "reason": "no unambiguous elongated-pole endpoint near a crossarm"}
    _, pole_index, crossarm_index, endpoint_index, point, pole_width, pole_height, crossarm = min(candidates)
    ax0, ay0, ax1, ay1 = crossarm["xyxy"]
    side = max(24., 3 * pole_width, 1.5 * (ay1 - ay0), .05 * min(width, height))
    x0, y0 = max(0., point[0] - side / 2), max(0., point[1] - side / 2)
    x1, y1 = min(float(width), point[0] + side / 2), min(float(height), point[1] + side / 2)
    return {"status": "geometry_candidate", "label": "pole-top search region",
            "xyxy": [x0, y0, x1, y1], "point_xy": list(point), "score": None,
            "derived": True, "physical_component_verified": False,
            "source_pole_prediction_index": pole_index,
            "source_crossarm_prediction_index": crossarm_index,
            "selected_axis_endpoint": "first" if endpoint_index == 0 else "second",
            "reason": "elongated pole-box axis endpoint nearest a crossarm; physical component unverified"}


def head_from(saved, torch):
    head = torch.nn.Sequential(torch.nn.Linear(768, 256), torch.nn.GELU(),
                               torch.nn.Linear(256, 4)).to("cuda")
    head.load_state_dict({name: torch.tensor(saved[name], device="cuda")
                          for name in ("0.weight", "0.bias", "2.weight", "2.bias")})
    return head.eval()


def verify_release(directory, expected_model_sha):
    if sha(directory / "model.safetensors") != expected_model_sha:
        raise ValueError(f"Model hash mismatch: {directory}")
    manifest = load(directory / "verified_manifest.json")
    for record in manifest["files"]:
        if sha(directory / record["file"]) != record["sha256"]:
            raise ValueError(f"Release file changed: {directory / record['file']}")


def verify_protocol(config_name):
    """Read-only hash and boundary preflight; performs no model inference."""
    config_path = ROOT / config_name
    cfg = load(config_path)
    source = ROOT / cfg["source_dataset"]
    insulator_run = ROOT / cfg["preserved_insulator_run"]
    pins = [
        (source / "manifest.json", cfg["source_manifest_sha256"]),
        (insulator_run / "results.json", cfg["preserved_insulator_result_sha256"]),
        (ROOT / cfg["component_detector"], cfg["component_detector_sha256"]),
        (ROOT / cfg["material_config"], cfg["material_config_sha256"]),
        (ROOT / cfg["material_head"], cfg["material_head_sha256"]),
        (ROOT / cfg["material_threshold_source"], cfg["material_threshold_source_sha256"]),
    ]
    for path, expected in pins:
        if sha(path) != expected:
            raise ValueError(f"Pinned input changed: {path}")
    verify_release(ROOT / cfg["material_encoder"], cfg["material_encoder_sha256"])
    verify_release(ROOT / cfg["grounding_model"], cfg["grounding_model_sha256"])
    manifest = load(source / "manifest.json")
    if (not manifest["selection_frozen_before_v2_adapted_model_inference"] or
            manifest["model_inference_performed_before_freeze"]):
        raise ValueError("UK v3 boundary is not frozen")
    records = [record for record in manifest["records"] if record["role"] != "excluded"]
    if (len(records), len({record["asset_group"] for record in records})) != (9, 9):
        raise ValueError("Frozen UK v3 image/group counts changed")
    for record in records:
        if sha(ROOT / record["image_file"]) != record["image_sha256"]:
            raise ValueError(f"Source image changed: {record['record_id']}")
    insulator_result = load(insulator_run / "results.json")
    if (insulator_result["status"] != "COMPLETE" or
            insulator_result["integrity"]["acceptance_used_for_training_or_checkpoint_selection"] or
            insulator_result["integrity"]["acceptance_inference_passes_per_checkpoint"] != 1):
        raise ValueError("Preserved v2 acceptance inference is not valid")
    predictions = {row["record_id"]: row for row in
                   insulator_result["acceptance_predictions"]["adapted_specialist"]}
    if set(predictions) != {record["record_id"] for record in records}:
        raise ValueError("Preserved insulator predictions do not cover UK v3 exactly")
    for record_id, row in predictions.items():
        prediction_path = insulator_run / row["prediction_file"]
        if sha(prediction_path) != row["sha256"]:
            raise ValueError(f"Preserved insulator prediction changed: {record_id}")
    material_cfg = load(ROOT / cfg["material_config"])
    material_result = load(ROOT / cfg["material_threshold_source"])
    if material_cfg["classes"] != cfg["material_policy"]["classes"]:
        raise ValueError("Material class mapping changed")
    if material_result["adapted_head_sha256"] != cfg["material_head_sha256"]:
        raise ValueError("Material head/result provenance changed")
    if (ROOT / cfg["run"]).exists():
        raise FileExistsError("Existing run: inspect it instead of submitting or rerunning")
    return {"status": "PROTOCOL_VERIFIED", "images": len(records),
            "asset_groups": len({record["asset_group"] for record in records}),
            "preserved_insulator_predictions": len(predictions), "gradient_steps": 0,
            "v3_reference_boxes_accessed_or_used": False, "output_exists": False}


def main(config_name):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "gputest":
        raise RuntimeError("Requires Roihu gputest; no local model fallback")
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    import ultralytics
    from PIL import Image
    from transformers import AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor
    from ultralytics import YOLOE

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    preflight = verify_protocol(config_name)
    config_path = ROOT / config_name
    cfg = load(config_path)
    source = ROOT / cfg["source_dataset"]
    source_manifest_path = source / "manifest.json"
    insulator_run = ROOT / cfg["preserved_insulator_run"]
    material_cfg_path = ROOT / cfg["material_config"]
    material_head_path = ROOT / cfg["material_head"]
    material_result_path = ROOT / cfg["material_threshold_source"]
    pins = [
        (source_manifest_path, cfg["source_manifest_sha256"]),
        (insulator_run / "results.json", cfg["preserved_insulator_result_sha256"]),
        (ROOT / cfg["component_detector"], cfg["component_detector_sha256"]),
        (material_cfg_path, cfg["material_config_sha256"]),
        (material_head_path, cfg["material_head_sha256"]),
        (material_result_path, cfg["material_threshold_source_sha256"]),
    ]
    for path, expected in pins:
        if sha(path) != expected:
            raise ValueError(f"Pinned input changed: {path}")
    verify_release(ROOT / cfg["material_encoder"], cfg["material_encoder_sha256"])
    verify_release(ROOT / cfg["grounding_model"], cfg["grounding_model_sha256"])

    manifest = load(source_manifest_path)
    if (not manifest["selection_frozen_before_v2_adapted_model_inference"] or
            manifest["model_inference_performed_before_freeze"]):
        raise ValueError("UK v3 boundary is not frozen")
    records = [record for record in manifest["records"] if record["role"] != "excluded"]
    if (len(records), len({r["asset_group"] for r in records})) != (9, 9):
        raise ValueError("Frozen UK v3 image/group counts changed")
    for record in records:
        if sha(ROOT / record["image_file"]) != record["image_sha256"]:
            raise ValueError(f"Source image changed: {record['record_id']}")

    insulator_result = load(insulator_run / "results.json")
    if (insulator_result["status"] != "COMPLETE" or
            insulator_result["integrity"]["acceptance_used_for_training_or_checkpoint_selection"] or
            insulator_result["integrity"]["acceptance_inference_passes_per_checkpoint"] != 1):
        raise ValueError("Preserved v2 acceptance inference is not valid")
    insulator_rows = {row["record_id"]: row for row in
                      insulator_result["acceptance_predictions"]["adapted_specialist"]}
    if set(insulator_rows) != {record["record_id"] for record in records}:
        raise ValueError("Preserved insulator predictions do not cover UK v3 exactly")

    out = ROOT / cfg["run"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "code").mkdir()
    snapshots = {}
    for path in (config_path, Path(__file__), ROOT / "scripts/material_head_v2_common.py",
                 ROOT / "scripts/uk_multicomponent_inference_v1.sbatch"):
        shutil.copy2(path, out / "code" / path.name)
        snapshots[str(path.relative_to(ROOT))] = sha(path)
    frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(config_path),
        "source_manifest_sha256": sha(source_manifest_path),
        "record_ids": [record["record_id"] for record in records],
        "asset_groups": [record["asset_group"] for record in records],
        "gradient_steps": 0,
        "v3_reference_boxes_accessed_or_used": False,
        "v3_roles_available_to_models": False,
        "threshold_or_model_selection_from_v3": False,
    }
    write(out / "frozen_choices.json", frozen)
    result = {
        "status": "COMPONENT_INFERENCE", "started_at": frozen["frozen_at"],
        "git_commit": os.environ.get("GRIDSIGHT_SUBMISSION_COMMIT"),
        "protocol": cfg, "protocol_sha256": sha(config_path),
        "source_snapshots": snapshots, "frozen_choices_sha256": sha(out / "frozen_choices.json"),
        "runtime": {"job_id": os.environ["SLURM_JOB_ID"], "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                    "transformers": transformers.__version__},
        "records": [], "gradient_steps": 0, "performance_metrics": None,
        "claim_boundary": cfg["claim_boundary"], "preflight": preflight
    }
    write(out / "results.json", result)
    started = time.perf_counter()
    try:
        torch.set_float32_matmul_precision("highest")
        images = {}
        for record in records:
            with Image.open(ROOT / record["image_file"]) as opened:
                images[record["record_id"]] = opened.convert("RGB")

        component_model = YOLOE(str(ROOT / cfg["component_detector"])).to("cuda:0")
        if component_model.names != dict(enumerate(cfg["component_classes"])):
            raise ValueError("Component detector class mapping changed")
        prepared = {}
        component_cfg = cfg["component_inference"]
        for record in records:
            record_id, image = record["record_id"], images[record["record_id"]]
            prediction = component_model.predict(
                image, imgsz=component_cfg["imgsz"], conf=component_cfg["raw_score_floor"],
                iou=component_cfg["nms_iou"], max_det=component_cfg["max_det"],
                device=0, half=False, verbose=False)[0]
            raw_components = []
            if prediction.boxes is not None:
                for index, (box, score, cls) in enumerate(zip(
                        prediction.boxes.xyxy.cpu().tolist(), prediction.boxes.conf.cpu().tolist(),
                        prediction.boxes.cls.cpu().tolist())):
                    class_id = int(cls)
                    raw_components.append({
                        "prediction_index": index, "class_id": class_id,
                        "class_name": cfg["component_classes"][class_id],
                        "xyxy": list(map(float, box)), "raw_score": float(score),
                        "source_model": "epri_component_detector_full_frame",
                        "calibrated_probability": False, "reference_truth": False,
                    })
            preserved_row = insulator_rows[record_id]
            preserved_path = insulator_run / preserved_row["prediction_file"]
            if sha(preserved_path) != preserved_row["sha256"]:
                raise ValueError(f"Preserved insulator prediction changed: {record_id}")
            preserved_payload = load(preserved_path)
            insulators = [prediction for prediction in preserved_payload["full_plus_tiles"]
                          if prediction["raw_score"] >= component_cfg["display_operating_score"]]
            poles = [{**prediction, "prediction_index": index} for index, prediction in enumerate(raw_components)
                     if prediction["class_name"] == "pole" and
                     prediction["raw_score"] >= component_cfg["display_operating_score"]]
            crossarms = [{**prediction, "prediction_index": index} for index, prediction in enumerate(raw_components)
                         if prediction["class_name"] == "crossarm" and
                         prediction["raw_score"] >= component_cfg["display_operating_score"]]
            prepared[record_id] = {
                "source": {"record_id": record_id, "photo_id": record["photo_id"],
                           "title": record["title"], "author": record["author"],
                           "photo_page_url": record["photo_page_url"], "licence": record["licence"],
                           "licence_url": record["licence_url"], "image_file": record["image_file"],
                           "image_sha256": record["image_sha256"], "width": record["width"],
                           "height": record["height"]},
                "raw_component_predictions": raw_components,
                "components": {"pole": poles, "crossarm": crossarms, "insulator": insulators},
                "preserved_insulator_prediction_file": str(preserved_path.relative_to(ROOT)),
                "preserved_insulator_prediction_sha256": sha(preserved_path),
                "pole_top": box_pole_top_region(poles, crossarms, image.width, image.height),
            }
        del component_model, prediction
        gc.collect(); torch.cuda.empty_cache()

        result["status"] = "STEELWORK_CANDIDATES"
        write(out / "results.json", result)
        grounding_processor = AutoProcessor.from_pretrained(
            ROOT / cfg["grounding_model"], local_files_only=True, trust_remote_code=False)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            ROOT / cfg["grounding_model"], local_files_only=True, trust_remote_code=False,
            use_safetensors=True, disable_custom_kernels=True).to("cuda").eval()
        steel_cfg = cfg["steelwork"]
        for record in records:
            record_id, image = record["record_id"], images[record["record_id"]]
            raw_candidates, raw_files = [], []
            for query_index, query in enumerate(steel_cfg["queries"]):
                batch = grounding_processor(images=image, text=query,
                                            size=steel_cfg["grounding_size"],
                                            return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    raw = grounding_model(**batch)
                logits = raw.logits[0].float().cpu().numpy()
                boxes = raw.pred_boxes[0].float().cpu().numpy()
                scores = raw.logits[0].sigmoid().amax(-1).float().cpu().numpy()
                raw_path = out / "model_raw" / record_id / f"steelwork_q{query_index}.npz"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(raw_path, token_logits=logits, boxes_cxcywh=boxes,
                                    input_ids=batch["input_ids"].cpu().numpy())
                raw_files.append({"file": str(raw_path.relative_to(out)), "sha256": sha(raw_path),
                                  "query": query})
                for index, (box, score) in enumerate(zip(boxes, scores)):
                    if score < steel_cfg["raw_score_floor"]:
                        continue
                    cx, cy, width, height = map(float, box)
                    xyxy = [max(0., (cx - width / 2) * image.width),
                            max(0., (cy - height / 2) * image.height),
                            min(float(image.width), (cx + width / 2) * image.width),
                            min(float(image.height), (cy + height / 2) * image.height)]
                    if xyxy[0] >= xyxy[2] or xyxy[1] >= xyxy[3]:
                        continue
                    raw_candidates.append({
                        "prediction_index": index, "class_id": 0, "box": xyxy,
                        "score": float(score), "query": query,
                        "query_index": index, "raw_file": str(raw_path.relative_to(out)),
                        "label": steel_cfg["final_label"],
                        "steel_composition_verified": False,
                        "calibrated_probability": False, "reference_truth": False,
                    })
            prepared[record_id]["raw_steelwork_candidates"] = raw_candidates
            prepared[record_id]["steelwork_candidates"] = [candidate for candidate in
                box_nms(raw_candidates, steel_cfg["nms_iou"])
                if candidate["score"] >= steel_cfg["display_operating_score"]]
            prepared[record_id]["steelwork_raw_files"] = raw_files
        del grounding_model, grounding_processor, batch, raw
        gc.collect(); torch.cuda.empty_cache()

        result["status"] = "MATERIAL_DIAGNOSTICS"
        write(out / "results.json", result)
        material_cfg = load(material_cfg_path)
        material_result = load(material_result_path)
        if cfg["material_policy"]["classes"] != material_cfg["classes"]:
            raise ValueError("Material class mapping changed")
        saved = np.load(material_head_path)
        head = head_from(saved, torch)
        centroids = saved["centroids"]
        thresholds = material_result["thresholds"]["adapted"]
        material_processor = AutoProcessor.from_pretrained(
            ROOT / cfg["material_encoder"], local_files_only=True, trust_remote_code=False)
        material_encoder = AutoModel.from_pretrained(
            ROOT / cfg["material_encoder"], local_files_only=True, trust_remote_code=False,
            use_safetensors=True).to("cuda").eval()
        crop_images, crop_meta = [], []
        for record in records:
            record_id, image = record["record_id"], images[record["record_id"]]
            for prediction_index, prediction in enumerate(prepared[record_id]["components"]["insulator"]):
                for view, padding in (("tight", 0.), ("context", cfg["material_policy"]["context_padding"])):
                    crop_box = extent(prediction["xyxy"], image.width, image.height, padding)
                    crop_meta.append({"record_id": record_id, "prediction_index": prediction_index,
                                      "view": view, "crop_xyxy": crop_box})
                    crop_images.append(image.crop(crop_box))
        feature_batches = []
        for begin in range(0, len(crop_images), 24):
            batch = material_processor(images=crop_images[begin:begin + 24],
                                       return_tensors="pt").to("cuda")
            with torch.inference_mode():
                features = material_encoder.get_image_features(**batch)
                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output
                feature_batches.append(F.normalize(features.float(), dim=-1).cpu().numpy())
        embeddings = np.concatenate(feature_batches) if feature_batches else np.zeros((0, 768), np.float32)
        similarities = embeddings @ centroids.T if len(embeddings) else np.zeros((0, len(material_cfg["classes"])))
        with torch.inference_mode():
            logits = (head(torch.tensor(embeddings, device="cuda")).cpu().numpy()
                      if len(embeddings) else np.zeros((0, len(material_cfg["classes"]))) )
        np.savez_compressed(out / "material_features.npz", embeddings=embeddings,
                            similarities=similarities, logits=logits)
        cursor = 0
        for record in records:
            record_id = record["record_id"]
            material_rows = []
            for prediction_index, prediction in enumerate(prepared[record_id]["components"]["insulator"]):
                decision = decide_v2(logits[cursor].tolist(), logits[cursor + 1].tolist(),
                                     similarities[cursor].tolist(), similarities[cursor + 1].tolist(),
                                     prediction["xyxy"], material_cfg, thresholds)
                material_rows.append({
                    "insulator_prediction_index": prediction_index,
                    "diagnostic_material": decision["material"],
                    "diagnostic_decision": decision,
                    "final_material": cfg["material_policy"]["final_material"],
                    "final_reason": cfg["material_policy"]["final_reason"],
                    "scores_are_probabilities": False,
                    "material_verified": False,
                    "tight_crop": crop_meta[cursor]["crop_xyxy"],
                    "context_crop": crop_meta[cursor + 1]["crop_xyxy"],
                    "raw": {"tight_logits": logits[cursor].tolist(),
                            "context_logits": logits[cursor + 1].tolist(),
                            "tight_similarity": similarities[cursor].tolist(),
                            "context_similarity": similarities[cursor + 1].tolist()},
                })
                cursor += 2
            prepared[record_id]["material"] = material_rows

        for record in records:
            record_id = record["record_id"]
            payload = {
                "schema": "gridsight-uk-multicomponent-v1",
                **prepared[record_id],
                "inference_contract": cfg["output_contract"],
                "v3_reference_boxes_accessed_or_used": False,
                "v3_role_written_to_output": False,
                "performance_metrics": None,
            }
            target = out / "records" / f"{record_id}.json"
            write(target, payload)
            result["records"].append({
                "record_id": record_id, "record_file": str(target.relative_to(out)),
                "record_sha256": sha(target),
                "counts": {"pole": len(payload["components"]["pole"]),
                           "crossarm": len(payload["components"]["crossarm"]),
                           "insulator": len(payload["components"]["insulator"]),
                           "material_diagnostics": len(payload["material"]),
                           "steelwork_candidates": len(payload["steelwork_candidates"]),
                           "pole_top_regions": int(payload["pole_top"]["status"] == "geometry_candidate")},
            })
        result.update(
            status="COMPLETE_UNSCORED_MULTICOMPONENT_DIAGNOSTIC",
            completed_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - started,
            material_features_sha256=sha(out / "material_features.npz"),
            integrity={"gradient_steps": 0, "v3_reference_boxes_accessed_or_used": False,
                       "v3_roles_available_to_models": False,
                       "threshold_or_model_selection_from_v3": False,
                       "outputs_are_calibrated_probabilities": False,
                       "multi_component_accuracy_computed": False,
                       "steel_composition_verified": False,
                       "pole_top_is_physical_component_detection": False},
        )
        write(out / "results.json", result)
        print(json.dumps({"status": result["status"], "records": result["records"]}, indent=2), flush=True)
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started)
        write(out / "results.json", result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/uk_multicomponent_inference_v1.json")
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify_only:
        print(json.dumps(verify_protocol(arguments.config), indent=2))
    else:
        main(arguments.config)
