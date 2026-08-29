"""Score preserved box/mask outputs against a publisher-mask-derived pole endpoint.

This does not create a physical pole-top annotation.  It freezes a reproducible
geometry target on the EPRI development split and keeps abstentions visible.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from component_mask_metrics import decode_masks, mask_iou, pole_end_candidate, raster_polygon, rectangle_mask

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scaled_boxes(refs, source_size, working_size, class_id):
    sw, sh = source_size
    w, h = working_size
    return [[b * s for b, s in zip(r["box"], [w / sw, h / sh, w / sw, h / sh])]
            for r in refs if r["class_id"] == class_id]


def best_pole(predictions, masks, target, threshold):
    candidates = [(mask_iou(masks[p["prediction_index"]], target), p)
                  for p in predictions if p["class_id"] == 0 and p["score"] >= threshold]
    return max(candidates, default=(0.0, None), key=lambda item: item[0])


def endpoint_from_box_predictions(predictions, source_size, working_size, threshold):
    sw, sh = source_size
    w, h = working_size
    scale = [w / sw, h / sh, w / sw, h / sh]
    poles = [p for p in predictions if p["class_id"] == 0 and p["score"] >= threshold]
    arms = [[x * s for x, s in zip(p["box"], scale)] for p in predictions
            if p["class_id"] == 1 and p["score"] >= threshold]
    if not poles:
        return None, "no pole above threshold", 0.0
    # The source target is never used to select among predictions. Highest score is fixed.
    pole = max(poles, key=lambda p: p["score"])
    box = [x * s for x, s in zip(pole["box"], scale)]
    result = pole_end_candidate(rectangle_mask(box, working_size), arms)
    return result, result["reason"], pole["score"]


def summarize(records, arm):
    eligible = len(records)
    accepted = [r for r in records if r[arm]["status"] == "accepted"]
    errors = [r[arm]["normalized_error"] for r in accepted]
    return {
        "eligible_targets": eligible,
        "accepted": len(accepted),
        "coverage": len(accepted) / eligible if eligible else 0.0,
        "median_normalized_error": float(np.median(errors)) if errors else None,
        "pck@0.25": sum(e <= 0.25 for e in errors) / eligible if eligible else 0.0,
        "pck@0.50": sum(e <= 0.50 for e in errors) / eligible if eligible else 0.0,
        "conditional_pck@0.25": sum(e <= 0.25 for e in errors) / len(errors) if errors else None,
        "conditional_pck@0.50": sum(e <= 0.50 for e in errors) / len(errors) if errors else None,
    }


def decision(candidate, target_point, normalizer, extra=None):
    if not candidate or candidate.get("status") != "geometry_candidate":
        return {"status": "abstained", "reason": (candidate or {}).get("reason", "no candidate"),
                **(extra or {})}
    point = np.asarray(candidate["point"], dtype=float)
    error = float(np.linalg.norm(point - target_point))
    return {"status": "accepted", "point": point.tolist(), "pixel_error": error,
            "normalized_error": error / normalizer, "reason": candidate["reason"], **(extra or {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pole_top_keypoint_v1.json")
    args = ap.parse_args()
    cfg_path = ROOT / args.config
    cfg = load(cfg_path)
    manifest_path = ROOT / cfg["source_manifest"]
    mask_root = ROOT / cfg["mask_predictions"]
    box_root = ROOT / cfg["box_predictions"]
    assert sha(manifest_path) == cfg["source_manifest_sha256"]
    assert sha(mask_root / "results.json") == cfg["mask_results_sha256"]
    # Parent result pins the separate box-prediction files and their protocol.
    box_result = ROOT / "runs/keen_components/epri_components_v1_20260827/results.json"
    assert sha(box_result) == cfg["box_results_sha256"]
    manifest = load(manifest_path)
    rows = [r for r in manifest["images"] if r["split"] == cfg["split"]]
    records, exclusions = [], []
    for row in rows:
        refs = row["references"]
        poles = [r for r in refs if r["class_id"] == 0]
        arms = [r for r in refs if r["class_id"] == 1]
        if len(poles) != 1 or not arms:
            exclusions.append({"image_id": row["image_id"], "reason": "requires exactly one pole and at least one crossarm"})
            continue
        mp = load(mask_root / "predictions" / row["image_id"] / "predictions.json")
        assert mp["source_image_sha256"] == row["sha256"]
        source_size, working_size = mp["source_size"], mp["working_size"]
        gt_mask = raster_polygon(poles[0]["polygon"], source_size, working_size)
        gt_arms = scaled_boxes(refs, source_size, working_size, 1)
        gt = pole_end_candidate(gt_mask, gt_arms)
        if gt["status"] != "geometry_candidate":
            exclusions.append({"image_id": row["image_id"], "reason": "publisher geometry: " + gt["reason"]})
            continue
        target_point = np.asarray(gt["point"], dtype=float)
        normalizer = math.sqrt(float(np.count_nonzero(gt_mask)))
        with np.load(mask_root / "predictions" / row["image_id"] / mp["raw_file"], allow_pickle=False) as raw:
            masks = decode_masks(raw)
        iou, pole_pred = best_pole(mp["predictions"], masks, gt_mask, cfg["score_threshold"])
        pred_arms = [p["box_working"] for p in mp["predictions"]
                     if p["class_id"] == 1 and p["score"] >= cfg["score_threshold"]]
        if pole_pred is None or iou < cfg["mask_match_iou"]:
            mask_decision = {"status": "abstained", "reason": "no pole mask matched publisher pole at frozen IoU",
                             "best_mask_iou": iou}
        else:
            candidate = pole_end_candidate(masks[pole_pred["prediction_index"]], pred_arms)
            mask_decision = decision(candidate, target_point, normalizer,
                                     {"best_mask_iou": iou, "pole_score": pole_pred["score"]})
        bp_path = box_root / f'{row["image_id"]}.json'
        bp = load(bp_path)
        assert bp["image_sha256"] == row["sha256"]
        box_candidate, _, box_score = endpoint_from_box_predictions(
            bp["predictions"], source_size, working_size, cfg["score_threshold"])
        box_decision = decision(box_candidate, target_point, normalizer, {"pole_score": box_score})
        records.append({"image_id": row["image_id"], "source_image_sha256": row["sha256"],
                        "working_size": working_size, "target_point": target_point.tolist(),
                        "normalizer": normalizer, "target_origin": "publisher polygon geometry",
                        "mask_model": mask_decision, "box_model": box_decision})
    out = ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "COMPLETE_DEVELOPMENT_GEOMETRY_AUDIT",
        "config": cfg,
        "config_sha256": sha(cfg_path),
        "source_manifest_sha256": sha(manifest_path),
        "target_is_physical_tip_annotation": False,
        "target_is_model_pseudo_label": False,
        "target_is_publisher_mask_derived": True,
        "uk_accuracy_claim": False,
        "records": records,
        "exclusions": exclusions,
        "summary": {"mask_model": summarize(records, "mask_model"),
                    "box_model": summarize(records, "box_model")},
    }
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
