from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

from v23_common import REPORTS, ROOT, WEIGHTS, runtime_env, sha256, write_json
from v25_hybrid_component_detector import TOWER_PROMPT, expand, iou, read_yolo
from v34_tiled_insulator_specialist import metrics, nms
from v35_yoloe_aggregated_vpe import INSULATOR_PROMPT, build_prompt_model, infer
from v36_geometry_prior_yoloe import apply_prior, training_prior

PROTOCOL_PATH = ROOT / "configs/v38_final_holdout_protocol.json"
FREEZE_PATH = ROOT / "data/final_holdout/final_holdout_freeze.json"
HOLDOUT_IMAGE_DIR = ROOT / "data/final_holdout/images"
PREDICTION_BUNDLE = REPORTS / "v3_8_final_holdout_predictions_locked.json"
RESULT_PATH = REPORTS / "v3_8_final_holdout_metrics.json"


def read_json(path: Path):
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_exact(url: str, dest: Path, expected_sha256: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and file_sha256(dest) == expected_sha256:
        return {"path": str(dest.relative_to(ROOT)), "sha256": expected_sha256, "downloaded": False}
    headers = {"User-Agent": "GridSight-UK/3.8 final-holdout-evaluation (research portfolio; contact via GitHub repo)"}
    errors = []
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response:
                payload = response.read()
            dest.write_bytes(payload)
            got = file_sha256(dest)
            if got != expected_sha256:
                raise RuntimeError(f"SHA256 mismatch for {dest.name}: expected {expected_sha256}, got {got}")
            return {"path": str(dest.relative_to(ROOT)), "sha256": got, "bytes": len(payload), "downloaded": True, "attempt": attempt + 1}
        except Exception as exc:
            errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
            time.sleep(min(8, 2 ** attempt))
    raise RuntimeError(f"Failed exact holdout download {url}: {' | '.join(errors)}")


def detect_parent_and_raw(weight: Path, image_path: Path):
    from PIL import Image
    from ultralytics import YOLOE

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    tower_model = YOLOE(str(weight))
    tower_model.set_classes([TOWER_PROMPT])
    tr = tower_model.predict(str(image_path), imgsz=768, conf=0.05, device="cpu", verbose=False)[0]
    if tr.boxes is None or len(tr.boxes) == 0:
        raise RuntimeError(f"No tower ROI for final holdout {image_path.stem}")
    idx = int(tr.boxes.conf.argmax().item())
    tower_box = [float(x) for x in tr.boxes.xyxy[idx].cpu().tolist()]
    tower_score = float(tr.boxes.conf[idx].item())
    roi = expand(tower_box, w, h, px=0.08, py=0.04)
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    crop = image.crop((x1, y1, x2, y2))
    crop_path = REPORTS / f"v3_8_{image_path.stem}_tower_roi.jpg"
    crop.save(crop_path, quality=95)
    model = build_prompt_model(weight, "text", None)
    raw = infer(model, crop_path, (x1, y1, x2, y2), conf=0.001)
    return {
        "dimensions": [w, h],
        "tower_box": tower_box,
        "tower_score": tower_score,
        "tower_roi": [x1, y1, x2, y2],
        "raw": raw,
        "crop_path": str(crop_path.relative_to(ROOT)),
    }


def locked_filter(raw, tower_box, prior, protocol):
    champion = protocol["champion_operating_point"]
    baseline = protocol["secondary_frozen_baseline"]
    fused_all = apply_prior(raw, tower_box, prior)
    fused = nms([x for x in fused_all if x["score"] >= float(champion["fused_score_threshold"])], float(champion["nms_iou"]))
    baseline_rows = [{**x, "score": float(x["score"])} for x in raw]
    raw_filtered = nms([x for x in baseline_rows if x["score"] >= float(baseline["model_score_threshold"])], float(baseline["nms_iou"]))
    return raw_filtered, fused, fused_all


def render(image_path: Path, tower_box, rows, suffix: str):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    tb = tuple(int(round(v)) for v in tower_box)
    draw.rectangle(tb, outline=(35, 180, 75), width=3)
    for row in rows:
        box = tuple(int(round(v)) for v in row["box"])
        draw.rectangle(box, outline=(220, 60, 150), width=3)
        score = row.get("model_score", row.get("score", 0.0))
        draw.text((box[0], box[1]), f"insulator {score:.3f}", fill=(220, 60, 150), font=font)
    out = REPORTS / f"v3_8_{image_path.stem}_{suffix}.jpg"
    image.save(out, quality=95)
    return str(out.relative_to(ROOT))


def sum_metric(per_source, idx: int):
    tp = sum(row[idx]["tp"] for row in per_source)
    fp = sum(row[idx]["fp"] for row in per_source)
    fn = sum(row[idx]["fn"] for row in per_source)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"iou_threshold": [0.30, 0.50][idx], "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main():
    protocol = read_json(PROTOCOL_PATH)
    freeze = read_json(FREEZE_PATH)
    protocol_sha = sha256(PROTOCOL_PATH)
    freeze_sha = sha256(FREEZE_PATH)
    if not protocol.get("frozen_before_final_holdout_inference"):
        raise RuntimeError("Protocol not marked frozen")
    if protocol["final_holdout"]["holdout_release_sha256"] != freeze["holdout_release_sha256"]:
        raise RuntimeError("Holdout release mismatch")
    expected_ids = protocol["final_holdout"]["record_ids"]
    freeze_ids = [r["record_id"] for r in freeze["records"]]
    if expected_ids != freeze_ids:
        raise RuntimeError(f"Holdout IDs differ: protocol={expected_ids}, freeze={freeze_ids}")

    weight = WEIGHTS / protocol["model"]["name"]
    if not weight.exists():
        raise FileNotFoundError(weight)
    if sha256(weight) != protocol["model"]["sha256"]:
        raise RuntimeError("YOLOE checkpoint hash differs from frozen protocol")

    # Refit only the predeclared training-only prior. No validation or holdout labels are involved here.
    prior = training_prior()
    if prior["n"] != protocol["geometry_prior"]["n_train_insulators"]:
        raise RuntimeError("Training prior sample count differs from protocol")

    downloaded = []
    prediction_rows = []
    for record in freeze["records"]:
        rid = record["record_id"]
        dest = HOLDOUT_IMAGE_DIR / f"{rid}.jpg"
        downloaded.append(download_exact(record["image_url"], dest, record["image_sha256"]))
        pred = detect_parent_and_raw(weight, dest)
        raw_filtered, fused, fused_all = locked_filter(pred["raw"], pred["tower_box"], prior, protocol)
        prediction_rows.append({
            "record_id": rid,
            "image_sha256": file_sha256(dest),
            "dimensions": pred["dimensions"],
            "tower_box": pred["tower_box"],
            "tower_score": pred["tower_score"],
            "tower_roi": pred["tower_roi"],
            "n_raw": len(pred["raw"]),
            "raw_baseline_predictions": raw_filtered,
            "champion_predictions": fused,
            "all_geometry_scored_candidates": fused_all,
            "raw_render": render(dest, pred["tower_box"], raw_filtered, "raw_baseline"),
            "champion_render": render(dest, pred["tower_box"], fused, "geometry_champion"),
        })

    # Freeze prediction bundle on disk before any final-holdout label is opened.
    bundle = {
        "evidence_type": "locked-final-holdout-predictions-before-label-read",
        "protocol_sha256": protocol_sha,
        "freeze_file_sha256": freeze_sha,
        "checkpoint_sha256": sha256(weight),
        "downloads": downloaded,
        "records": prediction_rows,
        "labels_loaded": False,
    }
    write_json(PREDICTION_BUNDLE, bundle)
    prediction_bundle_sha = sha256(PREDICTION_BUNDLE)

    # Only now load final-holdout labels and score the already-frozen predictions.
    evaluated = []
    raw_metrics_all = []
    champion_metrics_all = []
    for record, preds in zip(freeze["records"], prediction_rows):
        rid = record["record_id"]
        label_path = ROOT / record["label_path"]
        if sha256(label_path) != record["label_sha256"]:
            raise RuntimeError(f"Label hash mismatch for {rid}")
        w, h = preds["dimensions"]
        labels = read_yolo(label_path, w, h)
        tower_gt = next(x for x in labels if x["class_id"] == 0)
        ins_gt = [x["box"] for x in labels if x["class_id"] == 2]
        raw_m = [metrics(preds["raw_baseline_predictions"], ins_gt, t) for t in (0.30, 0.50)]
        champion_m = [metrics(preds["champion_predictions"], ins_gt, t) for t in (0.30, 0.50)]
        raw_metrics_all.append(raw_m); champion_metrics_all.append(champion_m)
        evaluated.append({
            "record_id": rid,
            "n_insulator_gt": len(ins_gt),
            "tower_iou_vs_reference": iou(preds["tower_box"], tower_gt["box"]),
            "raw_baseline": {"n_predictions": len(preds["raw_baseline_predictions"]), "metrics": raw_m},
            "champion": {"n_predictions": len(preds["champion_predictions"]), "metrics": champion_m},
        })

    result = {
        "evidence_type": "first-and-locked-final-holdout-evaluation",
        "protocol_commit_precedes_inference": True,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "freeze_path": str(FREEZE_PATH.relative_to(ROOT)),
        "freeze_file_sha256": freeze_sha,
        "prediction_bundle_path": str(PREDICTION_BUNDLE.relative_to(ROOT)),
        "prediction_bundle_sha256_before_label_read": prediction_bundle_sha,
        "labels_loaded_after_prediction_bundle": True,
        "records": evaluated,
        "pooled": {
            "n_towers": len(evaluated),
            "n_insulator_gt": sum(r["n_insulator_gt"] for r in evaluated),
            "raw_baseline": [sum_metric(raw_metrics_all, i) for i in range(2)],
            "champion": [sum_metric(champion_metrics_all, i) for i in range(2)],
        },
        "post_holdout_rule": "Do not tune v3.6 champion using these final-holdout outcomes if reporting them as final-holdout evidence. Any future method revision must be versioned as a new development cycle and evaluated on a newly frozen holdout.",
        "claim_boundary": protocol["claim_boundary"],
        "runtime": runtime_env(),
    }
    write_json(RESULT_PATH, result)
    print(json.dumps(result["pooled"], indent=2))


if __name__ == "__main__":
    main()
