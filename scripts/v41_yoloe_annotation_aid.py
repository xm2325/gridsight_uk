from __future__ import annotations

import json
from pathlib import Path

from v23_common import REPORTS, ROOT, WEIGHTS, runtime_env, sha256, write_json
from v25_hybrid_component_detector import TOWER_PROMPT, expand

SELECTED = ["POS_6610209", "POS_8091164"]
PROMPTS = {
    1: "crossarm of an electricity transmission tower",
    2: "insulator string on an electricity transmission tower",
    3: "earth wire peak at the top of an electricity transmission tower",
}
DISPLAY = {1: "crossarm", 2: "insulator", 3: "earthwire peak"}
COLORS = {1: (50, 120, 230), 2: (220, 60, 150), 3: (230, 145, 30)}


def source_path(record_id: str) -> Path:
    path = ROOT / f"reports/v4_0_morphology_candidates/{record_id}.jpg"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def run_one(record_id: str, weight: Path):
    from PIL import Image, ImageDraw, ImageFont
    from ultralytics import YOLOE

    path = source_path(record_id)
    image = Image.open(path).convert("RGB")
    w, h = image.size

    tower_model = YOLOE(str(weight))
    tower_model.set_classes([TOWER_PROMPT])
    tower_result = tower_model.predict(str(path), imgsz=768, conf=0.05, device="cpu", verbose=False)[0]
    if tower_result.boxes is None or len(tower_result.boxes) == 0:
        raise RuntimeError(f"No tower proposal for {record_id}")
    idx = int(tower_result.boxes.conf.argmax().item())
    tower_box = [float(x) for x in tower_result.boxes.xyxy[idx].cpu().tolist()]
    tower_score = float(tower_result.boxes.conf[idx].item())
    roi = expand(tower_box, w, h, px=0.10, py=0.06)
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    crop = image.crop((x1, y1, x2, y2))
    crop_path = REPORTS / f"v4_1_{record_id}_tower_roi.jpg"
    crop.save(crop_path, quality=95)

    predictions = []
    per_class = {}
    for class_id, prompt in PROMPTS.items():
        model = YOLOE(str(weight))
        model.set_classes([prompt])
        result = model.predict(str(crop_path), imgsz=960, conf=0.01, iou=0.35, max_det=25, device="cpu", verbose=False)[0]
        rows = []
        masks = None if result.masks is None else result.masks.data.cpu().numpy()
        if result.boxes is not None:
            for k, (box, score) in enumerate(zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())):
                row = {
                    "class_id": class_id,
                    "label": DISPLAY[class_id],
                    "prompt": prompt,
                    "score": float(score),
                    "box": [float(box[0]) + x1, float(box[1]) + y1, float(box[2]) + x1, float(box[3]) + y1],
                    "mask_pixels_at_predictor_resolution": int((masks[k] > 0.5).sum()) if masks is not None and k < len(masks) else None,
                }
                rows.append(row)
                predictions.append(row)
        per_class[DISPLAY[class_id]] = {"prompt": prompt, "n_predictions": len(rows), "predictions": rows}

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    tb = tuple(int(round(v)) for v in tower_box)
    draw.rectangle(tb, outline=(35, 180, 75), width=3)
    draw.text((tb[0], tb[1]), f"tower {tower_score:.3f}", fill=(35, 180, 75), font=font)
    for row in predictions:
        b = tuple(int(round(v)) for v in row["box"])
        color = COLORS[row["class_id"]]
        draw.rectangle(b, outline=color, width=3)
        draw.text((b[0], b[1]), f"{row['label']} {row['score']:.3f}", fill=color, font=font)
    out = REPORTS / f"v4_1_{record_id}_annotation_aid.jpg"
    canvas.save(out, quality=95)
    return {
        "record_id": record_id,
        "source_path": str(path.relative_to(ROOT)),
        "dimensions": [w, h],
        "tower": {"box": tower_box, "score": tower_score, "prompt": TOWER_PROMPT},
        "roi_xyxy": [x1, y1, x2, y2],
        "per_class": per_class,
        "n_total_component_proposals": len(predictions),
        "overlay": str(out.relative_to(ROOT)),
        "ground_truth_status": "none; annotation_aid_only",
    }


def main():
    weight = WEIGHTS / "yoloe-26n-seg.pt"
    if not weight.exists():
        raise FileNotFoundError(weight)
    rows = [run_one(rid, weight) for rid in SELECTED]
    report = {
        "version": "v4.1-yoloe-annotation-aid",
        "evidence_type": "model-proposals-for-assistant-visual-review-only",
        "claim_scope": "No proposal is ground truth until visually accepted/corrected; no performance metric; no retired-v3.8 holdout inference.",
        "selected_after_pixel_review": SELECTED,
        "weight": {"name": weight.name, "sha256": sha256(weight)},
        "records": rows,
        "runtime": runtime_env(),
        "retired_v3_8_holdout_used": False,
    }
    write_json(REPORTS / "v4_1_yoloe_annotation_aid.json", report)
    print(json.dumps({r["record_id"]: {k: v["n_predictions"] for k, v in r["per_class"].items()} for r in rows}, indent=2))


if __name__ == "__main__":
    main()
