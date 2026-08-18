from __future__ import annotations

import argparse
import json

from v23_common import REPORTS, ROOT, WEIGHTS, dataset_manifest, runtime_env, sha256, write_json
from v25_hybrid_component_detector import (
    DISPLAY,
    LOCAL_TO_GLOBAL,
    TOWER_PROMPT,
    crop_training_split,
    expand,
    greedy,
    iou,
    read_yolo,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    from PIL import Image, ImageDraw, ImageFont
    from ultralytics import YOLO, YOLOE

    crop_root = ROOT / "data/v27_component_crop"
    train_manifest = crop_training_split("train", crop_root)
    val_manifest = crop_training_split("val", crop_root)
    (crop_root / "data.yaml").write_text(
        "path: data/v27_component_crop\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: crossarm\n"
        "  1: insulator_string\n"
        "  2: earthwire_peak\n"
    )
    if len(train_manifest) != 5 or len(val_manifest) != 1:
        raise RuntimeError(f"Expected 5 train / 1 val sources, got {len(train_manifest)} / {len(val_manifest)}")
    if sum(x["n_components"] for x in train_manifest) != 50:
        raise RuntimeError(f"Expected 50 train component boxes, got {train_manifest}")

    # Fixed held-out source: model-generated YOLOE tower ROI only.
    test_path = ROOT / "data/images/test/POS_2326530.jpg"
    image = Image.open(test_path).convert("RGB")
    W, H = image.size
    yoloe_weight = WEIGHTS / "yoloe-26n-seg.pt"
    yolo26_weight = WEIGHTS / "yolo26n.pt"
    yoloe = YOLOE(str(yoloe_weight))
    yoloe.set_classes([TOWER_PROMPT])
    r = yoloe.predict(str(test_path), imgsz=768, conf=0.05, device="cpu", verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        raise RuntimeError("YOLOE produced no test tower ROI")
    idx = int(r.boxes.conf.argmax().item())
    tower_box = [float(x) for x in r.boxes.xyxy[idx].cpu().tolist()]
    tower_score = float(r.boxes.conf[idx].item())
    roi = expand(tower_box, W, H)
    x1, y1, x2, y2 = [int(round(x)) for x in roi]
    crop = image.crop((x1, y1, x2, y2))
    cw, ch = crop.size
    test_crop = crop_root / "images/test/POS_2326530.jpg"
    test_crop.parent.mkdir(parents=True, exist_ok=True)
    crop.save(test_crop, quality=95)

    model = YOLO(str(yolo26_weight))
    model.train(
        data=str(crop_root / "data.yaml"),
        epochs=args.epochs,
        imgsz=640,
        batch=1,
        workers=0,
        device="cpu",
        seed=17,
        deterministic=True,
        pretrained=True,
        project=str(ROOT / "runs"),
        name="v2_7_yolo26n_component_crop_5sources",
        exist_ok=True,
        plots=False,
        verbose=True,
        mosaic=0.0,
        close_mosaic=0,
        translate=0.05,
        scale=0.2,
        fliplr=0.5,
        lr0=0.001,
        lrf=0.01,
        optimizer="AdamW",
    )
    best = ROOT / "runs/v2_7_yolo26n_component_crop_5sources/weights/best.pt"
    last = ROOT / "runs/v2_7_yolo26n_component_crop_5sources/weights/last.pt"
    chosen = best if best.exists() else last
    if not chosen.exists():
        raise RuntimeError("No trained checkpoint")

    pred = YOLO(str(chosen)).predict(str(test_crop), imgsz=640, conf=0.005, device="cpu", verbose=False)[0]
    predictions = []
    if pred.boxes is not None:
        for b, s, c in zip(pred.boxes.xyxy.cpu().tolist(), pred.boxes.conf.cpu().tolist(), pred.boxes.cls.cpu().tolist()):
            local = int(c)
            global_id = LOCAL_TO_GLOBAL[local]
            full = [float(b[0]) + x1, float(b[1]) + y1, float(b[2]) + x1, float(b[3]) + y1]
            predictions.append({"class_id": global_id, "label": DISPLAY[global_id], "score": float(s), "box": full})

    gt_all = read_yolo(ROOT / "data/labels/test/POS_2326530.txt", W, H)
    tower_gt = next(g for g in gt_all if g["class_id"] == 0)
    component_gt = [g for g in gt_all if g["class_id"] != 0]
    metrics = [greedy(predictions, component_gt, t) for t in (0.30, 0.50)]

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    colors = {1: (50, 120, 230), 2: (220, 60, 150), 3: (230, 145, 30)}
    draw.rectangle(tuple(int(v) for v in tower_box), outline=(35, 180, 75), width=3)
    draw.text((int(tower_box[0]), int(tower_box[1])), f"YOLOE tower {tower_score*100:.1f}%", fill=(35, 180, 75), font=font)
    for p in predictions:
        b = tuple(int(round(v)) for v in p["box"])
        color = colors[p["class_id"]]
        draw.rectangle(b, outline=color, width=3)
        draw.text((b[0], b[1]), f"{p['label']} {p['score']*100:.1f}%", fill=color, font=font)
    out_img = REPORTS / "v2_7_source_diversity_POS_2326530.jpg"
    canvas.save(out_img, quality=95)

    report = {
        "evidence_type": "source-diversity-ablation-yoloe-roi-plus-yolo26",
        "claim_scope": "exploratory 5-train/1-val/1-fixed-test source-isolated pilot; same detector/training family as v2.5; scores uncalibrated; no material/condition claim",
        "ablation": {
            "changed": "training source diversity only: 3 -> 5 independent UK tower sources; component boxes 30 -> 50 inside train tower crops",
            "held_constant": ["fixed validation POS_5442616", "fixed test POS_2326530", "YOLOE tower prompt", "YOLO26n pretrained checkpoint family", "80 epochs", "640 input", "AdamW", "seed 17", "tower-crop pipeline"],
            "historical_v25_raw_iou30": {"precision": 0.04411764705882353, "recall": 0.5, "f1": 0.0810810810810811},
            "historical_v25_raw_iou50": {"precision": 0.014705882352941176, "recall": 0.16666666666666666, "f1": 0.02702702702702703},
        },
        "epochs": args.epochs,
        "train_crop_manifest": train_manifest,
        "val_crop_manifest": val_manifest,
        "held_out": {
            "source": "POS_2326530",
            "source_dimensions": [W, H],
            "tower_prompt": TOWER_PROMPT,
            "tower_box": tower_box,
            "tower_score": tower_score,
            "tower_iou_vs_manual_reference": iou(tower_box, tower_gt["box"]),
            "model_generated_roi_xyxy": [x1, y1, x2, y2],
            "crop_dimensions": [cw, ch],
        },
        "n_train_sources": len(train_manifest),
        "n_train_component_boxes": sum(x["n_components"] for x in train_manifest),
        "n_component_gt": len(component_gt),
        "n_predictions": len(predictions),
        "predictions": predictions,
        "metrics": metrics,
        "weights": {
            "yoloe": {"sha256": sha256(yoloe_weight)},
            "yolo26_input": {"sha256": sha256(yolo26_weight)},
            "yolo26_trained": {"path": str(chosen.relative_to(ROOT)), "sha256": sha256(chosen)},
        },
        "runtime": runtime_env(),
        "dataset_manifest": dataset_manifest(),
    }
    write_json(REPORTS / "v2_7_source_diversity_metrics.json", report)
    print(json.dumps({"n_train_sources": len(train_manifest), "n_train_components": report["n_train_component_boxes"], "tower_iou": report["held_out"]["tower_iou_vs_manual_reference"], "n_predictions": len(predictions), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
