from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from v23_common import REPORTS, ROOT, WEIGHTS, runtime_env, sha256, write_json
from v25_hybrid_component_detector import TOWER_PROMPT, expand, iou, read_yolo
from v34_tiled_insulator_specialist import metrics, nms

INSULATOR_CLASS_ID = 2
INSULATOR_PROMPT = "insulator string on an electricity transmission tower"
TRAIN_SOURCES = ["POS_1283842", "POS_190181", "POS_291727", "POS_3778704", "POS_7060068"]
VAL_SOURCE = "POS_5442616"
DEV_SOURCE = "POS_2326530"


def split_path(source: str) -> tuple[Path, Path]:
    for split in ("train", "val", "test"):
        image = ROOT / f"data/images/{split}/{source}.jpg"
        label = ROOT / f"data/labels/{split}/{source}.txt"
        if image.exists() and label.exists():
            return image, label
    raise FileNotFoundError(source)


def insulator_boxes(source: str):
    from PIL import Image

    image, label = split_path(source)
    with Image.open(image) as im:
        w, h = im.size
    rows = read_yolo(label, w, h)
    boxes = [r["box"] for r in rows if r["class_id"] == INSULATOR_CLASS_ID]
    if not boxes:
        raise RuntimeError(f"No insulator boxes for {source}")
    return image, boxes, (w, h)


def extract_single_source_vpe(weight: Path, source: str):
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    image, boxes, _ = insulator_boxes(source)
    model = YOLOE(str(weight))
    prompts = {
        "bboxes": np.asarray(boxes, dtype=np.float32),
        "cls": np.zeros(len(boxes), dtype=np.int64),
    }
    # Prediction output is intentionally ignored. This call uses Ultralytics' official
    # visual-prompt predictor to extract VPE from the reference image and set it on model.model.
    model.predict(
        str(image),
        refer_image=str(image),
        visual_prompts=prompts,
        predictor=YOLOEVPSegPredictor,
        imgsz=960,
        conf=0.99,
        device="cpu",
        verbose=False,
        max_det=1,
    )
    pe = getattr(model.model, "pe", None)
    if not isinstance(pe, torch.Tensor) or pe.ndim != 3 or pe.shape[0] != 1 or pe.shape[1] != 1:
        raise RuntimeError(f"Unexpected VPE shape for {source}: {None if pe is None else tuple(pe.shape)}")
    pe = F.normalize(pe.detach().cpu().float(), dim=-1, p=2)
    return pe, len(boxes)


def aggregate_vpes(rows):
    # Ultralytics' validator aggregates class VPE contributions across samples and then L2-normalizes.
    # Single-reference VPEs are already one vector/class, so we use box-count weighting as the closest
    # source-level equivalent to per-instance contribution weighting.
    num = sum(pe * float(n) for _, pe, n in rows)
    den = float(sum(n for _, _, n in rows))
    return F.normalize(num / den, dim=-1, p=2)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


def model_generated_tower_crop(weight: Path, source: str, suffix: str):
    from PIL import Image
    from ultralytics import YOLOE

    image_path, label_path = split_path(source)
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    tower_model = YOLOE(str(weight))
    tower_model.set_classes([TOWER_PROMPT])
    result = tower_model.predict(str(image_path), imgsz=768, conf=0.05, device="cpu", verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError(f"No tower ROI for {source}")
    idx = int(result.boxes.conf.argmax().item())
    tower_box = [float(x) for x in result.boxes.xyxy[idx].cpu().tolist()]
    tower_score = float(result.boxes.conf[idx].item())
    roi = expand(tower_box, w, h, px=0.08, py=0.04)
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    crop = image.crop((x1, y1, x2, y2))
    crop_path = REPORTS / f"v3_5_{suffix}_{source}_tower_roi.jpg"
    crop.save(crop_path, quality=95)
    gt = read_yolo(label_path, w, h)
    tower_gt = next(x for x in gt if x["class_id"] == 0)
    insulator_gt = [x["box"] for x in gt if x["class_id"] == INSULATOR_CLASS_ID]
    return crop_path, (x1, y1, x2, y2), tower_box, tower_score, tower_gt["box"], insulator_gt, (w, h)


def build_prompt_model(weight: Path, mode: str, vpe: torch.Tensor | None):
    from ultralytics import YOLOE

    model = YOLOE(str(weight))
    if mode == "text":
        model.set_classes([INSULATOR_PROMPT])
    else:
        if vpe is None:
            raise ValueError(mode)
        model.set_classes(["insulator"], vpe.to(next(model.model.parameters()).device))
    return model


def infer(model, crop_path: Path, roi, conf=0.001):
    x1, y1, _, _ = roi
    result = model.predict(str(crop_path), imgsz=960, conf=conf, device="cpu", verbose=False, max_det=300)[0]
    rows = []
    if result.boxes is not None:
        masks = None if result.masks is None else result.masks.data.cpu().numpy()
        for k, (b, s) in enumerate(zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())):
            rows.append(
                {
                    "score": float(s),
                    "box": [float(b[0]) + x1, float(b[1]) + y1, float(b[2]) + x1, float(b[3]) + y1],
                    "mask_pixels_at_predictor_resolution": int((masks[k] > 0.5).sum()) if masks is not None and k < len(masks) else None,
                }
            )
    return rows


def choose_operating_point(raw, gt):
    confs = sorted(
        set(
            [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020, 0.030, 0.050, 0.075, 0.10, 0.15, 0.20]
            + [round(x["score"], 6) for x in raw]
        )
    )
    nmses = [0.20, 0.25, 0.30, 0.40, 0.50]
    sweep = []
    for conf in confs:
        for ni in nmses:
            filt = nms([x for x in raw if x["score"] >= conf], ni)
            m = metrics(filt, gt, 0.30)
            sweep.append({"conf_threshold": conf, "nms_iou": ni, "n_predictions": len(filt), **{k: v for k, v in m.items() if k != "matches"}})
    best = max(sweep, key=lambda x: (x["f1"], x["recall"], x["precision"], -x["n_predictions"], x["conf_threshold"], -x["nms_iou"]))
    return best, sweep


def render(source: str, mode: str, predictions, tower_box, tower_score):
    from PIL import Image, ImageDraw, ImageFont

    image_path, _ = split_path(source)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    tb = tuple(int(round(v)) for v in tower_box)
    draw.rectangle(tb, outline=(35, 180, 75), width=3)
    draw.text((tb[0], tb[1]), f"tower ROI {tower_score:.3f}", fill=(35, 180, 75), font=font)
    for p in predictions:
        b = tuple(int(round(v)) for v in p["box"])
        draw.rectangle(b, outline=(220, 60, 150), width=3)
        draw.text((b[0], b[1]), f"insulator {p['score']:.3f}", fill=(220, 60, 150), font=font)
    out = REPORTS / f"v3_5_{mode}_{source}.jpg"
    image.save(out, quality=95)
    return out


def main():
    weight = WEIGHTS / "yoloe-26n-seg.pt"
    if not weight.exists():
        raise FileNotFoundError(weight)

    source_rows = []
    for source in TRAIN_SOURCES:
        pe, n = extract_single_source_vpe(weight, source)
        source_rows.append((source, pe, n))

    single_source = "POS_190181"
    single_vpe = next(pe for source, pe, _ in source_rows if source == single_source)
    aggregate_vpe = aggregate_vpes(source_rows)

    similarity = {
        "source_pair_cosine": {
            f"{a}__{b}": cosine(pa, pb)
            for i, (a, pa, _) in enumerate(source_rows)
            for b, pb, _ in source_rows[i + 1 :]
        },
        "single_to_aggregate_cosine": cosine(single_vpe, aggregate_vpe),
        "n_reference_boxes": {source: n for source, _, n in source_rows},
        "aggregation": "box-count-weighted source VPE mean followed by L2 normalization",
        "official_method_note": "Ultralytics YOLOE validator aggregates visual prompt embeddings across reference samples per class and L2-normalizes; this source-level implementation mirrors that pattern using per-source VPEs weighted by reference-box count.",
    }

    val_crop, val_roi, val_tower, val_tower_score, val_tower_gt, val_gt, _ = model_generated_tower_crop(weight, VAL_SOURCE, "val")
    dev_crop, dev_roi, dev_tower, dev_tower_score, dev_tower_gt, dev_gt, _ = model_generated_tower_crop(weight, DEV_SOURCE, "dev")

    modes = {
        "text": None,
        "single_vpe": single_vpe,
        "aggregate_vpe": aggregate_vpe,
    }
    results = {}
    for mode, vpe in modes.items():
        model = build_prompt_model(weight, mode, vpe)
        val_raw = infer(model, val_crop, val_roi, 0.001)
        op, sweep = choose_operating_point(val_raw, val_gt)
        val_filtered = nms([x for x in val_raw if x["score"] >= op["conf_threshold"]], op["nms_iou"])
        val_metrics = [metrics(val_filtered, val_gt, t) for t in (0.30, 0.50)]

        dev_raw = infer(model, dev_crop, dev_roi, 0.001)
        dev_filtered = nms([x for x in dev_raw if x["score"] >= op["conf_threshold"]], op["nms_iou"])
        dev_metrics = [metrics(dev_filtered, dev_gt, t) for t in (0.30, 0.50)]
        render(DEV_SOURCE, mode, dev_filtered, dev_tower, dev_tower_score)
        results[mode] = {
            "validation": {
                "source": VAL_SOURCE,
                "tower_iou": iou(val_tower, val_tower_gt),
                "tower_score": val_tower_score,
                "n_raw": len(val_raw),
                "operating_point": op,
                "metrics": val_metrics,
                "sweep": sweep,
            },
            "development": {
                "source": DEV_SOURCE,
                "semantic_status": "adaptive development showcase; not final holdout",
                "tower_iou": iou(dev_tower, dev_tower_gt),
                "tower_score": dev_tower_score,
                "n_raw": len(dev_raw),
                "n_filtered": len(dev_filtered),
                "metrics": dev_metrics,
                "predictions": dev_filtered,
            },
        }

    report = {
        "evidence_type": "yoloe-26-multi-source-visual-prompt-aggregation-ablation",
        "claim_scope": "exploratory prompt-representation ablation; all operating points selected on POS_5442616 validation only; POS_2326530 is adaptive development; frozen final holdout untouched; model scores uncalibrated; model masks are pseudo-labels, not GT",
        "weight": {"name": weight.name, "sha256": sha256(weight)},
        "reference_sources": TRAIN_SOURCES,
        "single_reference_source": single_source,
        "similarity": similarity,
        "results": results,
        "runtime": runtime_env(),
        "final_holdout_touched": False,
    }
    write_json(REPORTS / "v3_5_yoloe_aggregated_vpe_metrics.json", report)
    print(
        json.dumps(
            {
                "reference_boxes": similarity["n_reference_boxes"],
                "single_to_aggregate_cosine": similarity["single_to_aggregate_cosine"],
                "comparison": {
                    mode: {
                        "val_op": row["validation"]["operating_point"],
                        "dev_iou30": row["development"]["metrics"][0],
                        "dev_iou50": row["development"]["metrics"][1],
                    }
                    for mode, row in results.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
