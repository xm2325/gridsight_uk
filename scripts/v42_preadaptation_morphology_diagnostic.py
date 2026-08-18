from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from v23_common import REPORTS, ROOT, WEIGHTS, runtime_env, sha256, write_json
from v25_hybrid_component_detector import TOWER_PROMPT, expand
from v34_tiled_insulator_specialist import metrics, nms
from v35_yoloe_aggregated_vpe import INSULATOR_PROMPT, build_prompt_model, infer
from v36_geometry_prior_yoloe import apply_prior, box_features, training_prior

ANNOTATIONS = ROOT / "data/v4_annotations/assistant_provisional_insulators.json"
FREEZE = ROOT / "data/final_holdout/champion_freeze_v38.json"
SOURCES = ["POS_6610209", "POS_8091164"]


def load_json(path: Path):
    return json.loads(path.read_text())


def source_path(rid: str):
    path = ROOT / f"reports/v4_0_morphology_candidates/{rid}.jpg"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def annotation_map():
    rows = load_json(ANNOTATIONS)["records"]
    return {r["record_id"]: r for r in rows}


def model_tower_crop(weight: Path, rid: str, cfg):
    from PIL import Image
    from ultralytics import YOLOE

    path = source_path(rid)
    image = Image.open(path).convert("RGB")
    w, h = image.size
    tower = YOLOE(str(weight))
    tower.set_classes([cfg["tower_prompt"]])
    r = tower.predict(str(path), imgsz=cfg["tower_predict_imgsz"], conf=cfg["tower_predict_conf"], device="cpu", verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        raise RuntimeError(f"No tower proposal for {rid}")
    idx = int(r.boxes.conf.argmax().item())
    box = [float(x) for x in r.boxes.xyxy[idx].cpu().tolist()]
    score = float(r.boxes.conf[idx].item())
    roi = expand(box, w, h, px=cfg["tower_roi_expand_px_fraction"], py=cfg["tower_roi_expand_py_fraction"])
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    crop_path = REPORTS / f"v4_2_{rid}_tower_roi.jpg"
    image.crop((x1, y1, x2, y2)).save(crop_path, quality=95)
    return crop_path, (x1, y1, x2, y2), box, score, (w, h)


def render(rid, tower_box, rows, suffix):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(source_path(rid)).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    tb = tuple(int(round(v)) for v in tower_box)
    draw.rectangle(tb, outline=(35, 180, 75), width=3)
    for row in rows:
        b = tuple(int(round(v)) for v in row["box"])
        draw.rectangle(b, outline=(220, 60, 150), width=3)
        model_score = row.get("model_score", row.get("score", 0.0))
        draw.text((b[0], b[1]), f"insulator {model_score:.3f}", fill=(220, 60, 150), font=font)
    out = REPORTS / f"v4_2_{rid}_{suffix}.jpg"
    image.save(out, quality=95)
    return str(out.relative_to(ROOT))


def main():
    freeze = load_json(FREEZE)
    cfg = freeze["champion"]
    raw_cfg = freeze["raw_text_baseline"]
    annotations = annotation_map()
    if set(annotations) != set(SOURCES):
        raise RuntimeError(f"Unexpected v4 annotation IDs: {sorted(annotations)}")
    if any(rid in freeze["final_holdout_release"]["records"] for rid in SOURCES):
        raise RuntimeError("Retired v3.8 holdout leaked into v4 diagnostic")

    weight = WEIGHTS / cfg["checkpoint"]
    if sha256(weight) != cfg["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash differs from v3.8 freeze")
    if cfg["text_prompt"] != INSULATOR_PROMPT or cfg["tower_prompt"] != TOWER_PROMPT:
        raise RuntimeError("Prompt mismatch vs frozen implementation")

    prior = training_prior()
    model = build_prompt_model(weight, "text", None)
    records = []
    pooled_raw = []
    pooled_geom = []
    all_gt_d2 = []

    for rid in SOURCES:
        crop, roi, tower_box, tower_score, dims = model_tower_crop(weight, rid, cfg)
        raw = infer(model, crop, roi, conf=cfg["component_raw_conf"])
        raw_selected = nms([dict(x) for x in raw if x["score"] >= raw_cfg["score_threshold"]], raw_cfg["nms_iou"])
        geom_all = apply_prior(raw, tower_box, prior)
        geom_selected = nms([x for x in geom_all if x["score"] >= cfg["fused_threshold"]], cfg["nms_iou"])
        gt = [list(map(float, b["xyxy"])) for b in annotations[rid]["boxes_xyxy"]]
        raw_m = [metrics(raw_selected, gt, t) for t in (0.30, 0.50)]
        geom_m = [metrics(geom_selected, gt, t) for t in (0.30, 0.50)]
        pooled_raw.append(raw_m); pooled_geom.append(geom_m)

        gt_shift = []
        for box_obj in annotations[rid]["boxes_xyxy"]:
            f = box_features(box_obj["xyxy"], tower_box)
            delta = f - prior["mean"]
            d2 = float(delta @ prior["inv"] @ delta)
            gt_shift.append({"id": box_obj["id"], "orientation": box_obj["orientation"], "geometry_d2": d2, "beyond_training_q95": d2 > prior["train_d2_q95"], "beyond_training_max": d2 > prior["train_d2_max"]})
            all_gt_d2.append(d2)
        records.append({
            "record_id": rid,
            "annotation_role": annotations[rid]["role"],
            "n_gt": len(gt),
            "tower_box": tower_box,
            "tower_score": tower_score,
            "n_raw_candidates": len(raw),
            "raw_baseline": {"n_predictions": len(raw_selected), "metrics": raw_m, "render": render(rid, tower_box, raw_selected, "raw_frozen")},
            "geometry_champion": {"n_predictions": len(geom_selected), "metrics": geom_m, "render": render(rid, tower_box, geom_selected, "geometry_frozen")},
            "gt_geometry_shift": gt_shift,
        })

    def pool(groups, index):
        tp=sum(x[index]["tp"] for x in groups); fp=sum(x[index]["fp"] for x in groups); fn=sum(x[index]["fn"] for x in groups)
        p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0
        return {"iou_threshold": [0.30,0.50][index], "tp":tp,"fp":fp,"fn":fn,"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0.0}

    report = {
        "version": "v4.2-frozen-pre-adaptation-diagnostic",
        "evidence_type": "new-cycle diagnostic using v3.8-frozen model and operating points without retuning",
        "claim_scope": "Development diagnostic on newly pixel-reviewed assistant-provisional references; no thresholds/prompts/priors selected on these images; old v3.8 final holdout not used.",
        "frozen_source": {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha256(FREEZE)},
        "annotations": {"path": str(ANNOTATIONS.relative_to(ROOT)), "sha256": sha256(ANNOTATIONS), "n_total": sum(r["n_gt"] for r in records)},
        "training_geometry_reference": {"n": prior["n"], "d2_q95": prior["train_d2_q95"], "d2_max": prior["train_d2_max"]},
        "records": records,
        "pooled": {"raw_text": [pool(pooled_raw,i) for i in range(2)], "geometry_champion": [pool(pooled_geom,i) for i in range(2)]},
        "new_gt_shift": {"median_d2": float(np.median(all_gt_d2)), "n_beyond_train_q95": int(sum(x > prior["train_d2_q95"] for x in all_gt_d2)), "n_beyond_train_max": int(sum(x > prior["train_d2_max"] for x in all_gt_d2)), "n_total": len(all_gt_d2)},
        "retired_v3_8_holdout_used": False,
        "runtime": runtime_env(),
    }
    write_json(REPORTS / "v4_2_preadaptation_morphology_diagnostic.json", report)
    print(json.dumps({"pooled": report["pooled"], "new_gt_shift": report["new_gt_shift"], "per_record": {r["record_id"]: {"raw":r["raw_baseline"]["metrics"][0],"geometry":r["geometry_champion"]["metrics"][0]} for r in records}}, indent=2))


if __name__ == "__main__":
    main()
