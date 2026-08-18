from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from v23_common import REPORTS, ROOT, WEIGHTS, runtime_env, sha256, write_json
from v25_hybrid_component_detector import TOWER_PROMPT, expand, iou, read_yolo
from v34_tiled_insulator_specialist import metrics, nms
from v35_yoloe_aggregated_vpe import INSULATOR_CLASS_ID, INSULATOR_PROMPT, build_prompt_model, infer
from v36_geometry_prior_yoloe import apply_prior, training_prior

FREEZE = ROOT / "data/final_holdout/final_holdout_freeze.json"
CHAMPION = ROOT / "data/final_holdout/champion_freeze_v38.json"
FINAL_DIR = ROOT / "data/final_holdout/images"


def download_exact(url: str, path: Path, expected_sha256: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "GridSight-UK-v3.8-final-holdout/1.0 (research portfolio; contact via GitHub xm2325/gridsight_uk)"})
            with urllib.request.urlopen(req, timeout=90) as response:
                data = response.read()
            got = hashlib.sha256(data).hexdigest()
            if got != expected_sha256:
                raise RuntimeError(f"SHA256 mismatch: expected {expected_sha256}, got {got}")
            path.write_bytes(data)
            return {"bytes": len(data), "sha256": got, "attempt": attempt}
        except Exception as exc:
            failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < 5:
                time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(f"Could not hydrate exact final-holdout image {url}: {' | '.join(failures)}")


def pooled(per_image, threshold: float):
    rows = [x["metrics"][str(threshold)] for x in per_image.values()]
    tp = sum(x["tp"] for x in rows); fp = sum(x["fp"] for x in rows); fn = sum(x["fn"] for x in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"iou_threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0}


def final_tower_crop(weight: Path, image_path: Path, label_path: Path, cfg: dict):
    from PIL import Image
    from ultralytics import YOLOE

    image = Image.open(image_path).convert("RGB"); w, h = image.size
    labels = read_yolo(label_path, w, h)
    tower_gt = next(x["box"] for x in labels if x["class_id"] == 0)
    insulator_gt = [x["box"] for x in labels if x["class_id"] == INSULATOR_CLASS_ID]

    tower_model = YOLOE(str(weight))
    tower_model.set_classes([cfg["tower_prompt"]])
    result = tower_model.predict(str(image_path), imgsz=cfg["tower_predict_imgsz"], conf=cfg["tower_predict_conf"], device="cpu", verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError(f"No tower detection on final holdout {image_path.stem}")
    idx = int(result.boxes.conf.argmax().item())
    tower_box = [float(x) for x in result.boxes.xyxy[idx].cpu().tolist()]
    tower_score = float(result.boxes.conf[idx].item())
    roi = expand(tower_box, w, h, px=cfg["tower_roi_expand_px_fraction"], py=cfg["tower_roi_expand_py_fraction"])
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    crop = image.crop((x1, y1, x2, y2)); crop_path = REPORTS / f"v3_8_{image_path.stem}_tower_roi.jpg"; crop.save(crop_path, quality=95)
    return crop_path, (x1,y1,x2,y2), tower_box, tower_score, tower_gt, insulator_gt, (w,h)


def render_final(image_path: Path, record_id: str, tower_box, raw_rows, champion_rows, gt_rows):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(image_path).convert("RGB"); font = ImageFont.load_default()
    # Champion-only output: genuine model candidate boxes after frozen geometry postprocess.
    champion = image.copy(); draw = ImageDraw.Draw(champion)
    tb = tuple(int(round(v)) for v in tower_box); draw.rectangle(tb, outline=(35,180,75), width=3); draw.text((tb[0],tb[1]), "YOLOE tower ROI", fill=(35,180,75), font=font)
    for p in champion_rows:
        b = tuple(int(round(v)) for v in p["box"]); draw.rectangle(b, outline=(220,60,150), width=3)
        draw.text((b[0],b[1]), f"insulator fused={p['score']:.5f}", fill=(220,60,150), font=font)
    champion_path = REPORTS / f"v3_8_final_{record_id}_champion.jpg"; champion.save(champion_path, quality=95)

    # Audit comparison after the frozen inference: GT is shown only for evaluation, never used to alter predictions.
    audit = image.copy(); draw = ImageDraw.Draw(audit)
    for g in gt_rows:
        b = tuple(int(round(v)) for v in g); draw.rectangle(b, outline=(40,180,220), width=3)
    for p in champion_rows:
        b = tuple(int(round(v)) for v in p["box"]); draw.rectangle(b, outline=(220,60,150), width=3)
    draw.text((8,8), "cyan=pre-frozen reference GT; magenta=frozen champion prediction", fill=(255,255,255), font=font)
    audit_path = REPORTS / f"v3_8_final_{record_id}_prediction_vs_reference.jpg"; audit.save(audit_path, quality=95)
    return champion_path, audit_path


def main():
    freeze = json.loads(FREEZE.read_text()); champion_doc = json.loads(CHAMPION.read_text()); cfg = champion_doc["champion"]; raw_cfg = champion_doc["raw_text_baseline"]
    assert freeze["holdout_release_sha256"] == champion_doc["final_holdout_release"]["holdout_release_sha256"]
    assert cfg["tower_prompt"] == TOWER_PROMPT, (cfg["tower_prompt"], TOWER_PROMPT)
    assert cfg["text_prompt"] == INSULATOR_PROMPT, (cfg["text_prompt"], INSULATOR_PROMPT)

    weight = WEIGHTS / cfg["checkpoint"]
    if not weight.exists(): raise FileNotFoundError(weight)
    got_weight_sha = sha256(weight)
    if got_weight_sha != cfg["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint identity mismatch: {got_weight_sha}")

    # Training-only geometry prior is recomputed from the already-frozen development training labels.
    prior = training_prior()
    text_model = build_prompt_model(weight, "text", None)

    hydration = {}; raw_eval = {}; champion_eval = {}; record_meta = {}
    for rec in freeze["records"]:
        rid = rec["record_id"]; image_path = FINAL_DIR / f"{rid}.jpg"; label_path = ROOT / rec["label_path"]
        label_sha = sha256(label_path)
        if label_sha != rec["label_sha256"]: raise RuntimeError(f"Label SHA mismatch for {rid}: {label_sha}")
        hydration[rid] = download_exact(rec["image_url"], image_path, rec["image_sha256"])
        crop, roi, tower_box, tower_score, tower_gt, ins_gt, dims = final_tower_crop(weight, image_path, label_path, cfg)

        raw_all = infer(text_model, crop, roi, cfg["component_raw_conf"])
        raw_rows = nms([x for x in raw_all if x["score"] >= raw_cfg["score_threshold"]], raw_cfg["nms_iou"])
        fused_all = apply_prior(raw_all, tower_box, prior)
        champion_rows = nms([x for x in fused_all if x["score"] >= cfg["fused_threshold"]], cfg["nms_iou"])

        raw_metrics = {str(t): metrics(raw_rows, ins_gt, t) for t in (0.30,0.50)}
        champ_metrics = {str(t): metrics(champion_rows, ins_gt, t) for t in (0.30,0.50)}
        raw_eval[rid] = {"n_raw": len(raw_all), "n_filtered": len(raw_rows), "metrics": raw_metrics, "predictions": raw_rows}
        champion_eval[rid] = {"n_raw": len(fused_all), "n_filtered": len(champion_rows), "metrics": champ_metrics, "predictions": champion_rows}
        champion_img, audit_img = render_final(image_path, rid, tower_box, raw_rows, champion_rows, ins_gt)
        record_meta[rid] = {"country": rec["country"], "role": rec["role"], "dimensions": list(dims), "n_reference_insulators": len(ins_gt), "tower_iou": iou(tower_box, tower_gt), "tower_score": tower_score, "champion_image": champion_img.name, "audit_image": audit_img.name}

    report = {
        "evidence_type": "one-shot-frozen-final-holdout-evaluation",
        "holdout_release_sha256": freeze["holdout_release_sha256"],
        "champion_freeze_sha256": sha256(CHAMPION),
        "champion_name": cfg["name"],
        "checkpoint_sha256": got_weight_sha,
        "evaluation_started_after_champion_freeze": True,
        "no_further_tuning_permitted": True,
        "claim_scope": "independent two-image final component holdout selected and frozen before model inference; assistant-provisional reference labels and tiny n mean results are portfolio evidence, not production performance estimates",
        "hydration": hydration,
        "records": record_meta,
        "raw_text_baseline": {"per_image": raw_eval, "pooled": {str(t): pooled(raw_eval, t) for t in (0.30,0.50)}},
        "frozen_champion": {"per_image": champion_eval, "pooled": {str(t): pooled(champion_eval, t) for t in (0.30,0.50)}},
        "runtime": {**runtime_env(), "github_sha": os.environ.get("GITHUB_SHA"), "github_head_ref": os.environ.get("GITHUB_HEAD_REF"), "github_run_id": os.environ.get("GITHUB_RUN_ID")},
        "headline_claim_ready": False,
        "headline_block_reason": "Only two final-holdout images with assistant-provisional component references; no independent human adjudication or production-scale sample size.",
        "final_holdout_touched": True
    }
    write_json(REPORTS / "v3_8_final_holdout_metrics.json", report)
    print(json.dumps({"raw_pooled":report["raw_text_baseline"]["pooled"],"champion_pooled":report["frozen_champion"]["pooled"],"records":record_meta,"no_further_tuning_permitted":True}, indent=2))


if __name__ == "__main__":
    main()
