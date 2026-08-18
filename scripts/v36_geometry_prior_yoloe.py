from __future__ import annotations

import json
import math

import numpy as np

from v23_common import REPORTS, ROOT, WEIGHTS, runtime_env, sha256, write_json
from v25_hybrid_component_detector import read_yolo
from v34_tiled_insulator_specialist import metrics, nms
from v35_yoloe_aggregated_vpe import (
    DEV_SOURCE,
    INSULATOR_CLASS_ID,
    INSULATOR_PROMPT,
    TRAIN_SOURCES,
    VAL_SOURCE,
    build_prompt_model,
    infer,
    model_generated_tower_crop,
    render,
    split_path,
)


def box_features(box, tower_box):
    tx1, ty1, tx2, ty2 = tower_box
    bx1, by1, bx2, by2 = box
    tw = max(1e-6, tx2 - tx1)
    th = max(1e-6, ty2 - ty1)
    bw = max(1e-6, bx2 - bx1)
    bh = max(1e-6, by2 - by1)
    tcx = (tx1 + tx2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    return np.asarray(
        [
            abs(bcx - tcx) / tw,
            (bcy - ty1) / th,
            math.log(bw / tw),
            math.log(bh / th),
        ],
        dtype=np.float64,
    )


def training_prior():
    feats = []
    per_source = {}
    for source in TRAIN_SOURCES:
        image_path, label_path = split_path(source)
        from PIL import Image

        with Image.open(image_path) as im:
            w, h = im.size
        labels = read_yolo(label_path, w, h)
        tower = next(x["box"] for x in labels if x["class_id"] == 0)
        source_feats = [box_features(x["box"], tower) for x in labels if x["class_id"] == INSULATOR_CLASS_ID]
        feats.extend(source_feats)
        per_source[source] = len(source_feats)
    x = np.stack(feats)
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    # Stable shrinkage: keep empirical correlations while preventing the tiny pilot from creating singular directions.
    diag = np.diag(np.diag(cov))
    cov_reg = 0.75 * cov + 0.25 * diag + np.eye(cov.shape[0]) * 1e-5
    inv = np.linalg.inv(cov_reg)
    d2_train = np.einsum("ni,ij,nj->n", x - mean, inv, x - mean)
    return {
        "mean": mean,
        "cov": cov_reg,
        "inv": inv,
        "train_d2": d2_train,
        "train_d2_q95": float(np.quantile(d2_train, 0.95)),
        "train_d2_max": float(d2_train.max()),
        "n": len(x),
        "per_source": per_source,
        "feature_names": ["abs_x_offset_over_tower_width", "y_from_tower_top_over_tower_height", "log_width_over_tower_width", "log_height_over_tower_height"],
    }


def apply_prior(rows, tower_box, prior):
    out = []
    for row in rows:
        f = box_features(row["box"], tower_box)
        delta = f - prior["mean"]
        d2 = float(delta @ prior["inv"] @ delta)
        # Not a probability: a monotone geometry compatibility weight used only for ranking.
        geometry_weight = math.exp(-0.5 * min(d2, 50.0))
        fused_score = float(row["score"] * geometry_weight)
        out.append({**row, "model_score": float(row["score"]), "geometry_d2": d2, "geometry_weight": geometry_weight, "fused_score": fused_score, "score": fused_score})
    return out


def choose(rows, gt):
    confs = sorted(
        set(
            [0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
            + [round(x["score"], 9) for x in rows]
        )
    )
    nmses = [0.20, 0.25, 0.30, 0.40, 0.50]
    sweep = []
    for conf in confs:
        for ni in nmses:
            filt = nms([x for x in rows if x["score"] >= conf], ni)
            m = metrics(filt, gt, 0.30)
            sweep.append({"threshold": conf, "nms_iou": ni, "n_predictions": len(filt), **{k: v for k, v in m.items() if k != "matches"}})
    best = max(sweep, key=lambda x: (x["f1"], x["recall"], x["precision"], -x["n_predictions"], x["threshold"], -x["nms_iou"]))
    return best, sweep


def filter_with_op(rows, op):
    return nms([x for x in rows if x["score"] >= op["threshold"]], op["nms_iou"])


def main():
    weight = WEIGHTS / "yoloe-26n-seg.pt"
    if not weight.exists():
        raise FileNotFoundError(weight)
    prior = training_prior()

    val_crop, val_roi, val_tower, val_tower_score, val_tower_gt, val_gt, _ = model_generated_tower_crop(weight, VAL_SOURCE, "v36_val")
    dev_crop, dev_roi, dev_tower, dev_tower_score, dev_tower_gt, dev_gt, _ = model_generated_tower_crop(weight, DEV_SOURCE, "v36_dev")

    model = build_prompt_model(weight, "text", None)
    val_raw_model = infer(model, val_crop, val_roi, 0.001)
    dev_raw_model = infer(model, dev_crop, dev_roi, 0.001)

    # Baseline: original YOLOE model score only, selected on validation.
    raw_for_select = [{**x, "score": float(x["score"])} for x in val_raw_model]
    raw_op, raw_sweep = choose(raw_for_select, val_gt)
    raw_val = filter_with_op(raw_for_select, raw_op)
    raw_dev = filter_with_op([{**x, "score": float(x["score"])} for x in dev_raw_model], raw_op)

    # Geometry-aware ranking: training-only prior, operating threshold selected only on validation.
    val_fused_all = apply_prior(val_raw_model, val_tower, prior)
    dev_fused_all = apply_prior(dev_raw_model, dev_tower, prior)
    fused_op, fused_sweep = choose(val_fused_all, val_gt)
    fused_val = filter_with_op(val_fused_all, fused_op)
    fused_dev = filter_with_op(dev_fused_all, fused_op)

    render(DEV_SOURCE, "v36_raw_text", raw_dev, dev_tower, dev_tower_score)
    render(DEV_SOURCE, "v36_geometry_fused", fused_dev, dev_tower, dev_tower_score)

    report = {
        "evidence_type": "yoloe-26-text-plus-training-only-component-geometry-prior",
        "claim_scope": "exploratory post-processing ablation; geometry distribution fitted on five training towers/30 insulator GT only; operating thresholds selected on POS_5442616 validation only; POS_2326530 adaptive development; final holdout untouched; model and fused scores are not calibrated probabilities",
        "prior": {
            "n_train_insulators": prior["n"],
            "per_source": prior["per_source"],
            "feature_names": prior["feature_names"],
            "mean": prior["mean"].tolist(),
            "covariance": prior["cov"].tolist(),
            "train_d2_q95": prior["train_d2_q95"],
            "train_d2_max": prior["train_d2_max"],
        },
        "tower_roi": {
            "validation_iou": __import__("v25_hybrid_component_detector").iou(val_tower, val_tower_gt),
            "development_iou": __import__("v25_hybrid_component_detector").iou(dev_tower, dev_tower_gt),
            "validation_score": val_tower_score,
            "development_score": dev_tower_score,
        },
        "raw_text": {
            "validation": {"n_raw": len(val_raw_model), "operating_point": raw_op, "metrics": [metrics(raw_val, val_gt, t) for t in (0.30, 0.50)], "sweep": raw_sweep},
            "development": {"n_raw": len(dev_raw_model), "n_filtered": len(raw_dev), "metrics": [metrics(raw_dev, dev_gt, t) for t in (0.30, 0.50)], "predictions": raw_dev},
        },
        "geometry_fused": {
            "validation": {"n_raw": len(val_fused_all), "operating_point": fused_op, "metrics": [metrics(fused_val, val_gt, t) for t in (0.30, 0.50)], "sweep": fused_sweep},
            "development": {"n_raw": len(dev_fused_all), "n_filtered": len(fused_dev), "metrics": [metrics(fused_dev, dev_gt, t) for t in (0.30, 0.50)], "predictions": fused_dev},
        },
        "weight": {"name": weight.name, "sha256": sha256(weight)},
        "runtime": runtime_env(),
        "final_holdout_touched": False,
    }
    write_json(REPORTS / "v3_6_geometry_prior_metrics.json", report)
    print(json.dumps({
        "prior_q95": prior["train_d2_q95"],
        "raw_val_op": raw_op,
        "fused_val_op": fused_op,
        "raw_dev": report["raw_text"]["development"]["metrics"],
        "fused_dev": report["geometry_fused"]["development"]["metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
