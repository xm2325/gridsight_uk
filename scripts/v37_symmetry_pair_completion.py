from __future__ import annotations

import json
import math

import numpy as np

from v23_common import REPORTS, ROOT, WEIGHTS, runtime_env, sha256, write_json
from v25_hybrid_component_detector import read_yolo
from v34_tiled_insulator_specialist import metrics, nms
from v35_yoloe_aggregated_vpe import DEV_SOURCE, INSULATOR_CLASS_ID, TRAIN_SOURCES, VAL_SOURCE, build_prompt_model, infer, model_generated_tower_crop, render, split_path
from v36_geometry_prior_yoloe import apply_prior, choose, filter_with_op, training_prior


def center_size(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1e-6, x2 - x1), max(1e-6, y2 - y1)


def pair_features(a, b, tower):
    ax, ay, aw, ah = center_size(a)
    bx, by, bw, bh = center_size(b)
    tx1, ty1, tx2, ty2 = tower
    tw = max(1e-6, tx2 - tx1)
    th = max(1e-6, ty2 - ty1)
    tcx = (tx1 + tx2) / 2.0
    return np.asarray([
        abs(ay - by) / th,
        abs((ax + bx) / 2.0 - tcx) / tw,
        abs(math.log(aw / bw)),
        abs(math.log(ah / bh)),
        abs(ax - bx) / tw,
    ], dtype=np.float64)


def training_pair_prior():
    feats = []
    per_source = {}
    for source in TRAIN_SOURCES:
        image, label = split_path(source)
        from PIL import Image
        with Image.open(image) as im:
            w, h = im.size
        rows = read_yolo(label, w, h)
        tower = next(x["box"] for x in rows if x["class_id"] == 0)
        ins = sorted([x["box"] for x in rows if x["class_id"] == INSULATOR_CLASS_ID], key=lambda b: center_size(b)[1])
        if len(ins) % 2:
            raise RuntimeError(f"Odd insulator count for {source}: {len(ins)}")
        source_pairs = 0
        for i in range(0, len(ins), 2):
            a, b = ins[i], ins[i + 1]
            ax, _, _, _ = center_size(a); bx, _, _, _ = center_size(b)
            tcx = (tower[0] + tower[2]) / 2.0
            if (ax - tcx) * (bx - tcx) >= 0:
                raise RuntimeError(f"Training pair is not opposite-sided: {source} pair {i//2}")
            feats.append(pair_features(a, b, tower)); source_pairs += 1
        per_source[source] = source_pairs
    x = np.stack(feats)
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    diag = np.diag(np.diag(cov))
    cov_reg = 0.60 * cov + 0.40 * diag + np.eye(cov.shape[0]) * 1e-5
    inv = np.linalg.inv(cov_reg)
    d2 = np.einsum("ni,ij,nj->n", x - mean, inv, x - mean)
    return {"mean": mean, "cov": cov_reg, "inv": inv, "d2": d2, "n_pairs": len(x), "per_source": per_source, "q95": float(np.quantile(d2, 0.95)), "max": float(d2.max())}


def opposite_side(a, b, tower):
    ax, _, _, _ = center_size(a); bx, _, _, _ = center_size(b); tcx = (tower[0] + tower[2]) / 2.0
    return (ax - tcx) * (bx - tcx) < 0


def proposal_pairs(seeds, raw_geometry, tower, pair_prior):
    proposals = []
    for si, seed in enumerate(seeds):
        for ci, cand in enumerate(raw_geometry):
            # Don't re-add an already accepted box and require the counterpart to be across the tower centre.
            if any(_iou(cand["box"], s["box"]) >= 0.50 for s in seeds):
                continue
            if not opposite_side(seed["box"], cand["box"], tower):
                continue
            f = pair_features(seed["box"], cand["box"], tower)
            delta = f - pair_prior["mean"]
            d2 = float(delta @ pair_prior["inv"] @ delta)
            pair_weight = math.exp(-0.5 * min(d2, 50.0))
            # Geometry compatibility of the candidate itself plus pair compatibility; model score remains uncalibrated.
            score = float(cand["model_score"] * cand["geometry_weight"] * pair_weight)
            proposals.append({"seed_index": si, "candidate_index": ci, "pair_score": score, "pair_d2": d2, "pair_weight": pair_weight})
    return sorted(proposals, key=lambda x: x["pair_score"], reverse=True)


def _iou(a, b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]);inter=max(0.0,x2-x1)*max(0.0,y2-y1)
    aa=max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1]);bb=max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1]);den=aa+bb-inter
    return inter/den if den else 0.0


def complete(seeds, raw_geometry, tower, pair_prior, threshold):
    props = proposal_pairs(seeds, raw_geometry, tower, pair_prior)
    used_seeds = set(); used_candidates = set(); added = []
    for p in props:
        if p["pair_score"] < threshold:
            continue
        if p["seed_index"] in used_seeds or p["candidate_index"] in used_candidates:
            continue
        cand = raw_geometry[p["candidate_index"]]
        if any(_iou(cand["box"], x["box"]) >= 0.30 for x in seeds + added):
            continue
        row = {**cand, "pair_score": p["pair_score"], "pair_d2": p["pair_d2"], "pair_weight": p["pair_weight"], "completion": True}
        added.append(row); used_seeds.add(p["seed_index"]); used_candidates.add(p["candidate_index"])
    return nms(seeds + added, 0.20), props, added


def select_pair_threshold(seeds, raw_geometry, tower, pair_prior, gt):
    props = proposal_pairs(seeds, raw_geometry, tower, pair_prior)
    scores = [p["pair_score"] for p in props if p["pair_score"] > 0]
    candidates = sorted(set([0.0, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4] + [round(s, 12) for s in scores]))
    sweep = []
    for threshold in candidates:
        rows, _, added = complete(seeds, raw_geometry, tower, pair_prior, threshold)
        m = metrics(rows, gt, 0.30)
        sweep.append({"pair_threshold": threshold, "n_total": len(rows), "n_added": len(added), **{k:v for k,v in m.items() if k != "matches"}})
    best = max(sweep, key=lambda x: (x["f1"], x["recall"], x["precision"], -x["n_total"], x["pair_threshold"]))
    return best, sweep


def main():
    weight = WEIGHTS / "yoloe-26n-seg.pt"
    geometry_prior = training_prior(); pair_prior = training_pair_prior()

    val_crop, val_roi, val_tower, val_tower_score, val_tower_gt, val_gt, _ = model_generated_tower_crop(weight, VAL_SOURCE, "v37_val")
    dev_crop, dev_roi, dev_tower, dev_tower_score, dev_tower_gt, dev_gt, _ = model_generated_tower_crop(weight, DEV_SOURCE, "v37_dev")
    model = build_prompt_model(weight, "text", None)
    val_raw = infer(model, val_crop, val_roi, 0.001); dev_raw = infer(model, dev_crop, dev_roi, 0.001)
    val_geom = apply_prior(val_raw, val_tower, geometry_prior); dev_geom = apply_prior(dev_raw, dev_tower, geometry_prior)

    # Reproduce v3.6 seed selection on validation only.
    seed_op, seed_sweep = choose(val_geom, val_gt)
    val_seeds = filter_with_op(val_geom, seed_op); dev_seeds = filter_with_op(dev_geom, seed_op)

    pair_op, pair_sweep = select_pair_threshold(val_seeds, val_geom, val_tower, pair_prior, val_gt)
    val_completed, val_props, val_added = complete(val_seeds, val_geom, val_tower, pair_prior, pair_op["pair_threshold"])
    dev_completed, dev_props, dev_added = complete(dev_seeds, dev_geom, dev_tower, pair_prior, pair_op["pair_threshold"])

    render(DEV_SOURCE, "v37_precision_seeds", dev_seeds, dev_tower, dev_tower_score)
    render(DEV_SOURCE, "v37_symmetry_completed", dev_completed, dev_tower, dev_tower_score)

    report = {
        "evidence_type": "training-derived-insulator-symmetry-pair-completion",
        "claim_scope": "exploratory structural postprocess; component and pair priors fitted on five training towers only; geometry seed and pair acceptance operating points selected on one validation tower only; POS_2326530 adaptive development; final holdout untouched; scores are ranking heuristics, not calibrated probabilities",
        "geometry_seed_operating_point": seed_op,
        "pair_prior": {"n_training_pairs": pair_prior["n_pairs"], "per_source": pair_prior["per_source"], "mean": pair_prior["mean"].tolist(), "covariance": pair_prior["cov"].tolist(), "train_d2_q95": pair_prior["q95"], "train_d2_max": pair_prior["max"]},
        "pair_validation_selection": {"best": pair_op, "sweep": pair_sweep},
        "validation": {"n_seeds": len(val_seeds), "n_added": len(val_added), "seed_metrics": [metrics(val_seeds, val_gt, t) for t in (0.30,0.50)], "completed_metrics": [metrics(val_completed, val_gt, t) for t in (0.30,0.50)], "added": val_added},
        "development": {"source": DEV_SOURCE, "semantic_status": "adaptive development showcase; not final holdout", "n_seeds": len(dev_seeds), "n_added": len(dev_added), "seed_metrics": [metrics(dev_seeds, dev_gt, t) for t in (0.30,0.50)], "completed_metrics": [metrics(dev_completed, dev_gt, t) for t in (0.30,0.50)], "added": dev_added, "completed_predictions": dev_completed},
        "weight": {"name": weight.name, "sha256": sha256(weight)},
        "runtime": runtime_env(),
        "final_holdout_touched": False,
    }
    write_json(REPORTS / "v3_7_symmetry_pair_metrics.json", report)
    print(json.dumps({"seed_op":seed_op,"pair_op":pair_op,"val_seed":report["validation"]["seed_metrics"],"val_completed":report["validation"]["completed_metrics"],"dev_seed":report["development"]["seed_metrics"],"dev_completed":report["development"]["completed_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
