from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from v23_common import REPORTS, ROOT, WEIGHTS, runtime_env, sha256, write_json

OLD_TRAIN = ["POS_1283842", "POS_190181", "POS_291727", "POS_3778704", "POS_7060068"]
NEW_TRAIN = "POS_6610209"
NEW_VAL = "POS_8091164"
ANN_PATH = ROOT / "data/v4_annotations/assistant_provisional_insulators.json"


def filter_old_insulators(label_path: Path) -> list[str]:
    out = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        c, xc, yc, w, h = line.split()
        if int(float(c)) == 2:
            out.append(f"0 {xc} {yc} {w} {h}")
    return out


def new_record(rid: str):
    data = json.loads(ANN_PATH.read_text())
    return next(r for r in data["records"] if r["record_id"] == rid)


def pixel_boxes_to_yolo(record) -> list[str]:
    W, H = record["dimensions"]
    rows = []
    for item in record["boxes_xyxy"]:
        x1, y1, x2, y2 = map(float, item["xyxy"])
        xc = ((x1 + x2) / 2) / W
        yc = ((y1 + y2) / 2) / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H
        rows.append(f"0 {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}")
    return rows


def prepare_dataset(root: Path, include_new: bool):
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (root / f"labels/{split}").mkdir(parents=True, exist_ok=True)

    n_train = 0
    for rid in OLD_TRAIN:
        src = ROOT / f"data/images/train/{rid}.jpg"
        lab = ROOT / f"data/labels/train/{rid}.txt"
        if not src.exists() or not lab.exists():
            raise FileNotFoundError(rid)
        shutil.copy2(src, root / f"images/train/{rid}.jpg")
        rows = filter_old_insulators(lab)
        n_train += len(rows)
        (root / f"labels/train/{rid}.txt").write_text("\n".join(rows) + "\n")

    if n_train != 30:
        raise RuntimeError(f"Expected 30 old-train insulators, got {n_train}")

    if include_new:
        rec = new_record(NEW_TRAIN)
        src = ROOT / f"reports/v4_0_morphology_candidates/{NEW_TRAIN}.jpg"
        if sha256(src) != rec["source_sha256"]:
            raise RuntimeError("New-train source hash mismatch")
        shutil.copy2(src, root / f"images/train/{NEW_TRAIN}.jpg")
        rows = pixel_boxes_to_yolo(rec)
        n_train += len(rows)
        (root / f"labels/train/{NEW_TRAIN}.txt").write_text("\n".join(rows) + "\n")

    val_rec = new_record(NEW_VAL)
    val_src = ROOT / f"reports/v4_0_morphology_candidates/{NEW_VAL}.jpg"
    if sha256(val_src) != val_rec["source_sha256"]:
        raise RuntimeError("New-validation source hash mismatch")
    shutil.copy2(val_src, root / f"images/val/{NEW_VAL}.jpg")
    val_rows = pixel_boxes_to_yolo(val_rec)
    (root / f"labels/val/{NEW_VAL}.txt").write_text("\n".join(val_rows) + "\n")

    (root / "data.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: overhead_line_insulator_assembly\n"
    )
    return {
        "train_sources": OLD_TRAIN + ([NEW_TRAIN] if include_new else []),
        "n_train_sources": len(OLD_TRAIN) + int(include_new),
        "n_train_insulators": n_train,
        "val_source": NEW_VAL,
        "n_val_insulators": len(val_rows),
    }


def scalar(x):
    try:
        return float(x)
    except Exception:
        try:
            return float(x.mean())
        except Exception:
            return None


def eval_model(model, yaml_path: Path):
    m = model.val(data=str(yaml_path), split="val", imgsz=960, batch=1, workers=0, device="cpu", plots=False, verbose=False)
    rd = {str(k): scalar(v) for k, v in m.results_dict.items()}
    return {
        "precision": rd.get("metrics/precision(B)"),
        "recall": rd.get("metrics/recall(B)"),
        "map50": rd.get("metrics/mAP50(B)"),
        "map50_95": rd.get("metrics/mAP50-95(B)"),
        "results_dict": rd,
    }


def fixed_predictions(model, image_path: Path, conf: float = 0.01):
    r = model.predict(str(image_path), imgsz=960, conf=conf, iou=0.5, max_det=100, device="cpu", verbose=False)[0]
    rows = []
    if r.boxes is not None:
        for b, s in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist()):
            rows.append({"score": float(s), "box": [float(x) for x in b]})
    return rows


def render(image_path: Path, rows, out: Path):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im); font = ImageFont.load_default()
    for r in rows:
        b = tuple(int(round(x)) for x in r["box"])
        d.rectangle(b, outline=(220, 60, 150), width=3)
        d.text((b[0], b[1]), f"insulator {r['score']:.3f}", fill=(220, 60, 150), font=font)
    im.save(out, quality=95)


def run_arm(name: str, dataset_root: Path, epochs: int, weight: Path):
    from ultralytics import YOLO
    manifest = prepare_dataset(dataset_root, include_new=(name == "expanded"))
    run_name = f"v4_3_yolo26n_{name}"
    model = YOLO(str(weight))
    model.train(
        data=str(dataset_root / "data.yaml"), epochs=epochs, imgsz=960, batch=1, workers=0, device="cpu",
        seed=17, deterministic=True, pretrained=True, project=str(ROOT / "runs"), name=run_name,
        exist_ok=True, plots=False, verbose=True, mosaic=0.0, close_mosaic=0, translate=0.04, scale=0.15,
        fliplr=0.5, optimizer="AdamW", lr0=0.001, lrf=0.01, patience=epochs + 1,
    )
    best = ROOT / f"runs/{run_name}/weights/best.pt"
    last = ROOT / f"runs/{run_name}/weights/last.pt"
    chosen = best if best.exists() else last
    if not chosen.exists():
        raise RuntimeError(f"No checkpoint for {name}")
    trained = YOLO(str(chosen))
    validation = eval_model(trained, dataset_root / "data.yaml")
    val_image = dataset_root / f"images/val/{NEW_VAL}.jpg"
    preds = fixed_predictions(trained, val_image, 0.01)
    render(val_image, preds, REPORTS / f"v4_3_{name}_{NEW_VAL}.jpg")
    return {
        "manifest": manifest,
        "checkpoint": {"path": str(chosen.relative_to(ROOT)), "sha256": sha256(chosen)},
        "validation": validation,
        "fixed_conf_0_01": {"n_predictions": len(preds), "predictions": preds},
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=30); args = ap.parse_args()
    weight = WEIGHTS / "yolo26n.pt"
    if not weight.exists():
        raise FileNotFoundError(weight)
    baseline = run_arm("baseline", ROOT / "data/v43_baseline", args.epochs, weight)
    expanded = run_arm("expanded", ROOT / "data/v43_expanded", args.epochs, weight)
    delta = {
        k: (expanded["validation"][k] - baseline["validation"][k]) if baseline["validation"][k] is not None and expanded["validation"][k] is not None else None
        for k in ("precision", "recall", "map50", "map50_95")
    }
    report = {
        "version": "v4.3-controlled-morphology-diversity-yolo26",
        "evidence_type": "paired training-data ablation with identical pretrained YOLO26n/config and fixed new validation source",
        "claim_scope": "Development experiment on one assistant-provisional validation tower; not a production estimate. The only intended arm difference is adding POS_6610209 and its seven morphology-diverse insulator references to training.",
        "epochs": args.epochs,
        "input_weight": {"name": weight.name, "sha256": sha256(weight)},
        "baseline": baseline,
        "expanded": expanded,
        "delta_expanded_minus_baseline": delta,
        "old_v3_8_holdout_used": False,
        "runtime": runtime_env(),
    }
    write_json(REPORTS / "v4_3_morphology_diversity_yolo26.json", report)
    print(json.dumps({"baseline":baseline["validation"],"expanded":expanded["validation"],"delta":delta}, indent=2))


if __name__ == "__main__":
    main()
