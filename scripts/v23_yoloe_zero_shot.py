from __future__ import annotations

import json
from pathlib import Path

from v23_common import DISPLAY, PROMPTS, REPORTS, SHOWCASE, WEIGHTS, dataset_manifest, runtime_env, sha256, write_json


def main() -> None:
    from ultralytics import YOLOE

    weight = WEIGHTS / "yoloe-26n-seg.pt"
    if not weight.exists():
        raise FileNotFoundError(weight)

    model = YOLOE(str(weight))
    model.set_classes(PROMPTS)
    results = model.predict(str(SHOWCASE), imgsz=640, conf=0.05, device="cpu", verbose=False)
    if len(results) != 1:
        raise RuntimeError(f"Expected one result, got {len(results)}")
    r = results[0]

    rows = []
    boxes = r.boxes
    mask_data = None if r.masks is None else r.masks.data.cpu().numpy()
    if boxes is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            c = int(cls[i])
            row = {
                "prediction_index": i,
                "class_index": c,
                "prompt": PROMPTS[c] if 0 <= c < len(PROMPTS) else str(c),
                "display_label": DISPLAY[c] if 0 <= c < len(DISPLAY) else str(c),
                "model_score": float(conf[i]),
                "xyxy_px": [float(x) for x in xyxy[i]],
            }
            if mask_data is not None and i < len(mask_data):
                m = mask_data[i]
                row["mask_pixels_at_model_resolution"] = int((m > 0.5).sum())
                row["mask_shape"] = list(m.shape)
            rows.append(row)

    out_img = REPORTS / "v2_3_yoloe26_zero_shot_POS_2326530.jpg"
    r.save(filename=str(out_img))
    write_json(REPORTS / "v2_3_yoloe26_zero_shot_predictions.json", {
        "evidence_type": "pretrained-open-vocabulary-zero-shot",
        "claim_scope": "exploratory model output; not calibrated confidence and not human mask ground truth",
        "source_image": "POS_2326530",
        "prompts": PROMPTS,
        "weight": {"name": weight.name, "sha256": sha256(weight)},
        "runtime": runtime_env(),
        "dataset_manifest": dataset_manifest(),
        "n_predictions": len(rows),
        "predictions": rows,
    })
    print(json.dumps({"n_predictions": len(rows), "output": str(out_img)}, indent=2))


if __name__ == "__main__":
    main()
