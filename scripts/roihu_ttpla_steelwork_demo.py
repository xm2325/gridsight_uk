"""One bounded TTPLA lattice-structure segmentation comparison on Roihu."""
import argparse
import gc
import json
import math
import os
import shutil
import time
import traceback
from pathlib import Path

from paper_material_demo import ROOT, load, sha, write
from component_mask_metrics import decode_masks, mask_matches, raster_polygon


def counts(records, arm):
    total = {"tp": 0, "fp": 0, "fn": 0}
    positive_images = negative_images = negative_images_with_fp = 0
    matched_ious = []
    for record in records:
        metric = record[arm]["metrics"]
        for key in total:
            total[key] += metric[key]
        if record["reference_count"]:
            positive_images += 1
        else:
            negative_images += 1
            negative_images_with_fp += metric["fp"] > 0
        matched_ious.extend(m["iou"] for m in metric["matches"])
    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {**total, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "positive_images": positive_images, "negative_images": negative_images,
            "negative_images_with_false_positive": negative_images_with_fp,
            "negative_image_specificity": 1 - negative_images_with_fp / negative_images if negative_images else None,
            "mean_matched_mask_iou": sum(matched_ious) / len(matched_ious) if matched_ious else None,
            "operating_point_is_fixed_not_calibrated_probability": True}


def main(config_path):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "gputest":
        raise RuntimeError("Requires Roihu gputest; no local model fallback")
    cfg_path = ROOT / config_path
    cfg = load(cfg_path)
    data = ROOT / cfg["dataset_output"]
    out = ROOT / cfg["run_output"]
    assert sha(data / "manifest.json") == cfg["manifest_sha256"]
    assert sha(ROOT / cfg["model"]["checkpoint"]) == cfg["model"]["checkpoint_sha256"]
    manifest = load(data / "manifest.json")
    assert len(manifest["records"]) == 60 and manifest["source_group_overlap"] is False
    for row in manifest["records"]:
        assert sha(data / row["image_file"]) == row["image_sha256"]
        assert sha(data / row["label_file"]) == row["label_sha256"]
    from insplad_adapt_common import start_runtime
    runtime = start_runtime()
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer
    out.mkdir(parents=True, exist_ok=False)
    (out / "code").mkdir()
    for path in [cfg_path, Path(__file__), ROOT / "scripts/ttpla_steelwork_demo.sbatch",
                 ROOT / "scripts/prepare_ttpla_steelwork_demo.py", ROOT / "scripts/component_mask_metrics.py"]:
        shutil.copyfile(path, out / "code" / path.name)
    result = {"status": "TRAINING", "runtime": runtime, "config": cfg,
              "config_sha256": sha(cfg_path), "manifest_sha256": sha(data / "manifest.json"),
              "epoch_losses": [], "records": [], "test_used_for_training_or_selection": False,
              "uk_images": 0, "scope_warning": cfg["scope_warning"]}
    write(out / "results.json", result)
    start = time.perf_counter()
    try:
        model = YOLOE(str(ROOT / cfg["model"]["checkpoint"]))
        def on_start(trainer):
            expected_train = {str(data / r["image_file"]) for r in manifest["records"] if r["split"] == "train"}
            expected_val = {str(data / r["image_file"]) for r in manifest["records"] if r["split"] == "val"}
            assert set(trainer.train_loader.dataset.im_files) == expected_train
            assert set(trainer.test_loader.dataset.im_files) == expected_val
            result["training_setup"] = {"training_images": sorted(expected_train),
                                        "validation_images": sorted(expected_val),
                                        "args": vars(trainer.args)}
            write(out / "results.json", result)
        def on_batch(trainer):
            if trainer.loss is not None and not torch.isfinite(trainer.loss).all():
                raise ValueError("Non-finite loss")
        def on_epoch(trainer):
            event = {"epoch": trainer.epoch + 1, "losses": trainer.loss_items.detach().cpu().tolist()}
            if not result["epoch_losses"] or result["epoch_losses"][-1]["epoch"] != event["epoch"]:
                result["epoch_losses"].append(event)
            write(out / "results.json", result)
        model.add_callback("on_train_start", on_start)
        model.add_callback("on_train_batch_end", on_batch)
        model.add_callback("on_fit_epoch_end", on_epoch)
        model.train(data=str(data / "dataset.yaml"), trainer=YOLOEPESegTrainer,
                    epochs=cfg["model"]["training_epochs"], imgsz=cfg["model"]["imgsz"],
                    batch=cfg["model"]["batch"], nbs=cfg["model"]["batch"], optimizer="AdamW",
                    lr0=0.0005, lrf=0.1, weight_decay=0.0005, warmup_epochs=1.0,
                    seed=cfg["model"]["seed"], workers=12, amp=False, freeze=0, cache=False,
                    mosaic=0.0, mixup=0.0, copy_paste=0.0, translate=0.05, scale=0.2,
                    fliplr=0.5, hsv_h=0.015, hsv_s=0.3, hsv_v=0.2, overlap_mask=False,
                    mask_ratio=4, device=0, deterministic=True, compile=False,
                    project=str(out), name="training", exist_ok=False, pretrained=True,
                    patience=21, plots=False, verbose=False, save=True, save_period=-1,
                    cos_lr=False, close_mosaic=0, val=True, degrees=0.0, shear=0.0,
                    perspective=0.0, flipud=0.0)
        assert [e["epoch"] for e in result["epoch_losses"]] == list(range(1, 21))
        checkpoint = Path(model.trainer.last)
        result.update(status="PREDICTING", training_seconds=time.perf_counter() - start,
                      selected_checkpoint=str(checkpoint.relative_to(out)),
                      checkpoint_sha256=sha(checkpoint), checkpoint_selection="final epoch")
        write(out / "results.json", result)
        del model
        gc.collect(); torch.cuda.empty_cache()
        models = {"supervised": YOLOE(str(checkpoint)),
                  "open_vocabulary": YOLOE(str(ROOT / cfg["model"]["checkpoint"]))}
        prompt = [cfg["model"]["open_vocabulary_prompt"]]
        models["open_vocabulary"].set_classes(prompt, models["open_vocabulary"].get_text_pe(prompt))
        test_rows = [r for r in manifest["records"] if r["split"] == "test"]
        for row in test_rows:
            source = data / row["image_file"]
            with Image.open(source) as image:
                image = image.convert("RGB")
                assert image.size == (row["width"], row["height"])
                scale = min(1.0, cfg["model"]["imgsz"] / max(image.size))
                size = (round(image.width * scale), round(image.height * scale))
                working = image.resize(size, Image.Resampling.LANCZOS)
            refs = []
            ann = row["source_annotations"]
            for name, segmentation in zip(ann["category_name"], ann["segmentation"]):
                if name != "tower_lattice":
                    continue
                points = [{"x": max(0.0, min(float(row["width"]), float(segmentation[i]))),
                           "y": max(0.0, min(float(row["height"]), float(segmentation[i + 1])))}
                          for i in range(0, len(segmentation), 2)]
                refs.append({"class_id": 0, "polygon": points})
            ref_masks = [raster_polygon(r["polygon"], (row["width"], row["height"]), size) for r in refs]
            record = {"image_id": f'{row["split"]}_{row["dataset_row_index"]:04d}',
                      "file_name": row["file_name"], "source_group": row["source_group"],
                      "source_image_sha256": row["image_sha256"], "reference_count": len(refs),
                      "selection_kind": row["selection_kind"]}
            folder = out / "predictions" / record["image_id"]
            folder.mkdir(parents=True)
            for arm, active in models.items():
                pred = active.predict(working, imgsz=cfg["model"]["imgsz"], conf=0.01, iou=0.5,
                                      max_det=100, retina_masks=True, device=0, half=False, verbose=False)[0]
                boxes = pred.boxes.xyxy.cpu().numpy()
                scores = pred.boxes.conf.cpu().numpy()
                classes = pred.boxes.cls.cpu().numpy().astype(np.int64)
                masks = pred.masks.data.cpu().numpy().astype(np.uint8) if pred.masks is not None else np.zeros((0, size[1], size[0]), dtype=np.uint8)
                assert masks.shape == (len(boxes), size[1], size[0])
                raw = folder / f"{arm}.npz"
                np.savez_compressed(raw, boxes=boxes, scores=scores, classes=classes,
                                    mask_shape=np.array(masks.shape),
                                    mask_bits=np.packbits(masks.reshape((len(boxes), size[0] * size[1])), axis=1))
                predictions = [{"prediction_index": i, "class_id": int(c), "score": float(s),
                                "box_working": b.tolist(), "mask_pixels": int(m.sum())}
                               for i, (b, s, c, m) in enumerate(zip(boxes, scores, classes, masks))]
                metric = mask_matches(predictions, masks.astype(bool), refs, ref_masks,
                                      cfg["model"]["score_threshold"], cfg["model"]["mask_iou_threshold"])
                record[arm] = {"predictions": predictions, "raw_file": str(raw.relative_to(out)),
                               "raw_sha256": sha(raw), "metrics": metric}
            result["records"].append(record)
            write(out / "results.json", result)
        result["summary"] = {arm: counts(result["records"], arm) for arm in models}
        result.update(status="COMPLETE_SMALL_DEVELOPMENT_DEMO", elapsed_seconds=time.perf_counter() - start,
                      outputs_are_model_scores_not_probabilities=True,
                      deployment_claim=False, distribution_pole_steelwork_claim=False)
        write(out / "results.json", result)
        print(json.dumps(result["summary"], indent=2), flush=True)
    except BaseException as exc:
        result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write(out / "results.json", result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ttpla_steelwork_demo_v1.json")
    main(parser.parse_args().config)
