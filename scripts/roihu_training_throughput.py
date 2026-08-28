#!/usr/bin/env python3
"""Resource sizing only; no quality claim, no checkpoint selection, no holdout."""
import json
import time

from insplad_adapt_common import ROOT, start_runtime, load_protocol, write_json
from prepare_insplad_adaptation import DEFAULT_DATASET


def main():
    runtime = start_runtime()
    protocol = load_protocol()
    import torch
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPETrainer
    hyper = protocol["adaptation"]
    output = ROOT / "runs/insplad_adaptation/resource_probe" / runtime["slurm_job_id"]
    output.mkdir(parents=True, exist_ok=False)
    epoch_times = []
    epoch_started = None
    model = YOLOE(protocol["detection_yaml"]).load(str(ROOT / protocol["model_checkpoint"]))

    def epoch_start(trainer):
        nonlocal epoch_started
        torch.cuda.synchronize()
        epoch_started = time.perf_counter()

    def epoch_end(trainer):
        torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - epoch_started)
        write_json(output / "timing.json", {
            "status": "EPOCH_TIMED", "runtime": runtime, "images": 320, "batch": 8, "imgsz": 1280,
            "epoch_seconds": epoch_times, "batches": len(trainer.train_loader),
            "purpose": "Resource sizing only. This checkpoint is never a scientific candidate.",
            "heldout_used": False})

    model.add_callback("on_train_epoch_start", epoch_start)
    model.add_callback("on_train_epoch_end", epoch_end)
    model.train(
        data=str(DEFAULT_DATASET / "train.yaml"), trainer=YOLOEPETrainer,
        epochs=1, imgsz=hyper["imgsz"], batch=hyper["batch"], nbs=hyper["nbs"],
        optimizer=hyper["optimizer"], lr0=hyper["lr0"], lrf=hyper["lrf"],
        weight_decay=hyper["weight_decay"], warmup_epochs=0, device=0, seed=17,
        deterministic=True, amp=False, freeze=0, workers=4,
        project=str(output), name="resource_only", exist_ok=False, plots=False, verbose=False,
        save=True, val=False, cache=False, compile=False, close_mosaic=0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, translate=.05, scale=.2, fliplr=.5,
        hsv_h=.015, hsv_s=.3, hsv_v=.2)
    print(json.dumps({"event": "RESOURCE_PROBE_COMPLETE", "output": str(output), "epoch_seconds": epoch_times,
                      "cautious_20_epoch_estimate_seconds": 120 + 1.5 * 20 * max(epoch_times)}), flush=True)


if __name__ == "__main__":
    main()
