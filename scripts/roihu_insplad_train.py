#!/usr/bin/env python3
"""Gated two-image reconstruction, then one fixed supervised adaptation run."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import time
from pathlib import Path

from prepare_insplad_adaptation import DEFAULT_DATASET, verify_dataset
from prepare_insplad100 import verify_dataset as verify_diagnostic
from insplad_adapt_common import (ROOT, PROTOCOL, load_protocol, start_runtime, create_run, run_predictions,
                                  overfit_gate, verify_records, select_prompt, digest, write_json, summarize)


def check_gate(path, manifest, protocol):
    report = json.loads(path.read_text())
    if report["status"] != "OVERFIT_GATE_PASSED" or report["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("No passing reconstruction gate for the frozen protocol")
    if report["dataset_manifest_sha256"] != digest(DEFAULT_DATASET / "manifest.json"):
        raise ValueError("Overfit gate belongs to a different dataset")
    ids = set(manifest["overfit_image_ids"])
    if {r["image_id"] for r in report["results"]} != ids:
        raise ValueError("Overfit gate did not use the fixed training examples")
    verify_records(report["results"], manifest["images"], path.parent / "training_check", protocol)
    if not overfit_gate(report["results"], protocol)["passed"]:
        raise ValueError("Stored overfit gate does not recompute")
    return report


def check_prompt_selection(path, protocol):
    selection = json.loads(path.read_text())
    report = json.loads((path.parent / "results.json").read_text())
    if report["status"] != "COMPLETED_PROMPT_DIAGNOSTIC" or selection["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Prompt selection is incomplete or uses another protocol")
    diagnostic = verify_diagnostic(ROOT / "data/external/insplad100")
    verify_records(report["results"], diagnostic["images"], path.parent, protocol)
    winner, _ = select_prompt(report["results"], protocol)
    if winner != selection["selected_prompt"] or selection["specification"] != protocol["prompt_arms"][winner]:
        raise ValueError("Selected prompt differs from fixed diagnostic criterion")
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["overfit", "adapt"])
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--prompt-selection", type=Path)
    args = parser.parse_args()
    protocol = load_protocol()
    manifest = verify_dataset(DEFAULT_DATASET, protocol)
    if args.stage == "adapt":
        if args.gate is None or args.prompt_selection is None:
            parser.error("Adaptation requires the passing gate and frozen prompt selection")
        check_gate(args.gate, manifest, protocol)
        selection = check_prompt_selection(args.prompt_selection, protocol)
        addendum_path = ROOT / "configs/insplad_adapt_control_v1.json"
        addendum = json.loads(addendum_path.read_text())
        if addendum["base_protocol_sha256"] != digest(PROTOCOL):
            raise ValueError("Control addendum does not match base protocol")
        heldout_lock = ROOT / "runs/insplad_adaptation" / f"heldout-{digest(DEFAULT_DATASET / 'manifest.json')}.json"
        if heldout_lock.exists():
            raise FileExistsError("This dataset already has a heldout evaluation receipt; do not retrain or retest automatically")
    runtime = start_runtime()
    import torch
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPETrainer

    class NoAugmentationTrainer(YOLOEPETrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            # Setting train(augment=False) alone does not disable YOLO's training
            # transforms. Rebuild with dataset.augment=False: LetterBox + Format.
            dataset.augment = False
            dataset.transforms = dataset.build_transforms(hyp=self.args)
            return dataset

    output, report = create_run(args.stage, protocol, runtime)
    report.update(dataset_manifest_sha256=digest(DEFAULT_DATASET / "manifest.json"), script_sha256=digest(__file__),
                  training_examples=manifest["overfit_image_ids"] if args.stage == "overfit" else 320,
                  initial_checkpoint=str(ROOT / protocol["model_checkpoint"]),
                  initial_checkpoint_sha256=digest(ROOT / protocol["model_checkpoint"]))
    write_json(output / "dataset_manifest.json", manifest)
    if args.stage == "adapt":
        report.update(gate_source=str(args.gate), gate_source_sha256=digest(args.gate),
                      prompt_selection=selection, prompt_selection_sha256=digest(args.prompt_selection),
                      control_addendum=addendum, control_addendum_sha256=digest(addendum_path))
    started = time.perf_counter()
    try:
        hyper = protocol["overfit" if args.stage == "overfit" else "adaptation"]
        model = YOLOE(protocol["detection_yaml"]).load(str(ROOT / protocol["model_checkpoint"]))
        trainer_type = NoAugmentationTrainer if args.stage == "overfit" else YOLOEPETrainer
        train_kwargs = dict(data=str(DEFAULT_DATASET / ("overfit.yaml" if args.stage == "overfit" else "train.yaml")),
                            trainer=trainer_type, epochs=hyper["epochs"], imgsz=hyper["imgsz"], batch=hyper["batch"],
                            nbs=hyper["nbs"], optimizer=hyper["optimizer"], lr0=hyper["lr0"], lrf=hyper["lrf"],
                            weight_decay=hyper["weight_decay"], warmup_epochs=hyper["warmup_epochs"],
                            device=0, seed=17, deterministic=True, amp=False, freeze=0, workers=4,
                            project=str(output), name="training", exist_ok=False, pretrained=True,
                            patience=hyper["epochs"] + 1, plots=False, verbose=False, save=True, save_period=-1,
                            cache=False, compile=False, cos_lr=False, close_mosaic=0,
                            val=args.stage == "adapt", mosaic=0.0, mixup=0.0, copy_paste=0.0,
                            degrees=0.0, shear=0.0, perspective=0.0, flipud=0.0,
                            translate=0.0, scale=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
        if args.stage == "adapt":
            for key in ("mosaic", "mixup", "copy_paste", "translate", "scale", "fliplr", "hsv_h", "hsv_s", "hsv_v"):
                train_kwargs[key] = hyper[key]

        def on_start(trainer):
            dataset = trainer.train_loader.dataset
            transforms = [type(t).__name__ for t in dataset.transforms.transforms]
            report["training_setup"] = {"args": vars(trainer.args), "class_names": trainer.model.names,
                "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
                "frozen_parameters": sum(p.numel() for p in trainer.model.parameters() if not p.requires_grad),
                "trainable_names": [name for name, p in trainer.model.named_parameters() if p.requires_grad],
                "batches_per_epoch": len(trainer.train_loader), "accumulate": trainer.accumulate,
                "dataset_augment": dataset.augment, "transform_types": transforms,
                "actual_training_images": [str(path) for path in dataset.im_files]}
            if args.stage == "overfit" and (dataset.augment or transforms != ["LetterBox", "Format"]):
                raise RuntimeError("Overfit augmentations were not completely disabled")
            if len(dataset.im_files) != (2 if args.stage == "overfit" else 320):
                raise RuntimeError("Trainer loaded an unexpected number of images")
            if trainer.model.names != {0: "insulator"}:
                raise RuntimeError("Wrong training class mapping")
            if args.stage == "adapt":
                control_path = output / "untrained_detector.pt"
                torch.save({"model": deepcopy(trainer.model).float().cpu().eval(),
                            "train_args": vars(trainer.args), "epoch": -1, "optimizer": None,
                            "updates": 0, "ema": None}, control_path)
                report["untrained_control"] = {"path": str(control_path), "sha256": digest(control_path),
                                                "optimizer_steps": 0, "saved_at": "on_train_start"}
            write_json(output / "results.json", report)

        def on_batch_end(trainer):
            if trainer.loss is not None and not torch.isfinite(trainer.loss).all():
                raise RuntimeError("Non-finite training loss")

        def on_epoch_end(trainer):
            losses = trainer.loss_items.detach().cpu().tolist() if trainer.loss_items is not None else []
            report["training_progress"] = {"completed_epochs": trainer.epoch + 1, "last_batch_losses": losses,
                "validation_metrics": {key: float(value) for key, value in trainer.metrics.items()},
                "best_fitness": float(trainer.best_fitness or 0)}
            write_json(output / "results.json", report)
            if (trainer.epoch + 1) % 20 == 0 or args.stage == "adapt":
                print(json.dumps({"event": "TRAINING_PROGRESS", **report["training_progress"]}), flush=True)

        model.add_callback("on_train_start", on_start)
        model.add_callback("on_train_batch_end", on_batch_end)
        model.add_callback("on_fit_epoch_end", on_epoch_end)
        model.train(**train_kwargs)
        if report.get("training_progress", {}).get("completed_epochs") != hyper["epochs"]:
            raise RuntimeError("Training did not complete its fixed epoch budget")
        checkpoint = Path(model.trainer.best)
        if not checkpoint.is_file():
            raise FileNotFoundError("No checkpoint selected by the configured training routine")
        report.update(selected_checkpoint=str(checkpoint), selected_checkpoint_sha256=digest(checkpoint),
                      training_elapsed_seconds=time.perf_counter() - started)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = YOLOE(str(checkpoint)).to("cuda:0")
        if model.names != {0: "insulator"}:
            raise RuntimeError("Reloaded checkpoint class names differ")

        if args.stage == "overfit":
            rows = [r for r in manifest["images"] if r["image_id"] in manifest["overfit_image_ids"]]
            local_protocol = {**protocol, "inference_imgsz": hyper["imgsz"]}
            report["results"] = run_predictions(model, rows, DEFAULT_DATASET, output / "training_check",
                                                 "overfit", [0], local_protocol)
            verify_records(report["results"], rows, output / "training_check", protocol)
            report["gate"] = overfit_gate(report["results"], protocol)
            report["gate"]["inference_imgsz"] = hyper["imgsz"]
            report["status"] = "OVERFIT_GATE_PASSED" if report["gate"]["passed"] else "OVERFIT_GATE_FAILED"
            if not report["gate"]["passed"]:
                print(json.dumps({"event": "GATE_FAILED", "gate": report["gate"]}), flush=True)
        else:
            # Freeze all model choices before any inference on new heldout families.
            receipt = output / "frozen_evaluation_choices.json"
            write_json(receipt, {"checkpoint_sha256": digest(checkpoint), "prompt_selection": selection,
                                "untrained_control_sha256": report["untrained_control"]["sha256"],
                                "control_addendum_sha256": report["control_addendum_sha256"],
                                "dataset_manifest_sha256": report["dataset_manifest_sha256"],
                                "protocol_sha256": digest(PROTOCOL), "selection_uses_holdout": False})
            report["evaluation_choices_sha256"] = digest(receipt)
            report["evaluations"] = {}
            diagnostic_dir = ROOT / "data/external/insplad100"
            diagnostic = verify_diagnostic(diagnostic_dir)
            diagnostic_output = output / "diagnostic"
            diagnostic_results = run_predictions(model, diagnostic["images"], diagnostic_dir, diagnostic_output,
                                                   "supervised", [0], protocol)
            control = YOLOE(str(output / "untrained_detector.pt")).to("cuda:0")
            diagnostic_results += run_predictions(control, diagnostic["images"], diagnostic_dir, diagnostic_output,
                                                  "untrained_detector", [0], protocol)
            verify_records(diagnostic_results, diagnostic["images"], diagnostic_output, protocol)
            report["evaluations"]["diagnostic"] = {"results": diagnostic_results,
                "summary": summarize(diagnostic_results, ["supervised", "untrained_detector"], protocol),
                "scope": "Previously observed 100-image diagnostic, not an untouched test"}
            write_json(output / "results.json", report)
            holdout = [r for r in manifest["images"] if r["split"] == "holdout"]
            lock = ROOT / "runs/insplad_adaptation" / f"heldout-{report['dataset_manifest_sha256']}.json"
            with lock.open("x") as file:
                json.dump({"status": "EVALUATION_STARTED", "run": str(output),
                           "frozen_choices_sha256": report["evaluation_choices_sha256"]}, file, indent=2)
            holdout_output = output / "holdout"
            records = run_predictions(model, holdout, DEFAULT_DATASET, holdout_output, "supervised", [0], protocol)
            records += run_predictions(control, holdout, DEFAULT_DATASET, holdout_output, "untrained_detector", [0], protocol)
            del control
            del model
            gc.collect()
            torch.cuda.empty_cache()
            arms = ["supervised", "untrained_detector"]
            for label, prompt_name in (("original_prompt", "long_multi"), ("selected_prompt", selection["selected_prompt"])):
                if label == "selected_prompt" and prompt_name == "long_multi":
                    # The fixed selection retained the baseline. Show the alias
                    # transparently without spending GPU time on identical work.
                    (holdout_output / "predictions" / label).mkdir(parents=True)
                    aliases = []
                    for source_record in records:
                        if source_record["arm"] != "original_prompt":
                            continue
                        stored = json.loads((holdout_output / source_record["prediction_file"]).read_text())
                        stored.update(arm=label, alias_of="original_prompt")
                        path = f"predictions/{label}/{source_record['image_id']}.json"
                        write_json(holdout_output / path, stored)
                        aliases.append({**source_record, "arm": label, "prediction_file": path,
                                        "reused": True, "alias_of": "original_prompt"})
                    if len(aliases) != 100:
                        raise ValueError("Missing baseline predictions for selected-prompt alias")
                    records += aliases
                    arms.append(label)
                    continue
                specification = protocol["prompt_arms"][prompt_name]
                original = YOLOE(str(ROOT / protocol["model_checkpoint"])).to("cuda:0")
                original.set_classes(specification["prompts"])
                records += run_predictions(original, holdout, DEFAULT_DATASET, holdout_output, label,
                                           specification["target_ids"], protocol)
                del original
                gc.collect()
                torch.cuda.empty_cache()
                arms.append(label)
            verify_records(records, holdout, holdout_output, protocol)
            report["evaluations"]["holdout"] = {"results": records, "summary": summarize(records, arms, protocol),
                "scope": protocol["scope"], "independent_of_training_families": True,
                "used_for_prompt_or_checkpoint_selection": False}
            report["status"] = "COMPLETED_FIXED_ADAPTATION_AND_HELDOUT_EVALUATION"
            write_json(lock, {"status": "COMPLETED", "run": str(output), "records": len(records),
                              "frozen_choices_sha256": report["evaluation_choices_sha256"]})
    except Exception as error:
        report["status"] = "FAILED_PARTIAL_TRAINING_OR_EVALUATION"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "results.json", report)
    print(json.dumps({"event": "RUN_COMPLETE", "status": report["status"], "output": str(output),
                      "gate": report.get("gate"),
                      "evaluation_summary": {name: data["summary"] for name, data in report.get("evaluations", {}).items()}}), flush=True)


if __name__ == "__main__":
    main()
