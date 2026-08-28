#!/usr/bin/env python3
"""One reference image, 26 full-frame targets, no gradient training or GT metrics."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

from prepare_keen_components import ROOT, digest, write_json
from keen_component_metrics import validate_predictions
from insplad_adapt_common import start_runtime
from roihu_keen_components import predict_rows


def main():
    protocol_path=ROOT/"configs/keen_uk_visual_prompt_v1.json"
    protocol=json.loads(protocol_path.read_text())
    source=ROOT/"data/external/uk_distribution_pilot_v1"
    if digest(source/"manifest.json")!=protocol["uk_manifest_sha256"]:
        raise ValueError("UK source manifest changed")
    manifest=json.loads((source/"manifest.json").read_text())
    reference=next(r for r in manifest["images"] if r["image_id"]==protocol["reference"]["image_id"])
    reference_path=source/reference["image_file"]
    if digest(reference_path)!=protocol["reference"]["image_sha256"]:
        raise ValueError("Reference pixels changed")
    for b in protocol["reference"]["boxes"]:
        validate_predictions([{**b,"score":1.}],reference["width"],reference["height"],3)
    rows=[r for r in manifest["images"] if r["image_id"]!=reference["image_id"]]
    if len(rows)!=26 or digest(ROOT/protocol["base_checkpoint"])!=protocol["base_checkpoint_sha256"]:
        raise ValueError("Target count or original weights changed")
    prior=ROOT/"runs/keen_components/epri_components_v1_20260827/results.json"
    if digest(prior)!=protocol["prior_experiment_results_sha256"]:
        raise ValueError("Prior experiment changed")
    output=ROOT/"runs/keen_components/uk_visual_prompt_v1_20260827"
    if output.exists():
        raise FileExistsError("Visual-prompt diagnostic already exists; do not run it twice")
    runtime=start_runtime()
    import numpy as np
    import torch
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    output.mkdir(parents=True,exist_ok=False)
    code=output/"code";code.mkdir()
    snapshot_paths=[protocol_path,Path(__file__),*[ROOT/"scripts"/name for name in
                    ["roihu_keen_visual_prompt.sbatch","roihu_keen_components.py","roihu_demo_ablation.py",
                     "keen_component_metrics.py","prepare_keen_components.py","insplad_adapt_common.py"]]]
    for p in snapshot_paths:
        shutil.copyfile(p,code/p.name)
    write_json(output/"reference.json",protocol["reference"])
    write_json(output/"targets.json",rows)
    report={"status":"PREPARING_VISUAL_EMBEDDINGS", "protocol":protocol,"protocol_sha256":digest(protocol_path),
            "runtime":runtime,"reference":reference,"target_count":len(rows),"gradient_steps":0,"performance_metrics":None,
            "started_at":datetime.now(timezone.utc).isoformat(),
            "source_snapshots":{p.name:digest(p) for p in snapshot_paths}}
    write_json(output/"results.json",report)
    start=time.perf_counter()
    try:
        model=YOLOE(str(ROOT/protocol["base_checkpoint"])).to("cuda:0")
        prompts={"bboxes":np.array([b["box"] for b in protocol["reference"]["boxes"]],dtype=np.float32),
                 "cls":np.array([b["class_id"] for b in protocol["reference"]["boxes"]],dtype=np.int64)}
        # The only image supplying prompt boxes is the separate reference. Blank
        # synthetic input initializes the predictor without consuming a target.
        model.predict(np.zeros((720,1280,3),dtype=np.uint8),refer_image=str(reference_path),visual_prompts=prompts,
                      predictor=YOLOEVPSegPredictor,imgsz=1280,conf=.001,iou=.5,agnostic_nms=False,
                      half=False,device=0,verbose=False,max_det=300)
        names=model.names
        actual=list(names.values()) if isinstance(names,dict) else list(names)
        if actual!=["object0","object1","object2"]:
            raise ValueError(f"Unexpected visual class mapping: {actual}")
        frozen=output/"reference_conditioned_model.pt"
        torch.save({"model":deepcopy(model.model).float().cpu().eval(),"train_args":model.overrides,
                    "epoch":-1,"optimizer":None,"ema":None,"gradient_steps":0},frozen)
        choices={"protocol_sha256":digest(protocol_path),"reference_image_sha256":reference["sha256"],
                 "reference_boxes":protocol["reference"]["boxes"],"conditioned_model_sha256":digest(frozen),
                 "target_ids":[r["image_id"] for r in rows],"reference_in_targets":False,"target_gt_used":False,
                 "frozen_at":datetime.now(timezone.utc).isoformat()}
        write_json(output/"frozen_choices.json",choices)
        result=predict_rows(model,rows,source,output,"visual_prompt",protocol,labelled=False)
        counts={}
        for threshold in protocol["display_thresholds"]:
            by_class=[0,0,0];nonempty=0
            for r in result["records"]:
                p=json.loads((output/r["prediction_file"]).read_text())
                ps=[x for x in p["predictions"] if x["score"]>=threshold]
                nonempty+=bool(ps)
                for x in ps:by_class[x["class_id"]]+=1
            counts[f"{threshold:.2f}"]={"images_with_output":nonempty,"images_without_output":len(rows)-nonempty,
                                      "boxes_by_class":dict(zip(protocol["classes"],by_class)),"not_accuracy":True}
        report.update(status="COMPLETED_QUALITATIVE_VISUAL_PROMPT_DIAGNOSTIC",elapsed_seconds=time.perf_counter()-start,
                      conditioned_model_sha256=digest(frozen),frozen_choices_sha256=digest(output/"frozen_choices.json"),
                      verified_prediction_count=result["verified_prediction_count"],output_counts=counts)
        write_json(output/"results.json",report)
        print(json.dumps({"event":"VISUAL_PROMPT_COMPLETED","output":str(output),"counts":counts}),flush=True)
    except BaseException as exc:
        report.update(status="FAILED",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        write_json(output/"results.json",report)
        raise


if __name__=="__main__":main()
