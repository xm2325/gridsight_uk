#!/usr/bin/env python3
"""Generate auditable UK annotation proposals. No training and no UK GT metrics."""
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import shutil
import time
import traceback

from prepare_keen_components import ROOT,digest,write_json
from insplad_adapt_common import start_runtime
from keen_component_metrics import validate_predictions
from roihu_demo_ablation import infer,nms
from uk_review_common import pole_regions,map_crop_predictions


def main():
    protocol_path=ROOT/"configs/uk_component_review_v2.json"
    cfg=json.loads(protocol_path.read_text())
    source=ROOT/"data/external/uk_distribution_pilot_v1"
    prior=ROOT/"runs/keen_components/epri_components_v1_20260827"
    if digest(source/"manifest.json")!=cfg["uk_manifest_sha256"] or digest(prior/"results.json")!=cfg["prior_results_sha256"]:
        raise ValueError("Prior sources changed")
    if digest(ROOT/cfg["weight_file"])!=cfg["weight_sha256"] or digest(ROOT/cfg["epri_checkpoint"])!=cfg["epri_checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    weight_dir=(ROOT/cfg["weight_file"]).parent
    weights=json.loads((weight_dir/"verified_manifest.json").read_text())
    for f in weights["files"]:
        if digest(weight_dir/f["file"])!=f["sha256"]:raise ValueError("Model release file changed")
    manifest=json.loads((source/"manifest.json").read_text())
    if len(manifest["images"])!=27:raise ValueError("Expected all 27 UK development images")
    for r in manifest["images"]:
        if digest(source/r["image_file"])!=r["sha256"]:raise ValueError("Original image changed")
    output=ROOT/"runs/uk_component_review/v2_20260827"
    if output.exists():raise FileExistsError("Inspect existing output; do not rerun completed candidates")
    runtime=start_runtime()
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
    from ultralytics import YOLOE
    output.mkdir(parents=True,exist_ok=False)
    for p in [protocol_path,Path(__file__),ROOT/"scripts/uk_review_common.py",ROOT/"scripts/roihu_demo_ablation.py",ROOT/"scripts/roihu_uk_review_candidates.sbatch"]:
        (output/"code").mkdir(exist_ok=True);shutil.copyfile(p,output/"code"/p.name)
    shutil.copyfile(source/"manifest.json",output/"uk_manifest.json")
    shutil.copyfile(weight_dir/"verified_manifest.json",output/"model_manifest.json")
    frozen={"protocol_sha256":digest(protocol_path),"uk_manifest_sha256":cfg["uk_manifest_sha256"],
            "weights_sha256":cfg["weight_sha256"],"epri_checkpoint_sha256":cfg["epri_checkpoint_sha256"],
            "frozen_at":datetime.now(timezone.utc).isoformat(),"image_ids":[r["image_id"] for r in manifest["images"]],
            "ground_truth_used":False,"manual_rois_used":False,"gradient_steps":0}
    write_json(output/"frozen_choices.json",frozen)
    report={"status":"LOADING_MODELS","runtime":{**runtime,"transformers":transformers.__version__},
            "config":cfg,"frozen_choices_sha256":digest(output/"frozen_choices.json"),"completed_images":0,
            "performance_metrics":None,"records":[],"started_at":datetime.now(timezone.utc).isoformat()}
    write_json(output/"results.json",report)
    started=time.perf_counter()
    try:
        processor=AutoProcessor.from_pretrained(weight_dir,local_files_only=True,trust_remote_code=False)
        dino=AutoModelForZeroShotObjectDetection.from_pretrained(weight_dir,local_files_only=True,
            trust_remote_code=False,use_safetensors=True,disable_custom_kernels=True).to("cuda").eval()
        detector=YOLOE(str(ROOT/cfg["epri_checkpoint"])).to("cuda")
        expected_names=dict(enumerate(cfg["classes"]))
        if detector.names!=expected_names:raise ValueError("Wrong adapted detector class mapping")
        report["status"]="GENERATING_UNREVIEWED_PROPOSALS";write_json(output/"results.json",report)

        def ground(photo,image_id,region_index,class_ids):
            predictions=[];files=[]
            for cls in class_ids:
                inputs=processor(images=photo,text=cfg["text_queries"][cls],size=cfg["image_size"],return_tensors="pt").to("cuda")
                with torch.inference_mode():raw=dino(**inputs)
                logits=raw.logits[0].float().cpu().numpy()
                boxes=raw.pred_boxes[0].float().cpu().numpy()
                scores=raw.logits[0].sigmoid().amax(-1).float().cpu().numpy()
                raw_name=f"model_raw/{image_id}/r{region_index}_c{cls}.npz"
                raw_path=output/raw_name;raw_path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(raw_path,token_logits=logits,boxes_cxcywh=boxes,
                    input_ids=inputs["input_ids"].cpu().numpy(),original_size=np.array([photo.height,photo.width]))
                files.append({"file":raw_name,"sha256":digest(raw_path),"class_id":cls,"region":region_index})
                for query_index,(b,score) in enumerate(zip(boxes,scores)):
                    if float(score)<cfg["confidence_floor"]:continue
                    cx,cy,bw,bh=map(float,b)
                    box=[max(0.,(cx-bw/2)*photo.width),max(0.,(cy-bh/2)*photo.height),
                         min(float(photo.width),(cx+bw/2)*photo.width),min(float(photo.height),(cy+bh/2)*photo.height)]
                    if box[2]<=box[0] or box[3]<=box[1]:continue
                    predictions.append({"class_id":cls,"score":float(score),"box":box,"region":region_index,
                        "source":"grounding_dino","raw_file":raw_name,"query_index":query_index})
                del raw,inputs
            return predictions,files

        for index,row in enumerate(manifest["images"]):
            image_id=row["image_id"]
            with Image.open(source/row["image_file"]) as f:photo=f.convert("RGB")
            torch.cuda.synchronize();t0=time.perf_counter()
            full,raw_files=ground(photo,image_id,0,range(3))
            validate_predictions(full,photo.width,photo.height,3)
            regions=pole_regions(full,photo.width,photo.height,cfg["roi"])
            local_dino=[];local_supervised=[]
            for ri,region in enumerate(regions,1):
                cropped=photo.crop(region["box"])
                ps,files=ground(cropped,image_id,ri,cfg["roi"]["target_classes"])
                raw_files.extend(files)
                local_dino.extend(map_crop_predictions(ps,region["box"],photo.width,photo.height))
                raw,_,_=infer(detector,cropped,{"imgsz":1280,"tiled":False},0,cfg["confidence_floor"])
                raw_name=f"model_raw/{image_id}/r{ri}_supervised.json"
                write_json(output/raw_name,{"image_id":image_id,"region":region,"raw_predictions":raw})
                raw_files.append({"file":raw_name,"sha256":digest(output/raw_name),"region":ri})
                ps=[{**p,"region":ri,"source":"epri_supervised_roi","raw_file":raw_name}
                    for p in raw if p["class_id"] in cfg["roi"]["target_classes"]]
                local_supervised.extend(map_crop_predictions(ps,region["box"],photo.width,photo.height))
            old_path=prior/"uk/predictions/supervised"/(image_id+".json")
            old=json.loads(old_path.read_text())
            if old["image_sha256"]!=row["sha256"]:raise ValueError("Cached prediction source changed")
            old_ps=[{**p,"source":"epri_supervised_full","region":0} for p in old["raw_predictions"]
                    if p["score"]>=cfg["confidence_floor"]]
            arms={"dino_full":full,"dino_roi":full+local_dino,
                  "hybrid_roi":old_ps+[p for p in full if p["class_id"]==0]+local_supervised}
            prediction_files={}
            for arm,raw in arms.items():
                validate_predictions(raw,photo.width,photo.height,3)
                merged=nms(raw,cfg["nms_iou"])
                name=f"predictions/{arm}/{image_id}.json"
                write_json(output/name,{"image_id":image_id,"image_sha256":row["sha256"] ,"arm":arm,
                    "class_names":cfg["classes"],"raw_predictions":raw,"predictions":merged,"regions":regions,
                    "confidence_floor":cfg["confidence_floor"],"manual_rois_used":False,
                    "ground_truth":False,"material_status":"not classified","status":"UNREVIEWED_MACHINE_PROPOSALS"})
                prediction_files[arm]={"file":name,"sha256":digest(output/name)}
            torch.cuda.synchronize()
            record={"image_id":image_id,"image_sha256":row["sha256"],"prediction_files":prediction_files,
                    "model_raw_files":raw_files,"regions":regions,"elapsed_seconds":time.perf_counter()-t0,
                    "reused_supervised_prediction_sha256":digest(old_path)}
            report["records"].append(record);report["completed_images"]=index+1
            write_json(output/"results.json",report)
            print(json.dumps({"event":"UK_IMAGE_COMPLETED","index":index+1,"image_id":image_id,"roi_count":len(regions)}),flush=True)
        report.update(status="COMPLETED_UNREVIEWED_ANNOTATION_PROPOSALS",elapsed_seconds=time.perf_counter()-started,
                      prediction_files=27*len(cfg["arms"]),training_started=False)
        write_json(output/"results.json",report)
        print(json.dumps({"event":"UK_REVIEW_COMPLETED","prediction_files":report["prediction_files"],
                          "elapsed_seconds":report["elapsed_seconds"],"run":str(output)}),flush=True)
    except BaseException as exc:
        report.update(status="FAILED",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        write_json(output/"results.json",report);raise


if __name__=="__main__":main()
