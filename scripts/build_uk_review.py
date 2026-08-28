#!/usr/bin/env python3
"""Verify raw model tensors/coordinate transforms, then build the review workbench."""
import argparse
import json
from pathlib import Path
import shutil

from prepare_keen_components import ROOT,digest,write_json
from keen_component_metrics import validate_predictions
from roihu_demo_ablation import nms
from uk_review_common import pole_regions,map_crop_predictions
from build_keen_components_report import annotated_image,output_counts


def verify_dino_prediction(p,arrays,regions,width,height):
    import numpy as np
    raw=arrays[p["raw_file"]];qi=p["query_index"]
    if p["class_id"]!=raw["class_id"] or p["region"]!=raw["region"]:raise ValueError("Raw tensor class/region mismatch")
    if not 0<=qi<len(raw["boxes"]):raise ValueError("Invalid decoder query")
    if not np.isclose(p["score"],raw["scores"][qi],rtol=2e-5,atol=1e-7):raise ValueError("Score differs from raw model logits")
    h,w=raw["size"];cx,cy,bw,bh=map(float,raw["boxes"][qi])
    box=[max(0.,(cx-bw/2)*w),max(0.,(cy-bh/2)*h),min(float(w),(cx+bw/2)*w),min(float(h),(cy+bh/2)*h)]
    if p["region"]:
        box=map_crop_predictions([{"box":box}],regions[p["region"]-1]["box"],width,height)[0]["box"]
    if not np.allclose(box,p["box"],rtol=0,atol=1e-8):raise ValueError("Crop transform differs from raw model boxes")


def build(run):
    import numpy as np
    report=json.loads((run/"results.json").read_text());cfg=report["config"]
    if report["status"]!="COMPLETED_UNREVIEWED_ANNOTATION_PROPOSALS" or report["completed_images"]!=27 or report["training_started"]:
        raise ValueError("Cannot report an incomplete or trained candidate run")
    if report["performance_metrics"] is not None:raise ValueError("No GT exists for UK accuracy")
    frozen=json.loads((run/"frozen_choices.json").read_text())
    if digest(run/"frozen_choices.json")!=report["frozen_choices_sha256"] or frozen["protocol_sha256"]!=digest(run/"code/uk_component_review_v2.json"):
        raise ValueError("Frozen protocol changed")
    if frozen["ground_truth_used"] or frozen["manual_rois_used"] or frozen["gradient_steps"]:
        raise ValueError("Candidate experiment crossed its boundary")
    manifest=json.loads((run/"uk_manifest.json").read_text())
    if digest(run/"uk_manifest.json")!=cfg["uk_manifest_sha256"] or frozen["image_ids"]!=[r["image_id"] for r in manifest["images"]]:
        raise ValueError("Source list changed")
    prior=ROOT/"runs/keen_components/epri_components_v1_20260827"
    if digest(prior/"results.json")!=cfg["prior_results_sha256"]:raise ValueError("Previous evaluation changed")
    source=ROOT/"data/external/uk_distribution_pilot_v1"
    records={r["image_id"]:r for r in report["records"]}
    if len(records)!=27 or set(records)!=set(frozen["image_ids"]):raise ValueError("Unexpected prediction input set")
    out=run/"report";(out/"images").mkdir(parents=True,exist_ok=True)
    images=[];raw_count=0;prediction_count=0;decoder_predictions=0
    for row in manifest["images"]:
        key=row["image_id"];record=records[key]
        original=source/row["image_file"]
        if digest(original)!=row["sha256"] or record["image_sha256"]!=row["sha256"]:raise ValueError("Original pixels changed")
        arrays={};supervised_crops=[]
        for f in record["model_raw_files"]:
            path=run/f["file"]
            if digest(path)!=f["sha256"]:raise ValueError("Raw model tensor/file changed")
            raw_count+=1
            if path.suffix==".npz":
                with np.load(path,allow_pickle=False) as data:
                    logits=data["token_logits"];scores=(1/(1+np.exp(-np.clip(logits,-80,80)))).max(axis=-1)
                    arrays[f["file"]]={"scores":scores,"boxes":data["boxes_cxcywh"].copy(),
                        "size":data["original_size"].tolist(),"class_id":f["class_id"],"region":f["region"]}
            else:
                payload=json.loads(path.read_text());ri=f["region"];region=record["regions"][ri-1]
                if payload["image_id"]!=key or payload["region"]!=region:raise ValueError("Supervised crop source changed")
                b=region["box"];validate_predictions(payload["raw_predictions"],b[2]-b[0],b[3]-b[1],3)
                ps=[{**p,"region":ri,"source":"epri_supervised_roi","raw_file":f["file"]}
                    for p in payload["raw_predictions"] if p["class_id"] in cfg["roi"]["target_classes"]]
                supervised_crops.extend(map_crop_predictions(ps,b,row["width"],row["height"]))
        payloads={}
        for arm in cfg["arms"]:
            f=record["prediction_files"][arm];path=run/f["file"]
            if digest(path)!=f["sha256"]:raise ValueError("Prediction JSON changed")
            p=json.loads(path.read_text());payloads[arm]=p
            if p["image_sha256"]!=row["sha256"] or p["ground_truth"] or p["manual_rois_used"]:raise ValueError("Prediction source/role mismatch")
            validate_predictions(p["raw_predictions"],row["width"],row["height"],3)
            if nms(p["raw_predictions"],cfg["nms_iou"])!=p["predictions"]:raise ValueError("NMS does not reproduce")
            for prediction in p["raw_predictions"]:
                if prediction["source"]=="grounding_dino":
                    verify_dino_prediction(prediction,arrays,record["regions"],row["width"],row["height"])
                    decoder_predictions+=1
            prediction_count+=1
        if pole_regions(payloads["dino_full"]["raw_predictions"],row["width"],row["height"],cfg["roi"])!=record["regions"]:
            raise ValueError("Automatic region geometry does not reproduce")
        full=payloads["dino_full"]["raw_predictions"];regional=payloads["dino_roi"]["raw_predictions"]
        if regional[:len(full)]!=full or any(p["region"]==0 or p["class_id"] not in [1,2] for p in regional[len(full):]):
            raise ValueError("Full-frame context or regional class filter changed")
        old_path=prior/"uk/predictions/supervised"/(key+".json")
        if digest(old_path)!=record["reused_supervised_prediction_sha256"]:raise ValueError("Previously cached predictions changed")
        old=json.loads(old_path.read_text())
        expected=[{**p,"source":"epri_supervised_full","region":0} for p in old["raw_predictions"] if p["score"]>=cfg["confidence_floor"]]
        expected += [p for p in full if p["class_id"]==0]+supervised_crops
        if expected!=payloads["hybrid_roi"]["raw_predictions"]:raise ValueError("Hybrid assembly does not reproduce")
        target=out/"images"/(key+".jpg")
        if not target.exists():shutil.copyfile(original,target)
        if digest(target)!=row["sha256"]:raise ValueError("Display image differs from original")
        cached=run/"prior_supervised"/(key+".json");cached.parent.mkdir(exist_ok=True);shutil.copyfile(old_path,cached)
        sample={"image_id":key,"width":row["width"],"height":row["height"],"sha256":row["sha256"],
            "image_file":"images/"+target.name,"title":row["title"],"source_page":row["source_page"],
            "credit":row.get("attribution") or row["author"],"license":row["license"],"license_url":row["license_url"],
            "regions":record["regions"],"predictions":{arm:p["predictions"] for arm,p in payloads.items()},
            "raw_files":{arm:"../"+f["file"] for arm,f in record["prediction_files"].items()}}
        sample["predictions"]["supervised"]=old["predictions"];sample["raw_files"]["supervised"]="../prior_supervised/"+cached.name
        images.append(sample)
    bundle={"schema":"gridsight-uk-review-v2","classes":cfg["classes"],"runtime":report["runtime"],
            "images":images,"training_approved":False,"performance_metrics":None,"role":"Unreviewed annotation development",
            "output_counts":{arm:output_counts(images,arm,cfg["classes"],cfg["thresholds_for_display"]) for arm in cfg["arms"]+["supervised"]}}
    write_json(out/"data.json",bundle)
    template=(ROOT/"templates/uk_component_review.html").read_text()
    embedded=json.dumps(bundle,ensure_ascii=False,allow_nan=False,separators=(",",":")).replace("<","\\u003c")
    (out/"index.html").write_text(template.replace("__DATA_JSON__",embedded))
    shutil.copyfile(ROOT/"UK_COMPONENT_ANNOTATION_GUIDE.md",out/"UK_COMPONENT_ANNOTATION_GUIDE.md")
    # Operational task notes are local; a clean source checkout has the public overview.
    notes = ROOT/"UK_REVIEW_V2_STATUS.md"
    if not notes.exists():
        notes = ROOT/"docs/CAPABILITY_LAB.md"
    shutil.copyfile(notes,out/"RESULTS.md")
    default=next(r for r in images if r["image_id"]=="uk_geograph_7106830")
    annotated_image(out/default["image_file"],default["predictions"]["dino_roi"],cfg["classes"],
        "UK DEVELOPMENT | Grounding DINO + auto ROI | unreviewed model outputs",score_threshold=.2).save(out/"example_candidates.jpg",quality=94)
    for ident in ["5722811","6494360","4120413"]:
        example=next(r for r in images if r["image_id"]=="uk_geograph_"+ident)
        annotated_image(out/example["image_file"],example["predictions"]["dino_roi"],cfg["classes"],
            ident+" | DINO + auto ROI | score >= 0.30 | NOT GT",score_threshold=.3).save(out/("review_example_"+ident+".jpg"),quality=93)
    verification={"status":"VERIFIED_RAW_TENSORS_AND_COORDINATE_TRANSFORMS","original_images":len(images),
        "prediction_files":prediction_count,"raw_model_files":raw_count,"decoder_prediction_checks":decoder_predictions,
        "results_sha256":digest(run/"results.json"),"frozen_choices_sha256":digest(run/"frozen_choices.json"),
        "uk_manifest_sha256":digest(run/"uk_manifest.json"),"builder_sha256":digest(__file__),
        "template_sha256":digest(ROOT/"templates/uk_component_review.html"),"html_sha256":digest(out/"index.html"),
        "metrics_computed":False,"training_started":False,"material_classification":False,
        "model_weights_verified":"On Roihu before inference; local verification checks raw output hashes and tensors, not a second weight download"}
    write_json(out/"verification.json",verification)
    print(json.dumps({"event":"UK_REVIEW_REPORT_VERIFIED",**verification,"output_counts":bundle["output_counts"]}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,default=ROOT/"runs/uk_component_review/v2_20260827")
    build(p.parse_args().run)
