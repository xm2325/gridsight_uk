from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from v23_common import REPORTS, ROOT, WEIGHTS, runtime_env, sha256, write_json

PROTOCOL = ROOT / "configs/v45_material_visual_prompt_protocol.json"
ANNOTATIONS = ROOT / "data/v4_annotations/material_reference_v44.json"
EVAL_FREEZE = ROOT / "data/v4_holdout/material_eval_freeze_v44.json"
REF_ID = "POS_8090535"
EVAL_ID = "POS_2952166"
CLASS_TO_ID = {"glass": 0, "ceramic_family": 1}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
COLORS = {0: (30, 190, 95), 1: (190, 95, 35)}


def load_json(path: Path):
    return json.loads(path.read_text())


def image_path(rid: str) -> Path:
    p = ROOT / f"reports/v4_4_angle_material_candidates/{rid}.jpg"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def record(data: dict, rid: str):
    return next(r for r in data["records"] if r["record_id"] == rid)


def box_iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0.0,x2-x1)*max(0.0,y2-y1)
    aa=max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1]); bb=max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1]); den=aa+bb-inter
    return inter/den if den else 0.0


def class_nms(rows, threshold):
    out=[]
    for cid in sorted(CLASS_TO_ID.values()):
        cand=sorted([x for x in rows if x["class_id"]==cid], key=lambda x:x["score"], reverse=True)
        while cand:
            best=cand.pop(0); out.append(best)
            cand=[x for x in cand if box_iou(best["box"],x["box"])<threshold]
    return sorted(out,key=lambda x:x["score"],reverse=True)


def metrics(preds, gt, threshold):
    used=set(); matches=[]
    for pi,p in enumerate(sorted(preds,key=lambda x:x["score"],reverse=True)):
        candidates=[]
        for gi,g in enumerate(gt):
            if gi in used or p["class_id"]!=g["class_id"]:
                continue
            candidates.append((box_iou(p["box"],g["box"]),gi))
        if candidates:
            ov,gi=max(candidates)
            if ov>=threshold:
                used.add(gi); matches.append({"pred_index_sorted":pi,"gt_index":gi,"iou":ov,"class_id":p["class_id"],"class_name":p["label"]})
    tp=len(matches); fp=len(preds)-tp; fn=len(gt)-tp
    precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
    per_class={}
    for cid,name in ID_TO_CLASS.items():
        pp=[p for p in preds if p["class_id"]==cid]; gg=[g for g in gt if g["class_id"]==cid]
        mm=metrics_single_class(pp,gg,threshold)
        per_class[name]=mm
    return {"iou_threshold":threshold,"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0.0,"matches":matches,"per_material":per_class}


def metrics_single_class(preds, gt, threshold):
    used=set(); tp=0
    for p in sorted(preds,key=lambda x:x["score"],reverse=True):
        cand=[(box_iou(p["box"],g["box"]),i) for i,g in enumerate(gt) if i not in used]
        if cand:
            ov,i=max(cand)
            if ov>=threshold: used.add(i);tp+=1
    fp=len(preds)-tp;fn=len(gt)-tp;p=tp/(tp+fp) if tp+fp else 0.0;r=tp/(tp+fn) if tp+fn else 0.0
    return {"tp":tp,"fp":fp,"fn":fn,"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0.0,"n_predictions":len(preds),"n_gt":len(gt)}


def make_gt(eval_record):
    return [{"class_id":CLASS_TO_ID[b["material_task_label"]],"label":b["material_task_label"],"box":[float(x) for x in b["xyxy"]],"source_specific_material":b["source_specific_material"]} for b in eval_record["boxes"]]


def prompts_for(ref_record, material=None):
    rows=[b for b in ref_record["boxes"] if material is None or b["material_task_label"]==material]
    if material is None:
        cls=np.asarray([CLASS_TO_ID[b["material_task_label"]] for b in rows],dtype=np.int64)
    else:
        cls=np.zeros(len(rows),dtype=np.int64)
    boxes=np.asarray([b["xyxy"] for b in rows],dtype=np.float32)
    return {"bboxes":boxes,"cls":cls},rows


def rows_from_result(result, local_map):
    rows=[]; masks=None if result.masks is None else result.masks.data.cpu().numpy()
    if result.boxes is None:
        return rows
    for k,(b,s,c) in enumerate(zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist())):
        local=int(c)
        if local not in local_map:
            continue
        cid=local_map[local]
        rows.append({"class_id":cid,"label":ID_TO_CLASS[cid],"score":float(s),"box":[float(x) for x in b],"mask_pixels_at_predictor_resolution":int((masks[k]>0.5).sum()) if masks is not None and k<len(masks) else None})
    return rows


def joint_method(weight, ref_path, target_path, ref_record, cfg):
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    prompts,_=prompts_for(ref_record,None)
    model=YOLOE(str(weight))
    r=model.predict(str(target_path),refer_image=str(ref_path),visual_prompts=prompts,predictor=YOLOEVPSegPredictor,imgsz=cfg["imgsz"],conf=cfg["confidence_floor"],iou=cfg["nms_iou"],max_det=cfg["max_detections_per_pass"],device=cfg["device"],verbose=False)[0]
    rows=rows_from_result(r,{0:0,1:1})
    return class_nms(rows,cfg["nms_iou"])


def isolated_method(weight, ref_path, target_path, ref_record, cfg):
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    all_rows=[]
    for material,cid in CLASS_TO_ID.items():
        prompts,_=prompts_for(ref_record,material)
        model=YOLOE(str(weight))
        r=model.predict(str(target_path),refer_image=str(ref_path),visual_prompts=prompts,predictor=YOLOEVPSegPredictor,imgsz=cfg["imgsz"],conf=cfg["confidence_floor"],iou=cfg["nms_iou"],max_det=cfg["max_detections_per_pass"],device=cfg["device"],verbose=False)[0]
        rows=rows_from_result(r,{0:cid})
        all_rows.extend(rows)
    return class_nms(all_rows,cfg["nms_iou"])


def render(target_path, rows, out_path, method_name):
    from PIL import Image,ImageDraw,ImageFont
    image=Image.open(target_path).convert("RGB");draw=ImageDraw.Draw(image);font=ImageFont.load_default()
    banner="SOURCE-ASSISTED MATERIAL EVAL | UNCALIBRATED MODEL SCORES"
    draw.rectangle((0,0,image.width,20),fill=(255,255,255));draw.text((5,5),banner,fill=(0,0,0),font=font)
    for row in rows:
        b=tuple(int(round(x)) for x in row["box"]);color=COLORS[row["class_id"]]
        draw.rectangle(b,outline=color,width=3)
        draw.text((b[0],max(22,b[1]-12)),f"{row['label']} {row['score']*100:.1f}%",fill=color,font=font)
    out_path.parent.mkdir(parents=True,exist_ok=True);image.save(out_path,quality=95)
    return str(out_path.relative_to(ROOT))


def main():
    protocol=load_json(PROTOCOL); annotations=load_json(ANNOTATIONS); freeze=load_json(EVAL_FREEZE)
    if not protocol["frozen_before_material_inference"] or not freeze["frozen_before_material_model_inference"]:
        raise RuntimeError("Material protocol/evaluation source not frozen")
    ref=record(annotations,REF_ID); ev=record(annotations,EVAL_ID)
    ref_path=image_path(REF_ID); target_path=image_path(EVAL_ID)
    if sha256(ref_path)!=ref["source_sha256"] or sha256(target_path)!=ev["source_sha256"]:
        raise RuntimeError("Source pixel hash mismatch")
    if freeze["source_sha256"]!=ev["source_sha256"]:
        raise RuntimeError("Evaluation freeze hash mismatch")
    weight=WEIGHTS/protocol["model"]["name"]
    if sha256(weight)!=protocol["model"]["sha256"]:
        raise RuntimeError("YOLOE checkpoint differs from frozen protocol")
    cfg=protocol["inference"];gt=make_gt(ev)
    methods={
        "joint_two_class_vpe":joint_method(weight,ref_path,target_path,ref,cfg),
        "class_isolated_vpe":isolated_method(weight,ref_path,target_path,ref,cfg),
    }
    results={}
    for name,rows in methods.items():
        out=REPORTS/f"v4_5_{name}_{EVAL_ID}.jpg"
        results[name]={"n_predictions":len(rows),"predictions":rows,"metrics":[metrics(rows,gt,t) for t in protocol["evaluation"]["box_iou_thresholds"]],"overlay":render(target_path,rows,out,name)}
    report={
        "version":"v4.5-independent-material-visual-prompt-evaluation",
        "evidence_type":"cross-source YOLOE-26 visual-prompt material localisation",
        "protocol":{"path":str(PROTOCOL.relative_to(ROOT)),"sha256":sha256(PROTOCOL)},
        "evaluation_freeze":{"path":str(EVAL_FREEZE.relative_to(ROOT)),"sha256":sha256(EVAL_FREEZE)},
        "annotations":{"path":str(ANNOTATIONS.relative_to(ROOT)),"sha256":sha256(ANNOTATIONS)},
        "reference_source":REF_ID,"evaluation_source":EVAL_ID,"n_reference_boxes":len(ref["boxes"]),"n_eval_gt":len(gt),
        "methods":results,
        "selection_rule_respected":True,
        "material_label_provenance":"source-assisted and visually corroborated; ceramic_family is broader than porcelain",
        "score_semantics":"raw uncalibrated YOLOE model score; percentage-like rendering is display-only",
        "claim_boundary":protocol["claim_boundary"],
        "runtime":runtime_env(),"retired_v3_8_holdout_used":False,
    }
    write_json(REPORTS/"v4_5_material_visual_prompt_eval.json",report)
    print(json.dumps({name:{"n_predictions":x["n_predictions"],"metrics":x["metrics"]} for name,x in results.items()},indent=2))


if __name__=="__main__": main()
