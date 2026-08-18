from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from v23_common import REPORTS, ROOT, WEIGHTS, runtime_env, sha256, write_json
from v25_hybrid_component_detector import DISPLAY, TOWER_PROMPT, expand, greedy, iou, read_yolo

LOCAL_TO_GLOBAL={0:1,1:2,2:3}
GLOBAL_TO_LOCAL={1:0,2:1,3:2}
VP_NAMES={0:"crossarm",1:"insulator",2:"earthwire peak"}


def main():
    from PIL import Image,ImageDraw,ImageFont
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    weight=WEIGHTS/"yoloe-26n-seg.pt"
    if not weight.exists(): raise FileNotFoundError(weight)
    ref_path=ROOT/"data/images/train/POS_190181.jpg"
    target_path=ROOT/"data/images/test/POS_2326530.jpg"

    # Reference visual prompts come only from a training source.
    ref_im=Image.open(ref_path).convert("RGB"); RW,RH=ref_im.size
    ref_labels=read_yolo(ROOT/"data/labels/train/POS_190181.txt",RW,RH)
    ref_components=[x for x in ref_labels if x["class_id"] in GLOBAL_TO_LOCAL]
    vp_boxes=np.array([x["box"] for x in ref_components],dtype=np.float32)
    vp_cls=np.array([GLOBAL_TO_LOCAL[x["class_id"]] for x in ref_components],dtype=np.int64)
    visual_prompts={"bboxes":vp_boxes,"cls":vp_cls}

    # Stage 1 model-generated tower ROI on target, no development component GT used.
    target=Image.open(target_path).convert("RGB"); W,H=target.size
    tower_model=YOLOE(str(weight)); tower_model.set_classes([TOWER_PROMPT])
    tr=tower_model.predict(str(target_path),imgsz=768,conf=0.05,device="cpu",verbose=False)[0]
    if tr.boxes is None or len(tr.boxes)==0: raise RuntimeError("No target tower ROI")
    ti=int(tr.boxes.conf.argmax().item()); tower_box=[float(x) for x in tr.boxes.xyxy[ti].cpu().tolist()]; tower_score=float(tr.boxes.conf[ti].item())
    roi=expand(tower_box,W,H); x1,y1,x2,y2=[int(round(v)) for v in roi]
    crop=target.crop((x1,y1,x2,y2)); crop_path=REPORTS/"v3_0_visual_prompt_target_roi.jpg"; crop.save(crop_path,quality=95)

    # Independent YOLOE instance learns temporary visual class embeddings from training reference examples.
    vp_model=YOLOE(str(weight))
    rr=vp_model.predict(
        str(crop_path),
        refer_image=str(ref_path),
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        imgsz=960,
        conf=0.02,
        device="cpu",
        verbose=False,
    )[0]

    predictions=[]
    mask_counts=[]
    masks=None if rr.masks is None else rr.masks.data.cpu().numpy()
    if rr.boxes is not None:
        for k,(b,s,c) in enumerate(zip(rr.boxes.xyxy.cpu().tolist(),rr.boxes.conf.cpu().tolist(),rr.boxes.cls.cpu().tolist())):
            local=int(c)
            if local not in LOCAL_TO_GLOBAL: continue
            gid=LOCAL_TO_GLOBAL[local]
            predictions.append({"class_id":gid,"label":VP_NAMES[local],"score":float(s),"box":[float(b[0])+x1,float(b[1])+y1,float(b[2])+x1,float(b[3])+y1],"prompt_class_id":local,"mask_pixels_at_predictor_resolution":int((masks[k]>0.5).sum()) if masks is not None and k<len(masks) else None})

    gt_all=read_yolo(ROOT/"data/labels/test/POS_2326530.txt",W,H)
    tower_gt=next(g for g in gt_all if g["class_id"]==0); comp_gt=[g for g in gt_all if g["class_id"]!=0]
    metrics=[greedy(predictions,comp_gt,t) for t in (0.30,0.50)]

    # Custom full-image renderer with genuine model-generated boxes + uncalibrated scores.
    canvas=target.copy(); draw=ImageDraw.Draw(canvas); font=ImageFont.load_default(); colors={1:(50,120,230),2:(220,60,150),3:(230,145,30)}
    draw.rectangle(tuple(int(v) for v in tower_box),outline=(35,180,75),width=3); draw.text((int(tower_box[0]),int(tower_box[1])),f"tower ROI score {tower_score:.3f}",fill=(35,180,75),font=font)
    for p in predictions:
        b=tuple(int(round(v)) for v in p["box"]); color=colors[p["class_id"]]; draw.rectangle(b,outline=color,width=3); draw.text((b[0],b[1]),f"{p['label']} score {p['score']:.3f}",fill=color,font=font)
    out_img=REPORTS/"v3_0_yoloe_visual_prompt_POS_2326530.jpg"; canvas.save(out_img,quality=95)
    rr.save(filename=str(REPORTS/"v3_0_yoloe_visual_prompt_roi_native.jpg"))

    report={
      "evidence_type":"yoloe-26-visual-prompt-component-segmentation",
      "claim_scope":"exploratory development-showcase inference using visual prompts from training source POS_190181 only; model boxes/masks/scores are genuine but scores uncalibrated and masks are pseudo-labels, not GT",
      "reference":{"source":"POS_190181","dimensions":[RW,RH],"n_visual_prompts":len(ref_components),"class_counts":{"crossarm":sum(x["class_id"]==1 for x in ref_components),"insulator":sum(x["class_id"]==2 for x in ref_components),"earthwire_peak":sum(x["class_id"]==3 for x in ref_components)},"boxes":vp_boxes.tolist(),"classes":vp_cls.tolist()},
      "target":{"source":"POS_2326530","semantic_status":"adaptive development showcase; not final holdout","dimensions":[W,H],"tower_box":tower_box,"tower_score":tower_score,"tower_iou_vs_manual_reference":iou(tower_box,tower_gt["box"]),"roi_xyxy":[x1,y1,x2,y2]},
      "n_predictions":len(predictions),"predictions":predictions,"metrics":metrics,
      "weight":{"name":weight.name,"sha256":sha256(weight)},"runtime":runtime_env(),
      "final_holdout_touched":False
    }
    write_json(REPORTS/"v3_0_yoloe_visual_prompt_metrics.json",report)
    print(json.dumps({"reference_prompts":report["reference"]["class_counts"],"n_predictions":len(predictions),"metrics":metrics},indent=2))

if __name__=="__main__": main()
