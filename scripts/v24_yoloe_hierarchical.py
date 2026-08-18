from __future__ import annotations

import json
from pathlib import Path

from v23_common import REPORTS, ROOT, SHOWCASE, WEIGHTS, dataset_manifest, runtime_env, sha256, write_json

TOWER_PROMPT = "steel lattice electricity transmission pylon"
COMPONENT_PROMPTS = [
    "crossarm on an electricity transmission tower",
    "power line insulator string",
    "earth wire peak at the top of a transmission tower",
]
COMPONENT_TO_GLOBAL = {0: 1, 1: 2, 2: 3}
DISPLAY = {0: "steelwork", 1: "crossarm", 2: "insulator", 3: "earthwire peak"}


def read_gt(path: Path, width: int, height: int):
    out=[]
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        c,xc,yc,w,h=map(float,line.split())
        out.append({
            "class_id": int(c),
            "box": [(xc-w/2)*width,(yc-h/2)*height,(xc+w/2)*width,(yc+h/2)*height],
        })
    return out


def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0.0,x2-x1)*max(0.0,y2-y1)
    aa=max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1]); bb=max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1])
    den=aa+bb-inter
    return inter/den if den else 0.0


def greedy_metrics(preds, gt, threshold: float, class_aware: bool):
    order=sorted(range(len(preds)), key=lambda i: preds[i]["score"], reverse=True)
    used=set(); matches=[]
    for i in order:
        p=preds[i]
        cand=[]
        for j,g in enumerate(gt):
            if j in used: continue
            if class_aware and p["class_id"] != g["class_id"]: continue
            cand.append((iou(p["box"],g["box"]),j))
        if cand:
            best,j=max(cand)
            if best>=threshold:
                used.add(j); matches.append({"pred_index":i,"gt_index":j,"iou":best})
    tp=len(matches); fp=len(preds)-tp; fn=len(gt)-tp
    precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
    return {"threshold":threshold,"class_aware":class_aware,"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0.0,"matches":matches}


def main():
    from PIL import Image, ImageDraw, ImageFont
    from ultralytics import YOLOE

    weight=WEIGHTS/"yoloe-26n-seg.pt"
    if not weight.exists(): raise FileNotFoundError(weight)
    image=Image.open(SHOWCASE).convert("RGB")
    W,H=image.size
    model=YOLOE(str(weight))

    # Stage 1: model-generated tower ROI, no held-out bbox is used for inference.
    model.set_classes([TOWER_PROMPT])
    tower_result=model.predict(str(SHOWCASE),imgsz=768,conf=0.05,device="cpu",verbose=False)[0]
    tower_candidates=[]
    if tower_result.boxes is not None:
        for b,s in zip(tower_result.boxes.xyxy.cpu().tolist(),tower_result.boxes.conf.cpu().tolist()):
            tower_candidates.append({"class_id":0,"label":"steelwork","score":float(s),"box":[float(x) for x in b],"stage":"tower"})
    if not tower_candidates:
        raise RuntimeError("Stage-1 YOLOE produced no tower proposal")
    tower=max(tower_candidates,key=lambda x:x["score"])

    # Crop with a small geometry-only padding around the model proposal.
    x1,y1,x2,y2=tower["box"]
    pad_x=0.08*(x2-x1); pad_y=0.04*(y2-y1)
    cx1=max(0,int(x1-pad_x)); cy1=max(0,int(y1-pad_y)); cx2=min(W,int(x2+pad_x)); cy2=min(H,int(y2+pad_y))
    crop=image.crop((cx1,cy1,cx2,cy2))
    crop_path=REPORTS/"v2_4_yoloe_tower_roi_POS_2326530.jpg"
    crop.save(crop_path,quality=94)

    # Stage 2: components inside model-generated ROI at higher inference resolution.
    model.set_classes(COMPONENT_PROMPTS)
    comp_result=model.predict(str(crop_path),imgsz=1280,conf=0.025,device="cpu",verbose=False)[0]
    components=[]
    if comp_result.boxes is not None:
        for b,s,c in zip(comp_result.boxes.xyxy.cpu().tolist(),comp_result.boxes.conf.cpu().tolist(),comp_result.boxes.cls.cpu().tolist()):
            local=int(c); global_id=COMPONENT_TO_GLOBAL.get(local)
            if global_id is None: continue
            bx=[float(b[0])+cx1,float(b[1])+cy1,float(b[2])+cx1,float(b[3])+cy1]
            components.append({"class_id":global_id,"label":DISPLAY[global_id],"score":float(s),"box":bx,"stage":"component","prompt":COMPONENT_PROMPTS[local]})

    predictions=[tower]+components
    gt=read_gt(ROOT/"data/labels/test/POS_2326530.txt",W,H)
    tower_gt=next(g for g in gt if g["class_id"]==0)
    tower_iou=iou(tower["box"],tower_gt["box"])
    component_gt=[g for g in gt if g["class_id"]!=0]
    metrics=[]
    for threshold in (0.30,0.50):
        metrics.append(greedy_metrics(components,component_gt,threshold,True))
        metrics.append(greedy_metrics(components,component_gt,threshold,False))

    # LinkedIn-style renderer. Scores are explicitly uncalibrated model scores.
    canvas=image.copy(); draw=ImageDraw.Draw(canvas)
    font=ImageFont.load_default()
    palette={0:(40,180,70),1:(60,120,230),2:(220,70,160),3:(230,150,40)}
    for p in predictions:
        color=palette[p["class_id"]]
        b=tuple(int(round(v)) for v in p["box"])
        draw.rectangle(b,outline=color,width=3)
        text=f"{p['label']} {p['score']*100:.1f}%"
        tb=draw.textbbox((b[0],b[1]),text,font=font)
        draw.rectangle((tb[0]-2,tb[1]-2,tb[2]+2,tb[3]+2),fill=color)
        draw.text((b[0],b[1]),text,fill=(255,255,255),font=font)
    out_img=REPORTS/"v2_4_yoloe_hierarchical_POS_2326530.jpg"
    canvas.save(out_img,quality=95)

    report={
        "evidence_type":"pretrained-open-vocabulary-hierarchical-zero-shot",
        "claim_scope":"exploratory model-generated boxes/scores; scores are uncalibrated; no material/condition claim",
        "source_image":"POS_2326530",
        "source_dimensions":[W,H],
        "weight":{"name":weight.name,"sha256":sha256(weight)},
        "stage1":{"prompt":TOWER_PROMPT,"n_candidates":len(tower_candidates),"selected":tower,"tower_body_iou_vs_manual_reference":tower_iou},
        "stage2":{"prompts":COMPONENT_PROMPTS,"roi_xyxy":[cx1,cy1,cx2,cy2],"n_component_predictions":len(components),"predictions":components},
        "component_metrics":metrics,
        "runtime":runtime_env(),
        "dataset_manifest":dataset_manifest(),
    }
    write_json(REPORTS/"v2_4_yoloe_hierarchical_metrics.json",report)
    print(json.dumps({"tower_iou":tower_iou,"n_components":len(components),"metrics":metrics},indent=2))

if __name__=="__main__": main()
