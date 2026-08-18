from __future__ import annotations

import argparse
import json
from pathlib import Path

from v23_common import REPORTS, ROOT, WEIGHTS, dataset_manifest, runtime_env, sha256, write_json

GLOBAL_TO_LOCAL = {1: 0, 2: 1, 3: 2}
LOCAL_TO_GLOBAL = {0: 1, 1: 2, 2: 3}
DISPLAY = {1: "crossarm", 2: "insulator", 3: "earthwire peak"}
TOWER_PROMPT = "steel lattice electricity transmission pylon"


def read_yolo(path: Path, width: int, height: int):
    rows=[]
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        c,xc,yc,w,h=map(float,line.split())
        rows.append({"class_id":int(c),"box":[(xc-w/2)*width,(yc-h/2)*height,(xc+w/2)*width,(yc+h/2)*height]})
    return rows


def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0.0,x2-x1)*max(0.0,y2-y1)
    aa=max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1]); bb=max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1])
    den=aa+bb-inter
    return inter/den if den else 0.0


def expand(box,W,H,px=0.10,py=0.05):
    x1,y1,x2,y2=box; dx=px*(x2-x1); dy=py*(y2-y1)
    return [max(0,x1-dx),max(0,y1-dy),min(W,x2+dx),min(H,y2+dy)]


def crop_training_split(split: str, dst: Path):
    from PIL import Image
    manifest=[]
    for image_path in sorted((ROOT/f"data/images/{split}").glob("*.jpg")):
        im=Image.open(image_path).convert("RGB"); W,H=im.size
        labels=read_yolo(ROOT/f"data/labels/{split}/{image_path.stem}.txt",W,H)
        tower=next(x for x in labels if x["class_id"]==0)
        roi=expand(tower["box"],W,H)
        x1,y1,x2,y2=[int(round(x)) for x in roi]; cw=x2-x1; ch=y2-y1
        crop=im.crop((x1,y1,x2,y2))
        out_img=dst/f"images/{split}/{image_path.name}"; out_img.parent.mkdir(parents=True,exist_ok=True); crop.save(out_img,quality=94)
        out_labels=[]
        kept=[]
        for obj in labels:
            if obj["class_id"] not in GLOBAL_TO_LOCAL: continue
            bx=obj["box"]; cx=(bx[0]+bx[2])/2; cy=(bx[1]+bx[3])/2
            if not (x1<=cx<=x2 and y1<=cy<=y2): continue
            clipped=[max(x1,bx[0]),max(y1,bx[1]),min(x2,bx[2]),min(y2,bx[3])]
            if clipped[2]<=clipped[0] or clipped[3]<=clipped[1]: continue
            lx1,ly1,lx2,ly2=clipped[0]-x1,clipped[1]-y1,clipped[2]-x1,clipped[3]-y1
            xc=((lx1+lx2)/2)/cw; yc=((ly1+ly2)/2)/ch; w=(lx2-lx1)/cw; h=(ly2-ly1)/ch
            local=GLOBAL_TO_LOCAL[obj["class_id"]]
            out_labels.append(f"{local} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}")
            kept.append({"global_class":obj["class_id"],"local_class":local,"full_box":bx})
        out_lab=dst/f"labels/{split}/{image_path.stem}.txt"; out_lab.parent.mkdir(parents=True,exist_ok=True); out_lab.write_text("\n".join(out_labels)+"\n")
        manifest.append({"source":image_path.stem,"source_dimensions":[W,H],"roi_xyxy":[x1,y1,x2,y2],"crop_dimensions":[cw,ch],"n_components":len(kept)})
    return manifest


def greedy(preds, gt, threshold):
    order=sorted(range(len(preds)),key=lambda i:preds[i]["score"],reverse=True); used=set(); matches=[]
    for i in order:
        p=preds[i]; cand=[]
        for j,g in enumerate(gt):
            if j in used or p["class_id"]!=g["class_id"]: continue
            cand.append((iou(p["box"],g["box"]),j))
        if cand:
            best,j=max(cand)
            if best>=threshold: used.add(j); matches.append({"pred":i,"gt":j,"iou":best})
    tp=len(matches); fp=len(preds)-tp; fn=len(gt)-tp
    precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
    return {"iou_threshold":threshold,"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0.0,"matches":matches}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--epochs",type=int,default=80); args=ap.parse_args()
    from PIL import Image,ImageDraw,ImageFont
    from ultralytics import YOLO,YOLOE

    crop_root=ROOT/"data/v25_component_crop"
    train_manifest=crop_training_split("train",crop_root)
    val_manifest=crop_training_split("val",crop_root)
    (crop_root/"data.yaml").write_text("path: data/v25_component_crop\ntrain: images/train\nval: images/val\nnames:\n  0: crossarm\n  1: insulator_string\n  2: earthwire_peak\n")

    # Held-out crop is model-generated: YOLOE tower ROI only, never test parent GT.
    test_path=ROOT/"data/images/test/POS_2326530.jpg"; image=Image.open(test_path).convert("RGB"); W,H=image.size
    yoloe_weight=WEIGHTS/"yoloe-26n-seg.pt"; yolo26_weight=WEIGHTS/"yolo26n.pt"
    yoloe=YOLOE(str(yoloe_weight)); yoloe.set_classes([TOWER_PROMPT])
    r=yoloe.predict(str(test_path),imgsz=768,conf=0.05,device="cpu",verbose=False)[0]
    if r.boxes is None or len(r.boxes)==0: raise RuntimeError("YOLOE produced no test tower ROI")
    idx=int(r.boxes.conf.argmax().item()); tower_box=[float(x) for x in r.boxes.xyxy[idx].cpu().tolist()]; tower_score=float(r.boxes.conf[idx].item())
    roi=expand(tower_box,W,H); x1,y1,x2,y2=[int(round(x)) for x in roi]; crop=image.crop((x1,y1,x2,y2)); cw,ch=crop.size
    test_crop=crop_root/"images/test/POS_2326530.jpg"; test_crop.parent.mkdir(parents=True,exist_ok=True); crop.save(test_crop,quality=95)

    # Train component-only detector, no tower-body class and no mosaic on this tiny cohort.
    model=YOLO(str(yolo26_weight))
    model.train(data=str(crop_root/"data.yaml"),epochs=args.epochs,imgsz=640,batch=1,workers=0,device="cpu",seed=17,deterministic=True,pretrained=True,project=str(ROOT/"runs"),name="v2_5_yolo26n_component_crop",exist_ok=True,plots=False,verbose=True,mosaic=0.0,close_mosaic=0,translate=0.05,scale=0.2,fliplr=0.5,lr0=0.001,lrf=0.01,optimizer="AdamW")
    best=ROOT/"runs/v2_5_yolo26n_component_crop/weights/best.pt"; last=ROOT/"runs/v2_5_yolo26n_component_crop/weights/last.pt"; chosen=best if best.exists() else last
    if not chosen.exists(): raise RuntimeError("No trained checkpoint")

    pred=YOLO(str(chosen)).predict(str(test_crop),imgsz=640,conf=0.005,device="cpu",verbose=False)[0]
    predictions=[]
    if pred.boxes is not None:
        for b,s,c in zip(pred.boxes.xyxy.cpu().tolist(),pred.boxes.conf.cpu().tolist(),pred.boxes.cls.cpu().tolist()):
            local=int(c); global_id=LOCAL_TO_GLOBAL[local]
            full=[float(b[0])+x1,float(b[1])+y1,float(b[2])+x1,float(b[3])+y1]
            predictions.append({"class_id":global_id,"label":DISPLAY[global_id],"score":float(s),"box":full})

    gt_all=read_yolo(ROOT/"data/labels/test/POS_2326530.txt",W,H); tower_gt=next(g for g in gt_all if g["class_id"]==0); component_gt=[g for g in gt_all if g["class_id"]!=0]
    metrics=[greedy(predictions,component_gt,t) for t in (0.30,0.50)]

    canvas=image.copy(); draw=ImageDraw.Draw(canvas); font=ImageFont.load_default(); colors={1:(50,120,230),2:(220,60,150),3:(230,145,30)}
    draw.rectangle(tuple(int(v) for v in tower_box),outline=(35,180,75),width=3); draw.text((int(tower_box[0]),int(tower_box[1])),f"YOLOE tower {tower_score*100:.1f}%",fill=(35,180,75),font=font)
    for p in predictions:
        b=tuple(int(round(v)) for v in p["box"]); color=colors[p["class_id"]]; draw.rectangle(b,outline=color,width=3); draw.text((b[0],b[1]),f"{p['label']} {p['score']*100:.1f}%",fill=color,font=font)
    out_img=REPORTS/"v2_5_yoloe_yolo26_hybrid_POS_2326530.jpg"; canvas.save(out_img,quality=95)

    report={
      "evidence_type":"yoloe-tower-roi-plus-pretrained-yolo26-component-finetune",
      "claim_scope":"exploratory source-isolated 3-train/1-val/1-test pilot; model scores uncalibrated; no material/condition claim",
      "epochs":args.epochs,
      "train_crop_manifest":train_manifest,"val_crop_manifest":val_manifest,
      "held_out":{"source":"POS_2326530","source_dimensions":[W,H],"tower_prompt":TOWER_PROMPT,"tower_box":tower_box,"tower_score":tower_score,"tower_iou_vs_manual_reference":iou(tower_box,tower_gt["box"]),"model_generated_roi_xyxy":[x1,y1,x2,y2],"crop_dimensions":[cw,ch]},
      "n_component_gt":len(component_gt),"n_predictions":len(predictions),"predictions":predictions,"metrics":metrics,
      "weights":{"yoloe":{"sha256":sha256(yoloe_weight)},"yolo26_input":{"sha256":sha256(yolo26_weight)},"yolo26_trained":{"path":str(chosen.relative_to(ROOT)),"sha256":sha256(chosen)}},
      "runtime":runtime_env(),"dataset_manifest":dataset_manifest(),
    }
    write_json(REPORTS/"v2_5_yoloe_yolo26_hybrid_metrics.json",report)
    print(json.dumps({"tower_iou":report["held_out"]["tower_iou_vs_manual_reference"],"n_predictions":len(predictions),"metrics":metrics},indent=2))

if __name__=="__main__": main()
