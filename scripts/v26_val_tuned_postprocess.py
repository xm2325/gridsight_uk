from __future__ import annotations

import argparse
import json
from pathlib import Path

from v23_common import REPORTS, ROOT, runtime_env, sha256, write_json

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


def class_nms(preds, nms_iou):
    kept=[]
    for cls in sorted({p["class_id"] for p in preds}):
        candidates=sorted([p for p in preds if p["class_id"]==cls], key=lambda p:p["score"], reverse=True)
        while candidates:
            best=candidates.pop(0); kept.append(best)
            candidates=[p for p in candidates if iou(best["box"],p["box"]) < nms_iou]
    return sorted(kept,key=lambda p:p["score"],reverse=True)


def filter_preds(raw, conf, nms_iou):
    return class_nms([p for p in raw if p["score"] >= conf], nms_iou)


def greedy(preds, gt, threshold):
    order=sorted(range(len(preds)),key=lambda i:preds[i]["score"],reverse=True); used=set(); matches=[]
    for i in order:
        p=preds[i]; cand=[]
        for j,g in enumerate(gt):
            if j in used or p["class_id"]!=g["class_id"]: continue
            cand.append((iou(p["box"],g["box"]),j))
        if cand:
            best,j=max(cand)
            if best>=threshold:
                used.add(j); matches.append({"pred":i,"gt":j,"iou":best})
    tp=len(matches); fp=len(preds)-tp; fn=len(gt)-tp
    precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
    return {"iou_threshold":threshold,"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0.0,"matches":matches}


def gt_crop(image_path: Path, label_path: Path, out_path: Path):
    from PIL import Image
    im=Image.open(image_path).convert("RGB"); W,H=im.size
    gt=read_yolo(label_path,W,H); tower=next(g for g in gt if g["class_id"]==0)
    roi=expand(tower["box"],W,H); x1,y1,x2,y2=[int(round(v)) for v in roi]
    out_path.parent.mkdir(parents=True,exist_ok=True); im.crop((x1,y1,x2,y2)).save(out_path,quality=95)
    return (x1,y1,x2,y2), [g for g in gt if g["class_id"]!=0], (W,H)


def yoloe_test_crop(yoloe, image_path: Path, out_path: Path):
    from PIL import Image
    im=Image.open(image_path).convert("RGB"); W,H=im.size
    yoloe.set_classes([TOWER_PROMPT])
    r=yoloe.predict(str(image_path),imgsz=768,conf=0.05,device="cpu",verbose=False)[0]
    if r.boxes is None or len(r.boxes)==0: raise RuntimeError("YOLOE produced no tower ROI")
    idx=int(r.boxes.conf.argmax().item()); tower=[float(x) for x in r.boxes.xyxy[idx].cpu().tolist()]; score=float(r.boxes.conf[idx].item())
    roi=expand(tower,W,H); x1,y1,x2,y2=[int(round(v)) for v in roi]
    out_path.parent.mkdir(parents=True,exist_ok=True); im.crop((x1,y1,x2,y2)).save(out_path,quality=95)
    gt=read_yolo(ROOT/"data/labels/test/POS_2326530.txt",W,H)
    return (x1,y1,x2,y2), [g for g in gt if g["class_id"]!=0], (W,H), tower, score


def infer(model, crop_path: Path, roi, conf=0.001):
    x1,y1,_,_=roi
    r=model.predict(str(crop_path),imgsz=640,conf=conf,device="cpu",verbose=False,max_det=300)[0]
    rows=[]
    if r.boxes is not None:
        for b,s,c in zip(r.boxes.xyxy.cpu().tolist(),r.boxes.conf.cpu().tolist(),r.boxes.cls.cpu().tolist()):
            gid=LOCAL_TO_GLOBAL[int(c)]
            rows.append({"class_id":gid,"label":DISPLAY[gid],"score":float(s),"box":[float(b[0])+x1,float(b[1])+y1,float(b[2])+x1,float(b[3])+y1]})
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--artifact-root",required=True); args=ap.parse_args()
    from PIL import Image,ImageDraw,ImageFont
    from ultralytics import YOLO,YOLOE
    root=Path(args.artifact_root)
    trained=next(root.rglob("runs/v2_5_yolo26n_component_crop/weights/best.pt"))
    yoloe_weight=next(root.rglob("weights/yoloe-26n-seg.pt"))
    model=YOLO(str(trained)); yoloe=YOLOE(str(yoloe_weight))

    val_crop=ROOT/"reports/v2_6_val_crop_POS_5442616.jpg"
    val_roi,val_gt,_=gt_crop(ROOT/"data/images/val/POS_5442616.jpg",ROOT/"data/labels/val/POS_5442616.txt",val_crop)
    val_raw=infer(model,val_crop,val_roi,conf=0.001)

    # Search is validation-only. Primary objective: class-aware F1 at IoU>=0.30.
    score_candidates=sorted(set([0.001,0.002,0.003,0.004,0.005,0.006,0.008,0.010,0.012,0.015,0.020,0.030,0.050] + [round(p["score"],6) for p in val_raw]))
    nms_candidates=[0.30,0.40,0.50,0.60,0.70]
    sweep=[]
    for conf in score_candidates:
        for nms_iou in nms_candidates:
            filtered=filter_preds(val_raw,conf,nms_iou); m=greedy(filtered,val_gt,0.30)
            sweep.append({"conf_threshold":conf,"nms_iou":nms_iou,"n_predictions":len(filtered),**{k:v for k,v in m.items() if k!="matches"}})
    # Deterministic tie-break: F1, recall, precision, fewer predictions, higher conf, lower NMS IoU.
    best=max(sweep,key=lambda x:(x["f1"],x["recall"],x["precision"],-x["n_predictions"],x["conf_threshold"],-x["nms_iou"]))

    test_crop=ROOT/"reports/v2_6_test_yoloe_crop_POS_2326530.jpg"
    test_roi,test_gt,(W,H),tower_box,tower_score=yoloe_test_crop(yoloe,ROOT/"data/images/test/POS_2326530.jpg",test_crop)
    test_raw=infer(model,test_crop,test_roi,conf=0.001)
    test_filtered=filter_preds(test_raw,best["conf_threshold"],best["nms_iou"])
    test_metrics=[greedy(test_filtered,test_gt,t) for t in (0.30,0.50)]

    image=Image.open(ROOT/"data/images/test/POS_2326530.jpg").convert("RGB"); draw=ImageDraw.Draw(image); font=ImageFont.load_default(); colors={1:(55,120,230),2:(220,60,150),3:(230,145,30)}
    draw.rectangle(tuple(int(v) for v in tower_box),outline=(35,180,75),width=3); draw.text((int(tower_box[0]),int(tower_box[1])),f"tower ROI {tower_score*100:.1f}%",fill=(35,180,75),font=font)
    for p in test_filtered:
        b=tuple(int(round(v)) for v in p["box"]); color=colors[p["class_id"]]; draw.rectangle(b,outline=color,width=3); draw.text((b[0],b[1]),f"{p['label']} score {p['score']:.3f}",fill=color,font=font)
    out_img=REPORTS/"v2_6_val_tuned_hybrid_POS_2326530.jpg"; image.save(out_img,quality=95)

    report={
      "evidence_type":"validation-only-operating-point-selection",
      "claim_scope":"exploratory; threshold/NMS chosen only on POS_5442616, then frozen before fixed test POS_2326530; scores uncalibrated",
      "trained_weight":{"path":str(trained),"sha256":sha256(trained)},
      "yoloe_weight":{"path":str(yoloe_weight),"sha256":sha256(yoloe_weight)},
      "selection":{"objective":"max class-aware F1 at IoU>=0.30 on validation source","tie_break":"F1, recall, precision, fewer predictions, higher conf, lower NMS IoU","n_val_raw":len(val_raw),"n_grid":len(sweep),"best":best,"sweep":sweep},
      "test":{"source":"POS_2326530","n_raw":len(test_raw),"n_filtered":len(test_filtered),"tower_box":tower_box,"tower_score":tower_score,"operating_point":{"conf_threshold":best["conf_threshold"],"nms_iou":best["nms_iou"]},"metrics":test_metrics,"predictions":test_filtered},
      "runtime":runtime_env(),
    }
    write_json(REPORTS/"v2_6_val_tuned_hybrid_metrics.json",report)
    print(json.dumps({"val_best":best,"test_n":len(test_filtered),"test_metrics":test_metrics},indent=2))

if __name__=="__main__": main()
