from __future__ import annotations

import json
from pathlib import Path

from v23_common import REPORTS, ROOT, WEIGHTS, runtime_env, sha256, write_json
from v25_hybrid_component_detector import TOWER_PROMPT, expand, iou, read_yolo
from v34_tiled_insulator_specialist import metrics, nms

INSULATOR_PROMPT = "insulator string on an electricity transmission tower"
VAL_IDS = ["POS_354803", "POS_543992"]


def infer_source(model_weight: Path, rid: str):
    from PIL import Image
    from ultralytics import YOLOE

    image_path = ROOT / f"data/v4_1_yolo/images/val/{rid}.jpg"
    label_path = ROOT / f"data/v4_0_labels/val/{rid}.txt"
    image = Image.open(image_path).convert("RGB"); w, h = image.size
    gt_rows = read_yolo(label_path, w, h)
    tower_gt = next(x["box"] for x in gt_rows if x["class_id"] == 0)
    ins_gt = [x["box"] for x in gt_rows if x["class_id"] == 2]

    tower = YOLOE(str(model_weight)); tower.set_classes([TOWER_PROMPT])
    tr = tower.predict(str(image_path), imgsz=768, conf=0.05, device="cpu", verbose=False)[0]
    if tr.boxes is None or len(tr.boxes) == 0: raise RuntimeError(f"no tower ROI: {rid}")
    ti = int(tr.boxes.conf.argmax().item()); tower_box=[float(x) for x in tr.boxes.xyxy[ti].cpu().tolist()]; tower_score=float(tr.boxes.conf[ti])
    roi=expand(tower_box,w,h,px=0.08,py=0.04); x1,y1,x2,y2=[int(round(v)) for v in roi]
    crop=image.crop((x1,y1,x2,y2)); crop_path=REPORTS/f"v4_1_yoloe_{rid}_tower_roi.jpg"; crop.save(crop_path,quality=95)

    model=YOLOE(str(model_weight)); model.set_classes([INSULATOR_PROMPT])
    result=model.predict(str(crop_path),imgsz=960,conf=0.001,device="cpu",verbose=False,max_det=300)[0]
    preds=[]
    if result.boxes is not None:
        for b,s in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
            preds.append({"box":[float(b[0])+x1,float(b[1])+y1,float(b[2])+x1,float(b[3])+y1],"score":float(s)})
    return {"rid":rid,"image_path":image_path,"tower_box":tower_box,"tower_gt":tower_gt,"tower_score":tower_score,"tower_iou":iou(tower_box,tower_gt),"gt":ins_gt,"raw":preds}


def pool(rows, threshold, nms_iou, iou_thr):
    total={"tp":0,"fp":0,"fn":0}; per={}
    for row in rows:
        pred=nms([x for x in row["raw"] if x["score"]>=threshold],nms_iou)
        m=metrics(pred,row["gt"],iou_thr); per[row["rid"]]={**{k:v for k,v in m.items() if k!="matches"},"n_predictions":len(pred)}
        for k in total: total[k]+=m[k]
    tp,fp,fn=total["tp"],total["fp"],total["fn"]; p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0
    return {**total,"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0.0,"per_source":per}


def render(row, threshold, nms_iou):
    from PIL import Image,ImageDraw,ImageFont
    im=Image.open(row["image_path"]).convert("RGB"); d=ImageDraw.Draw(im); font=ImageFont.load_default()
    tb=tuple(int(round(v)) for v in row["tower_box"]); d.rectangle(tb,outline=(35,180,75),width=3)
    pred=nms([x for x in row["raw"] if x["score"]>=threshold],nms_iou)
    for p in pred:
        b=tuple(int(round(v)) for v in p["box"]);d.rectangle(b,outline=(220,60,150),width=3);d.text((b[0],b[1]),f"YOLOE insulator {p['score']:.3f}",fill=(220,60,150),font=font)
    out=REPORTS/f"v4_1_yoloe_{row['rid']}_selected.jpg";im.save(out,quality=95);return out.name


def main():
    weight=WEIGHTS/"yoloe-26n-seg.pt"; rows=[infer_source(weight,r) for r in VAL_IDS]
    thresholds=sorted(set([0.001,0.002,0.003,0.005,0.008,0.01,0.015,0.02,0.03,0.05,0.08,0.1,0.15,0.2]+[round(x["score"],6) for r in rows for x in r["raw"]]))
    sweep=[]
    for t in thresholds:
        for ni in [0.2,0.3,0.4,0.5]:
            m=pool(rows,t,ni,0.30);sweep.append({"threshold":t,"nms_iou":ni,**{k:v for k,v in m.items() if k!="per_source"}})
    best=max(sweep,key=lambda x:(x["f1"],x["recall"],x["precision"],x["threshold"],-x["nms_iou"]))
    selected={str(x):pool(rows,best["threshold"],best["nms_iou"],x) for x in (0.30,0.50)}
    images=[render(r,best["threshold"],best["nms_iou"]) for r in rows]
    report={"evidence_type":"v4.1-yoloe26n-text-validation","claim_scope":"validation-only model/operating-point evidence; two assistant-provisional sources; v4 final holdout untouched; scores uncalibrated","prompt":INSULATOR_PROMPT,"checkpoint":{"name":weight.name,"sha256":sha256(weight)},"sources":[{"record_id":r["rid"],"tower_iou":r["tower_iou"],"tower_score":r["tower_score"],"n_gt":len(r["gt"]),"n_raw":len(r["raw"])} for r in rows],"selected_operating_point":best,"metrics":selected,"sweep":sweep,"images":images,"runtime":runtime_env(),"v4_final_holdout_touched":False}
    write_json(REPORTS/"v4_1_yoloe_text_validation.json",report);print(json.dumps({"best":best,"metrics":selected},indent=2))

if __name__=="__main__":main()
