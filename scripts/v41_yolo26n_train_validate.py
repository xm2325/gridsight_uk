from __future__ import annotations

import json
from pathlib import Path

from v23_common import REPORTS, ROOT, WEIGHTS, runtime_env, sha256, write_json
from v25_hybrid_component_detector import read_yolo
from v34_tiled_insulator_specialist import metrics

VAL_IDS=["POS_354803","POS_543992"]


def load_gt(rid:str):
    from PIL import Image
    image=ROOT/f"data/v4_1_yolo/images/val/{rid}.jpg"; label=ROOT/f"data/v4_1_yolo/labels/val/{rid}.txt"
    with Image.open(image) as im:w,h=im.size
    rows=read_yolo(label,w,h)
    return image,[x["box"] for x in rows if x["class_id"]==0]


def infer(model,rid:str):
    image,gt=load_gt(rid)
    result=model.predict(str(image),imgsz=960,conf=0.001,device="cpu",verbose=False,max_det=300)[0]
    raw=[]
    if result.boxes is not None:
        for b,s in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):raw.append({"box":[float(x) for x in b],"score":float(s)})
    return {"rid":rid,"image":image,"gt":gt,"raw":raw}


def pooled(rows,threshold,iou_thr):
    total={"tp":0,"fp":0,"fn":0};per={}
    for r in rows:
        pred=[x for x in r["raw"] if x["score"]>=threshold]
        m=metrics(pred,r["gt"],iou_thr);per[r["rid"]]={**{k:v for k,v in m.items() if k!="matches"},"n_predictions":len(pred)}
        for k in total:total[k]+=m[k]
    tp,fp,fn=total["tp"],total["fp"],total["fn"];p=tp/(tp+fp) if tp+fp else 0.0;r=tp/(tp+fn) if tp+fn else 0.0
    return {**total,"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0.0,"per_source":per}


def render(model,rid,threshold):
    from PIL import Image,ImageDraw,ImageFont
    image,gt=load_gt(rid);im=Image.open(image).convert("RGB");d=ImageDraw.Draw(im);font=ImageFont.load_default()
    result=model.predict(str(image),imgsz=960,conf=threshold,device="cpu",verbose=False,max_det=300)[0]
    if result.boxes is not None:
        for b,s in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
            bb=tuple(int(round(x)) for x in b);d.rectangle(bb,outline=(220,60,150),width=3);d.text((bb[0],bb[1]),f"YOLO26 insulator {float(s):.3f}",fill=(220,60,150),font=font)
    out=REPORTS/f"v4_1_yolo26n_{rid}_selected.jpg";im.save(out,quality=95);return out.name


def main():
    from ultralytics import YOLO
    weight=WEIGHTS/"yolo26n.pt";data=ROOT/"data/v4_1_yolo/data.yaml";run_dir=ROOT/"runs/v4_1_yolo26n"
    model=YOLO(str(weight))
    train_result=model.train(data=str(data),epochs=30,imgsz=960,batch=4,device="cpu",workers=0,seed=17,deterministic=True,patience=10,project=str(ROOT/"runs"),name="v4_1_yolo26n",exist_ok=True,cache=False,verbose=True,plots=False)
    best=run_dir/"weights/best.pt"
    if not best.exists():raise FileNotFoundError(best)
    best_model=YOLO(str(best));rows=[infer(best_model,r) for r in VAL_IDS]
    thresholds=sorted(set([0.001,0.002,0.003,0.005,0.008,0.01,0.015,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.5]+[round(x["score"],6) for r in rows for x in r["raw"]]))
    sweep=[]
    for t in thresholds:
        m=pooled(rows,t,0.30);sweep.append({"threshold":t,**{k:v for k,v in m.items() if k!="per_source"}})
    best_op=max(sweep,key=lambda x:(x["f1"],x["recall"],x["precision"],x["threshold"]))
    selected={str(x):pooled(rows,best_op["threshold"],x) for x in (0.30,0.50)}
    images=[render(best_model,r,best_op["threshold"]) for r in VAL_IDS]
    train_metrics=getattr(train_result,"results_dict",None)
    report={"evidence_type":"v4.1-pretrained-yolo26n-one-class-insulator-validation","claim_scope":"exploratory validation-only closed-set fine-tuning; 11 train sources / 60 provisional insulator boxes; 2 validation sources / 12 provisional boxes; v4 final holdout untouched","pretrained_checkpoint":{"name":weight.name,"sha256":sha256(weight)},"best_checkpoint":{"path":str(best.relative_to(ROOT)),"sha256":sha256(best)},"training":{"epochs_requested":30,"imgsz":960,"batch":4,"seed":17,"train_metrics":train_metrics},"validation_sources":[{"record_id":r["rid"],"n_gt":len(r["gt"]),"n_raw":len(r["raw"])} for r in rows],"selected_operating_point":best_op,"metrics":selected,"sweep":sweep,"images":images,"runtime":runtime_env(),"v4_final_holdout_touched":False}
    write_json(REPORTS/"v4_1_yolo26n_validation.json",report);print(json.dumps({"best_checkpoint":report["best_checkpoint"],"best_op":best_op,"metrics":selected},indent=2))

if __name__=="__main__":main()
