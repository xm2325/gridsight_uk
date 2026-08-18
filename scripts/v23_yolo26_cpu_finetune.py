from __future__ import annotations

import argparse
import json
from pathlib import Path

from v23_common import DATA_YAML, REPORTS, ROOT, SHOWCASE, WEIGHTS, dataset_manifest, runtime_env, sha256, write_json


def yolo_gt(label_path: Path, width: int, height: int):
    rows=[]
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        c, xc, yc, w, h = map(float, line.split())
        x1=(xc-w/2)*width; y1=(yc-h/2)*height
        x2=(xc+w/2)*width; y2=(yc+h/2)*height
        rows.append((int(c), [x1,y1,x2,y2]))
    return rows


def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    den=aa+bb-inter
    return inter/den if den>0 else 0.0


def evaluate(preds, gt, threshold=0.5):
    order=sorted(range(len(preds)), key=lambda i: preds[i][2], reverse=True)
    used=set(); tp=0
    for i in order:
        pc,pb,_=preds[i]
        candidates=[(iou(pb,gb),j) for j,(gc,gb) in enumerate(gt) if gc==pc and j not in used]
        if candidates:
            best,j=max(candidates)
            if best>=threshold:
                used.add(j); tp+=1
    fp=len(preds)-tp; fn=len(gt)-tp
    p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0
    return {"iou_threshold":threshold,"tp":tp,"fp":fp,"fn":fn,"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0.0}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=640)
    args=ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    weight=WEIGHTS/"yolo26n.pt"
    if not weight.exists(): raise FileNotFoundError(weight)
    model=YOLO(str(weight))
    model.train(
        data=str(DATA_YAML), epochs=args.epochs, imgsz=args.imgsz, batch=1,
        workers=0, device="cpu", seed=17, deterministic=True, pretrained=True,
        freeze=10, project=str(ROOT/"runs"), name="v2_3_yolo26n_cpu_smoke",
        exist_ok=True, plots=False, verbose=True,
    )
    best=ROOT/"runs/v2_3_yolo26n_cpu_smoke/weights/best.pt"
    last=ROOT/"runs/v2_3_yolo26n_cpu_smoke/weights/last.pt"
    chosen=best if best.exists() else last
    if not chosen.exists(): raise FileNotFoundError("training produced no checkpoint")

    infer=YOLO(str(chosen)).predict(str(SHOWCASE), imgsz=args.imgsz, conf=0.01, device="cpu", verbose=False)[0]
    preds=[]
    if infer.boxes is not None:
        for box, conf, cls in zip(infer.boxes.xyxy.cpu().tolist(), infer.boxes.conf.cpu().tolist(), infer.boxes.cls.cpu().tolist()):
            preds.append((int(cls), [float(x) for x in box], float(conf)))
    with Image.open(SHOWCASE) as im:
        w,h=im.size
    gt=yolo_gt(ROOT/"data/labels/test/POS_2326530.txt", w, h)
    metrics=[evaluate(preds,gt,t) for t in (0.30,0.50)]
    out_img=REPORTS/"v2_3_yolo26n_cpu_finetune_POS_2326530.jpg"
    infer.save(filename=str(out_img))
    status={
        "evidence_type":"pretrained-yolo26-cpu-finetune-smoke",
        "claim_scope":"short CPU Actions training path; not headline model performance",
        "epochs":args.epochs,"imgsz":args.imgsz,
        "n_train_images":3,"n_val_images":1,"n_test_images":1,
        "test_source":"POS_2326530","n_test_gt":len(gt),"n_predictions":len(preds),
        "metrics":metrics,
        "input_weight":{"name":weight.name,"sha256":sha256(weight)},
        "trained_weight":{"path":str(chosen.relative_to(ROOT)),"sha256":sha256(chosen)},
        "runtime":runtime_env(),"dataset_manifest":dataset_manifest(),
    }
    write_json(REPORTS/"v2_3_yolo26n_cpu_finetune_status.json",status)
    print(json.dumps(status,indent=2))

if __name__=="__main__": main()
