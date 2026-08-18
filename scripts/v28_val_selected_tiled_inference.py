from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from v23_common import REPORTS, ROOT, runtime_env, sha256, write_json
from v26_val_tuned_postprocess import DISPLAY, LOCAL_TO_GLOBAL, TOWER_PROMPT, class_nms, greedy, iou, read_yolo, expand


def find_one(root: Path, pattern: str) -> Path:
    hits=list(root.rglob(pattern))
    if len(hits)!=1:
        raise RuntimeError(f"Expected exactly one {pattern} under {root}, got {hits}")
    return hits[0]


def tower_crop_gt(image_path: Path, label_path: Path, out_path: Path):
    from PIL import Image
    im=Image.open(image_path).convert("RGB"); W,H=im.size
    gt=read_yolo(label_path,W,H); tower=next(g for g in gt if g["class_id"]==0)
    roi=expand(tower["box"],W,H); x1,y1,x2,y2=[int(round(v)) for v in roi]
    out_path.parent.mkdir(parents=True,exist_ok=True); im.crop((x1,y1,x2,y2)).save(out_path,quality=95)
    return (x1,y1,x2,y2),[g for g in gt if g["class_id"]!=0],(W,H)


def tower_crop_yoloe(yoloe,image_path: Path,label_path: Path,out_path: Path):
    from PIL import Image
    im=Image.open(image_path).convert("RGB"); W,H=im.size
    yoloe.set_classes([TOWER_PROMPT])
    r=yoloe.predict(str(image_path),imgsz=768,conf=0.05,device="cpu",verbose=False)[0]
    if r.boxes is None or len(r.boxes)==0: raise RuntimeError("YOLOE produced no tower ROI")
    idx=int(r.boxes.conf.argmax().item()); tower=[float(x) for x in r.boxes.xyxy[idx].cpu().tolist()]; score=float(r.boxes.conf[idx].item())
    roi=expand(tower,W,H); x1,y1,x2,y2=[int(round(v)) for v in roi]
    out_path.parent.mkdir(parents=True,exist_ok=True); im.crop((x1,y1,x2,y2)).save(out_path,quality=95)
    gt=read_yolo(label_path,W,H); manual_tower=next(g for g in gt if g["class_id"]==0)
    return (x1,y1,x2,y2),[g for g in gt if g["class_id"]!=0],(W,H),tower,score,iou(tower,manual_tower["box"])


def tile_windows(width:int,height:int,height_frac:float,overlap:float):
    if height_frac>=0.999:
        return [(0,0,width,height)]
    th=max(32,min(height,int(round(height*height_frac))))
    step=max(1,int(round(th*(1-overlap))))
    ys=list(range(0,max(1,height-th+1),step))
    last=max(0,height-th)
    if not ys or ys[-1]!=last: ys.append(last)
    return [(0,y,width,y+th) for y in sorted(set(ys))]


def infer_tiled(model,crop_path:Path,roi,height_frac:float,overlap:float,conf_floor=0.001):
    from PIL import Image
    x0,y0,_,_=roi
    im=Image.open(crop_path).convert("RGB"); W,H=im.size
    tmp=REPORTS/"v2_8_tiles"; tmp.mkdir(parents=True,exist_ok=True)
    rows=[]
    for k,(tx1,ty1,tx2,ty2) in enumerate(tile_windows(W,H,height_frac,overlap)):
        p=tmp/f"tile_{crop_path.stem}_{height_frac:.2f}_{overlap:.2f}_{k}.jpg"
        im.crop((tx1,ty1,tx2,ty2)).save(p,quality=94)
        r=model.predict(str(p),imgsz=640,conf=conf_floor,device="cpu",verbose=False,max_det=300)[0]
        if r.boxes is None: continue
        for b,s,c in zip(r.boxes.xyxy.cpu().tolist(),r.boxes.conf.cpu().tolist(),r.boxes.cls.cpu().tolist()):
            gid=LOCAL_TO_GLOBAL[int(c)]
            rows.append({"class_id":gid,"label":DISPLAY[gid],"score":float(s),"box":[float(b[0])+tx1+x0,float(b[1])+ty1+y0,float(b[2])+tx1+x0,float(b[3])+ty1+y0],"tile_index":k})
    return rows


def filter_preds(raw,conf,nms_iou):
    return class_nms([p for p in raw if p["score"]>=conf],nms_iou)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--v25-root",required=True); ap.add_argument("--v27-root",required=True); args=ap.parse_args()
    from PIL import Image,ImageDraw,ImageFont
    from ultralytics import YOLO,YOLOE

    roots={"v25-3sources":Path(args.v25_root),"v27-5sources":Path(args.v27_root)}
    model_paths={
      "v25-3sources":find_one(roots["v25-3sources"],"best.pt"),
      "v27-5sources":find_one(roots["v27-5sources"],"best.pt"),
    }
    yoloe_path=find_one(roots["v25-3sources"],"yoloe-26n-seg.pt")
    models={k:YOLO(str(p)) for k,p in model_paths.items()}; yoloe=YOLOE(str(yoloe_path))

    val_crop=REPORTS/"v2_8_val_tower_crop.jpg"
    val_roi,val_gt,_=tower_crop_gt(ROOT/"data/images/val/POS_5442616.jpg",ROOT/"data/labels/val/POS_5442616.txt",val_crop)

    height_fracs=[0.35,0.45,0.55,0.70,1.00]
    overlaps=[0.20,0.35]
    confs=[0.001,0.002,0.003,0.004,0.005,0.006,0.008,0.010,0.012,0.015,0.020,0.030,0.050]
    nmses=[0.30,0.40,0.50,0.60]
    sweep=[]; val_cache={}
    for model_name,model in models.items():
      for hf in height_fracs:
        for ov in overlaps:
          if hf>=0.999 and ov!=overlaps[0]: continue
          raw=infer_tiled(model,val_crop,val_roi,hf,ov)
          val_cache[(model_name,hf,ov)]=raw
          for conf in confs:
            for nms in nmses:
              filt=filter_preds(raw,conf,nms); m=greedy(filt,val_gt,0.30)
              sweep.append({"model":model_name,"height_frac":hf,"overlap":ov,"conf_threshold":conf,"nms_iou":nms,"n_raw":len(raw),"n_predictions":len(filt),**{k:v for k,v in m.items() if k!="matches"}})
    # Selection uses validation only. Tie-break favors F1, recall, precision, fewer outputs, simpler/larger tiles.
    best=max(sweep,key=lambda x:(x["f1"],x["recall"],x["precision"],-x["n_predictions"],x["conf_threshold"],x["height_frac"],-x["overlap"]))

    dev_crop=REPORTS/"v2_8_dev_yoloe_tower_crop.jpg"
    dev_roi,dev_gt,(W,H),tower_box,tower_score,tower_iou=tower_crop_yoloe(yoloe,ROOT/"data/images/test/POS_2326530.jpg",ROOT/"data/labels/test/POS_2326530.txt",dev_crop)
    selected=models[best["model"]]
    dev_raw=infer_tiled(selected,dev_crop,dev_roi,best["height_frac"],best["overlap"])
    dev_filtered=filter_preds(dev_raw,best["conf_threshold"],best["nms_iou"])
    dev_metrics=[greedy(dev_filtered,dev_gt,t) for t in (0.30,0.50)]

    image=Image.open(ROOT/"data/images/test/POS_2326530.jpg").convert("RGB"); draw=ImageDraw.Draw(image); font=ImageFont.load_default(); colors={1:(50,120,230),2:(220,60,150),3:(230,145,30)}
    draw.rectangle(tuple(int(v) for v in tower_box),outline=(35,180,75),width=3); draw.text((int(tower_box[0]),int(tower_box[1])),f"tower ROI score {tower_score:.3f}",fill=(35,180,75),font=font)
    for p in dev_filtered:
      b=tuple(int(round(v)) for v in p["box"]); color=colors[p["class_id"]]; draw.rectangle(b,outline=color,width=3); draw.text((b[0],b[1]),f"{p['label']} score {p['score']:.3f}",fill=color,font=font)
    out=REPORTS/"v2_8_val_selected_tiled_POS_2326530.jpg"; image.save(out,quality=95)

    report={
      "evidence_type":"validation-only-model-and-sahi-style-tile-selection",
      "claim_scope":"POS_2326530 is now a development showcase, not an untouched final test; all model/tile/threshold/NMS selection used only POS_5442616 validation; scores uncalibrated",
      "selection":{"objective":"max class-aware F1 at IoU>=0.30 on POS_5442616 only","candidate_models":{k:{"sha256":sha256(p)} for k,p in model_paths.items()},"height_fracs":height_fracs,"overlaps":overlaps,"conf_thresholds":confs,"nms_ious":nmses,"best":best,"sweep":sweep},
      "development_showcase":{"source":"POS_2326530","semantic_status":"adaptive development showcase; excluded from final headline evaluation","tower_iou":tower_iou,"tower_score":tower_score,"n_raw":len(dev_raw),"n_filtered":len(dev_filtered),"metrics":dev_metrics,"predictions":dev_filtered},
      "yoloe_weight":{"sha256":sha256(yoloe_path)},"runtime":runtime_env()
    }
    write_json(REPORTS/"v2_8_val_selected_tiled_metrics.json",report)
    print(json.dumps({"val_best":best,"dev_n":len(dev_filtered),"dev_metrics":dev_metrics},indent=2))

if __name__=="__main__": main()
