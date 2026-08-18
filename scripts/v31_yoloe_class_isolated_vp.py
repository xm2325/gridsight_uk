from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from v23_common import REPORTS, ROOT, WEIGHTS, runtime_env, sha256, write_json
from v25_hybrid_component_detector import TOWER_PROMPT, expand, greedy, iou, read_yolo

CLASSES={1:"crossarm",2:"insulator",3:"earthwire peak"}


def class_nms(rows,thr=0.30):
    def biou(a,b):
        x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
        inter=max(0,x2-x1)*max(0,y2-y1);aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);den=aa+bb-inter
        return inter/den if den else 0.0
    out=[]
    for cid in CLASSES:
        cand=sorted([r for r in rows if r['class_id']==cid],key=lambda x:x['score'],reverse=True)
        while cand:
            b=cand.pop(0);out.append(b);cand=[x for x in cand if biou(b['box'],x['box'])<thr]
    return sorted(out,key=lambda x:x['score'],reverse=True)


def main():
    from PIL import Image,ImageDraw,ImageFont
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    weight=WEIGHTS/"yoloe-26n-seg.pt"; ref=ROOT/"data/images/train/POS_190181.jpg"; target=ROOT/"data/images/test/POS_2326530.jpg"
    ref_im=Image.open(ref).convert('RGB');RW,RH=ref_im.size; target_im=Image.open(target).convert('RGB');W,H=target_im.size
    ref_labels=read_yolo(ROOT/"data/labels/train/POS_190181.txt",RW,RH)

    tower_model=YOLOE(str(weight));tower_model.set_classes([TOWER_PROMPT]);tr=tower_model.predict(str(target),imgsz=768,conf=0.05,device='cpu',verbose=False)[0]
    ti=int(tr.boxes.conf.argmax().item());tower=[float(x) for x in tr.boxes.xyxy[ti].cpu().tolist()];tower_score=float(tr.boxes.conf[ti].item());roi=expand(tower,W,H);x1,y1,x2,y2=[int(round(v)) for v in roi]
    crop=target_im.crop((x1,y1,x2,y2));crop_path=REPORTS/"v3_1_target_tower_roi.jpg";crop.save(crop_path,quality=95)

    all_preds=[]; per_class={}
    for gid,name in CLASSES.items():
        examples=[x for x in ref_labels if x['class_id']==gid]
        vp={"bboxes":np.array([x['box'] for x in examples],dtype=np.float32),"cls":np.zeros(len(examples),dtype=np.int64)}
        model=YOLOE(str(weight))
        r=model.predict(str(crop_path),refer_image=str(ref),visual_prompts=vp,predictor=YOLOEVPSegPredictor,imgsz=960,conf=0.015,device='cpu',verbose=False)[0]
        masks=None if r.masks is None else r.masks.data.cpu().numpy();rows=[]
        if r.boxes is not None:
            for k,(b,s) in enumerate(zip(r.boxes.xyxy.cpu().tolist(),r.boxes.conf.cpu().tolist())):
                rows.append({"class_id":gid,"label":name,"score":float(s),"box":[float(b[0])+x1,float(b[1])+y1,float(b[2])+x1,float(b[3])+y1],"mask_pixels_at_predictor_resolution":int((masks[k]>0.5).sum()) if masks is not None and k<len(masks) else None})
        per_class[name]={"n_reference_examples":len(examples),"n_raw_predictions":len(rows),"predictions":rows};all_preds.extend(rows)

    filtered=class_nms(all_preds,0.30)
    gt=read_yolo(ROOT/"data/labels/test/POS_2326530.txt",W,H);tower_gt=next(g for g in gt if g['class_id']==0);comp_gt=[g for g in gt if g['class_id']!=0]
    metrics=[greedy(filtered,comp_gt,t) for t in (0.30,0.50)]

    canvas=target_im.copy();draw=ImageDraw.Draw(canvas);font=ImageFont.load_default();colors={1:(50,120,230),2:(220,60,150),3:(230,145,30)}
    draw.rectangle(tuple(int(v) for v in tower),outline=(35,180,75),width=3);draw.text((int(tower[0]),int(tower[1])),f"tower ROI {tower_score:.3f}",fill=(35,180,75),font=font)
    for p in filtered:
        b=tuple(int(round(v)) for v in p['box']);c=colors[p['class_id']];draw.rectangle(b,outline=c,width=3);draw.text((b[0],b[1]),f"{p['label']} {p['score']:.3f}",fill=c,font=font)
    canvas.save(REPORTS/"v3_1_class_isolated_vp_POS_2326530.jpg",quality=95)

    report={"evidence_type":"class-isolated-yoloe-visual-prompt-ablation","claim_scope":"each component class prompted in a separate YOLOE pass using only training-source examples; POS_2326530 remains adaptive development showcase; masks are pseudo-labels; scores uncalibrated","reference_source":"POS_190181","target_source":"POS_2326530","tower_iou":iou(tower,tower_gt['box']),"tower_score":tower_score,"per_class":per_class,"n_raw":len(all_preds),"n_after_class_nms":len(filtered),"predictions":filtered,"metrics":metrics,"weight":{"sha256":sha256(weight)},"runtime":runtime_env(),"final_holdout_touched":False}
    write_json(REPORTS/"v3_1_class_isolated_vp_metrics.json",report)
    print(json.dumps({"per_class":{k:{"n_ref":v['n_reference_examples'],"n_raw":v['n_raw_predictions']} for k,v in per_class.items()},"n_filtered":len(filtered),"metrics":metrics},indent=2))

if __name__=='__main__':main()
