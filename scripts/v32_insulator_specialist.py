from __future__ import annotations

import argparse
import json
from pathlib import Path

from v23_common import REPORTS, ROOT, WEIGHTS, runtime_env, sha256, write_json
from v25_hybrid_component_detector import TOWER_PROMPT, expand, iou, read_yolo


def box_iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0.0,x2-x1)*max(0.0,y2-y1)
    aa=max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1]);bb=max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1]);den=aa+bb-inter
    return inter/den if den else 0.0


def nms(rows,thr):
    cand=sorted(rows,key=lambda x:x['score'],reverse=True);out=[]
    while cand:
        best=cand.pop(0);out.append(best);cand=[x for x in cand if box_iou(best['box'],x['box'])<thr]
    return out


def metrics(preds,gt,thr=0.30):
    used=set();matches=[]
    for i,p in enumerate(sorted(preds,key=lambda x:x['score'],reverse=True)):
        cand=[(box_iou(p['box'],g),j) for j,g in enumerate(gt) if j not in used]
        if cand:
            bi,j=max(cand)
            if bi>=thr:used.add(j);matches.append({'pred_rank':i,'gt_index':j,'iou':bi})
    tp=len(matches);fp=len(preds)-tp;fn=len(gt)-tp;p=tp/(tp+fp) if tp+fp else 0.0;r=tp/(tp+fn) if tp+fn else 0.0
    return {'iou_threshold':thr,'tp':tp,'fp':fp,'fn':fn,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.0,'matches':matches}


def make_crop_dataset(split:str,dst:Path):
    from PIL import Image
    manifest=[]
    for img_path in sorted((ROOT/f'data/images/{split}').glob('*.jpg')):
        im=Image.open(img_path).convert('RGB');W,H=im.size
        labels=read_yolo(ROOT/f'data/labels/{split}/{img_path.stem}.txt',W,H)
        tower=next(x for x in labels if x['class_id']==0);roi=expand(tower['box'],W,H,px=0.08,py=0.04);x1,y1,x2,y2=[int(round(v)) for v in roi];cw=x2-x1;ch=y2-y1
        out_img=dst/f'images/{split}/{img_path.name}';out_img.parent.mkdir(parents=True,exist_ok=True);im.crop((x1,y1,x2,y2)).save(out_img,quality=95)
        ys=[];kept=[]
        for obj in labels:
            if obj['class_id']!=2:continue
            bx=obj['box'];cx=(bx[0]+bx[2])/2;cy=(bx[1]+bx[3])/2
            if not(x1<=cx<=x2 and y1<=cy<=y2):continue
            cl=[max(x1,bx[0]),max(y1,bx[1]),min(x2,bx[2]),min(y2,bx[3])]
            if cl[2]<=cl[0] or cl[3]<=cl[1]:continue
            lx1,ly1,lx2,ly2=cl[0]-x1,cl[1]-y1,cl[2]-x1,cl[3]-y1
            xc=((lx1+lx2)/2)/cw;yc=((ly1+ly2)/2)/ch;ww=(lx2-lx1)/cw;hh=(ly2-ly1)/ch
            ys.append(f'0 {xc:.8f} {yc:.8f} {ww:.8f} {hh:.8f}');kept.append(bx)
        out_lab=dst/f'labels/{split}/{img_path.stem}.txt';out_lab.parent.mkdir(parents=True,exist_ok=True);out_lab.write_text('\n'.join(ys)+'\n')
        manifest.append({'source':img_path.stem,'roi_xyxy':[x1,y1,x2,y2],'crop_dimensions':[cw,ch],'n_insulators':len(kept)})
    return manifest


def model_generated_dev_crop(yoloe,image_path:Path,out_path:Path):
    from PIL import Image
    im=Image.open(image_path).convert('RGB');W,H=im.size;yoloe.set_classes([TOWER_PROMPT]);r=yoloe.predict(str(image_path),imgsz=768,conf=0.05,device='cpu',verbose=False)[0]
    if r.boxes is None or len(r.boxes)==0:raise RuntimeError('No YOLOE tower ROI')
    idx=int(r.boxes.conf.argmax().item());tb=[float(x) for x in r.boxes.xyxy[idx].cpu().tolist()];ts=float(r.boxes.conf[idx].item());roi=expand(tb,W,H,px=0.08,py=0.04);x1,y1,x2,y2=[int(round(v)) for v in roi]
    out_path.parent.mkdir(parents=True,exist_ok=True);im.crop((x1,y1,x2,y2)).save(out_path,quality=95)
    return (x1,y1,x2,y2),tb,ts,(W,H)


def infer(model,crop_path:Path,roi,conf=0.001):
    x1,y1,_,_=roi;r=model.predict(str(crop_path),imgsz=960,conf=conf,device='cpu',verbose=False,max_det=300)[0];rows=[]
    if r.boxes is not None:
        for b,s in zip(r.boxes.xyxy.cpu().tolist(),r.boxes.conf.cpu().tolist()):
            rows.append({'score':float(s),'box':[float(b[0])+x1,float(b[1])+y1,float(b[2])+x1,float(b[3])+y1]})
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=100);args=ap.parse_args()
    from PIL import Image,ImageDraw,ImageFont
    from ultralytics import YOLO,YOLOE

    dst=ROOT/'data/v32_insulator_specialist';train=make_crop_dataset('train',dst);val=make_crop_dataset('val',dst)
    ntrain=sum(x['n_insulators'] for x in train);nval=sum(x['n_insulators'] for x in val)
    if len(train)!=5 or ntrain!=30:raise RuntimeError(f'Expected 5 train/30 insulators, got {len(train)}/{ntrain}')
    if len(val)!=1 or nval<4:raise RuntimeError(f'Unexpected val manifest {val}')
    (dst/'data.yaml').write_text('path: data/v32_insulator_specialist\ntrain: images/train\nval: images/val\nnames:\n  0: insulator_string\n')

    input_weight=WEIGHTS/'yolo26n.pt';yoloe_weight=WEIGHTS/'yoloe-26n-seg.pt'
    model=YOLO(str(input_weight));model.train(data=str(dst/'data.yaml'),epochs=args.epochs,imgsz=960,batch=1,workers=0,device='cpu',seed=17,deterministic=True,pretrained=True,project=str(ROOT/'runs'),name='v3_2_yolo26n_insulator_specialist',exist_ok=True,plots=False,verbose=True,mosaic=0.0,close_mosaic=0,translate=0.05,scale=0.20,fliplr=0.5,optimizer='AdamW',lr0=0.001,lrf=0.01)
    best=ROOT/'runs/v3_2_yolo26n_insulator_specialist/weights/best.pt';last=ROOT/'runs/v3_2_yolo26n_insulator_specialist/weights/last.pt';chosen=best if best.exists() else last
    if not chosen.exists():raise RuntimeError('No specialist checkpoint')
    trained=YOLO(str(chosen))

    # Validation selection uses the single frozen validation source only.
    val_crop=ROOT/'data/v32_insulator_specialist/images/val/POS_5442616.jpg';v=val[0];val_roi=tuple(v['roi_xyxy']);vim=Image.open(ROOT/'data/images/val/POS_5442616.jpg');VW,VH=vim.size;vgt=[x['box'] for x in read_yolo(ROOT/'data/labels/val/POS_5442616.txt',VW,VH) if x['class_id']==2]
    val_raw=infer(trained,val_crop,val_roi,conf=0.001)
    confs=sorted(set([0.001,0.002,0.003,0.004,0.005,0.006,0.008,0.010,0.012,0.015,0.020,0.030,0.050,0.075,0.10,0.15,0.20]+[round(x['score'],6) for x in val_raw]));nmses=[0.25,0.30,0.40,0.50,0.60]
    sweep=[]
    for conf in confs:
        for ni in nmses:
            filt=nms([x for x in val_raw if x['score']>=conf],ni);m=metrics(filt,vgt,0.30);sweep.append({'conf_threshold':conf,'nms_iou':ni,'n_predictions':len(filt),**{k:v for k,v in m.items() if k!='matches'}})
    bestop=max(sweep,key=lambda x:(x['f1'],x['recall'],x['precision'],-x['n_predictions'],x['conf_threshold'],-x['nms_iou']))

    # Development showcase crop is model generated by YOLOE; no dev parent GT used to generate it.
    yoloe=YOLOE(str(yoloe_weight));dev_crop=REPORTS/'v3_2_dev_yoloe_tower_roi.jpg';dev_roi,tower_box,tower_score,(W,H)=model_generated_dev_crop(yoloe,ROOT/'data/images/test/POS_2326530.jpg',dev_crop)
    dev_gt_all=read_yolo(ROOT/'data/labels/test/POS_2326530.txt',W,H);dev_gt=[x['box'] for x in dev_gt_all if x['class_id']==2];tower_gt=next(x for x in dev_gt_all if x['class_id']==0)
    dev_raw=infer(trained,dev_crop,dev_roi,conf=0.001);dev_filt=nms([x for x in dev_raw if x['score']>=bestop['conf_threshold']],bestop['nms_iou']);dev_metrics=[metrics(dev_filt,dev_gt,t) for t in (0.30,0.50)]

    image=Image.open(ROOT/'data/images/test/POS_2326530.jpg').convert('RGB');draw=ImageDraw.Draw(image);font=ImageFont.load_default();draw.rectangle(tuple(int(v) for v in tower_box),outline=(35,180,75),width=3);draw.text((int(tower_box[0]),int(tower_box[1])),f'tower ROI {tower_score:.3f}',fill=(35,180,75),font=font)
    for p in dev_filt:
        b=tuple(int(round(v)) for v in p['box']);draw.rectangle(b,outline=(220,60,150),width=3);draw.text((b[0],b[1]),f'insulator {p["score"]:.3f}',fill=(220,60,150),font=font)
    image.save(REPORTS/'v3_2_insulator_specialist_POS_2326530.jpg',quality=95)

    report={'evidence_type':'pretrained-yolo26-one-class-insulator-specialist','claim_scope':'exploratory 5-train/1-val/adaptive-development pilot; operating point selected on validation only; final holdout untouched; scores uncalibrated','epochs':args.epochs,'train_manifest':train,'val_manifest':val,'n_train_insulators':ntrain,'selection':{'objective':'max insulator F1@IoU0.30 on POS_5442616 only','n_val_raw':len(val_raw),'best':bestop,'sweep':sweep},'development_showcase':{'source':'POS_2326530','tower_iou':iou(tower_box,tower_gt['box']),'tower_score':tower_score,'n_raw':len(dev_raw),'n_filtered':len(dev_filt),'metrics':dev_metrics,'predictions':dev_filt},'weights':{'input_yolo26':sha256(input_weight),'input_yoloe':sha256(yoloe_weight),'trained_specialist':sha256(chosen)},'runtime':runtime_env(),'final_holdout_touched':False}
    write_json(REPORTS/'v3_2_insulator_specialist_metrics.json',report);print(json.dumps({'n_train_insulators':ntrain,'val_best':bestop,'dev_n':len(dev_filt),'dev_metrics':dev_metrics},indent=2))

if __name__=='__main__':main()
