from __future__ import annotations

import argparse
import json
from pathlib import Path

from v23_common import REPORTS, ROOT, WEIGHTS, runtime_env, sha256, write_json
from v25_hybrid_component_detector import TOWER_PROMPT, expand, iou, read_yolo


def biou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0.0,x2-x1)*max(0.0,y2-y1);aa=max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1]);bb=max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1]);den=aa+bb-inter
    return inter/den if den else 0.0


def nms(rows,thr):
    cand=sorted(rows,key=lambda x:x['score'],reverse=True);out=[]
    while cand:
        b=cand.pop(0);out.append(b);cand=[x for x in cand if biou(b['box'],x['box'])<thr]
    return out


def metrics(preds,gt,thr):
    used=set();matches=[]
    for i,p in enumerate(sorted(preds,key=lambda x:x['score'],reverse=True)):
        cand=[(biou(p['box'],g),j) for j,g in enumerate(gt) if j not in used]
        if cand:
            ov,j=max(cand)
            if ov>=thr:used.add(j);matches.append({'pred_rank':i,'gt_index':j,'iou':ov})
    tp=len(matches);fp=len(preds)-tp;fn=len(gt)-tp;p=tp/(tp+fp) if tp+fp else 0.0;r=tp/(tp+fn) if tp+fn else 0.0
    return {'iou_threshold':thr,'tp':tp,'fp':fp,'fn':fn,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.0,'matches':matches}


def tile_windows(W,H):
    # 2x2 overlap tiles: 65% width x 60% height, preserving context while magnifying small insulators.
    tw=max(32,int(round(W*0.65)));th=max(32,int(round(H*0.60)))
    xs=[0,max(0,W-tw)];ys=[0,max(0,H-th)]
    return [(x,y,min(W,x+tw),min(H,y+th)) for y in ys for x in xs]


def parent_crop(image_path:Path,label_path:Path):
    from PIL import Image
    im=Image.open(image_path).convert('RGB');W,H=im.size;labels=read_yolo(label_path,W,H);tower=next(x for x in labels if x['class_id']==0);roi=expand(tower['box'],W,H,px=0.08,py=0.04);x1,y1,x2,y2=[int(round(v)) for v in roi]
    return im.crop((x1,y1,x2,y2)),(x1,y1,x2,y2),[x for x in labels if x['class_id']==2],(W,H)


def build_tiled_split(split:str,dst:Path):
    manifest=[];unique_gt=0;tiled_instances=0
    for image_path in sorted((ROOT/f'data/images/{split}').glob('*.jpg')):
        crop,roi,insulators,_=parent_crop(image_path,ROOT/f'data/labels/{split}/{image_path.stem}.txt');x0,y0,_,_=roi;W,H=crop.size;unique_gt+=len(insulators)
        source_rows=[]
        for ti,(tx1,ty1,tx2,ty2) in enumerate(tile_windows(W,H)):
            tile=crop.crop((tx1,ty1,tx2,ty2));tw,th=tile.size;ys=[]
            for g in insulators:
                # convert full-image GT to tower-crop coordinates
                b=[g['box'][0]-x0,g['box'][1]-y0,g['box'][2]-x0,g['box'][3]-y0];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
                if not(tx1<=cx<=tx2 and ty1<=cy<=ty2):continue
                cl=[max(tx1,b[0]),max(ty1,b[1]),min(tx2,b[2]),min(ty2,b[3])]
                if cl[2]<=cl[0] or cl[3]<=cl[1]:continue
                lx1,ly1,lx2,ly2=cl[0]-tx1,cl[1]-ty1,cl[2]-tx1,cl[3]-ty1
                xc=((lx1+lx2)/2)/tw;yc=((ly1+ly2)/2)/th;ww=(lx2-lx1)/tw;hh=(ly2-ly1)/th
                ys.append(f'0 {xc:.8f} {yc:.8f} {ww:.8f} {hh:.8f}')
            if not ys:continue
            stem=f'{image_path.stem}_tile{ti}';out_img=dst/f'images/{split}/{stem}.jpg';out_lab=dst/f'labels/{split}/{stem}.txt';out_img.parent.mkdir(parents=True,exist_ok=True);out_lab.parent.mkdir(parents=True,exist_ok=True);tile.save(out_img,quality=95);out_lab.write_text('\n'.join(ys)+'\n');tiled_instances+=len(ys);source_rows.append({'tile':stem,'xyxy_in_parent_crop':[tx1,ty1,tx2,ty2],'n_labels':len(ys),'dimensions':[tw,th]})
        manifest.append({'source':image_path.stem,'parent_roi_xyxy':list(roi),'parent_crop_dimensions':[W,H],'n_unique_insulators':len(insulators),'tiles':source_rows})
    return manifest,unique_gt,tiled_instances


def yoloe_dev_parent(yoloe,image_path:Path,out_path:Path):
    from PIL import Image
    im=Image.open(image_path).convert('RGB');W,H=im.size;yoloe.set_classes([TOWER_PROMPT]);r=yoloe.predict(str(image_path),imgsz=768,conf=0.05,device='cpu',verbose=False)[0]
    idx=int(r.boxes.conf.argmax().item());tb=[float(x) for x in r.boxes.xyxy[idx].cpu().tolist()];ts=float(r.boxes.conf[idx].item());roi=expand(tb,W,H,px=0.08,py=0.04);x1,y1,x2,y2=[int(round(v)) for v in roi];crop=im.crop((x1,y1,x2,y2));out_path.parent.mkdir(parents=True,exist_ok=True);crop.save(out_path,quality=95)
    return crop,(x1,y1,x2,y2),tb,ts,(W,H)


def infer_tiles(model,parent_crop,roi,tmp_prefix,imgsz=960,conf=0.001):
    x0,y0,_,_=roi;W,H=parent_crop.size;tmp=REPORTS/'v3_4_tiles';tmp.mkdir(parents=True,exist_ok=True);rows=[]
    for ti,(tx1,ty1,tx2,ty2) in enumerate(tile_windows(W,H)):
        p=tmp/f'{tmp_prefix}_tile{ti}.jpg';parent_crop.crop((tx1,ty1,tx2,ty2)).save(p,quality=95);r=model.predict(str(p),imgsz=imgsz,conf=conf,device='cpu',verbose=False,max_det=300)[0]
        if r.boxes is not None:
            for b,s in zip(r.boxes.xyxy.cpu().tolist(),r.boxes.conf.cpu().tolist()):
                rows.append({'score':float(s),'box':[float(b[0])+tx1+x0,float(b[1])+ty1+y0,float(b[2])+tx1+x0,float(b[3])+ty1+y0],'tile_index':ti})
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=80);args=ap.parse_args()
    from PIL import Image,ImageDraw,ImageFont
    from ultralytics import YOLO,YOLOE
    dst=ROOT/'data/v34_tiled_insulator';train,train_unique,train_instances=build_tiled_split('train',dst);val,val_unique,val_instances=build_tiled_split('val',dst)
    if train_unique!=30 or len(train)!=5:raise RuntimeError(f'Expected 5 sources/30 unique train insulators, got {len(train)}/{train_unique}')
    if val_unique!=4 or len(val)!=1:raise RuntimeError(f'Expected 1 source/4 val insulators, got {len(val)}/{val_unique}')
    (dst/'data.yaml').write_text('path: data/v34_tiled_insulator\ntrain: images/train\nval: images/val\nnames:\n  0: insulator_string\n')
    yolo26=WEIGHTS/'yolo26n.pt';yoloe_weight=WEIGHTS/'yoloe-26n-seg.pt';model=YOLO(str(yolo26));model.train(data=str(dst/'data.yaml'),epochs=args.epochs,imgsz=960,batch=1,workers=0,device='cpu',seed=17,deterministic=True,pretrained=True,project=str(ROOT/'runs'),name='v3_4_yolo26n_tiled_insulator',exist_ok=True,plots=False,verbose=True,mosaic=0.0,close_mosaic=0,translate=0.04,scale=0.15,fliplr=0.5,optimizer='AdamW',lr0=0.001,lrf=0.01)
    best=ROOT/'runs/v3_4_yolo26n_tiled_insulator/weights/best.pt';last=ROOT/'runs/v3_4_yolo26n_tiled_insulator/weights/last.pt';chosen=best if best.exists() else last;trained=YOLO(str(chosen))
    # validation: manual parent ROI allowed; inference tiled exactly as training.
    val_img=ROOT/'data/images/val/POS_5442616.jpg';vcrop,vroi,vgt_all,(VW,VH)=parent_crop(val_img,ROOT/'data/labels/val/POS_5442616.txt');vgt=[x['box'] for x in vgt_all];vraw=infer_tiles(trained,vcrop,vroi,'val',960,0.001)
    confs=sorted(set([0.001,0.002,0.003,0.004,0.005,0.006,0.008,0.010,0.012,0.015,0.020,0.030,0.050,0.075,0.10,0.15,0.20]+[round(x['score'],6) for x in vraw]));nmses=[0.20,0.25,0.30,0.40,0.50];sweep=[]
    for conf in confs:
        for ni in nmses:
            f=nms([x for x in vraw if x['score']>=conf],ni);m=metrics(f,vgt,0.30);sweep.append({'conf_threshold':conf,'nms_iou':ni,'n_predictions':len(f),**{k:v for k,v in m.items() if k!='matches'}})
    op=max(sweep,key=lambda x:(x['f1'],x['recall'],x['precision'],-x['n_predictions'],x['conf_threshold'],-x['nms_iou']))
    # development uses model-generated YOLOE parent ROI.
    yoloe=YOLOE(str(yoloe_weight));dcrop,droi,tb,ts,(W,H)=yoloe_dev_parent(yoloe,ROOT/'data/images/test/POS_2326530.jpg',REPORTS/'v3_4_dev_parent_crop.jpg');dgt_all=read_yolo(ROOT/'data/labels/test/POS_2326530.txt',W,H);dgt=[x['box'] for x in dgt_all if x['class_id']==2];tower_gt=next(x for x in dgt_all if x['class_id']==0);draw_raw=infer_tiles(trained,dcrop,droi,'dev',960,0.001);df=nms([x for x in draw_raw if x['score']>=op['conf_threshold']],op['nms_iou']);dm=[metrics(df,dgt,t) for t in (0.30,0.50)]
    image=Image.open(ROOT/'data/images/test/POS_2326530.jpg').convert('RGB');draw=ImageDraw.Draw(image);font=ImageFont.load_default();draw.rectangle(tuple(int(v) for v in tb),outline=(35,180,75),width=3)
    for p in df:
        b=tuple(int(round(v)) for v in p['box']);draw.rectangle(b,outline=(220,60,150),width=3);draw.text((b[0],b[1]),f'insulator {p["score"]:.3f}',fill=(220,60,150),font=font)
    image.save(REPORTS/'v3_4_tiled_insulator_POS_2326530.jpg',quality=95)
    report={'evidence_type':'pretrained-yolo26-tiled-one-class-insulator-specialist','claim_scope':'controlled scale ablation: same 5 train sources/30 unique insulator GT as v3.2; 2x2 overlapping tower-ROI tiles increase pixel scale; validation-only operating point; final holdout untouched','epochs':args.epochs,'training':{'n_sources':len(train),'n_unique_insulators':train_unique,'n_tiled_label_instances':train_instances,'manifest':train},'validation':{'n_unique_insulators':val_unique,'n_tiled_label_instances':val_instances,'n_raw':len(vraw),'best':op,'sweep':sweep},'development_showcase':{'source':'POS_2326530','tower_iou':iou(tb,tower_gt['box']),'tower_score':ts,'n_raw':len(draw_raw),'n_filtered':len(df),'metrics':dm,'predictions':df},'weights':{'input_yolo26':sha256(yolo26),'input_yoloe':sha256(yoloe_weight),'trained':sha256(chosen)},'runtime':runtime_env(),'final_holdout_touched':False}
    write_json(REPORTS/'v3_4_tiled_insulator_metrics.json',report);print(json.dumps({'train_unique':train_unique,'train_tiled_instances':train_instances,'val_best':op,'dev_n':len(df),'dev_metrics':dm},indent=2))

if __name__=='__main__':main()
