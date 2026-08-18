from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from v23_common import ROOT
from v25_hybrid_component_detector import expand, read_yolo

OUT=ROOT/'data/v4_2_tower_roi'
LEGACY_SPLITS={
 'POS_1283842':'train','POS_190181':'train','POS_7060068':'train','POS_3778704':'train','POS_291727':'train','POS_5442616':'val','POS_2326530':'test'
}
NEW_TRAIN=['POS_6610209','POS_5952661','POS_7480474','POS_1352733']
NEW_VAL=['POS_354803','POS_543992']
FINAL={'POS_8091164','POS_8239540'}


def xyxy_to_yolo(box,w,h):
 x1,y1,x2,y2=box;return ((x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h)


def crop_one(rid:str,image_path:Path,label_path:Path,role:str):
 im=Image.open(image_path).convert('RGB');w,h=im.size;rows=read_yolo(label_path,w,h)
 tower=next(x['box'] for x in rows if x['class_id']==0);ins=[x['box'] for x in rows if x['class_id']==2]
 roi=expand(tower,w,h,px=0.08,py=0.04);x1,y1,x2,y2=[int(round(v)) for v in roi];crop=im.crop((x1,y1,x2,y2));cw,ch=crop.size
 out_img=OUT/f'images/{role}/{rid}.jpg';out_lab=OUT/f'labels/{role}/{rid}.txt';out_img.parent.mkdir(parents=True,exist_ok=True);out_lab.parent.mkdir(parents=True,exist_ok=True);crop.save(out_img,quality=95)
 lines=[];kept=[]
 for b in ins:
  bx1=max(float(b[0]),x1);by1=max(float(b[1]),y1);bx2=min(float(b[2]),x2);by2=min(float(b[3]),y2)
  if bx2<=bx1 or by2<=by1:continue
  local=[bx1-x1,by1-y1,bx2-x1,by2-y1];xc,yc,bw,bh=xyxy_to_yolo(local,cw,ch)
  if not(0<=xc<=1 and 0<=yc<=1 and 0<bw<=1 and 0<bh<=1):raise RuntimeError((rid,local,crop.size))
  lines.append(f'0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}');kept.append(local)
 if len(lines)!=len(ins):raise RuntimeError(f'{rid}: lost insulators in tower crop {len(lines)}/{len(ins)}')
 out_lab.write_text('\n'.join(lines)+'\n')
 return {'record_id':rid,'role':role,'original_size':[w,h],'tower_box':[float(x) for x in tower],'crop_box':[x1,y1,x2,y2],'crop_size':[cw,ch],'n_insulators':len(lines)}


def main():
 split=json.loads((ROOT/'data/v4_0_split_freeze.json').read_text());ann=json.loads((ROOT/'data/v4_0_annotation_freeze.json').read_text())
 assert split['frozen_before_v4_model_inference'] and ann['created_before_v4_model_inference']
 if OUT.exists():shutil.rmtree(OUT)
 rows=[]
 # v4.1 preparation must already have hydrated/copy-ready legacy images.
 for rid,old_split in LEGACY_SPLITS.items():
  image=ROOT/f'data/images/{old_split}/{rid}.jpg';label=ROOT/f'data/labels/{old_split}/{rid}.txt'
  if not image.exists():raise FileNotFoundError(image)
  rows.append(crop_one(rid,image,label,'train'))
 for rid in NEW_TRAIN:
  image=ROOT/f'data/v4_1_yolo/images/train/{rid}.jpg';label=ROOT/f'data/v4_0_labels/train/{rid}.txt';rows.append(crop_one(rid,image,label,'train'))
 for rid in NEW_VAL:
  image=ROOT/f'data/v4_1_yolo/images/val/{rid}.jpg';label=ROOT/f'data/v4_0_labels/val/{rid}.txt';rows.append(crop_one(rid,image,label,'val'))
 for rid in FINAL:
  if (OUT/f'images/final/{rid}.jpg').exists() or (ROOT/f'data/v4_1_yolo/images/final/{rid}.jpg').exists():raise RuntimeError(f'final visible: {rid}')
 (OUT/'data.yaml').write_text(f'path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: insulator\n')
 report={'status':'PASS','evidence_type':'v4.2-manual-parent-tower-roi-dataset','crop_policy':'manual parent tower box expanded 8% horizontal / 4% vertical; training only; validation manual crop used only as oracle-parent diagnostic','rows':rows,'counts':{'train_sources':sum(r['role']=='train' for r in rows),'train_insulators':sum(r['n_insulators'] for r in rows if r['role']=='train'),'val_sources':sum(r['role']=='val' for r in rows),'val_insulators':sum(r['n_insulators'] for r in rows if r['role']=='val')},'v4_final_holdout_visible':False}
 (ROOT/'reports/v4_2_tower_roi_dataset.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
