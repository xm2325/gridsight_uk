from __future__ import annotations

import json
from pathlib import Path

from v23_common import REPORTS,ROOT,WEIGHTS,runtime_env,sha256,write_json
from v25_hybrid_component_detector import TOWER_PROMPT,expand,iou,read_yolo
from v34_tiled_insulator_specialist import metrics

VAL_IDS=['POS_354803','POS_543992']


def oneclass_gt(path:Path,w:int,h:int):
 return [x['box'] for x in read_yolo(path,w,h) if x['class_id']==0]


def original_gt(rid:str):
 from PIL import Image
 image=ROOT/f'data/v4_1_yolo/images/val/{rid}.jpg';label=ROOT/f'data/v4_0_labels/val/{rid}.txt';im=Image.open(image);w,h=im.size;rows=read_yolo(label,w,h)
 return image,next(x['box'] for x in rows if x['class_id']==0),[x['box'] for x in rows if x['class_id']==2],(w,h)


def oracle_rows(model):
 from PIL import Image
 rows=[]
 for rid in VAL_IDS:
  image=ROOT/f'data/v4_2_tower_roi/images/val/{rid}.jpg';label=ROOT/f'data/v4_2_tower_roi/labels/val/{rid}.txt';im=Image.open(image);w,h=im.size;gt=oneclass_gt(label,w,h)
  result=model.predict(str(image),imgsz=960,conf=0.001,device='cpu',verbose=False,max_det=300)[0];raw=[]
  if result.boxes is not None:
   for b,s in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):raw.append({'box':[float(x) for x in b],'score':float(s)})
  rows.append({'rid':rid,'image':image,'gt':gt,'raw':raw})
 return rows


def model_parent_rows(model,yoloe_weight:Path):
 from PIL import Image
 from ultralytics import YOLOE
 tower_model=YOLOE(str(yoloe_weight));tower_model.set_classes([TOWER_PROMPT]);rows=[]
 for rid in VAL_IDS:
  image_path,tower_gt,ins_gt,(w,h)=original_gt(rid);image=Image.open(image_path).convert('RGB')
  tr=tower_model.predict(str(image_path),imgsz=768,conf=0.05,device='cpu',verbose=False)[0]
  if tr.boxes is None or len(tr.boxes)==0:raise RuntimeError(f'no tower ROI {rid}')
  ti=int(tr.boxes.conf.argmax().item());tower_box=[float(x) for x in tr.boxes.xyxy[ti].cpu().tolist()];tower_score=float(tr.boxes.conf[ti])
  roi=expand(tower_box,w,h,px=.08,py=.04);x1,y1,x2,y2=[int(round(v)) for v in roi];crop=image.crop((x1,y1,x2,y2));crop_path=REPORTS/f'v4_2_{rid}_model_parent_crop.jpg';crop.save(crop_path,quality=95)
  result=model.predict(str(crop_path),imgsz=960,conf=.001,device='cpu',verbose=False,max_det=300)[0];raw=[]
  if result.boxes is not None:
   for b,s in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):raw.append({'box':[float(b[0])+x1,float(b[1])+y1,float(b[2])+x1,float(b[3])+y1],'score':float(s)})
  rows.append({'rid':rid,'image':image_path,'gt':ins_gt,'raw':raw,'tower_iou':iou(tower_box,tower_gt),'tower_score':tower_score,'tower_box':tower_box})
 return rows


def pooled(rows,threshold,iou_thr):
 total={'tp':0,'fp':0,'fn':0};per={}
 for r in rows:
  pred=[x for x in r['raw'] if x['score']>=threshold];m=metrics(pred,r['gt'],iou_thr);per[r['rid']]={**{k:v for k,v in m.items() if k!='matches'},'n_predictions':len(pred)}
  for k in total:total[k]+=m[k]
 tp,fp,fn=total['tp'],total['fp'],total['fn'];p=tp/(tp+fp) if tp+fp else 0.;rec=tp/(tp+fn) if tp+fn else 0.
 return {**total,'precision':p,'recall':rec,'f1':2*p*rec/(p+rec) if p+rec else 0.,'per_source':per}


def choose(rows):
 thresholds=sorted(set([.001,.002,.003,.005,.008,.01,.015,.02,.03,.05,.08,.1,.15,.2,.3,.4,.5]+[round(x['score'],6) for r in rows for x in r['raw']]))
 sweep=[]
 for t in thresholds:
  m=pooled(rows,t,.30);sweep.append({'threshold':t,**{k:v for k,v in m.items() if k!='per_source'}})
 best=max(sweep,key=lambda x:(x['f1'],x['recall'],x['precision'],x['threshold']))
 return best,sweep


def render(rows,threshold,prefix):
 from PIL import Image,ImageDraw,ImageFont
 outs=[]
 for r in rows:
  im=Image.open(r['image']).convert('RGB');d=ImageDraw.Draw(im);font=ImageFont.load_default()
  if 'tower_box' in r:
   tb=tuple(int(round(v)) for v in r['tower_box']);d.rectangle(tb,outline=(35,180,75),width=3)
  for p in [x for x in r['raw'] if x['score']>=threshold]:
   b=tuple(int(round(v)) for v in p['box']);d.rectangle(b,outline=(220,60,150),width=3);d.text((b[0],b[1]),f'insulator {p["score"]:.3f}',fill=(220,60,150),font=font)
  out=REPORTS/f'{prefix}_{r["rid"]}.jpg';im.save(out,quality=95);outs.append(out.name)
 return outs


def main():
 from ultralytics import YOLO
 weight=WEIGHTS/'yolo26n.pt';data=ROOT/'data/v4_2_tower_roi/data.yaml';run_dir=ROOT/'runs/v4_2_yolo26n_tower_roi'
 model=YOLO(str(weight));train_result=model.train(data=str(data),epochs=40,imgsz=960,batch=4,device='cpu',workers=0,seed=17,deterministic=True,patience=12,project=str(ROOT/'runs'),name='v4_2_yolo26n_tower_roi',exist_ok=True,cache=False,verbose=True,plots=False)
 best=run_dir/'weights/best.pt'
 if not best.exists():raise FileNotFoundError(best)
 best_model=YOLO(str(best));oracle=oracle_rows(best_model);pipeline=model_parent_rows(best_model,WEIGHTS/'yoloe-26n-seg.pt')
 pipeline_op,pipeline_sweep=choose(pipeline);oracle_op,oracle_sweep=choose(oracle)
 pipeline_metrics={str(t):pooled(pipeline,pipeline_op['threshold'],t) for t in (.30,.50)}
 oracle_same={str(t):pooled(oracle,pipeline_op['threshold'],t) for t in (.30,.50)}
 oracle_best={str(t):pooled(oracle,oracle_op['threshold'],t) for t in (.30,.50)}
 report={'evidence_type':'v4.2-yolo26n-tower-roi-specialist-validation','claim_scope':'validation-only ablation. Training crops use frozen manual parent boxes; end-to-end validation uses YOLOE-generated parent tower ROI. Oracle-parent result is diagnostic only. v4 final holdout untouched.','pretrained_checkpoint':{'name':weight.name,'sha256':sha256(weight)},'best_checkpoint':{'path':str(best.relative_to(ROOT)),'sha256':sha256(best)},'training':{'epochs_requested':40,'imgsz':960,'batch':4,'seed':17,'train_metrics':getattr(train_result,'results_dict',None)},'pipeline_parent':{'tower_iou':{r['rid']:r['tower_iou'] for r in pipeline},'tower_score':{r['rid']:r['tower_score'] for r in pipeline},'selected_operating_point':pipeline_op,'metrics':pipeline_metrics,'sweep':pipeline_sweep,'images':render(pipeline,pipeline_op['threshold'],'v4_2_pipeline')},'oracle_parent':{'selected_operating_point_diagnostic':oracle_op,'metrics_at_pipeline_threshold':oracle_same,'metrics_at_oracle_best_threshold':oracle_best,'sweep':oracle_sweep,'images':render(oracle,oracle_op['threshold'],'v4_2_oracle')},'runtime':runtime_env(),'v4_final_holdout_touched':False}
 write_json(REPORTS/'v4_2_yolo26n_tower_roi_validation.json',report);print(json.dumps({'best_checkpoint':report['best_checkpoint'],'pipeline_op':pipeline_op,'pipeline_metrics':pipeline_metrics,'oracle_op':oracle_op,'oracle_best':oracle_best},indent=2))

if __name__=='__main__':main()
