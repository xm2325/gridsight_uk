"""Inference-only prospective UK porcelain check. CUDA/Slurm only."""
from __future__ import annotations

import json,math,os,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from paper_material_demo import ROOT,load,sha,write

def iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.

def gate(box,predictions,cfg):
    settings=cfg['preregistered_diagnostic_gate'];width=box['xyxy'][2]-box['xyxy'][0];height=box['xyxy'][3]-box['xyxy'][1]
    reasons=[]
    if box['raw_detector_score']<settings['display_score_floor']:reasons.append('raw score below fixed display floor')
    if min(width,height)<settings['minimum_native_side'] or width*height<settings['minimum_native_area']:reasons.append('insufficient native box pixels')
    conflicts=[other for other in predictions if other is not box and other['class_name']!=box['class_name'] and
      other['raw_detector_score']>=settings['conflicting_class_score_floor'] and iou(box['xyxy'],other['xyxy'])>=settings['overlap_conflict_iou']]
    if conflicts:
        margin=box['raw_detector_score']-max(other['raw_detector_score'] for other in conflicts)
        if margin<settings['minimum_class_margin']:reasons.append('overlapping material-class conflict')
    return {'status':'unknown' if reasons else box['class_name'],'reasons':reasons or ['passes fixed diagnostic gate; score remains uncalibrated']}

def main():
    import torch,ultralytics
    from ultralytics import YOLOE
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run on Roihu gputest only')
    protocol=ROOT/'configs/uk_prospective_porcelain_v1.json';cfg=load(protocol);out=ROOT/cfg['run']
    if out.exists():raise FileExistsError('Existing prospective result; never overwrite or repeat automatically')
    for key in ('checkpoint','image','freeze'):
        if sha(ROOT/cfg[key])!=cfg[key+'_sha256']:raise ValueError(f'{key} bytes differ from frozen protocol')
    freeze=load(ROOT/cfg['freeze']);reference=freeze['references'][0];out.mkdir(parents=True)
    result={'status':'INFERENCE','started_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(protocol),
      'freeze_sha256':cfg['freeze_sha256'],'checkpoint_sha256':cfg['checkpoint_sha256'],'raw_predictions':[],
      'runtime':{'job_id':os.environ['SLURM_JOB_ID'],'device':torch.cuda.get_device_name(),'torch':torch.__version__,'ultralytics':ultralytics.__version__},
      'claim_boundary':cfg['claim_boundary']};write(out/'results.json',result);started=time.perf_counter()
    try:
        model=YOLOE(str(ROOT/cfg['checkpoint'])).to('cuda:0')
        pred=model.predict(str(ROOT/cfg['image']),device=0,verbose=False,save=False,half=False,**cfg['inference'])[0]
        boxes=[{'class_id':int(cls),'class_name':cfg['classes'][int(cls)],'raw_detector_score':float(score),'xyxy':list(map(float,box)),
          'material_probability_calibrated':False} for box,score,cls in zip(pred.boxes.xyxy.cpu().tolist(),pred.boxes.conf.cpu().tolist(),pred.boxes.cls.cpu().tolist())]
        for box in boxes:box['diagnostic_gate']=gate(box,boxes,cfg)
        candidates=[box for box in boxes if iou(reference['xyxy'],box['xyxy'])>=cfg['preregistered_diagnostic_gate']['reference_iou']]
        selected=max(candidates,key=lambda box:box['raw_detector_score']) if candidates else None
        result.update(status='COMPLETE',raw_predictions=boxes,reference=reference,
          prospective_diagnostic={'localised_at_iou_0_5':bool(candidates),'matching_prediction_count':len(candidates),
            'selected_prediction':selected,'selected_iou':iou(reference['xyxy'],selected['xyxy']) if selected else None,
            'material_correct_if_localised':selected['class_name']==reference['material'] if selected else None,
            'gate_output':selected['diagnostic_gate'] if selected else {'status':'unknown','reasons':['no prediction overlaps frozen complete-assembly box at IoU 0.5']}},
          elapsed_seconds=time.perf_counter()-started)
        write(out/'results.json',result)
    except BaseException as error:
        result.update(status='FAILED',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started);write(out/'results.json',result);raise

if __name__=='__main__':main()
