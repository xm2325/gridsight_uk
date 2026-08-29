"""Bounded MPID direct material detector diagnostic. CUDA and Slurm only."""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path

from paper_material_demo import ROOT,load,sha,write
from prepare_mpid_material import prepare,verify

def main():
    import torch,ultralytics
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPETrainer
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run in Roihu gputest only')
    protocol=ROOT/'configs/mpid_material_detector_v1.json';cfg=load(protocol);out=ROOT/cfg['run'];dataset=ROOT/cfg['dataset']
    if out.exists():raise FileExistsError('Existing run: inspect before any resubmission')
    manifest=verify() if (dataset/'manifest.json').exists() else prepare()
    if sha(ROOT/cfg['uk_target_manifest'])!=cfg['uk_target_manifest_sha256']:raise ValueError('UK target manifest differs')
    if sha(ROOT/cfg['pretrained'])!=cfg['pretrained_sha256']:raise ValueError('Pretrained checkpoint differs')
    targets=load(ROOT/cfg['uk_target_manifest'])['images']
    for row in targets:
        path=ROOT/row['image_file']
        if sha(path)!=row['image_sha256']:raise ValueError(f"UK target bytes differ: {row['record_id']}")
    out.mkdir(parents=True);(out/'code').mkdir()
    for path in [Path(__file__),protocol,ROOT/'scripts/prepare_mpid_material.py',ROOT/'scripts/audit_mpid_archives.py',ROOT/'scripts/mpid_material_detector.sbatch']:
        shutil.copyfile(path,out/'code'/path.name)
    result={'status':'TRAINING','started_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(protocol),
      'dataset_manifest_sha256':sha(dataset/'manifest.json'),'dataset_counts':manifest['counts'],
      'runtime':{'job_id':os.environ['SLURM_JOB_ID'],'device':torch.cuda.get_device_name(),'torch':torch.__version__,'ultralytics':ultralytics.__version__},
      'training_progress':{},'uk_predictions':[],'claim_boundary':cfg['claim_boundary']}
    write(out/'results.json',result);started=time.perf_counter()
    try:
        torch.set_num_threads(8);model=YOLOE(cfg['yaml']).load(str(ROOT/cfg['pretrained']))
        def on_epoch(trainer):
            result['training_progress']={'epochs_completed':trainer.epoch+1,'losses':trainer.loss_items.detach().cpu().tolist(),'seconds':time.perf_counter()-started}
            write(out/'results.json',result);print(json.dumps(result['training_progress']),flush=True)
        def on_batch(trainer):
            if trainer.loss is not None and not torch.isfinite(trainer.loss).all():raise ValueError('Non-finite loss')
        model.add_callback('on_fit_epoch_end',on_epoch);model.add_callback('on_train_batch_end',on_batch)
        model.train(data=str(dataset/'dataset.yaml'),trainer=YOLOEPETrainer,epochs=cfg['epochs'],imgsz=cfg['imgsz'],batch=cfg['batch'],nbs=cfg['batch'],
          optimizer='AdamW',lr0=.0005,lrf=.1,weight_decay=.0005,warmup_epochs=.5,seed=cfg['seed'],workers=16,amp=False,freeze=0,
          cache=False,mosaic=.2,mixup=0.,copy_paste=0.,translate=.05,scale=.2,fliplr=.5,hsv_h=0.,hsv_s=.1,hsv_v=.2,
          device=0,deterministic=True,compile=False,project=str(out),name='training',exist_ok=False,pretrained=True,
          patience=cfg['epochs']+1,plots=False,verbose=False,save=True,save_period=-1,cos_lr=False,close_mosaic=0,val=False)
        if result['training_progress'].get('epochs_completed')!=cfg['epochs']:raise RuntimeError('Fixed training budget did not complete')
        checkpoint=Path(model.trainer.last);result.update(status='VALIDATING',checkpoint=str(checkpoint.relative_to(out)),checkpoint_sha256=sha(checkpoint),training_seconds=time.perf_counter()-started)
        write(out/'results.json',result);model=YOLOE(str(checkpoint)).to('cuda:0')
        metrics=model.val(data=str(dataset/'dataset.yaml'),split='val',imgsz=cfg['imgsz'],batch=cfg['batch'],device=0,plots=False,verbose=False)
        result['mpid_development_metrics']={key:float(value) for key,value in metrics.results_dict.items()}
        result['mpid_metrics_scope']='Internal filename-family development diagnostic; not authoritative upstream-source independence and not UK accuracy.'
        result['status']='UK_INFERENCE';write(out/'results.json',result)
        for row in targets:
            tick=time.perf_counter();prediction=model.predict(str(ROOT/row['image_file']),device=0,verbose=False,save=False,half=False,**cfg['inference'])[0]
            boxes=[{'class_id':int(cls),'class_name':cfg['classes'][int(cls)],'raw_detector_score':float(score),
              'xyxy':list(map(float,box)),'material_probability_calibrated':False,'reference_truth':False}
              for box,score,cls in zip(prediction.boxes.xyxy.cpu().tolist(),prediction.boxes.conf.cpu().tolist(),prediction.boxes.cls.cpu().tolist())]
            target=out/'predictions'/f"{row['record_id']}.json"
            write(target,{'record_id':row['record_id'],'image_sha256':row['image_sha256'],'boxes':boxes,
              'inference_seconds':time.perf_counter()-tick,'checkpoint_sha256':result['checkpoint_sha256'],'protocol_sha256':sha(protocol),
              'warning':'Raw uncalibrated model output. Source evidence is separate; these predictions are not ground truth.'})
            result['uk_predictions'].append({'record_id':row['record_id'],'file':str(target.relative_to(out)),'sha256':sha(target),'count':len(boxes)})
            write(out/'results.json',result);print(json.dumps({'target':row['record_id'],'boxes':len(boxes)}),flush=True)
        result.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-started);write(out/'results.json',result)
    except BaseException as error:
        result.update(status='FAILED',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started);write(out/'results.json',result);raise

if __name__=='__main__':main()
