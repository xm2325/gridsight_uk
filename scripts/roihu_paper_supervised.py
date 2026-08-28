"""Fixed-budget public-label material localisation pilot. CUDA/Slurm only."""
import json
import os
import shutil
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
from paper_material_demo import ROOT,load,write,sha


def main():
    import torch
    import ultralytics
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPETrainer
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run in Roihu gputest only')
    protocol=ROOT/'configs/paper_supervised_demo_v1.json';cfg=load(protocol)
    data=ROOT/cfg['dataset'];out=ROOT/cfg['run'];source=ROOT/cfg['source_demo']
    if out.exists():raise FileExistsError('Existing run/checkpoint: inspect before any resubmission')
    assert sha(data/'manifest.json')==cfg['manifest_sha256']
    assert sha(source/'manifest.json')==cfg['source_demo_manifest_sha256']
    assert sha(ROOT/cfg['pretrained'])==cfg['pretrained_sha256']
    manifest=load(data/'manifest.json');targets=[r for r in load(source/'manifest.json')['images'] if r['role']=='target']
    for r in manifest['images']:
        assert sha(data/r['image_file'])==r['image_sha256'] and sha(data/r['label_file'])==r['label_sha256']
    for r in targets:assert sha(source/r['file'])==r['sha256']
    assert not {r['sha256'] for r in targets}.intersection(r['image_sha256'] for r in manifest['images'])
    out.mkdir(parents=True);(out/'code').mkdir()
    for p in [Path(__file__),protocol,ROOT/'scripts/prepare_paper_supervised.py',ROOT/'scripts/paper_material_demo.py',ROOT/'scripts/paper_supervised.sbatch']:
        shutil.copyfile(p,out/'code'/p.name)
    trainlist=out/'train.txt';vallist=out/'sanity_validation.txt'
    trainlist.write_text('\n'.join(str(data/r['image_file']) for r in manifest['images'])+'\n')
    vallist.write_text('\n'.join(str(data/f) for f in manifest['sanity_validation_images'])+'\n')
    # JSON is a YAML subset. No external YAML parser or untrusted object tags.
    write(out/'dataset.yaml',{'path':str(data),'train':str(trainlist),'val':str(vallist),'names':cfg['classes']})
    result={'status':'TRAINING','config':cfg,'protocol_sha256':sha(protocol),'started_at':datetime.now(timezone.utc).isoformat(),
        'runtime':{'job_id':os.environ['SLURM_JOB_ID'],'device':torch.cuda.get_device_name(),'torch':torch.__version__,'ultralytics':ultralytics.__version__},
        'training_progress':{},'predictions':[],'validation_is_in_sample':True,'target_labels_used_for_training':False}
    write(out/'results.json',result);start=time.perf_counter()
    try:
        torch.set_num_threads(8)
        model=YOLOE(cfg['yaml']).load(str(ROOT/cfg['pretrained']))
        def on_start(trainer):
            expected={str(data/r['image_file']) for r in manifest['images']}
            assert set(trainer.train_loader.dataset.im_files)==expected
            assert len(trainer.train_loader.dataset.im_files)==len(manifest['images'])
            assert set(trainer.test_loader.dataset.im_files)=={str(data/f) for f in manifest['sanity_validation_images']}
            assert trainer.model.names==dict(enumerate(cfg['classes']))
            result['actual_training_setup']={'image_count':len(expected),'args':vars(trainer.args),'names':trainer.model.names}
            write(out/'results.json',result)
        def on_batch(trainer):
            if trainer.loss is not None and not torch.isfinite(trainer.loss).all():raise ValueError('Non-finite loss; no silent continuation')
        def on_epoch(trainer):
            result['training_progress']={'epochs_completed':trainer.epoch+1,'losses':trainer.loss_items.detach().cpu().tolist(), 'seconds':time.perf_counter()-start}
            write(out/'results.json',result);print(json.dumps(result['training_progress']),flush=True)
        model.add_callback('on_train_start',on_start);model.add_callback('on_train_batch_end',on_batch);model.add_callback('on_fit_epoch_end',on_epoch)
        model.train(data=str(out/'dataset.yaml'),trainer=YOLOEPETrainer,epochs=cfg['epochs'],imgsz=cfg['imgsz'],batch=cfg['batch'],nbs=cfg['batch'],
            optimizer='AdamW',lr0=.0005,lrf=.1,weight_decay=.0005,warmup_epochs=1.,seed=cfg['seed'],workers=8,amp=False,freeze=0,
            cache=False,mosaic=0.,mixup=0.,copy_paste=0.,translate=.05,scale=.15,fliplr=.5,hsv_h=0.,hsv_s=.1,hsv_v=.2,
            device=0,deterministic=True,compile=False,project=str(out),name='training',exist_ok=False,pretrained=True,
            patience=cfg['epochs']+1,plots=False,verbose=False,save=True,save_period=-1,cos_lr=False,close_mosaic=0,
            val=False,degrees=0.,shear=0.,perspective=0.,flipud=0.)
        assert result['training_progress']['epochs_completed']==cfg['epochs']
        checkpoint=Path(model.trainer.last);assert checkpoint.exists()
        result.update(status='INFERENCE',checkpoint=str(checkpoint.relative_to(out)),checkpoint_sha256=sha(checkpoint),training_seconds=time.perf_counter()-start)
        write(out/'results.json',result)
        model=YOLOE(str(checkpoint)).to('cuda:0')
        for row in targets:
            t=time.perf_counter();pred=model.predict(str(source/row['file']),device=0,verbose=False,save=False,half=False,**cfg['inference'])[0]
            predictions=[{'class_id':int(c),'class_name':cfg['classes'][int(c)],'score':float(s),'box':list(map(float,b)),
                'material_verified':False,'material_probability_calibrated':False} for b,s,c in zip(pred.boxes.xyxy.cpu().tolist(),pred.boxes.conf.cpu().tolist(),pred.boxes.cls.cpu().tolist())]
            f=out/f'predictions/{row["id"]}.json'
            write(f,{'image_id':row['id'],'image_sha256':row['sha256'],'predictions':predictions,'checkpoint_sha256':result['checkpoint_sha256'],
                'protocol_sha256':sha(protocol),'seconds':time.perf_counter()-t,'output_stage':'All post-NMS outputs above .05, no target annotations used.'})
            result['predictions'].append({'image_id':row['id'],'file':str(f.relative_to(out)),'sha256':sha(f),'count':len(predictions)})
            write(out/'results.json',result);print(json.dumps({'inferred':row['id'],'boxes':len(predictions)}),flush=True)
        result.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-start);write(out/'results.json',result)
    except BaseException as e:
        result.update(status='FAILED',error=str(e),traceback=traceback.format_exc());write(out/'results.json',result);raise


if __name__=='__main__':main()
