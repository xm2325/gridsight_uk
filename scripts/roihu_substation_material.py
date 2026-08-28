"""One fixed public-label experiment; execution requires a gputest CUDA job."""
import argparse
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from paper_material_demo import ROOT, load, sha, write
from prepare_substation_material import CONFIG, label_text, polygon_references


def main(manifest_sha):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'gputest':
        raise RuntimeError('Run only through a Roihu gputest allocation')
    import torch
    import ultralytics
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPETrainer
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    cfg = load(CONFIG); data = ROOT/cfg['dataset']; out = ROOT/cfg['run']
    assert sha(data/'manifest.json') == manifest_sha
    manifest = load(data/'manifest.json')
    assert manifest['protocol_sha256'] == sha(CONFIG)
    assert sha(data/'selection_audit.json') == manifest['selection_audit_sha256']
    assert sha(ROOT/cfg['pretrained']) == cfg['pretrained_sha256']
    source = ROOT/cfg['source_demo']
    assert sha(source/'manifest.json') == cfg['source_demo_manifest_sha256']
    train = [r for r in manifest['images'] if r['split']=='train']
    dev = [r for r in manifest['images'] if r['split']=='development']
    external = [dict(r, split='external_demo') for r in load(source/'manifest.json')['images'] if r['role']=='target']
    assert train and dev and len(external)==8
    assert not {r['capture_group'] for r in train} & {r['capture_group'] for r in dev}
    assert not {r['duplicate_cluster'] for r in train} & {r['duplicate_cluster'] for r in dev}
    assert not {r['image_sha256'] for r in train} & ({r['image_sha256'] for r in dev}|{r['sha256'] for r in external})
    for row in train+dev:
        for f,h in [('image_file','image_sha256'),('annotation_file','annotation_sha256'),('label_file','label_sha256')]:
            assert sha(data/row[f]) == row[h]
        original = load(data/row['annotation_file'])
        assert polygon_references(original,cfg['publisher_classes']) == row['references']
        assert (data/row['label_file']).read_text() == label_text(row['references'],row['width'],row['height'])
    for row in external:
        assert sha(source/row['file'])==row['sha256']
    out.mkdir(parents=True,exist_ok=False)
    (out/'code').mkdir()
    for p in [CONFIG,Path(__file__),ROOT/'scripts/prepare_substation_material.py',ROOT/'scripts/substation_material.sbatch']:
        shutil.copyfile(p,out/'code'/p.name)
    for split,rows in [('train',train),('development',dev)]:
        (out/f'{split}.txt').write_text('\n'.join(str(data/r['image_file']) for r in rows)+'\n')
    write(out/'dataset.yaml',{'path':str(data),'train':str(out/'train.txt'),'val':str(out/'development.txt'),'names':cfg['model_classes']})
    result={'status':'TRAINING','config':cfg,'protocol_sha256':sha(CONFIG),'manifest_sha256':manifest_sha,
            'started_at':datetime.now(timezone.utc).isoformat(), 'predictions':[], 'training_progress':{},
            'runtime':{'job_id':os.environ['SLURM_JOB_ID'],'partition':os.environ['SLURM_JOB_PARTITION'],
                       'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'ultralytics':ultralytics.__version__},
            'source_labels_recomputed':len(train)+len(dev),'training_images':len(train),'development_images':len(dev),
            'external_target_annotations_used_by_model':False,'is_uk_or_independent_asset_accuracy':False}
    write(out/'results.json',result); start=time.perf_counter()
    try:
        torch.set_num_threads(8)
        model=YOLOE(cfg['yaml']).load(str(ROOT/cfg['pretrained']))
        def on_start(trainer):
            assert set(trainer.train_loader.dataset.im_files)=={str(data/r['image_file']) for r in train}
            assert set(trainer.test_loader.dataset.im_files)=={str(data/r['image_file']) for r in dev}
            assert trainer.model.names==dict(enumerate(cfg['model_classes']))
            result['actual_training_args']=vars(trainer.args);write(out/'results.json',result)
        def on_batch(trainer):
            if trainer.loss is not None and not torch.isfinite(trainer.loss).all():
                raise ValueError('Non-finite training loss; stop without resubmission')
        def on_epoch(trainer):
            result['training_progress']={'epochs_completed':trainer.epoch+1,'losses':trainer.loss_items.detach().cpu().tolist(),
                                         'seconds':time.perf_counter()-start}
            write(out/'results.json',result)
            print(result['training_progress'],flush=True)
        model.add_callback('on_train_start',on_start);model.add_callback('on_train_batch_end',on_batch);model.add_callback('on_fit_epoch_end',on_epoch)
        opt=cfg['optimizer']
        model.train(data=str(out/'dataset.yaml'),trainer=YOLOEPETrainer,epochs=cfg['epochs'],imgsz=cfg['imgsz'],batch=cfg['batch'],nbs=cfg['batch'],
                    optimizer=opt['name'],lr0=opt['lr0'],lrf=opt['lrf'],weight_decay=opt['weight_decay'],warmup_epochs=1.,
                    seed=17,workers=8,amp=False,freeze=0,cache=False,mosaic=.5,close_mosaic=5,mixup=0.,copy_paste=0.,
                    translate=.05,scale=.15,fliplr=.5,hsv_h=0.,hsv_s=.1,hsv_v=.2,device=0,deterministic=True,
                    compile=False,project=str(out),name='training',exist_ok=False,pretrained=True,patience=cfg['epochs']+1,
                    plots=False,verbose=False,save=True,save_period=-1,cos_lr=False,val=False,degrees=0.,shear=0.,perspective=0.,flipud=0.)
        assert result['training_progress']['epochs_completed']==cfg['epochs']
        checkpoint=Path(model.trainer.last)
        result.update(status='INFERENCE',checkpoint=str(checkpoint.relative_to(out)),checkpoint_sha256=sha(checkpoint),training_seconds=time.perf_counter()-start)
        write(out/'results.json',result)
        model=YOLOE(str(checkpoint)).to('cuda:0')
        targets=[dict(r,inference_file=str(data/r['image_file']),expected_sha=r['image_sha256']) for r in dev]
        targets += [dict(r,inference_file=str(source/r['file']),expected_sha=r['sha256']) for r in external]
        for row in targets:
            t=time.perf_counter()
            pred=model.predict(row['inference_file'],device=0,verbose=False,save=False,half=False,**cfg['inference'])[0]
            predictions=[{'class_id':int(c),'class_name':cfg['classes'][int(c)],'score':float(s),'box':list(map(float,b)),
                          'material_verified':False,'material_probability_calibrated':False}
                         for b,s,c in zip(pred.boxes.xyxy.cpu().tolist(),pred.boxes.conf.cpu().tolist(),pred.boxes.cls.cpu().tolist())]
            f=f'predictions/{row["split"]}/{row["id"]}.json'
            write(out/f,{'image_id':row['id'],'image_sha256':row['expected_sha'],'split':row['split'],'predictions':predictions,
                         'checkpoint_sha256':result['checkpoint_sha256'],'protocol_sha256':sha(CONFIG),'seconds':time.perf_counter()-t,
                         'output_stage':'All library post-NMS boxes above fixed .05 floor; no pre-NMS tensors retained. Detector scores are not calibrated material probabilities.'})
            result['predictions'].append({'image_id':row['id'],'split':row['split'],'file':f,'sha256':sha(out/f),'count':len(predictions)})
            write(out/'results.json',result)
        result.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-start);write(out/'results.json',result)
        print({'status':'COMPLETE','seconds':result['elapsed_seconds'],'predictions':len(result['predictions'])},flush=True)
    except BaseException as e:
        result.update(status='FAILED',error=str(e),traceback=traceback.format_exc());write(out/'results.json',result);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest-sha',required=True)
    main(parser.parse_args().manifest_sha)
