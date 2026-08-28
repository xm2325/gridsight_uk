"""One original-polygon segmentation run on gputest; no held-out evaluation inputs."""
import argparse
import gc
import os
import shutil
import time
import traceback
from pathlib import Path
from paper_material_demo import ROOT,load,sha,write
from prepare_component_masks import CONFIG,segmentation_line


def main(manifest_sha):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='gputest':
        raise RuntimeError('Requires Roihu gputest; no local model fallback')
    cfg=load(CONFIG);data=ROOT/cfg['dataset'];out=ROOT/cfg['run'];ukroot=ROOT/cfg['uk_dataset']
    assert sha(data/'manifest.json')==manifest_sha
    m=load(data/'manifest.json');assert m['protocol_sha256']==sha(CONFIG)
    assert sha(ROOT/cfg['checkpoint'])==cfg['checkpoint_sha256']
    assert sha(ukroot/'manifest.json')==cfg['uk_manifest_sha256']
    uk=load(ukroot/'manifest.json')['images']
    assert len(uk)==27 and all(r['ground_truth_status']=='NONE' for r in uk)
    for r in m['images']:
        assert r['split'] in ['train','dev'] and sha(data/r['image_file'])==r['sha256']
        lp=data/r['label_file'];assert sha(lp)==r['label_sha256']
        assert lp.read_text().splitlines()==[segmentation_line(a,r['width'],r['height']) for a in r['references']]
    from insplad_adapt_common import start_runtime
    runtime=start_runtime()
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer
    out.mkdir(parents=True,exist_ok=False);(out/'code').mkdir()
    for p in [CONFIG,Path(__file__),ROOT/'scripts/prepare_component_masks.py',ROOT/'scripts/component_masks.sbatch']:
        shutil.copyfile(p,out/'code'/p.name)
    result={'status':'TRAINING','runtime':runtime,'config':cfg,'protocol_sha256':sha(CONFIG),
            'manifest_sha256':manifest_sha,'source_label_files_verified':len(m['images']),
            'epoch_losses':[],'predictions':[],'evaluation_images_used':False,'uk_training_images':0}
    write(out/'results.json',result);start=time.perf_counter()
    try:
        model=YOLOE(str(ROOT/cfg['checkpoint']))
        def on_start(trainer):
            expected={str(data/r['image_file']) for r in m['images'] if r['split']=='train'}
            expected_dev={str(data/r['image_file']) for r in m['images'] if r['split']=='dev'}
            assert set(trainer.train_loader.dataset.im_files)==expected
            assert set(trainer.test_loader.dataset.im_files)==expected_dev
            assert trainer.model.names==dict(enumerate(cfg['classes'])) and trainer.args.task=='segment'
            result['training_setup']={'training_images':sorted(expected),'development_images':sorted(expected_dev),
                 'args':vars(trainer.args),'trainable_parameters':sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)}
            write(out/'results.json',result)
        def on_batch(trainer):
            if trainer.loss is not None and not torch.isfinite(trainer.loss).all():raise ValueError('Non-finite loss')
        def on_epoch(trainer):
            event={'epoch':trainer.epoch+1,'losses':trainer.loss_items.detach().cpu().tolist()}
            if result['epoch_losses'] and result['epoch_losses'][-1]['epoch']==event['epoch']:
                assert event['epoch']==cfg['training']['epochs']
                result.setdefault('post_training_callbacks',[]).append(event)
            else:
                result['epoch_losses'].append(event)
            write(out/'results.json',result)
            print({'epoch':trainer.epoch+1,'losses':result['epoch_losses'][-1]['losses']},flush=True)
        model.add_callback('on_train_start',on_start);model.add_callback('on_train_batch_end',on_batch);model.add_callback('on_fit_epoch_end',on_epoch)
        model.train(data=str(data/'train.yaml'),trainer=YOLOEPESegTrainer,**cfg['training'],
                    device=0,deterministic=True,compile=False,project=str(out),name='training',exist_ok=False,
                    pretrained=True,patience=21,plots=False,verbose=False,save=True,save_period=-1,
                    cos_lr=False,close_mosaic=0,val=True,degrees=0.,shear=0.,perspective=0.,flipud=0.)
        assert [r['epoch'] for r in result['epoch_losses']]==list(range(1,21))
        checkpoint=Path(model.trainer.last);assert checkpoint.is_file()
        result.update(status='PREDICTING',training_seconds=time.perf_counter()-start,checkpoint_sha256=sha(checkpoint),
                      selected_checkpoint=str(checkpoint.relative_to(out)))
        write(out/'results.json',result);del model;gc.collect();torch.cuda.empty_cache()
        model=YOLOE(str(checkpoint));assert model.names==dict(enumerate(cfg['classes']))
        rows=[('development',r,data) for r in m['images'] if r['split']=='dev']+[('uk_qualitative',r,ukroot) for r in uk]
        inf=cfg['inference']
        for index,(split,r,source) in enumerate(rows):
            ip=source/r['image_file'];assert sha(ip)==r['sha256']
            with Image.open(ip) as im:
                im=im.convert('RGB');assert im.size==(r['width'],r['height'])
                scale=min(1.,inf['long_side']/max(im.size));size=(round(im.width*scale),round(im.height*scale))
                working=im.resize(size,Image.Resampling.LANCZOS);working.info.clear()
            folder=out/'predictions'/r['image_id'];folder.mkdir(parents=True)
            working.save(folder/'input.png');torch.cuda.synchronize();t=time.perf_counter()
            pred=model.predict(working,imgsz=inf['long_side'],conf=inf['confidence_floor'],iou=inf['nms_iou'],
                               max_det=inf['max_det'],retina_masks=True,device=0,half=False,verbose=False)[0]
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
            boxes=pred.boxes.xyxy.cpu().numpy();scores=pred.boxes.conf.cpu().numpy();classes=pred.boxes.cls.cpu().numpy().astype(np.int64)
            masks=pred.masks.data.cpu().numpy().astype(np.uint8) if pred.masks is not None else np.zeros((0,size[1],size[0]),dtype=np.uint8)
            assert masks.shape==(len(boxes),size[1],size[0]) and np.isin(masks,[0,1]).all()
            raw=folder/'postprocessed_masks.npz'
            np.savez_compressed(raw,boxes=boxes,scores=scores,classes=classes,mask_shape=np.array(masks.shape),
                                mask_bits=np.packbits(masks.reshape((len(boxes),size[0]*size[1])),axis=1))
            predictions=[]
            for j,(b,s,c,mask) in enumerate(zip(boxes,scores,classes,masks)):
                sx,sy=r['width']/size[0],r['height']/size[1]
                predictions.append({'prediction_index':j,'class_id':int(c),'score':float(s),
                   'box_working':b.tolist(),'box':[float(b[0])*sx,float(b[1])*sy,float(b[2])*sx,float(b[3])*sy],
                   'mask_pixels':int(mask.sum()),'material':'unknown','material_verified':False})
            payload={'image_id':r['image_id'],'source_image_sha256':r['sha256'],'split':split,
                     'source_size':[r['width'],r['height']],'working_size':list(size),'input_sha256':sha(folder/'input.png'),
                     'raw_file':'postprocessed_masks.npz','raw_sha256':sha(raw),'predictions':predictions,
                     'outputs_are_postprocessed':True,'source_references_used_for_inference':False,
                     'elapsed_seconds':elapsed,'pole_top_supervised':False,'steel_material_supervised':False}
            write(folder/'predictions.json',payload)
            result['predictions'].append({'image_id':r['image_id'],'split':split,'file':str((folder/'predictions.json').relative_to(out)),
                                         'sha256':sha(folder/'predictions.json')})
            write(out/'results.json',result)
            if (index+1)%20==0 or index+1==len(rows):print({'predicted':index+1,'total':len(rows)},flush=True)
        assert len(result['predictions'])==107
        result.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-start);write(out/'results.json',result)
    except BaseException as e:
        result.update(status='FAILED',error=str(e),traceback=traceback.format_exc());write(out/'results.json',result);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest-sha',required=True);main(p.parse_args().manifest_sha)
