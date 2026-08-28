"""Resume inference only after a verified completed training run; never call train."""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
from paper_material_demo import ROOT,load,sha,write
from prepare_component_masks import CONFIG


def training_audit(parent):
    failed=load(parent/'results.json')
    assert failed['status']=='FAILED' and failed['predictions']==[]
    events=failed['epoch_losses'];assert [e['epoch'] for e in events[:20]]==list(range(1,21))
    assert all(e['epoch']==20 for e in events[20:])
    assert all(math.isfinite(x) for e in events for x in e['losses'])
    with (parent/'training/results.csv').open() as f:
        rows=[{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(f)]
    assert [int(r['epoch']) for r in rows]==list(range(1,21))
    assert all(math.isfinite(float(v)) for r in rows for k,v in r.items() if 'loss' in k)
    return failed,rows


def submit():
    cfg=load(CONFIG);parent=ROOT/cfg['run'];training_audit(parent)
    record=ROOT/'runtime/component_masks_inference_submission.json'
    if record.exists() or (parent/'inference').exists():raise FileExistsError('Inspect existing inference receipt; do not resubmit')
    if subprocess.check_output(['squeue','--noheader','--name=gridsight-component-masks-infer','--format=%i'],text=True).strip():
        raise RuntimeError('Matching inference job exists')
    receipt={'status':'SUBMITTING','created_at':datetime.now(timezone.utc).isoformat(),
             'checkpoint_sha256':sha(parent/'training/weights/last.pt'),'training_results_sha256':sha(parent/'results.json'),
             'training_csv_sha256':sha(parent/'training/results.csv'),'runner_sha256':sha(Path(__file__)),
             'training_steps':0,'original_job_id':'921053'}
    with record.open('x') as f:json.dump(receipt,f,indent=2)
    r=subprocess.run(['sbatch','--parsable','scripts/component_masks_inference.sbatch',receipt['checkpoint_sha256']],cwd=ROOT,text=True,capture_output=True)
    receipt.update(status='SUBMITTED' if r.returncode==0 else 'FAILED_INSPECT_BEFORE_RETRY',stdout=r.stdout.strip(),stderr=r.stderr.strip(),returncode=r.returncode)
    if r.returncode==0:receipt['job_id']=r.stdout.strip().split(';')[0]
    write(record,receipt);print(json.dumps(receipt,indent=2));r.check_returncode()


def main(checkpoint_sha):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='gputest':
        raise RuntimeError('Requires Roihu gputest; no local model fallback')
    cfg=load(CONFIG);parent=ROOT/cfg['run'];out=parent/'inference';data=ROOT/cfg['dataset'];ukroot=ROOT/cfg['uk_dataset']
    failed,rows_csv=training_audit(parent);checkpoint=parent/'training/weights/last.pt'
    assert sha(checkpoint)==checkpoint_sha and failed['protocol_sha256']==sha(CONFIG)
    assert sha(data/'manifest.json')==failed['manifest_sha256']
    assert sha(ukroot/'manifest.json')==cfg['uk_manifest_sha256']
    m=load(data/'manifest.json');uk=load(ukroot/'manifest.json')['images']
    assert len(uk)==27 and all(r['ground_truth_status']=='NONE' for r in uk)
    from insplad_adapt_common import start_runtime
    runtime=start_runtime()
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLOE
    out.mkdir(exist_ok=False);(out/'code').mkdir()
    for p in [CONFIG,Path(__file__),ROOT/'scripts/component_masks_inference.sbatch']:
        shutil.copyfile(p,out/'code'/p.name)
    result={'status':'PREDICTING','runtime':runtime,'config':cfg,'protocol_sha256':sha(CONFIG),
            'manifest_sha256':failed['manifest_sha256'],'epoch_losses':failed['epoch_losses'][:20],
            'training_seconds':float(rows_csv[-1]['time']),'training_result_sha256':sha(parent/'results.json'),
            'training_csv_sha256':sha(parent/'training/results.csv'),'original_training_job':'921053',
            'extra_training_steps':0,'duplicate_final_callbacks':len(failed['epoch_losses'])-20,
            'selected_checkpoint':'../training/weights/last.pt','checkpoint_sha256':checkpoint_sha,'predictions':[]}
    write(out/'results.json',result);start=time.perf_counter()
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


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--submit',action='store_true');p.add_argument('--checkpoint-sha')
    a=p.parse_args()
    if a.submit:submit()
    elif a.checkpoint_sha:main(a.checkpoint_sha)
    else:p.error('Specify --submit or --checkpoint-sha')
