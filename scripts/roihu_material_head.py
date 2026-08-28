"""Frozen SigLIP2 + one bounded supervised linear head, on gputest only."""
import argparse
import hashlib
import io
import os
import shutil
import time
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path

from paper_material_demo import ROOT, load, sha, write, extent
from prepare_material_head import CONFIG
from material_head_common import decide


def main(manifest_sha):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='gputest':
        raise RuntimeError('Requires a Roihu gputest allocation; no local fallback')
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModel,AutoProcessor
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    cfg=load(CONFIG);data=ROOT/cfg['dataset'];out=ROOT/cfg['run'];source=ROOT/cfg['external_demo']
    assert sha(data/'manifest.json')==manifest_sha
    manifest=load(data/'manifest.json');assert manifest['protocol_sha256']==sha(CONFIG)
    assert sha(source/'manifest.json')==cfg['external_manifest_sha256']
    assert load(source/'protocol.json')['siglip_sha256']==cfg['encoder_sha256']
    weights=ROOT/cfg['encoder'];assert sha(weights/'model.safetensors')==cfg['encoder_sha256']
    for f in load(weights/'verified_manifest.json')['files']:assert sha(weights/f['file'])==f['sha256']
    train_images={r['id'] for r in manifest['images'] if r['split']=='train'}
    dev_images={r['id'] for r in manifest['images'] if r['split']=='development'}
    assert not train_images&dev_images
    assert not {r['capture_group'] for r in manifest['images'] if r['split']=='train'}&{r['capture_group'] for r in manifest['images'] if r['split']=='development'}
    by_source=defaultdict(list)
    for c in manifest['crops']:by_source[c['image_id']].append(c)
    # Independent crop-to-original pixel comparison, not only self-reported hashes.
    verified_pixels=0
    with zipfile.ZipFile(ROOT/'data/external/substation15_cache/substation-semantic-dataset.zip') as archive:
        for r in manifest['images']:
            raw=archive.read(r['archive_image']);assert hashlib.sha256(raw).hexdigest()==r['image_sha256']
            im=Image.open(io.BytesIO(raw)).convert('RGB')
            for c in by_source[r['id']]:
                for view,d in c['views'].items():
                    assert sha(data/d['file'])==d['sha256']
                    actual=Image.open(data/d['file']).convert('RGB'); expected=im.crop(d['box'])
                    assert actual.size==expected.size and actual.tobytes()==expected.tobytes()
                    verified_pixels+=1
    out.mkdir(parents=True,exist_ok=False);(out/'code').mkdir()
    for p in [CONFIG,Path(__file__),ROOT/'scripts/prepare_material_head.py',ROOT/'scripts/material_head_common.py',ROOT/'scripts/material_head.sbatch']:
        shutil.copyfile(p,out/'code'/p.name)
    result={'status':'ENCODING','config':cfg,'protocol_sha256':sha(CONFIG),'manifest_sha256':manifest_sha,
            'source_to_crop_pixel_checks':verified_pixels,'encoder_gradient_steps':0,'training_losses':[],
            'runtime':{'job_id':os.environ['SLURM_JOB_ID'],'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'transformers':transformers.__version__},
            'external_automatic':[],'external_oracle':[]}
    start=time.perf_counter();write(out/'results.json',result)
    try:
        torch.set_num_threads(8);torch.manual_seed(cfg['classifier']['seed'])
        processor=AutoProcessor.from_pretrained(weights,local_files_only=True,trust_remote_code=False)
        encoder=AutoModel.from_pretrained(weights,local_files_only=True,trust_remote_code=False,use_safetensors=True).to('cuda').eval()
        for parameter in encoder.parameters():parameter.requires_grad_(False)
        def encode(images):
            batch=processor(images=images,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                v=encoder.get_image_features(**batch)
                if not isinstance(v,torch.Tensor):v=v.pooler_output
                return F.normalize(v.float(),dim=-1).cpu().numpy()
        index=[];features=[];paths=[]
        for c in manifest['crops']:
            for view in ['tight','context']:
                paths.append(data/c['views'][view]['file']);index.append({'crop_id':c['id'],'image_id':c['image_id'],'split':c['split'],'class_id':c['class_id'],'view':view})
        for begin in range(0,len(paths),cfg['batch_size']):
            images=[Image.open(p).convert('RGB') for p in paths[begin:begin+cfg['batch_size']]]
            features.append(encode(images))
            if begin%256==0:print({'encoded':min(begin+cfg['batch_size'],len(paths)),'total':len(paths)},flush=True)
        features=np.concatenate(features);np.savez_compressed(out/'features.npz',embeddings=features)
        write(out/'feature_index.json',index)
        result['features_sha256']=sha(out/'features.npz');result['feature_index_sha256']=sha(out/'feature_index.json')
        train_idx=[i for i,r in enumerate(index) if r['split']=='train' and r['view']=='tight']
        x=torch.tensor(features[train_idx],device='cuda');y=torch.tensor([index[i]['class_id'] for i in train_idx],device='cuda')
        head=torch.nn.Linear(x.shape[1],3,device='cuda');opt=cfg['classifier']
        optimiser=torch.optim.AdamW(head.parameters(),lr=opt['learning_rate'],weight_decay=opt['weight_decay'])
        counts=torch.bincount(y,minlength=3);assert torch.all(counts>0)
        class_weights=len(y)/(3*counts.float())
        result.update(status='FITTING_HEAD',train_crop_counts=counts.cpu().tolist(),training_views='tight only; context views are rejection checks')
        write(out/'results.json',result)
        for step in range(opt['steps']):
            optimiser.zero_grad();loss=F.cross_entropy(head(x),y,weight=class_weights)
            if not torch.isfinite(loss):raise ValueError('Non-finite classifier loss')
            loss.backward();optimiser.step();result['training_losses'].append(float(loss.detach()))
        weight=head.weight.detach().cpu().numpy();bias=head.bias.detach().cpu().numpy()
        np.savez_compressed(out/'head.npz',weight=weight,bias=bias)
        logits=features@weight.T+bias;np.savez_compressed(out/'source_logits.npz',logits=logits)
        result.update(status='EXTERNAL_DIAGNOSTICS',head_sha256=sha(out/'head.npz'),source_logits_sha256=sha(out/'source_logits.npz'))
        write(out/'results.json',result)
        # Automatic branch only reuses the original detector's crops/features.
        targets=[r for r in load(source/'manifest.json')['images'] if r['role']=='target']
        for row in targets:
            mf=source/f'materials/grounding/{row["id"]}.json';m=load(mf)
            assert sha(source/m['raw_file'])==m['raw_sha256']
            pf=source/f'predictions/grounding/{row["id"]}.json';assert sha(pf)==m['prediction_sha256']
            old_features=np.load(source/m['raw_file'])['image_embeddings'];raw_logits=old_features@weight.T+bias
            decisions=[]
            for p in m['predictions']:
                ti=p['crops']['tight']['embedding_index'];ci=p['crops']['context']['embedding_index']
                d=decide(raw_logits[ti].tolist(),raw_logits[ci].tolist(),p['box'],cfg)
                decisions.append(dict(d,prediction_index=p['prediction_index'],box=p['box'],detector_score=p['detector_score'],
                                      tight_logits=raw_logits[ti].tolist(),context_logits=raw_logits[ci].tolist(),
                                      previous_reference_material=p['reference_material']))
            f=f'automatic/{row["id"]}.json'
            write(out/f,{'image_id':row['id'],'image_sha256':row['sha256'],'source_prediction_sha256':sha(pf),
                         'source_material_sha256':sha(mf),'source_embedding_sha256':m['raw_sha256'],'decisions':decisions})
            result['external_automatic'].append({'image_id':row['id'],'file':f,'sha256':sha(out/f)})
        # Separate oracle branch: source annotations are NEVER passed to the automatic branch or training.
        oracle=[];oracle_images=[]
        for row in targets:
            im=Image.open(source/row['file']).convert('RGB');assert sha(source/row['file'])==row['sha256']
            af=source/row['annotation_file'];assert sha(af)==row['annotation_sha256']
            for a in load(af)['annotations']:
                x0,y0,w,h=a['bbox'];box=[x0,y0,x0+w,y0+h];record={'id':f'{row["id"]}_{a["id"]}','image_id':row['id'],
                       'annotation_id':a['id'],'class_id':a['category_id']-1,'box':box,'view_indices':{}}
                for view,pad in [('tight',0),('context',cfg['context_padding'])]:
                    record['view_indices'][view]=len(oracle_images);oracle_images.append(im.crop(extent(box,im.width,im.height,pad)))
                oracle.append(record)
        oracle_features=np.concatenate([encode(oracle_images[i:i+cfg['batch_size']]) for i in range(0,len(oracle_images),cfg['batch_size'])])
        oracle_logits=oracle_features@weight.T+bias
        np.savez_compressed(out/'oracle_features.npz',embeddings=oracle_features,logits=oracle_logits)
        for r in oracle:
            ti,ci=r['view_indices']['tight'],r['view_indices']['context']
            r.update(decide(oracle_logits[ti].tolist(),oracle_logits[ci].tolist(),r['box'],cfg),tight_logits=oracle_logits[ti].tolist(),context_logits=oracle_logits[ci].tolist())
        write(out/'oracle.json',oracle)
        result.update(status='COMPLETE',oracle_sha256=sha(out/'oracle.json'),oracle_features_sha256=sha(out/'oracle_features.npz'),
                      elapsed_seconds=time.perf_counter()-start,head_gradient_steps=opt['steps'])
        write(out/'results.json',result);print({'status':'COMPLETE','elapsed_seconds':result['elapsed_seconds']},flush=True)
    except BaseException as e:
        result.update(status='FAILED',error=str(e),traceback=traceback.format_exc());write(out/'results.json',result);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest-sha',required=True);main(p.parse_args().manifest_sha)
