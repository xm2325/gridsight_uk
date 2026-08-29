"""Train and evaluate one frozen four-class material head on Roihu gputest."""
from __future__ import annotations

import hashlib,json,os,shutil,time,traceback
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

from material_head_v2_common import decide_v2,diagnostic_counts,margin

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def rank(seed,value):return hashlib.sha256(f'{seed}|{value}'.encode()).hexdigest()
def extent(box,width,height,padding):
    x0,y0,x1,y1=box;dx=(x1-x0)*padding;dy=(y1-y0)*padding
    return [max(0,int(x0-dx)),max(0,int(y0-dy)),min(width,int(x1+dx+.999999)),min(height,int(y1+dy+.999999))]
def yolo_boxes(path,width,height):
    boxes=[]
    for line in Path(path).read_text().splitlines():
        cls,x,y,w,h=map(float,line.split());x*=width;y*=height;w*=width;h*=height
        boxes.append((int(cls),[x-w/2,y-h/2,x+w/2,y+h/2]))
    return boxes

def main(config_name):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='gputest':
        raise RuntimeError('Requires a Roihu gputest allocation; no local fallback')
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModel,AutoProcessor
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    protocol=ROOT/config_name;cfg=load(protocol);out=ROOT/cfg['run'];mpid=ROOT/cfg['mpid_dataset']
    dependencies=[(mpid/'manifest.json',cfg['mpid_manifest_sha256']),(ROOT/cfg['substation_features'],cfg['substation_features_sha256']),
      (ROOT/cfg['substation_index'],cfg['substation_index_sha256']),(ROOT/cfg['uk_targets'],cfg['uk_targets_sha256']),
      (ROOT/cfg['encoder']/'model.safetensors',cfg['encoder_sha256'])]
    for path,expected in dependencies:
        if sha(path)!=expected:raise ValueError(f'Hash mismatch: {path}')
    targets=load(ROOT/cfg['uk_targets']);assert not targets['training_use'] and not targets['model_v2_inference_performed_before_freeze']
    for row in targets['images']:
        if sha(ROOT/row['image_file'])!=row['image_sha256']:raise ValueError(f"UK image differs: {row['record_id']}")
    out.mkdir(parents=True,exist_ok=False);(out/'code').mkdir()
    for path in [protocol,Path(__file__),ROOT/'scripts/material_head_v2_common.py',ROOT/'scripts/material_head_v2.sbatch']:
        shutil.copy2(path,out/'code'/path.name)
    result={'status':'PREPARING','started_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(protocol),
      'runtime':{'job_id':os.environ['SLURM_JOB_ID'],'device':torch.cuda.get_device_name(),'torch':torch.__version__,'transformers':transformers.__version__},
      'encoder_gradient_steps':0,'head_gradient_steps':0,'training_losses':[],'claim_boundary':cfg['claim_boundary']}
    write(out/'results.json',result);started=time.perf_counter()
    try:
        manifest=load(mpid/'manifest.json');names=cfg['classes'];seed=cfg['classifier']['seed'];selected=[];images=[]
        for split in ('train','development'):
            limit=cfg['mpid_crop_limits_per_class'][split]
            for class_id,material in enumerate(names[:3]):
                rows=sorted((r for r in manifest['rows'] if r['split']==split and r['material']==material),key=lambda r:rank(seed,r['image_sha256']))
                count=0
                for row in rows:
                    image=Image.open(mpid/row['image_file']).convert('RGB');candidates=[box for cls,box in yolo_boxes(mpid/row['label_file'],image.width,image.height) if cls==class_id]
                    candidates=[b for b in candidates if min(b[2]-b[0],b[3]-b[1])>=cfg['minimum_native_side'] and (b[2]-b[0])*(b[3]-b[1])>=cfg['minimum_native_area']]
                    if not candidates:continue
                    box=max(candidates,key=lambda b:(b[2]-b[0])*(b[3]-b[1]));record={'id':f"mpid_{split}_{material}_{row['image_sha256'][:16]}",
                      'source':'MPID largest eligible box per image','split':split,'class_id':class_id,'expected_material':material,
                      'image_sha256':row['image_sha256'],'box':box,'views':{}}
                    for view,padding in [('tight',0),('context',cfg['context_padding'])]:
                        crop_box=extent(box,image.width,image.height,padding);record['views'][view]={'feature_index':len(images),'crop_xyxy':crop_box}
                        images.append(image.crop(crop_box))
                    selected.append(record);count+=1
                    if count==limit:break
                if count!=limit:raise RuntimeError(f'Insufficient eligible MPID {split} {material}: {count}/{limit}')
        write(out/'mpid_crop_manifest.json',selected)
        result['mpid_crop_counts']=dict(Counter(f"{r['split']}:{r['expected_material']}" for r in selected));result['status']='ENCODING_MPID';write(out/'results.json',result)
        weights=ROOT/cfg['encoder'];processor=AutoProcessor.from_pretrained(weights,local_files_only=True,trust_remote_code=False)
        encoder=AutoModel.from_pretrained(weights,local_files_only=True,trust_remote_code=False,use_safetensors=True).to('cuda').eval()
        for parameter in encoder.parameters():parameter.requires_grad_(False)
        def encode(batch_images):
            batch=processor(images=batch_images,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                value=encoder.get_image_features(**batch)
                if not isinstance(value,torch.Tensor):value=value.pooler_output
                return F.normalize(value.float(),dim=-1).cpu().numpy()
        chunks=[]
        for begin in range(0,len(images),cfg['batch_size']):
            chunks.append(encode(images[begin:begin+cfg['batch_size']]))
            if begin%480==0:print(json.dumps({'encoded':min(begin+cfg['batch_size'],len(images)),'total':len(images)}),flush=True)
        mpid_features=np.concatenate(chunks);np.savez_compressed(out/'mpid_features.npz',embeddings=mpid_features)
        old_features=np.load(ROOT/cfg['substation_features'])['embeddings'];old_index=load(ROOT/cfg['substation_index'])
        old_map={0:0,1:1,2:3};train_features=[];train_labels=[];dev_pairs=[]
        for record in selected:
            tight=mpid_features[record['views']['tight']['feature_index']];context=mpid_features[record['views']['context']['feature_index']]
            if record['split']=='train':
                train_features.extend([tight,context]);train_labels.extend([record['class_id']]*2)
            else:dev_pairs.append({'id':record['id'],'source':'MPID','expected_class':record['class_id'],'expected_material':record['expected_material'],
              'box':record['box'],'tight_feature':tight,'context_feature':context})
        old_by=defaultdict(dict)
        for index,row in enumerate(old_index):old_by[(row['crop_id'],row['split'],old_map[row['class_id']])][row['view']]=old_features[index]
        for (crop_id,split,class_id),views in old_by.items():
            if set(views)!={'tight','context'}:raise ValueError(f'Incomplete old feature pair: {crop_id}')
            if split=='train':
                train_features.extend([views['tight'],views['context']]);train_labels.extend([class_id]*2)
            else:dev_pairs.append({'id':'substation_'+crop_id,'source':'Substation15','expected_class':class_id,'expected_material':names[class_id],
              'box':[0,0,64,64],'tight_feature':views['tight'],'context_feature':views['context']})
        x=torch.tensor(np.stack(train_features),device='cuda');y=torch.tensor(train_labels,device='cuda');torch.manual_seed(seed)
        hidden=cfg['classifier']['hidden_dimensions'];head=torch.nn.Sequential(torch.nn.Linear(x.shape[1],hidden),torch.nn.GELU(),torch.nn.Linear(hidden,len(names))).to('cuda')
        counts=torch.bincount(y,minlength=len(names));class_weights=len(y)/(len(names)*counts.float())
        optimiser=torch.optim.AdamW(head.parameters(),lr=cfg['classifier']['learning_rate'],weight_decay=cfg['classifier']['weight_decay'])
        result.update(status='FITTING_HEAD',train_view_counts=counts.cpu().tolist(),development_pair_counts=dict(Counter(p['expected_material'] for p in dev_pairs)))
        write(out/'results.json',result)
        for step in range(cfg['classifier']['steps']):
            optimiser.zero_grad();loss=F.cross_entropy(head(x),y,weight=class_weights)
            if not torch.isfinite(loss):raise ValueError('Non-finite classifier loss')
            loss.backward();optimiser.step()
            if step%20==0 or step+1==cfg['classifier']['steps']:result['training_losses'].append({'step':step+1,'loss':float(loss.detach())})
        result['head_gradient_steps']=cfg['classifier']['steps']
        state={name:value.detach().cpu().numpy() for name,value in head.state_dict().items()}
        train_np=x.cpu().numpy();labels_np=y.cpu().numpy();centroids=[]
        for class_id in range(len(names)):
            centroid=train_np[labels_np==class_id].mean(0);centroid/=max(np.linalg.norm(centroid),1e-12);centroids.append(centroid)
        centroids=np.stack(centroids)
        np.savez_compressed(out/'head.npz',centroids=centroids,**state)
        def logits(features):
            with torch.inference_mode():return head(torch.tensor(np.stack(features),device='cuda')).cpu().numpy()
        all_dev_features=[f for p in dev_pairs for f in (p['tight_feature'],p['context_feature'])];dev_logits=logits(all_dev_features)
        dev_sim=np.stack(all_dev_features)@centroids.T
        correct_support=defaultdict(lambda:{'margin':[],'similarity':[]})
        for index,pair in enumerate(dev_pairs):
            tl,cl=dev_logits[2*index],dev_logits[2*index+1];ts,cs=dev_sim[2*index],dev_sim[2*index+1];expected=pair['expected_class']
            if int(tl.argmax())==expected and int(cl.argmax())==expected:
                correct_support[expected]['margin'].append(min(margin(tl),margin(cl)))
                correct_support[expected]['similarity'].append(min(ts[expected],cs[expected]))
        thresholds={'margin':[],'similarity':[]}
        for class_id in range(len(names)):
            if len(correct_support[class_id]['margin'])<10:raise RuntimeError(f'Insufficient correct development support: {names[class_id]}')
            thresholds['margin'].append(max(cfg['rejection']['minimum_logit_margin_floor'],float(np.quantile(correct_support[class_id]['margin'],cfg['rejection']['correct_development_margin_quantile']))))
            thresholds['similarity'].append(float(np.quantile(correct_support[class_id]['similarity'],cfg['rejection']['correct_development_similarity_quantile'])))
        dev_records=[]
        for index,pair in enumerate(dev_pairs):
            decision=decide_v2(dev_logits[2*index].tolist(),dev_logits[2*index+1].tolist(),dev_sim[2*index].tolist(),dev_sim[2*index+1].tolist(),pair['box'],cfg,thresholds)
            dev_records.append({k:v for k,v in pair.items() if not k.endswith('_feature')}|{'decision':decision})
        write(out/'development_decisions.json',dev_records);result.update(status='UK_INFERENCE',thresholds=thresholds,development_diagnostics=diagnostic_counts(dev_records),head_sha256=sha(out/'head.npz'))
        write(out/'results.json',result)
        uk_images=[];uk_inputs=[];uk_records=[]
        for row in targets['images']:
            image=Image.open(ROOT/row['image_file']).convert('RGB');image_record={k:v for k,v in row.items() if k!='boxes'};image_record['boxes']=[]
            for box in row['boxes']:
                material=box['material'];class_id=names.index(material);record={**box,'expected_material':material,'expected_class':class_id,'views':{}}
                for view,padding in [('tight',0),('context',cfg['context_padding'])]:
                    crop_box=extent(box['xyxy'],image.width,image.height,padding);record['views'][view]={'feature_index':len(uk_inputs),'crop_xyxy':crop_box};uk_inputs.append(image.crop(crop_box))
                image_record['boxes'].append(record);uk_records.append((image_record,record))
            uk_images.append(image_record)
        uk_features=np.concatenate([encode(uk_inputs[i:i+cfg['batch_size']]) for i in range(0,len(uk_inputs),cfg['batch_size'])]);uk_logits=logits(uk_features);uk_sim=uk_features@centroids.T
        np.savez_compressed(out/'uk_raw.npz',embeddings=uk_features,logits=uk_logits,similarities=uk_sim)
        flat=[]
        for image_record,record in uk_records:
            ti=record['views']['tight']['feature_index'];ci=record['views']['context']['feature_index']
            record['raw']={'tight_logits':uk_logits[ti].tolist(),'context_logits':uk_logits[ci].tolist(),
              'tight_similarity':uk_sim[ti].tolist(),'context_similarity':uk_sim[ci].tolist()}
            record['decision']=decide_v2(record['raw']['tight_logits'],record['raw']['context_logits'],record['raw']['tight_similarity'],record['raw']['context_similarity'],record['xyxy'],cfg,thresholds)
            flat.append({'record_id':image_record['record_id'],'id':record['id'],'expected_material':record['expected_material'],'decision':record['decision']})
        write(out/'uk_decisions.json',{'target_manifest_sha256':sha(ROOT/cfg['uk_targets']),'images':uk_images,
          'claim_boundary':targets['claim_boundary'],'raw_npz_sha256':sha(out/'uk_raw.npz')})
        result.update(status='COMPLETE',uk_diagnostics=diagnostic_counts(flat),uk_decisions_sha256=sha(out/'uk_decisions.json'),
          uk_raw_sha256=sha(out/'uk_raw.npz'),elapsed_seconds=time.perf_counter()-started)
        write(out/'results.json',result);print(json.dumps({'status':'COMPLETE','uk':result['uk_diagnostics'],'development':result['development_diagnostics']}),flush=True)
    except BaseException as error:
        result.update(status='FAILED',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started);write(out/'results.json',result);raise

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/material_head_v2.json');main(parser.parse_args().config)
