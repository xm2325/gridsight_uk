"""Three fixed leave-one-asset-out material adaptation folds on Roihu only."""
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
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')

def main(config_name):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='gputest':raise RuntimeError('Requires Roihu gputest; no local fallback')
    import numpy as np
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    protocol=ROOT/config_name;cfg=load(protocol);v2cfg=load(ROOT/cfg['v2_config']);v2=load(ROOT/cfg['v2_results'])
    pinned=[('v2_config','v2_config_sha256'),('v2_results','v2_results_sha256'),('v2_head','v2_head_sha256'),
      ('mpid_features','mpid_features_sha256'),('mpid_crop_manifest','mpid_crop_manifest_sha256'),
      ('substation_features','substation_features_sha256'),('substation_index','substation_index_sha256'),
      ('low_resolution_uk_decisions','low_resolution_uk_decisions_sha256'),('low_resolution_uk_raw','low_resolution_uk_raw_sha256'),
      ('high_resolution_decisions','high_resolution_decisions_sha256'),('high_resolution_raw','high_resolution_raw_sha256')]
    for path_key,hash_key in pinned:
        if sha(ROOT/cfg[path_key])!=cfg[hash_key]:raise ValueError(f'Hash mismatch: {cfg[path_key]}')
    out=ROOT/cfg['run'];out.mkdir(parents=True,exist_ok=False);(out/'code').mkdir()
    for path in [protocol,Path(__file__),ROOT/'scripts/material_head_v2_common.py',ROOT/'scripts/material_head_v2_loao.sbatch']:
        shutil.copy2(path,out/'code'/path.name)
    result={'status':'PREPARING','started_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(protocol),
      'runtime':{'job_id':os.environ['SLURM_JOB_ID'],'device':torch.cuda.get_device_name(),'torch':torch.__version__},
      'encoder_gradient_steps':0,'folds':[],'claim_boundary':cfg['claim_boundary']}
    write(out/'results.json',result);started=time.perf_counter()
    try:
        names=cfg['classes'];mpid_features=np.load(ROOT/cfg['mpid_features'])['embeddings'];mpid_manifest=load(ROOT/cfg['mpid_crop_manifest'])
        old_features=np.load(ROOT/cfg['substation_features'])['embeddings'];old_index=load(ROOT/cfg['substation_index']);old_map={0:0,1:1,2:3}
        base_train_x=[];base_train_y=[];dev_pairs=[]
        for record in mpid_manifest:
            tight=mpid_features[record['views']['tight']['feature_index']];context=mpid_features[record['views']['context']['feature_index']]
            if record['split']=='train':base_train_x.extend([tight,context]);base_train_y.extend([record['class_id']]*2)
            else:dev_pairs.append({'id':record['id'],'expected_class':record['class_id'],'expected_material':record['expected_material'],'box':record['box'],'tight':tight,'context':context})
        old_by=defaultdict(dict)
        for index,row in enumerate(old_index):old_by[(row['crop_id'],row['split'],old_map[row['class_id']])][row['view']]=old_features[index]
        for (crop_id,split,class_id),views in old_by.items():
            if split=='train':base_train_x.extend([views['tight'],views['context']]);base_train_y.extend([class_id]*2)
            else:dev_pairs.append({'id':'substation_'+crop_id,'expected_class':class_id,'expected_material':names[class_id],'box':[0,0,64,64],
              'tight':views['tight'],'context':views['context']})
        base_x=torch.tensor(np.stack(base_train_x),device='cuda');base_y=torch.tensor(base_train_y,device='cuda')
        base_counts=torch.bincount(base_y,minlength=len(names));base_weights=len(base_y)/(len(names)*base_counts.float())
        base_dev_features=np.stack([feature for pair in dev_pairs for feature in (pair['tight'],pair['context'])])
        centroids=np.load(ROOT/cfg['v2_head'])['centroids'];base_dev_similarity=base_dev_features@centroids.T
        low=load(ROOT/cfg['low_resolution_uk_decisions']);low_raw=np.load(ROOT/cfg['low_resolution_uk_raw'])['embeddings']
        high=load(ROOT/cfg['high_resolution_decisions']);high_raw=np.load(ROOT/cfg['high_resolution_raw'])['embeddings']
        uk_groups={}
        for image in low['images']:
            if image['record_id']=='uk_material_8090535':continue
            boxes=[]
            for box in image['boxes']:
                ti=box['views']['tight']['feature_index'];ci=box['views']['context']['feature_index']
                boxes.append({'id':box['id'],'record_id':image['record_id'],'asset_group':image['asset_group'],'expected_material':box['expected_material'],
                  'expected_class':box['expected_class'],'box':box['xyxy'],'tight':low_raw[ti],'context':low_raw[ci],'baseline_decision':box['decision'],'resolution':'source derivative'})
            uk_groups[image['record_id']]={'asset_group':image['asset_group'],'boxes':boxes}
        high_source=high['source'];boxes=[]
        for box in high['boxes']:
            ti=box['views']['tight']['feature_index'];ci=box['views']['context']['feature_index'];expected=names.index(box['material'])
            boxes.append({'id':box['id'],'record_id':high['record_id'],'asset_group':high_source['asset_group'],'expected_material':box['material'],
              'expected_class':expected,'box':box['xyxy'],'tight':high_raw[ti],'context':high_raw[ci],'baseline_decision':box['decision'],'resolution':'2560x1920 original'})
        uk_groups[high['record_id']]={'asset_group':high_source['asset_group'],'boxes':boxes}
        if set(uk_groups)!=set(f['test_record_id'] for f in cfg['folds']):raise ValueError('Fold records differ from frozen UK groups')
        saved=np.load(ROOT/cfg['v2_head']);all_heldout=[]
        for fold_index,fold_cfg in enumerate(cfg['folds']):
            test_id=fold_cfg['test_record_id'];train_ids=sorted(set(uk_groups)-{test_id})
            if uk_groups[test_id]['asset_group']!=fold_cfg['test_asset_group']:raise ValueError('Test asset group differs')
            train_boxes=[box for rid in train_ids for box in uk_groups[rid]['boxes']];test_boxes=uk_groups[test_id]['boxes']
            if any(box['asset_group']==fold_cfg['test_asset_group'] for box in train_boxes):raise ValueError('Held-out asset leaked into adaptation')
            uk_x=torch.tensor(np.stack([feature for box in train_boxes for feature in (box['tight'],box['context'])]),device='cuda')
            uk_y=torch.tensor([box['expected_class'] for box in train_boxes for _ in (0,1)],device='cuda')
            torch.manual_seed(cfg['adaptation']['seed']+fold_index)
            head=torch.nn.Sequential(torch.nn.Linear(768,256),torch.nn.GELU(),torch.nn.Linear(256,4)).to('cuda')
            head.load_state_dict({name:torch.tensor(saved[name],device='cuda') for name in ('0.weight','0.bias','2.weight','2.bias')})
            for parameter in head[0].parameters():parameter.requires_grad_(False)
            optimiser=torch.optim.AdamW(head[2].parameters(),lr=cfg['adaptation']['learning_rate'],weight_decay=cfg['adaptation']['weight_decay'])
            losses=[]
            for step in range(cfg['adaptation']['steps']):
                optimiser.zero_grad();base_loss=F.cross_entropy(head(base_x),base_y,weight=base_weights);uk_loss=F.cross_entropy(head(uk_x),uk_y)
                loss=cfg['adaptation']['base_loss_weight']*base_loss+cfg['adaptation']['uk_loss_weight']*uk_loss
                if not torch.isfinite(loss):raise ValueError('Non-finite adaptation loss')
                loss.backward();optimiser.step()
                if step in (0,cfg['adaptation']['steps']-1):losses.append({'step':step+1,'total':float(loss.detach()),'base':float(base_loss.detach()),'uk':float(uk_loss.detach())})
            with torch.inference_mode():base_dev_logits=head(torch.tensor(base_dev_features,device='cuda')).cpu().numpy()
            support=defaultdict(lambda:{'margin':[],'similarity':[]})
            for index,pair in enumerate(dev_pairs):
                tl,cl=base_dev_logits[2*index],base_dev_logits[2*index+1];ts,cs=base_dev_similarity[2*index],base_dev_similarity[2*index+1];expected=pair['expected_class']
                if int(tl.argmax())==expected and int(cl.argmax())==expected:
                    support[expected]['margin'].append(min(margin(tl),margin(cl)));support[expected]['similarity'].append(min(ts[expected],cs[expected]))
            thresholds={'margin':[],'similarity':[]}
            for class_id in range(len(names)):
                if len(support[class_id]['margin'])<10:raise RuntimeError(f'Insufficient fold development support: {names[class_id]}')
                thresholds['margin'].append(max(v2cfg['rejection']['minimum_logit_margin_floor'],float(np.quantile(support[class_id]['margin'],v2cfg['rejection']['correct_development_margin_quantile']))))
                thresholds['similarity'].append(float(np.quantile(support[class_id]['similarity'],v2cfg['rejection']['correct_development_similarity_quantile'])))
            test_features=np.stack([feature for box in test_boxes for feature in (box['tight'],box['context'])])
            with torch.inference_mode():test_logits=head(torch.tensor(test_features,device='cuda')).cpu().numpy()
            test_similarity=test_features@centroids.T;predictions=[]
            for index,box in enumerate(test_boxes):
                decision=decide_v2(test_logits[2*index].tolist(),test_logits[2*index+1].tolist(),test_similarity[2*index].tolist(),test_similarity[2*index+1].tolist(),box['box'],v2cfg,thresholds)
                record={k:v for k,v in box.items() if k not in ('tight','context')}|{'decision':decision,
                  'raw':{'tight_logits':test_logits[2*index].tolist(),'context_logits':test_logits[2*index+1].tolist(),
                    'tight_similarity':test_similarity[2*index].tolist(),'context_similarity':test_similarity[2*index+1].tolist()}}
                predictions.append(record);all_heldout.append({'expected_material':box['expected_material'],'decision':decision})
            fold={'fold_index':fold_index,'test_record_id':test_id,'test_asset_group':fold_cfg['test_asset_group'],'train_record_ids':train_ids,
              'train_asset_groups':[uk_groups[rid]['asset_group'] for rid in train_ids],'uk_train_box_count':len(train_boxes),'test_box_count':len(test_boxes),
              'adaptation_losses':losses,'thresholds':thresholds,'predictions':predictions,
              'diagnostics':diagnostic_counts([{'expected_material':p['expected_material'],'decision':p['decision']} for p in predictions])}
            np.savez_compressed(out/f'head_fold_{fold_index}.npz',**{name:value.detach().cpu().numpy() for name,value in head.state_dict().items()})
            fold['head_sha256']=sha(out/f'head_fold_{fold_index}.npz');result['folds'].append(fold);write(out/'results.json',result)
        result.update(status='COMPLETE',head_gradient_steps_per_fold=cfg['adaptation']['steps'],aggregate_diagnostics=diagnostic_counts(all_heldout),
          elapsed_seconds=time.perf_counter()-started);write(out/'results.json',result)
        print(json.dumps({'status':'COMPLETE','aggregate':result['aggregate_diagnostics'],'folds':[f['diagnostics'] for f in result['folds']]}),flush=True)
    except BaseException as error:
        result.update(status='FAILED',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started);write(out/'results.json',result);raise

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/material_head_v2_loao.json');main(parser.parse_args().config)
