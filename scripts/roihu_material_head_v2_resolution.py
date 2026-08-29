"""Run the fixed material-head v2 resolution intervention on Roihu only."""
import hashlib,json,os,shutil,time,traceback
from datetime import datetime,timezone
from pathlib import Path

from material_head_v2_common import decide_v2,diagnostic_counts

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
def extent(box,width,height,padding):
    x0,y0,x1,y1=box;dx=(x1-x0)*padding;dy=(y1-y0)*padding
    return [max(0,int(x0-dx)),max(0,int(y0-dy)),min(width,int(x1+dx+.999999)),min(height,int(y1+dy+.999999))]

def main(config_name):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='gputest':raise RuntimeError('Requires Roihu gputest; no local fallback')
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModel,AutoProcessor
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    protocol=ROOT/config_name;cfg=load(protocol);v2cfg=load(ROOT/cfg['v2_config']);v2=load(ROOT/cfg['v2_results']);intervention=load(ROOT/cfg['intervention'])
    dependencies=[(ROOT/cfg['intervention'],cfg['intervention_sha256']),(ROOT/cfg['v2_results'],cfg['v2_results_sha256']),
      (ROOT/cfg['v2_head'],cfg['v2_head_sha256']),(ROOT/cfg['low_resolution_decisions'],cfg['low_resolution_decisions_sha256']),
      (ROOT/cfg['encoder']/'model.safetensors',cfg['encoder_sha256']),
      (ROOT/intervention['high_resolution']['image_file'],intervention['high_resolution']['sha256']),
      (ROOT/intervention['high_resolution']['api_snapshot_file'],intervention['high_resolution']['api_snapshot_sha256'])]
    for path,expected in dependencies:
        if sha(path)!=expected:raise ValueError(f'Hash mismatch: {path}')
    if intervention['high_resolution_model_inference_performed_before_freeze'] or not intervention['head_and_thresholds_already_fixed']:
        raise ValueError('Intervention was not frozen before high-resolution inference')
    out=ROOT/cfg['run'];out.mkdir(parents=True,exist_ok=False);(out/'code').mkdir()
    for path in [protocol,Path(__file__),ROOT/'scripts/material_head_v2_common.py',ROOT/'scripts/material_head_v2_resolution.sbatch']:
        shutil.copy2(path,out/'code'/path.name)
    result={'status':'ENCODING','started_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(protocol),
      'runtime':{'job_id':os.environ['SLURM_JOB_ID'],'device':torch.cuda.get_device_name(),'torch':torch.__version__,'transformers':transformers.__version__},
      'head_gradient_steps':0,'encoder_gradient_steps':0,'claim_boundary':cfg['claim_boundary']}
    write(out/'results.json',result);started=time.perf_counter()
    try:
        image=Image.open(ROOT/intervention['high_resolution']['image_file']).convert('RGB')
        if list(image.size)!=[intervention['high_resolution']['width'],intervention['high_resolution']['height']]:raise ValueError('High-resolution dimensions differ')
        inputs=[];records=[]
        for box in intervention['boxes']:
            record={**box,'views':{}}
            for view,padding in [('tight',0),('context',cfg['context_padding'])]:
                crop_box=extent(box['xyxy'],image.width,image.height,padding);record['views'][view]={'feature_index':len(inputs),'crop_xyxy':crop_box};inputs.append(image.crop(crop_box))
            records.append(record)
        weights=ROOT/cfg['encoder'];processor=AutoProcessor.from_pretrained(weights,local_files_only=True,trust_remote_code=False)
        encoder=AutoModel.from_pretrained(weights,local_files_only=True,trust_remote_code=False,use_safetensors=True).to('cuda').eval()
        for parameter in encoder.parameters():parameter.requires_grad_(False)
        batch=processor(images=inputs,return_tensors='pt').to('cuda')
        with torch.inference_mode():
            features=encoder.get_image_features(**batch)
            if not isinstance(features,torch.Tensor):features=features.pooler_output
            features=F.normalize(features.float(),dim=-1)
        saved=np.load(ROOT/cfg['v2_head']);head=torch.nn.Sequential(torch.nn.Linear(768,256),torch.nn.GELU(),torch.nn.Linear(256,4)).to('cuda').eval()
        state={name:torch.tensor(saved[name],device='cuda') for name in ('0.weight','0.bias','2.weight','2.bias')};head.load_state_dict(state)
        with torch.inference_mode():logits=head(features).cpu().numpy()
        features=features.cpu().numpy();centroids=saved['centroids'];similarities=features@centroids.T
        np.savez_compressed(out/'raw.npz',embeddings=features,logits=logits,similarities=similarities)
        flat=[]
        for record in records:
            ti=record['views']['tight']['feature_index'];ci=record['views']['context']['feature_index']
            record['raw']={'tight_logits':logits[ti].tolist(),'context_logits':logits[ci].tolist(),
              'tight_similarity':similarities[ti].tolist(),'context_similarity':similarities[ci].tolist()}
            record['decision']=decide_v2(record['raw']['tight_logits'],record['raw']['context_logits'],record['raw']['tight_similarity'],record['raw']['context_similarity'],record['xyxy'],v2cfg,v2['thresholds'])
            flat.append({'expected_material':record['material'],'decision':record['decision']})
        low=load(ROOT/cfg['low_resolution_decisions']);low_image=next(row for row in low['images'] if row['record_id']==intervention['record_id'])
        low_by={box['id']:box for box in low_image['boxes']}
        comparisons=[]
        for record in records:
            previous=low_by[record['id']]
            comparisons.append({'id':record['id'],'expected_material':record['material'],'low_resolution_decision':previous['decision'],
              'high_resolution_decision':record['decision'],'low_resolution_xyxy':record['xyxy_low_resolution'],'high_resolution_xyxy':record['xyxy']})
        write(out/'decisions.json',{'record_id':intervention['record_id'],'source':intervention,'boxes':records,'comparisons':comparisons,
          'diagnostics':diagnostic_counts(flat),'raw_npz_sha256':sha(out/'raw.npz'),'claim_boundary':cfg['claim_boundary']})
        result.update(status='COMPLETE',diagnostics=diagnostic_counts(flat),decisions_sha256=sha(out/'decisions.json'),raw_sha256=sha(out/'raw.npz'),
          elapsed_seconds=time.perf_counter()-started);write(out/'results.json',result)
        print(json.dumps({'status':'COMPLETE','diagnostics':result['diagnostics'],'decisions':[r['decision']['material'] for r in records]}),flush=True)
    except BaseException as error:
        result.update(status='FAILED',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started);write(out/'results.json',result);raise

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/material_head_v2_resolution_8090535.json');main(parser.parse_args().config)
