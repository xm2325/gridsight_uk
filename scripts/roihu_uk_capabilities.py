"""One bounded inference diagnostic; no model training and no material ground truth."""
import gc,json,shutil,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from prepare_keen_components import ROOT,digest,write_json
from insplad_adapt_common import start_runtime
from roihu_demo_ablation import nms
from uk_capability_common import crop_extent,material_quality,diagnostic_decision


def main():
    protocol=ROOT/'configs/uk_capabilities_v3.json';cfg=json.loads(protocol.read_text())
    source=ROOT/cfg['source_run'];out=ROOT/cfg['output']
    if out.exists():raise FileExistsError('Inspect existing checkpoint; never duplicate completed inference')
    if digest(source/'results.json')!=cfg['source_results_sha256'] or digest(source/'report/data.json')!=cfg['source_bundle_sha256']:raise ValueError('Source predictions changed')
    for key,sha in [('grounding_model','grounding_sha256'),('siglip_model','siglip_sha256')]:
        directory=ROOT/cfg[key]
        if digest(directory/'model.safetensors')!=cfg[sha]:raise ValueError('Weight hash mismatch')
        for f in json.loads((directory/'verified_manifest.json').read_text())['files']:
            if digest(directory/f['file'])!=f['sha256']:raise ValueError('Release file changed')
    bundle=json.loads((source/'report/data.json').read_text());assert len(bundle['images'])==27
    runtime=start_runtime()
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    from PIL import Image
    from transformers import AutoProcessor,AutoModel,AutoModelForZeroShotObjectDetection
    torch.set_float32_matmul_precision('highest')
    out.mkdir(parents=True)
    for p in [protocol,Path(__file__),ROOT/'scripts/uk_capability_common.py',ROOT/'scripts/roihu_uk_capabilities.sbatch']:
        (out/'code').mkdir(exist_ok=True);shutil.copyfile(p,out/'code'/p.name)
    for key in ['grounding_model','siglip_model']:shutil.copyfile(ROOT/cfg[key]/'verified_manifest.json',out/(key+'_manifest.json'))
    frozen={'frozen_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':digest(protocol),
        'source_results_sha256':cfg['source_results_sha256'],'source_bundle_sha256':cfg['source_bundle_sha256'],
        'image_ids':[r['image_id'] for r in bundle['images']],'gradient_steps':0,'ground_truth_used':False,'manual_target_boxes':False}
    write_json(out/'frozen_choices.json',frozen)
    report={'status':'LOADING_MODELS','config':cfg,'runtime':{**runtime,'transformers':transformers.__version__},
        'frozen_choices_sha256':digest(out/'frozen_choices.json'),'records':[],'completed_images':0,
        'training_started':False,'performance_metrics':None,'started_at':frozen['frozen_at']}
    write_json(out/'results.json',report);started=time.perf_counter()
    try:
        proc=AutoProcessor.from_pretrained(ROOT/cfg['grounding_model'],local_files_only=True,trust_remote_code=False)
        model=AutoModelForZeroShotObjectDetection.from_pretrained(ROOT/cfg['grounding_model'],local_files_only=True,
            trust_remote_code=False,use_safetensors=True,disable_custom_kernels=True).to('cuda').eval()
        hardware={}
        for row in bundle['images']:
            key=row['image_id'];path=source/'report'/row['image_file'];assert digest(path)==row['sha256']
            with Image.open(path) as f:photo=f.convert('RGB')
            raw_predictions=[];raw_files=[]
            for qi,text in enumerate(cfg['steelwork_queries']):
                inputs=proc(images=photo,text=text,size=cfg['grounding_size'],return_tensors='pt').to('cuda')
                with torch.inference_mode():raw=model(**inputs)
                logits=raw.logits[0].float().cpu().numpy();boxes=raw.pred_boxes[0].float().cpu().numpy()
                scores=raw.logits[0].sigmoid().amax(-1).float().cpu().numpy()
                name=f'model_raw/{key}/hardware_q{qi}.npz';target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target,token_logits=logits,boxes_cxcywh=boxes,input_ids=inputs['input_ids'].cpu().numpy())
                raw_files.append({'file':name,'sha256':digest(target),'query':text})
                for i,(box,score) in enumerate(zip(boxes,scores)):
                    if score<cfg['grounding_confidence_floor']:continue
                    cx,cy,w,h=map(float,box);b=[max(0.,(cx-w/2)*photo.width),max(0.,(cy-h/2)*photo.height),min(float(photo.width),(cx+w/2)*photo.width),min(float(photo.height),(cy+h/2)*photo.height)]
                    if b[0]>=b[2] or b[1]>=b[3]:continue
                    raw_predictions.append({'class_id':3,'box':b,'score':float(score),'query':text,'query_index':i,'raw_file':name,'source':'grounding_dino_metal_hypothesis','region':0,'steel_composition_verified':False})
            hardware[key]={'image_id':key,'image_sha256':row['sha256'],'raw_predictions':raw_predictions,
                'predictions':nms(raw_predictions,cfg['nms_iou']),'status':'UNREVIEWED_STEELWORK_HYPOTHESES','raw_files':raw_files}
            write_json(out/f'hardware/{key}.json',hardware[key])
        del model,proc,inputs,raw;gc.collect();torch.cuda.empty_cache()
        proc=AutoProcessor.from_pretrained(ROOT/cfg['siglip_model'],local_files_only=True,trust_remote_code=False)
        model=AutoModel.from_pretrained(ROOT/cfg['siglip_model'],local_files_only=True,trust_remote_code=False,use_safetensors=True).to('cuda').eval()
        def pooled(v):
            if isinstance(v,torch.Tensor):return v
            if getattr(v,'pooler_output',None) is not None:return v.pooler_output
            raise ValueError('Unrecognised SigLIP2 pooled output')
        labels=list(cfg['material_prompts']);texts=[p for c in labels for p in cfg['material_prompts'][c]]
        inputs=proc(text=texts,padding='max_length',return_tensors='pt').to('cuda')
        with torch.inference_mode():text_vectors=F.normalize(pooled(model.get_text_features(**inputs)).float(),dim=-1)
        prototypes=F.normalize(text_vectors.reshape(len(labels),2,-1).mean(1),dim=-1)
        np.savez_compressed(out/'text_embeddings.npz',text_embeddings=text_vectors.cpu().numpy(),class_prototypes=prototypes.cpu().numpy())
        report['text_embeddings_sha256']=digest(out/'text_embeddings.npz')
        report['status']='MATERIAL_CROP_DIAGNOSTICS';write_json(out/'results.json',report)
        for number,row in enumerate(bundle['images'],1):
            key=row['image_id'];path=source/'report'/row['image_file']
            with Image.open(path) as f:photo=f.convert('RGB')
            descriptors=[];crops=[]
            for i,p in enumerate(row['predictions'][cfg['material_proposal_arm']]):
                if p['class_id']!=2 or p['score']<cfg['material_proposal_score']:continue
                d={'candidate_index':i,'source_arm':cfg['material_proposal_arm'],'box':p['box'],'detector_score':p['score'],
                   'material':'unknown','accepted':False,'scores_are_probabilities':False,'source_image_sha256':row['sha256']}
                if not material_quality(p['box'],cfg):d['reason']='insufficient_native_pixels'
                else:
                    d['crops']={}
                    for view,padding in [('tight',0.),('context',cfg['context_padding_fraction'])]:
                        extent=crop_extent(p['box'],photo.width,photo.height,padding)
                        crop=photo.crop(extent);name=f'crops/{key}/i{i}_{view}.png';target=out/name;target.parent.mkdir(parents=True,exist_ok=True);crop.save(target)
                        d['crops'][view]={'box':extent,'file':name,'sha256':digest(target),'embedding_index':len(crops)};crops.append(crop)
                descriptors.append(d)
            vectors=[]
            for start in range(0,len(crops),16):
                batch=proc(images=crops[start:start+16],return_tensors='pt').to('cuda')
                with torch.inference_mode():v=F.normalize(pooled(model.get_image_features(**batch)).float(),dim=-1)
                vectors.append(v.cpu().numpy())
            embeddings=np.concatenate(vectors) if vectors else np.zeros((0,prototypes.shape[1]),dtype=np.float32)
            scores=embeddings@prototypes.cpu().numpy().T
            for d in descriptors:
                if 'crops' not in d:continue
                tight=scores[d['crops']['tight']['embedding_index']].tolist();context=scores[d['crops']['context']['embedding_index']].tolist()
                d.update(diagnostic_decision(tight,context,labels));d['cosine_scores']={'tight':dict(zip(labels,tight)),'context':dict(zip(labels,context))}
            name=f'model_raw/{key}/material_embeddings.npz';np.savez_compressed(out/name,image_embeddings=embeddings,cosine_scores=scores)
            payload={'image_id':key,'image_sha256':row['sha256'],'labels':labels,'diagnostics':descriptors,
                'raw_embeddings':{'file':name,'sha256':digest(out/name)},'no_ground_truth':True,'no_calibrated_predictions':True}
            write_json(out/f'materials/{key}.json',payload)
            report['records'].append({'image_id':key,'hardware_file':f'hardware/{key}.json','hardware_sha256':digest(out/f'hardware/{key}.json'),
                'material_file':f'materials/{key}.json','material_sha256':digest(out/f'materials/{key}.json'),'candidate_crops':len(descriptors),'encoded_views':len(crops)})
            report['completed_images']=number;write_json(out/'results.json',report)
            print(json.dumps({'event':'CAPABILITY_IMAGE_COMPLETE','image_id':key,'candidates':len(descriptors),'encoded_views':len(crops)}),flush=True)
        report.update(status='COMPLETED_UNVALIDATED_CAPABILITY_DIAGNOSTICS',elapsed_seconds=time.perf_counter()-started)
        write_json(out/'results.json',report);print(json.dumps({'event':'CAPABILITIES_COMPLETE','seconds':report['elapsed_seconds']}),flush=True)
    except BaseException as exc:
        report.update(status='FAILED',error=str(exc),traceback=traceback.format_exc());write_json(out/'results.json',report);raise


if __name__=='__main__':main()
