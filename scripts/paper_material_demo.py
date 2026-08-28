"""Small public-label material demo. Separate prepare/detect/material phases resume safely.

Target annotations are NEVER read by either automatic prediction phase.
The viewer may display them separately as publisher references, not predictions.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/paper_material_demo/v2_roihu_20260828'
CATEGORIES = ['glass', 'porcelain']
ZIP_SHA = '858d24ce426d28ad746d539cd2012a6ce8f390146a8cafdadf1691379875cbee'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n'); tmp.replace(path)


def load(path): return json.loads(Path(path).read_text())


def extent(box, width, height, padding=0):
    x1, y1, x2, y2 = box
    dx = (x2 - x1) * padding; dy = (y2 - y1) * padding
    return [max(0, math.floor(x1-dx)), max(0, math.floor(y1-dy)),
            min(width, math.ceil(x2+dx)), min(height, math.ceil(y2+dy))]


def material_decision(tight, context, box, min_margin=.02):
    """Demonstration heuristic only; does not claim calibrated material certainty."""
    if min(box[2]-box[0], box[3]-box[1]) < 16 or (box[2]-box[0])*(box[3]-box[1]) < 512:
        return 'unknown', 'insufficient native pixels'
    a = max(range(len(tight)), key=tight.__getitem__)
    b = max(range(len(context)), key=context.__getitem__)
    if a != b: return 'unknown', 'crop/context disagreement'
    if min(abs(tight[0]-tight[1]), abs(context[0]-context[1])) < min_margin:
        return 'unknown', 'small reference cosine margin'
    return CATEGORIES[a], 'provisional reference match; uncalibrated'


def grounding_protocol():
    return {'directory':'weights/grounding-dino-base-12bdfa3',
        'sha256':'5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21',
        'queries':['electrical insulator.','glass insulator.','porcelain insulator.'],
        'size':{'shortest_edge':800,'longest_edge':1333},'confidence_floor':.15,'nms_iou':.5,
        'material_source':'All queries produce generic insulator proposals. Material names from queries are NOT used as classifier labels.'}


def prepare():
    if (RUN/'manifest.json').exists():
        print('Prepared manifest already exists; validating, not extracting again.'); validate(); return
    if RUN.exists(): raise FileExistsError('Inspect incomplete preparation before continuing')
    source = ROOT/'data/external/uvinsdet_cache/UVInsDet_v1.0.0.zip'
    assert sha(source) == ZIP_SHA
    selection = load(ROOT/'runtime/paper_demo_selection/selection.json')
    assert len(selection) == 12
    RUN.mkdir(parents=True)
    (RUN/'code').mkdir(); shutil.copyfile(__file__, RUN/'code/preparation.py')
    arms = {
        'insplad': {'file':'runs/insplad_adaptation/adapt/20260827T155412415982Z/training/weights/best.pt', 'insulator_id':0},
        'epri': {'file':'runs/keen_components/epri_components_v1_20260827/training/weights/best.pt', 'insulator_id':2},
    }
    for arm in arms.values(): arm['sha256'] = sha(ROOT/arm['file'])
    grounding_cfg = grounding_protocol()
    assert sha(ROOT/grounding_cfg['directory']/'model.safetensors') == grounding_cfg['sha256']
    arms['grounding'] = {'type':'grounding_dino','insulator_id':0,'sha256':grounding_cfg['sha256']}
    protocol = {
        'created_at':datetime.now(timezone.utc).isoformat(), 'dataset':'UVInsDet v1.0.0',
        'source_url':'https://zenodo.org/records/18197601', 'doi':'10.5281/zenodo.18197601',
        'license':'CC BY 4.0', 'zip_sha256':ZIP_SHA, 'selection_sha256':sha(ROOT/'runtime/paper_demo_selection/selection.json'),
        'selection_rule':'Before inference: largest polygon area, pure-glass images, dHash diversity >=10 within split; 3 train and 6 test. All porcelain-containing images plus first unannotated test image.',
        'scope':'Small, correlated substation demonstration; NOT UK accuracy, NOT independent asset generalisation. Publisher test images are demonstration inputs, not an untouched future evaluation set.',
        'unit_warning':'Porcelain polygons are separate sheds/discs with null group IDs. No regrouped publisher target truth. Image_1 has no supplied annotations, not a verified negative.',
        'detectors':arms, 'grounding':grounding_cfg,
        'detector_settings':{'imgsz':1280,'conf':.05,'iou':.5,'max_det':20,'augment':False},
        'siglip_directory':'weights/siglip2-base-naflex-b53b807',
        'siglip_sha256':sha(ROOT/'weights/siglip2-base-naflex-b53b807/model.safetensors'),
        'material':{'context_padding':.12, 'minimum_native_side':16, 'minimum_native_area':512, 'reference_margin':.02,
            'reference_prototypes':'Mean normalised embeddings within each reference image and class, then equal image mean and normalise. Include an explicitly derived union of same-material reference polygons for multi-polygon images; never a publisher instance.',
            'reference_usage':'Only the four publisher-train images and their supplied labels enter prototypes. No target boxes or labels enter automatic inference. No gradient training.',
            'text_prompts':{
                'glass':['a close-up photo of a glass electrical insulator','a translucent green glass insulator string'],
                'porcelain':['a close-up photo of a porcelain electrical insulator','a glazed ceramic porcelain electrical insulator'],
                'polymer':['a close-up photo of a polymer electrical insulator','a silicone rubber composite electrical insulator'],
                'not_insulator':['a close-up photo of metal electrical equipment without insulators','a metal support, cable or background, not an insulator']},
            'scores':'Cosine similarities and detector scores, never material probabilities. Labels remain provisional even with consistent views.'},
        'gradient_steps':0, 'number_of_reference_images':4, 'number_of_target_images':8,
    }
    write(RUN/'protocol.json', protocol)
    rows=[]
    with zipfile.ZipFile(source) as z:
        for f in ['LICENSE','CITATION.cff','README.md','tools/labelme_to_coco.py']:
            p=RUN/'source'/f; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(z.read('UVInsDet/'+f))
        for r in selection:
            split=r['split']; name=r['image']['file_name']; key=split+'_'+Path(name).stem
            p=RUN/'images'/name; p.parent.mkdir(exist_ok=True)
            p.write_bytes(z.read(f'UVInsDet/data/images/{split}/{name}'))
            af=RUN/'annotations'/f'{key}.json'; write(af,r)
            lm=RUN/'annotations'/f'{key}_labelme.json'; lm.write_bytes(z.read(f'UVInsDet/data/annotations/labelme/{split}/{Path(name).stem}.json'))
            rows.append({'id':key,'role':'reference' if split=='train' else 'target','publisher_split':split,
                'file':str(p.relative_to(RUN)), 'sha256':sha(p), 'width':r['image']['width'],'height':r['image']['height'],
                'annotation_file':str(af.relative_to(RUN)), 'annotation_sha256':sha(af),
                'labelme_file':str(lm.relative_to(RUN)), 'labelme_sha256':sha(lm)})
    shutil.copyfile(ROOT/'runtime/target_sources/uvinsdet_zenodo.json', RUN/'source/zenodo_record.json')
    write(RUN/'manifest.json', {'protocol_sha256':sha(RUN/'protocol.json'),'images':rows})
    print(json.dumps({'prepared':len(rows),'manifest_sha256':sha(RUN/'manifest.json')}))


def validate():
    m=load(RUN/'manifest.json'); assert sha(RUN/'protocol.json')==m['protocol_sha256']
    for r in m['images']:
        for filekey, hashkey in [('file','sha256'),('annotation_file','annotation_sha256'),('labelme_file','labelme_sha256')]:
            assert sha(RUN/r[filekey])==r[hashkey], r['id']
    return load(RUN/'protocol.json'), m


def upgrade():
    """Freeze an additive remote arm, keeping the already completed baselines intact."""
    if RUN.exists():
        validate(); print('Remote protocol exists; no recopy.'); return
    source=ROOT/'runs/paper_material_demo/v1_20260828'
    m=load(source/'manifest.json');cfg=load(source/'protocol.json')
    assert sha(source/'protocol.json')==m['protocol_sha256']
    baseline=load(source/'detectors_complete.json')
    for f,h in baseline['outputs'].items(): assert sha(source/f)==h
    shutil.copytree(source,RUN)
    shutil.copyfile(source/'protocol.json',RUN/'baseline_protocol.json')
    cfg['created_at']=datetime.now(timezone.utc).isoformat()
    cfg['additive_experiment']='User requested Roihu gputest instead of local Apple GPU. Preserve finished local two-arm baseline. Add Grounding DINO proposal arm, run all material embeddings on Roihu. No target annotation input.'
    cfg['baseline_manifest_sha256']=sha(source/'manifest.json')
    cfg['grounding']=grounding_protocol()
    cfg['detectors']['grounding']={'type':'grounding_dino','insulator_id':0,'sha256':cfg['grounding']['sha256']}
    write(RUN/'protocol.json',cfg);m['protocol_sha256']=sha(RUN/'protocol.json');write(RUN/'manifest.json',m)
    shutil.copyfile(__file__,RUN/'code/remote_inference.py')
    print(json.dumps({'upgraded':str(RUN.relative_to(ROOT)), 'manifest_sha256':sha(RUN/'manifest.json')}))


def box_iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=w*h;union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.


def suppress(predictions,threshold):
    kept=[]
    for p in sorted(predictions,key=lambda p:p['score'],reverse=True):
        if all(box_iou(p['box'],q['box'])<=threshold for q in kept): kept.append(p)
    return kept


def grounding(device):
    cfg,m=validate();rt=runtime(device);g=cfg['grounding'];weights=ROOT/g['directory']
    assert sha(weights/'model.safetensors')==g['sha256']
    for f in load(weights/'verified_manifest.json')['files']: assert sha(weights/f['file'])==f['sha256']
    remaining=[r for r in m['images'] if r['role']=='target' and not (RUN/f'predictions/grounding/{r["id"]}.json').exists()]
    if not remaining: print('Grounding outputs complete; skipping inference',flush=True)
    else:
        import numpy as np
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
        rt['transformers']=transformers.__version__
        proc=AutoProcessor.from_pretrained(weights,local_files_only=True,trust_remote_code=False)
        model=AutoModelForZeroShotObjectDetection.from_pretrained(weights,local_files_only=True,trust_remote_code=False,
            use_safetensors=True,disable_custom_kernels=True).to(device).eval()
        for r in remaining:
            t=time.perf_counter();photo=Image.open(RUN/r['file']).convert('RGB');candidates=[];raw_files=[]
            for qi,query in enumerate(g['queries']):
                inputs=proc(images=photo,text=query,size=g['size'],return_tensors='pt').to(device)
                with torch.inference_mode():raw=model(**inputs)
                logits=raw.logits[0].float().cpu().numpy();boxes=raw.pred_boxes[0].float().cpu().numpy()
                scores=raw.logits[0].sigmoid().amax(-1).float().cpu().numpy()
                f=f'raw/grounding/{r["id"]}_q{qi}.npz';(RUN/f).parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(RUN/f,token_logits=logits,boxes_cxcywh=boxes,input_ids=inputs['input_ids'].cpu().numpy())
                raw_files.append({'file':f,'sha256':sha(RUN/f),'query':query})
                for idx,(b,s) in enumerate(zip(boxes,scores)):
                    if s<g['confidence_floor']:continue
                    cx,cy,w,h=map(float,b);box=[max(0.,(cx-w/2)*photo.width),max(0.,(cy-h/2)*photo.height),min(float(photo.width),(cx+w/2)*photo.width),min(float(photo.height),(cy+h/2)*photo.height)]
                    if box[0]>=box[2] or box[1]>=box[3]:continue
                    candidates.append({'box':box,'score':float(s),'class_id':0,'class_name':'insulator','query':query,'query_index':idx,'raw_file':f})
            kept=suppress(candidates,g['nms_iou'])
            write(RUN/f'predictions/grounding/{r["id"]}.json',{'image_id':r['id'],'image_sha256':r['sha256'],
                'arm':'grounding','model_sha256':g['sha256'],'protocol_sha256':m['protocol_sha256'],'runtime':rt,
                'seconds':time.perf_counter()-t,'raw_files':raw_files,'raw_predictions':candidates,'predictions':kept,'insulator_class_id':0})
            print(json.dumps({'grounding':r['id'],'kept':len(kept),'seconds':time.perf_counter()-t}),flush=True)
    write(RUN/'detectors_complete.json',{'completed_at':datetime.now(timezone.utc).isoformat(),'runtime':rt,
        'outputs':{str(p.relative_to(RUN)):sha(p) for p in sorted((RUN/'predictions').rglob('*.json'))}})


def runtime(device):
    if device != 'cuda' or not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'gputest':
        raise RuntimeError('Model execution requires a Roihu gputest CUDA allocation; local CPU/Apple inference is disabled.')
    import torch
    torch.set_num_threads(4)
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; run through gputest')
    return {'device':device,'torch':torch.__version__,'threads':torch.get_num_threads(),'started_at':datetime.now(timezone.utc).isoformat()}


def detect(device):
    cfg, m=validate(); rt=runtime(device)
    import ultralytics
    from ultralytics import YOLO
    rt['ultralytics']=ultralytics.__version__
    for arm,info in cfg['detectors'].items():
        if info.get('type')=='grounding_dino':continue
        assert sha(ROOT/info['file'])==info['sha256']
        remaining=[r for r in m['images'] if r['role']=='target' and not (RUN/f'predictions/{arm}/{r["id"]}.json').exists()]
        if not remaining: print(arm+' complete; skipped',flush=True); continue
        model=YOLO(ROOT/info['file'])
        print(json.dumps({'arm':arm,'names':model.names,'remaining':len(remaining)}),flush=True)
        for r in remaining:
            t=time.perf_counter()
            output=model.predict(str(RUN/r['file']),device=device,verbose=False,save=False,**cfg['detector_settings'])[0]
            predictions=[{'box':list(map(float,b)), 'score':float(s),'class_id':int(c),'class_name':model.names[int(c)]}
                for b,s,c in zip(output.boxes.xyxy.cpu().tolist(),output.boxes.conf.cpu().tolist(),output.boxes.cls.cpu().tolist())]
            payload={'image_id':r['id'],'image_sha256':r['sha256'],'arm':arm,'model_sha256':info['sha256'],
                'protocol_sha256':m['protocol_sha256'],'runtime':rt,'seconds':time.perf_counter()-t,
                'output_stage':'All post-NMS outputs at frozen floor; pre-NMS tensors not retained. No target annotations used.',
                'predictions':predictions,'insulator_class_id':info['insulator_id']}
            write(RUN/f'predictions/{arm}/{r["id"]}.json',payload)
            print(json.dumps({'image':r['id'],'arm':arm,'boxes':len(predictions),'seconds':payload['seconds']}),flush=True)
    write(RUN/'detectors_complete.json',{'completed_at':datetime.now(timezone.utc).isoformat(),'runtime':rt,
        'outputs':{str(p.relative_to(RUN)):sha(p) for p in sorted((RUN/'predictions').rglob('*.json'))}})


def materials(device):
    cfg,m=validate(); rt=runtime(device)
    if (RUN/'materials_complete.json').exists(): print('Materials already complete; skipped.'); return
    for f,h in load(RUN/'detectors_complete.json')['outputs'].items(): assert sha(RUN/f)==h
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    from PIL import Image
    from transformers import AutoModel,AutoProcessor
    weights=ROOT/cfg['siglip_directory']; assert sha(weights/'model.safetensors')==cfg['siglip_sha256']
    for f in load(weights/'verified_manifest.json')['files']: assert sha(weights/f['file'])==f['sha256']
    rt['transformers']=transformers.__version__
    proc=AutoProcessor.from_pretrained(weights,local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(weights,local_files_only=True,trust_remote_code=False,use_safetensors=True).to(device).eval()
    def pooled(v): return v if isinstance(v,torch.Tensor) else v.pooler_output
    def encode(crops):
        vectors=[]
        for i in range(0,len(crops),4):
            batch=proc(images=crops[i:i+4],return_tensors='pt').to(device)
            with torch.inference_mode(): v=F.normalize(pooled(model.get_image_features(**batch)).float(),dim=-1)
            vectors.append(v.cpu().numpy())
        return np.concatenate(vectors)
    texts=cfg['material']['text_prompts']; labels=list(texts)
    batch=proc(text=[p for k in labels for p in texts[k]],padding='max_length',return_tensors='pt').to(device)
    with torch.inference_mode(): tv=F.normalize(pooled(model.get_text_features(**batch)).float(),dim=-1)
    tp=F.normalize(tv.reshape(len(labels),2,-1).mean(1),dim=-1).cpu().numpy()
    refs=[]; ref_vectors=[]; group_vectors={c:[] for c in CATEGORIES}
    for r in m['images']:
        if r['role']!='reference': continue
        photo=Image.open(RUN/r['file']).convert('RGB')
        annotations=load(RUN/r['annotation_file'])['annotations']
        for ci,category in enumerate(CATEGORIES,1):
            boxes=[[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in annotations if a['category_id']==ci]
            if not boxes: continue
            descriptions=['publisher polygon bounding box']*len(boxes)
            if len(boxes)>1:
                boxes.append([min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)])
                descriptions.append('derived same-material union for reference support ONLY; not a publisher instance')
            crops=[]
            for j,(box,desc) in enumerate(zip(boxes,descriptions)):
                ex=extent(box,photo.width,photo.height);crop=photo.crop(ex);f=f'crops/reference/{r["id"]}_{category}_{j}.png'
                (RUN/f).parent.mkdir(parents=True,exist_ok=True);crop.save(RUN/f);crops.append(crop)
                refs.append({'image_id':r['id'],'material':category,'box':ex,'kind':desc,'file':f,'sha256':sha(RUN/f),'embedding_index':len(refs)})
            vec=encode(crops); ref_vectors.extend(vec);avg=vec.mean(0);group_vectors[category].append(avg/np.linalg.norm(avg))
            print(json.dumps({'reference':r['id'],'material':category,'crops':len(crops)}),flush=True)
    rp=np.stack([np.mean(group_vectors[c],0) for c in CATEGORIES]);rp/=np.linalg.norm(rp,axis=1,keepdims=True)
    np.savez_compressed(RUN/'prototypes.npz',reference_embeddings=np.stack(ref_vectors),reference_prototypes=rp,text_embeddings=tv.cpu().numpy(),text_prototypes=tp)
    write(RUN/'reference_support.json',{'references':refs,'prototype_sha256':sha(RUN/'prototypes.npz'),'labels':CATEGORIES,'text_labels':labels})
    for r in m['images']:
        if r['role']!='target': continue
        photo=Image.open(RUN/r['file']).convert('RGB')
        for arm in cfg['detectors']:
            dest=RUN/f'materials/{arm}/{r["id"]}.json'
            if dest.exists():
                assert load(dest)['prototype_sha256']==sha(RUN/'prototypes.npz'), 'Reference embedding checkpoint changed; inspect before resuming'
                print(f'{r["id"]}/{arm} material checkpoint exists; skipped',flush=True);continue
            t=time.perf_counter();predfile=RUN/f'predictions/{arm}/{r["id"]}.json';pred=load(predfile);descriptors=[];crops=[]
            for i,p in enumerate(pred['predictions']):
                if p['class_id']!=pred['insulator_class_id']: continue
                d={'prediction_index':i,'box':p['box'],'detector_score':p['score'],'crops':{}}
                for view,pad in [('tight',0),('context',cfg['material']['context_padding'])]:
                    ex=extent(p['box'],photo.width,photo.height,pad);crop=photo.crop(ex)
                    f=f'crops/{arm}/{r["id"]}_{i}_{view}.png';(RUN/f).parent.mkdir(parents=True,exist_ok=True);crop.save(RUN/f)
                    d['crops'][view]={'box':ex,'file':f,'sha256':sha(RUN/f),'embedding_index':len(crops)};crops.append(crop)
                descriptors.append(d)
            vec=encode(crops) if crops else np.zeros((0,rp.shape[1]),dtype=np.float32)
            rs=vec@rp.T;ts=vec@tp.T
            for d in descriptors:
                i=d['crops']['tight']['embedding_index'];j=d['crops']['context']['embedding_index']
                label,reason=material_decision(rs[i].tolist(),rs[j].tolist(),d['box'],cfg['material']['reference_margin'])
                d.update(reference_material=label,reference_reason=reason,material_verified=False,scores_are_probabilities=False,
                    reference_cosines={'tight':dict(zip(CATEGORIES,rs[i].tolist())),'context':dict(zip(CATEGORIES,rs[j].tolist()))},
                    text_cosines={'tight':dict(zip(labels,ts[i].tolist())),'context':dict(zip(labels,ts[j].tolist()))})
                a=int(ts[i].argmax());b=int(ts[j].argmax())
                d['text_material']=labels[a] if a==b else 'unknown'
            raw=f'raw/{arm}/{r["id"]}.npz';(RUN/raw).parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(RUN/raw,image_embeddings=vec,reference_cosines=rs,text_cosines=ts)
            write(dest,{'image_id':r['id'],'image_sha256':r['sha256'],'arm':arm,'prediction_sha256':sha(predfile),
                'prototype_sha256':sha(RUN/'prototypes.npz'),'protocol_sha256':m['protocol_sha256'],
                'runtime':rt,'seconds':time.perf_counter()-t,'raw_file':raw,'raw_sha256':sha(RUN/raw),'predictions':descriptors})
            print(json.dumps({'materials':r['id'],'arm':arm,'candidates':len(descriptors),'seconds':time.perf_counter()-t}),flush=True)
    write(RUN/'materials_complete.json',{'runtime':rt,'completed_at':datetime.now(timezone.utc).isoformat(),
        'outputs':{str(p.relative_to(RUN)):sha(p) for p in sorted((RUN/'materials').rglob('*.json'))},'gradient_steps':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','detect','material','upgrade','grounding']);p.add_argument('--device',choices=['cuda'],default='cuda')
    a=p.parse_args()
    if a.phase=='prepare': prepare()
    elif a.phase=='upgrade': upgrade()
    elif a.phase=='grounding': grounding(a.device)
    elif a.phase=='detect': detect(a.device)
    else: materials(a.device)
