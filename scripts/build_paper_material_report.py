"""Verify recorded model outputs and build a separate, all-English public-data viewer."""
import json
import argparse
from collections import Counter
from pathlib import Path
import shutil
from paper_material_demo import ROOT, RUN, load, write, sha, validate, material_decision, suppress, box_iou, extent


def build(supervised=False):
    import numpy as np
    from PIL import Image
    cfg,m=validate()
    complete=load(RUN/'materials_complete.json')
    for marker in ['detectors_complete.json','materials_complete.json']:
        for f,h in load(RUN/marker)['outputs'].items(): assert sha(RUN/f)==h,(marker,f)
    z=np.load(RUN/'prototypes.npz');refs=load(RUN/'reference_support.json')
    assert sha(RUN/'prototypes.npz')==refs['prototype_sha256']
    ref=z['reference_embeddings'];rp=z['reference_prototypes'];tp=z['text_prototypes']
    assert np.allclose(np.linalg.norm(ref,axis=1),1,atol=1e-5)
    groups={c:{} for c in refs['labels']}
    crop_checks=0
    rows_by_id={r['id']:r for r in m['images']}
    def check_crop(d,image_id):
        nonlocal crop_checks
        assert sha(RUN/d['file'])==d['sha256']
        with Image.open(RUN/rows_by_id[image_id]['file']) as im:
            expected=im.convert('RGB').crop(d['box'])
        with Image.open(RUN/d['file']) as actual:
            assert expected.size==actual.size and expected.tobytes()==actual.convert('RGB').tobytes()
        crop_checks+=1
    for d in refs['references']:
        assert rows_by_id[d['image_id']]['role']=='reference'
        groups[d['material']].setdefault(d['image_id'],[]).append(ref[d['embedding_index']]);check_crop(d,d['image_id'])
    expected=[]
    for c in refs['labels']:
        per_image=[]
        for values in groups[c].values():
            v=np.mean(values,0);per_image.append(v/np.linalg.norm(v))
        v=np.mean(per_image,0);expected.append(v/np.linalg.norm(v))
    assert np.allclose(rp,expected,atol=1e-6)
    expected_text=z['text_embeddings'].reshape(len(refs['text_labels']),2,-1).mean(1)
    expected_text/=np.linalg.norm(expected_text,axis=1,keepdims=True)
    assert np.allclose(tp,expected_text,atol=1e-6)
    counters={arm:Counter() for arm in cfg['detectors']};rows=[];raw_checks=0
    for source in m['images']:
        row=dict(source);row['annotations']=load(RUN/source['annotation_file'])['annotations'];row['predictions']={};row['materials']={}
        for arm in cfg['detectors']:
            if source['role']!='target': continue
            pf=RUN/f'predictions/{arm}/{source["id"]}.json';mf=RUN/f'materials/{arm}/{source["id"]}.json'
            pred=load(pf);mat=load(mf)
            assert pred['image_sha256']==source['sha256']==mat['image_sha256']
            assert mat['prediction_sha256']==sha(pf) and mat['prototype_sha256']==sha(RUN/'prototypes.npz')
            assert sha(RUN/mat['raw_file'])==mat['raw_sha256']
            raw=np.load(RUN/mat['raw_file']);raw_checks+=1
            rs=raw['image_embeddings']@rp.T;ts=raw['image_embeddings']@tp.T
            assert np.allclose(rs,raw['reference_cosines'],atol=1e-6) and np.allclose(ts,raw['text_cosines'],atol=1e-6)
            if arm=='grounding':
                assert suppress(pred['raw_predictions'],cfg['grounding']['nms_iou'])==pred['predictions']
                for d in pred['raw_files']:
                    assert sha(RUN/d['file'])==d['sha256'];raw_checks+=1
                raw_by_file={d['file']:np.load(RUN/d['file']) for d in pred['raw_files']}
                for p in pred['raw_predictions']:
                    f=raw_by_file[p['raw_file']];idx=p['query_index']
                    logits=f['token_logits'][idx];expected_score=float(np.max(1/(1+np.exp(-np.clip(logits,-80,80)))))
                    assert abs(p['score']-expected_score)<1e-6
                    cx,cy,w,h=map(float,f['boxes_cxcywh'][idx])
                    assert np.allclose(p['box'],[max(0.,(cx-w/2)*1280),max(0.,(cy-h/2)*720),min(1280.,(cx+w/2)*1280),min(720.,(cy+h/2)*720)],atol=1e-4)
            for d in mat['predictions']:
                p=pred['predictions'][d['prediction_index']]
                assert p['class_id']==pred['insulator_class_id'] and p['box']==d['box'] and p['score']==d['detector_score']
                assert not d['material_verified'] and not d['scores_are_probabilities']
                i=d['crops']['tight']['embedding_index'];j=d['crops']['context']['embedding_index']
                expected_label,expected_reason=material_decision(rs[i].tolist(),rs[j].tolist(),d['box'],cfg['material']['reference_margin'])
                assert d['reference_material']==expected_label and d['reference_reason']==expected_reason
                for view,index in [('tight',i),('context',j)]:
                    check_crop(d['crops'][view],source['id'])
                    assert np.allclose(list(d['reference_cosines'][view].values()),rs[index],atol=1e-6)
                    assert np.allclose(list(d['text_cosines'][view].values()),ts[index],atol=1e-6)
                counters[arm][d['reference_material']]+=1
            counters[arm]['images']+=1;counters[arm]['candidates']+=len(mat['predictions'])
            if not mat['predictions']: counters[arm]['empty_images']+=1
            row['predictions'][arm]=pred['predictions'];row['materials'][arm]=mat['predictions']
        rows.append(row)
    supervised_verification=None
    if supervised:
        sup=ROOT/'runs/paper_material_supervised/v1_20260828';results=load(sup/'results.json');scfg=results['config']
        assert results['status']=='COMPLETE' and results['training_progress']['epochs_completed']==10
        assert results['checkpoint'].endswith('/last.pt')
        assert sha(sup/results['checkpoint'])==results['checkpoint_sha256']
        assert sha(ROOT/'configs/paper_supervised_demo_v1.json')==results['protocol_sha256']
        assert scfg['source_demo_manifest_sha256']==sha(RUN/'manifest.json')
        sm=load(ROOT/scfg['dataset']/'manifest.json');assert sha(ROOT/scfg['dataset']/'manifest.json')==scfg['manifest_sha256']
        import csv
        history=list(csv.DictReader((sup/'training/results.csv').open()))
        assert len(history)==10
        assert all(np.isfinite(float(v)) for row in history for k,v in row.items() if 'loss' in k)
        assert len({r['source_file'] for r in sm['images']})==398 and len(sm['images'])==417
        assert len({r['source_file'] for r in sm['images'] if any(a['category_id']==2 for a in r['source_annotations'])})==1
        for r in sm['images']:
            for filekey,hashkey in [('image_file','image_sha256'),('label_file','label_sha256')]: assert sha(ROOT/scfg['dataset']/r[filekey])==r[hashkey]
            label_lines=(ROOT/scfg['dataset']/r['label_file']).read_text().splitlines()
            assert len(label_lines)==len(r['source_annotations'])
            for line,a in zip(label_lines,r['source_annotations']):
                cls,cx,cy,w,h=map(float,line.split());x,y,bw,bh=a['bbox']
                x1=max(0,min(1280,x));y1=max(0,min(720,y));x2=max(0,min(1280,x+bw));y2=max(0,min(720,y+bh))
                assert int(cls)==a['category_id']-1
                assert np.allclose([cx,cy,w,h],[(x1+x2)/2560,(y1+y2)/1440,(x2-x1)/1280,(y2-y1)/720],atol=1e-8)
        assert not {r['image_sha256'] for r in sm['images']}.intersection(r['sha256'] for r in rows if r['role']=='target')
        expected=set(r['id'] for r in rows if r['role']=='target');assert {r['image_id'] for r in results['predictions']}==expected
        by_id={r['id']:r for r in rows};counts=Counter();agreement=[]
        for entry in results['predictions']:
            assert sha(sup/entry['file'])==entry['sha256'];p=load(sup/entry['file']);row=by_id[entry['image_id']]
            assert p['image_sha256']==row['sha256'] and p['checkpoint_sha256']==results['checkpoint_sha256']
            ds=[]
            for i,pred in enumerate(p['predictions']):
                assert pred['class_name']==scfg['classes'][pred['class_id']]
                assert not pred['material_verified'] and not pred['material_probability_calibrated']
                x1,y1,x2,y2=pred['box'];assert 0<=x1<x2<=1280 and 0<=y1<y2<=720 and .05<=pred['score']<=1
                ds.append({'prediction_index':i,'box':pred['box'],'detector_score':pred['score'],
                    'reference_material':pred['class_name'],'text_material':pred['class_name'],'direct_supervised':True,
                    'material_verified':False,'scores_are_probabilities':False,
                    'reference_reason':'direct supervised class; uncalibrated detector score',
                    'raw_file':'supervised_source/'+entry['file']})
                ds[-1]['crops']={}
                with Image.open(RUN/row['file']) as im:
                    im=im.convert('RGB')
                    for view,padding in [('tight',0),('context',.12)]:
                        ex=extent(pred['box'],im.width,im.height,padding);f=f'crops/supervised/{row["id"]}_{i}_{view}.png'
                        (RUN/f).parent.mkdir(parents=True,exist_ok=True);im.crop(ex).save(RUN/f)
                        descriptor={'box':ex,'file':f,'sha256':sha(RUN/f)};ds[-1]['crops'][view]=descriptor;check_crop(descriptor,row['id'])
                counts[pred['class_name']]+=1
            row['materials']['supervised']=ds;row['predictions']['supervised']=p['predictions'];counts['images']+=1;counts['candidates']+=len(ds)
            if not ds:counts['empty_images']+=1
            if row['annotations']:
                matched=set();detected=[p for p in p['predictions'] if p['score']>=.25];tp_count=0
                for pred in sorted(detected,key=lambda p:p['score'],reverse=True):
                    best=-1;overlap=.5
                    for j,a in enumerate(row['annotations']):
                        if j in matched or pred['class_id']!=a['category_id']-1:continue
                        x,y,w,h=a['bbox'];iou=box_iou(pred['box'],[x,y,x+w,y+h])
                        if iou>=overlap:best=j;overlap=iou
                    if best>=0:matched.add(best);tp_count+=1
                item={'image_id':row['id'],'supplied_annotations':len(row['annotations']),'detections_at_025':len(detected),'matched_at_iou_050':tp_count,
                    'unmatched_predictions':len(detected)-tp_count,'missed_annotations':len(row['annotations'])-tp_count}
                agreement.append(item);row['supervised_annotation_agreement']=item
        counters['supervised']=counts
        link=RUN/'supervised_source'
        if link.is_symlink(): assert link.resolve()==sup.resolve()
        elif link.exists():raise FileExistsError('Do not overwrite an existing source directory')
        else:link.symlink_to(Path('../../paper_material_supervised/v1_20260828'),target_is_directory=True)
        supervised_verification={'job_id':results['runtime']['job_id'],'training_seconds':results['training_seconds'],'epochs':10,
            'unique_training_images':398,'weighted_training_entries':417,'porcelain_source_images':1,
            'checkpoint_sha256':results['checkpoint_sha256'],'source_labels':'Published UVInsDet train polygon boxes; not machine labels',
            'demo_annotation_agreement':agreement,'agreement_is_independent_accuracy':False,'score_threshold':.25,'iou_threshold':.5,
            'in_sample_validation_metrics_not_used':True,'all_training_losses_finite':True}
    glass_localisation=[]
    for r in rows:
        if r['role']!='target' or len(r['annotations'])!=1 or r['annotations'][0]['category_id']!=1:continue
        a=r['annotations'][0];x,y,w,h=a['bbox'];box=[x,y,x+w,y+h]
        candidates=[d for d in r['materials']['grounding'] if d['detector_score']>=.25]
        best=max(candidates,key=lambda d:box_iou(d['box'],box)) if candidates else None
        glass_localisation.append({'image_id':r['id'],'annotation_id':a['id'],'iou':box_iou(best['box'],box) if best else 0,
            'prediction_index':best['prediction_index'] if best else None,'reference_material':best['reference_material'] if best else None})
    report=RUN/'report';report.mkdir(exist_ok=True)
    data={'protocol':cfg,'images':rows,'references':refs,'summary':counters,'runtime':complete['runtime'],'supervised':supervised_verification,'glass_localisation':glass_localisation}
    write(report/'data.json',data)
    template=(ROOT/'templates/paper_material_demo.html').read_text()
    html=template.replace('__DATA__',json.dumps(data).replace('</','<\\/'))
    assert '__DATA__' not in html
    (report/'index.html').write_text(html)
    shutil.copyfile(ROOT/'scripts/build_paper_material_report.py',RUN/'code/report_builder.py')
    shutil.copyfile(ROOT/'templates/paper_material_demo.html',RUN/'code/report_template.html')
    verification={'status':'VERIFIED','images':len(rows),'targets':sum(r['role']=='target' for r in rows),
        'source_images_and_annotations_hash_checked':True,'exact_crop_pixel_checks':crop_checks,'raw_array_files_checked':raw_checks,
        'prototype_math_recomputed':True,'all_material_cosines_recomputed':True,'target_annotations_used_for_automatic_inference':False,
        'reference_images':4,'siglip_gradient_steps':0,'summary':counters,'html_sha256':sha(report/'index.html'),'data_sha256':sha(report/'data.json'),
        'manifest_sha256':sha(RUN/'manifest.json'),'limitations':cfg['scope'],'supervised':supervised_verification,
        'glass_localisation_diagnostic':glass_localisation,'glass_diagnostic_is_independent_accuracy':False}
    write(report/'verification.json',verification);print(json.dumps(verification,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--supervised',action='store_true');a=p.parse_args();build(a.supervised)
