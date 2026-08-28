"""Verify saved head arithmetic and source labels; no model runtime or new inference."""
import json
import math
import shutil
from collections import Counter

from paper_material_demo import ROOT, load, sha, write, extent
from prepare_material_head import CONFIG, intersection
from prepare_substation_material import polygon_references
from material_head_common import decide
from keen_component_metrics import summarize, match_image, iou


def classification_stats(rows):
    names=['glass','porcelain','other'];matrix=[[0]*3 for _ in names]
    acceptance=[[0]*3 for _ in names]
    for r in rows:
        matrix[r['class_id']][names.index(r['tight_argmax'])]+=1
        acceptance[r['class_id']][['glass','porcelain','unknown'].index(r['material'])]+=1
    accepted=sum(sum(r[:2]) for r in acceptance)
    correct=acceptance[0][0]+acceptance[1][1]
    return {'crops':len(rows),'tight_argmax_accuracy':sum(matrix[i][i] for i in range(3))/len(rows) if rows else None,
            'accepted':accepted,'correct_accepted':correct,'accepted_coverage':accepted/len(rows) if rows else None,
            'accepted_accuracy':correct/accepted if accepted else None,
            'raw_columns':names,'raw_confusion':matrix,'decision_columns':['glass','porcelain','unknown'],
            'decision_confusion':acceptance,'decision_counts':dict(Counter(r['material'] for r in rows)),
            'conditional_on_source_crop':True}


def build():
    import numpy as np
    from PIL import Image,ImageDraw
    cfg=load(CONFIG);out=ROOT/cfg['run'];data=ROOT/cfg['dataset'];source=ROOT/cfg['external_demo']
    result=load(out/'results.json');m=load(data/'manifest.json')
    assert result['status']=='COMPLETE' and result['head_gradient_steps']==400 and result['encoder_gradient_steps']==0
    assert result['protocol_sha256']==sha(CONFIG)==m['protocol_sha256'] and result['manifest_sha256']==sha(data/'manifest.json')
    assert result['config']==cfg
    assert len(result['training_losses'])==400 and all(math.isfinite(x) for x in result['training_losses'])
    for f,key in [('head.npz','head_sha256'),('features.npz','features_sha256'),('feature_index.json','feature_index_sha256'),
                  ('source_logits.npz','source_logits_sha256'),('oracle.json','oracle_sha256'),('oracle_features.npz','oracle_features_sha256')]:
        assert sha(out/f)==result[key]
    head=np.load(out/'head.npz');features=np.load(out/'features.npz')['embeddings'];index=load(out/'feature_index.json')
    logits=np.load(out/'source_logits.npz')['logits']
    assert np.isfinite(features).all() and np.isfinite(logits).all() and np.allclose(np.linalg.norm(features,axis=1),1,atol=1e-5)
    assert np.allclose(features@head['weight'].T+head['bias'],logits,atol=1e-5)
    assert len(index)==len(m['crops'])*2==len(features)
    lookup={(r['crop_id'],r['view']):i for i,r in enumerate(index)};images={r['id']:r for r in m['images']}
    assert len(lookup)==len(index)
    assert not {r['capture_group'] for r in m['images'] if r['split']=='train'}&{r['capture_group'] for r in m['images'] if r['split']=='development'}
    report=out/'report';report.mkdir(exist_ok=True);(report/'crops').mkdir(exist_ok=True);(report/'images').mkdir(exist_ok=True)
    annotations={};source_rows=[];crop_checks=0
    for c in m['crops']:
        src=images[c['image_id']]
        assert src['split']==c['split']
        if src['id'] not in annotations:
            af=data/src['annotation_file'];assert sha(af)==src['annotation_sha256'];annotations[src['id']]=load(af)
        a=annotations[src['id']];shape=a['shapes'][c['annotation_index']]
        assert shape['label']==c['publisher_label']
        refs=polygon_references(a,cfg['publisher_classes'])
        if c['class_id']<2:
            ref=next(r for r in refs if r['annotation_index']==c['annotation_index'])
            assert ref['class_id']==c['class_id'] and ref['box']==c['box']
        else:
            assert c['derived_negative'] and shape['label'] in cfg['negative_source_classes']
            assert not any(intersection(c['box'],r['box']) for r in refs)
            mask=Image.new('L',(src['width'],src['height']));ImageDraw.Draw(mask).polygon([tuple(p) for p in shape['points']],fill=255)
            assert mask.crop(c['box']).getextrema()==(255,255)
        for view,d in c['views'].items():
            assert sha(data/d['file'])==d['sha256']
            im=Image.open(data/d['file']);assert im.getexif().get(274) is None
            assert im.size==(d['box'][2]-d['box'][0],d['box'][3]-d['box'][1]);crop_checks+=1
        ti=lookup[(c['id'],'tight')];ci=lookup[(c['id'],'context')]
        for i in [ti,ci]:assert index[i]['split']==c['split'] and index[i]['class_id']==c['class_id']
        if c['split']=='development':
            r=dict(decide(logits[ti].tolist(),logits[ci].tolist(),c['box'],cfg),id=c['id'],image_id=c['image_id'],
                   class_id=c['class_id'],source_name=src['source_name'],publisher_label=c['publisher_label'],box=c['box'],
                   tight_logits=logits[ti].tolist(),context_logits=logits[ci].tolist(),crop_file=f'crops/{c["id"]}.png')
            shutil.copyfile(data/c['views']['tight']['file'],report/r['crop_file']);source_rows.append(r)
    oracle=load(out/'oracle.json');of=np.load(out/'oracle_features.npz')
    assert np.allclose(of['embeddings']@head['weight'].T+head['bias'],of['logits'],atol=1e-5)
    targets={r['id']:r for r in load(source/'manifest.json')['images'] if r['role']=='target'}
    assert sha(source/'manifest.json')==cfg['external_manifest_sha256']
    for r in oracle:
        row=targets[r['image_id']];af=source/row['annotation_file'];assert sha(af)==row['annotation_sha256']
        a=next(a for a in load(af)['annotations'] if a['id']==r['annotation_id']);x,y,w,h=a['bbox']
        assert r['box']==[x,y,x+w,y+h] and r['class_id']==a['category_id']-1
        ti,ci=r['view_indices']['tight'],r['view_indices']['context']
        assert np.allclose(r['tight_logits'],of['logits'][ti]) and np.allclose(r['context_logits'],of['logits'][ci])
        expected=decide(r['tight_logits'],r['context_logits'],r['box'],cfg)
        assert all(r[k]==v for k,v in expected.items())
        with Image.open(source/row['file']) as im:
            f=f'crops/oracle_{r["id"]}.png';im.convert('RGB').crop(extent(r['box'],im.width,im.height)).save(report/f);r['crop_file']=f
    automatic=[];metric_rows={'head':[],'prototype':[]};localisation=[]
    assert len(result['external_automatic'])==8
    for entry in result['external_automatic']:
        assert sha(out/entry['file'])==entry['sha256'];record=load(out/entry['file']);row=targets[record['image_id']]
        assert record['image_sha256']==sha(source/row['file'])==row['sha256']
        mf=source/f'materials/grounding/{row["id"]}.json'
        pf=source/f'predictions/grounding/{row["id"]}.json'
        old=load(mf)
        assert sha(mf)==record['source_material_sha256']
        assert sha(pf)==record['source_prediction_sha256']==old['prediction_sha256']
        assert sha(source/old['raw_file'])==record['source_embedding_sha256']==old['raw_sha256']
        vectors=np.load(source/old['raw_file'])['image_embeddings'];values=vectors@head['weight'].T+head['bias']
        assert len(record['decisions'])==len(old['predictions'])
        for d,p in zip(record['decisions'],old['predictions']):
            assert d['prediction_index']==p['prediction_index'] and d['box']==p['box'] and d['detector_score']==p['detector_score']
            assert d['previous_reference_material']==p['reference_material']
            ti,ci=p['crops']['tight']['embedding_index'],p['crops']['context']['embedding_index']
            assert np.allclose(d['tight_logits'],values[ti],atol=1e-5) and np.allclose(d['context_logits'],values[ci],atol=1e-5)
            expected=decide(d['tight_logits'],d['context_logits'],d['box'],cfg);assert all(d[k]==v for k,v in expected.items())
        references=[]
        af=source/row['annotation_file'];assert sha(af)==row['annotation_sha256']
        for a in load(af)['annotations']:
            x,y,w,h=a['bbox'];references.append({'class_id':a['category_id']-1,'box':[x,y,x+w,y+h]})
        for ref in references:
            # Proposal coverage only: not one-to-one matching or material accuracy.
            overlap=max((iou(ref['box'],d['box']) for d in record['decisions'] if d['detector_score']>=.25),default=0)
            localisation.append(dict(ref,image_id=row['id'],best_proposal_iou=overlap))
        f=f'images/{row["id"]}.jpg';shutil.copyfile(source/row['file'],report/f);assert sha(report/f)==row['sha256']
        output={'image_id':row['id'],'file':f,'width':row['width'],'height':row['height'],'references':references,
                'decisions':record['decisions'],'source_sha256':row['sha256'],'raw_file':'../'+entry['file'],'agreement':{}}
        for arm,key in [('head','material'),('prototype','previous_reference_material')]:
            predictions=[{'box':d['box'],'class_id':['glass','porcelain'].index(d[key]),'score':d['detector_score']}
                         for d in record['decisions'] if d[key] in ['glass','porcelain']]
            r={'image_id':row['id'],'references':references,'predictions':predictions}
            output['agreement'][arm]=match_image(predictions,references,.5,.25) if references else None
            if references:metric_rows[arm].append(r)
        automatic.append(output)
    summary={'data':m['summary'],'development':classification_stats(source_rows),'oracle':classification_stats(oracle),
             'automatic':{arm:summarize(rows,['glass','porcelain'],thresholds=(.25,)) for arm,rows in metric_rows.items()},
             'proposal_coverage':{name:{'source_objects':sum(r['class_id']==i for r in localisation),
                                      'covered_at_iou_50':sum(r['class_id']==i and r['best_proposal_iou']>=.5 for r in localisation)}
                                  for i,name in enumerate(['glass','porcelain'])},
             'job_id':result['runtime']['job_id'],'elapsed_seconds':result['elapsed_seconds'],'head_sha256':result['head_sha256']}
    payload={'config':cfg,'summary':summary,'images':automatic,'development_crops':source_rows,'oracle_crops':oracle,
             'proposal_coverage_references':localisation}
    write(report/'data.json',payload)
    html=(ROOT/'templates/material_head_report.html').read_text().replace('__DATA__',json.dumps(payload).replace('</','<\\/'))
    (report/'index.html').write_text(html)
    write(report/'verification.json',{'status':'VERIFIED','source_crop_file_checks':crop_checks,'source_to_original_pixel_checks_on_roihu':result['source_to_crop_pixel_checks'],
          'classifier_arithmetic_recomputed':True,'automatic_ground_truth_boxes_used':False,'oracle_is_separate':True,
          'html_sha256':sha(report/'index.html'),'data_sha256':sha(report/'data.json'),'summary':summary})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':build()
