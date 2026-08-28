"""Check completed masks, measure source agreement, and build an English viewer."""
import hashlib
import json
import math
import shutil
from collections import Counter,defaultdict
from paper_material_demo import ROOT,load,sha,write
from prepare_component_masks import CONFIG,segmentation_line
from component_mask_metrics import decode_masks,raster_polygon,rectangle_mask,mask_iou,mask_matches,pole_end_candidate
from keen_component_metrics import summarize,match_image,validate_predictions,nms
from roihu_demo_ablation import counts_to_metrics
from resume_component_masks import training_audit


def main():
    import numpy as np
    from PIL import Image
    cfg=load(CONFIG);data=ROOT/cfg['dataset'];out=ROOT/cfg['run']/'inference';report=out/'report'
    result=load(out/'results.json');m=load(data/'manifest.json');original=load(ROOT/cfg['source_dataset']/'manifest.json')
    assert result['status']=='COMPLETE' and result['config']==cfg
    failed,training_rows=training_audit(out.parent)
    assert result['extra_training_steps']==0
    assert sha(out.parent/'results.json')==result['training_result_sha256']
    assert sha(out.parent/'training/results.csv')==result['training_csv_sha256']
    assert result['epoch_losses']==failed['epoch_losses'][:20]
    assert result['protocol_sha256']==sha(CONFIG)==m['protocol_sha256']
    assert result['manifest_sha256']==sha(data/'manifest.json')
    assert sha(ROOT/cfg['source_dataset']/'manifest.json')==cfg['source_manifest_sha256']
    assert sha(out/result['selected_checkpoint'])==result['checkpoint_sha256']
    assert [r['epoch'] for r in result['epoch_losses']]==list(range(1,21))
    assert all(math.isfinite(v) for r in result['epoch_losses'] for v in r['losses'])
    pixels=load(out/'source_pixel_verification.json');assert pixels['status']=='VERIFIED' and pixels['results_sha256']==sha(out/'results.json')
    pixel_by_id={r['image_id']:r for r in pixels['checked']};assert len(pixel_by_id)==107
    source_rows={r['image_id']:r for r in original['images'] if r['split'] in ['train','dev']}
    for r in m['images']:
        source=source_rows[r['image_id']];assert source['references']==r['references'] and source['sha256']==r['sha256']
        assert r['split']==source['split'] and r['circuit']==source['circuit']
        lp=data/r['label_file'];assert sha(lp)==r['label_sha256']
        assert lp.read_text().splitlines()==[segmentation_line(a,r['width'],r['height']) for a in source['references']]
    assert Counter(r['split'] for r in m['images'])=={'train':320,'dev':80}
    train_circuits={r['circuit'] for r in m['images'] if r['split']=='train'}
    assert not train_circuits&{r['circuit'] for r in m['images'] if r['split']=='dev'}
    ukm=ROOT/cfg['uk_dataset']/'manifest.json';assert sha(ukm)==cfg['uk_manifest_sha256']
    uk={r['image_id']:r for r in load(ukm)['images']}
    oldrun=ROOT/'runs/keen_components/epri_components_v1_20260827'
    assert sha(oldrun/'results.json')=='d5bc2d70c50a23ed483b8c2378b4d8e0e49d264193314c1c9f828d6122d44a07'
    oldrecords={split:{r['image_id']:r for r in load(oldrun/split/'supervised.json')['records']} for split in ['dev','uk']}
    hardware=ROOT/'runs/uk_capabilities/v3_20260827'
    assert sha(hardware/'results.json')=='9f42a8a41b9390d9e8a15aae27ea1f48d9309fab7f52279b20683d1bd0bab14a'
    hardware_records={r['image_id']:r for r in load(hardware/'results.json')['records']}
    (report/'images').mkdir(parents=True,exist_ok=True);(report/'masks').mkdir(exist_ok=True)
    images=[];box_rows=[];legacy_rows=[];nms_rows=[];mask_counts=defaultdict(Counter);nms_counts=defaultdict(Counter);extent=defaultdict(list);geometry_counts=Counter();mask_checks=0
    colours=[(243,189,82),(92,217,153),(163,146,255)]
    for record in result['predictions']:
        p=out/record['file'];assert sha(p)==record['sha256'];d=load(p);key=d['image_id']
        is_dev=d['split']=='development';source=source_rows[key] if is_dev else uk[key]
        assert (source['split']=='dev') if is_dev else source['ground_truth_status']=='NONE'
        assert d['source_image_sha256']==source['sha256'] and d['source_size']==[source['width'],source['height']]
        w,h=d['working_size'];sw,sh=d['source_size']
        image_file=p.parent/'input.png';assert sha(image_file)==d['input_sha256']
        with Image.open(image_file) as im:
            assert im.size==(w,h) and hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==pixel_by_id[key]['working_pixel_sha256']
        rawfile=p.parent/d['raw_file'];assert sha(rawfile)==d['raw_sha256'];raw=np.load(rawfile);masks=decode_masks(raw)
        assert len(masks)==len(d['predictions']) and np.isfinite(raw['boxes']).all() and np.isfinite(raw['scores']).all()
        validate_predictions([dict(a,box=a['box_working']) for a in d['predictions']],w,h,3)
        for pred in d['predictions']:
            # Preserve float32 serialization, then map raw working boxes in float64.
            corrected=[float(v)*scale for v,scale in zip(pred['box_working'],[sw/w,sh/h,sw/w,sh/h])]
            assert np.allclose(pred['box'],corrected,rtol=0,atol=.001)
            pred['serialized_source_box']=pred['box'];pred['box']=corrected
        validate_predictions(d['predictions'],sw,sh,3)
        for j,pred in enumerate(d['predictions']):
            assert pred['prediction_index']==j and pred['class_id']==raw['classes'][j] and pred['score']==float(raw['scores'][j])
            assert np.array_equal(pred['box_working'],raw['boxes'][j])
            assert np.allclose(pred['box'],raw['boxes'][j]*np.array([sw/w,sh/h,sw/w,sh/h]),rtol=0,atol=1e-6)
            assert pred['mask_pixels']==int(masks[j].sum()) and pred['material']=='unknown' and not pred['material_verified']
            rgba=np.zeros((h,w,4),dtype=np.uint8);rgba[masks[j],:3]=colours[pred['class_id']];rgba[masks[j],3]=110
            f=f'masks/{key}_{j}.png';Image.fromarray(rgba).save(report/f);pred['mask_file']=f
            with Image.open(report/f) as saved:assert np.array_equal(np.asarray(saved)[:,:,3]>0,masks[j])
            mask_checks+=1
        refs=source['references'] if is_dev else []
        reference_masks=[raster_polygon(a['polygon'],[sw,sh],[w,h]) for a in refs]
        oldsplit='dev' if is_dev else 'uk';oldrec=oldrecords[oldsplit][key];oldfile=oldrun/oldsplit/oldrec['prediction_file']
        assert sha(oldfile)==oldrec['prediction_sha256'];old=load(oldfile);assert old['image_sha256']==source['sha256']
        old_predictions=old['predictions']
        for pred in old_predictions:pred['box_working']=[v*s for v,s in zip(pred['box'],[w/sw,h/sh,w/sw,h/sh])]
        bm=match_image(d['predictions'],refs,.5,.25) if is_dev else None
        mm=mask_matches(d['predictions'],masks,refs,reference_masks) if is_dev else None
        deduplicated=nms(d['predictions'],cfg['inference']['nms_iou'])
        nm=mask_matches(deduplicated,masks,refs,reference_masks) if is_dev else None
        nb=match_image(deduplicated,refs,.5,.25) if is_dev else None
        if is_dev:
            box_rows.append({'image_id':key,'predictions':d['predictions'],'references':refs})
            legacy_rows.append({'image_id':key,'predictions':old_predictions,'references':refs})
            nms_rows.append({'image_id':key,'predictions':deduplicated,'references':refs})
            for match in nm['matches']:nms_counts[match['class_id']]['tp']+=1
            for j in nm['false_predictions']:nms_counts[d['predictions'][j]['class_id']]['fp']+=1
            for j in nm['missed_references']:nms_counts[refs[j]['class_id']]['fn']+=1
            for match in mm['matches']:mask_counts[match['class_id']]['tp']+=1
            for j in mm['false_predictions']:mask_counts[d['predictions'][j]['class_id']]['fp']+=1
            for j in mm['missed_references']:mask_counts[refs[j]['class_id']]['fn']+=1
            for match in bm['matches']:
                j,k=match['prediction_index'],match['reference_index'];pred=d['predictions'][j]
                extent[pred['class_id']].append({'image_id':key,'prediction_index':j,'reference_index':k,
                     'mask_iou':mask_iou(masks[j],reference_masks[k]),
                     'rectangle_iou':mask_iou(rectangle_mask(pred['box_working'],[w,h]),reference_masks[k])})
        component_boxes=[a['box_working'] for a in d['predictions'] if a['class_id'] in [1,2] and a['score']>=.25]
        endpoints=[]
        for pred in d['predictions']:
            if pred['class_id']!=0 or pred['score']<.25:continue
            endpoint=dict(pole_end_candidate(masks[pred['prediction_index']],component_boxes),prediction_index=pred['prediction_index'])
            endpoints.append(endpoint);geometry_counts[(d['split'],endpoint['status'])]+=1
        steel=[]
        if not is_dev:
            hr=hardware_records[key];hf=hardware/hr['hardware_file'];assert sha(hf)==hr['hardware_sha256']
            hd=load(hf);assert hd['image_sha256']==source['sha256']
            steel=[dict(a,box_working=[v*s for v,s in zip(a['box'],[w/sw,h/sh,w/sw,h/sh])]) for a in hd['predictions'] if a['score']>=.3]
        f=f'images/{key}.png';shutil.copyfile(image_file,report/f)
        images.append({'image_id':key,'split':d['split'],'working_size':[w,h],'source_size':[sw,sh],'image_file':f,
                       'source_sha256':source['sha256'],'predictions':d['predictions'],'legacy_predictions':old_predictions,
                       'references':refs,'box_agreement':bm,'mask_agreement':mm,'pole_end_candidates':endpoints,'steelwork_proposals':steel,
                       'nms_prediction_indices':[p['prediction_index'] for p in deduplicated],'nms_box_agreement':nb,'nms_mask_agreement':nm,
                       'legacy_box_agreement':match_image(old_predictions,refs,.5,.25) if is_dev else None,
                       'raw_file':'../'+record['file'],'source_page':original['source_page'] if is_dev else source['source_page'],
                       'credit':original['publisher'] if is_dev else source['attribution'],
                       'license':original['license'] if is_dev else source['license'],
                       'source_label_review_flag':key=='epri_c4_176'})
    assert len(images)==107 and Counter(r['split'] for r in images)=={'development':80,'uk_qualitative':27}
    summary={'job_id':result['runtime']['slurm_job_id'],'training_seconds':result['training_seconds'],'elapsed_seconds':result['elapsed_seconds'],
             'source_counts':m['summary'],'box':summarize(box_rows,cfg['classes'],thresholds=(.25,)),
             'legacy_box':summarize(legacy_rows,cfg['classes'],thresholds=(.25,)),
             'nms_box':summarize(nms_rows,cfg['classes'],thresholds=(.25,)),
             'mask':{name:counts_to_metrics(*(mask_counts[c][k] for k in ['tp','fp','fn'])) for c,name in enumerate(cfg['classes'])},
             'nms_mask':{name:counts_to_metrics(*(nms_counts[c][k] for k in ['tp','fp','fn'])) for c,name in enumerate(cfg['classes'])},
             'extent':{name:{'matched_boxes':len(extent[c]),'mean_mask_iou':sum(r['mask_iou'] for r in extent[c])/len(extent[c]) if extent[c] else None,
                            'mean_rectangle_iou':sum(r['rectangle_iou'] for r in extent[c])/len(extent[c]) if extent[c] else None} for c,name in enumerate(cfg['classes'])},
             'pole_end_counts':{f'{s}/{k}':v for (s,k),v in geometry_counts.items()},'rendered_masks':mask_checks}
    payload={'config':cfg,'summary':summary,'images':images,'extent_pairs':dict(extent)}
    write(report/'data.json',payload)
    html=(ROOT/'templates/component_masks_report.html').read_text().replace('__DATA__',json.dumps(payload).replace('</','<\\/'))
    (report/'index.html').write_text(html)
    write(report/'verification.json',{'status':'VERIFIED','source_label_files':400,'original_to_working_pixel_checks':107,
                                    'mask_png_pixel_checks':mask_checks,'checkpoint_sha256':result['checkpoint_sha256'],
                                    'html_sha256':sha(report/'index.html'),'data_sha256':sha(report/'data.json'),'summary':summary})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
