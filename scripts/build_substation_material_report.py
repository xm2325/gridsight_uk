"""Independently verify the public material experiment and render an English report."""
import csv
import math
import shutil
from collections import Counter

from paper_material_demo import ROOT, load, sha, write
from prepare_substation_material import CONFIG, polygon_references, label_text, duplicate_components
from keen_component_metrics import match_image, summarize, validate_predictions, geometric_confusion


def build():
    from PIL import Image
    cfg=load(CONFIG); out=ROOT/cfg['run']; data=ROOT/cfg['dataset']; result=load(out/'results.json')
    assert result['status']=='COMPLETE' and result['training_progress']['epochs_completed']==cfg['epochs']==20
    assert result['protocol_sha256']==sha(CONFIG) and result['config']==cfg
    assert sha(data/'manifest.json')==result['manifest_sha256']
    assert result['checkpoint'].endswith('/last.pt') and sha(out/result['checkpoint'])==result['checkpoint_sha256']
    manifest=load(data/'manifest.json'); audit=load(data/'selection_audit.json')
    assert sha(data/'selection_audit.json')==manifest['selection_audit_sha256']
    assert manifest['protocol_sha256']==result['protocol_sha256']
    history=list(csv.DictReader((out/'training/results.csv').open()))
    assert len(history)==20 and all(math.isfinite(float(v)) for row in history for k,v in row.items() if 'loss' in k)
    train=[r for r in manifest['images'] if r['split']=='train']; dev=[r for r in manifest['images'] if r['split']=='development']
    assert len(train)==600 and len(dev)==25
    assert not {r['capture_group'] for r in train}&{r['capture_group'] for r in dev}
    assert not {r['pixel_sha256'] for r in train}&{r['pixel_sha256'] for r in dev}
    _,edges=duplicate_components(manifest['images'],cfg['dhash_distance'],cfg['dhash_aspect_ratio_tolerance'])
    assert not edges
    for row in manifest['images']:
        for f,h in [('label_file','label_sha256'),('annotation_file','annotation_sha256')]:
            assert sha(data/row[f])==row[h]
        a=load(data/row['annotation_file']); assert polygon_references(a,cfg['publisher_classes'])==row['references']
        assert (data/row['label_file']).read_text()==label_text(row['references'],row['width'],row['height'])
    source=ROOT/cfg['source_demo']; assert sha(source/'manifest.json')==cfg['source_demo_manifest_sha256']
    external=[r for r in load(source/'manifest.json')['images'] if r['role']=='target']
    expected={('development',r['id']) for r in dev}|{('external_demo',r['id']) for r in external}
    assert {(r['split'],r['image_id']) for r in result['predictions']}==expected and len(result['predictions'])==len(expected)
    targets={('development',r['id']):r for r in dev};targets.update({('external_demo',r['id']):r for r in external})
    report=out/'report'; report.mkdir(exist_ok=True); (report/'images').mkdir(exist_ok=True)
    rows=[]
    for entry in result['predictions']:
        assert sha(out/entry['file'])==entry['sha256']
        payload=load(out/entry['file']); row=targets[(entry['split'],entry['image_id'])]
        assert payload['checkpoint_sha256']==result['checkpoint_sha256'] and payload['protocol_sha256']==result['protocol_sha256']
        original=data/row['image_file'] if entry['split']=='development' else source/row['file']
        expected_sha=row['image_sha256'] if entry['split']=='development' else row['sha256']
        assert sha(original)==expected_sha==payload['image_sha256']
        with Image.open(original) as im:
            assert im.size==(row['width'],row['height'])
        predictions=payload['predictions'];validate_predictions(predictions,row['width'],row['height'],2)
        assert all(p['class_name']==cfg['classes'][p['class_id']] and not p['material_verified'] and not p['material_probability_calibrated'] and p['score']>=.05 for p in predictions)
        if entry['split']=='development':
            refs=row['references']; source_name=row['source_name']; annotation_file=data/row['annotation_file']; inspected=row['previously_inspected']
            assert sha(annotation_file)==row['annotation_sha256']
        else:
            annotation_file=source/row['annotation_file']; assert sha(annotation_file)==row['annotation_sha256']
            supplied=load(annotation_file)['annotations']; refs=[]
            for a in supplied:
                x,y,w,h=a['bbox'];refs.append({'class_id':a['category_id']-1,'box':[x,y,x+w,y+h],'segmentation':a['segmentation']})
            source_name=original.name; inspected=True
        image_file=f'images/{entry["split"]}_{row["id"]}{original.suffix.lower()}'
        shutil.copyfile(original,report/image_file);assert sha(report/image_file)==expected_sha
        af=f'images/{entry["split"]}_{row["id"]}_source.json';shutil.copyfile(annotation_file,report/af)
        record={'image_id':row['id'],'split':entry['split'],'source_name':source_name,'image_file':image_file,'image_sha256':expected_sha,
                'width':row['width'],'height':row['height'],'predictions':predictions,'references':refs,
                'annotation_file':af,'raw_file':'../'+entry['file'],'previously_inspected':inspected,
                'metrics_available':bool(refs) or entry['split']=='development',
                'source_url':cfg['source'] if entry['split']=='development' else 'https://zenodo.org/records/18197601'}
        record['agreement_025']=match_image(predictions,refs,.5,.25) if record['metrics_available'] else None
        rows.append(record)
    metrics={s:summarize([r for r in rows if r['split']==s and r['metrics_available']],cfg['classes'],thresholds=(.25,.5)) for s in ['development','external_demo']}
    confusion={s:geometric_confusion([r for r in rows if r['split']==s and r['metrics_available']],cfg['classes']) for s in metrics}
    summary={'training_images':len(train),'development_images':len(dev),'source_geometry_exclusions':len(audit['exclusions']),
             'duplicate_drops':len(audit['same_split_duplicate_drops']),'cross_split_quarantined':len(audit['cross_split_quarantined']),
             'training_image_classes':audit['selected_image_classes']['train'],'metrics':metrics,'confusion':confusion,
             'checkpoint_sha256':result['checkpoint_sha256'],'job_id':result['runtime']['job_id'],'runtime':result['runtime'],
             'training_seconds':result['training_seconds'],'elapsed_seconds':result['elapsed_seconds'],
             'raw_counts':dict(Counter(p['class_name'] for r in rows for p in r['predictions']))}
    view={'config':cfg,'summary':summary,'images':rows}
    write(report/'data.json',view)
    import json
    html=(ROOT/'templates/substation_material_report.html').read_text().replace('__DATA__',json.dumps(view).replace('</','<\\/'))
    assert '__DATA__' not in html
    (report/'index.html').write_text(html)
    verification={'status':'VERIFIED','source_annotation_and_derived_label_checks':len(manifest['images']),
                  'display_image_hash_checks':len(rows),'prediction_file_hash_checks':len(rows),'frozen_final_checkpoint_checked':True,
                  'training_losses_finite_epochs':len(history),'selected_duplicate_graph_recomputed':True,'independent_asset_accuracy':False,
                  'training_image_bytes_checked_locally':False,'source_image_bytes_checked_by_remote_runner':result['source_labels_recomputed'],
                  'html_sha256':sha(report/'index.html'),'data_sha256':sha(report/'data.json'),'manifest_sha256':result['manifest_sha256'],
                  'summary':summary,'ap_scope':'101-point AP from saved post-NMS boxes at .05 floor; no COCO crowd/area ignore; not library default AP.'}
    write(report/'verification.json',verification)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':build()
