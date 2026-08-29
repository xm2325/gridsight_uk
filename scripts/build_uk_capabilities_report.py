"""Verify v3 tensors and automatic crops, then render a separately versioned English UI."""
import json,shutil,os
from collections import Counter
from pathlib import Path
from prepare_keen_components import ROOT,digest,write_json
from roihu_demo_ablation import nms
from uk_capability_common import crop_extent,material_quality,diagnostic_decision


def build():
    import numpy as np
    from PIL import Image
    run=ROOT/'runs/uk_capabilities/v3_20260827';report=json.loads((run/'results.json').read_text());cfg=report['config']
    assert report['status']=='COMPLETED_UNVALIDATED_CAPABILITY_DIAGNOSTICS' and report['completed_images']==27
    assert report['performance_metrics'] is None and not report['training_started']
    assert digest(run/'frozen_choices.json')==report['frozen_choices_sha256']
    frozen=json.loads((run/'frozen_choices.json').read_text());assert digest(run/'code/uk_capabilities_v3.json')==frozen['protocol_sha256']
    assert cfg==json.loads((run/'code/uk_capabilities_v3.json').read_text())
    source=ROOT/cfg['source_run'];assert digest(source/'results.json')==cfg['source_results_sha256']
    assert digest(source/'report/data.json')==cfg['source_bundle_sha256']
    old_archive=source/'archive_manifest.json';assert digest(old_archive)=='5a340c9f1aad7f45dc9d0d3fb03e19f81a9d581f8118eaf0656ecde7b75c88d3'
    archived=json.loads(old_archive.read_text())
    for f in archived['files']:assert digest(source/f['file'])==f['sha256'],f['file']
    bundle=json.loads((source/'report/data.json').read_text());assert frozen['image_ids']==[r['image_id'] for r in bundle['images']]
    assert digest(run/'text_embeddings.npz')==report['text_embeddings_sha256']
    with np.load(run/'text_embeddings.npz',allow_pickle=False) as data:
        text=data['text_embeddings'];prototypes=data['class_prototypes'];expected=text.reshape(len(cfg['material_prompts']),2,-1).mean(1);expected/=np.linalg.norm(expected,axis=1,keepdims=True)
        np.testing.assert_allclose(prototypes,expected,atol=2e-6)
    rows={r['image_id']:r for r in report['records']};reasons=Counter();hypotheses=Counter();hardware_counts=Counter();raw_checks=0;crop_checks=0;encoded=0
    assert set(rows)==set(frozen['image_ids'])
    for r in bundle['images']:
        key=r['image_id'];record=rows[key];photo_path=source/'report'/r['image_file'];assert digest(photo_path)==r['sha256']
        with Image.open(photo_path) as f:photo=f.convert('RGB')
        for file_key,hash_key in [('hardware_file','hardware_sha256'),('material_file','material_sha256')]:assert digest(run/record[file_key])==record[hash_key]
        hardware=json.loads((run/record['hardware_file']).read_text());assert hardware['image_sha256']==r['sha256']
        arrays={}
        for f in hardware['raw_files']:
            assert digest(run/f['file'])==f['sha256']
            with np.load(run/f['file'],allow_pickle=False) as a:arrays[f['file']]={k:a[k].copy() for k in a.files}
        for p in hardware['raw_predictions']:
            a=arrays[p['raw_file']];i=p['query_index'];box=a['boxes_cxcywh'][i];cx,cy,w,h=map(float,box)
            expected=[max(0.,(cx-w/2)*photo.width),max(0.,(cy-h/2)*photo.height),min(float(photo.width),(cx+w/2)*photo.width),min(float(photo.height),(cy+h/2)*photo.height)]
            np.testing.assert_allclose(p['box'],expected,atol=1e-8)
            score=float((1/(1+np.exp(-np.clip(a['token_logits'][i],-80,80)))).max())
            assert abs(score-p['score'])<2e-6 and p['class_id']==3 and not p['steel_composition_verified'];raw_checks+=1
        assert nms(hardware['raw_predictions'],cfg['nms_iou'])==hardware['predictions']
        for threshold in [.2,.3,.5]:hardware_counts[str(threshold)]+=sum(p['score']>=threshold for p in hardware['predictions'])
        material=json.loads((run/record['material_file']).read_text());assert material['image_sha256']==r['sha256']
        f=material['raw_embeddings'];assert digest(run/f['file'])==f['sha256']
        with np.load(run/f['file'],allow_pickle=False) as a:
            embeddings=a['image_embeddings'];scores=a['cosine_scores'];np.testing.assert_allclose(scores,embeddings@prototypes.T,atol=2e-6)
        expected_indices=[i for i,p in enumerate(r['predictions'][cfg['material_proposal_arm']]) if p['class_id']==2 and p['score']>=cfg['material_proposal_score']]
        assert expected_indices==[d['candidate_index'] for d in material['diagnostics']]
        for d in material['diagnostics']:
            p=r['predictions'][cfg['material_proposal_arm']][d['candidate_index']];assert d['box']==p['box'] and d['source_image_sha256']==r['sha256']
            assert d['material']=='unknown' and d['accepted'] is False and d['scores_are_probabilities'] is False
            if material_quality(p['box'],cfg):
                tight=scores[d['crops']['tight']['embedding_index']].tolist();context=scores[d['crops']['context']['embedding_index']].tolist()
                decision=diagnostic_decision(tight,context,material['labels'])
                assert all(d[k]==v for k,v in decision.items());hypotheses[d['tight_hypothesis']]+=1
                for view,padding in [('tight',0),('context',cfg['context_padding_fraction'])]:
                    crop=d['crops'][view];assert crop['box']==crop_extent(p['box'],photo.width,photo.height,padding)
                    assert digest(run/crop['file'])==crop['sha256']
                    with Image.open(run/crop['file']) as im:np.testing.assert_array_equal(np.asarray(im),np.asarray(photo.crop(crop['box'])))
                    np.testing.assert_allclose(list(d['cosine_scores'][view].values()),scores[crop['embedding_index']],atol=2e-6);crop_checks+=1
            else:assert d['reason']=='insufficient_native_pixels' and 'crops' not in d
            reasons[d['reason']]+=1
        encoded+=len(embeddings)
        r['image_file']='../source/report/'+r['image_file']
        r['raw_files']={arm:'../source/'+file.removeprefix('../') for arm,file in r['raw_files'].items()}
        r['predictions']['dino_hardware']=hardware['predictions'];r['raw_files']['dino_hardware']='../'+record['hardware_file']
        r['material_diagnostics']=material['diagnostics'];r['material_raw']='../'+record['material_file']
    link=run/'source'
    if not link.exists():link.symlink_to(os.path.relpath(source,run),target_is_directory=True)
    assert link.resolve()==source.resolve()
    out=run/'report';out.mkdir(exist_ok=True)
    bundle.update(schema='gridsight-uk-review-v3',classes=['pole','crossarm','insulator','steelwork','pole-top'],runtime=report['runtime'],
        role='Unvalidated capability diagnostics and annotation drafts',capability_counts={'material_candidates':sum(reasons.values()),'encoded_crop_views':encoded,'reasons':dict(reasons),'tight_hypotheses':dict(hypotheses),'steelwork_proposal_counts':dict(hardware_counts)})
    template=ROOT/'templates/uk_component_review_v3.html';html=template.read_text();assert not any('\u3400'<=c<='\u9fff' for c in html)
    write_json(out/'data.json',bundle)
    (out/'index.html').write_text(html.replace('__DATA_JSON__',json.dumps(bundle,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c')))
    for file in ['example_candidates.jpg','review_example_5722811.jpg','review_example_6494360.jpg','review_example_4120413.jpg']:shutil.copyfile(source/'report'/file,out/file)
    import build_keen_components_report as renderer
    if len(renderer.COLORS)==3:renderer.COLORS.append('#f28537')
    for ident in ['7106830','5722811']:
        r=next(r for r in bundle['images'] if r['image_id']=='uk_geograph_'+ident)
        renderer.annotated_image(out/r['image_file'],r['predictions']['dino_hardware'],bundle['classes'],
            ident+' | METAL STRUCTURE HYPOTHESES | NOT VERIFIED STEEL',score_threshold=.3).save(out/('hardware_'+ident+'.jpg'),quality=93)
    import build_capability_upgrade_report
    extensions,extension_verification=build_capability_upgrade_report.build(out)
    bundle['verified_extensions']={
        'steelwork':extensions['steelwork']['summary'],
        'pole_top':extensions['pole_top']['summary'],
        'material':extensions['material'],
        'report':'upgrade/index.html',
        'verification':'upgrade/verification.json'
    }
    write_json(out/'data.json',bundle)
    (out/'index.html').write_text(html.replace('__DATA_JSON__',json.dumps(bundle,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c')))
    shutil.copyfile(ROOT/'UK_COMPONENT_ANNOTATION_GUIDE_EN.md',out/'UK_COMPONENT_ANNOTATION_GUIDE.md')
    shutil.copyfile(ROOT/'KEEN_CAPABILITY_DESIGN_EN.md',out/'RESULTS.md')
    verification={'status':'VERIFIED_V3_RAW_TENSORS_CROPS_AND_ENGLISH_UI','source_archive_files_verified':len(archived['files']),
        'images':27,'hardware_raw_prediction_checks':raw_checks,'exact_crop_pixel_checks':crop_checks,'encoded_views':encoded,
        'counts_are_accuracy':False,'training_started':False,'results_sha256':digest(run/'results.json'),
        'template_sha256':digest(template),'builder_sha256':digest(__file__),'html_sha256':digest(out/'index.html'),**bundle['capability_counts']}
    verification['extension_verification']=extension_verification
    write_json(out/'verification.json',verification);print(json.dumps(verification))


if __name__=='__main__':build()
