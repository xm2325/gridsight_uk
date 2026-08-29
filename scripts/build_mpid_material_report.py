"""Build an English, source-aware report for the bounded MPID material diagnostic."""
from __future__ import annotations

import hashlib,json,shutil
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/mpid_material_detector/v1_20260829'
PROSPECTIVE=ROOT/'runs/mpid_material_detector/prospective_770272_v1/results.json'
TWO_STAGE=ROOT/'runs/material_head/uk_two_stage_770272_v1/results.json'
V2=ROOT/'runs/material_head/v2_mpid_substation_20260829/results.json'
V2_UK=ROOT/'runs/material_head/v2_mpid_substation_20260829/uk_decisions.json'
RESOLUTION=ROOT/'runs/material_head/v2_resolution_8090535_20260829/results.json'
RESOLUTION_DECISIONS=ROOT/'runs/material_head/v2_resolution_8090535_20260829/decisions.json'
LOAO=ROOT/'runs/material_head/v2_loao_20260829/results.json'
OUT=ROOT/'runs/mpid_material_detector/report_v1_20260829/report'

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path:Path):return json.loads(path.read_text())
def write(path:Path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    if OUT.exists():shutil.rmtree(OUT)
    (OUT/'images').mkdir(parents=True)
    run=load(RUN/'results.json');source=load(ROOT/'data/external/uk_material_sources_v1/manifest.json')
    prospective=load(PROSPECTIVE);two_stage=load(TWO_STAGE);mpid=load(ROOT/'data/external/mpid_material_detector_v1/manifest.json')
    v2=load(V2);v2_uk=load(V2_UK);resolution=load(RESOLUTION);resolution_decisions=load(RESOLUTION_DECISIONS);loao=load(LOAO)
    if two_stage['status']!='COMPLETE' or two_stage['runtime']['job_id']!='940144':
        raise ValueError('Refuse to publish an incomplete or unexpected two-stage diagnostic')
    expected_jobs=[(v2,'940239'),(resolution,'940273'),(loao,'940286')]
    if any(run['status']!='COMPLETE' or run['runtime']['job_id']!=job for run,job in expected_jobs):
        raise ValueError('Refuse to publish incomplete or unexpected v2 diagnostics')
    old_refs=load(ROOT/'data/v4_annotations/material_reference_v44.json')
    references={record['record_id'].replace('POS_','uk_material_'):[{'id':box['id'],'material':box['material_task_label'],
      'source_specific_material':box['source_specific_material'],'xyxy':box['xyxy'],'status':'legacy_source_assisted_reference'} for box in record['boxes']]
      for record in old_refs['records']}
    freeze=load(ROOT/'data/uk_material_eval_v1/prospective_freeze_770272.json')
    references[freeze['record_id']]=[{**box,'status':'prospective_frozen_before_this_image_inference'} for box in freeze['references']]
    rows=[]
    initial_ids={row['record_id'] for row in run['uk_predictions']}
    for source_row in source['images']:
        record_id=source_row['record_id'];target=OUT/'images'/Path(source_row['image_file']).name
        shutil.copyfile(ROOT/source_row['image_file'],target)
        if record_id in initial_ids:predictions=load(RUN/'predictions'/f'{record_id}.json')['boxes'];prediction_scope='post_source_selection; no complete-assembly boxes were frozen before inference'
        elif record_id==freeze['record_id']:predictions=prospective['raw_predictions'];prediction_scope='one source-supported complete-assembly box frozen before inference on this image; sibling asset already observed'
        else:predictions=[];prediction_scope='not inferred'
        counts=Counter(box['class_name'] for box in predictions)
        rows.append({'record_id':record_id,'file':'images/'+target.name,'width':source_row['width'],'height':source_row['height'],
          'image_sha256':source_row['image_sha256'],'photo_page_url':source_row['photo_page_url'],'author':source_row['author'],
          'licence':source_row['licence'],'asset_group':source_row['asset_group'],'material_evidence':source_row['evidence'],
          'source_materials':source_row['materials'],'role':source_row['use'],'prediction_scope':prediction_scope,
          'predictions':predictions,'references':references.get(record_id,[]),'prediction_counts':dict(counts),
          'maximum_raw_score':max((box['raw_detector_score'] for box in predictions),default=None)})
    high_source=ROOT/resolution_decisions['source']['high_resolution']['image_file'];high_target=OUT/'images'/high_source.name
    shutil.copyfile(high_source,high_target)
    payload={'generated_from_verified_results':True,'run':{'job_id':run['runtime']['job_id'],'status':run['status'],
      'training_seconds':run['training_seconds'],'epochs':run['training_progress']['epochs_completed'],'dataset_counts':run['dataset_counts'],
      'metrics':run['mpid_development_metrics'],'metrics_scope':run['mpid_metrics_scope'],'checkpoint_sha256':run['checkpoint_sha256']},
      'mpid_audit':{'prepared_images':len(mpid['rows']),'exact_duplicate_groups':len(mpid['exact_duplicate_groups']),
        'cross_material_conflicts':mpid['cross_material_exact_conflicts_quarantined'],'origin_families':mpid['origin_family_count'],
        'split_warning':mpid['split_warning']},'prospective':prospective['prospective_diagnostic'],
      'two_stage':{'job_id':two_stage['runtime']['job_id'],'status':two_stage['status'],'classes':two_stage['classes'],
        'comparisons':two_stage['comparisons'],'claim_boundary':two_stage['claim_boundary'],
        'head_gradient_steps':two_stage['head_gradient_steps'],'encoder_gradient_steps':two_stage['encoder_gradient_steps']},
      'material_v2':{'job_id':v2['runtime']['job_id'],'development':v2['development_diagnostics'],'uk':v2['uk_diagnostics'],
        'thresholds':v2['thresholds'],'train_view_counts':v2['train_view_counts'],'mpid_crop_counts':v2['mpid_crop_counts'],
        'uk_images':v2_uk['images'],'claim_boundary':v2['claim_boundary']},
      'resolution':{'job_id':resolution['runtime']['job_id'],'diagnostics':resolution['diagnostics'],
        'record_id':resolution_decisions['record_id'],'image_file':'images/'+high_target.name,
        'width':resolution_decisions['source']['high_resolution']['width'],'height':resolution_decisions['source']['high_resolution']['height'],
        'boxes':resolution_decisions['boxes'],'comparisons':resolution_decisions['comparisons'],
        'source':{key:resolution_decisions['source'][key] for key in ('commons_file_page_url','source_page_url','author','licence','licence_url','material_evidence')},
        'claim_boundary':resolution_decisions['claim_boundary']},
      'loao':{'job_id':loao['runtime']['job_id'],'aggregate':loao['aggregate_diagnostics'],'folds':loao['folds'],
        'claim_boundary':loao['claim_boundary']},'images':rows,
      'interpretation':{
        'direct_detector':'Useful as an additional proposal generator, but not a deployable UK material decision system.',
        'prospective_result':'The frozen porcelain string was barely localised at IoU 0.504, classified as polymer/composite, and rejected to unknown only because the matching score was below the fixed gate.',
        'critical_failure':'A shorter partial box on the same porcelain string received raw polymer/composite score 0.730 and passed the per-box gate. Confidence alone cannot detect incomplete assemblies.',
        'two_stage_result':'The existing glass / porcelain / other SigLIP2 head classified the frozen complete porcelain assembly and both detector fragments as glass. Full-assembly cropping did not repair material transfer.',
        'v2_result':'The four-class MPID + Substation15 head reached 98.5% accepted accuracy on internal development but only 50.0% on thirteen previously observed UK boxes.',
        'resolution_result':'Replacing the 640×480 derivative with the licensed 2560×1920 original left accepted accuracy at 50.0%; two porcelain strings were still accepted as glass.',
        'loao_result':'Three fixed leave-one-asset-out adaptation folds reduced accepted accuracy to 40.0%; sparse UK last-layer adaptation is rejected.',
        'next_architecture':'Use material-agnostic complete-string proposals, then train an encoder adapter on a larger source-supported UK asset pool with asset-separated folds. Require native pixels, completeness, tight/context agreement, target-domain support and calibrated abstention; otherwise return unknown.'}}
    data=OUT/'data.json';write(data,payload)
    template=(ROOT/'templates/mpid_material_report.html').read_text();(OUT/'index.html').write_text(template.replace('__DATA__',json.dumps(payload).replace('</','<\\/')))
    verification={'status':'VERIFIED_REPORT_BUILD','source_result_sha256':sha(RUN/'results.json'),'prospective_result_sha256':sha(PROSPECTIVE),
      'two_stage_result_sha256':sha(TWO_STAGE),'two_stage_job_id':two_stage['runtime']['job_id'],
      'v2_result_sha256':sha(V2),'v2_uk_sha256':sha(V2_UK),'resolution_result_sha256':sha(RESOLUTION),
      'resolution_decisions_sha256':sha(RESOLUTION_DECISIONS),'loao_result_sha256':sha(LOAO),
      'prospective_freeze_sha256':sha(ROOT/'data/uk_material_eval_v1/prospective_freeze_770272.json'),'data_sha256':sha(data),
      'image_count':len(rows),'language':'English','raw_predictions_preserved':True,'pseudo_labels_used_as_truth':False}
    write(OUT/'verification.json',verification);print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
