"""Combine UK provenance-only and material-evidenced pools without copying pixels."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/external/uk_source_pool_v1'

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    qualitative_path=ROOT/'data/external/uk_distribution_pilot_v1/manifest.json'
    material_path=ROOT/'data/external/uk_material_sources_v1/manifest.json'
    qualitative=json.loads(qualitative_path.read_text());material=json.loads(material_path.read_text())
    rows=[]
    for row in qualitative['images']:
        rows.append({'record_id':row['image_id'],'pool':'provenance_only','image_file':str((qualitative_path.parent/row['image_file']).relative_to(ROOT)),
          'image_sha256':row['sha256'],'photo_page_url':row['geograph_page'],'source_page_url':row['source_page'],
          'author':row['author'],'licence':row['license'],'asset_group':f"geograph_{row['geograph_id']}",
          'material_evidence':None,'ground_truth_status':'NONE','prior_model_use':'qualitative_development'})
    for row in material['images']:
        rows.append({'record_id':row['record_id'],'pool':'source_evidenced_material_candidate','image_file':row['image_file'],
          'image_sha256':row['image_sha256'],'photo_page_url':row['photo_page_url'],'source_page_url':row['photo_page_url'],
          'source_page_sha256_at_acquisition':row['source_page_sha256_at_acquisition'],'author':row['author'],'licence':row['licence'],
          'asset_group':row['asset_group'],'material_evidence':{'classes':row['materials'],'excerpt':row['evidence']},
          'ground_truth_status':row['material_truth_status'],'prior_model_use':row['use']})
    ids=[r['record_id'] for r in rows];hashes=[r['image_sha256'] for r in rows]
    if len(ids)!=len(set(ids)) or len(hashes)!=len(set(hashes)):raise RuntimeError('Duplicate ID or exact image across UK pools')
    new_primary=[r['record_id'] for r in rows if r['prior_model_use'].startswith('new_candidate')]
    report={'version':'uk-source-pool-v1','created_at':datetime.now(timezone.utc).isoformat(),
      'count':len(rows),'counts':{'provenance_only':len(qualitative['images']),'source_evidenced_material_candidate':len(material['images']),
        'new_primary_material_candidates':len(new_primary)},'new_primary_material_candidates':new_primary,'records':rows,
      'input_manifests':{str(qualitative_path.relative_to(ROOT)):sha(qualitative_path),str(material_path.relative_to(ROOT)):sha(material_path)},
      'semantics':'Image provenance does not imply material truth. Only source-evidenced candidates may proceed to box review; model outputs never become evaluation truth.',
      'model_inference_performed':False,'split_frozen':False}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'count':report['count'],'counts':report['counts']}))

if __name__=='__main__':main()
