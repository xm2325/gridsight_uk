"""Audit MPID ZIPs without extracting or running a model."""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/external/mpid_v1'
EXPECTED={
 'glass':{'images':1612,'instances':3773},
 'porcelain':{'images':2549,'instances':3172},
 'composite':{'images':646,'instances':905},
}
IMAGE_SUFFIXES={'.jpg','.jpeg','.png','.bmp','.webp'}

def origin_key(name:str)->str:
    base=Path(name).name
    base=re.sub(r'\.rf\.[0-9a-f]+(?=\.[^.]+$)','',base,flags=re.I)
    stem=Path(base).stem
    return re.sub(r'_(?:jpg|jpeg|png|bmp)$','',stem,flags=re.I).lower()

def split_of(name:str)->str:
    parts=Path(name).parts
    for split in ('train','valid','val','test'):
        if split in parts:return 'valid' if split=='val' else split
    return 'other'

def inspect_archive(path:Path,material:str)->dict:
    image_rows=[];labels={};invalid=[];class_ids=Counter();instances=0
    with zipfile.ZipFile(path) as archive:
        names=[n for n in archive.namelist() if not n.endswith('/')]
        yaml={n:archive.read(n).decode('utf-8','replace') for n in names if n.endswith('data.yaml')}
        readmes={n:archive.read(n).decode('utf-8','replace') for n in names if 'README' in Path(n).name}
        for name in names:
            suffix=Path(name).suffix.lower()
            if suffix in IMAGE_SUFFIXES and '/images/' in name:
                payload=archive.read(name);image_rows.append({'name':name,'split':split_of(name),'origin_key':origin_key(name),
                  'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
            elif suffix=='.txt' and '/labels/' in name:
                text=archive.read(name).decode('utf-8','replace');key=(split_of(name),Path(name).stem);labels[key]=text
                for lineno,line in enumerate(text.splitlines(),1):
                    if not line.strip():continue
                    fields=line.split();instances+=1
                    if len(fields)!=5:
                        invalid.append({'file':name,'line':lineno,'reason':'field_count','value':line[:120]});continue
                    try:cls=int(fields[0]);box=[float(x) for x in fields[1:]]
                    except ValueError:
                        invalid.append({'file':name,'line':lineno,'reason':'parse','value':line[:120]});continue
                    class_ids[cls]+=1;x,y,w,h=box
                    if not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1 and x-w/2>=-1e-6 and y-h/2>=-1e-6 and x+w/2<=1+1e-6 and y+h/2<=1+1e-6):
                        invalid.append({'file':name,'line':lineno,'reason':'normalised_bounds','value':box})
    image_keys={(r['split'],Path(r['name']).stem) for r in image_rows}
    label_keys=set(labels)
    origin_splits=defaultdict(set)
    origin_members=defaultdict(list)
    for row in image_rows:
        origin_splits[row['origin_key']].add(row['split']);origin_members[row['origin_key']].append(row['name'])
    split_leak={k:sorted(v) for k,v in origin_splits.items() if len(v)>1}
    origin_duplicates={k:v for k,v in origin_members.items() if len(v)>1}
    sha_groups=defaultdict(list)
    for row in image_rows:sha_groups[row['sha256']].append(row['name'])
    exact={k:v for k,v in sha_groups.items() if len(v)>1}
    return {'archive':path.name,'archive_bytes':path.stat().st_size,'material':material,
      'images':len(image_rows),'instances':instances,'split_images':dict(Counter(r['split'] for r in image_rows)),
      'class_ids':dict(class_ids),'unpaired_images':len(image_keys-label_keys),'orphan_labels':len(label_keys-image_keys),
      'invalid_label_rows':invalid,'origin_family_count':len(origin_splits),'origin_families_across_splits':split_leak,
      'origin_duplicate_families':origin_duplicates,
      'exact_duplicate_groups':exact,'yaml':yaml,'readmes':readmes,
      'expected':EXPECTED[material],'paper_count_match':len(image_rows)==EXPECTED[material]['images'] and instances==EXPECTED[material]['instances']}

def main():
    archive_by_material={p.name.split('_')[0]:p for p in DATA.glob('*_MPID-*.zip')}
    if set(archive_by_material)!=set(EXPECTED):raise RuntimeError(f'Expected three material archives, got {sorted(archive_by_material)}')
    rows=[inspect_archive(archive_by_material[m],m) for m in ('glass','porcelain','composite')]
    all_names=defaultdict(list);all_sha=defaultdict(list)
    for row in rows:
        for origin,splits in row['origin_families_across_splits'].items():all_names[origin].append({'material':row['material'],'splits':splits})
        for sha,names in row['exact_duplicate_groups'].items():all_sha[sha].extend(f"{row['material']}:{n}" for n in names)
    report={'status':'PASS' if all(not r['invalid_label_rows'] and not r['unpaired_images'] and not r['orphan_labels'] for r in rows) else 'FAIL',
      'created_at':datetime.now(timezone.utc).isoformat(),'archives_extracted':False,'model_inference_performed':False,
      'archives':rows,'totals':{'images':sum(r['images'] for r in rows),'instances':sum(r['instances'] for r in rows)},
      'split_warning':'MPID source dataset identity is not explicit in the flattened Roboflow export. Its built-in split is not accepted as evidence of source- or asset-independent generalisation.',
      'recommended_use':'Training-source candidate after whole-assembly visual sampling and duplicate quarantine; evaluate on separately sourced datasets and frozen UK evidence.',
      'training_authorized':False,'next_gate':'inspect representative full-resolution images and origin families; build a source-external evaluation protocol before training'}
    (DATA/'archive_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'totals':report['totals'],
      'materials':{r['material']:{'images':r['images'],'instances':r['instances'],'split_families':len(r['origin_families_across_splits']),
        'exact_duplicate_groups':len(r['exact_duplicate_groups']),'invalid':len(r['invalid_label_rows']),'paper_count_match':r['paper_count_match']} for r in rows}},indent=2))

if __name__=='__main__':main()
