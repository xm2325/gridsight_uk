"""Prepare a deterministic, exact-deduplicated MPID detector dataset on Roihu."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

from audit_mpid_archives import IMAGE_SUFFIXES,origin_key

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/mpid_material_detector_v1.json'
SOURCE=ROOT/'data/external/mpid_v1'

def sha_bytes(payload:bytes)->str:return hashlib.sha256(payload).hexdigest()
def sha_file(path:Path)->str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def rank(seed:int,value:str)->str:return sha_bytes(f'{seed}|{value}'.encode())
def write(path:Path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def remap_label(text:str,class_id:int)->tuple[str,int]:
    output=[]
    for line in text.splitlines():
        if not line.strip():continue
        fields=line.split()
        if len(fields)!=5 or fields[0]!='0':raise ValueError(f'Unexpected MPID label row: {line[:100]}')
        values=[float(value) for value in fields[1:]];x,y,w,h=values
        if not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1 and x-w/2>=-1e-6 and y-h/2>=-1e-6 and x+w/2<=1+1e-6 and y+h/2<=1+1e-6):
            raise ValueError(f'Invalid MPID box: {values}')
        output.append(str(class_id)+' '+' '.join(f'{value:.10f}' for value in values))
    return '\n'.join(output)+('\n' if output else ''),len(output)

def split_origins(origins:set[str],fraction:float,seed:int)->set[str]:
    ordered=sorted(origins,key=lambda value:rank(seed,'origin|'+value))
    count=max(1,min(len(ordered)-1,math.ceil(len(ordered)*fraction)))
    return set(ordered[:count])

def prepare():
    cfg=json.loads(CONFIG.read_text());out=ROOT/cfg['dataset']
    if out.exists():raise FileExistsError(f'Existing prepared dataset: {out}; verify rather than overwrite')
    archive_paths={name:SOURCE/name for name in cfg['source_archives']}
    for name,path in archive_paths.items():
        if sha_file(path)!=cfg['source_archives'][name]:raise ValueError(f'Archive hash differs: {name}')
    material_id={name:index for index,name in enumerate(cfg['classes'])}
    archive_material={'glass':'glass','porcelain':'porcelain_ceramic','composite':'polymer_composite'}
    candidates=[]
    for archive_name,path in archive_paths.items():
        material=archive_material[archive_name.split('_',1)[0]];class_id=material_id[material]
        with zipfile.ZipFile(path) as archive:
            names=set(archive.namelist())
            for image_name in sorted(name for name in names if '/images/' in name and Path(name).suffix.lower() in IMAGE_SUFFIXES):
                label_name=str(Path(image_name.replace('/images/','/labels/')).with_suffix('.txt'))
                if label_name not in names:raise ValueError(f'Missing label: {label_name}')
                image=archive.read(image_name);label,count=remap_label(archive.read(label_name).decode(),class_id)
                candidates.append({'archive':archive_name,'material':material,'class_id':class_id,'image_name':image_name,
                  'label_name':label_name,'origin':origin_key(image_name),'image_sha256':sha_bytes(image),'image':image,
                  'label':label,'instances':count,'suffix':Path(image_name).suffix.lower()})
    by_sha=defaultdict(list)
    for row in candidates:by_sha[row['image_sha256']].append(row)
    conflicts=[];deduplicated=[];duplicates=[]
    for digest,rows in sorted(by_sha.items()):
        if len({row['material'] for row in rows})>1:
            conflicts.append({'sha256':digest,'members':[row['image_name'] for row in rows]});continue
        rows.sort(key=lambda row:(row['archive'],row['image_name']))
        deduplicated.append(rows[0])
        if len(rows)>1:duplicates.append({'sha256':digest,'kept':rows[0]['image_name'],'removed':[row['image_name'] for row in rows[1:]]})
    origins={row['origin'] for row in deduplicated};dev=split_origins(origins,cfg['split']['development_fraction'],cfg['split']['seed'])
    rows=[]
    for row in sorted(deduplicated,key=lambda item:(item['material'],item['image_name'])):
        split='development' if row['origin'] in dev else 'train'
        stem=f"{row['material']}_{row['image_sha256'][:16]}"
        image_rel=Path('images')/split/(stem+row['suffix']);label_rel=Path('labels')/split/(stem+'.txt')
        image_path=out/image_rel;label_path=out/label_rel
        image_path.parent.mkdir(parents=True,exist_ok=True);label_path.parent.mkdir(parents=True,exist_ok=True)
        image_path.write_bytes(row.pop('image'));label_path.write_text(row.pop('label'))
        rows.append({**row,'split':split,'image_file':str(image_rel),'label_file':str(label_rel),
          'label_sha256':sha_file(label_path)})
    counts={split:{'images':sum(row['split']==split for row in rows),'instances':sum(row['instances'] for row in rows if row['split']==split),
      'by_material':dict(Counter(row['material'] for row in rows if row['split']==split))} for split in ('train','development')}
    for split in counts:
        if set(counts[split]['by_material'])!=set(cfg['classes']):raise ValueError(f'Material missing from {split}')
    manifest={'version':'mpid-material-detector-v1','created_at':datetime.now(timezone.utc).isoformat(),
      'config_sha256':sha_file(CONFIG),'source_archive_sha256':cfg['source_archives'],'rows':rows,'counts':counts,
      'exact_duplicate_groups':duplicates,'cross_material_exact_conflicts_quarantined':conflicts,
      'origin_family_count':len(origins),'development_origin_count':len(dev),'development_origins':sorted(dev),
      'split_warning':cfg['split']['warning'],'evaluation_scope':'MPID internal diagnostic only; not UK accuracy'}
    write(out/'manifest.json',manifest)
    dataset={'path':str(out),'train':'images/train','val':'images/development','names':cfg['classes']}
    write(out/'dataset.yaml',dataset)
    return manifest

def verify():
    cfg=json.loads(CONFIG.read_text());out=ROOT/cfg['dataset'];manifest=json.loads((out/'manifest.json').read_text())
    if manifest['config_sha256']!=sha_file(CONFIG):raise ValueError('Prepared dataset config differs')
    origins=defaultdict(set);hashes=set()
    for row in manifest['rows']:
        image=out/row['image_file'];label=out/row['label_file']
        if sha_file(image)!=row['image_sha256'] or sha_file(label)!=row['label_sha256']:raise ValueError('Prepared bytes differ')
        if row['image_sha256'] in hashes:raise ValueError('Exact duplicate survived')
        hashes.add(row['image_sha256']);origins[row['origin']].add(row['split'])
    leaked={key:value for key,value in origins.items() if len(value)>1}
    if leaked:raise ValueError(f'Origin leakage: {list(leaked)[:3]}')
    return manifest

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    result=verify() if args.verify_only else prepare()
    print(json.dumps({'status':'VERIFIED' if args.verify_only else 'PREPARED','counts':result['counts'],'duplicates':len(result['exact_duplicate_groups'])},indent=2))
