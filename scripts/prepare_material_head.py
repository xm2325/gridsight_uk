"""Prepare source-labelled crop classification, preserving automatic/oracle separation."""
import hashlib
import io
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
import zipfile

from audit_substation15 import ARCHIVE
from paper_material_demo import ROOT, load, sha, write, extent
from prepare_substation_material import duplicate_components, polygon_references, select_rows

CONFIG=ROOT/'configs/material_head_v1.json'


def intersection(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def native_ok(box,cfg):
    w,h=box[2]-box[0],box[3]-box[1]
    return min(w,h)>=cfg['minimum_native_side'] and w*h>=cfg['minimum_native_area']


def negative_boxes(annotation, refs, key, maximum=2, source_classes=("Background",)):
    from PIL import Image, ImageDraw
    found=[]; W,H=annotation['imageWidth'],annotation['imageHeight']
    for index,s in enumerate(annotation['shapes']):
        if s['label'] not in source_classes: continue
        mask=Image.new('L',(W,H));ImageDraw.Draw(mask).polygon([tuple(p) for p in s['points']],fill=255)
        # Same global 64px grid, but points outside this polygon's bounds cannot qualify.
        xs,ys=zip(*s['points'])
        xlo=max(0,math.floor(min(xs)/64)*64); ylo=max(0,math.floor(min(ys)/64)*64)
        xhi=min(W-63,math.ceil(max(xs))+1); yhi=min(H-63,math.ceil(max(ys))+1)
        coords=[(x,y) for y in range(ylo,yhi,64) for x in range(xlo,xhi,64)]
        coords.sort(key=lambda p:hashlib.sha256(f'{key}:{index}:{p}'.encode()).hexdigest())
        for x,y in coords:
            b=[x,y,x+64,y+64]
            if any(intersection(b,r['box'])>0 for r in refs):continue
            if mask.crop(tuple(b)).getextrema()!=(255,255):continue
            if any(intersection(b,old['box'])>0 for old in found):continue
            found.append({'annotation_index':index,'class_id':2,'publisher_label':s['label'], 'box':b,
                          'derived_negative':True,'polygon':s['points']})
            if len(found)>=maximum:return found
    return found


def balanced_crops(rows, maximum, seed):
    groups=defaultdict(list)
    for row in rows:groups[row['image_id']].append(row)
    queues=[]
    for key in sorted(groups,key=lambda k:hashlib.sha256((seed+k).encode()).hexdigest()):
        queues.append(deque(sorted(groups[key],key=lambda r:hashlib.sha256((seed+r['id']).encode()).hexdigest())))
    selected=[]
    while len(selected)<maximum and any(queues):
        for q in queues:
            if q and len(selected)<maximum:selected.append(q.popleft())
    return selected


def main():
    from PIL import Image
    cfg=load(CONFIG);out=ROOT/cfg['dataset'];audit=load(ROOT/cfg['orientation_audit'])
    if out.exists():raise FileExistsError('Existing material-head dataset: inspect, do not rebuild')
    assert sha(ARCHIVE)==cfg['archive_sha256']==audit['archive_sha256']
    rows=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        for a in audit['rows']:
            if not a['eligible']:continue
            row=dict(a);row['split']='development' if row['capture_group'] in cfg['development_groups'] else 'train'
            annotation=json.loads(z.read(row['archive_annotation']))
            row['references']=polygon_references(annotation,cfg['publisher_classes'])
            rows.append(row)
        clusters,edges=duplicate_components(rows,4,.1); representatives=[];quarantine=[];drops=[]
        for cluster in clusters:
            if len({r['split'] for r in cluster})>1:
                quarantine.extend(r['id'] for r in cluster);continue
            ordered=sorted(cluster,key=lambda r:r['id']); r=dict(ordered[0]);r['duplicate_cluster']=min(x['id'] for x in cluster)
            representatives.append(r);drops.extend(x['id'] for x in ordered[1:])
        selected=[]
        for split,maximum in cfg['source_images_max'].items():
            selected+=select_rows([r for r in representatives if r['split']==split],maximum,cfg['seed'])
        assert not {r['capture_group'] for r in selected if r['split']=='train'}&{r['capture_group'] for r in selected if r['split']=='development'}
        candidates=[]
        for row in selected:
            a=json.loads(z.read(row['archive_annotation'])); refs=row['references']
            for c in [0,1]:
                eligible=[r for r in refs if r['class_id']==c and native_ok(r['box'],cfg)]
                eligible.sort(key=lambda r:hashlib.sha256(f'{cfg["seed"]}:{row["id"]}:{r["annotation_index"]}'.encode()).hexdigest())
                for r in eligible[:cfg['max_material_crops_per_image_class']]:
                    candidates.append(dict(r,image_id=row['id'],split=row['split'],id=f'{row["id"]}_{c}_{r["annotation_index"]}',derived_negative=False))
            for i,r in enumerate(negative_boxes(a,refs,row['id'],cfg['max_negative_crops_per_image'],cfg['negative_source_classes'])):
                candidates.append(dict(r,image_id=row['id'],split=row['split'],id=f'{row["id"]}_2_{i}'))
        crops=[]
        for split,limits in cfg['crop_limits'].items():
            for c,maximum in enumerate(limits):
                crops+=balanced_crops([r for r in candidates if r['split']==split and r['class_id']==c],maximum,cfg['seed'])
        for split in cfg['crop_limits']:
            print({'preflight_split':split,'class_counts':dict(Counter(r['class_id'] for r in crops if r['split']==split))},flush=True)
            assert all(sum(r['split']==split and r['class_id']==c for r in crops)>=30 for c in range(3))
        out.mkdir(parents=True)
        by_image=defaultdict(list)
        for r in crops:by_image[r['image_id']].append(r)
        image_records=[]
        for source in selected:
            if source['id'] not in by_image:continue
            content=z.read(source['archive_image']);assert hashlib.sha256(content).hexdigest()==source['image_sha256']
            im=Image.open(io.BytesIO(content)).convert('RGB'); im.info.clear()
            annotation_bytes=z.read(source['archive_annotation']);assert hashlib.sha256(annotation_bytes).hexdigest()==source['annotation_sha256']
            af=f'annotations/{source["id"]}.json';(out/af).parent.mkdir(exist_ok=True);(out/af).write_bytes(annotation_bytes)
            rec={k:v for k,v in source.items() if k!='references'};rec['annotation_file']=af
            if source['split']=='development':
                f=f'images/{source["id"]}.png';(out/f).parent.mkdir(exist_ok=True);im.save(out/f);rec['display_file']=f;rec['display_sha256']=sha(out/f)
            image_records.append(rec)
            for r in by_image[source['id']]:
                r['views']={}
                for view,pad in [('tight',0),('context',cfg['context_padding'])]:
                    box=extent(r['box'],im.width,im.height,pad);crop=im.crop(box);crop.info.clear()
                    f=f'crops/{r["split"]}/{r["id"]}_{view}.png';(out/f).parent.mkdir(parents=True,exist_ok=True);crop.save(out/f)
                    r['views'][view]={'file':f,'sha256':sha(out/f),'box':box,'pixel_sha256':hashlib.sha256(crop.tobytes()).hexdigest()}
        summary={'source_images':dict(Counter(r['split'] for r in image_records)),
                 'crops':{s:{cfg['classes'][c]:sum(r['split']==s and r['class_id']==c for r in crops) for c in range(3)} for s in cfg['crop_limits']},
                 'orientation_counts':audit['counts'],'quarantined':len(quarantine),'duplicate_drops':len(drops)}
        write(out/'manifest.json',{'protocol_sha256':sha(CONFIG),'orientation_audit_sha256':sha(ROOT/cfg['orientation_audit']),
              'archive_sha256':cfg['archive_sha256'],'images':image_records,'crops':crops,'summary':summary,
              'automatic_target_annotations_used_for_training':False,'negative_labels_are_source_derived':True})
        print(json.dumps({'summary':summary,'manifest_sha256':sha(out/'manifest.json')},indent=2))


if __name__=='__main__':main()
