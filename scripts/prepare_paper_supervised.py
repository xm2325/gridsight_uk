"""Prepare one explicitly bounded public-label training pilot, never target labels."""
import json
import shutil
import zipfile
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from paper_material_demo import ROOT,RUN,ZIP_SHA,load,write,sha

DATA=ROOT/'data/external/uvinsdet_supervised_demo_v1'
CONFIG=ROOT/'configs/paper_supervised_demo_v1.json'


def main():
    if DATA.exists() or CONFIG.exists(): raise FileExistsError('Inspect prepared manifest; do not overwrite')
    archive=ROOT/'data/external/uvinsdet_cache/UVInsDet_v1.0.0.zip';assert sha(archive)==ZIP_SHA
    DATA.mkdir(parents=True);rows=[];by_id=defaultdict(list)
    with zipfile.ZipFile(archive) as z:
        coco=json.loads(z.read('UVInsDet/data/annotations/coco/instances_train.json'))
        assert len(coco['images'])==398
        for a in coco['annotations']:by_id[a['image_id']].append(a)
        for source in coco['images']:
            name=source['file_name'];annotations=by_id[source['id']]
            for replicate in range(20 if any(a['category_id']==2 for a in annotations) else 1):
                key=Path(name).stem+(f'_weight{replicate:02d}' if replicate else '')
                f=DATA/'images/train'/f'{key}.jpg';f.parent.mkdir(parents=True,exist_ok=True)
                f.write_bytes(z.read('UVInsDet/data/images/train/'+name))
                lines=[]
                for a in annotations:
                    x,y,w,h=a['bbox'];W=source['width'];H=source['height']
                    x1=max(0,min(W,x));y1=max(0,min(H,y));x2=max(0,min(W,x+w));y2=max(0,min(H,y+h))
                    if x2<=x1 or y2<=y1:raise ValueError('Degenerate source bbox')
                    lines.append(f'{a["category_id"]-1} {(x1+x2)/2/W:.9f} {(y1+y2)/2/H:.9f} {(x2-x1)/W:.9f} {(y2-y1)/H:.9f}')
                label=DATA/'labels/train'/f'{key}.txt';label.parent.mkdir(parents=True,exist_ok=True);label.write_text('\n'.join(lines)+'\n' if lines else '')
                rows.append({'source_file':name,'source_image_id':source['id'],'publisher_split':'train','replicate':replicate,
                    'image_file':str(f.relative_to(DATA)),'image_sha256':sha(f),'label_file':str(label.relative_to(DATA)),
                    'label_sha256':sha(label),'source_annotations':annotations})
        for f in ['LICENSE','CITATION.cff','README.md']:
            (DATA/f).write_bytes(z.read('UVInsDet/'+f))
    target=load(RUN/'manifest.json')
    target_hashes={r['sha256'] for r in target['images'] if r['role']=='target'}
    assert not target_hashes.intersection({r['image_sha256'] for r in rows})
    refs={Path(r['file']).name for r in target['images'] if r['role']=='reference'}
    validation=[r['image_file'] for r in rows if r['source_file'] in refs and r['replicate']==0]
    write(DATA/'manifest.json',{'unique_source_images':398,'weighted_entries':len(rows),'porcelain_source_images':1,
        'source_zip_sha256':ZIP_SHA,'annotation_unit':'Original polygon bounding boxes. Porcelain sheds remain separate; no union targets.',
        'images':rows,'sanity_validation_images':validation,'validation_role':'In-sample training sanity ONLY; never accuracy or checkpoint selection.',
        'exact_train_target_image_hash_overlap':False,'near_duplicate_or_asset_independence_not_established':True})
    cfg={'created_at':datetime.now(timezone.utc).isoformat(),'run':'runs/paper_material_supervised/v1_20260828',
        'dataset':str(DATA.relative_to(ROOT)),'manifest_sha256':sha(DATA/'manifest.json'),
        'source_demo':str(RUN.relative_to(ROOT)),'source_demo_manifest_sha256':sha(RUN/'manifest.json'),
        'classes':['glass','porcelain'],'pretrained':'weights/yoloe-26m-seg.pt',
        'pretrained_sha256':'585f5ec9028fd358035da8d860c27c56be285a795cba2076fba536a4391c2c83',
        'yaml':'yoloe-26m.yaml','epochs':10,'imgsz':960,'batch':16,'seed':17,
        'porcelain_weighting':'20 appearances of one original image, not 20 independent images.',
        'checkpoint_selection':'Fixed final epoch, not selected against target images or material accuracy.',
        'authorization':'User permits small academic/public labelled-data demo and requests Roihu gputest. One 10-epoch supervised pilot following recorded transfer failures; no automatic repeat.',
        'assessment':'All eight previously inspected demo inputs are development/demo images. This is not a fresh holdout or UK validation.',
        'inference':{'imgsz':1280,'conf':.05,'iou':.5,'max_det':100,'augment':False}}
    write(CONFIG,cfg);print(json.dumps({'config':str(CONFIG.relative_to(ROOT)),'sources':398,'weighted_entries':len(rows),'manifest_sha256':cfg['manifest_sha256']}))


if __name__=='__main__':main()
