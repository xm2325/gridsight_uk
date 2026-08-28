"""Create segmentation labels from cached publisher polygons; no image acquisition."""
import json
import math
from pathlib import Path
from collections import Counter
from paper_material_demo import ROOT, load, sha, write

CONFIG=ROOT/'configs/component_masks_v1.json'


def segmentation_line(ref, width, height):
    points=ref['polygon']
    if len(points)<3:
        raise ValueError('Insufficient source polygon vertices')
    values=[]
    for p in points:
        x,y=float(p['x']),float(p['y'])
        if not math.isfinite(x+y) or not 0<=x<=width or not 0<=y<=height:
            raise ValueError('Source polygon outside raw image canvas')
        values.extend([x/width,y/height])
    return str(ref['class_id'])+' '+' '.join(f'{v:.10f}' for v in values)


def main():
    from PIL import Image
    cfg=load(CONFIG);source=ROOT/cfg['source_dataset'];out=ROOT/cfg['dataset']
    if out.exists():raise FileExistsError('Existing component-mask dataset; inspect before resumption')
    assert sha(source/'manifest.json')==cfg['source_manifest_sha256']
    assert sha(source/'selection_plan.json')==cfg['source_plan_sha256']
    m=load(source/'manifest.json');rows=[r for r in m['images'] if r['split'] in ['train','dev']]
    original={r['image_id']:r for r in load(source/'selection_plan.json')['images'] if r['split'] in ['train','dev']}
    assert Counter(r['split'] for r in rows)=={'train':320,'dev':80}
    assert not {r['circuit'] for r in rows if r['split']=='train'}&{r['circuit'] for r in rows if r['split']=='dev'}
    records=[]
    for r in rows:
        image=source/r['image_file'];assert sha(image)==r['sha256']
        with Image.open(image) as im:
            if im.size!=(r['width'],r['height']) or im.getexif().get(274) not in [None,1]:
                raise ValueError('Raw-pixel geometry/EXIF needs audit: '+r['image_id'])
        refs=[]
        for index,obj in enumerate(original[r['image_id']]['source_labels']['objects']):
            if obj['value'] not in cfg['classes']:continue
            ref=next(a for a in r['references'] if a['annotation_id']==f'{r["image_id"]}_{index}')
            assert ref['polygon']==obj['polygon'] and ref['class_name']==obj['value']
            refs.append(ref)
        assert len(refs)==len(r['references'])
        lines=[segmentation_line(a,r['width'],r['height']) for a in refs]
        records.append(dict(r,segmentation_lines=lines))
    out.mkdir(parents=True)
    for r in records:
        ip=out/r['image_file'];ip.parent.mkdir(parents=True,exist_ok=True)
        ip.symlink_to(Path('../../../epri_components_v1')/r['image_file'])
        assert ip.resolve()==(source/r['image_file']).resolve()
        lp=out/'labels'/r['split']/(r['image_id']+'.txt');lp.parent.mkdir(parents=True,exist_ok=True)
        lines=r.pop('segmentation_lines');lp.write_text('\n'.join(lines)+ ('\n' if lines else ''))
        r['label_file']=str(lp.relative_to(out));r['label_sha256']=sha(lp)
    (out/'train.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/dev\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(cfg['classes'])))
    manifest={'protocol_sha256':sha(CONFIG),'source_manifest_sha256':sha(source/'manifest.json'),
              'source_plan_sha256':sha(source/'selection_plan.json'),'classes':cfg['classes'],
              'license':m['license'],'license_url':m['license_url'],'source_page':m['source_page'],
              'images':records,'summary':{s:{n:sum(a['class_name']==n for r in records if r['split']==s for a in r['references']) for n in cfg['classes']} for s in ['train','dev']},
              'evaluation_images_used':False,'source_label_corrections':False}
    write(out/'manifest.json',manifest)
    print(json.dumps({'manifest_sha256':sha(out/'manifest.json'),'images':len(records),'summary':manifest['summary']}))


if __name__=='__main__':main()
