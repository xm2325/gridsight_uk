"""Create deterministic boxed samples for MPID annotation-unit review."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont,ImageOps

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/external/mpid_v1'
OUT=DATA/'unit_samples'

def label_name(image_name:str)->str:
    return image_name.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt'

def boxes(text:str):
    return [[float(v) for v in line.split()[1:]] for line in text.splitlines() if line.strip()]

def features(rows):
    areas=[b[2]*b[3] for b in rows];aspects=[b[2]/b[3] for b in rows]
    return {'instances':len(rows),'max_area':max(areas),'min_area':min(areas),'min_aspect':min(aspects),'max_aspect':max(aspects)}

def choose(rows):
    rules=[('smallest_box',lambda r:r['feature']['min_area']),('small_whole_extent',lambda r:r['feature']['max_area']),
      ('median_extent',lambda r:abs(r['feature']['max_area']-.08)),('large_extent',lambda r:-r['feature']['max_area']),
      ('most_instances',lambda r:-r['feature']['instances']),('widest',lambda r:r['feature']['min_aspect']),
      ('tallest',lambda r:-r['feature']['max_aspect']),('hash_sample',lambda r:r['sha256'])]
    selected=[];used=set()
    for rule,key in rules:
        for row in sorted(rows,key=key):
            if row['sha256'] not in used:
                used.add(row['sha256']);selected.append({**row,'selection_rule':rule});break
    return selected

def sheet(material,selected,archive):
    tw,th,label_h=600,450,70;canvas=Image.new('RGB',(tw*2,(th+label_h)*4),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.load_default()
    for i,row in enumerate(selected):
        payload=archive.read(row['name']);image=Image.open(io.BytesIO(payload)).convert('RGB');w,h=image.size
        boxed=image.copy();bd=ImageDraw.Draw(boxed)
        for x,y,bw,bh in row['boxes']:
            bd.rectangle(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h),outline='#ff2d7a',width=max(2,w//500))
        fitted=ImageOps.contain(boxed,(tw,th));x0=(i%2)*tw+(tw-fitted.width)//2;y0=(i//2)*(th+label_h)+(th-fitted.height)//2
        canvas.paste(fitted,(x0,y0));draw.text(((i%2)*tw+7,(i//2)*(th+label_h)+th+5),
          f"{material} | {row['selection_rule']} | {w}x{h} | {len(row['boxes'])} boxes\n{Path(row['name']).name[:70]}",fill='black',font=font)
    path=OUT/f'{material}_unit_samples.jpg';canvas.save(path,quality=92);return path

def main():
    OUT.mkdir(parents=True,exist_ok=True);report={'status':'SAMPLES_ONLY','model_inference_performed':False,'materials':{}}
    for material in ('glass','porcelain','composite'):
        path=next(DATA.glob(f'{material}_MPID-*.zip'))
        with zipfile.ZipFile(path) as archive:
            names=[n for n in archive.namelist() if '/images/' in n and Path(n).suffix.lower() in {'.jpg','.jpeg','.png'}]
            rows=[]
            for name in names:
                payload=archive.read(name);sha=hashlib.sha256(payload).hexdigest();bs=boxes(archive.read(label_name(name)).decode())
                rows.append({'name':name,'sha256':sha,'boxes':bs,'feature':features(bs)})
            selected=choose(rows);sheet_path=sheet(material,selected,archive)
        report['materials'][material]={'archive':path.name,'sheet':str(sheet_path.relative_to(ROOT)),
          'samples':[{k:v for k,v in row.items() if k!='boxes'}|{'boxes':row['boxes']} for row in selected]}
    report['source']='https://zenodo.org/records/14604384';report['next_gate']='visual annotation-unit decision; samples do not validate the full archive'
    (OUT/'sample_manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({m:len(v['samples']) for m,v in report['materials'].items()}))

if __name__=='__main__':main()
