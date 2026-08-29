"""Acquire source-evidenced UK material images without running a model."""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/external/uk_material_sources_v1'
UA='GridSight-UK-material-source-audit/1.0 (+https://github.com/xm2325/gridsight_uk)'

# Evidence is copied only as a short identifying excerpt. The source page remains authoritative.
SOURCES=[
 {'photo_id':'2952166','author':'Peter Facey','country':'England','asset_group':'denmead_fawley_lovedoan_20120516',
  'materials':['glass','porcelain_ceramic'],'evidence':'ceramic insulators on its left side, and green glass ones on its right',
  'use':'legacy_adaptive_development_not_new_evaluation'},
 {'photo_id':'8090535','author':'Daniel Beardsmore','country':'England','asset_group':'hitchin_l6_d30_20250510',
  'materials':['glass','porcelain_ceramic'],'evidence':'blue-tinted glass insulators rather than the brown-glazed porcelain insulators used originally',
  'use':'legacy_material_reference_not_new_evaluation'},
 {'photo_id':'3209028','author':'Peter Facey','country':'England','asset_group':'mayles_lane_4ye030_20120524',
  'materials':['glass'],'evidence':'These insulators are made of toughened glass.',
  'use':'new_candidate_requires_pixel_and_box_review'},
 {'photo_id':'3208894','author':'Peter Facey','country':'England','asset_group':'mayles_lane_4ye030_20120524',
  'materials':['porcelain_ceramic'],'evidence':'There are 22 porcelain insulators in each of the two strings.',
  'use':'new_candidate_grouped_with_3209028'},
 {'photo_id':'3809215','author':'Peter Facey','country':'England','asset_group':'park_hills_wood_20140110',
  'materials':['glass'],'evidence':'showing up the glass insulators',
  'use':'new_candidate_requires_pixel_and_box_review'},
 {'photo_id':'7880016','author':'DS Pugh','country':'England','asset_group':'marazion_old_telegraph_20240903',
  'materials':['porcelain_ceramic'],'evidence':'a fairly old pole, with ceramic insulators in place',
  'use':'auxiliary_telegraph_morphology_not_primary_power_evaluation'},
]

def digest(payload:bytes)->str:return hashlib.sha256(payload).hexdigest()

def fetch(url:str)->bytes:
    errors=[]
    for attempt in range(4):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':UA,'Accept':'text/html,image/jpeg,image/*;q=0.8'})
            with urllib.request.urlopen(request,timeout=60) as response:return response.read()
        except Exception as exc:
            errors.append(f'{type(exc).__name__}: {exc}');time.sleep(2**attempt)
    raise RuntimeError(f'Could not fetch {url}: {" | ".join(errors)}')

def extract(page:bytes,photo_id:str)->tuple[str,str,str]:
    text=page.decode('latin-1')
    image=re.search(r'https://s\d\.geograph\.org\.uk/geophotos/[^"&]+/'+re.escape(photo_id)+r'_[0-9a-f]+\.jpg',text)
    licence=re.search(r'rel="license" href="([^"]+)"',text)
    title=re.search(r'<meta property="og:title" content="([^"]+)"',text)
    if not (image and licence and title):raise ValueError(f'Incomplete Geograph metadata for {photo_id}')
    return html.unescape(image.group(0)),html.unescape(licence.group(1)),html.unescape(title.group(1))

def main():
    (OUT/'images').mkdir(parents=True,exist_ok=True)
    rows=[]
    for source in SOURCES:
        photo_id=source['photo_id'];page_url=f'https://www.geograph.org.uk/photo/{photo_id}'
        page=fetch(page_url);image_url,licence_url,title=extract(page,photo_id)
        image=fetch(image_url);path=OUT/'images'/f'uk_material_{photo_id}.jpg';path.write_bytes(image)
        with Image.open(path) as opened:
            opened.verify();width,height=opened.size;fmt=opened.format
        if fmt!='JPEG':raise ValueError(f'Expected JPEG for {photo_id}, got {fmt}')
        rows.append({**source,'record_id':f'uk_material_{photo_id}','title':title,
          'photo_page_url':page_url,'image_url':image_url,'licence_url':licence_url,
          'licence':'CC BY-SA 2.0','image_file':str(path.relative_to(ROOT)),
          'image_sha256':digest(image),'source_page_sha256_at_acquisition':digest(page),
          'width':width,'height':height,'bytes':len(image),'model_inference_performed':False,
          'material_truth_status':'source_evidenced_candidate_not_box_reviewed'})
    groups={}
    for row in rows:groups.setdefault(row['asset_group'],[]).append(row['record_id'])
    manifest={'version':'uk-material-sources-v1','created_at':datetime.now(timezone.utc).isoformat(),
      'selection':'Public UK pages with explicit material descriptions; not selected from model output.',
      'count':len(rows),'independent_asset_groups':len(groups),'asset_groups':groups,'images':rows,
      'model_inference_performed':False,
      'evaluation_status':'candidate_pool_only; freeze compatible boxes and splits before inference',
      'provenance_rule':'Keep page URL, original image URL, author, licence, page and image hashes, evidence excerpt and asset group.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'count':len(rows),'groups':len(groups),'manifest':str((OUT/'manifest.json').relative_to(ROOT))},indent=2))

if __name__=='__main__':main()
