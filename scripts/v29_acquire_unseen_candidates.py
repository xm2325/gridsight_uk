from __future__ import annotations

import hashlib
import json
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/v2_9_unseen_candidates"
OUT.mkdir(parents=True,exist_ok=True)
UA="GridSight-UK-v2.9-candidate-acquisition/1.0 (+https://github.com/xm2325/gridsight_uk)"

CANDIDATES=[
 {"photo_id":"7072688","country":"Scotland","photographer":"Richard Sutcliffe","url":"https://s0.geograph.org.uk/geophotos/07/07/26/7072688_5aae390b.jpg","licence":"CC BY-SA 2.0"},
 {"photo_id":"7561805","country":"Scotland","photographer":"Richard Sutcliffe","url":"https://s0.geograph.org.uk/geophotos/07/56/18/7561805_65ecc0eb.jpg","licence":"CC BY-SA 2.0"},
 {"photo_id":"7478407","country":"Scotland","photographer":"Richard Sutcliffe","url":"https://s0.geograph.org.uk/geophotos/07/47/84/7478407_cba3b805.jpg","licence":"CC BY-SA 2.0"},
 {"photo_id":"6610209","country":"Wales","photographer":"Alan Hughes","url":"https://s0.geograph.org.uk/geophotos/06/61/02/6610209_b56889e7.jpg","licence":"CC BY-SA 2.0"},
 {"photo_id":"3437435","country":"England","photographer":"Philip Halling","url":"https://s0.geograph.org.uk/geophotos/03/43/74/3437435_c782d726.jpg","licence":"CC BY-SA 2.0"},
 {"photo_id":"8091164","country":"England","photographer":"Daniel Beardsmore","url":"https://s0.geograph.org.uk/geophotos/08/09/11/8091164_0e5e19aa.jpg","licence":"CC BY-SA 2.0"}
]

def fetch(url):
 last=None
 for k in range(4):
  try:
   req=urllib.request.Request(url,headers={"User-Agent":UA,"Accept":"image/jpeg,image/*;q=0.8,*/*;q=0.1"})
   with urllib.request.urlopen(req,timeout=60) as r:return r.read()
  except Exception as e:last=e; time.sleep(2**k)
 raise last

def jpeg_size(data):
 if not data.startswith(b'\xff\xd8'):raise ValueError('not JPEG')
 i=2; sof={0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF}
 while i+9<len(data):
  if data[i]!=0xFF:i+=1;continue
  while i<len(data) and data[i]==0xFF:i+=1
  if i>=len(data):break
  marker=data[i];i+=1
  if marker in {0xD8,0xD9} or 0xD0<=marker<=0xD7:continue
  seglen=struct.unpack('>H',data[i:i+2])[0]
  if marker in sof:
   h=struct.unpack('>H',data[i+3:i+5])[0];w=struct.unpack('>H',data[i+5:i+7])[0];return w,h
  i+=seglen
 raise ValueError('size not found')

rows=[]
for c in CANDIDATES:
 data=fetch(c['url']); w,h=jpeg_size(data); path=OUT/f"POS_{c['photo_id']}.jpg"; path.write_bytes(data)
 rows.append({**c,"record_id":f"POS_{c['photo_id']}","path":str(path.relative_to(ROOT)),"width_px":w,"height_px":h,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"sha1":hashlib.sha1(data).hexdigest(),"photo_page_url":f"https://www.geograph.org.uk/photo/{c['photo_id']}","ground_truth_status":"candidate_only_visual_review_required","model_inference_allowed":False})
report={"status":"PASS","selection_basis":"pre-existing geographic acquisition queue; no model outputs used","count":len(rows),"candidates":rows,"final_holdout_rule":"At least one visually suitable source will be frozen before any model inference on it."}
(ROOT/'reports/v2_9_unseen_candidate_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
