from __future__ import annotations

import hashlib
import json
import struct
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/v4_0_candidates"
OUT.mkdir(parents=True, exist_ok=True)
UA = "GridSight-UK-v4.0-morphology-acquisition/1.0 (+https://github.com/xm2325/gridsight_uk)"

# Pre-existing geographically diverse Geograph seed records that were not used in the closed v2-v3
# model-selection/final-holdout cycle. Selection here is metadata/geography only; no model scores are used.
CANDIDATES = [
    {"photo_id":"7072688","country":"Scotland","photographer":"Richard Sutcliffe","direct_url":"https://s0.geograph.org.uk/geophotos/07/07/26/7072688_5aae390b.jpg"},
    {"photo_id":"7478407","country":"Scotland","photographer":"Richard Sutcliffe","direct_url":"https://s0.geograph.org.uk/geophotos/07/47/84/7478407_cba3b805.jpg"},
    {"photo_id":"7528296","country":"Scotland","photographer":"Jim Smillie","direct_url":None},
    {"photo_id":"6610209","country":"Wales","photographer":"Alan Hughes","direct_url":"https://s0.geograph.org.uk/geophotos/06/61/02/6610209_b56889e7.jpg"},
    {"photo_id":"5952661","country":"England","photographer":"Malc McDonald","direct_url":"https://s0.geograph.org.uk/geophotos/05/95/26/5952661_bd4ace05.jpg"},
    {"photo_id":"8091164","country":"England","photographer":"Daniel Beardsmore","direct_url":"https://s0.geograph.org.uk/geophotos/08/09/11/8091164_0e5e19aa.jpg"},
    {"photo_id":"7945993","country":"England","photographer":"David Smith","direct_url":None},
    {"photo_id":"8239540","country":"England","photographer":"N Chadwick","direct_url":None},
    {"photo_id":"354803","country":"England","photographer":"Mick Garratt","direct_url":None},
    {"photo_id":"7480474","country":"England","photographer":"Luke Shaw","direct_url":None}
]


def fetch(url: str, accept: str = "*/*") -> bytes:
    last=None
    for attempt in range(5):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":UA,"Accept":accept})
            with urllib.request.urlopen(req,timeout=60) as r:
                return r.read()
        except Exception as exc:
            last=exc; time.sleep(min(2**attempt,12))
    raise last


def discover(photo_id: str) -> tuple[str,str]:
    import re
    page=f"https://www.geograph.org.uk/photo/{photo_id}"
    html=fetch(page,"text/html,*/*;q=0.1").decode("utf-8","replace")
    matches=re.findall(r"https://s\d+\.geograph\.org\.uk/(?:photos|geophotos)/[^\"'<> ]+?\.jpg",html)
    if not matches: raise RuntimeError(f"No Geograph JPEG surfaced by {page}")
    return matches[0].replace("&amp;","&"),page


def jpeg_size(data: bytes):
    if not data.startswith(b"\xff\xd8"): raise ValueError("not JPEG")
    i=2; sof={0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF}
    while i+9<len(data):
        if data[i]!=0xFF: i+=1; continue
        while i<len(data) and data[i]==0xFF: i+=1
        if i>=len(data): break
        marker=data[i]; i+=1
        if marker in {0xD8,0xD9} or 0xD0<=marker<=0xD7: continue
        seglen=struct.unpack('>H',data[i:i+2])[0]
        if marker in sof:
            h=struct.unpack('>H',data[i+3:i+5])[0]; w=struct.unpack('>H',data[i+5:i+7])[0]; return w,h
        i+=seglen
    raise ValueError("JPEG dimensions not found")


def main():
    rows=[]
    for index,c in enumerate(CANDIDATES):
        pid=c['photo_id']; page=f"https://www.geograph.org.uk/photo/{pid}"; attempts=[]; data=None; used=None
        urls=[]
        if c.get('direct_url'): urls.append((c['direct_url'],'preexisting_direct_seed'))
        try:
            discovered,_=discover(pid); urls.append((discovered,'photo_page_discovered'))
        except Exception as exc:
            attempts.append({'stage':'discover','error':repr(exc)})
        seen=set()
        for url,kind in urls:
            if url in seen: continue
            seen.add(url)
            try:
                data=fetch(url,"image/jpeg,image/*;q=0.8,*/*;q=0.1"); used=(url,kind); break
            except Exception as exc:
                attempts.append({'stage':kind,'url':url,'error':repr(exc)})
        if data is None: raise RuntimeError(f"Could not acquire {pid}: {attempts}")
        w,h=jpeg_size(data); out=OUT/f"POS_{pid}.jpg"; out.write_bytes(data)
        rows.append({
          "record_id":f"POS_{pid}","photo_id":pid,"country":c['country'],"photographer":c['photographer'],
          "photo_page_url":page,"runtime_image_url":used[0],"runtime_source_kind":used[1],
          "width_px":w,"height_px":h,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"sha1":hashlib.sha1(data).hexdigest(),
          "candidate_order":index,"visual_review_status":"pending_pixel_review","model_inference_before_v4_split_freeze":False,
          "eligible_for_v4_split":False,"attempt_failures":attempts,"licence":"CC BY-SA 2.0"
        })
        time.sleep(0.4)
    report={
      "version":"v4.0-morphology-diverse-candidate-acquisition",
      "status":"PASS","count":len(rows),"selection_basis":"pre-existing geographically diverse Geograph seed queue; old v2-v3 train/val/development/final sources excluded; no model outputs used",
      "old_v3_final_holdout_retired_from_model_selection":["POS_3437435","POS_7561805"],
      "candidates":rows,
      "next_gate":"assistant pixel review may assess morphology/annotatability, but v4 train/validation/new-final roles must be frozen before any v4 model inference"
    }
    (ROOT/'reports/v4_0_candidate_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
