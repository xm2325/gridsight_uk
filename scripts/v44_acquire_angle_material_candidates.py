from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/v4_4_angle_material_candidates"
OUT.mkdir(parents=True, exist_ok=True)
UA = "GridSight-UK-v4.4-candidate-acquisition/1.0 (+https://github.com/xm2325/gridsight_uk)"
EXCLUDED = {"3437435", "7561805", "6610209", "8091164", "7072688", "7478407"}

# Discovered from public Geograph pages before any v4.4 model inference.
# Text descriptions are discovery metadata only, never component-level ground truth.
CANDIDATES = [
    {
        "photo_id": "8090535",
        "country": "England",
        "photographer": "Daniel Beardsmore",
        "title": "Balfour Beatty L6 D30 pylon, Hitchin",
        "page_url": "https://www.geograph.org.uk/photo/8090535",
        "image_url": "https://s0.geograph.org.uk/geophotos/08/09/05/8090535_97e777ce.jpg",
        "licence": "CC BY-SA 2.0",
        "discovery_context": "400 kV 30-degree angle tower; page description mentions refurbished blue-tinted glass versus older brown-glazed porcelain insulators.",
        "target_reason": "angle/strain morphology plus potential future material-subtype visual review"
    },
    {
        "photo_id": "7630781",
        "country": "England",
        "photographer": "Rod Grealish",
        "title": "Terminal Electricity Pylon from Isabel Trail on Doxey Marsh Stafford",
        "page_url": "https://www.geograph.org.uk/photo/7630781",
        "image_url": "https://s0.geograph.org.uk/geophotos/07/63/07/7630781_792709d1.jpg",
        "licence": "Creative Commons reuse per Geograph page",
        "discovery_context": "Terminal pylon at the end of an overhead line before underground supply.",
        "target_reason": "terminal/strain morphology"
    },
    {
        "photo_id": "2952166",
        "country": "England",
        "photographer": "Peter Facey",
        "title": "400 KV Pylon with mobile phone aerials",
        "page_url": "https://www.geograph.org.uk/photo/2952166",
        "image_url": "https://s0.geograph.org.uk/geophotos/02/95/21/2952166_879bdeb6.jpg",
        "licence": "CC BY-SA 2.0",
        "discovery_context": "Page states ceramic insulators on the left side and green glass insulators on the right side.",
        "target_reason": "candidate for future visually adjudicated material subtype, never automatic material GT"
    }
]


def fetch(url: str) -> bytes:
    errors=[]
    for attempt in range(5):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":UA,"Accept":"image/jpeg,image/*;q=0.8,*/*;q=0.1"})
            with urllib.request.urlopen(req,timeout=60) as r: return r.read()
        except Exception as e:
            errors.append(f"attempt {attempt+1}: {type(e).__name__}: {e}"); time.sleep(min(8,2**attempt))
    raise RuntimeError(f"download failed: {url}: {' | '.join(errors)}")


def contact_sheet(rows):
    tw,th,lh=640,480,88
    canvas=Image.new("RGB",(tw*2,(th+lh)*2),"white");draw=ImageDraw.Draw(canvas);font=ImageFont.load_default()
    for i,row in enumerate(rows):
        im=Image.open(ROOT/row["path"]).convert("RGB");fit=ImageOps.contain(im,(tw,th));col=i%2;rowi=i//2;x=col*tw+(tw-fit.width)//2;y=rowi*(th+lh)+(th-fit.height)//2;canvas.paste(fit,(x,y))
        text=f"{row['record_id']} | {row['title']}\n{row['country']} | {row['width_px']}x{row['height_px']} | candidate-only\nNO MODEL INFERENCE / NO MATERIAL GT"
        draw.text((col*tw+8,rowi*(th+lh)+th+5),text,fill="black",font=font)
    path=ROOT/"reports/v4_4_angle_material_contact_sheet.jpg";canvas.save(path,quality=94);return str(path.relative_to(ROOT))


def main():
    if any(c["photo_id"] in EXCLUDED for c in CANDIDATES): raise RuntimeError("Previously used source leaked into v4.4 candidate set")
    rows=[]
    for c in CANDIDATES:
        payload=fetch(c["image_url"]);path=OUT/f"POS_{c['photo_id']}.jpg";path.write_bytes(payload)
        with Image.open(path) as im: w,h=im.size;fmt=im.format
        if fmt!="JPEG": raise RuntimeError(f"Expected JPEG, got {fmt}")
        rows.append({**c,"record_id":f"POS_{c['photo_id']}","path":str(path.relative_to(ROOT)),"width_px":w,"height_px":h,"bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest(),"sha1":hashlib.sha1(payload).hexdigest(),"ground_truth_status":"candidate_only_pixel_review_required","model_inference_performed":False,"material_ground_truth_status":"none; page-level descriptive context only"})
    report={"version":"v4.4-angle-material-candidate-acquisition","count":len(rows),"selection_time_rule":"Candidates identified and committed before any model inference on their pixels.","excluded_previous_sources":sorted(EXCLUDED),"candidates":rows,"contact_sheet":contact_sheet(rows),"model_inference_performed":False,"next_gate":"assistant pixel review for morphology, scale, component annotatability, and whether source-level material description is visually resolvable at component level"}
    (ROOT/"reports/v4_4_angle_material_candidate_manifest.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))

if __name__=="__main__": main()
