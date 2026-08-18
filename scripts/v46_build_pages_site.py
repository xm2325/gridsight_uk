from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path
from string import Template

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SITE = ROOT / "_site"
ASSETS = SITE / "assets"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def extract_shift(md: str) -> dict[str, float]:
    def grab(pattern: str) -> float:
        m = re.search(pattern, md)
        if not m:
            raise RuntimeError(f"Could not parse morphology-shift value: {pattern}")
        return float(m.group(1))

    return {
        "train_d2_max": grab(r"maximum: \*\*([0-9.]+)"),
        "train_width_ratio": grab(r"Median width/tower-width: \*\*([0-9.]+)"),
        "shifted_median_d2": grab(r"POS_3437435[\s\S]*?Median d²: \*\*([0-9.]+)"),
        "shifted_width_ratio": grab(r"POS_3437435[\s\S]*?Median width/tower-width: \*\*([0-9.]+)"),
    }


def save_zoom_crop(src: Path, box: list[int], out: Path, pad: int = 24, scale: int = 4) -> tuple[int, int]:
    with Image.open(src) as im:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(im.width, x2 + pad)
        y2 = min(im.height, y2 + pad)
        crop = im.crop((x1, y1, x2, y2)).convert("RGB")
        crop = crop.resize(
            (crop.width * scale, crop.height * scale),
            Image.Resampling.LANCZOS,
        )
        crop.save(out, "WEBP", quality=96, method=6)
        return crop.size


def copy_source(src: Path, name: str) -> tuple[int, int]:
    dst = ASSETS / name
    shutil.copy2(src, dst)
    with Image.open(src) as im:
        return im.size


def main() -> None:
    material = load_json(REPORTS / "v4_6_siglip2_material_crop_results.json")
    final = load_json(REPORTS / "v3_8_final_holdout_summary.json")
    annotations = load_json(ROOT / "data/v4_annotations/material_reference_v44.json")
    shift = extract_shift((REPORTS / "v3_9_morphology_shift.md").read_text(encoding="utf-8"))

    records = {r["record_id"]: r for r in annotations["records"]}
    dev = records["POS_2952166"]
    ref = records["POS_8090535"]

    source_paths = {
        "POS_2952166": REPORTS / "v4_4_angle_material_candidates/POS_2952166.jpg",
        "POS_8090535": REPORTS / "v4_4_angle_material_candidates/POS_8090535.jpg",
        "POS_7630781": REPORTS / "v4_4_angle_material_candidates/POS_7630781.jpg",
    }
    offline = REPORTS / "GridSight_UK_v46_SELF_CONTAINED_SHOWCASE.html"
    for path in [*source_paths.values(), offline]:
        if not path.exists():
            raise FileNotFoundError(f"Required Pages source missing: {path}")

    shutil.rmtree(SITE, ignore_errors=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(offline, SITE / "offline.html")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")

    dims = {}
    for rid, src in source_paths.items():
        dims[rid] = copy_source(src, f"{rid}.jpg")

    # Keep the full-tower images at their exact source pixels. Local component
    # views are magnified only to make the existing pixels easier to inspect.
    details = []
    for record, role in [(dev, "adaptive_prediction"), (ref, "reference_source")]:
        src = source_paths[record["record_id"]]
        for i, box in enumerate(record["boxes"], start=1):
            out_name = f"{record['record_id']}_detail_{i}.webp"
            w, h = save_zoom_crop(src, box["xyxy"], ASSETS / out_name)
            details.append(
                {
                    "record_id": record["record_id"],
                    "id": box["id"],
                    "role": role,
                    "truth": box["material_task_label"],
                    "display_truth": "glass" if box["material_task_label"] == "glass" else "ceramic-family",
                    "webp": f"assets/{out_name}",
                    "width": w,
                    "height": h,
                }
            )

    methods = material["methods"]
    champion = material["development_selected_champion"]
    method_order = [
        ("Text prompt", "siglip2_text_prompt_prototype"),
        ("Image reference", "siglip2_image_reference_prototype"),
        ("Hybrid", "siglip2_equal_weight_text_image_hybrid"),
    ]

    web_data = {
        "width": dims["POS_2952166"][0],
        "height": dims["POS_2952166"][1],
        "boxes": {b["id"]: b["xyxy"] for b in dev["boxes"]},
        "truth": {b["id"]: b["material_task_label"] for b in dev["boxes"]},
        "methods": {k: methods[k]["predictions"] for _, k in method_order},
        "details": details,
    }
    data_json = json.dumps(web_data, separators=(",", ":"))

    raw30 = final["raw_text_baseline"]["iou_0_30"]
    geo30 = final["frozen_geometry_champion"]["iou_0_30"]

    cards = []
    for label, key in method_order:
        m = methods[key]
        selected = " selected" if key == champion else ""
        cards.append(
            f'<button class="method-card{selected}" data-method="{key}">'
            f'<span>{html.escape(label)}</span><b>{pct(m["accuracy"])}</b>'
            f'<small>balanced {pct(m["balanced_accuracy"])}</small></button>'
        )
    method_cards = "".join(cards)

    dev_detail_cards = []
    ref_detail_cards = []
    for item in details:
        truth = html.escape(item["display_truth"])
        common = (
            f'<article class="detail-card">'
            f'<img class="clickable" src="{item["webp"]}" width="{item["width"]}" height="{item["height"]}" '
            f'loading="lazy" alt="Magnified component crop {html.escape(item["id"])}">'
            f'<div class="detail-meta"><b>{truth}</b><span class="small-id">{html.escape(item["id"])}</span>'
        )
        if item["role"] == "adaptive_prediction":
            dev_detail_cards.append(
                common
                + f'<div class="pill-row"><span class="pill green" data-pred-for="{html.escape(item["id"])}"></span>'
                + f'<span class="pill amber" data-score-for="{html.escape(item["id"])}"></span></div></div></article>'
            )
        else:
            ref_detail_cards.append(
                common
                + '<div class="pill-row"><span class="pill blue">reference source</span><span class="pill">no model score</span></div></div></article>'
            )

    gallery_specs = [
        (
            "POS_2952166",
            "Mixed-material 400 kV tower",
            "Adaptive material-development source; interactive SigLIP2 predictions are shown in the inspection view above.",
            "https://www.geograph.org.uk/photo/2952166",
        ),
        (
            "POS_8090535",
            "Angle / strain tower",
            "Development reference source with source-assisted glass and ceramic-family component references.",
            "https://www.geograph.org.uk/photo/8090535",
        ),
        (
            "POS_7630781",
            "Terminal tower",
            "Morphology candidate added to widen tower-configuration coverage; no material label or model result is claimed here.",
            "https://www.geograph.org.uk/photo/7630781",
        ),
    ]
    gallery_cards = []
    for rid, title, description, source_url in gallery_specs:
        w, h = dims[rid]
        gallery_cards.append(
            f'<article class="gallery-card"><div class="gallery-copy"><h3>{html.escape(title)}</h3>'
            f'<p class="muted">{html.escape(description)}</p></div>'
            f'<img class="clickable gallery-img" src="assets/{rid}.jpg" width="{w}" height="{h}" loading="lazy" '
            f'alt="{html.escape(title)}">'
            f'<a class="source-link" href="{source_url}">source / attribution</a></article>'
        )

    template = Template(r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="GridSight-UK: UK overhead-line component vision, material classification and domain-shift evaluation.">
<title>GridSight-UK — Overhead-line vision under domain shift</title>
<link rel="preload" as="image" href="assets/POS_2952166.jpg" fetchpriority="high">
<style>
:root{--bg:#071019;--panel:#0d1722;--panel2:#111f2d;--text:#eff6fb;--muted:#9fb0bf;--line:#25394a;--green:#56d0a4;--blue:#76b9ff;--amber:#f1bd66;--red:#ec7e7e}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:linear-gradient(180deg,#071019,#0b141e 55%,#071019);color:var(--text);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:inherit}.wrap{max-width:1240px;margin:auto;padding:22px 20px 60px}nav{display:flex;gap:9px;flex-wrap:wrap;position:sticky;top:0;z-index:10;padding:10px 0;background:rgba(7,16,25,.92);backdrop-filter:blur(10px)}nav a{text-decoration:none;border:1px solid var(--line);border-radius:999px;padding:7px 11px;color:var(--muted)}.hero{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(300px,.8fr);gap:18px;margin-top:8px}.panel{background:rgba(13,23,34,.92);border:1px solid var(--line);border-radius:18px;padding:18px}.visual-panel{display:flex;align-items:center;justify-content:center;background:#050a0f;min-height:660px;padding:14px}.visual-canvas{position:relative;width:min(100%,$WIDTHpx);aspect-ratio:$WIDTH/$HEIGHT}.visual-canvas img{display:block;width:100%;height:100%;object-fit:contain}.visual-canvas svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.eyebrow{font-size:12px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;color:var(--green)}h1{font-size:38px;line-height:1.05;margin:5px 0 10px}h2{font-size:27px;margin:0 0 8px}h3{margin:0 0 7px}.muted{color:var(--muted)}.small{font-size:12px}.stack{display:grid;gap:11px}.metric{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:14px}.metric b{display:block;font-size:30px;line-height:1;margin-bottom:5px}.metric.good b{color:var(--green)}.metric.warn b{color:var(--amber)}section{margin-top:42px;scroll-margin-top:68px}.method-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}.method-card{appearance:none;text-align:left;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:14px;padding:13px;cursor:pointer}.method-card.selected{border-color:var(--green);box-shadow:0 0 0 1px rgba(86,208,164,.2) inset}.method-card span,.method-card small{display:block}.method-card b{display:block;font-size:25px;margin:3px 0}.method-card small{color:var(--muted)}.detail-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}.detail-card{overflow:hidden;border:1px solid var(--line);border-radius:15px;background:var(--panel)}.detail-card img{display:block;width:100%;height:250px;object-fit:contain;background:#050a0f}.detail-meta{padding:10px 12px 12px}.detail-meta b,.small-id{display:block}.small-id{font-size:11px;color:var(--muted);margin-top:2px}.pill-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}.pill{display:inline-flex;border:1px solid var(--line);border-radius:999px;padding:3px 7px;font-size:11px;color:var(--muted)}.pill.green{color:var(--green);background:rgba(86,208,164,.07)}.pill.amber{color:var(--amber);background:rgba(241,189,102,.07)}.pill.blue{color:var(--blue);background:rgba(118,185,255,.07)}.gallery-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.gallery-card{display:flex;flex-direction:column;border:1px solid var(--line);border-radius:16px;background:var(--panel);overflow:hidden}.gallery-copy{padding:14px 14px 10px;min-height:130px}.gallery-img{display:block;width:100%;height:390px;object-fit:contain;background:#050a0f}.source-link{padding:10px 14px;border-top:1px solid var(--line);font-size:12px;color:var(--muted);text-decoration:none}.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}.barrow{display:grid;grid-template-columns:150px 1fr 64px;gap:10px;align-items:center;margin:13px 0}.bar{height:12px;background:#172736;border-radius:999px;overflow:hidden}.bar i{display:block;height:100%;background:var(--blue)}.bar.geo i{background:var(--amber)}.shift{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.shift .metric b{font-size:24px}.contract{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.step{border:1px solid var(--line);border-radius:14px;padding:13px;background:var(--panel)}.step span{display:block;color:var(--muted);font-size:12px;margin-top:4px}.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:15px}.btn{display:inline-block;text-decoration:none;border:1px solid var(--line);border-radius:10px;padding:9px 12px;background:#10202d}.btn.primary{border-color:var(--green);background:rgba(86,208,164,.1)}.clickable{cursor:zoom-in}.modal{position:fixed;inset:0;z-index:100;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.88);padding:20px}.modal.open{display:flex}.modal img{max-width:96vw;max-height:92vh;width:auto;height:auto;border-radius:10px;box-shadow:0 16px 60px rgba(0,0,0,.55)}.modal button{position:absolute;top:18px;right:18px;border:1px solid var(--line);border-radius:10px;background:#10202d;color:var(--text);padding:8px 11px;cursor:pointer}footer{border-top:1px solid var(--line);margin-top:42px;padding-top:16px;color:var(--muted);font-size:12px}@media(max-width:920px){.hero,.two{grid-template-columns:1fr}.gallery-grid{grid-template-columns:1fr}.method-grid{grid-template-columns:1fr}.contract,.shift{grid-template-columns:1fr 1fr}.visual-panel{min-height:auto}h1{font-size:31px}}@media(max-width:520px){.contract,.shift{grid-template-columns:1fr}.barrow{grid-template-columns:115px 1fr 55px}.wrap{padding-left:13px;padding-right:13px}.detail-grid{grid-template-columns:1fr 1fr}.detail-card img{height:210px}}
</style>
</head>
<body><div class="wrap">
<nav><a href="#inspection">Inspection view</a><a href="#prediction-details">Prediction details</a><a href="#gallery">Tower gallery</a><a href="#reference-details">Reference details</a><a href="#detection">Detection</a><a href="#shift">Domain shift</a><a href="#evidence">Evidence</a></nav>
<section id="inspection" class="hero">
<div class="panel visual-panel"><div class="visual-canvas"><img class="clickable" src="assets/POS_2952166.jpg" width="$WIDTH" height="$HEIGHT" fetchpriority="high" decoding="async" alt="UK 400 kV tower with material prediction overlay"><svg id="overlay" viewBox="0 0 $WIDTH $HEIGHT" preserveAspectRatio="xMidYMid meet" aria-label="Vector material predictions"></svg></div></div>
<div class="panel stack"><div><div class="eyebrow">GridSight-UK · v4.7 Pages presentation</div><h1>Sharper component inspection, without hiding source limits</h1><p class="muted">The full tower is shown at no more than its native source width. Vector boxes stay sharp, and magnified component crops make the small insulators easier to inspect. The page now includes three tower configurations rather than one.</p></div><div class="metric good"><b>6 / 6</b><span>SigLIP2 text-prompt predictions correct on the adaptive six-crop development source.</span></div><div class="metric"><b>$RAW_RECALL</b><span>Raw YOLOE recall at IoU ≥ 0.30 on the frozen 12-insulator final holdout.</span></div><div class="metric warn"><b>$GEO_RECALL</b><span>Frozen geometry-filter recall on the same holdout after morphology shift.</span></div><div class="metric"><b>$WIDTH × $HEIGHT</b><span>Native pixels of the main public source image. The detail crops magnify existing pixels; they do not add new image information.</span></div></div>
</section>
<section id="material"><div class="eyebrow">Material branch</div><h2>Interactive SigLIP2 comparison</h2><p class="muted">Switching methods updates only the SVG overlay and the prediction labels below. Boxes are manual/oracle component boxes; material labels are model outputs; displayed scores are relative and uncalibrated.</p><div class="method-grid">$METHOD_CARDS</div><div class="panel"><p id="method-note" class="muted"></p></div></section>
<section id="prediction-details"><div class="eyebrow">Magnified prediction crops</div><h2>Six component-level views from POS_2952166</h2><p class="muted">These views are 4× display magnifications of local source pixels. They are included because the insulator assemblies are only a small part of the full 627 × 640 tower image. Click any crop to inspect it larger.</p><div class="detail-grid">$DEV_DETAILS</div></section>
<section id="gallery"><div class="eyebrow">Tower gallery</div><h2>Three source images, three tower configurations</h2><p class="muted">The first tower is the adaptive material-development source, the second is the material reference source, and the third is an unlabelled terminal-morphology candidate. The cards keep those evidence roles separate.</p><div class="gallery-grid">$GALLERY_CARDS</div></section>
<section id="reference-details"><div class="eyebrow">Reference-source detail</div><h2>Six magnified component references from POS_8090535</h2><p class="muted">These are source-assisted reference annotations, not predictions. No model percentage is shown on this source.</p><div class="detail-grid">$REF_DETAILS</div></section>
<section id="detection"><div class="eyebrow">Frozen final holdout</div><h2>A development heuristic failed under new tower morphology</h2><div class="two"><div class="panel"><h3>Recall at IoU ≥ 0.30</h3><div class="barrow"><span>Raw YOLOE text</span><div class="bar"><i style="width:$RAW_RECALL_NUM%"></i></div><b>$RAW_RECALL</b></div><div class="barrow"><span>Geometry filter</span><div class="bar geo"><i style="width:$GEO_RECALL_NUM%"></i></div><b>$GEO_RECALL</b></div></div><div class="panel"><h3>What changed?</h3><p class="muted">Raw YOLOE kept 10 true positives with 2 false negatives. The frozen geometry method kept only 1 true positive and rejected 11 reference insulators. The negative result was kept instead of tuning against the final holdout.</p></div></div></section>
<section id="shift"><div class="eyebrow">Failure diagnosis</div><h2>Quantified morphology shift</h2><div class="shift"><div class="metric"><b>$TRAIN_D2</b><span>maximum training Mahalanobis d²</span></div><div class="metric warn"><b>$SHIFT_D2</b><span>median d² on the most shifted final tower</span></div><div class="metric"><b>$TRAIN_WIDTH</b><span>training median insulator width / tower width</span></div><div class="metric warn"><b>$SHIFT_WIDTH</b><span>shifted-tower median width / tower width</span></div></div></section>
<section id="evidence"><div class="eyebrow">Evidence contract</div><h2>What each visual element means</h2><div class="contract"><div class="step"><b>Source pixels</b><span>Real UK overhead-line image with recorded source and SHA-256.</span></div><div class="step"><b>Manual component box</b><span>v4.6 material classification does not claim automatic localisation.</span></div><div class="step"><b>SigLIP2 label</b><span>glass vs ceramic-family prediction from the frozen encoder/prototype method.</span></div><div class="step"><b>Relative score</b><span>Two-class softmax display score; not a calibrated probability.</span></div></div><div class="actions"><a class="btn primary" href="https://github.com/xm2325/gridsight_uk">GitHub repository</a><a class="btn" href="offline.html">Offline self-contained version</a><a class="btn" href="https://www.geograph.org.uk/photo/2952166">Main image attribution</a></div></section>
<footer>GridSight-UK · public tower imagery via Geograph under the recorded Creative Commons licences. Source images are shown at their available project resolution; magnified crops do not create new source detail. Material references are source-assisted and visually corroborated, not laboratory-verified. No condition, corrosion, defect, failure, mechanical-integrity or safety-risk performance is claimed.</footer>
</div>
<div id="modal" class="modal" aria-hidden="true"><button id="modal-close" type="button">Close</button><img id="modal-img" alt="Expanded project image"></div>
<script>
const DATA=$DATA_JSON;
const NS='http://www.w3.org/2000/svg';
const overlay=document.getElementById('overlay');
const note=document.getElementById('method-note');
const labels={siglip2_text_prompt_prototype:'Text-prompt prototype',siglip2_image_reference_prototype:'Image-reference prototype',siglip2_equal_weight_text_image_hybrid:'Equal-weight hybrid'};
function el(name,attrs={}){const n=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);return n;}
function displayLabel(raw){return raw==='glass'?'glass':'ceramic';}
function render(key){
  overlay.innerHTML='';
  const preds=Object.fromEntries(DATA.methods[key].map(x=>[x.id,x]));
  for(const [id,box] of Object.entries(DATA.boxes)){
    const p=preds[id]; if(!p) continue;
    const [x1,y1,x2,y2]=box;
    const glass=p.predicted_label==='glass';
    const c=glass?'#45e39f':'#ffb35d';
    overlay.appendChild(el('rect',{x:x1,y:y1,width:x2-x1,height:y2-y1,fill:'none',stroke:c,'stroke-width':3,'vector-effect':'non-scaling-stroke'}));
    const label=displayLabel(p.predicted_label)+' '+(p.relative_score*100).toFixed(1)+'%';
    const tx=Math.max(4,x1),ty=Math.max(18,y1-7);
    overlay.appendChild(el('rect',{x:tx-2,y:ty-15,width:Math.max(78,label.length*6.5),height:18,rx:3,fill:'#071019','fill-opacity':.9,stroke:c,'stroke-width':1,'vector-effect':'non-scaling-stroke'}));
    const t=el('text',{x:tx,y:ty,fill:c,'font-size':12,'font-weight':800,'font-family':'ui-sans-serif,system-ui'});t.textContent=label;overlay.appendChild(t);
  }
  const rows=DATA.methods[key];
  note.textContent=labels[key]+' · '+rows.filter(x=>x.correct).length+'/'+rows.length+' correct on the adaptive development source. Scores are relative and uncalibrated.';
  document.querySelectorAll('.method-card').forEach(b=>b.classList.toggle('selected',b.dataset.method===key));
  for(const item of DATA.details){
    if(item.role!=='adaptive_prediction') continue;
    const p=preds[item.id]; if(!p) continue;
    const pe=document.querySelector('[data-pred-for="'+item.id+'"]');
    const se=document.querySelector('[data-score-for="'+item.id+'"]');
    if(pe) pe.textContent='pred: '+displayLabel(p.predicted_label);
    if(se) se.textContent='score: '+(p.relative_score*100).toFixed(1)+'%';
  }
}
document.querySelectorAll('.method-card').forEach(b=>b.addEventListener('click',()=>render(b.dataset.method)));
const modal=document.getElementById('modal'),modalImg=document.getElementById('modal-img'),modalClose=document.getElementById('modal-close');
function openModal(img){modalImg.src=img.src;modalImg.alt=img.alt||'Expanded project image';modal.classList.add('open');modal.setAttribute('aria-hidden','false');}
function closeModal(){modal.classList.remove('open');modal.setAttribute('aria-hidden','true');modalImg.removeAttribute('src');}
document.querySelectorAll('.clickable').forEach(img=>img.addEventListener('click',()=>openModal(img)));
modalClose.addEventListener('click',closeModal);modal.addEventListener('click',e=>{if(e.target===modal)closeModal();});document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal();});
render('$CHAMPION');
</script></body></html>''')

    width, height = dims["POS_2952166"]
    page = template.substitute(
        WIDTH=width,
        HEIGHT=height,
        RAW_RECALL=pct(raw30["recall"]),
        GEO_RECALL=pct(geo30["recall"]),
        RAW_RECALL_NUM=f"{100 * raw30['recall']:.1f}",
        GEO_RECALL_NUM=f"{100 * geo30['recall']:.1f}",
        TRAIN_D2=f"{shift['train_d2_max']:.3f}",
        SHIFT_D2=f"{shift['shifted_median_d2']:.3f}",
        TRAIN_WIDTH=f"{shift['train_width_ratio']:.3f}",
        SHIFT_WIDTH=f"{shift['shifted_width_ratio']:.3f}",
        METHOD_CARDS=method_cards,
        DEV_DETAILS="".join(dev_detail_cards),
        REF_DETAILS="".join(ref_detail_cards),
        GALLERY_CARDS="".join(gallery_cards),
        DATA_JSON=data_json,
        CHAMPION=champion,
    )

    (SITE / "index.html").write_text(page, encoding="utf-8")
    (ASSETS / "material_results.json").write_text(json.dumps(material, indent=2) + "\n", encoding="utf-8")

    detail_files = sorted(ASSETS.glob("POS_*_detail_*.webp"))
    if len(detail_files) != 12:
        raise RuntimeError(f"Expected 12 magnified component crops, found {len(detail_files)}")
    if (SITE / "index.html").stat().st_size > 220_000:
        raise RuntimeError("Online index unexpectedly large; keep heavy assets outside the HTML")

    print(
        json.dumps(
            {
                "site": str(SITE),
                "index_bytes": (SITE / "index.html").stat().st_size,
                "offline_bytes": (SITE / "offline.html").stat().st_size,
                "full_source_images": {
                    rid: {"dimensions": dims[rid], "bytes": (ASSETS / f"{rid}.jpg").stat().st_size}
                    for rid in source_paths
                },
                "magnified_component_crops": len(detail_files),
                "crop_bytes_total": sum(p.stat().st_size for p in detail_files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
