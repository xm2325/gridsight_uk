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


def main() -> None:
    material = load_json(REPORTS / "v4_6_siglip2_material_crop_results.json")
    final = load_json(REPORTS / "v3_8_final_holdout_summary.json")
    annotations = load_json(ROOT / "data/v4_annotations/material_reference_v44.json")
    shift = extract_shift((REPORTS / "v3_9_morphology_shift.md").read_text(encoding="utf-8"))

    dev = next(r for r in annotations["records"] if r["record_id"] == "POS_2952166")
    src = REPORTS / "v4_4_angle_material_candidates/POS_2952166.jpg"
    offline = REPORTS / "GridSight_UK_v46_SELF_CONTAINED_SHOWCASE.html"
    if not src.exists() or not offline.exists():
        raise FileNotFoundError("Run the v4.6 evidence pipeline before building Pages")

    shutil.rmtree(SITE, ignore_errors=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, ASSETS / "POS_2952166.jpg")
    shutil.copy2(offline, SITE / "offline.html")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")

    with Image.open(src) as im:
        im.convert("RGB").save(ASSETS / "POS_2952166.webp", "WEBP", quality=90, method=6)
        width, height = im.size

    methods = material["methods"]
    champion = material["development_selected_champion"]
    method_order = [
        ("Text prompt", "siglip2_text_prompt_prototype"),
        ("Image reference", "siglip2_image_reference_prototype"),
        ("Hybrid", "siglip2_equal_weight_text_image_hybrid"),
    ]

    web_data = {
        "width": width,
        "height": height,
        "boxes": {b["id"]: b["xyxy"] for b in dev["boxes"]},
        "truth": {b["id"]: b["material_task_label"] for b in dev["boxes"]},
        "methods": {k: methods[k]["predictions"] for _, k in method_order},
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

    template = Template(r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="GridSight-UK: UK overhead-line component vision, material classification and domain-shift evaluation.">
<title>GridSight-UK — Overhead-line vision under domain shift</title>
<link rel="preload" as="image" href="assets/POS_2952166.webp" type="image/webp" fetchpriority="high">
<style>
:root{--bg:#071019;--panel:#0d1722;--panel2:#111f2d;--text:#eff6fb;--muted:#9fb0bf;--line:#25394a;--green:#56d0a4;--blue:#76b9ff;--amber:#f1bd66;--red:#ec7e7e}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:linear-gradient(180deg,#071019,#0b141e 55%,#071019);color:var(--text);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:inherit}.wrap{max-width:1180px;margin:auto;padding:22px 20px 60px}nav{display:flex;gap:9px;flex-wrap:wrap;position:sticky;top:0;z-index:10;padding:10px 0;background:rgba(7,16,25,.92);backdrop-filter:blur(10px)}nav a{text-decoration:none;border:1px solid var(--line);border-radius:999px;padding:7px 11px;color:var(--muted)}.hero{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(290px,.65fr);gap:18px;margin-top:8px}.panel{background:rgba(13,23,34,.92);border:1px solid var(--line);border-radius:18px;padding:18px}.visual{padding:0;overflow:hidden;position:relative;background:#050a0f;aspect-ratio:$WIDTH/$HEIGHT}.visual picture,.visual img{display:block;width:100%;height:100%}.visual img{object-fit:contain}.visual svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.eyebrow{font-size:12px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;color:var(--green)}h1{font-size:38px;line-height:1.05;margin:5px 0 10px}h2{font-size:27px;margin:0 0 8px}h3{margin:0 0 8px}.muted{color:var(--muted)}.stack{display:grid;gap:11px}.metric{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:14px}.metric b{display:block;font-size:30px;line-height:1;margin-bottom:5px}.metric.good b{color:var(--green)}.metric.warn b{color:var(--amber)}section{margin-top:42px;scroll-margin-top:68px}.method-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}.method-card{appearance:none;text-align:left;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:14px;padding:13px;cursor:pointer}.method-card.selected{border-color:var(--green);box-shadow:0 0 0 1px rgba(86,208,164,.2) inset}.method-card span,.method-card small{display:block}.method-card b{display:block;font-size:25px;margin:3px 0}.method-card small{color:var(--muted)}.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}.barrow{display:grid;grid-template-columns:150px 1fr 64px;gap:10px;align-items:center;margin:13px 0}.bar{height:12px;background:#172736;border-radius:999px;overflow:hidden}.bar i{display:block;height:100%;background:var(--blue)}.bar.geo i{background:var(--amber)}.shift{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.shift .metric b{font-size:24px}.contract{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.step{border:1px solid var(--line);border-radius:14px;padding:13px;background:var(--panel)}.step span{display:block;color:var(--muted);font-size:12px;margin-top:4px}.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:15px}.btn{display:inline-block;text-decoration:none;border:1px solid var(--line);border-radius:10px;padding:9px 12px;background:#10202d}.btn.primary{border-color:var(--green);background:rgba(86,208,164,.1)}footer{border-top:1px solid var(--line);margin-top:42px;padding-top:16px;color:var(--muted);font-size:12px}@media(max-width:850px){.hero,.two{grid-template-columns:1fr}.method-grid{grid-template-columns:1fr}.contract,.shift{grid-template-columns:1fr 1fr}h1{font-size:31px}}@media(max-width:520px){.contract,.shift{grid-template-columns:1fr}.barrow{grid-template-columns:115px 1fr 55px}.wrap{padding-left:13px;padding-right:13px}}
</style>
</head>
<body><div class="wrap">
<nav><a href="#inspection">Inspection view</a><a href="#material">Material</a><a href="#detection">Detection</a><a href="#shift">Domain shift</a><a href="#evidence">Evidence</a></nav>
<section id="inspection" class="hero">
<div class="panel visual">
<picture><source srcset="assets/POS_2952166.webp" type="image/webp"><img src="assets/POS_2952166.jpg" width="$WIDTH" height="$HEIGHT" fetchpriority="high" decoding="async" alt="UK 400 kV tower with model material overlays"></picture>
<svg id="overlay" viewBox="0 0 $WIDTH $HEIGHT" preserveAspectRatio="xMidYMid meet" aria-label="Vector material predictions"></svg>
</div>
<div class="panel stack"><div><div class="eyebrow">GridSight-UK · v4.6</div><h1>Overhead-line vision under domain shift</h1><p class="muted">Fast portfolio view for model outputs, failure analysis and evidence controls. Bounding boxes and labels are SVG vectors, so they stay sharp when zoomed.</p></div><div class="metric good"><b>6 / 6</b><span>SigLIP2 text-prompt predictions correct on the adaptive six-crop development source.</span></div><div class="metric"><b>$RAW_RECALL</b><span>Raw YOLOE recall at IoU ≥ 0.30 on the frozen 12-insulator final holdout.</span></div><div class="metric warn"><b>$GEO_RECALL</b><span>Frozen geometry-filter recall on the same holdout after morphology shift.</span></div></div>
</section>
<section id="material"><div class="eyebrow">Material branch</div><h2>Interactive SigLIP2 material comparison</h2><p class="muted">The source image downloads once. Switching methods updates only the SVG overlay. Boxes are manual/oracle component boxes; material labels are model outputs; displayed scores are relative and uncalibrated.</p><div class="method-grid">$METHOD_CARDS</div><div class="panel"><p id="method-note" class="muted"></p></div></section>
<section id="detection"><div class="eyebrow">Frozen final holdout</div><h2>A development heuristic failed under new tower morphology</h2><div class="two"><div class="panel"><h3>Recall at IoU ≥ 0.30</h3><div class="barrow"><span>Raw YOLOE text</span><div class="bar"><i style="width:$RAW_RECALL_NUM%"></i></div><b>$RAW_RECALL</b></div><div class="barrow"><span>Geometry filter</span><div class="bar geo"><i style="width:$GEO_RECALL_NUM%"></i></div><b>$GEO_RECALL</b></div></div><div class="panel"><h3>What changed?</h3><p class="muted">Raw YOLOE kept 10 true positives with 2 false negatives. The frozen geometry method kept only 1 true positive and rejected 11 reference insulators. The negative result was preserved instead of tuning against the final holdout.</p></div></div></section>
<section id="shift"><div class="eyebrow">Failure diagnosis</div><h2>Quantified morphology shift</h2><div class="shift"><div class="metric"><b>$TRAIN_D2</b><span>maximum training Mahalanobis d²</span></div><div class="metric warn"><b>$SHIFT_D2</b><span>median d² on the most shifted final tower</span></div><div class="metric"><b>$TRAIN_WIDTH</b><span>training median insulator width / tower width</span></div><div class="metric warn"><b>$SHIFT_WIDTH</b><span>shifted-tower median width / tower width</span></div></div></section>
<section id="evidence"><div class="eyebrow">Evidence contract</div><h2>What each visual element means</h2><div class="contract"><div class="step"><b>Source pixels</b><span>Real UK overhead-line image with recorded source and SHA-256.</span></div><div class="step"><b>Manual component box</b><span>v4.6 material classification does not claim automatic localisation.</span></div><div class="step"><b>SigLIP2 label</b><span>glass vs ceramic-family prediction from the frozen encoder/prototype method.</span></div><div class="step"><b>Relative score</b><span>Two-class softmax display score; not a calibrated probability.</span></div></div><div class="actions"><a class="btn primary" href="https://github.com/xm2325/gridsight_uk">GitHub repository</a><a class="btn" href="offline.html">Offline self-contained version</a><a class="btn" href="https://www.geograph.org.uk/photo/2952166">Image source / attribution</a></div></section>
<footer>GridSight-UK v4.6 · POS_2952166 image by Peter Facey, CC BY-SA 2.0 via Geograph. Material references are source-assisted and visually corroborated, not laboratory-verified. No condition, corrosion, defect, failure, mechanical-integrity or safety-risk performance is claimed.</footer>
</div>
<script>
const DATA=$DATA_JSON;
const NS='http://www.w3.org/2000/svg';
const overlay=document.getElementById('overlay');
const note=document.getElementById('method-note');
const labels={siglip2_text_prompt_prototype:'Text-prompt prototype',siglip2_image_reference_prototype:'Image-reference prototype',siglip2_equal_weight_text_image_hybrid:'Equal-weight hybrid'};
function el(name,attrs={}){const n=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);return n;}
function render(key){
  overlay.innerHTML='';
  const preds=Object.fromEntries(DATA.methods[key].map(x=>[x.id,x]));
  for(const [id,box] of Object.entries(DATA.boxes)){
    const p=preds[id]; if(!p) continue;
    const [x1,y1,x2,y2]=box;
    const glass=p.predicted_label==='glass';
    const c=glass?'#45e39f':'#ffb35d';
    overlay.appendChild(el('rect',{x:x1,y:y1,width:x2-x1,height:y2-y1,fill:'none',stroke:c,'stroke-width':3,'vector-effect':'non-scaling-stroke'}));
    const label=(glass?'glass':'ceramic')+' '+(p.relative_score*100).toFixed(1)+'%';
    const tx=Math.max(4,x1), ty=Math.max(18,y1-7);
    overlay.appendChild(el('rect',{x:tx-2,y:ty-15,width:Math.max(74,label.length*6.4),height:18,rx:3,fill:'#071019','fill-opacity':.88,stroke:c,'stroke-width':1,'vector-effect':'non-scaling-stroke'}));
    const t=el('text',{x:tx,y:ty,fill:c,'font-size':12,'font-weight':800,'font-family':'ui-sans-serif,system-ui'});
    t.textContent=label; overlay.appendChild(t);
  }
  note.textContent=labels[key]+' · '+DATA.methods[key].filter(x=>x.correct).length+'/'+DATA.methods[key].length+' correct on the adaptive development source. Scores are relative and uncalibrated.';
  document.querySelectorAll('.method-card').forEach(b=>b.classList.toggle('selected',b.dataset.method===key));
}
document.querySelectorAll('.method-card').forEach(b=>b.addEventListener('click',()=>render(b.dataset.method)));
render('siglip2_text_prompt_prototype');
</script></body></html>''')

    page = template.substitute(
        WIDTH=width,
        HEIGHT=height,
        RAW_RECALL=pct(raw30["recall"]),
        GEO_RECALL=pct(geo30["recall"]),
        RAW_RECALL_NUM=f"{100 * raw30['recall']:.1f}",
        GEO_RECALL_NUM=f"{100 * geo30['recall']:.1f}",
        METHOD_CARDS=method_cards,
        TRAIN_D2=f"{shift['train_d2_max']:.3f}",
        SHIFT_D2=f"{shift['shifted_median_d2']:.3f}",
        TRAIN_WIDTH=f"{shift['train_width_ratio']:.3f}",
        SHIFT_WIDTH=f"{shift['shifted_width_ratio']:.3f}",
        DATA_JSON=data_json,
    )

    (SITE / "index.html").write_text(page, encoding="utf-8")
    (ASSETS / "material_results.json").write_text(json.dumps(material, indent=2) + "\n", encoding="utf-8")

    if (SITE / "index.html").stat().st_size > 250_000:
        raise RuntimeError("Online index unexpectedly large; keep heavy assets outside the HTML")
    print(json.dumps({
        "site": str(SITE),
        "index_bytes": (SITE / "index.html").stat().st_size,
        "webp_bytes": (ASSETS / "POS_2952166.webp").stat().st_size,
        "offline_bytes": (SITE / "offline.html").stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
