#!/usr/bin/env python3
"""Build the English audit for the development-only crossarm guardrail."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/crossarm_association/v2_20260831"
POLE_TOP_RUN = ROOT / "runs/pole_top_development/v2_20260831"
SOURCE = ROOT / "data/external/uk_distribution_pilot_v1"
OUT = ROOT / "runs/uk_capabilities/v3_20260827/report/crossarm_association_v2"
RESULT_SHA = "1f3094c5cd85d79bbf9804176d0714e4eef8ff9bfe9066e3684a4a343d98949d"
POLE_TOP_RESULT_SHA = "ef371bd66e45df17ddbdbba133ad76fde03be98d72aa63e5c121b78fb4419beb"


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path):
    return json.loads(Path(path).read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def font(size=14):
    for name in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def box(draw, xyxy, label, colour, width):
    draw.rectangle(xyxy, outline=colour, width=width)
    x, y = int(xyxy[0]), max(22, int(xyxy[1]))
    bounds = draw.textbbox((0, 0), label, font=font(13))
    draw.rectangle((x, y - 20, x + bounds[2] - bounds[0] + 8, y), fill="#07111f")
    draw.text((x + 4, y - 18), label, fill=colour, font=font(13))


def render(source, poles, predictions, title, target, pole_top=None):
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(3, image.width // 400)
    for row in poles:
        box(draw, row["box"], f"pole input · raw {row['score']:.3f}", "#fbbf24", width)
    for row in predictions:
        box(draw, row["box"], f"crossarm candidate · raw {row['score']:.3f}", "#00e5d4", width + 1)
    if pole_top and pole_top["status"] == "geometry_candidate":
        box(draw, pole_top["xyxy"], "pole-top region · geometry · unscored", "#c084fc", width + 1)
    draw.rectangle((0, 0, image.width, 34), fill="#07111f")
    draw.text((10, 8), title, fill="white", font=font(15))
    if not predictions:
        draw.text((14, 48), "No displayed proposal", fill="#facc15", font=font(16),
                  stroke_width=2, stroke_fill="#07111f")
    image.save(target, quality=92)


def verify():
    result = load(RUN / "results.json")
    if sha(RUN / "results.json") != RESULT_SHA:
        raise ValueError("Pinned guardrail result changed")
    if result["status"] != "COMPLETE_DEVELOPMENT_MORPHOLOGY_GUARDRAIL":
        raise ValueError("Guardrail status changed")
    if result["model_inference"] or result["uk_v3_accessed"] or result["uk_ground_truth_used"]:
        raise ValueError("Execution/evidence boundary changed")
    if result["uk_development"]["accuracy"] is not None:
        raise ValueError("UK accuracy must remain unknown")
    if (result["uk_development"]["before_proposals"],
            result["uk_development"]["after_proposals"]) != (3, 1):
        raise ValueError("Pinned proposal counts changed")
    for relative, expected in result["source_snapshots"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Executed source changed: {relative}")
    checked = 1
    for item in result["uk_records"]:
        path = RUN / item["record_file"]
        if sha(path) != item["record_sha256"]:
            raise ValueError(f"Guardrail record changed: {path}")
        record = load(path)
        if (record["ground_truth_status"] != "NONE" or record["accuracy"] is not None
                or record["scores_are_probabilities"] or record["reference_truth"]):
            raise ValueError("Record claim boundary changed")
        checked += 1
    pole_top = load(POLE_TOP_RUN / "results.json")
    if sha(POLE_TOP_RUN / "results.json") != POLE_TOP_RESULT_SHA:
        raise ValueError("Pinned pole-top result changed")
    if (pole_top["status"] != "COMPLETE_DEVELOPMENT_POLE_TOP_GEOMETRY" or
            pole_top["model_inference"] or pole_top["uk_v3_accessed"] or
            pole_top["uk_ground_truth_used"] or pole_top["uk_development"]["accuracy"] is not None):
        raise ValueError("Pole-top evidence boundary changed")
    pole_top_records = {}
    for item in pole_top["records"]:
        path = POLE_TOP_RUN / item["record_file"]
        if sha(path) != item["record_sha256"]:
            raise ValueError(f"Pole-top record changed: {path}")
        pole_top_records[item["image_id"]] = load(path)
        checked += 1
    return result, pole_top, pole_top_records, checked


def build(output=OUT):
    result, pole_top_result, pole_top_records, checked = verify()
    source_manifest = load(SOURCE / "manifest.json")
    sources = {row["image_id"]: row for row in source_manifest["images"]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    gallery, artifacts = [], {}
    for item in result["uk_records"]:
        record = load(RUN / item["record_file"])
        source = sources[record["image_id"]]
        source_path = SOURCE / source["image_file"]
        if sha(source_path) != source["sha256"] or record["image_sha256"] != source["sha256"]:
            raise ValueError(f"UK source changed: {record['image_id']}")
        before = output / "images" / f"{record['image_id']}_before.jpg"
        after = output / "images" / f"{record['image_id']}_guarded.jpg"
        pole_top = pole_top_records[record["image_id"]]["pole_top"]
        render(source_path, record["input_poles"], record["before_predictions"],
               "V1 · EPRI-SELECTED ASSOCIATION", before)
        render(source_path, record["input_poles"], record["guarded_predictions"],
               "V2 · GUARDED CROSSARM + UNSCORED POLE-TOP REGION", after, pole_top)
        artifacts[str(before.relative_to(output))] = sha(before)
        artifacts[str(after.relative_to(output))] = sha(after)
        gallery.append({
            "image_id": record["image_id"], "title": source["title"],
            "author": source["author"], "source_page": source["source_page"],
            "licence": source["license"], "licence_url": source["license_url"],
            "source_sha256": source["sha256"], "before_count": len(record["before_predictions"]),
            "after_count": len(record["guarded_predictions"]),
            "before": str(before.relative_to(output)), "after": str(after.relative_to(output)),
            "pole_top_status": pole_top["status"],
            "ground_truth_status": "NONE", "accuracy": None,
        })
    data = {
        "schema": "gridsight-crossarm-association-v2-audit", "language": "English",
        "status": result["status"], "result_sha256": RESULT_SHA,
        "model_inference": False, "epri_fixed_rule_metrics": result["epri_fixed_rule_metrics"],
        "input_epri_metrics": result["input_epri_selected_metrics"],
        "uk_development": result["uk_development"],
        "pole_top_development": pole_top_result["uk_development"],
        "pole_top_result_sha256": POLE_TOP_RESULT_SHA,
        "decision": "Use the guardrail and pole-top region only as development display aids. They remove the two audited morphology failures and retain one plausible compact support proposal plus one unscored geometry region, but UK accuracy and recall remain unknown. Do not promote them to UK v3 without a separately frozen comparison.",
        "gallery": gallery,
    }
    write(output / "data.json", data)
    payload = json.dumps(gallery, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    options = "".join(f'<option value="{i}">{html.escape(row["title"])}</option>' for i, row in enumerate(gallery))
    epri = result["epri_fixed_rule_metrics"]
    original = result["input_epri_selected_metrics"]
    page = f'''<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Crossarm association v2 audit</title><style>
:root{{--bg:#07111f;--panel:#0f1c2e;--line:#2a3a52;--text:#edf4ff;--muted:#a7b6c9;--cyan:#00e5d4;--amber:#facc15;--red:#fb7185}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin:.2rem 0}}p{{color:var(--muted)}}a{{color:#67e8f9}}.warning,.panel,.metric{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}.warning{{border-left:5px solid var(--amber);margin:20px 0}}.metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}}.metric b{{display:block;font-size:22px}}.metric small{{color:var(--muted)}}.toolbar{{display:flex;gap:10px;align-items:center;margin:22px 0;flex-wrap:wrap}}button,select{{background:#14243a;color:var(--text);border:1px solid #40516c;border-radius:8px;padding:9px 13px}}select{{min-width:420px;max-width:70%}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}figure{{margin:0;background:#0a1626;border:1px solid var(--line);border-radius:10px;overflow:hidden}}figure img{{display:block;width:100%;height:auto}}figcaption{{padding:9px 12px;color:var(--muted)}}code{{word-break:break-all}}@media(max-width:950px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}select{{min-width:0;max-width:none;width:100%}}}}</style></head><body><main>
<div>GridSight-UK · development morphology audit</div><h1>Crossarm association v2</h1><p>A deterministic guardrail applied to the pinned Grounding DINO records. No model inference, training or UK v3 access occurred.</p>
<div class="warning"><b>A cleaner overlay is not a new accuracy result.</b> The guardrail removes two morphology failures from the 27-image UK development set and retains one compact support proposal. That proposal enables one purple pole-top assembly search region, which is geometry-derived, unscored and not a verified physical target. There are no UK reference boxes.</div>
<section class="metrics"><div class="metric"><small>V1 EPRI precision</small><b>{original['precision']:.3f}</b></div><div class="metric"><small>V2 EPRI precision</small><b>{epri['precision']:.3f}</b></div><div class="metric"><small>V1 EPRI recall</small><b>{original['recall']:.3f}</b></div><div class="metric"><small>V2 EPRI recall</small><b>{epri['recall']:.3f}</b></div><div class="metric"><small>UK proposals</small><b>3 → 1</b></div><div class="metric"><small>Pole-top regions</small><b>1 / 27</b></div></section>
<section class="panel" style="margin-top:14px"><h2>What changed</h2><p>The display now requires an upright pole input (height/width at least 1.5) and rejects candidate boxes taller than 45% of the associated pole box. These rules were fixed after the v1 UK-development failure audit, so this page is a development comparison, not an independent test.</p><p><b>Trade-off on EPRI circuit 4:</b> false positives fall from {original['fp']} to {epri['fp']}; true positives fall from {original['tp']} to {epri['tp']}. The filter prioritises morphology and precision at the cost of recall.</p><p><b>Pole-top semantics:</b> the purple box is centred on a guarded crossarm near the unambiguous upper endpoint of one upright pole. It carries no score and must not be read as a pole-top detection.</p></section>
<div class="toolbar"><button id="prev">← Previous</button><select id="pick">{options}</select><button id="next">Next →</button><span id="counter"></span></div>
<section class="panel"><h2 id="title"></h2><div id="meta"></div><div class="grid"><figure><img id="before"><figcaption id="beforeCap"></figcaption></figure><figure><img id="after"><figcaption id="afterCap"></figcaption></figure></div></section>
<p><a href="../crossarm_grounding_v1/index.html">Grounding DINO v1 audit</a> · <a href="../multicomponent_v1/index.html">Current UK multi-component overlay</a> · <a href="data.json">Report data</a> · <a href="verification.json">Build verification</a></p>
<script>const rows={payload};let i=0;const pick=document.querySelector('#pick');function link(t,u){{const a=document.createElement('a');a.textContent=t;a.href=u;return a}}function show(n){{i=(n+rows.length)%rows.length;pick.value=i;const r=rows[i];document.querySelector('#counter').textContent=`${{i+1}} / ${{rows.length}}`;document.querySelector('#title').textContent=r.title;const m=document.querySelector('#meta');m.replaceChildren(document.createTextNode(`${{r.author}} · `),link('source page',r.source_page),document.createTextNode(' · '),link(r.licence,r.licence_url),document.createElement('br'));const c=document.createElement('code');c.textContent=`SHA-256 ${{r.source_sha256}} · UK truth NONE · accuracy unavailable`;m.append(c);document.querySelector('#before').src=r.before;document.querySelector('#after').src=r.after;document.querySelector('#beforeCap').textContent=`V1 proposals: ${{r.before_count}}`;document.querySelector('#afterCap').textContent=`V2 guarded proposals: ${{r.after_count}} · pole-top: ${{r.pole_top_status==='geometry_candidate'?'geometry / unscored':'abstained'}}`}}document.querySelector('#prev').onclick=()=>show(i-1);document.querySelector('#next').onclick=()=>show(i+1);pick.onchange=()=>show(+pick.value);document.addEventListener('keydown',e=>{{if(e.key==='ArrowLeft')show(i-1);if(e.key==='ArrowRight')show(i+1)}});show(0);</script></main></body></html>'''
    if any("\u3400" <= character <= "\u9fff" for character in page):
        raise ValueError("Rendered page is not English-only")
    (output / "index.html").write_text(page)
    artifacts["data.json"] = sha(output / "data.json")
    artifacts["index.html"] = sha(output / "index.html")
    verification = {
        "status": "VERIFIED_CROSSARM_ASSOCIATION_V2_DEVELOPMENT_AUDIT",
        "result_sha256": RESULT_SHA, "pole_top_result_sha256": POLE_TOP_RESULT_SHA,
        "verified_guardrail_and_pole_top_records": checked,
        "model_inference": False, "uk_v3_accessed": False, "uk_ground_truth_used": False,
        "uk_accuracy_reported": False, "scores_presented_as_probabilities": False,
        "promoted_to_main_overlay": False, "artifacts": artifacts,
    }
    write(output / "verification.json", verification)
    return data, verification


if __name__ == "__main__":
    data, verification = build()
    print(json.dumps({"status": verification["status"], "images": len(data["gallery"]),
                      "html_sha256": sha(OUT / "index.html")}, indent=2))
