#!/usr/bin/env python3
"""Build an English evidence-separated multi-component report from a pinned real run."""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from prepare_keen_components import ROOT, digest, write_json

RUN = ROOT / "runs/uk_multicomponent/v1_20260830"
DEFAULT_OUT = ROOT / "runs/uk_capabilities/v3_20260827/report/multicomponent_v1"
COLOURS = {"pole": "#3b82f6", "crossarm": "#22c55e", "insulator": "#d946ef",
           "steelwork": "#f97316", "pole_top": "#facc15", "unknown": "#e5e7eb"}


def load(path):
    return json.loads(Path(path).read_text())


def font(size=14):
    for name in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_label(draw, box, text, colour):
    x0, y0 = int(box[0]), max(20, int(box[1]))
    active_font = font(13)
    bounds = draw.textbbox((0, 0), text, font=active_font)
    width = bounds[2] - bounds[0] + 8
    draw.rectangle((x0, y0 - 19, x0 + width, y0), fill="#07111f")
    draw.text((x0 + 4, y0 - 17), text, fill=colour, font=active_font)


def draw_box(draw, box, text, colour, width=3):
    draw.rectangle(box, outline=colour, width=width)
    draw_label(draw, box, text, colour)


def render_panel(image, record, panel, target):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    components = record["components"]
    if panel in {"components", "combined"}:
        for class_name in ("pole", "crossarm", "insulator"):
            for prediction in components[class_name]:
                draw_box(draw, prediction["xyxy"],
                         f"{class_name} · raw {prediction['raw_score']:.2f}", COLOURS[class_name])
    if panel in {"material", "combined"}:
        for material in record["material"]:
            prediction = components["insulator"][material["insulator_prediction_index"]]
            diagnostic = material["diagnostic_material"]
            detail = "material unknown"
            if diagnostic != "unknown":
                detail += f" · diagnostic {diagnostic.replace('_', '/')}"
            draw_box(draw, prediction["xyxy"], detail, COLOURS["unknown"], width=2)
    if panel in {"structure", "combined"}:
        for prediction in record["steelwork_candidates"]:
            draw_box(draw, prediction["box"],
                     f"steelwork candidate · raw {prediction['score']:.2f}", COLOURS["steelwork"])
        pole_top = record["pole_top"]
        if pole_top["status"] == "geometry_candidate":
            draw_box(draw, pole_top["xyxy"], "pole-top search region · unscored",
                     COLOURS["pole_top"], width=3)
    banner = {"components": "MODEL COMPONENT DETECTIONS",
              "material": "MATERIAL GATE · FINAL OUTPUT UNKNOWN",
              "structure": "STRUCTURE CANDIDATES · UNVERIFIED",
              "combined": "EVIDENCE-SEPARATED MULTI-COMPONENT OVERLAY"}[panel]
    draw.rectangle((0, 0, canvas.width, 28), fill="#07111f")
    draw.text((8, 7), banner, fill="white", font=font(14))
    canvas.save(target, quality=94)


def build(result_sha256, output=DEFAULT_OUT):
    result_path = RUN / "results.json"
    if digest(result_path) != result_sha256:
        raise ValueError("Result hash does not match the required real-run pin")
    result = load(result_path)
    if result["status"] != "COMPLETE_UNSCORED_MULTICOMPONENT_DIAGNOSTIC":
        raise ValueError("Expected a complete multi-component diagnostic")
    integrity = result["integrity"]
    required_false = ("v3_reference_boxes_accessed_or_used", "v3_roles_available_to_models",
                      "threshold_or_model_selection_from_v3", "outputs_are_calibrated_probabilities",
                      "multi_component_accuracy_computed", "steel_composition_verified",
                      "pole_top_is_physical_component_detection")
    if integrity["gradient_steps"] != 0 or any(integrity[key] for key in required_false):
        raise ValueError("Run integrity contract changed")
    if len(result["records"]) != 9 or result["performance_metrics"] is not None:
        raise ValueError("Expected nine unscored records")
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    (output / "raw").mkdir(exist_ok=True)
    gallery, totals = [], {name: 0 for name in
                           ("pole", "crossarm", "insulator", "material_diagnostics",
                            "steelwork_candidates", "pole_top_regions")}
    for result_record in result["records"]:
        record_path = RUN / result_record["record_file"]
        if digest(record_path) != result_record["record_sha256"]:
            raise ValueError(f"Record hash changed: {result_record['record_id']}")
        record = load(record_path)
        if (record["v3_reference_boxes_accessed_or_used"] or record["v3_role_written_to_output"] or
                record["performance_metrics"] is not None):
            raise ValueError("Record crosses the evidence boundary")
        source = record["source"]
        image_path = ROOT / source["image_file"]
        if digest(image_path) != source["image_sha256"]:
            raise ValueError(f"Source image changed: {source['record_id']}")
        image = Image.open(image_path).convert("RGB")
        panels = []
        for panel, label in (("components", "Pole, crossarm and insulator detections"),
                             ("material", "Material gate with final unknown output"),
                             ("structure", "Steelwork candidates and pole-top search region"),
                             ("combined", "Combined evidence-separated overlay")):
            target = output / "images" / f"{source['record_id']}_{panel}.jpg"
            render_panel(image, record, panel, target)
            panels.append({"label": label, "image": str(target.relative_to(output))})
        raw_target = output / "raw" / f"{source['record_id']}.json.txt"
        shutil.copyfile(record_path, raw_target)
        for key, value in result_record["counts"].items():
            totals[key] += value
        gallery.append({"record_id": source["record_id"], "photo_id": source["photo_id"],
                        "title": html.unescape(source["title"]), "author": source["author"],
                        "photo_page_url": source["photo_page_url"], "licence": source["licence"],
                        "licence_url": source["licence_url"], "image_sha256": source["image_sha256"],
                        "counts": result_record["counts"], "panels": panels,
                        "raw_record": str(raw_target.relative_to(output)),
                        "pole_top_status": record["pole_top"]["status"]})
    shutil.copyfile(result_path, output / "raw" / "results.json.txt")
    data = {"status": result["status"], "language": "English", "result_sha256": result_sha256,
            "job_id": result["runtime"]["job_id"], "git_commit": result["git_commit"],
            "gallery": gallery, "totals": totals, "performance_metrics": None,
            "claim_boundary": result["claim_boundary"], "integrity": integrity,
            "conclusion": "This page visualises real model outputs and explicit abstentions. It does not establish multi-component accuracy, verified material, steel composition or physical pole-top detection."}
    write_json(output / "data.json", data)
    options = "".join(f'<option value="{index}">{html.escape(row["title"])}</option>'
                      for index, row in enumerate(gallery))
    payload = json.dumps(gallery, ensure_ascii=False).replace("<", "\\u003c")
    page = f'''<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GridSight UK multi-component evidence overlay</title><style>
:root{{--bg:#07111f;--panel:#0f1c2e;--line:#2a3a52;--text:#edf4ff;--muted:#a7b6c9;--cyan:#00c2ff;--amber:#facc15}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1600px;margin:auto;padding:28px}}h1{{margin:.2rem 0}}p{{color:var(--muted)}}a{{color:var(--cyan)}}.warning,.panel,.metric{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}.warning{{border-left:5px solid var(--amber);margin:20px 0}}.metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}}.metric b{{display:block;font-size:24px}}.metric small{{color:var(--muted)}}.toolbar{{display:flex;gap:10px;align-items:center;margin:22px 0;flex-wrap:wrap}}button,select{{background:#14243a;color:var(--text);border:1px solid #40516c;border-radius:8px;padding:9px 13px}}select{{min-width:360px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}figure{{margin:0;background:#0a1626;border:1px solid var(--line);border-radius:10px;overflow:hidden}}figure img{{display:block;width:100%;height:auto}}figcaption{{padding:9px 12px;color:var(--muted)}}code{{word-break:break-all}}@media(max-width:1000px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}select{{min-width:0;width:100%}}}}</style></head><body><main>
<div>GridSight-UK · real-output evidence lab</div><h1>UK multi-component overlay</h1><p>Nine source-preserved UK images. Detector values are raw operating scores, not probabilities.</p>
<div class="warning"><b>This is a review overlay, not Keen-level validation.</b> Material remains unknown without independent instance evidence. Steelwork boxes are open-vocabulary candidates. Pole-top output is an unscored geometry search region. No multi-component accuracy is reported.</div>
<section class="metrics">{''.join(f'<div class="metric"><small>{key.replace("_", " ")}</small><b>{value}</b></div>' for key,value in totals.items())}</section>
<div class="toolbar"><button id="prev">← Previous</button><select id="pick">{options}</select><button id="next">Next →</button><span id="counter"></span></div>
<section class="panel"><h2 id="title"></h2><div id="meta"></div><div class="grid" id="grid"></div></section>
<p><a href="../localisation_adaptation_v2/index.html">UK insulator v2 audit</a> · <a href="../material_prospective/index.html">Source-evidenced material audit</a> · <a href="../upgrade/index.html">Capability gap summary</a> · <a href="raw/results.json.txt">Raw run result</a> · <a href="verification.json">Build verification</a> · <a href="ui_qa.json">Browser QA</a></p>
<script>const rows={payload};let i=0;const pick=document.querySelector('#pick');function show(n){{i=(n+rows.length)%rows.length;pick.value=i;const r=rows[i];document.querySelector('#counter').textContent=`${{i+1}} / ${{rows.length}}`;document.querySelector('#title').textContent=r.title;document.querySelector('#meta').innerHTML=`${{r.author}} · <a href="${{r.photo_page_url}}">source page</a> · <a href="${{r.licence_url}}">${{r.licence}}</a> · pole-top ${{r.pole_top_status}} · <a href="${{r.raw_record}}">raw record</a><br><code>SHA-256 ${{r.image_sha256}}</code>`;document.querySelector('#grid').innerHTML=r.panels.map(p=>`<figure><img src="${{p.image}}" alt="${{p.label}}"><figcaption>${{p.label}}</figcaption></figure>`).join('')}}document.querySelector('#prev').onclick=()=>show(i-1);document.querySelector('#next').onclick=()=>show(i+1);pick.onchange=()=>show(+pick.value);document.addEventListener('keydown',e=>{{if(e.key==='ArrowLeft')show(i-1);if(e.key==='ArrowRight')show(i+1)}});show(0);</script></main></body></html>'''
    if any("\u3400" <= character <= "\u9fff" for character in page):
        raise ValueError("Rendered UI is not English-only")
    (output / "index.html").write_text(page)
    verification = {"status": "VERIFIED", "language": "English", "gallery_images": len(gallery),
                    "panels": 4 * len(gallery), "result_sha256": result_sha256,
                    "source_predictions_rewritten": False, "reference_boxes_used": False,
                    "scores_presented_as_probabilities": False,
                    "data_sha256": digest(output / "data.json"),
                    "html_sha256": digest(output / "index.html")}
    write_json(output / "verification.json", verification)
    return data, verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    arguments = parser.parse_args()
    _, verified = build(arguments.result_sha256, arguments.output)
    print(json.dumps(verified, indent=2))
