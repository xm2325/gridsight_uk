#!/usr/bin/env python3
"""Build an English evidence-separated multi-component report from a pinned real run."""
from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageDraw, ImageFont

from prepare_keen_components import ROOT, digest, write_json

RUN = ROOT / "runs/uk_multicomponent/v1_20260830"
DEFAULT_OUT = ROOT / "runs/uk_capabilities/v3_20260827/report/multicomponent_v1"
PROTOCOL = ROOT / "configs/uk_multicomponent_inference_v1.json"
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


def checked_score(value, label):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid raw score for {label}")
    return value


def checked_source_url(value, label):
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid source URL for {label}")


def validate_record_contract(result_record, record):
    """Reject any record that turns a diagnostic or candidate into truth."""
    if record.get("schema") != "gridsight-uk-multicomponent-v1":
        raise ValueError("Unexpected multi-component record schema")
    source = record["source"]
    if source["record_id"] != result_record["record_id"]:
        raise ValueError("Result/record identity mismatch")
    checked_source_url(source["photo_page_url"], "photo page")
    checked_source_url(source["licence_url"], "licence")
    if (record["v3_reference_boxes_accessed_or_used"] or
            record["v3_role_written_to_output"] or
            record["performance_metrics"] is not None):
        raise ValueError("Record crosses the evidence boundary")

    components = record["components"]
    if set(components) != {"pole", "crossarm", "insulator"}:
        raise ValueError("Unexpected component classes")
    for class_name, predictions in components.items():
        for prediction in predictions:
            observed = prediction.get("class_name", prediction.get("source_class_name"))
            if observed != class_name:
                raise ValueError(f"Component class mismatch for {class_name}")
            checked_score(prediction["raw_score"], class_name)
            if prediction.get("calibrated_probability") is not False:
                raise ValueError("Component score was presented as a probability")
            if prediction.get("reference_truth") is not False:
                raise ValueError("Component prediction was presented as truth")
    for prediction in record["raw_component_predictions"]:
        if prediction["class_name"] not in components:
            raise ValueError("Unexpected raw component class")
        checked_score(prediction["raw_score"], "raw component")
        if (prediction.get("calibrated_probability") is not False or
                prediction.get("reference_truth") is not False):
            raise ValueError("Raw component prediction was presented as truth or probability")

    material_rows = record["material"]
    indices = [row["insulator_prediction_index"] for row in material_rows]
    if sorted(indices) != list(range(len(components["insulator"]))):
        raise ValueError("Material diagnostics do not cover insulator predictions exactly once")
    for row in material_rows:
        decision = row["diagnostic_decision"]
        if (row["final_material"] != "unknown" or row["material_verified"] is not False or
                row["scores_are_probabilities"] is not False or
                decision["material_verified"] is not False or
                decision["scores_are_probabilities"] is not False or
                row["diagnostic_material"] != decision["material"]):
            raise ValueError("Material evidence/abstention contract changed")

    for prediction in [*record["raw_steelwork_candidates"],
                       *record["steelwork_candidates"]]:
        checked_score(prediction["score"], "steelwork candidate")
        if (prediction["label"] != "steelwork candidate" or
                prediction["steel_composition_verified"] is not False or
                prediction["calibrated_probability"] is not False or
                prediction["reference_truth"] is not False):
            raise ValueError("Steelwork candidate was presented as verified truth")

    pole_top = record["pole_top"]
    if (pole_top["status"] not in {"unknown", "geometry_candidate"} or
            pole_top["score"] is not None or pole_top["derived"] is not True or
            pole_top["physical_component_verified"] is not False):
        raise ValueError("Pole-top search region was presented as a scored physical component")
    if ((pole_top["status"] == "geometry_candidate") != (pole_top["xyxy"] is not None)):
        raise ValueError("Pole-top geometry/status mismatch")

    actual_counts = {
        "pole": len(components["pole"]),
        "crossarm": len(components["crossarm"]),
        "insulator": len(components["insulator"]),
        "material_diagnostics": len(material_rows),
        "steelwork_candidates": len(record["steelwork_candidates"]),
        "pole_top_regions": int(pole_top["status"] == "geometry_candidate"),
    }
    if result_record["counts"] != actual_counts:
        raise ValueError("Reported component totals do not match the raw record")
    return actual_counts


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
    protocol = load(PROTOCOL)
    if result["protocol_sha256"] != digest(PROTOCOL) or result["protocol"] != protocol:
        raise ValueError("Run protocol does not match the pinned report protocol")
    if digest(RUN / "frozen_choices.json") != result["frozen_choices_sha256"]:
        raise ValueError("Frozen choices changed")
    for source_name, expected in result["source_snapshots"].items():
        if digest(RUN / "code" / Path(source_name).name) != expected:
            raise ValueError(f"Run code snapshot changed: {source_name}")
    if digest(RUN / "material_features.npz") != result["material_features_sha256"]:
        raise ValueError("Material feature tensor changed")
    integrity = result["integrity"]
    required_false = ("v3_reference_boxes_accessed_or_used", "v3_roles_available_to_models",
                      "threshold_or_model_selection_from_v3", "outputs_are_calibrated_probabilities",
                      "multi_component_accuracy_computed", "steel_composition_verified",
                      "pole_top_is_physical_component_detection")
    if integrity["gradient_steps"] != 0 or any(integrity[key] for key in required_false):
        raise ValueError("Run integrity contract changed")
    record_ids = [record["record_id"] for record in result["records"]]
    if (len(record_ids) != 9 or len(set(record_ids)) != 9 or
            result["performance_metrics"] is not None):
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
        validate_record_contract(result_record, record)
        preserved = ROOT / record["preserved_insulator_prediction_file"]
        if digest(preserved) != record["preserved_insulator_prediction_sha256"]:
            raise ValueError(f"Preserved insulator prediction changed: {result_record['record_id']}")
        for raw_file in record["steelwork_raw_files"]:
            if digest(RUN / raw_file["file"]) != raw_file["sha256"]:
                raise ValueError(f"Raw steelwork tensor changed: {result_record['record_id']}")
        source = record["source"]
        image_path = ROOT / source["image_file"]
        if not image_path.resolve().is_relative_to((ROOT / protocol["source_dataset"]).resolve()):
            raise ValueError(f"Source image is outside the pinned cohort: {source['record_id']}")
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
    payload = json.dumps(gallery, ensure_ascii=True).replace("<", "\\u003c")
    page = f'''<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GridSight UK multi-component evidence overlay</title><style>
:root{{--bg:#07111f;--panel:#0f1c2e;--line:#2a3a52;--text:#edf4ff;--muted:#a7b6c9;--cyan:#00c2ff;--amber:#facc15}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1600px;margin:auto;padding:28px}}h1{{margin:.2rem 0}}p{{color:var(--muted)}}a{{color:var(--cyan)}}.warning,.panel,.metric{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}.warning{{border-left:5px solid var(--amber);margin:20px 0}}.metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}}.metric b{{display:block;font-size:24px}}.metric small{{color:var(--muted)}}.toolbar{{display:flex;gap:10px;align-items:center;margin:22px 0;flex-wrap:wrap}}button,select{{background:#14243a;color:var(--text);border:1px solid #40516c;border-radius:8px;padding:9px 13px}}select{{min-width:360px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}figure{{margin:0;background:#0a1626;border:1px solid var(--line);border-radius:10px;overflow:hidden}}figure img{{display:block;width:100%;height:auto}}figcaption{{padding:9px 12px;color:var(--muted)}}code{{word-break:break-all}}@media(max-width:1000px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}select{{min-width:0;width:100%}}}}</style></head><body><main>
<div>GridSight-UK · real-output evidence lab</div><h1>UK multi-component overlay</h1><p>Nine source-preserved UK images. Detector values are raw operating scores, not probabilities.</p>
<div class="warning"><b>This is a review overlay, not Keen-level validation.</b> Material remains unknown without independent instance evidence. Steelwork boxes are open-vocabulary candidates. Pole-top output is an unscored geometry search region. No multi-component accuracy is reported.</div>
<section class="metrics">{''.join(f'<div class="metric"><small>{key.replace("_", " ")}</small><b>{value}</b></div>' for key,value in totals.items())}</section>
<div class="toolbar"><button id="prev">← Previous</button><select id="pick">{options}</select><button id="next">Next →</button><span id="counter"></span></div>
<section class="panel"><h2 id="title"></h2><div id="meta"></div><div class="grid" id="grid"></div></section>
<p><a href="../localisation_adaptation_v2/index.html">UK insulator v2 audit</a> · <a href="../material_prospective/index.html">Source-evidenced material audit</a> · <a href="../upgrade/index.html">Capability gap summary</a> · <a href="raw/results.json.txt">Raw run result</a> · <a href="verification.json">Build verification</a> · <a href="ui_qa.json">Browser QA</a></p>
<script>const rows={payload};let i=0;const pick=document.querySelector('#pick');function link(label,url){{const parsed=new URL(url,location.href);if(!['http:','https:'].includes(parsed.protocol))throw new Error('Unsafe report URL');const a=document.createElement('a');a.href=url;a.textContent=label;return a}}function show(n){{i=(n+rows.length)%rows.length;pick.value=i;const r=rows[i];document.querySelector('#counter').textContent=`${{i+1}} / ${{rows.length}}`;document.querySelector('#title').textContent=r.title;const meta=document.querySelector('#meta');meta.replaceChildren(document.createTextNode(`${{r.author}} · `),link('source page',r.photo_page_url),document.createTextNode(' · '),link(r.licence,r.licence_url),document.createTextNode(` · pole-top ${{r.pole_top_status}} · `),link('raw record',r.raw_record),document.createElement('br'));const code=document.createElement('code');code.textContent=`SHA-256 ${{r.image_sha256}}`;meta.append(code);const grid=document.querySelector('#grid');grid.replaceChildren(...r.panels.map(p=>{{const figure=document.createElement('figure');const image=document.createElement('img');image.src=p.image;image.alt=p.label;const caption=document.createElement('figcaption');caption.textContent=p.label;figure.append(image,caption);return figure}}))}}document.querySelector('#prev').onclick=()=>show(i-1);document.querySelector('#next').onclick=()=>show(i+1);pick.onchange=()=>show(+pick.value);document.addEventListener('keydown',e=>{{if(e.key==='ArrowLeft')show(i-1);if(e.key==='ArrowRight')show(i+1)}});show(0);</script></main></body></html>'''
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
