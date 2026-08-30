#!/usr/bin/env python3
"""Build the English UK prospective material-transfer evidence gallery.

This is presentation-only code. It reads the frozen source manifest and preserved
Roihu outputs; it does not run a model or alter any prediction.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from html import escape as escape_html

from PIL import Image, ImageDraw, ImageFont

from prepare_keen_components import ROOT, digest, write_json


RUN = ROOT / "runs/material_head/v3_uk_prospective_20260830"
SOURCE = ROOT / "data/external/uk_material_prospective_v1"
REPORT = ROOT / "runs/uk_capabilities/v3_20260827/report/material_prospective"
EXPECTED = {
    "results.json": "f959a8a7ea8e1b6f476567b8a01833e88249ab50e8358a3becc605d56c10b6f5",
    "oracle_decisions.json": "be97884b5260cd7b014385967ec0579d01cd9036dd4193cf5e031f83e4418ca7",
    "component_predictions.json": "52e89c2735134cc76872d1030c5879fd9e55c9916644bd7e9b62ca661d1fd2fe",
}
MANIFEST_SHA = "54fd6a24adc1e49a4c1af7cf21338d1bfa22b9382f7e5ee4a315fae22eb3a45c"


def load(path: Path):
    return json.loads(path.read_text())


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def banner(draw, width, title, detail, colour):
    draw.rectangle((0, 0, width, 78), fill=(8, 20, 34, 232))
    draw.text((16, 10), title, font=font(21, True), fill=colour)
    draw.text((16, 43), detail, font=font(14), fill="white")


def box(draw, xyxy, colour, label, width=4):
    x1, y1, x2, y2 = map(float, xyxy)
    draw.rectangle((x1, y1, x2, y2), outline=colour, width=width)
    label_w = max(85, 8 * len(label) + 10)
    top = max(80, y1 - 25)
    draw.rectangle((x1, top, min(x1 + label_w, draw._image.width), top + 24), fill=(8, 20, 34, 220))
    draw.text((x1 + 5, top + 3), label, font=font(13, True), fill=colour)


def decision_colour(expected, predicted):
    if predicted == "unknown":
        return "#f2b84b"
    return "#35d391" if predicted == expected else "#f26568"


def source_panel(image, record):
    panel = image.copy()
    draw = ImageDraw.Draw(panel, "RGBA")
    for i, region in enumerate(record["regions"], 1):
        box(draw, region, "#32d5e1", f"source region {i}")
    banner(draw, panel.width, "SOURCE-EVIDENCED REGIONS",
           "publisher text supports material; rectangles are analyst regions", "#82eef4")
    return panel


def oracle_panel(image, record, rows, title):
    panel = image.copy()
    draw = ImageDraw.Draw(panel, "RGBA")
    accepted = correct = 0
    for row in rows:
        predicted = row["decision"]["material"]
        expected = row["expected_material"]
        if predicted != "unknown":
            accepted += 1
            correct += int(predicted == expected)
        label = predicted.replace("porcelain_ceramic", "porcelain")
        box(draw, row["xyxy"], decision_colour(expected, predicted), label)
    detail = f"{correct}/{accepted} accepted correct" if accepted else "all regions rejected"
    banner(draw, panel.width, title, detail + " · output scores are not probabilities", "#b9d9ff")
    return panel


def detector_panel(image, record, component):
    panel = image.copy()
    draw = ImageDraw.Draw(panel, "RGBA")
    for region in record["regions"]:
        draw.rectangle(tuple(region), outline="#32d5e1", width=2)
    matches = {m["prediction_index"]: m for m in component["matches"]}
    material = {row["prediction_index"]: row["decision"]["material"]
                for row in component["material"]["adapted"]}
    for prediction in component["accepted_insulator_predictions"]:
        idx = prediction["prediction_index"]
        matched = idx in matches
        colour = "#35d391" if matched else "#f26568"
        label = material.get(idx, "unmatched").replace("porcelain_ceramic", "porcelain")
        box(draw, prediction["box"], colour, f"detector {label}")
    banner(draw, panel.width, "END-TO-END DETECTOR",
           f"{len(matches)} matched / {len(record['regions'])} regions · {len(component['accepted_insulator_predictions'])} accepted boxes", "#d8c4ff")
    return panel


def montage(panels):
    target_w = 620
    resized = []
    for panel in panels:
        scale = target_w / panel.width
        resized.append(panel.resize((target_w, round(panel.height * scale)), Image.Resampling.LANCZOS))
    target_h = max(p.height for p in resized)
    canvas = Image.new("RGB", (target_w * 2, target_h * 2), "#091523")
    for i, panel in enumerate(resized):
        canvas.paste(panel, ((i % 2) * target_w, (i // 2) * target_h))
    return canvas


def html(data):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return """<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GridSight · UK material prospective audit</title>
<style>:root{--ink:#172c43;--muted:#62758a;--line:#d9e3ec;--bg:#eef3f7;--navy:#102238}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{background:var(--navy);color:white;padding:24px}header p{margin:6px 0;color:#c7d6e6}.wrap{max-width:1450px;margin:auto;padding:20px}.panel,.card{background:white;border:1px solid var(--line);border-radius:12px;padding:17px}.panel{margin-bottom:18px}.warning{background:#fff3db;border-color:#e5c981}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px;margin-bottom:18px}.card strong{display:block;font-size:27px}.muted,small{color:var(--muted)}.viewer img{width:100%;max-height:900px;object-fit:contain;background:#091523;border-radius:8px}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:12px 0}button,select{font:inherit;padding:8px 11px;border:1px solid #c5d3df;border-radius:7px;background:white;color:var(--ink)}select{min-width:370px;max-width:100%}.legend{display:flex;gap:16px;flex-wrap:wrap}.dot{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px}table{border-collapse:collapse;width:100%}th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);padding:8px}code{background:#edf2f6;padding:2px 5px;border-radius:4px}a{color:#1e64bd}@media(max-width:800px){.cards{grid-template-columns:1fr}.wrap{padding:12px}select{min-width:0;width:100%}}</style></head><body>
<header><h1>UK insulator material · prospective transfer audit</h1><p>Frozen source evidence, oracle-region material decisions and actual full-image detector output</p></header><main class="wrap">
<section class="panel warning"><strong>Useful material transfer, failed end-to-end localisation.</strong><p>On 18 source-assisted analyst regions, the adapted head accepted 14 and classified 12 correctly. The existing component detector matched only 1 of 18 regions, so this is not a deployable Keen-style pipeline. Publisher text supports each asset's material; the rectangles are not expert inspection ground truth. There are no UK polymer targets.</p><a href="../upgrade/index.html">← Capability upgrade</a> · <a href="../index.html">UK review workbench</a></section>
<section class="cards"><div class="card"><small>Oracle regions · adapted</small><strong>12 / 14</strong><p>85.7% accepted accuracy at 77.8% region coverage. Baseline: 6/8 at 44.4% coverage.</p></div><div class="card"><small>Full-image localisation</small><strong>1 / 18</strong><p>5.6% reference-region coverage at the frozen score 0.25 and IoU 0.30.</p></div><div class="card"><small>Prospective assets</small><strong>5 groups</strong><p>Three glass and two porcelain/ceramic source-evidenced assets, disjoint from three adaptation groups.</p></div></section>
<section class="panel"><h2>All five prospective UK assets</h2><p class="muted">Top left: source-assisted regions. Top right: frozen v2 head. Bottom left: adapted final-layer head. Bottom right: actual EPRI component detector. Cyan is a reference region; green is correct/matched, red is wrong/unmatched, and yellow is rejected. Labels are categorical decisions, never probabilities.</p><div class="legend"><span><i class="dot" style="background:#32d5e1"></i>source-assisted region</span><span><i class="dot" style="background:#35d391"></i>correct or matched</span><span><i class="dot" style="background:#f26568"></i>wrong or unmatched</span><span><i class="dot" style="background:#f2b84b"></i>unknown / rejected</span></div><div class="toolbar"><button id="prev">← Previous</button><select id="select"></select><button id="next">Next →</button><span id="counter"></span></div><div class="viewer"><img id="image" alt="UK material transfer audit"></div><h3 id="title"></h3><p id="caption"></p><p id="links"></p></section>
<section class="panel"><h2>What the result changes</h2><table><thead><tr><th>Question</th><th>Verified answer</th><th>Next bounded upgrade</th></tr></thead><tbody><tr><td>Can the material head transfer?</td><td>Partly. Adaptation raised accepted region coverage from 44.4% to 77.8% and accepted accuracy from 75.0% to 85.7% on this five-asset diagnostic.</td><td>Acquire independent UK polymer/composite evidence and a larger untouched asset-group test before calibration claims.</td></tr><tr><td>Can it reproduce the Keen-style full-image output?</td><td>No. The current detector matched 1/18 analyst regions; seven accepted detector boxes were unmatched.</td><td>Train or adapt a material-agnostic UK insulator localiser using full-image labels, small-object tiling and pole/crossarm context. Keep the material head separate.</td></tr><tr><td>Can the displayed numbers be called confidence?</td><td>No. Detector operating scores and classifier margins are not calibrated probabilities.</td><td>Calibrate only on a separate target-domain development set, then publish coverage/error curves and preserve unknown.</td></tr></tbody></table></section>
<section class="panel"><h2>Frozen protocol</h2><p>SigLIP2 encoder gradient steps: <strong>0</strong>. Final-head gradient steps: <strong>120</strong>. Test used for training or model selection: <strong>no</strong>. Roihu job: <code>944218</code>, NVIDIA GH200, completed in 19 seconds (Python workflow 3.10 seconds).</p><p><a href="raw/results.html">Raw metrics</a> · <a href="raw/oracle_decisions.html">Raw oracle decisions</a> · <a href="raw/component_predictions.html">Raw detector predictions</a> · <a href="verification.json">Build verification</a></p></section>
<footer class="muted">No UK population accuracy, deployment reliability, calibrated probability or polymer performance is claimed. Source image bytes, publisher pages, author, licence, URLs and hashes are retained.</footer></main>
<script id="payload" type="application/json">""" + payload + """</script><script>'use strict';const D=JSON.parse(document.getElementById('payload').textContent);let i=0,s=document.getElementById('select'),img=document.getElementById('image'),count=document.getElementById('counter'),title=document.getElementById('title'),caption=document.getElementById('caption'),links=document.getElementById('links');D.gallery.forEach((r,j)=>{let o=document.createElement('option');o.value=j;o.textContent=r.record_id+' · '+r.expected_material;s.append(o)});function show(n){i=(n+D.gallery.length)%D.gallery.length;let r=D.gallery[i];s.value=i;img.src=r.image;count.textContent=(i+1)+' / '+D.gallery.length;title.textContent=r.title;caption.textContent='Expected from publisher text: '+r.expected_material+' · baseline asset decision: '+r.baseline+' · adapted asset decision: '+r.adapted+' · detector matched '+r.matched+'/'+r.regions+' regions.';links.replaceChildren();let a=document.createElement('a');a.href=r.photo_page_url;a.target='_blank';a.rel='noopener';a.textContent='Publisher photo page';links.append(a,document.createTextNode(' · '+r.author+' · '+r.licence+' · Evidence: “'+r.evidence_excerpt+'”'))}document.getElementById('prev').onclick=()=>show(i-1);document.getElementById('next').onclick=()=>show(i+1);s.onchange=()=>show(Number(s.value));show(0);</script></body></html>"""


def build():
    assert digest(SOURCE / "manifest.json") == MANIFEST_SHA
    # The results checksum is recorded in the result itself and checked separately
    # because presentation code must never rewrite source predictions.
    assert digest(RUN / "results.json") == EXPECTED["results.json"]
    assert digest(RUN / "oracle_decisions.json") == EXPECTED["oracle_decisions.json"]
    assert digest(RUN / "component_predictions.json") == EXPECTED["component_predictions.json"]
    manifest, results = load(SOURCE / "manifest.json"), load(RUN / "results.json")
    oracle, components = load(RUN / "oracle_decisions.json"), load(RUN / "component_predictions.json")
    assert results["status"] == "COMPLETE"
    assert results["test_used_for_training_or_selection"] is False
    assert results["source_manifest_sha256"] == MANIFEST_SHA
    assert manifest["asset_group_overlap"] is False
    assert manifest["polymer_test_targets"] == 0
    tests = [r for r in manifest["records"] if r["role"] == "prospective_test"]
    assert len(tests) == 5 and sum(len(r["regions"]) for r in tests) == 18
    oracle_by_arm = {arm: {r["record_id"]: [] for r in tests} for arm in ("baseline", "adapted")}
    for arm in oracle_by_arm:
        for row in oracle[arm]["regions"]:
            oracle_by_arm[arm][row["record_id"]].append(row)
    component_by_id = {r["record_id"]: r for r in components}
    asset_by_arm = {arm: {r["record_id"]: r for r in oracle[arm]["asset_diagnostics"]["decisions"]}
                    for arm in ("baseline", "adapted")}
    REPORT.mkdir(parents=True, exist_ok=True)
    images = REPORT / "images"
    images.mkdir(exist_ok=True)
    raw_dir = REPORT / "raw"
    raw_dir.mkdir(exist_ok=True)
    for name, expected_sha in EXPECTED.items():
        target = raw_dir / (name + ".txt")
        shutil.copyfile(RUN / name, target)
        assert digest(target) == expected_sha
        title = name.removesuffix(".json").replace("_", " ").title()
        viewer = ("<!doctype html><html lang=\"en-GB\"><head><meta charset=\"utf-8\">"
                  f"<title>{title} · preserved Roihu output</title>"
                  "<style>body{margin:0;background:#101c2a;color:#dbe9f5;font:13px/1.45 ui-monospace,monospace}"
                  "header{position:sticky;top:0;padding:14px 20px;background:#172c43}a{color:#75dbe8}"
                  "pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:20px;margin:0}</style></head><body>"
                  f"<header><strong>{title}</strong> · exact source SHA-256 <code>{expected_sha}</code> · "
                  "<a href=\"../index.html\">back to audit</a></header>"
                  f"<pre>{escape_html(target.read_text())}</pre></body></html>")
        (raw_dir / name.replace(".json", ".html")).write_text(viewer)
    gallery = []
    for record in tests:
        source = ROOT / record["image_file"]
        assert digest(source) == record["image_sha256"]
        with Image.open(source) as raw:
            image = raw.convert("RGB")
        component = component_by_id[record["record_id"]]
        panels = [
            source_panel(image, record),
            oracle_panel(image, record, oracle_by_arm["baseline"][record["record_id"]], "FROZEN V2 HEAD"),
            oracle_panel(image, record, oracle_by_arm["adapted"][record["record_id"]], "ADAPTED FINAL HEAD"),
            detector_panel(image, record, component),
        ]
        target = images / f"{record['record_id']}.jpg"
        montage(panels).save(target, quality=88, optimize=True)
        gallery.append({
            "record_id": record["record_id"], "title": record["title"],
            "image": "images/" + target.name, "image_sha256": digest(target),
            "source_image_sha256": record["image_sha256"], "expected_material": record["material"],
            "regions": len(record["regions"]), "matched": len(component["matches"]),
            "baseline": asset_by_arm["baseline"][record["record_id"]]["material"],
            "adapted": asset_by_arm["adapted"][record["record_id"]]["material"],
            "photo_page_url": record["photo_page_url"], "author": record["author"],
            "licence": record["licence"], "licence_url": record["licence_url"],
            "evidence_excerpt": record["evidence_excerpt"],
        })
    data = {
        "status": "VERIFIED_PRESENTATION", "language": "English", "gallery": gallery,
        "job_id": results["runtime"]["job_id"], "results_sha256": digest(RUN / "results.json"),
        "source_manifest_sha256": MANIFEST_SHA, "test_used_for_training_or_selection": False,
        "deployment_claim": False, "uk_population_accuracy_claim": False,
        "reference_regions_are_expert_ground_truth": False,
        "oracle_diagnostics": results["oracle_diagnostics"],
        "localisation_diagnostics": results["localisation_diagnostics"],
        "end_to_end_diagnostics": results["end_to_end_diagnostics"],
    }
    write_json(REPORT / "data.json", data)
    (REPORT / "index.html").write_text(html(data))
    verification = {
        "status": "VERIFIED", "language": "English", "gallery_images": len(gallery),
        "source_predictions_rewritten": False, "oracle_regions_presented_as_expert_truth": False,
        "scores_presented_as_probabilities": False, "failed_localisations_visible": True,
        "data_sha256": digest(REPORT / "data.json"), "html_sha256": digest(REPORT / "index.html"),
        "gallery_sha256": {r["record_id"]: r["image_sha256"] for r in gallery},
        "raw_copy_sha256": {name: digest(raw_dir / (name + ".txt")) for name in EXPECTED},
    }
    write_json(REPORT / "verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    build()
