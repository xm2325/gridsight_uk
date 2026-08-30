"""Build the English prospective UK adaptation audit from verified Roihu output."""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from prepare_keen_components import ROOT, digest, write_json
from roihu_uk_insulator_localisation_v1 import match_counts

RUN = ROOT / "runs/uk_insulator_adaptation/v1_20260830"
SOURCE = ROOT / "data/external/uk_insulator_localisation_v2"
DEFAULT_OUT = ROOT / "runs/uk_capabilities/v3_20260827/report/localisation_adaptation"
RESULT_SHA = "4e25c9f42b247b0c93e3308a83a8a37584bbf5bae9b986baf1811909049cf948"
MANIFEST_SHA = "2fde93a4332e4499cb047a4a684808c798b2e13c387375fcb6ef98395697ffdf"


def load(path):
    return json.loads(Path(path).read_text())


def render(image, references, predictions, path, title, threshold):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    counts = match_counts(predictions, references, threshold)
    matched_predictions = {row["prediction_index"] for row in counts["matches"]}
    matched_references = {row["reference_index"] for row in counts["matches"]}
    for index, box in enumerate(references):
        draw.rectangle(box, outline="#00c2ff" if index in matched_references else "#f5a623", width=3)
    for index, prediction in enumerate(predictions):
        box = prediction["xyxy"]
        colour = "#27d17f" if index in matched_predictions else "#ff4d6d"
        draw.rectangle(box, outline=colour, width=3)
        label = f"raw score {prediction['raw_score']:.2f}"
        x, y = int(box[0]), max(22, int(box[1]))
        draw.rectangle((x, y - 16, x + max(86, len(label) * 6), y), fill="#111827")
        draw.text((x + 2, y - 14), label, fill=colour, font=font)
    draw.rectangle((0, 0, canvas.width, 22), fill="#07111f")
    draw.text((6, 6), f"{title} | TP {counts['tp']} · FP {counts['fp']} · FN {counts['fn']}", fill="white", font=font)
    canvas.save(path, quality=94)
    return {key: counts[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")}


def render_references(image, references, path):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for box in references:
        draw.rectangle(box, outline="#00c2ff", width=3)
    draw.rectangle((0, 0, canvas.width, 22), fill="#07111f")
    draw.text((6, 6), f"Analyst visible-object references · {len(references)}", fill="white", font=font)
    canvas.save(path, quality=94)


def build(output=DEFAULT_OUT):
    if digest(RUN / "results.json") != RESULT_SHA or digest(SOURCE / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Pinned adaptation result or acceptance manifest changed")
    result = load(RUN / "results.json")
    manifest = load(SOURCE / "manifest.json")
    if result["status"] != "COMPLETE" or result["integrity"]["acceptance_used_for_training_or_checkpoint_selection"]:
        raise ValueError("Expected a complete leakage-protected run")
    if result["integrity"]["thresholds_selected_from_acceptance_results"]:
        raise ValueError("Acceptance-selected threshold is not permitted")
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    (output / "raw").mkdir(exist_ok=True)
    records = {row["record_id"]: row for row in manifest["records"]
               if row["role"] in {"prospective_test", "hard_negative"}}
    score = result["protocol"]["evaluation"]["primary_operating_score"]
    overlap = result["protocol"]["evaluation"]["primary_evaluation_iou"]
    baseline_rows = {row["record_id"]: row for row in result["acceptance_predictions"]["baseline_epri"]}
    adapted_rows = {row["record_id"]: row for row in result["acceptance_predictions"]["adapted_specialist"]}
    gallery = []
    for record_id, record in records.items():
        image = Image.open(ROOT / record["image_file"]).convert("RGB")
        reference_path = output / "images" / f"{record_id}_references.jpg"
        render_references(image, record["boxes"], reference_path)
        panels = [{"label": "Analyst references", "image": str(reference_path.relative_to(output)),
                   "reference": True, "reference_count": len(record["boxes"])}]
        raw_links = {}
        for model_name, label, rows in (("baseline_epri", "EPRI baseline · full + tiles", baseline_rows),
                                         ("adapted_specialist", "UK-adapted specialist · full + tiles", adapted_rows)):
            source_path = RUN / rows[record_id]["prediction_file"]
            if digest(source_path) != rows[record_id]["sha256"]:
                raise ValueError(f"Raw prediction changed: {model_name} {record_id}")
            payload = load(source_path)
            predictions = [row for row in payload["full_plus_tiles"] if row["raw_score"] >= score]
            target = output / "images" / f"{record_id}_{model_name}.jpg"
            metrics = render(image, record["boxes"], predictions, target, label, overlap)
            panels.append({"label": label, "image": str(target.relative_to(output)), "metrics": metrics})
            raw_target = output / "raw" / f"{record_id}_{model_name}.json.txt"
            shutil.copyfile(source_path, raw_target)
            raw_links[model_name] = str(raw_target.relative_to(output))
        gallery.append({"record_id": record_id, "photo_id": record["photo_id"],
                        "title": html.unescape(record["title"]), "author": record["author"],
                        "role": record["role"], "asset_group": record["asset_group"],
                        "photo_page_url": record["photo_page_url"], "licence": record["licence"],
                        "licence_url": record["licence_url"], "image_sha256": record["image_sha256"],
                        "reference_count": len(record["boxes"]), "reference_status": record["reference_status"],
                        "panels": panels, "raw_predictions": raw_links})
    for name in ("results.json", "frozen_choices_before_acceptance.json", "acceptance_evaluation_receipt.json"):
        shutil.copyfile(RUN / name, output / "raw" / f"{name}.txt")
    primary = {name: values[str(score)][str(overlap)] for name, values in result["metrics"].items()}
    data = {"status": result["status"], "job_id": result["runtime"]["slurm_job_id"],
            "git_commit": result["git_commit"], "runtime": result["runtime"],
            "selected_checkpoint_sha256": result["selected_checkpoint_sha256"],
            "result_sha256": RESULT_SHA, "acceptance_manifest_sha256": MANIFEST_SHA,
            "primary_score": score, "primary_iou": overlap, "primary": primary,
            "gallery": gallery, "claim_boundary": result["claim_boundary"], "integrity": result["integrity"],
            "conclusion": "Target-domain adaptation produced a real prospective gain, but recovered only 5 of 14 analyst references at the primary point. One positive asset remained completely missed; this is not yet Keen-style dense component localisation."}
    write_json(output / "data.json", data)
    cards = "".join(
        f'<div class="metric"><small>{label}</small><b>{primary[name]["tp"]}/14</b><span>recall {primary[name]["recall"]:.1%} · FP {primary[name]["fp"]}</span></div>'
        for name, label in (("baseline_epri_full_plus_tiles", "EPRI baseline · full + tiles"),
                            ("adapted_specialist_full_plus_tiles", "UK-adapted specialist · full + tiles"),
                            ("adapted_specialist_full", "UK-adapted specialist · full frame")))
    options = "".join(f'<option value="{index}">{html.escape(row["title"])}</option>'
                      for index, row in enumerate(gallery))
    payload = json.dumps(gallery, ensure_ascii=False).replace("<", "\\u003c")
    page = f'''<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UK insulator adaptation audit</title><style>
:root{{--bg:#07111f;--panel:#0f1c2e;--line:#27364d;--text:#edf4ff;--muted:#a7b6c9;--cyan:#00c2ff;--green:#27d17f;--red:#ff4d6d;--amber:#f5a623}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{font-size:32px;margin:.2rem 0}}h2{{margin:0 0 8px}}p{{color:var(--muted)}}a{{color:var(--cyan)}}.warning{{border-left:5px solid var(--amber);background:#211b10;padding:14px 18px;margin:20px 0}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.metric,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}.metric b{{display:block;font-size:28px}}.metric small,.metric span{{display:block;color:var(--muted)}}.toolbar{{display:flex;gap:10px;align-items:center;margin:24px 0;flex-wrap:wrap}}button,select{{background:#14243a;color:var(--text);border:1px solid #40516c;border-radius:8px;padding:9px 13px}}select{{min-width:340px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}figure{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}}figure img{{display:block;width:100%;height:auto}}figcaption{{padding:9px 12px;color:var(--muted)}}.legend span{{margin-right:16px}}.dot{{display:inline-block;width:11px;height:11px;margin-right:5px}}code{{word-break:break-all}}@media(max-width:980px){{.metrics,.grid{{grid-template-columns:1fr}}select{{min-width:0;width:100%}}}}
</style></head><body><main><div>GridSight-UK · prospective adaptation audit</div><h1>UK small-insulator adaptation</h1>
<p>Five untouched UK assets · 14 analyst visible-object references · one hard negative. The checkpoint and all settings were frozen before acceptance inference. Primary point: raw score ≥ {score}, IoU ≥ {overlap}.</p>
<div class="warning"><b>Improved, but not yet the supplied Keen AI result.</b> The specialist recovered 5/14 references with 3 false positives, versus 0/14 with 1 false positive for the baseline. It still missed 9 objects and one complete positive asset. Scores are uncalibrated and are not probabilities.</div>
<section class="metrics">{cards}</section>
<div class="toolbar"><button id="prev">← Previous</button><select id="pick">{options}</select><button id="next">Next →</button><span id="counter"></span></div>
<section class="panel"><h2 id="title"></h2><div id="meta"></div><p class="legend"><span><i class="dot" style="background:var(--cyan)"></i>matched reference</span><span><i class="dot" style="background:var(--amber)"></i>missed reference</span><span><i class="dot" style="background:var(--green)"></i>matched prediction</span><span><i class="dot" style="background:var(--red)"></i>unmatched prediction</span></p><div class="grid" id="grid"></div></section>
<p><a href="../localisation_prospective/index.html">← Earlier localisation audit</a> · <a href="../material_prospective/index.html">UK material audit</a> · <a href="../upgrade/index.html">Capability gap summary</a> · <a href="raw/results.json.txt">Raw verified Roihu result</a> · <a href="verification.json">Build verification</a> · <a href="ui_qa.json">Browser QA</a></p>
<script>const rows={payload};let i=0;const pick=document.querySelector('#pick');function show(n){{i=(n+rows.length)%rows.length;pick.value=i;const r=rows[i];document.querySelector('#counter').textContent=`${{i+1}} / ${{rows.length}}`;document.querySelector('#title').textContent=r.title;document.querySelector('#meta').innerHTML=`${{r.role}} · ${{r.reference_count}} references · ${{r.author}} · <a href="${{r.photo_page_url}}">source page</a> · <a href="${{r.licence_url}}">${{r.licence}}</a> · <a href="${{r.raw_predictions.baseline_epri}}">baseline raw</a> · <a href="${{r.raw_predictions.adapted_specialist}}">adapted raw</a><br><code>SHA-256 ${{r.image_sha256}}</code>`;document.querySelector('#grid').innerHTML=r.panels.map(p=>`<figure><img src="${{p.image}}" alt="${{p.label}}"><figcaption>${{p.reference?`${{p.label}} · ${{p.reference_count}} boxes`:`${{p.label}} · TP ${{p.metrics.tp}} · FP ${{p.metrics.fp}} · FN ${{p.metrics.fn}}`}}</figcaption></figure>`).join('')}}document.querySelector('#prev').onclick=()=>show(i-1);document.querySelector('#next').onclick=()=>show(i+1);pick.onchange=()=>show(+pick.value);document.addEventListener('keydown',e=>{{if(e.key==='ArrowLeft')show(i-1);if(e.key==='ArrowRight')show(i+1)}});show(0);</script></main></body></html>'''
    (output / "index.html").write_text(page)
    verification = {"status": "VERIFIED", "language": "English", "gallery_images": len(gallery),
                    "source_predictions_rewritten": False, "pseudo_labels_used_as_truth": False,
                    "result_sha256": RESULT_SHA, "data_sha256": digest(output / "data.json"),
                    "html_sha256": digest(output / "index.html")}
    write_json(output / "verification.json", verification)
    return data, verification


if __name__ == "__main__":
    _, built = build()
    print(json.dumps(built, indent=2))
