"""Build the English UK localisation technique audit from verified Roihu outputs."""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from roihu_uk_insulator_localisation_v1 import match_counts

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/uk_insulator_localisation/v1_20260830"
SOURCE = ROOT / "data/external/uk_insulator_localisation_v1"
DEFAULT_OUT = ROOT / "runs/uk_capabilities/v3_20260827/report/localisation_prospective"

ARM_LABELS = {
    "epri_full": "EPRI · full frame",
    "epri_full_plus_tiles": "EPRI · full + tiles",
    "mpid_full_plus_tiles": "MPID · generic proposals",
    "proposal_fusion": "Fixed-priority fusion",
}


def load(path): return json.loads(Path(path).read_text())


def render(image, references, predictions, path, title, threshold=.3):
    canvas = image.copy(); draw = ImageDraw.Draw(canvas); font = ImageFont.load_default()
    counts = match_counts(predictions, references, threshold)
    matched_p = {m["prediction_index"] for m in counts["matches"]}
    matched_r = {m["reference_index"] for m in counts["matches"]}
    for index, box in enumerate(references):
        colour = "#f5a623" if index not in matched_r else "#00c2ff"
        draw.rectangle(box, outline=colour, width=3)
    for index, prediction in enumerate(predictions):
        box = prediction["xyxy"]; colour = "#27d17f" if index in matched_p else "#ff4d6d"
        draw.rectangle(box, outline=colour, width=3)
        label = f"{prediction['source_model']} s={prediction['raw_score']:.2f}"
        x, y = int(box[0]), max(22, int(box[1]))
        draw.rectangle((x, y-16, x+max(75, len(label)*6), y), fill="#111827")
        draw.text((x+2, y-14), label, fill=colour, font=font)
    draw.rectangle((0, 0, canvas.width, 22), fill="#07111f")
    draw.text((6, 6), f"{title} | TP {counts['tp']} · FP {counts['fp']} · FN {counts['fn']}", fill="white", font=font)
    canvas.save(path, quality=94)
    return {k: counts[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")}


def render_references(image, references, path):
    canvas=image.copy(); draw=ImageDraw.Draw(canvas); font=ImageFont.load_default()
    for box in references: draw.rectangle(box, outline="#00c2ff", width=3)
    draw.rectangle((0,0,canvas.width,22),fill="#07111f")
    draw.text((6,6),f"Analyst visible-object references · {len(references)}",fill="white",font=font)
    canvas.save(path,quality=94)


def main(output):
    result = load(RUN/"results.json"); manifest = load(SOURCE/"manifest.json")
    if result["status"] != "COMPLETE" or result["integrity"]["training_or_parameter_updates"] != 0:
        raise ValueError("Expected a complete inference-only run")
    output.mkdir(parents=True, exist_ok=True); (output/"images").mkdir(exist_ok=True); (output/"raw").mkdir(exist_ok=True)
    records = {r["record_id"]: r for r in manifest["records"]}
    gallery = []
    primary_score = result["protocol"]["primary_operating_score"]
    primary_iou = result["protocol"]["primary_evaluation_iou"]
    for item in result["records"]:
        record = records[item["record_id"]]; payload = load(RUN/item["prediction_file"])
        source_image = Image.open(ROOT/record["image_file"]).convert("RGB")
        original = output/"images"/f"{record['record_id']}_reference.jpg"
        render_references(source_image,record["boxes"],original)
        panels = [{"label": "Analyst references", "image": str(original.relative_to(output)),
                   "reference": True, "reference_count": len(record["boxes"])}]
        for arm in result["protocol"]["arms"]:
            predictions = [p for p in payload["arms"][arm] if p["raw_score"] >= primary_score]
            target = output/"images"/f"{record['record_id']}_{arm}.jpg"
            metrics = render(source_image, record["boxes"], predictions, target, ARM_LABELS[arm], primary_iou)
            panels.append({"label": ARM_LABELS[arm], "image": str(target.relative_to(output)), "metrics": metrics})
        raw_name = f"{record['record_id']}.json.txt"
        shutil.copyfile(RUN/item["prediction_file"], output/"raw"/raw_name)
        gallery.append({"record_id": record["record_id"], "photo_id": record["photo_id"],
                        "title": html.unescape(record["title"]), "author": record["author"], "role": record["role"],
                        "asset_group": record["asset_group"], "photo_page_url": record["photo_page_url"],
                        "licence": record["licence"], "licence_url": record["licence_url"],
                        "image_sha256": record["image_sha256"], "reference_count": len(record["boxes"]),
                        "reference_status": record["reference_status"], "panels": panels,
                        "raw_prediction": f"raw/{raw_name}"})
    shutil.copyfile(RUN/"results.json", output/"raw/results.json.txt")
    primary = {name: result["metrics"][name][str(primary_score)][str(primary_iou)]
               for name in result["protocol"]["arms"]}
    data = {"status": result["status"], "job_id": result["runtime"]["slurm_job_id"],
            "git_commit": result["git_commit"], "runtime": result["runtime"], "primary_score": primary_score,
            "primary_iou": primary_iou, "primary": primary, "gallery": gallery,
            "claim_boundary": result["claim_boundary"], "integrity": result["integrity"],
            "conclusion": "Tiling gave only one additional true positive. MPID fusion doubled true positives but produced twenty-two false positives, including four on the hard negative. Target-domain supervised localisation is required."}
    (output/"data.json").write_text(json.dumps(data, indent=2)+"\n")
    cards = "".join(
        f'<div class="metric"><small>{html.escape(ARM_LABELS[name])}</small><b>{m["tp"]}/40</b><span>recall {m["recall"]:.1%} · FP {m["fp"]}</span></div>'
        for name, m in primary.items())
    options = "".join(f'<option value="{i}">{html.escape(row["title"])}</option>' for i, row in enumerate(gallery))
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UK insulator localisation audit</title><style>
:root{{--bg:#07111f;--panel:#0f1c2e;--line:#27364d;--text:#edf4ff;--muted:#a7b6c9;--cyan:#00c2ff;--green:#27d17f;--red:#ff4d6d;--amber:#f5a623}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1600px;margin:auto;padding:28px}}h1{{font-size:32px;margin:.2rem 0}}h2{{margin:0 0 8px}}p{{color:var(--muted)}}a{{color:var(--cyan)}}.warning{{border-left:5px solid var(--amber);background:#211b10;padding:14px 18px;margin:20px 0}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}.metric b{{display:block;font-size:28px}}.metric small,.metric span{{display:block;color:var(--muted)}}.toolbar{{display:flex;gap:10px;align-items:center;margin:24px 0;flex-wrap:wrap}}button,select{{background:#14243a;color:var(--text);border:1px solid #40516c;border-radius:8px;padding:9px 13px}}select{{min-width:340px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}figure{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}}figure img{{display:block;width:100%;height:auto}}figcaption{{padding:9px 12px;color:var(--muted)}}.legend span{{margin-right:16px}}.dot{{display:inline-block;width:11px;height:11px;margin-right:5px}}code{{word-break:break-all}}@media(max-width:900px){{.metrics,.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><div>GridSight-UK · prospective technique audit</div><h1>UK small-insulator localisation</h1>
<p>Eight source-preserved UK images · 40 analyst visible-object references · seven asset groups · one hard negative. Primary point: raw score ≥ {primary_score}, IoU ≥ {primary_iou}. Scores are not probabilities.</p>
<div class="warning"><b>This experiment did not close the Keen AI gap.</b> Tiling recovered one extra object. Fusion reached 4/40 but added 22 false positives. The result supports target-domain supervised localiser adaptation, not further threshold tuning on this acceptance set.</div>
<section class="metrics">{cards}</section>
<div class="toolbar"><button id="prev">← Previous</button><select id="pick">{options}</select><button id="next">Next →</button></div>
<section class="panel"><h2 id="title"></h2><div id="meta"></div><p class="legend"><span><i class="dot" style="background:var(--cyan)"></i>matched reference</span><span><i class="dot" style="background:var(--amber)"></i>missed reference</span><span><i class="dot" style="background:var(--green)"></i>matched prediction</span><span><i class="dot" style="background:var(--red)"></i>unmatched prediction</span></p><div class="grid" id="grid"></div></section>
<p><a href="../material_prospective/index.html">← UK material audit</a> · <a href="../upgrade/index.html">Capability gap summary</a> · <a href="raw/results.json.txt">Raw verified Roihu result</a></p>
<script>const rows={json.dumps(gallery)};let i=0;const pick=document.querySelector('#pick');function show(n){{i=(n+rows.length)%rows.length;pick.value=i;const r=rows[i];document.querySelector('#title').textContent=r.title;document.querySelector('#meta').innerHTML=`${{r.role}} · ${{r.reference_count}} references · ${{r.author}} · <a href="${{r.photo_page_url}}">source page</a> · <a href="${{r.licence_url}}">${{r.licence}}</a> · <a href="${{r.raw_prediction}}">raw predictions</a><br><code>SHA-256 ${{r.image_sha256}}</code>`;document.querySelector('#grid').innerHTML=r.panels.map(p=>`<figure><img src="${{p.image}}"><figcaption>${{p.reference?`${{p.label}} · ${{p.reference_count}} boxes`: `${{p.label}} · TP ${{p.metrics.tp}} · FP ${{p.metrics.fp}} · FN ${{p.metrics.fn}}`}}</figcaption></figure>`).join('')}}document.querySelector('#prev').onclick=()=>show(i-1);document.querySelector('#next').onclick=()=>show(i+1);pick.onchange=()=>show(+pick.value);document.addEventListener('keydown',e=>{{if(e.key==='ArrowLeft')show(i-1);if(e.key==='ArrowRight')show(i+1)}});show(0);</script>
</main></body></html>'''
    (output/"index.html").write_text(page)
    print(json.dumps({"output": str(output), "images": len(gallery),
                      "primary": {k: {x:v[x] for x in ("tp","fp","fn")} for k,v in primary.items()}}, indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    main(parser.parse_args().output)
