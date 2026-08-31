#!/usr/bin/env python3
"""Build the English evidence audit for crossarm Grounding DINO v1."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/crossarm_grounding/v1_20260831"
SOURCE = ROOT / "data/external/uk_distribution_pilot_v1"
OUT = ROOT / "runs/uk_capabilities/v3_20260827/report/crossarm_grounding_v1"
RESULT_SHA = "0c9424dc9571dc37fc54c6036413e4f803587c3690a70cf351b820d606a28421"
FROZEN_SHA = "2794a62b09f8b17ca7d5d396de8f7324b58d1e77177b34874e1217784213de31"


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


def draw_box(draw, box, label, colour, width):
    draw.rectangle(box, outline=colour, width=width)
    x, y = int(box[0]), max(22, int(box[1]))
    bounds = draw.textbbox((0, 0), label, font=font(13))
    draw.rectangle((x, y - 20, x + bounds[2] - bounds[0] + 8, y), fill="#07111f")
    draw.text((x + 4, y - 18), label, fill=colour, font=font(13))


def verify_run():
    result = load(RUN / "results.json")
    if sha(RUN / "results.json") != RESULT_SHA or not result["status"].startswith("COMPLETE"):
        raise ValueError("Pinned Grounding DINO result changed")
    if sha(RUN / "frozen_choices.json") != FROZEN_SHA:
        raise ValueError("Pinned frozen choices changed")
    if result["uk_v3_accessed"] or result["uk_ground_truth_used"]:
        raise ValueError("UK evidence boundary changed")
    if result["performance_metrics"]["uk_development"] is not None:
        raise ValueError("UK development data must not have accuracy metrics")
    if (result["uk_proposal_count"], result["uk_images_with_proposals"]) != (3, 2):
        raise ValueError("Pinned UK proposal counts changed")
    checked = 2
    for item in result["epri_records"]:
        record_path = RUN / item["record_file"]
        if sha(record_path) != item["record_sha256"]:
            raise ValueError(f"EPRI record changed: {record_path}")
        record = load(record_path)
        if record["reference_source"] != "publisher EPRI polygon boxes":
            raise ValueError("EPRI reference source changed")
        checked += 1
        for prompt in record["prompt_records"]:
            raw_path = RUN / prompt["raw_file"]
            if sha(raw_path) != prompt["raw_sha256"]:
                raise ValueError(f"EPRI raw tensor changed: {raw_path}")
            checked += 1
    for item in result["uk_records"]:
        record_path = RUN / item["record_file"]
        if sha(record_path) != item["record_sha256"]:
            raise ValueError(f"UK record changed: {record_path}")
        record = load(record_path)
        if (record["ground_truth_status"] != "NONE" or record["scores_are_probabilities"]
                or record["reference_truth"]):
            raise ValueError("UK truth/probability boundary changed")
        raw_path = RUN / record["raw_file"]
        if sha(raw_path) != record["raw_sha256"]:
            raise ValueError(f"UK raw tensor changed: {raw_path}")
        checked += 2
    for relative, expected in result["source_snapshots"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Executed source snapshot changed: {relative}")
        checked += 1
    return result, checked


def render(source: Path, record: dict, target: Path) -> None:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(3, image.width // 400)
    for pole in record["poles"]:
        draw_box(draw, pole["box"], f"pole input · raw {pole['score']:.3f}", "#fbbf24", width)
    for candidate in record["display_predictions"]:
        draw_box(draw, candidate["box"],
                 f"crossarm candidate · raw {candidate['score']:.3f}", "#00e5d4", width + 1)
    if not record["display_predictions"]:
        message = "No proposal at the EPRI-selected operating point"
        bounds = draw.textbbox((0, 0), message, font=font(16))
        draw.rectangle((10, 10, bounds[2] + 28, 44), fill="#07111f")
        draw.text((18, 18), message, fill="#facc15", font=font(16))
    image.save(target, quality=92)


def build(output=OUT):
    result, verified_files = verify_run()
    manifest_path = SOURCE / "manifest.json"
    manifest = load(manifest_path)
    rows = {row["image_id"]: row for row in manifest["images"]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    gallery, artifacts = [], {}
    for item in result["uk_records"]:
        record = load(RUN / item["record_file"])
        source = rows[record["image_id"]]
        source_path = SOURCE / source["image_file"]
        if sha(source_path) != source["sha256"] or record["image_sha256"] != source["sha256"]:
            raise ValueError(f"Source image changed: {record['image_id']}")
        target = output / "images" / f"{record['image_id']}.jpg"
        render(source_path, record, target)
        artifacts[str(target.relative_to(output))] = sha(target)
        gallery.append({
            "image_id": record["image_id"], "title": source["title"],
            "author": source["author"], "source_page": source["source_page"],
            "licence": source["license"], "licence_url": source["license_url"],
            "source_sha256": source["sha256"], "image": str(target.relative_to(output)),
            "pole_count": len(record["poles"]),
            "proposal_count": len(record["display_predictions"]),
            "raw_scores": [p["score"] for p in record["display_predictions"]],
            "ground_truth_status": "NONE", "accuracy": None,
        })
    data = {
        "schema": "gridsight-crossarm-grounding-v1-audit", "language": "English",
        "status": result["status"], "job_id": result["runtime"]["slurm_job_id"],
        "result_sha256": RESULT_SHA, "frozen_choices_sha256": FROZEN_SHA,
        "selected": result["selected"], "uk": {
            "images": len(gallery), "images_with_proposals": result["uk_images_with_proposals"],
            "proposal_count": result["uk_proposal_count"], "ground_truth_status": "NONE",
            "accuracy": None, "uk_v3_accessed": False,
        },
        "qualitative_audit": "One compact proposal overlaps a plausible pole-top support region. A second proposal follows the pole shaft, and a third spans a fallen pole rather than a crossarm. These observations are a visual failure audit, not reference annotations or accuracy.",
        "decision": "Do not promote this arm to UK v3 acceptance or the main overlay. It improves proposal coverage over the source-trained specialist, but two of three displayed boxes have the wrong morphology and UK recall is unknown.",
        "gallery": gallery,
    }
    write(output / "data.json", data)
    payload = json.dumps(gallery, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    options = "".join(f'<option value="{i}">{html.escape(row["title"])}</option>' for i, row in enumerate(gallery))
    selected = result["selected"]
    page = f'''<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Crossarm Grounding DINO v1 audit</title><style>
:root{{--bg:#07111f;--panel:#0f1c2e;--line:#2a3a52;--text:#edf4ff;--muted:#a7b6c9;--cyan:#00e5d4;--amber:#facc15;--red:#fb7185}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1450px;margin:auto;padding:28px}}h1{{margin:.2rem 0}}p{{color:var(--muted)}}a{{color:#67e8f9}}.warning,.panel,.metric{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}.warning{{border-left:5px solid var(--red);margin:20px 0}}.metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}}.metric b{{display:block;font-size:22px}}.metric small{{color:var(--muted)}}.toolbar{{display:flex;gap:10px;align-items:center;margin:22px 0;flex-wrap:wrap}}button,select{{background:#14243a;color:var(--text);border:1px solid #40516c;border-radius:8px;padding:9px 13px}}select{{min-width:420px;max-width:70%}}figure{{margin:0;background:#0a1626;border:1px solid var(--line);border-radius:10px;overflow:hidden}}figure img{{display:block;width:100%;height:auto}}figcaption{{padding:9px 12px;color:var(--muted)}}code{{word-break:break-all}}@media(max-width:900px){{.metrics{{grid-template-columns:repeat(2,1fr)}}select{{min-width:0;max-width:none;width:100%}}}}</style></head><body><main>
<div>GridSight-UK · evidence-gated transfer audit</div><h1>Crossarm Grounding DINO v1</h1><p>Prompt and operating point were selected only on publisher-annotated EPRI development data. The frozen arm was then run once on 27 UK development images without labels. UK v3 was not accessed.</p>
<div class="warning"><b>Candidate coverage is not UK accuracy.</b> The system emitted three proposals on two of 27 UK images. Visual inspection found one compact plausible support-region proposal and two morphology mismatches. This arm is not promoted to the UK acceptance cohort or main overlay.</div>
<section class="metrics"><div class="metric"><small>EPRI precision</small><b>{selected['precision']:.3f}</b></div><div class="metric"><small>EPRI recall</small><b>{selected['recall']:.3f}</b></div><div class="metric"><small>EPRI F1</small><b>{selected['f1']:.3f}</b></div><div class="metric"><small>UK images</small><b>27</b></div><div class="metric"><small>UK proposal images</small><b>2</b></div><div class="metric"><small>UK accuracy</small><b>—</b></div></section>
<section class="panel" style="margin-top:14px"><h2>Frozen arm</h2><p><b>Prompt:</b> {html.escape(selected['prompt'])}<br><b>Variant:</b> pole-associated · <b>raw threshold:</b> {selected['threshold']:.2f}. Scores are uncalibrated model values, not probabilities.</p><p><b>Decision:</b> do not run this arm on UK v3. The remaining problem is target-domain crossarm morphology and reliable pole association, not score presentation.</p></section>
<div class="toolbar"><button id="prev">← Previous</button><select id="pick">{options}</select><button id="next">Next →</button><span id="counter"></span></div>
<section class="panel"><h2 id="title"></h2><div id="meta"></div><figure><img id="image" alt="UK crossarm proposal audit"><figcaption id="caption"></figcaption></figure></section>
<p><a href="../crossarm_v2/index.html">Source-trained specialist audit</a> · <a href="../multicomponent_v1/index.html">Current UK multi-component overlay</a> · <a href="data.json">Report data</a> · <a href="verification.json">Build verification</a></p>
<script>const rows={payload};let i=0;const pick=document.querySelector('#pick');function link(t,u){{const a=document.createElement('a');a.textContent=t;a.href=u;return a}}function show(n){{i=(n+rows.length)%rows.length;pick.value=i;const r=rows[i];document.querySelector('#counter').textContent=`${{i+1}} / ${{rows.length}}`;document.querySelector('#title').textContent=r.title;const m=document.querySelector('#meta');m.replaceChildren(document.createTextNode(`${{r.author}} · `),link('source page',r.source_page),document.createTextNode(' · '),link(r.licence,r.licence_url),document.createElement('br'));const c=document.createElement('code');c.textContent=`SHA-256 ${{r.source_sha256}} · pole inputs ${{r.pole_count}} · crossarm proposals ${{r.proposal_count}} · UK truth NONE`;m.append(c);const im=document.querySelector('#image');im.src=r.image;im.alt=`${{r.title}} crossarm proposal audit`;document.querySelector('#caption').textContent=r.proposal_count?`Cyan: crossarm proposal; amber: pole association input. Raw scores: ${{r.raw_scores.map(x=>x.toFixed(3)).join(', ')}}.`:'Abstained at the EPRI-selected operating point.'}}document.querySelector('#prev').onclick=()=>show(i-1);document.querySelector('#next').onclick=()=>show(i+1);pick.onchange=()=>show(+pick.value);document.addEventListener('keydown',e=>{{if(e.key==='ArrowLeft')show(i-1);if(e.key==='ArrowRight')show(i+1)}});show(0);</script></main></body></html>'''
    if any("\u3400" <= character <= "\u9fff" for character in page):
        raise ValueError("Rendered page is not English-only")
    (output / "index.html").write_text(page)
    artifacts["data.json"] = sha(output / "data.json")
    artifacts["index.html"] = sha(output / "index.html")
    verification = {
        "status": "VERIFIED_CROSSARM_GROUNDING_V1_FAILURE_AUDIT",
        "result_sha256": RESULT_SHA, "frozen_choices_sha256": FROZEN_SHA,
        "verified_files": verified_files, "uk_v3_accessed": False,
        "uk_ground_truth_used": False, "uk_accuracy_reported": False,
        "scores_presented_as_probabilities": False, "promoted_to_main_overlay": False,
        "artifacts": artifacts,
    }
    write(output / "verification.json", verification)
    return data, verification


if __name__ == "__main__":
    data, verification = build()
    print(json.dumps({"status": verification["status"], "images": len(data["gallery"]),
                      "html_sha256": sha(OUT / "index.html")}, indent=2))
