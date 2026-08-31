#!/usr/bin/env python3
"""Build an English source/evaluation audit for the frozen crossarm-v2 run."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/epri_crossarm_specialist/v2_20260831"
OLD_RUN = ROOT / "runs/uk_multicomponent/v1_20260830"
SOURCE = ROOT / "data/external/uk_insulator_localisation_v3"
OUT = ROOT / "runs/uk_capabilities/v3_20260827/report/crossarm_v2"
RESULT_SHA = "6561ce49bfe8cae93c2897bae049e870914b6e2458b68dd0edf7c0a830fa148f"
OLD_RESULT_SHA = "8955cad7b399c60680e27d68a93953ba1c04d775eb45e9b17656a5b9d2494ee9"
BASELINE_RESULT_SHA = "d5bc2d70c50a23ed483b8c2378b4d8e0e49d264193314c1c9f828d6122d44a07"


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path):
    return json.loads(Path(path).read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def font(size=14):
    for name in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_box(draw, box, label, colour):
    draw.rectangle(box, outline=colour, width=3)
    x, y = int(box[0]), max(22, int(box[1]))
    bounds = draw.textbbox((0, 0), label, font=font(13))
    width = bounds[2] - bounds[0] + 8
    draw.rectangle((x, y - 20, x + width, y), fill="#07111f")
    draw.text((x + 4, y - 18), label, fill=colour, font=font(13))


def render(image, predictions, title, target, colour):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 30), fill="#07111f")
    draw.text((8, 7), title, fill="white", font=font(14))
    if not predictions:
        message = "No crossarm proposal at the frozen threshold"
        bounds = draw.textbbox((0, 0), message, font=font(16))
        draw.rectangle((10, 42, 30 + bounds[2], 74), fill="#07111fcc")
        draw.text((18, 49), message, fill="#facc15", font=font(16))
    for prediction in predictions:
        draw_box(draw, prediction["box"], f"crossarm · raw {prediction['score']:.2f}", colour)
    canvas.save(target, quality=94)


def verify_run():
    result = load(RUN / "results.json")
    if sha(RUN / "results.json") != RESULT_SHA or result["status"] != "COMPLETE_FROZEN_CROSSARM_SPECIALIST":
        raise ValueError("Crossarm-v2 result pin/status changed")
    config_path = ROOT / "configs/epri_crossarm_specialist_v2.json"
    if result["protocol_sha256"] != sha(config_path):
        raise ValueError("Executed protocol differs from the tracked protocol")
    if sha(RUN / "frozen_choices.json") != result["frozen_choices_sha256"]:
        raise ValueError("Frozen choices changed")
    frozen = load(RUN / "frozen_choices.json")
    if (frozen["evaluation_used_for_selection"] or frozen["uk_used_for_selection"] or
            frozen["uk_reference_boxes_accessed"] or frozen["selected_score_threshold"] != 0.1):
        raise ValueError("Selection/evidence boundary changed")
    if sha(RUN / "training/weights/best.pt") != result["selected_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    verified = 0
    for split, count in (("dev", 80), ("eval", 100), ("uk", 9)):
        summary = load(RUN / split / "results.json")
        if len(summary["records"]) != count:
            raise ValueError(f"Unexpected {split} record count")
        for row in summary["records"]:
            path = RUN / row["prediction_file"]
            if sha(path) != row["prediction_sha256"]:
                raise ValueError(f"Prediction changed: {path}")
            payload = load(path)
            if payload["manual_roi_used"] or not payload["full_frame_only"]:
                raise ValueError("Inference mode changed")
            if split == "uk" and payload["reference_boxes_accessed_or_written"]:
                raise ValueError("UK reference boxes crossed the inference boundary")
            verified += 1
    if (result["uk_proposal_count_at_selected_threshold"] != 0 or
            result["uk_images_with_proposals"] != 0 or result["claims"]["uk_accuracy"]):
        raise ValueError("UK abstention/claim boundary changed")
    for relative, expected in result["source_snapshots"].items():
        if sha(RUN / "code" / Path(relative).name) != expected:
            raise ValueError(f"Executed code snapshot changed: {relative}")
    return result, frozen, verified


def build(output=OUT):
    result, frozen, verified_predictions = verify_run()
    if sha(OLD_RUN / "results.json") != OLD_RESULT_SHA:
        raise ValueError("Pinned UK multi-component baseline changed")
    if sha(ROOT / "runs/keen_components/epri_components_v1_20260827/results.json") != BASELINE_RESULT_SHA:
        raise ValueError("Pinned EPRI baseline changed")
    old = load(OLD_RUN / "results.json")
    old_records = {row["record_id"]: row for row in old["records"]}
    source_manifest = load(SOURCE / "manifest.json")
    source_rows = {row["record_id"]: row for row in source_manifest["records"] if row["role"] != "excluded"}
    uk_summary = load(RUN / "uk/results.json")
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    gallery = []
    artifacts = {}
    for row in uk_summary["records"]:
        record_id = row["image_id"]
        source = source_rows[record_id]
        image_path = ROOT / source["image_file"]
        if sha(image_path) != source["image_sha256"]:
            raise ValueError(f"UK source image changed: {record_id}")
        old_meta = old_records[record_id]
        old_path = OLD_RUN / old_meta["record_file"]
        if sha(old_path) != old_meta["record_sha256"]:
            raise ValueError(f"Old UK component record changed: {record_id}")
        old_payload = load(old_path)
        baseline = [{"score": p["raw_score"], "box": p["xyxy"]}
                    for p in old_payload["components"]["crossarm"]]
        specialist_payload = load(RUN / row["prediction_file"])
        specialist = [p for p in specialist_payload["predictions"] if p["score"] >= frozen["selected_score_threshold"]]
        image = Image.open(image_path).convert("RGB")
        baseline_target = output / "images" / f"{record_id}_baseline.jpg"
        specialist_target = output / "images" / f"{record_id}_specialist.jpg"
        render(image, baseline, "V1 MULTI-CLASS BASELINE · THRESHOLD 0.05", baseline_target, "#22c55e")
        render(image, specialist, "V2 CROSSARM SPECIALIST · DEV-FROZEN THRESHOLD 0.10", specialist_target, "#00c2ff")
        artifacts[str(baseline_target.relative_to(output))] = sha(baseline_target)
        artifacts[str(specialist_target.relative_to(output))] = sha(specialist_target)
        gallery.append({"record_id": record_id, "title": source["title"],
                        "author": source["author"],
                        "source_page": source["photo_page_url"],
                        "licence": source["licence"],
                        "licence_url": source["licence_url"],
                        "image_sha256": source["image_sha256"],
                        "baseline_count": len(baseline), "specialist_count": len(specialist),
                        "panels": [{"label": "V1 multi-class baseline", "image": str(baseline_target.relative_to(output))},
                                   {"label": "V2 crossarm specialist", "image": str(specialist_target.relative_to(output))}]})
    baseline_eval = load(ROOT / "runs/keen_components/epri_components_v1_20260827/eval/supervised.json")["summary"]
    specialist_eval = load(RUN / "eval/results.json")["summary"]
    baseline_operating = baseline_eval["operating_points"]["0.05"]["per_class"]["crossarm"]
    specialist_operating = result["evaluation_selected_metrics"]
    metrics = {
        "baseline": {"ap50": baseline_eval["ap"]["crossarm"]["ap50"],
                     "ap50_95": baseline_eval["ap"]["crossarm"]["ap50_95"],
                     "operating_threshold": 0.05, **baseline_operating},
        "specialist": {"ap50": specialist_eval["ap"]["crossarm"]["ap50"],
                       "ap50_95": specialist_eval["ap"]["crossarm"]["ap50_95"],
                       "operating_threshold": 0.1, **specialist_operating},
    }
    data = {"schema": "gridsight-crossarm-specialist-v2-audit", "language": "English",
            "status": result["status"], "job_id": result["runtime"]["slurm_job_id"],
            "result_sha256": RESULT_SHA, "checkpoint_sha256": result["selected_checkpoint_sha256"],
            "metrics": metrics, "gallery": gallery,
            "uk_output": {"images": 9, "baseline_proposals": sum(r["baseline_count"] for r in gallery),
                          "specialist_proposals": 0, "accuracy": None},
            "conclusion": "The specialist improves source-domain EPRI crossarm detection but does not transfer at its frozen operating point to this nine-image UK cohort. It is retained as a verified experiment, not promoted into the UK overlay."}
    write(output / "data.json", data)
    payload = json.dumps(gallery, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    options = "".join(f'<option value="{i}">{html.escape(row["title"])}</option>' for i, row in enumerate(gallery))
    b, s = metrics["baseline"], metrics["specialist"]
    page = f'''<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Crossarm specialist v2 audit</title><style>
:root{{--bg:#07111f;--panel:#0f1c2e;--line:#2a3a52;--text:#edf4ff;--muted:#a7b6c9;--cyan:#00c2ff;--amber:#facc15}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin:.2rem 0}}p{{color:var(--muted)}}a{{color:var(--cyan)}}.warning,.panel,.metric{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}.warning{{border-left:5px solid var(--amber);margin:20px 0}}.metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}}.metric b{{display:block;font-size:22px}}.metric small{{color:var(--muted)}}.toolbar{{display:flex;gap:10px;align-items:center;margin:22px 0;flex-wrap:wrap}}button,select{{background:#14243a;color:var(--text);border:1px solid #40516c;border-radius:8px;padding:9px 13px}}select{{min-width:360px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}figure{{margin:0;background:#0a1626;border:1px solid var(--line);border-radius:10px;overflow:hidden}}figure img{{display:block;width:100%;height:auto}}figcaption{{padding:9px 12px;color:var(--muted)}}code{{word-break:break-all}}@media(max-width:1000px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}select{{min-width:0;width:100%}}}}</style></head><body><main>
<div>GridSight-UK · frozen transfer audit</div><h1>Crossarm specialist v2</h1><p>One 40-epoch Roihu GH200 run. EPRI circuits 1/2/3/6 train, circuit 4 development, circuits 5/7 frozen evaluation.</p>
<div class="warning"><b>Source-domain improvement did not become UK output.</b> EPRI AP improved, but the specialist emitted zero crossarm proposals above its development-frozen 0.10 threshold on all nine UK images. No UK accuracy is claimed, and this checkpoint is not promoted into the main overlay.</div>
<section class="metrics"><div class="metric"><small>Baseline AP50</small><b>{b['ap50']:.3f}</b></div><div class="metric"><small>Specialist AP50</small><b>{s['ap50']:.3f}</b></div><div class="metric"><small>Baseline AP50–95</small><b>{b['ap50_95']:.3f}</b></div><div class="metric"><small>Specialist AP50–95</small><b>{s['ap50_95']:.3f}</b></div><div class="metric"><small>Baseline F1 @ 0.05</small><b>{b['f1']:.3f}</b></div><div class="metric"><small>Specialist F1 @ 0.10</small><b>{s['f1']:.3f}</b></div></section>
<section class="panel" style="margin-top:14px"><h2>Frozen EPRI evaluation</h2><p>Baseline: precision {b['precision']:.3f}, recall {b['recall']:.3f}. Specialist: precision {s['precision']:.3f}, recall {s['recall']:.3f}. The specialist improves precision and ranking quality on the independent EPRI circuits, while recall at the displayed operating point remains {s['recall']:.3f}.</p><p><b>Interpretation:</b> the run learns EPRI crossarms more cleanly, but the UK domain gap remains the binding problem. More source-domain confidence or prettier labels cannot be presented as UK deployment evidence.</p></section>
<div class="toolbar"><button id="prev">← Previous</button><select id="pick">{options}</select><button id="next">Next →</button><span id="counter"></span></div>
<section class="panel"><h2 id="title"></h2><div id="meta"></div><div class="grid" id="grid"></div></section>
<p><a href="../multicomponent_v1/index.html">Current UK multi-component overlay</a> · <a href="data.json">Report data</a> · <a href="verification.json">Build verification</a></p>
<script>const rows={payload};let i=0;const pick=document.querySelector('#pick');function link(t,u){{const a=document.createElement('a');a.textContent=t;a.href=u;return a}}function show(n){{i=(n+rows.length)%rows.length;pick.value=i;const r=rows[i];document.querySelector('#counter').textContent=`${{i+1}} / ${{rows.length}}`;document.querySelector('#title').textContent=r.title;const m=document.querySelector('#meta');m.replaceChildren(document.createTextNode(`${{r.author}} · `),link('source page',r.source_page),document.createTextNode(' · '),link(r.licence,r.licence_url),document.createElement('br'));const c=document.createElement('code');c.textContent=`SHA-256 ${{r.image_sha256}} · baseline ${{r.baseline_count}} · specialist ${{r.specialist_count}}`;m.append(c);const g=document.querySelector('#grid');g.replaceChildren(...r.panels.map(p=>{{const f=document.createElement('figure'),im=document.createElement('img'),cap=document.createElement('figcaption');im.src=p.image;im.alt=p.label;cap.textContent=p.label;f.append(im,cap);return f}}))}}document.querySelector('#prev').onclick=()=>show(i-1);document.querySelector('#next').onclick=()=>show(i+1);pick.onchange=()=>show(+pick.value);document.addEventListener('keydown',e=>{{if(e.key==='ArrowLeft')show(i-1);if(e.key==='ArrowRight')show(i+1)}});show(0);</script></main></body></html>'''
    if any("\u3400" <= character <= "\u9fff" for character in page):
        raise ValueError("Rendered page is not English-only")
    (output / "index.html").write_text(page)
    artifacts["data.json"] = sha(output / "data.json")
    artifacts["index.html"] = sha(output / "index.html")
    verification = {"status": "VERIFIED_FROZEN_CROSSARM_V2_AUDIT",
                    "result_sha256": RESULT_SHA, "old_result_sha256": OLD_RESULT_SHA,
                    "verified_prediction_files": verified_predictions,
                    "uk_reference_boxes_used_for_model_or_selection": False,
                    "uk_accuracy_reported": False, "specialist_promoted_to_uk_overlay": False,
                    "scores_presented_as_probabilities": False,
                    "artifacts": artifacts}
    write(output / "verification.json", verification)
    return data, verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    data, verification = build(args.output)
    print(json.dumps({"status": verification["status"], "images": len(data["gallery"]),
                      "html_sha256": sha(args.output / "index.html")}, indent=2))
