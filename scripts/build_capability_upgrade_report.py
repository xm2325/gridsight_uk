"""Build an English evidence gallery for the verified steelwork and pole-top audits."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from prepare_keen_components import ROOT, digest, write_json

POLE_RESULT_SHA = "244481031133e857bde18a35d66e23dc0b7c61886a200c914acf85a83b91d10d"
STEEL_RESULT_SHA = "f327f40f01d8f432c902878e9548e942bea6b9486829497f48bc609f7640ef39"
STEEL_MANIFEST_SHA = "f1b21f61b78038a8a86372483a5c4c9f618cec16b116b9f1f37b5bd7744d4c69"
MATERIAL_RESULT_SHA = "f959a8a7ea8e1b6f476567b8a01833e88249ab50e8358a3becc605d56c10b6f5"
LOCALISATION_RESULT_SHA = "d496f329db51ffc8b0849c1e3b9c9304d4b80906e24d4b9d9432090b12c9277f"
LOCALISATION_ADAPTATION_RESULT_SHA = "4e25c9f42b247b0c93e3308a83a8a37584bbf5bae9b986baf1811909049cf948"


def load(path):
    return json.loads(path.read_text())


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def banner(image, title, detail, colour):
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, 72), fill=(10, 24, 40, 230))
    draw.text((18, 10), title, font=font(22, True), fill=colour)
    draw.text((18, 40), detail, font=font(15), fill="white")


def outline(mask):
    m = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    return np.asarray(m.filter(ImageFilter.MaxFilter(7))) > np.asarray(m.filter(ImageFilter.MinFilter(7)))


def overlay_mask(image, mask, colour, alpha=75, edge=230):
    rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    rgba[mask, :3] = colour
    rgba[mask, 3] = alpha
    e = outline(mask)
    rgba[e, :3] = colour
    rgba[e, 3] = edge
    image.alpha_composite(Image.fromarray(rgba, "RGBA"))


def decode(raw):
    n, h, w = map(int, raw["mask_shape"])
    return np.unpackbits(raw["mask_bits"], axis=1, count=h * w).reshape(n, h, w).astype(bool)


def source_masks(row, size):
    masks = []
    for name, segmentation in zip(row["source_annotations"]["category_name"], row["source_annotations"]["segmentation"]):
        if name != "tower_lattice":
            continue
        canvas = Image.new("1", size)
        points = [(max(0, min(row["width"], segmentation[i])) * size[0] / row["width"],
                   max(0, min(row["height"], segmentation[i + 1])) * size[1] / row["height"])
                  for i in range(0, len(segmentation), 2)]
        ImageDraw.Draw(canvas).polygon(points, fill=1)
        masks.append(np.asarray(canvas, dtype=bool))
    return masks


def steel_panel(source, record, row, run, arm):
    raw_path = run / record[arm]["raw_file"]
    assert digest(raw_path) == record[arm]["raw_sha256"]
    with np.load(raw_path, allow_pickle=False) as raw:
        masks = decode(raw)
        h, w = map(int, raw["mask_shape"][1:])
    with Image.open(source) as im:
        canvas = im.convert("RGB").resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")
    for ref in source_masks(row, (w, h)):
        overlay_mask(canvas, ref, (42, 208, 220), alpha=24, edge=240)
    metrics = record[arm]["metrics"]
    matched = {m["prediction_index"] for m in metrics["matches"]}
    accepted_predictions = [p for p in record[arm]["predictions"] if p["score"] >= .25]
    for prediction in accepted_predictions:
        colour = (35, 206, 137) if prediction["prediction_index"] in matched else (239, 88, 94)
        overlay_mask(canvas, masks[prediction["prediction_index"]], colour)
    title = "SUPERVISED" if arm == "supervised" else "OPEN VOCABULARY"
    state = f'{metrics["tp"]} matched · {metrics["fp"]} false positive · {metrics["fn"]} missed'
    banner(canvas, title, state + " · fixed score ≥ 0.25 / mask IoU ≥ 0.50", "#6ff0b7" if metrics["tp"] else "#ff9b9e")
    return canvas.convert("RGB")


def build_steel(out):
    run = ROOT / "runs/ttpla_steelwork/v1_20260829"
    data = ROOT / "data/external/ttpla_steelwork_demo_v1"
    assert digest(run / "results.json") == STEEL_RESULT_SHA
    assert digest(data / "manifest.json") == STEEL_MANIFEST_SHA
    result, manifest = load(run / "results.json"), load(data / "manifest.json")
    assert result["status"] == "COMPLETE_SMALL_DEVELOPMENT_DEMO" and len(result["records"]) == 12
    rows = {(r["split"], r["dataset_row_index"]): r for r in manifest["records"]}
    gallery = []
    steel_dir = out / "steelwork"
    steel_dir.mkdir(parents=True, exist_ok=True)
    for record in result["records"]:
        idx = int(record["image_id"].split("_")[-1])
        row = rows[("test", idx)]
        source = data / row["image_file"]
        assert digest(source) == row["image_sha256"] == record["source_image_sha256"]
        left = steel_panel(source, record, row, run, "open_vocabulary")
        right = steel_panel(source, record, row, run, "supervised")
        target_h = 600
        panes = []
        for pane in [left, right]:
            scale = target_h / pane.height
            panes.append(pane.resize((round(pane.width * scale), target_h), Image.Resampling.LANCZOS))
        montage = Image.new("RGB", (panes[0].width + panes[1].width, target_h), "#0b1625")
        montage.paste(panes[0], (0, 0)); montage.paste(panes[1], (panes[0].width, 0))
        name = record["image_id"] + ".jpg"
        montage.save(steel_dir / name, quality=88, optimize=True)
        gallery.append({
            "image_id": record["image_id"], "file_name": record["file_name"],
            "image": "steelwork/" + name, "source_group": record["source_group"],
            "selection_kind": record["selection_kind"], "reference_count": record["reference_count"],
            "open_vocabulary": record["open_vocabulary"]["metrics"],
            "supervised": record["supervised"]["metrics"],
            "image_sha256": row["image_sha256"], "dataset_row_index": row["dataset_row_index"],
        })
    return result["summary"], gallery


def marker(draw, point, colour, label, y_offset=0):
    x, y = map(float, point)
    r = 11
    draw.ellipse((x-r, y-r, x+r, y+r), outline=colour, width=4)
    draw.line((x-17, y, x+17, y), fill=colour, width=3)
    draw.line((x, y-17, x, y+17), fill=colour, width=3)
    draw.rounded_rectangle((x+14, y-20+y_offset, x+14+max(110, len(label)*8), y+5+y_offset), 5, fill=(8, 20, 34, 215))
    draw.text((x+20, y-17+y_offset), label, font=font(14, True), fill=colour)


def build_pole_top(out):
    run = ROOT / "runs/pole_top_keypoint/v1_20260829"
    source_root = ROOT / "data/external/epri_component_masks_v1"
    assert digest(run / "results.json") == POLE_RESULT_SHA
    result = load(run / "results.json")
    manifest = load(source_root / "manifest.json")
    rows = {r["image_id"]: r for r in manifest["images"]}
    gallery = []
    target_dir = out / "pole_top"
    target_dir.mkdir(parents=True, exist_ok=True)
    for record in result["records"]:
        row = rows[record["image_id"]]
        source = source_root / row["image_file"]
        if not source.exists():
            source = ROOT / "data/external/epri_components_v1" / row["image_file"]
        assert digest(source) == row["sha256"] == record["source_image_sha256"]
        with Image.open(source) as im:
            canvas = im.convert("RGB").resize(tuple(record["working_size"]), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(canvas, "RGBA")
        marker(draw, record["target_point"], "#43d8e4", "publisher-derived target", -28)
        for arm, colour, label, offset in [("mask_model", "#55e6a2", "mask output", 0),
                                            ("box_model", "#f1ae48", "box output", 24)]:
            decision = record[arm]
            if decision["status"] == "accepted":
                marker(draw, decision["point"], colour, label, offset)
        mask_state = "accepted" if record["mask_model"]["status"] == "accepted" else "abstained"
        box_state = "accepted" if record["box_model"]["status"] == "accepted" else "abstained"
        banner(canvas, "POLE-SHAFT ENDPOINT AUDIT", f"mask {mask_state} · box {box_state} · target is geometry-derived, not a physical-tip annotation", "#9ae9ef")
        name = record["image_id"] + ".jpg"
        canvas.save(target_dir / name, quality=86, optimize=True)
        gallery.append({"image_id": record["image_id"], "image": "pole_top/" + name,
                        "target_point": record["target_point"], "mask_model": record["mask_model"],
                        "box_model": record["box_model"], "source_sha256": row["sha256"]})
    return result["summary"], gallery, len(result["exclusions"])


def load_material_result():
    path = ROOT / "runs/material_head/v3_uk_prospective_20260830/results.json"
    assert digest(path) == MATERIAL_RESULT_SHA
    result = load(path)
    assert result["status"] == "COMPLETE"
    assert result["test_used_for_training_or_selection"] is False
    diagnostics = result["oracle_diagnostics"]["adapted"]["regions"]
    required = {
        "material_targets", "accepted_material_targets",
        "correct_accepted_material_targets", "coverage", "accepted_accuracy",
    }
    assert required <= diagnostics.keys()
    total = diagnostics["material_targets"]
    accepted = diagnostics["accepted_material_targets"]
    correct = diagnostics["correct_accepted_material_targets"]
    assert total > 0 and 0 <= correct <= accepted <= total
    assert abs(diagnostics["coverage"] - accepted / total) < 1e-12
    assert abs(diagnostics["accepted_accuracy"] - correct / accepted) < 1e-12
    localisation = result["localisation_diagnostics"]
    assert localisation["reference_regions"] == total
    return diagnostics, localisation


def load_localisation_result():
    path = ROOT / "runs/uk_insulator_localisation/v1_20260830/results.json"
    assert digest(path) == LOCALISATION_RESULT_SHA
    result = load(path)
    assert result["status"] == "COMPLETE"
    assert result["integrity"]["training_or_parameter_updates"] == 0
    assert result["integrity"]["thresholds_selected_from_acceptance_results"] is False
    score = str(result["protocol"]["primary_operating_score"])
    overlap = str(result["protocol"]["primary_evaluation_iou"])
    return {name: result["metrics"][name][score][overlap] for name in result["protocol"]["arms"]}


def load_localisation_adaptation_result():
    path = ROOT / "runs/uk_insulator_adaptation/v1_20260830/results.json"
    assert digest(path) == LOCALISATION_ADAPTATION_RESULT_SHA
    result = load(path)
    assert result["status"] == "COMPLETE"
    assert result["integrity"]["acceptance_used_for_training_or_checkpoint_selection"] is False
    assert result["integrity"]["thresholds_selected_from_acceptance_results"] is False
    score = str(result["protocol"]["evaluation"]["primary_operating_score"])
    overlap = str(result["protocol"]["evaluation"]["primary_evaluation_iou"])
    return {name: values[score][overlap] for name, values in result["metrics"].items()}


def html(data):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GridSight · Verified capability upgrade</title>
<style>:root{--ink:#172c43;--muted:#62758a;--line:#d9e3ec;--bg:#eef3f7;--navy:#102238}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{background:var(--navy);color:white;padding:24px}header p{margin:6px 0;color:#c7d6e6}.wrap{max-width:1440px;margin:auto;padding:20px}.panel{background:white;border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:18px}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.card{border:1px solid var(--line);border-radius:10px;padding:14px}.card strong{display:block;font-size:25px}.muted,small{color:var(--muted)}.warning{background:#fff3db;border-color:#e7ca8b}.viewer img{width:100%;max-height:680px;object-fit:contain;background:#0b1625;border-radius:8px}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:12px 0}button,select{font:inherit;padding:8px 11px;border:1px solid #c5d3df;border-radius:7px;background:white;color:var(--ink)}select{min-width:320px;max-width:100%}.legend{display:flex;gap:16px;flex-wrap:wrap}.dot{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid var(--line);padding:8px}a{color:#1e64bd}code{background:#edf2f6;padding:2px 5px;border-radius:4px}@media(max-width:800px){.cards{grid-template-columns:1fr}.wrap{padding:12px}select{min-width:0;width:100%}}</style></head><body>
<header><h1>Verified capability upgrade</h1><p>Evidence-backed limits for UK material transfer, lattice steel-structure segmentation and pole-shaft endpoint geometry</p></header><main class="wrap">
<section class="panel warning"><strong>This is not yet the supplied Keen AI result.</strong><p>A fixed target-domain insulator specialist improved a new prospective UK cohort from __ADAPT_BASE_TP__/14 baseline true positives to __ADAPT_TP__/14, with __ADAPT_FP__ false positives. It still missed __ADAPT_FN__ objects and one complete positive asset. TTPLA steel structure is a transmission-tower assembly task, and the pole-top benchmark is a publisher-mask-derived shaft endpoint rather than a physical component detector.</p><a href="../index.html">← UK review workbench</a> · <a href="../material_prospective/index.html">UK material audit</a> · <a href="../localisation_adaptation/index.html">UK adaptation audit</a> · <a href="../localisation_prospective/index.html">Earlier localisation audit</a> · <a href="http://127.0.0.1:8771/report/index.html">EPRI component explorer ↗</a></section>
<section class="cards"><div class="card"><small>UK localisation adaptation · prospective</small><strong>__ADAPT_TP__ / 14</strong><p>The fixed specialist produced __ADAPT_FP__ false positives and missed __ADAPT_FN__ references; the EPRI baseline recovered __ADAPT_BASE_TP__/14. <a href="../localisation_adaptation/index.html">Inspect all five assets</a>.</p></div><div class="card"><small>UK material · prospective oracle regions</small><strong>__MAT_CORRECT__ / __MAT_ACCEPTED__</strong><p>__MAT_ACC__ accepted accuracy at __MAT_COVERAGE__ coverage across __MAT_TOTAL__ source-assisted regions. Full-image localisation matched __LOC_MATCHED__/__MAT_TOTAL__. <a href="../material_prospective/index.html">Inspect all five assets</a>.</p></div><div class="card"><small>Lattice steel structure · fixed test</small><strong>__STEEL_TP__ / __STEEL_TOTAL__</strong><p>Supervised matched instances; open vocabulary matched __OPEN_TP__/__STEEL_TOTAL__. Both had zero detections on __STEEL_NEG__ hard-negative images at score 0.25.</p></div><div class="card"><small>Pole-shaft endpoint · development</small><strong>__POLE_ACCEPTED__ / __POLE_ELIGIBLE__</strong><p>Mask geometry coverage versus __BOX_ACCEPTED__/__POLE_ELIGIBLE__ for box geometry. Accepted mask outputs had median normalised error __POLE_MEDIAN__.</p></div></section>
<section class="panel"><h2>Steel-structure segmentation: all 12 fixed test images</h2><p class="muted">Left: open vocabulary. Right: supervised. Cyan shows the publisher <code>tower_lattice</code> extent; green is a matched mask and red is an unmatched output. Missing output remains a visible miss. These model scores are operating scores, not material probabilities.</p><div class="legend"><span><i class="dot" style="background:#2ad0dc"></i>publisher structure</span><span><i class="dot" style="background:#23ce89"></i>matched model mask</span><span><i class="dot" style="background:#ef585e"></i>unmatched model mask</span></div><div class="toolbar"><button id="steelPrev">← Previous</button><select id="steelSelect"></select><button id="steelNext">Next →</button><span id="steelCounter"></span></div><div class="viewer"><img id="steelImage" alt="TTPLA steel-structure model comparison"></div><p id="steelCaption"></p><p><a href="../../../../ttpla_steelwork/v1_20260829/results.json">Raw result record</a> · <a href="https://github.com/R3ab/ttpla_dataset" target="_blank" rel="noopener">Official TTPLA repository</a> · <a href="https://huggingface.co/datasets/grantmwilkinson/epri-transmission-ttpla" target="_blank" rel="noopener">Mirror provenance</a></p></section>
<section class="panel"><h2>Pole-shaft endpoint geometry: all 18 eligible development targets</h2><p class="muted">The cyan target is deterministically derived from publisher pole and crossarm polygons. It is not an independently annotated physical pole tip. Green is the mask-model endpoint; orange is the box-model endpoint. Absence means the method abstained.</p><div class="toolbar"><button id="polePrev">← Previous</button><select id="poleSelect"></select><button id="poleNext">Next →</button><span id="poleCounter"></span></div><div class="viewer"><img id="poleImage" alt="Pole endpoint geometry comparison"></div><p id="poleCaption"></p><p><a href="../../../../pole_top_keypoint/v1_20260829/results.json">Raw geometry audit</a></p></section>
<section class="panel"><h2>What moves the demo toward Keen AI</h2><table><thead><tr><th>Layer</th><th>Now</th><th>Required next evidence</th></tr></thead><tbody><tr><td>Component localisation</td><td>The prospective UK specialist recovers __ADAPT_TP__/14 with __ADAPT_FP__ false positives; one positive asset is completely missed</td><td>Broaden asset and morphology coverage, add multi-scale small-object training, and freeze a larger asset-disjoint UK acceptance set</td></tr><tr><td>Insulator material</td><td>Adapted oracle crops reach __MAT_ACC__ accepted accuracy at __MAT_COVERAGE__ coverage on 18 source-assisted regions</td><td>Independent UK polymer evidence, a larger untouched test and target-domain calibration; preserve unknown</td></tr><tr><td>Steelwork</td><td>TTPLA lattice assembly mask only</td><td>Distribution-pole structural-member or connected-assembly labels with material evidence</td></tr><tr><td>Pole-top</td><td>Scored shaft-end geometry benchmark; current UK window remains unscored</td><td>Choose and label either physical shaft-tip keypoints or upper-assembly extents on UK assets</td></tr><tr><td>Presentation</td><td>Evidence-separated review overlays</td><td>Only show Keen-style percentages after target-domain calibration; keep unknown and rejected states</td></tr></tbody></table></section>
<footer class="muted">All source hashes, raw predictions, misses, exclusions and frozen boundaries are retained. No UK steelwork or pole-top accuracy is claimed. · <a href="verification.json">Build verification</a> · <a href="ui_qa.json">Browser QA</a></footer></main>
<script id="payload" type="application/json">""" + payload + """</script><script>'use strict';const D=JSON.parse(document.getElementById('payload').textContent);function setup(prefix,rows,caption){let i=0,s=document.getElementById(prefix+'Select'),img=document.getElementById(prefix+'Image'),count=document.getElementById(prefix+'Counter'),cap=document.getElementById(prefix+'Caption');rows.forEach((r,j)=>{let o=document.createElement('option');o.value=j;o.textContent=r.image_id+(r.file_name?' · '+r.file_name:'');s.append(o)});function show(n){i=(n+rows.length)%rows.length;let r=rows[i];s.value=i;img.src=r.image;count.textContent=(i+1)+' / '+rows.length;cap.textContent=caption(r)}document.getElementById(prefix+'Prev').onclick=()=>show(i-1);document.getElementById(prefix+'Next').onclick=()=>show(i+1);s.onchange=()=>show(Number(s.value));show(0)}setup('steel',D.steelwork.gallery,r=>r.selection_kind+' · publisher instances '+r.reference_count+' · open TP/FP/FN '+r.open_vocabulary.tp+'/'+r.open_vocabulary.fp+'/'+r.open_vocabulary.fn+' · supervised '+r.supervised.tp+'/'+r.supervised.fp+'/'+r.supervised.fn);setup('pole',D.pole_top.gallery,r=>'mask '+r.mask_model.status+(r.mask_model.normalized_error!==undefined?' · normalised error '+r.mask_model.normalized_error.toFixed(3):'')+' · box '+r.box_model.status+(r.box_model.normalized_error!==undefined?' · normalised error '+r.box_model.normalized_error.toFixed(3):''));</script></body></html>"""
    material = data["material"]
    localisation_v1 = data["localisation_v1"]
    localisation_adaptation = data["localisation_adaptation_v1"]
    steel = data["steelwork"]["summary"]
    pole = data["pole_top"]["summary"]
    steel_total = steel["supervised"]["tp"] + steel["supervised"]["fn"]
    return (page.replace("__MAT_ACC__", f'{material["accepted_accuracy"]:.1%}')
                .replace("__MAT_ACCEPTED__", str(material["accepted_targets"]))
                .replace("__MAT_CORRECT__", str(material["correct_accepted_targets"]))
                .replace("__MAT_TOTAL__", str(material["material_targets"]))
                .replace("__MAT_COVERAGE__", f'{material["coverage"]:.1%}')
                .replace("__LOC_MATCHED__", str(material["localisation_matched"]))
                .replace("__LOC_COVERAGE__", f'{material["localisation_coverage"]:.1%}')
                .replace("__LOC_TILE_TP__", str(localisation_v1["epri_full_plus_tiles"]["tp"]))
                .replace("__LOC_FUSION_TP__", str(localisation_v1["proposal_fusion"]["tp"]))
                .replace("__LOC_FUSION_FP__", str(localisation_v1["proposal_fusion"]["fp"]))
                .replace("__ADAPT_BASE_TP__", str(localisation_adaptation["baseline_epri_full_plus_tiles"]["tp"]))
                .replace("__ADAPT_TP__", str(localisation_adaptation["adapted_specialist_full_plus_tiles"]["tp"]))
                .replace("__ADAPT_FP__", str(localisation_adaptation["adapted_specialist_full_plus_tiles"]["fp"]))
                .replace("__ADAPT_FN__", str(localisation_adaptation["adapted_specialist_full_plus_tiles"]["fn"]))
                .replace("__STEEL_TP__", str(steel["supervised"]["tp"]))
                .replace("__OPEN_TP__", str(steel["open_vocabulary"]["tp"]))
                .replace("__STEEL_TOTAL__", str(steel_total))
                .replace("__STEEL_NEG__", str(steel["supervised"]["negative_images"]))
                .replace("__POLE_ACCEPTED__", str(pole["mask_model"]["accepted"]))
                .replace("__BOX_ACCEPTED__", str(pole["box_model"]["accepted"]))
                .replace("__POLE_ELIGIBLE__", str(pole["mask_model"]["eligible_targets"]))
                .replace("__POLE_MEDIAN__", f'{pole["mask_model"]["median_normalized_error"]:.3f}'))


def build(report_root):
    out = report_root / "upgrade"
    out.mkdir(parents=True, exist_ok=True)
    steel_summary, steel_gallery = build_steel(out)
    pole_summary, pole_gallery, pole_excluded = build_pole_top(out)
    material_diagnostics, localisation = load_material_result()
    localisation_v1 = load_localisation_result()
    localisation_adaptation_v1 = load_localisation_adaptation_result()
    data = {
        "status": "VERIFIED_CAPABILITY_UPGRADE_PRESENTATION",
        "language": "English", "uk_accuracy_claim": False,
        "steelwork": {"summary": steel_summary, "gallery": steel_gallery,
                      "scope": "TTPLA lattice transmission-tower structural assembly only"},
        "pole_top": {"summary": pole_summary, "gallery": pole_gallery, "excluded": pole_excluded,
                     "scope": "publisher-mask-derived visible shaft endpoint; not physical tip"},
        "material": {"accepted_accuracy": material_diagnostics["accepted_accuracy"],
                     "coverage": material_diagnostics["coverage"],
                     "material_targets": material_diagnostics["material_targets"],
                     "accepted_targets": material_diagnostics["accepted_material_targets"],
                     "correct_accepted_targets": material_diagnostics["correct_accepted_material_targets"],
                     "localisation_matched": localisation["matched_regions"],
                     "localisation_coverage": localisation["region_coverage"],
                     "reference_regions_are_expert_ground_truth": False,
                     "deployment": False, "output_policy": "unknown"},
        "localisation_v1": localisation_v1,
        "localisation_adaptation_v1": localisation_adaptation_v1,
        "raw_results": {"steelwork_sha256": STEEL_RESULT_SHA, "pole_top_sha256": POLE_RESULT_SHA,
                        "steel_manifest_sha256": STEEL_MANIFEST_SHA,
                        "material_v2_sha256": MATERIAL_RESULT_SHA,
                        "uk_localisation_v1_sha256": LOCALISATION_RESULT_SHA,
                        "uk_localisation_adaptation_v1_sha256": LOCALISATION_ADAPTATION_RESULT_SHA},
    }
    write_json(out / "data.json", data)
    (out / "index.html").write_text(html(data))
    verification = {"status": "VERIFIED", "steel_images": len(steel_gallery),
                    "pole_top_images": len(pole_gallery), "source_predictions_rewritten": False,
                    "pseudo_labels_used_as_truth": False, "language": "English",
                    "data_sha256": digest(out / "data.json"), "html_sha256": digest(out / "index.html")}
    write_json(out / "verification.json", verification)
    return data, verification


if __name__ == "__main__":
    data, verification = build(ROOT / "runs/uk_capabilities/v3_20260827/report")
    print(json.dumps(verification, indent=2))
