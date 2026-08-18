from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "_site"
DATASET = SITE / "assets" / "dataset"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def metric(label: str, value: str, note: str = "") -> dict:
    return {"label": label, "value": value, "note": note}


def base_story(entry: dict) -> dict:
    rid = entry["record_id"]
    group = entry["filter_group"]
    if group == "legacy-train":
        return {
            "record_id": rid,
            "title": "Original training source",
            "status": "training",
            "phase": "Legacy component benchmark",
            "question": "Can a small source-separated UK tower cohort support an initial component-localisation benchmark?",
            "dataset": "One of the five original training images. It is allowed to influence model fitting and therefore cannot be used as independent evaluation evidence.",
            "experiment": "Used in the original component-training cohort. The gallery may show a SHA-256-verified higher-resolution Commons original, while the model runtime remains the recorded project derivative.",
            "result": "The current main branch preserves cohort-level results rather than a trustworthy per-image headline metric for this training record.",
            "decision": "Keep it visible as training provenance, not as a test example.",
            "metrics": [metric("evidence role", "TRAIN", "model fitting allowed")],
            "boundary": "Training source. No independent-performance claim can be made from this image.",
        }
    if group == "legacy-val":
        return {
            "record_id": rid,
            "title": "Original source-separated validation",
            "status": "validation",
            "phase": "Legacy component benchmark",
            "question": "Does the early pipeline transfer beyond the five training towers?",
            "dataset": "Held outside the original training cohort as the source-separated validation image.",
            "experiment": "Used for development validation/model selection in the early component cycle.",
            "result": "It is validation evidence, not a final holdout. The current main branch does not preserve a per-image metric that should be promoted as a headline result.",
            "decision": "Retain the role label so validation is never presented as unseen final-test evidence.",
            "metrics": [metric("evidence role", "VALIDATION", "development model selection")],
            "boundary": "Validation source. It may inform development and is therefore not independent final evidence.",
        }
    if group == "legacy-test":
        return {
            "record_id": rid,
            "title": "Early source-separated test / showcase",
            "status": "historical test",
            "phase": "Legacy component benchmark",
            "question": "Can the first benchmark be shown on a tower kept outside the original train/validation images?",
            "dataset": "Predeclared source-separated test/showcase image in the early project cycle.",
            "experiment": "Used as an early visual test/showcase after the original training and validation split.",
            "result": "The current main branch does not retain a source-level metric that is safe to reconstruct here. The page intentionally shows provenance instead of inventing a number.",
            "decision": "Keep this as historical evidence and do not recycle it as a fresh independent holdout after repeated inspection.",
            "metrics": [metric("evidence role", "TEST / SHOWCASE", "historical, repeatedly inspected")],
            "boundary": "Historical test/showcase source; not a new independent evaluation in the current cycle.",
        }
    if group == "stress":
        return {
            "record_id": rid,
            "title": "Scale-stress source",
            "status": "stress review",
            "phase": "Morphology and deployment-scale review",
            "question": "What happens when the tower is too distant for reliable component-level annotation?",
            "dataset": "Selected for scale-stress review, not for component ground truth.",
            "experiment": "Direct pixel review assessed whether individual insulator assemblies were large enough to annotate reliably.",
            "result": "Component pixels were judged too small for a defensible component-level benchmark on this source.",
            "decision": "Keep the image as a deployment-scale stress example and exclude it from headline component metrics.",
            "metrics": [metric("evidence role", "STRESS ONLY", "no component metric")],
            "boundary": "Visual scale-stress evidence only; no component-performance estimate is reported.",
        }
    return {
        "record_id": rid,
        "title": "Repository source image",
        "status": entry.get("split", "development"),
        "phase": "GridSight-UK data cycle",
        "question": "Why is this source present in the project?",
        "dataset": entry["description"],
        "experiment": "See the repository role and provenance labels for its permitted use.",
        "result": "No additional source-level metric is asserted here unless it is available from a frozen or reproducible report.",
        "decision": "Preserve the source role and evidence boundary.",
        "metrics": [metric("evidence role", entry.get("split", "development").upper())],
        "boundary": "Do not collapse training, validation, holdout and development evidence into one performance claim.",
    }


def main() -> None:
    index_path = SITE / "index.html"
    manifest_path = DATASET / "manifest.json"
    if not index_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Run the v4.8 Pages and dataset-gallery builders first")

    manifest = load_json(manifest_path)
    entries = manifest["entries"]
    by_id = {e["record_id"]: e for e in entries}
    if len(by_id) != 16:
        raise RuntimeError(f"Expected 16 dataset records, found {len(by_id)}")

    final = load_json(ROOT / "reports/v3_8_final_holdout_summary.json")
    v42 = load_json(ROOT / "reports/v4_2_preadaptation_summary.json")
    v43 = load_json(ROOT / "reports/v4_3_morphology_diversity_summary.json")
    v44 = load_json(ROOT / "reports/v4_4_pixel_review.json")
    material_labels = load_json(ROOT / "data/v4_annotations/material_reference_v44.json")
    material_protocol = load_json(ROOT / "configs/v46_siglip2_material_crop_protocol.json")
    material_results = load_json(ROOT / "reports/v4_6_siglip2_material_crop_results.json")

    stories = {rid: base_story(entry) for rid, entry in by_id.items()}

    # Frozen final holdout: preserve the one-shot result and the later analysis-only diagnosis.
    p343 = final["per_image"]["POS_3437435"]
    p756 = final["per_image"]["POS_7561805"]
    stories["POS_3437435"] = {
        "record_id": "POS_3437435",
        "title": "The geometry heuristic failed on a new morphology",
        "status": "retired final holdout",
        "phase": "v3.8 final holdout → v3.9 failure diagnosis",
        "question": "Would a development-selected tower-relative geometry filter generalise to a genuinely unseen UK tower design?",
        "dataset": "Primary frozen final holdout in England with six assistant-provisional insulator references. It was evaluated once and then retired from tuning.",
        "experiment": "Compare raw open-vocabulary YOLOE text proposals with the frozen geometry champion at IoU ≥ 0.30, without retuning either method on this image.",
        "result": f"Raw YOLOE found {p343['raw_text_iou_0_30']['tp']}/6 references; the geometry filter found {p343['champion_iou_0_30']['tp']}/6. Post-final analysis found median Mahalanobis d² 39.629 versus a maximum 7.008 in the training geometry distribution.",
        "decision": "Retire the narrow geometry prior and start a new morphology-diverse data cycle instead of tuning against this holdout.",
        "metrics": [
            metric("raw YOLOE recall", pct(p343["raw_text_iou_0_30"]["recall"]), "IoU ≥ 0.30"),
            metric("geometry recall", pct(p343["champion_iou_0_30"]["recall"]), "IoU ≥ 0.30"),
            metric("holdout median d²", "39.629", "training max 7.008"),
            metric("references", "6", "all 6 beyond training max d²"),
        ],
        "boundary": final["claim_boundary"],
    }
    stories["POS_7561805"] = {
        "record_id": "POS_7561805",
        "title": "A second holdout showed the failure was morphology-dependent",
        "status": "retired final holdout",
        "phase": "v3.8 final holdout → v3.9 failure diagnosis",
        "question": "Was the geometry-prior failure universal, or strongest on specific unseen tower geometries?",
        "dataset": "Secondary frozen final holdout in Scotland with six assistant-provisional insulator references, selected to add scale/generalisation stress.",
        "experiment": "The same frozen raw YOLOE and geometry methods were evaluated at IoU ≥ 0.30 with no per-image tuning.",
        "result": f"Raw YOLOE found {p756['raw_text_iou_0_30']['tp']}/6 references (recall {pct(p756['raw_text_iou_0_30']['recall'])}); the geometry filter retained {p756['champion_iou_0_30']['tp']}/6. Its median d² was 2.482, much closer to the training geometry distribution than POS_3437435.",
        "decision": "Treat the failure as a distribution-shift problem, not as evidence that geometric structure is always useless.",
        "metrics": [
            metric("raw YOLOE recall", pct(p756["raw_text_iou_0_30"]["recall"]), "IoU ≥ 0.30"),
            metric("geometry recall", pct(p756["champion_iou_0_30"]["recall"]), "IoU ≥ 0.30"),
            metric("holdout median d²", "2.482", "closer to training"),
        ],
        "boundary": final["claim_boundary"],
    }

    # v4 morphology cycle: use the frozen pre-adaptation diagnostic and controlled YOLO26 ablation.
    for rid, role in [("POS_6610209", "morphology training"), ("POS_8091164", "fixed validation")]:
        per = v42["per_source_iou_0_50"][rid]
        nref = v42["new_references"][rid]
        stories[rid] = {
            "record_id": rid,
            "title": "Morphology-diverse source used to test the next data strategy" if rid == "POS_6610209" else "Fixed v4 validation source exposed data insufficiency",
            "status": role,
            "phase": "v4.0–v4.3 morphology-diverse cycle",
            "question": "After the final-holdout failure, would adding visibly different tower morphology make the old geometry heuristic or a tiny closed-set detector viable?",
            "dataset": ("Close-range, non-standard angle/strain-like source accepted as a new training morphology stratum." if rid == "POS_6610209" else "Compact/angle source fixed as v4 validation so the next detector comparison could not tune against it."),
            "experiment": f"First, run the unchanged v3.8 raw YOLOE and frozen geometry methods on {nref} new references at IoU ≥ 0.50. Then use POS_8091164 as the fixed 3-insulator validation source in a controlled YOLO26n ablation: 5 towers/30 insulators versus 6 towers/37 insulators, 30 epochs per arm.",
            "result": f"On this source, raw YOLOE recall was {pct(per['raw_yoloe_text']['recall'])}; the frozen geometry method recall was {pct(per['frozen_geometry_champion']['recall'])}. In the controlled YOLO26 ablation, both baseline and expanded arms produced mAP50 = {v43['baseline']['map50']:.1f} on the fixed validation source.",
            "decision": "Stop spending additional CPU epochs at this labelled-data scale; keep YOLO26 as a future candidate and use foundation/open-vocabulary signals to drive the next data-acquisition cycle.",
            "metrics": [
                metric("new references", str(nref), "assistant-provisional"),
                metric("raw YOLOE recall", pct(per["raw_yoloe_text"]["recall"]), "IoU ≥ 0.50"),
                metric("geometry recall", pct(per["frozen_geometry_champion"]["recall"]), "IoU ≥ 0.50"),
                metric("YOLO26 mAP50", "0.0", "both 30-epoch arms"),
            ],
            "boundary": v43["claim_boundary"],
        }

    # Stress-only records retain the pre-inference visual-review semantics.
    stories["POS_7072688"].update({
        "title": "Long-range scale stress, not ground truth",
        "dataset": "Long-range Scottish source retained to represent deployment scale.",
        "experiment": "Pixel review asked whether insulator assemblies were large enough for reliable component annotation before any model inference.",
        "result": "Annotatability was too low for a defensible component benchmark.",
        "decision": "Keep it in the portfolio as a scale limitation, not in the metric denominator.",
    })
    stories["POS_7478407"].update({
        "title": "Extreme scale stress kept out of component metrics",
        "dataset": "Extreme long-range Scottish source used to show the lower end of visual component scale.",
        "experiment": "Direct pixel review before model inference.",
        "result": "Individual component detail was judged too small for reliable manual reference boxes.",
        "decision": "Use as a deployment-context example only.",
    })

    review_by_id = {r["record_id"]: r for r in v44["records"]}
    label_by_id = {r["record_id"]: r for r in material_labels["records"]}
    r809 = review_by_id["POS_8090535"]
    l809 = label_by_id["POS_8090535"]
    stories["POS_8090535"] = {
        "record_id": "POS_8090535",
        "title": "Material reference chosen before model inference",
        "status": "development reference",
        "phase": "v4.4–v4.6 material branch",
        "question": "Can the portfolio separate visually distinct glass and ceramic-family insulator assemblies without pretending the labels are laboratory-verified?",
        "dataset": "D30 angle/strain source accepted before inference because component annotatability was high and the public source description matched a visible left/right material contrast.",
        "experiment": "Six manual assembly boxes were created before material inference: three source-assisted glass references and three ceramic-family/porcelain references. These crops form the development reference set for the frozen SigLIP2 encoder.",
        "result": "This image is a reference source, not an evaluation result. Its value is the controlled, source-assisted 3+3 material prototype set.",
        "decision": "Keep it as development/reference data and separate it from the adaptive POS_2952166 predictions.",
        "metrics": [
            metric("manual boxes", str(len(l809["boxes"])), "3 glass + 3 ceramic-family"),
            metric("annotatability", r809["component_annotatability"].upper()),
            metric("material visibility", r809["material_visual_resolvability"].replace("_", " ").upper()),
        ],
        "boundary": material_labels["claim_boundary"],
    }

    methods = material_results["methods"]
    text_acc = methods["siglip2_text_prompt_prototype"]["accuracy"]
    image_acc = methods["siglip2_image_reference_prototype"]["accuracy"]
    hybrid_acc = methods["siglip2_equal_weight_text_image_hybrid"]["accuracy"]
    stories["POS_2952166"] = {
        "record_id": "POS_2952166",
        "title": "Adaptive material experiment: text prompts beat image prototypes",
        "status": "adaptive development",
        "phase": "v4.4–v4.6 material branch",
        "question": "For six manually localised material crops on a different tower, which frozen SigLIP2 prototype strategy transfers best: text, reference images, or a 50/50 hybrid?",
        "dataset": "Six source-assisted and visually corroborated boxes: three ceramic-family assemblies on image-left and three glass assemblies on image-right.",
        "experiment": "Frozen google/siglip2-base-patch16-naflex encoder; three text prompts per class; image-reference prototypes from POS_8090535; equal-weight text/image hybrid; argmax decision with no threshold tuning.",
        "result": f"On this six-crop adaptive development source, text-prompt prototype classified {round(text_acc*6)}/6 correctly, image-reference prototype {round(image_acc*6)}/6, and the equal-weight hybrid {round(hybrid_acc*6)}/6.",
        "decision": "Carry the text-prompt prototype forward as the development-selected method, but do not describe the result as independent material-holdout performance because this source had already been consumed by v4.5.",
        "metrics": [
            metric("text prototype", f"{round(text_acc*6)}/6", "adaptive crops"),
            metric("image prototype", f"{round(image_acc*6)}/6", "adaptive crops"),
            metric("50/50 hybrid", f"{round(hybrid_acc*6)}/6", "adaptive crops"),
            metric("encoder", "SigLIP2", "weights frozen"),
        ],
        "boundary": material_protocol["claim_boundary"] + " " + material_protocol["evaluation_semantics"],
    }

    r763 = review_by_id["POS_7630781"]
    stories["POS_7630781"] = {
        "record_id": "POS_7630781",
        "title": "A future detection holdout that is still unspent",
        "status": "unspent frozen holdout",
        "phase": "v4.4 terminal-morphology freeze",
        "question": "Can a later frozen v4 detector localise insulator strings on terminal-tower morphology it never saw during method development?",
        "dataset": "Terminal tower with three clear vertical insulator strings. It was selected by direct pixel review before any model inference on the image.",
        "experiment": "No headline detector experiment is shown yet. The usage rule forbids using this source for prompt, threshold, post-processing, feature-prior or detector-training decisions before a v4 operating point is frozen.",
        "result": "No result by design. Preserving an untouched future test is the result of this stage of the data protocol.",
        "decision": "Keep it frozen until a new detection method and operating point are declared; if used for headline evaluation, retire it immediately afterward.",
        "metrics": [
            metric("reference strings", "3", "assistant-provisional boxes"),
            metric("annotatability", r763["component_annotatability"].upper()),
            metric("current model result", "NOT RUN", "holdout preserved"),
        ],
        "boundary": "Single-source assistant-provisional portfolio holdout; not independently adjudicated engineering ground truth and not a production performance estimate.",
    }

    # Attach common source/image fields and ensure every record has an auditable story.
    for rid, story in stories.items():
        entry = by_id[rid]
        story["image"] = entry["full_path"]
        story["thumb"] = entry["thumb_path"]
        story["source_page"] = entry["source_page"]
        story["source_family"] = entry["source_family"]
        story["display_sha256"] = entry["sha256"]
        story["roles"] = entry["roles"]
        story["dimensions"] = [entry["width"], entry["height"]]

    featured = ["POS_2326530", "POS_3437435", "POS_8091164", "POS_2952166"]
    feature_subtitles = {
        "POS_2326530": "split discipline",
        "POS_3437435": "holdout failure",
        "POS_8091164": "data insufficiency",
        "POS_2952166": "material adaptation",
    }
    feature_html = "".join(
        f'''<button class="case-feature" type="button" data-case="{rid}">
          <img src="{html.escape(stories[rid]['thumb'])}" loading="lazy" decoding="async" alt="{rid} case-study thumbnail">
          <span class="case-feature-kicker">{html.escape(feature_subtitles[rid])}</span>
          <b>{rid}</b><small>{html.escape(stories[rid]['title'])}</small>
        </button>'''
        for rid in featured
    )

    section = f'''
<section id="cases">
  <div class="eyebrow">Case-study explorer · dataset → experiment → result → decision</div>
  <h2>Every source has a reason to exist — and a rule for what it can prove</h2>
  <p class="muted case-intro">Use the four featured transitions below, or open any card in the 16-image dataset gallery. The explorer deliberately shows “no result” for an unspent holdout and avoids reconstructing metrics that are not preserved in the current evidence chain.</p>
  <div class="case-feature-grid">{feature_html}</div>
  <div class="case-panel" id="case-panel" aria-live="polite">
    <div class="case-media"><button type="button" id="case-image-button" aria-label="Open current case image full size"><img id="case-image" alt="Current GridSight-UK case-study source"></button><div id="case-sha" class="case-sha"></div></div>
    <div class="case-content">
      <div class="case-topline"><span id="case-status" class="case-status"></span><span id="case-phase" class="case-phase"></span></div>
      <h3 id="case-title"></h3>
      <p id="case-question" class="case-question"></p>
      <div class="case-flow">
        <div><span>1 · DATASET</span><p id="case-dataset"></p></div>
        <div><span>2 · EXPERIMENT</span><p id="case-experiment"></p></div>
        <div><span>3 · RESULT</span><p id="case-result"></p></div>
        <div><span>4 · DECISION</span><p id="case-decision"></p></div>
      </div>
      <div id="case-metrics" class="case-metrics"></div>
      <div class="case-boundary"><b>Claim boundary</b><span id="case-boundary"></span></div>
      <div class="case-actions"><a id="case-source" target="_blank" rel="noopener">Geograph source ↗</a><a id="case-full" target="_blank" rel="noopener">Full project image ↗</a></div>
    </div>
  </div>
</section>
'''

    css = r'''
.case-intro{max-width:980px}.case-feature-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}.case-feature{display:grid;grid-template-columns:74px 1fr;grid-template-rows:auto auto 1fr;column-gap:10px;text-align:left;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:9px;color:inherit;cursor:pointer;min-width:0}.case-feature:hover,.case-feature.active{border-color:var(--green);background:rgba(86,208,164,.055)}.case-feature img{grid-row:1/4;width:74px;height:74px;object-fit:cover;border-radius:9px;background:#050a0f}.case-feature-kicker{text-transform:uppercase;letter-spacing:.08em;font-size:9px;color:var(--green)}.case-feature b{font-size:12px}.case-feature small{font-size:10px;color:var(--muted);line-height:1.25}.case-panel{display:grid;grid-template-columns:minmax(280px,.85fr) minmax(0,1.65fr);gap:18px;background:linear-gradient(180deg,rgba(17,29,42,.96),rgba(10,18,27,.98));border:1px solid var(--line);border-radius:18px;padding:14px}.case-media{min-width:0}.case-media button{width:100%;padding:0;border:0;background:#050a0f;border-radius:13px;overflow:hidden;cursor:zoom-in}.case-media img{display:block;width:100%;height:430px;object-fit:contain}.case-sha{margin-top:8px;color:#73889a;font:10px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere}.case-topline{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.case-status{border:1px solid var(--green);color:var(--green);border-radius:999px;padding:3px 8px;font-size:10px;text-transform:uppercase;letter-spacing:.06em}.case-phase{font-size:11px;color:var(--muted)}.case-content h3{font-size:25px;margin:9px 0 6px}.case-question{font-size:14px;color:#d7e1e9;margin:0 0 13px}.case-flow{display:grid;grid-template-columns:1fr 1fr;gap:8px}.case-flow>div{border:1px solid var(--line);border-radius:11px;padding:10px;background:rgba(5,10,15,.28)}.case-flow span{display:block;font-size:9px;letter-spacing:.09em;color:var(--blue);margin-bottom:5px}.case-flow p{margin:0;color:var(--muted);font-size:12px;line-height:1.45}.case-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px;margin:10px 0}.case-metric{border:1px solid var(--line);border-radius:10px;padding:8px;background:#0a121b}.case-metric b{display:block;color:#eef6fb;font-size:18px}.case-metric span{display:block;color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.05em}.case-metric small{display:block;color:#70879a;font-size:9px;margin-top:2px}.case-boundary{display:grid;grid-template-columns:110px 1fr;gap:8px;border-left:3px solid var(--amber);background:rgba(241,183,79,.05);padding:9px 10px;border-radius:0 9px 9px 0;font-size:11px}.case-boundary b{color:var(--amber)}.case-boundary span{color:var(--muted);line-height:1.4}.case-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:10px}.case-actions a,.case-open{color:var(--green);font-size:11px;text-decoration:none}.case-open{display:inline-flex;margin-top:9px;border:1px solid rgba(86,208,164,.35);background:rgba(86,208,164,.06);border-radius:8px;padding:5px 8px;cursor:pointer}.case-open:hover{background:rgba(86,208,164,.12)}@media(max-width:980px){.case-feature-grid{grid-template-columns:1fr 1fr}.case-panel{grid-template-columns:1fr}.case-media img{height:min(58vw,520px)}}@media(max-width:620px){.case-feature-grid{grid-template-columns:1fr}.case-flow{grid-template-columns:1fr}.case-metrics{grid-template-columns:1fr 1fr}.case-boundary{grid-template-columns:1fr}.case-content h3{font-size:21px}}
'''

    cases_json = json.dumps(stories, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    js = f'''
<script>
(function(){{
  const CASES={cases_json};
  const featured=[...document.querySelectorAll('.case-feature')];
  const panel=document.getElementById('case-panel');
  const img=document.getElementById('case-image');
  const imageButton=document.getElementById('case-image-button');
  function esc(s){{return String(s??'')}}
  function renderCase(id,scroll){{
    const c=CASES[id]; if(!c) return;
    featured.forEach(b=>b.classList.toggle('active',b.dataset.case===id));
    document.getElementById('case-status').textContent=c.status;
    document.getElementById('case-phase').textContent=c.phase;
    document.getElementById('case-title').textContent=c.title;
    document.getElementById('case-question').textContent=c.question;
    document.getElementById('case-dataset').textContent=c.dataset;
    document.getElementById('case-experiment').textContent=c.experiment;
    document.getElementById('case-result').textContent=c.result;
    document.getElementById('case-decision').textContent=c.decision;
    document.getElementById('case-boundary').textContent=c.boundary;
    document.getElementById('case-sha').textContent='Displayed SHA-256 · '+c.display_sha256+' · '+c.dimensions[0]+'×'+c.dimensions[1];
    img.src=c.image; img.alt=id+' case-study source image'; imageButton.dataset.full=c.image;
    const src=document.getElementById('case-source');src.href=c.source_page;
    const full=document.getElementById('case-full');full.href=c.image;
    document.getElementById('case-metrics').innerHTML=c.metrics.map(m=>'<div class="case-metric"><span>'+esc(m.label)+'</span><b>'+esc(m.value)+'</b><small>'+esc(m.note||'')+'</small></div>').join('');
    if(scroll) panel.scrollIntoView({{behavior:'smooth',block:'start'}});
  }}
  featured.forEach(b=>b.addEventListener('click',()=>renderCase(b.dataset.case,false)));
  document.querySelectorAll('.case-open').forEach(b=>b.addEventListener('click',()=>renderCase(b.dataset.case,true)));
  imageButton.addEventListener('click',()=>{{
    const modal=document.getElementById('modal'),modalImg=document.getElementById('modal-img');
    if(modal&&modalImg){{modalImg.src=imageButton.dataset.full;modalImg.alt='Expanded case-study source';modal.classList.add('open');modal.setAttribute('aria-hidden','false');}}
    else window.open(imageButton.dataset.full,'_blank');
  }});
  renderCase('POS_3437435',false);
}})();
</script>
'''

    page = index_path.read_text(encoding="utf-8")
    if 'id="cases"' in page:
        raise RuntimeError("Case-study explorer already present")
    if '<section id="dataset">' not in page:
        raise RuntimeError("Dataset section not found")

    # Add one case-study control to every provenance card using its unique displayed hash prefix.
    for rid, entry in by_id.items():
        needle = f'<div class="dataset-hash">Displayed SHA-256 · {entry["sha256"][:16]}…</div>'
        if page.count(needle) != 1:
            raise RuntimeError(f"Could not uniquely locate dataset card for {rid}")
        page = page.replace(needle, needle + f'<button class="case-open" type="button" data-case="{rid}">Open case study →</button>', 1)

    page = page.replace('<section id="dataset">', section + '\n<section id="dataset">', 1)
    page = page.replace('<a href="#dataset">Dataset</a>', '<a href="#cases">Case studies</a><a href="#dataset">Dataset</a>', 1)
    page = page.replace('</style>', css + '\n</style>', 1)
    page = page.replace('</body>', js + '\n</body>', 1)
    index_path.write_text(page, encoding="utf-8")

    output = {
        "version": "v4.9-case-study-explorer",
        "case_count": len(stories),
        "featured": featured,
        "default_case": "POS_3437435",
        "cases": stories,
        "claim_boundary": "Each case inherits its source-specific evidence semantics; training, validation, adaptive development, retired holdout and unspent holdout roles must remain distinct.",
    }
    (DATASET / "cases.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "index_bytes": index_path.stat().st_size,
        "case_count": len(stories),
        "featured_cases": featured,
        "dataset_case_buttons": page.count('class="case-open"'),
        "unspent_holdout_result": stories["POS_7630781"]["metrics"][-1]["value"],
        "material_text_correct": round(text_acc * 6),
        "material_image_correct": round(image_acc * 6),
        "material_hybrid_correct": round(hybrid_acc * 6),
    }, indent=2))


if __name__ == "__main__":
    main()
