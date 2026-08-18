from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = REPORTS / "GridSight_UK_v46_SELF_CONTAINED_SHOWCASE.html"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def img_data_uri(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def extract_shift(md: str) -> dict[str, float]:
    def grab(pattern: str) -> float:
        m = re.search(pattern, md)
        if not m:
            raise RuntimeError(f"Could not parse morphology-shift value: {pattern}")
        return float(m.group(1))

    return {
        "train_d2_q95": grab(r"Training Mahalanobis d² q95: \*\*([0-9.]+)"),
        "train_d2_max": grab(r"maximum: \*\*([0-9.]+)"),
        "train_width_ratio": grab(r"Median width/tower-width: \*\*([0-9.]+)"),
        "shifted_median_d2": grab(r"POS_3437435[\s\S]*?Median d²: \*\*([0-9.]+)"),
        "shifted_width_ratio": grab(r"POS_3437435[\s\S]*?Median width/tower-width: \*\*([0-9.]+)"),
    }


def method_card(label: str, method: dict, selected: bool) -> str:
    acc = float(method["accuracy"])
    bal = float(method["balanced_accuracy"])
    cm = method["confusion_matrix"]
    badge = '<span class="badge good">development champion</span>' if selected else '<span class="badge">comparison arm</span>'
    return f"""
    <article class="method-card {'champion' if selected else ''}">
      <div class="method-head"><div><h3>{html.escape(label)}</h3>{badge}</div><div class="big">{pct(acc)}</div></div>
      <div class="track"><span style="width:{acc * 100:.1f}%"></span></div>
      <div class="method-meta"><span>accuracy {pct(acc)}</span><span>balanced {pct(bal)}</span></div>
      <div class="cm" aria-label="Confusion matrix">
        <div></div><b>pred glass</b><b>pred ceramic</b>
        <b>true glass</b><span>{cm['glass']['glass']}</span><span>{cm['glass']['ceramic_family']}</span>
        <b>true ceramic</b><span>{cm['ceramic_family']['glass']}</span><span>{cm['ceramic_family']['ceramic_family']}</span>
      </div>
    </article>
    """


def main() -> None:
    material = load_json(REPORTS / "v4_6_siglip2_material_crop_results.json")
    final = load_json(REPORTS / "v3_8_final_holdout_summary.json")
    shift_md = (REPORTS / "v3_9_morphology_shift.md").read_text(encoding="utf-8")
    shift = extract_shift(shift_md)

    methods = material["methods"]
    champion = material["development_selected_champion"]
    method_specs = [
        ("Text-prompt prototype", "siglip2_text_prompt_prototype"),
        ("Image-reference prototype", "siglip2_image_reference_prototype"),
        ("Equal-weight hybrid", "siglip2_equal_weight_text_image_hybrid"),
    ]

    image_paths = {
        key: REPORTS / f"v4_6_{key}_POS_2952166.jpg"
        for _, key in method_specs
    }
    for p in image_paths.values():
        if not p.exists():
            raise FileNotFoundError(p)
    images = {k: img_data_uri(v) for k, v in image_paths.items()}

    raw30 = final["raw_text_baseline"]["iou_0_30"]
    geo30 = final["frozen_geometry_champion"]["iou_0_30"]
    raw50 = final["raw_text_baseline"]["iou_0_50"]
    geo50 = final["frozen_geometry_champion"]["iou_0_50"]

    methods_html = "".join(
        method_card(label, methods[key], key == champion) for label, key in method_specs
    )

    prediction_rows = []
    for row in methods[champion]["predictions"]:
        prediction_rows.append(
            f"<tr><td>{html.escape(row['id'])}</td><td>{html.escape(row['true_label'])}</td>"
            f"<td>{html.escape(row['predicted_label'])}</td><td>{'✓' if row['correct'] else '×'}</td>"
            f"<td>{pct(row['relative_score'])}</td></tr>"
        )
    prediction_table = "".join(prediction_rows)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GridSight-UK v4.6 — Self-contained evidence showcase</title>
<style>
:root{{--bg:#071019;--panel:#0d1722;--panel2:#101d2b;--text:#eef5fb;--muted:#9fb0bf;--line:#243747;--accent:#58d0a6;--accent2:#6fb5ff;--warn:#f2bd68;--bad:#e47a7a;--good:#58d0a6;}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:linear-gradient(180deg,#071019 0%,#0a121b 45%,#071019 100%);color:var(--text);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
a{{color:inherit}} .wrap{{max-width:1180px;margin:auto;padding:28px 22px 64px}} .nav{{display:flex;gap:10px;flex-wrap:wrap;position:sticky;top:0;z-index:5;padding:10px 0;background:rgba(7,16,25,.92);backdrop-filter:blur(10px)}}
.nav a{{text-decoration:none;border:1px solid var(--line);padding:7px 11px;border-radius:999px;color:var(--muted)}} .nav a:hover{{border-color:var(--accent);color:var(--text)}}
.hero{{display:grid;grid-template-columns:1.45fr .85fr;gap:20px;align-items:stretch;margin-top:10px}} .panel{{background:rgba(13,23,34,.88);border:1px solid var(--line);border-radius:18px;padding:18px}} .hero h1{{font-size:36px;line-height:1.08;margin:0 0 8px}} .kicker{{color:var(--accent);font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:12px}} .sub{{color:var(--muted);max-width:820px}}
.hero-image{{overflow:hidden;padding:0}} .hero-image img{{width:100%;height:100%;min-height:430px;object-fit:contain;background:#050a0f;display:block}} .stack{{display:grid;gap:12px}} .metric{{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:14px}} .metric .value{{font-size:32px;font-weight:800;line-height:1}} .metric .label{{color:var(--muted);margin-top:5px}} .metric.good .value{{color:var(--good)}} .metric.warn .value{{color:var(--warn)}}
section{{margin-top:42px;scroll-margin-top:70px}} h2{{font-size:26px;margin:0 0 8px}} h3{{margin:0 0 6px;font-size:16px}} .section-intro{{color:var(--muted);max-width:900px;margin-bottom:18px}} .grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}} .method-card{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px}} .method-card.champion{{border-color:var(--accent)}} .method-head{{display:flex;justify-content:space-between;gap:12px;align-items:start}} .big{{font-size:27px;font-weight:800}} .badge{{display:inline-block;margin-top:6px;padding:3px 7px;border-radius:999px;background:#152434;color:var(--muted);font-size:11px}} .badge.good{{background:rgba(88,208,166,.12);color:var(--good)}}
.track{{height:9px;background:#172735;border-radius:999px;overflow:hidden;margin:13px 0 6px}} .track span{{display:block;height:100%;background:var(--accent)}} .method-meta{{display:flex;justify-content:space-between;color:var(--muted);font-size:12px}} .cm{{display:grid;grid-template-columns:1.2fr 1fr 1fr;margin-top:14px;border-top:1px solid var(--line);border-left:1px solid var(--line);font-size:12px}} .cm>*{{padding:6px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);text-align:center}} .cm b{{color:var(--muted);font-weight:600}}
.gallery-controls{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}} button{{appearance:none;border:1px solid var(--line);background:#101d2b;color:var(--text);padding:8px 11px;border-radius:10px;cursor:pointer}} button.active{{border-color:var(--accent);background:rgba(88,208,166,.12)}} .gallery{{display:grid;grid-template-columns:1.5fr .7fr;gap:14px}} .gallery img{{width:100%;display:block;border-radius:14px;background:#050a0f}} .callout{{border-left:3px solid var(--accent);padding:12px 14px;background:rgba(88,208,166,.07);border-radius:0 12px 12px 0}} .warning{{border-left-color:var(--warn);background:rgba(242,189,104,.07)}}
.compare{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}} .barrow{{display:grid;grid-template-columns:165px 1fr 70px;gap:10px;align-items:center;margin:11px 0}} .bar{{height:11px;background:#172735;border-radius:999px;overflow:hidden}} .bar span{{height:100%;display:block;background:var(--accent2)}} .bar.geo span{{background:var(--warn)}} .small{{font-size:12px;color:var(--muted)}}
.shift{{display:grid;grid-template-columns:.8fr 1.2fr;gap:14px}} .shift-visual{{display:flex;gap:14px;align-items:end;min-height:250px;padding:20px}} .pillar{{flex:1;display:flex;flex-direction:column;justify-content:end;align-items:center;gap:8px;height:210px}} .pillar .shape{{width:68%;min-height:10px;border-radius:10px 10px 3px 3px;background:var(--accent2)}} .pillar.shifted .shape{{background:var(--bad)}} .pillar b{{font-size:22px}} .pillar span{{color:var(--muted);text-align:center;font-size:12px}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:left}} th{{color:var(--muted);font-weight:600}} .evidence{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}} .step{{border:1px solid var(--line);border-radius:14px;padding:13px;background:var(--panel)}} .step b{{display:block;margin-bottom:4px}} .step span{{color:var(--muted);font-size:12px}} footer{{margin-top:46px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:18px}}
@media(max-width:850px){{.hero,.gallery,.shift{{grid-template-columns:1fr}} .grid3{{grid-template-columns:1fr}} .compare{{grid-template-columns:1fr}} .evidence{{grid-template-columns:1fr 1fr}} .hero h1{{font-size:30px}} .hero-image img{{min-height:0}}}} @media(max-width:520px){{.evidence{{grid-template-columns:1fr}} .barrow{{grid-template-columns:125px 1fr 55px}} .wrap{{padding-left:14px;padding-right:14px}}}}
</style>
</head>
<body>
<div class="wrap">
  <nav class="nav"><a href="#overview">Overview</a><a href="#material">Material</a><a href="#detection">Detection</a><a href="#shift">Domain shift</a><a href="#evidence">Evidence contract</a></nav>

  <section id="overview" class="hero">
    <div class="panel hero-image"><img src="{images['siglip2_text_prompt_prototype']}" alt="SigLIP2 text-prompt material classification overlay on POS_2952166"></div>
    <div class="panel stack">
      <div><div class="kicker">GridSight-UK · v4.6</div><h1>Overhead-line vision under domain shift</h1><p class="sub">A compact evidence view of component localisation failure analysis and adaptive material classification. The page is fully self-contained: images, metrics, CSS and JavaScript are embedded in this HTML.</p></div>
      <div class="metric good"><div class="value">6 / 6</div><div class="label">SigLIP2 text-prompt material crops correct on the adaptive development source — not an independent holdout.</div></div>
      <div class="metric"><div class="value">{pct(raw30['recall'])}</div><div class="label">Raw YOLOE recall at IoU ≥ 0.30 on the frozen 12-insulator final holdout.</div></div>
      <div class="metric warn"><div class="value">{pct(geo30['recall'])}</div><div class="label">Frozen geometry-filter recall on the same final holdout, exposing the domain-shift failure.</div></div>
    </div>
  </section>

  <section id="material">
    <div class="kicker">Material branch</div><h2>SigLIP2 cross-source crop classification</h2>
    <p class="section-intro">Manual/oracle component boxes are held fixed. Only the material label is predicted. Scores shown on images are two-class relative, uncalibrated display scores. v4.6 uses POS_8090535 as the reference source and POS_2952166 as an adaptive development source.</p>
    <div class="grid3">{methods_html}</div>
    <div style="height:18px"></div>
    <div class="gallery-controls">
      <button class="active" data-key="siglip2_text_prompt_prototype">Text prompt</button>
      <button data-key="siglip2_image_reference_prototype">Image reference</button>
      <button data-key="siglip2_equal_weight_text_image_hybrid">Hybrid</button>
    </div>
    <div class="gallery">
      <div class="panel"><img id="material-image" src="{images['siglip2_text_prompt_prototype']}" alt="Selected material method overlay"></div>
      <div class="panel">
        <div class="callout"><b>What this proves</b><br>On this adaptive six-crop comparison, the frozen text-prompt prototype separated the three glass and three ceramic-family references correctly.</div>
        <div style="height:12px"></div>
        <div class="callout warning"><b>What it does not prove</b><br>This is not independent material holdout performance, not automatic component localisation, and not a defect, condition or safety result.</div>
        <div style="height:14px"></div>
        <table><thead><tr><th>crop</th><th>true</th><th>pred</th><th></th><th>relative score</th></tr></thead><tbody>{prediction_table}</tbody></table>
      </div>
    </div>
  </section>

  <section id="detection">
    <div class="kicker">Frozen evidence</div><h2>Development success did not survive the final holdout</h2>
    <p class="section-intro">The v3.8 final evaluator ran once on two previously unseen towers containing 12 assistant-provisional insulator references. The holdout is now retired from further tuning.</p>
    <div class="compare">
      <div class="panel"><h3>IoU ≥ 0.30</h3>
        <div class="barrow"><span>Raw YOLOE recall</span><div class="bar"><span style="width:{raw30['recall']*100:.1f}%"></span></div><b>{pct(raw30['recall'])}</b></div>
        <div class="barrow"><span>Geometry recall</span><div class="bar geo"><span style="width:{geo30['recall']*100:.1f}%"></span></div><b>{pct(geo30['recall'])}</b></div>
        <div class="barrow"><span>Raw YOLOE F1</span><div class="bar"><span style="width:{raw30['f1']*100:.1f}%"></span></div><b>{pct(raw30['f1'])}</b></div>
        <div class="barrow"><span>Geometry F1</span><div class="bar geo"><span style="width:{geo30['f1']*100:.1f}%"></span></div><b>{pct(geo30['f1'])}</b></div>
        <p class="small">Raw: {raw30['tp']} TP / {raw30['fp']} FP / {raw30['fn']} FN. Geometry: {geo30['tp']} TP / {geo30['fp']} FP / {geo30['fn']} FN.</p>
      </div>
      <div class="panel"><h3>IoU ≥ 0.50</h3>
        <div class="barrow"><span>Raw YOLOE recall</span><div class="bar"><span style="width:{raw50['recall']*100:.1f}%"></span></div><b>{pct(raw50['recall'])}</b></div>
        <div class="barrow"><span>Geometry recall</span><div class="bar geo"><span style="width:{geo50['recall']*100:.1f}%"></span></div><b>{pct(geo50['recall'])}</b></div>
        <div class="barrow"><span>Raw YOLOE F1</span><div class="bar"><span style="width:{raw50['f1']*100:.1f}%"></span></div><b>{pct(raw50['f1'])}</b></div>
        <div class="barrow"><span>Geometry F1</span><div class="bar geo"><span style="width:{geo50['f1']*100:.1f}%"></span></div><b>{pct(geo50['f1'])}</b></div>
        <p class="small">The geometry champion reaches zero TP at IoU ≥ 0.50. This result is preserved rather than tuned away.</p>
      </div>
    </div>
  </section>

  <section id="shift">
    <div class="kicker">Failure diagnosis</div><h2>The issue was morphology shift, not just a threshold</h2>
    <div class="shift">
      <div class="panel shift-visual">
        <div class="pillar"><b>{shift['train_d2_max']:.3f}</b><div class="shape" style="height:{max(12, shift['train_d2_max']/shift['shifted_median_d2']*170):.0f}px"></div><span>training maximum<br>Mahalanobis d²</span></div>
        <div class="pillar shifted"><b>{shift['shifted_median_d2']:.3f}</b><div class="shape" style="height:170px"></div><span>shifted tower median<br>Mahalanobis d²</span></div>
      </div>
      <div class="panel"><h3>POS_3437435 · England</h3><p>All <b>6 / 6</b> insulators exceeded even the maximum training geometry distance. Their median width/tower-width changed from <b>{shift['train_width_ratio']:.3f}</b> in training to <b>{shift['shifted_width_ratio']:.3f}</b>, consistent with wide horizontal/strain-like morphology absent from the narrow training cohort.</p>
      <div class="callout"><b>Engineering decision</b><br>Retire the narrow geometry filter from the mainline. Start a new morphology-diverse cycle with angle, strain and terminal structures and a new preregistered holdout.</div></div>
    </div>
  </section>

  <section id="evidence">
    <div class="kicker">Evidence contract</div><h2>What is model output, and what is not</h2>
    <p class="section-intro">The visual is designed to look polished without crossing the evidence boundary.</p>
    <div class="evidence">
      <div class="step"><b>1 · Source pixels</b><span>Public UK tower imagery with recorded provenance and hash identity.</span></div>
      <div class="step"><b>2 · Manual/oracle box</b><span>v4.6 material crops use manually defined component locations; localisation is not predicted here.</span></div>
      <div class="step"><b>3 · SigLIP2 label</b><span>Frozen vision-language features classify glass vs ceramic-family.</span></div>
      <div class="step"><b>4 · Relative score</b><span>Softmax-normalised two-class display score. It is explicitly not a calibrated probability.</span></div>
    </div>
    <div style="height:14px"></div>
    <div class="callout warning"><b>Claim boundary:</b> source-assisted material references and assistant-provisional component references are portfolio evidence. No condition, corrosion, defect, failure, mechanical-integrity or safety-risk performance is claimed.</div>
  </section>

  <footer>Generated automatically from persisted GridSight-UK evidence files and v4.6 GitHub Actions outputs. Final-holdout images remain retired from tuning. This HTML contains no external CSS, JavaScript, fonts or image dependencies.</footer>
</div>
<script>
const images = {json.dumps(images)};
const image = document.getElementById('material-image');
document.querySelectorAll('[data-key]').forEach(btn => btn.addEventListener('click', () => {{
  document.querySelectorAll('[data-key]').forEach(x => x.classList.remove('active'));
  btn.classList.add('active');
  image.src = images[btn.dataset.key];
  image.alt = btn.textContent + ' material classification overlay';
}}));
</script>
</body>
</html>"""

    OUT.write_text(html_text, encoding="utf-8")
    # Hard self-contained checks: no remote assets or external scripts/styles.
    text = OUT.read_text(encoding="utf-8")
    forbidden = ["<script src=", "<link rel=", "src=\"http://", "src=\"https://"]
    for token in forbidden:
        if token in text:
            raise RuntimeError(f"Self-contained check failed: {token}")
    if text.count("data:image/jpeg;base64,") < 3:
        raise RuntimeError("Expected three embedded v4.6 JPEGs")
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "bytes": OUT.stat().st_size, "embedded_jpegs": 3, "self_contained": True}, indent=2))


if __name__ == "__main__":
    main()
