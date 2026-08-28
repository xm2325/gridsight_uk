#!/usr/bin/env python3
"""Audit saved development predictions and UK review needs; never create labels."""
from __future__ import annotations

import html
import json
import ast
import csv
from collections import Counter
from pathlib import Path

from prepare_keen_components import ROOT, digest, write_json, polygon_box
from keen_component_metrics import match_image, iou
from roihu_keen_components import verify_predictions

RUN = ROOT / 'runs/keen_components/epri_components_v1_20260827'
UK = ROOT / 'runs/uk_capabilities/v3_20260827'
OUT = ROOT / 'runs/keen_components/epri_components_en_20260827/report'
LABELS = {
    'matched': 'Matched at score 0.25',
    'assignment_conflict': 'High-score overlap, but assigned to another reference',
    'low_score_overlap': 'Geometric support exists below score 0.25',
    'suppressed_overlap': 'Geometric support exists only before saved NMS',
    'wrong_class_overlap': 'High-score overlap from another component class',
    'partial_overlap': 'Same-class partial overlap, below IoU 0.50',
    'no_overlap': 'No saved same-class candidate reaches IoU 0.10',
}


def miss_reason(reference, predictions, raw_predictions, confidence=.25):
    """Describe saved geometric evidence, not a causal diagnosis or recovery count.

    Call only for references missed by the existing one-to-one matcher. A low
    score box may cover multiple references; it is not an independently
    recoverable true positive for every such reference.
    """
    def rank(items):
        return sorted(items, key=lambda p: (-iou(p['box'], reference['box']), -p['score']))
    same = rank([p for p in predictions if p['class_id'] == reference['class_id']])
    aligned = [p for p in same if iou(p['box'], reference['box']) >= .5]
    high = [p for p in aligned if p['score'] >= confidence]
    raw_aligned = rank([p for p in raw_predictions if p['class_id'] == reference['class_id']
                        and iou(p['box'], reference['box']) >= .5])
    wrong = rank([p for p in predictions if p['class_id'] != reference['class_id']
                  and p['score'] >= confidence and iou(p['box'], reference['box']) >= .5])
    if high:
        reason, candidate = 'assignment_conflict', high[0]
    elif aligned:
        reason, candidate = 'low_score_overlap', aligned[0]
    elif raw_aligned:
        reason, candidate = 'suppressed_overlap', raw_aligned[0]
    elif wrong:
        reason, candidate = 'wrong_class_overlap', wrong[0]
    elif same and iou(same[0]['box'], reference['box']) >= .1:
        reason, candidate = 'partial_overlap', same[0]
    else:
        reason, candidate = 'no_overlap', same[0] if same else None
    return {'reason': reason, 'supporting_candidate': candidate,
            'supporting_iou': iou(candidate['box'], reference['box']) if candidate else None,
            'candidate_scope': 'before_nms' if reason == 'suppressed_overlap' else 'after_nms',
            'recoverable_true_positive_claim': False}


def review_needs(row):
    diagnostics = row['material_diagnostics']
    reasons = Counter(d['reason'] for d in diagnostics)
    native_pass = len(diagnostics) - reasons['insufficient_native_pixels']
    hardware = [p for p in row['predictions']['dino_hardware'] if p['score'] >= .3]
    # A broad box is a geometric review flag, not proof of a model error.
    broad = sum((p['box'][2]-p['box'][0])*(p['box'][3]-p['box'][1]) /
                (row['width']*row['height']) >= .25 for p in hardware)
    needs = ['Review the complete image and add missing objects; candidates are not references',
             'Verify asset identity before any train/evaluation split',
             'Provide independent material evidence or record unknown']
    if reasons['insufficient_native_pixels']:
        needs.append('Inspect inadequate crops; request a closer original view where necessary')
    if reasons['crop_context_disagreement']:
        needs.append('Check crop boundaries and background dependence for material disagreements')
    if hardware:
        needs.append('Separate structural member/assembly extent from pole and attached equipment')
    return {k: row[k] for k in ['image_id','sha256','width','height','title','credit','source_page','license','license_url']} | {
        'proposal_count': len(diagnostics), 'native_gate_pass': native_pass,
        'native_gate_is_validated': False, 'material_reasons': dict(reasons),
        'hardware_proposals_at_030': len(hardware), 'broad_hardware_review_flags': broad,
        'material_labels_verified': 0, 'asset_id': None,
        'review_status': 'UNREVIEWED', 'training_approved': False, 'needs': needs,
        'priority_key': [-reasons['crop_context_disagreement'], -native_pass, row['image_id']],
        'priority_is_accuracy_or_suitability': False,
    }


def trace_publisher_labels(label, row, indices):
    """Verify selected references are exact conversions, without correcting labels."""
    traced = []
    for index in indices:
        reference = row['references'][index]
        object_index = int(reference['annotation_id'].rsplit('_',1)[1])
        source = label['objects'][object_index]
        box, _ = polygon_box(source['polygon'], row['width'], row['height'])
        if source['value'] != reference['class_name'] or source['polygon'] != reference['polygon'] or box != reference['box']:
            raise ValueError('Derived reference does not reproduce its publisher object')
        traced.append({'reference_index':index, 'source_object_index':object_index,
                       'publisher_value':source['value'], 'publisher_polygon':source['polygon'],
                       'derived_box':box, 'mapping_verified':True,
                       'expert_review_status':'PENDING', 'original_label_changed':False})
    return traced


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    assert digest(RUN/'dataset_manifest.json') == '56e0517fcbf864f6c60aa1e2b0869cf9061138a32eb2b2acd40ad37efcb8cffa'
    uk_archive = UK/'archive_manifest.json'
    assert digest(uk_archive) == '279a0b10ccc8c30813269bf3eedebec4dbfa79a153298d1cb04b8bc3b7ad9ebb'
    uk_files = {r['file']:r['sha256'] for r in json.loads(uk_archive.read_text())['files']}
    assert digest(UK/'report/data.json') == uk_files['report/data.json']
    manifest = json.loads((RUN/'dataset_manifest.json').read_text())
    rows = [r for r in manifest['images'] if r['split'] == 'dev']
    assert len(rows) == 80 and {r['circuit'] for r in rows} == {4}
    saved = json.loads((RUN/'dev/supervised.json').read_text())
    records = {r['image_id']: r for r in saved['records']}
    assert set(records) == {r['image_id'] for r in rows}
    assert verify_predictions(RUN/'dev', rows, saved['records'], ['pole','crossarm','insulator'], .001) == 80
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in
              [Path(__file__).resolve(), RUN/'dataset_manifest.json', RUN/'dev/supervised.json', RUN/'frozen_choices.json', uk_archive,
               UK/'report/data.json', UK/'results.json', ROOT/'data/external/uk_distribution_pilot_v1/manifest.json']}
    counts = Counter(); objects = []; operating = {k: Counter() for k in ('0.05','0.25','0.50')}
    for row in rows:
        path = RUN/'dev'/records[row['image_id']]['prediction_file']
        hashes[str(path.relative_to(ROOT))] = digest(path)
        raw = json.loads(path.read_text()); predictions = raw['predictions']
        matches = match_image(predictions, row['references'], confidence=.25, class_id=1)
        matched = {m['reference_index']: m for m in matches['matches']}
        for threshold in operating:
            m = match_image(predictions, row['references'], confidence=float(threshold), class_id=1)
            for key in ('tp','fp','fn'): operating[threshold][key] += m[key]
        for index, reference in enumerate(row['references']):
            if reference['class_id'] != 1: continue
            entry = {'image_id': row['image_id'], 'reference_index': index, 'reference': reference,
                     'width': row['width'], 'height': row['height'], 'image_sha256': row['sha256'],
                     'image_file': 'images/'+row['image_id']+'.jpg',
                     'raw_prediction_file': '../dev/'+records[row['image_id']]['prediction_file']}
            if index in matched:
                p = predictions[matched[index]['prediction_index']]
                entry.update(reason='matched', supporting_candidate=p,
                             supporting_iou=matched[index]['iou'], candidate_scope='after_nms')
            else:
                entry.update(miss_reason(reference, predictions, raw['raw_predictions']))
            counts[entry['reason']] += 1; objects.append(entry)
    for threshold, values in operating.items():
        expected = saved['summary']['operating_points'][threshold]['per_class']['crossarm']
        assert all(values[k] == expected[k] for k in ('tp','fp','fn'))
    assert sum(counts.values()) == saved['summary']['ap']['crossarm']['support'] == 45
    # This bounded flag comes from inspecting the rendered development example,
    # not from automatically assuming that a confident model is ground truth.
    csv_path = ROOT/'runtime/target_sources/Overhead-Distribution-Labels.csv'
    assert digest(csv_path) == 'a2b75c6d6aa08e2e7620ca007eb9b6b52546b4ee119f2105ec1ac689bedd3a52'
    source_row = next(r for r in rows if r['image_id']=='epri_c4_176')
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        label = next(ast.literal_eval(r['Label']) for r in csv.DictReader(stream) if r['External ID']==source_row['file_name'])
    source_label_review = {'image_id':source_row['image_id'], 'source_csv_sha256':digest(csv_path),
                          'observation':'Three independently mounted circular units visually resemble insulators, but the publisher labels them crossarm. This is an assistant review hypothesis, not corrected ground truth.',
                          'reviewed_source':'Full original development image and exact publisher CSV objects',
                          'objects':trace_publisher_labels(label,source_row,[1,2,3]),
                          'material_inferred':False, 'metrics_recomputed_with_corrected_labels':False}
    hashes[str(csv_path.relative_to(ROOT))] = digest(csv_path)
    uk = json.loads((UK/'report/data.json').read_text())
    assert digest(UK/'results.json') == '9f42a8a41b9390d9e8a15aae27ea1f48d9309fab7f52279b20683d1bd0bab14a'
    original_manifest = json.loads((ROOT/'data/external/uk_distribution_pilot_v1/manifest.json').read_text())
    original_rows = {r['image_id']: r for r in original_manifest['images']}
    queue = []
    for row in uk['images']:
        original = original_rows[row['image_id']]
        image = UK/'report'/row['image_file']
        assert digest(image) == row['sha256'] == original['sha256']
        assert (row['width'],row['height']) == (original['width'],original['height'])
        assert image.stat().st_size == original['published_bytes']
        item = review_needs(row)
        item['original_publisher_dimensions_verified'] = True
        item['image_file'] = '../uk_capabilities_v3/'+str(Path('report')/row['image_file'])
        queue.append(item)
    queue.sort(key=lambda r: r['priority_key'])
    cases = []
    for reason in LABELS:
        options = sorted((r for r in objects if r['reason']==reason), key=lambda r:(r['image_id'],r['reference_index']))
        if options:
            selected = options[0]
            assert digest(OUT/selected['image_file']) == selected['image_sha256']
            cases.append(selected)
    result = {'status':'VERIFIED_SAVED_DEVELOPMENT_ERROR_AUDIT', 'development_images':80,
              'development_circuit':4, 'crossarm_references':45, 'reasons':dict(counts),
              'crossarm_operating_points': {t:saved['summary']['operating_points'][t]['per_class']['crossarm'] for t in operating},
              'objects':objects, 'cases':cases,
              'source_label_review':source_label_review,
              'case_rule':'First image ID and reference index per geometric category; diagnostic examples, not a new benchmark',
              'uk_review_queue':queue, 'uk_review_order':'Crop-context disagreements descending, native-gate candidates descending, image ID; not an accuracy ranking',
              'evaluation_images_or_predictions_read':False, 'inference_rerun':False,
              'labels_created':False, 'training_started':False, 'training_approved':False,
              'source_sha256':hashes,
              'limitations':['Geometric categories are evidence patterns, not proven root causes',
                             'Low-score support is not a recoverable TP count; one candidate may cover multiple references',
                             'UK candidates and quality gates are unvalidated; every image needs full review',
                             'No independent material labels or verified UK asset IDs are available']}
    write_json(OUT/'development_audit.json', result)
    render(result)
    for name, sha in hashes.items(): assert digest(ROOT/name) == sha
    print(json.dumps({'status':result['status'],'reasons':dict(counts),'operating_points':result['crossarm_operating_points'],
                      'uk_images':len(queue),'html':str(OUT/'development_audit.html')},indent=2))


def render(data):
    esc = lambda x: html.escape(str(x), quote=True)
    parts = ['<!doctype html><html lang="en-GB"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>GridSight · Development error audit</title>',
             '<style>body{margin:0;background:#f1f5fa;color:#14273d;font:15px/1.6 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:28px}h1{line-height:1.2}h2{margin-top:34px}a{color:#185fb2}section,article{background:white;border:1px solid #d7e0eb;border-radius:10px;padding:20px;margin:18px 0}.notice{background:#fff4dc;padding:14px;border-radius:8px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:11px;border-bottom:1px solid #dee5ef}th{background:#f6f9fc}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid article{margin:0}svg{width:100%;height:300px;background:#102238}small{display:block;color:#52667d}code{word-break:break-all}.tag{background:#fff0c9;padding:2px 5px}@media(max-width:700px){.grid{grid-template-columns:1fr}main{padding:14px}}</style><main>',
             '<a href="index.html?presentation=english-v2">← English component explorer</a> · <a href="http://127.0.0.1:8772/report/index.html">UK review workbench ↗</a>',
             '<h1>Why are crossarms missed?</h1><p>Saved supervised predictions on <b>80 EPRI development images / circuit 4</b>. No inference rerun, no new labels, no frozen-evaluation images or predictions used.</p>',
             '<div class="notice">This audit diagnoses existing development behaviour. It does not improve the trained model or establish UK performance. Geometric patterns are not proven causes.</div>',
             '<h2>45 crossarm references</h2><div class="scroll"><table><tr><th>Pattern at score 0.25 / IoU 0.50</th><th>References</th></tr>']
    for key,label in LABELS.items(): parts.append(f'<tr><td>{label}</td><td>{data["reasons"].get(key,0)}</td></tr>')
    parts += ['</table></div><h2>The cost of lowering the threshold</h2><div class="scroll"><table><tr><th>Score threshold</th><th>TP / FP / FN</th><th>Precision</th><th>Recall</th></tr>']
    for threshold,m in data['crossarm_operating_points'].items():
        parts.append(f'<tr><td>{threshold}</td><td>{m["tp"]} / {m["fp"]} / {m["fn"]}</td><td>{m["precision"]:.1%}</td><td>{m["recall"]:.1%}</td></tr>')
    parts += ['</table></div><p>These are the three thresholds already saved in the original protocol, not a new threshold search. Low-score geometric support does not imply that every missed reference can be recovered: matching remains one-to-one and false positives also increase.</p>',
              '<section><h2>Engineering implications</h2><p>Audit crossarm boundaries and class-specific score behaviour before simply increasing model size. After target-domain labels exist, compare a development-selected per-class operating point, automatic local crops and supervised adaptation. Retain full-image misses and test on new independent assets. Do not apply an EPRI threshold to UK photographs as a validated setting.</p></section>',
              '<section class="notice"><h2>Source-label review flag: epri_c4_176</h2><p>Three individually mounted units visually resemble insulators, yet all three are labelled <b>crossarm</b> in the publisher CSV. The saved polygons and derived boxes exactly reproduce those source objects: this is not a conversion mismatch. The confident insulator predictions alone do not establish the correct labels.</p><p><b>Expert review is pending.</b> No labels were changed, no material was inferred and no corrected-label metrics were calculated. See <code>source_label_review</code> in the audit JSON for the original polygons and CSV fingerprint. This may explain some apparent class-confusion errors, but it does not establish the prevalence of label noise.</p></section>',
              '<h2>Inspect the actual geometry</h2><p>One deterministic example per category. These <b>reference-guided audit crops</b> are for error inspection only; no reference boxes were supplied to inference. Dashed white: publisher-derived reference. Cyan: saved supporting candidate, which may be the wrong class or a pre-NMS candidate.</p><div class="grid">']
    for r in data['cases']:
        ref=r['reference']['box'];p=r['supporting_candidate'];b=p['box'] if p else ref
        left,top=min(ref[0],b[0]),min(ref[1],b[1]);right,bottom=max(ref[2],b[2]),max(ref[3],b[3])
        pad=max(right-left,bottom-top)*.15
        x,y=max(0,left-pad),max(0,top-pad);right,bottom=min(r['width'],right+pad),min(r['height'],bottom+pad)
        parts += [f'<article><h3>{esc(LABELS[r["reason"]])}</h3><small>{esc(r["image_id"])} · reference {r["reference_index"]}</small>',
                  f'<svg role="img" aria-label="Reference and saved candidate comparison for {esc(r["image_id"])}" viewBox="{x} {y} {right-x} {bottom-y}"><image href="{esc(r["image_file"])}" width="{r["width"]}" height="{r["height"]}"/>',
                  f'<rect x="{ref[0]}" y="{ref[1]}" width="{ref[2]-ref[0]}" height="{ref[3]-ref[1]}" fill="none" stroke="white" stroke-width="3" stroke-dasharray="9 5" vector-effect="non-scaling-stroke"/>']
        if p:
            parts += [f'<rect x="{b[0]}" y="{b[1]}" width="{b[2]-b[0]}" height="{b[3]-b[1]}" fill="none" stroke="#28d8e8" stroke-width="2" vector-effect="non-scaling-stroke"/>']
        parts += ['</svg>']
        detail = f'{["pole","crossarm","insulator"][p["class_id"]]} · score {p["score"]:.4f} · IoU {r["supporting_iou"]:.3f} · {r["candidate_scope"]}' if p else 'No saved crossarm candidate'
        parts += [f'<p>{esc(detail)}</p><a href="{esc(r["image_file"])}">Full original</a> · <a href="{esc(r["raw_prediction_file"])}">Raw prediction</a><small>EPRI / P. Kulkarni / D. Lewis · CC BY-SA 4.0</small></article>']
    parts += ['</div><h2>UK full-image review queue</h2><p>All 27 images remain unreviewed development data. Order prioritises crop-context disagreement, then native-gate candidate count; it does not rank accuracy or prove image suitability. Every original matches publisher dimensions and bytes. No material or asset identity has been inferred.</p>',
              '<div class="scroll"><table><tr><th>Image</th><th>Material proposals</th><th>Pass native gate*</th><th>Crop disagreement</th><th>Hardware at 0.30</th><th>Broad box flags*</th></tr>']
    for r in data['uk_review_queue']:
        parts += [f'<tr><td><a href="{esc(r["image_file"])}">{esc(r["image_id"])}</a><small>{esc(r["credit"])} · <a href="{esc(r["license_url"])}">{esc(r["license"])}</a> · <a href="{esc(r["source_page"])}">Source</a></small></td><td>{r["proposal_count"]}</td><td>{r["native_gate_pass"]}</td><td>{r["material_reasons"].get("crop_context_disagreement",0)}</td><td>{r["hardware_proposals_at_030"]}</td><td>{r["broad_hardware_review_flags"]}</td></tr>']
    parts += ['</table></div><small>* Heuristic review flags, not validated material quality or error labels. Broad hardware boxes cover at least 25% of the full image. Native-gate settings are those of the frozen v3 diagnostic.</small>',
              '<section><h2>Evidence still needed before training</h2><p>Reviewed component boundaries, independently supported material labels or explicit unknowns, verified asset identities, and agreed definitions for structural members/assemblies and pole-top. The existing workbench can record drafts, but saving a draft is not training approval.</p><p>For steelwork, distinguish structural role from composition. For pole-top, agree shaft tip versus upper assembly; derived search windows are neither trained detections nor material evidence.</p></section>',
              '<p><a href="development_audit.json">Complete audit, all 45 references, all 27 review tasks and source hashes</a> · <a href="KEEN_CAPABILITY_DESIGN_EN.md">Capability design</a></p>',
              '<footer>GridSight · Local diagnostic · No training, new inference, ground-truth creation or risk claim. EPRI images: EPRI / P. Kulkarni / D. Lewis, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>, <a href="https://www.kaggle.com/datasets/dexterlewis/epri-distribution-inspection-imagery">publisher release</a>. Original pixels are unchanged; reference and prediction overlays are separate presentation elements.</footer></main></html>']
    (OUT/'development_audit.html').write_text('\n'.join(parts))


if __name__ == '__main__':
    build()
