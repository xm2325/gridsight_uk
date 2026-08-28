#!/usr/bin/env python3
"""Create a separate English presentation without modifying frozen experiment files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/keen_components/epri_components_v1_20260827'
OUTPUT = ROOT / 'runs/keen_components/epri_components_en_20260827'
SOURCE_MANIFEST_SHA = 'e97cb9ba5920ad105991d88891a338216d516d8213fc02804f41b5aa9eb861cb'
CAPABILITY_SHA = '9f42a8a41b9390d9e8a15aae27ea1f48d9309fab7f52279b20683d1bd0bab14a'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source(source):
    manifest = source / 'report/asset_manifest.json'
    if digest(manifest) != SOURCE_MANIFEST_SHA:
        raise ValueError('Frozen report asset manifest changed')
    files = json.loads(manifest.read_text())['files']
    for name, sha in files.items():
        if digest(source / 'report' / name) != sha:
            raise ValueError(f'Frozen report asset changed: {name}')
    data = json.loads((source / 'report/data.json').read_text())
    if digest(source / 'results.json') != data['meta']['raw_results_sha256']:
        raise ValueError('Original results changed')
    checks = json.loads((source / 'report/verification.json').read_text())
    for name, key in [('frozen_choices.json', 'frozen_choices_sha256'),
                      ('dataset_manifest.json', 'dataset_manifest_sha256'),
                      ('uk_manifest.json', 'uk_manifest_sha256'),
                      ('training/weights/best.pt', 'selected_checkpoint_sha256')]:
        if digest(source / name) != checks[key]:
            raise ValueError(f'Original experiment evidence changed: {name}')
    raw_files = set()
    for rows in data['datasets'].values():
        for row in rows:
            if digest(source / 'report' / row['image_file']) != row['sha256']:
                raise ValueError('Display image no longer matches source pixels')
            for arm, name in row['raw_files'].items():
                path = source / 'report' / name
                raw = json.loads(path.read_text())
                if raw['predictions'] != row['predictions'][arm]:
                    raise ValueError('Displayed predictions differ from saved raw outputs')
                raw_files.add(str(path.resolve()))
    return data, len(files), len(raw_files)


def link_existing(target, source):
    """Idempotent links, never replace an existing file or retarget a link."""
    relative = os.path.relpath(source, target.parent)
    if target.is_symlink():
        if os.readlink(target) != relative:
            raise ValueError(f'Refusing to retarget an existing link: {target}')
    elif target.exists():
        raise ValueError(f'Refusing to replace existing content: {target}')
    else:
        target.symlink_to(relative, target_is_directory=source.is_dir())


def english_results(data):
    lines = ['# GridSight distribution component experiment', '',
             'English presentation of the completed experiment; no inference or training rerun.', '',
             '500 EPRI originals: 320 train, 80 development, 100 evaluation, split by publisher circuit groups. '
             'Another 27 UK photographs are qualitative development examples without reviewed reference labels.', '',
             '| EPRI evaluation metric | Original open vocabulary | Supervised adaptation |',
             '|---|---:|---:|']
    base = data['aggregate']['eval']['open_vocabulary']['summary']
    trained = data['aggregate']['eval']['supervised']['summary']
    for key, name in [('map50', 'Three-class mAP@0.50'), ('map50_95', 'Three-class mAP@0.50:0.95')]:
        lines.append(f'| {name} | {base[key]:.3%} | {trained[key]:.3%} |')
    lines += ['', '## Crossarm recall remains a limitation', '']
    m = trained['operating_points']['0.25']['per_class']['crossarm']
    lines.append(f"At score 0.25 and IoU 0.50: {m['tp']} TP, {m['fp']} FP, {m['fn']} FN; recall {m['recall']:.1%}. "
                 'Aggregate mAP does not hide this class-specific weakness.')
    c = data['meta']['uk_output_counts']['supervised']['0.25']
    lines += ['', '## UK transfer is not UK accuracy', '',
              f"At score 0.25, {c['images_without_output']}/27 UK images have no supervised output. "
              f"Box counts: {c['boxes_by_class']}. Empty output is not a measured false negative without reference labels.", '',
              'The separate visual-prompt diagnostic uses one reference and 26 targets, zero gradient steps. '
              'It does not establish accuracy improvement. The reference is excluded from target results.', '',
              '## Boundaries', '',
              'Only pole, crossarm and insulator were trained in this experiment. Material, steelwork and pole-top '
              'are not validated outputs here. No defect or safety claim is made. Model scores are not calibrated probabilities.', '',
              'AP uses 101-point interpolation across IoU 0.50:0.95, all sizes, and saved candidates down to 0.001. '
              'This is not an official EPRI leaderboard result. Original segmentation-base boxes and a supervised detection head '
              'are compared, not compute-matched models. Publisher circuit groups are not separately verified asset identities. '
              'Train/evaluation samples have no completely target-free images; pretraining overlap is unknown.', '',
              'The separate UK v3 review adds material diagnostics and structural hypotheses, not reviewed ground truth. '
              'All 216 material proposals remain unknown; steelwork boxes can include whole poles. '
              '[Capability design and research](KEEN_CAPABILITY_DESIGN_EN.md).', '',
              '## Sources and verification', '',
              '[EPRI release](https://www.kaggle.com/datasets/dexterlewis/epri-distribution-inspection-imagery): '
              'EPRI / P. Kulkarni / D. Lewis, CC BY-SA 4.0, DOI 10.34740/kaggle/dsv/3803175. '
              'UK photographs retain individual author/source credit and CC BY-SA 2.0.', '',
              '[English presentation verification](presentation_verification.json). '
              'Original predictions, images, metrics and frozen choices remain unchanged. No publication performed.']
    return '\n'.join(lines) + '\n'


def build(source=SOURCE, output=OUTPUT):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Presentation must be a separate directory outside the frozen source')
    data, archive_count, raw_count = verify_source(source)
    capability = ROOT / 'runs/uk_capabilities/v3_20260827/results.json'
    if digest(capability) != CAPABILITY_SHA:
        raise ValueError('Capability comparison must refer to the verified v3 result')
    capability_archive = capability.parent / 'archive_manifest.json'
    if digest(capability_archive) != '279a0b10ccc8c30813269bf3eedebec4dbfa79a153298d1cb04b8bc3b7ad9ebb':
        raise ValueError('Frozen capability archive manifest changed')
    capability_files = json.loads(capability_archive.read_text())['files']
    for item in capability_files:
        if digest(capability.parent / item['file']) != item['sha256']:
            raise ValueError(f"Frozen capability evidence changed: {item['file']}")
    report = output / 'report'
    report.mkdir(parents=True, exist_ok=True)
    link_existing(output / 'uk_capabilities_v3', capability.parent)
    for item in source.iterdir():
        if item.name != 'report':
            link_existing(output / item.name, item)
    replaced = {'index.html', 'RESULTS.md', 'UK_COMPONENT_ANNOTATION_GUIDE.md'}
    for item in (source / 'report').iterdir():
        if item.name not in replaced:
            link_existing(report / item.name, item)
    template = ROOT / 'templates/keen_components_report_en.html'
    text = template.read_text()
    if re.search(r'[\u3400-\u9fff]', text) or 'lang="en-GB"' not in text:
        raise ValueError('English presentation contains untranslated interface text')
    embedded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    generated = {'index.html': text.replace('__DATA_JSON__', embedded),
                 'RESULTS.md': english_results(data),
                 'UK_COMPONENT_ANNOTATION_GUIDE.md': (ROOT / 'EPRI_UK_ANNOTATION_GUIDE_EN.md').read_text(),
                 'KEEN_CAPABILITY_DESIGN_EN.md': (ROOT / 'KEEN_CAPABILITY_DESIGN_EN.md').read_text()}
    for name, value in generated.items():
        destination = report / name
        if destination.is_symlink():
            raise ValueError(f'Refusing to write through a symlink: {destination}')
        destination.write_text(value)
    verify_source(source)
    receipt = {'status': 'VERIFIED_ENGLISH_PRESENTATION_FROM_FROZEN_SOURCE',
               'frozen_report_files_verified': archive_count, 'raw_prediction_files_verified': raw_count,
               'data_sha256': digest(source / 'report/data.json'), 'source_asset_manifest_sha256': SOURCE_MANIFEST_SHA,
               'capability_results_sha256': CAPABILITY_SHA,
               'capability_archive_files_verified': len(capability_files),
               'template_sha256': digest(template), 'builder_sha256': digest(__file__),
               'files': {name: digest(report / name) for name in generated},
               'class_filter_semantics': 'Display only; image and aggregate scoring use all three classes',
               'training_rerun': False, 'inference_rerun': False, 'source_files_modified': False,
               'links': 'Relative symlinks to the original experiment; keep both run directories together',
               'browser_verification': 'Recorded separately in english_ui_qa.json after actual UI checks'}
    (report / 'presentation_verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.source, args.output)
