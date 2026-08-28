"""Prepare a fixed public-label material experiment without running an ML model."""
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path

from audit_substation15 import ARCHIVE
from paper_material_demo import ROOT, load, sha, write

CONFIG = ROOT/'configs/substation_material_v1.json'
AUDITED = {'FLIR0335_rgb.jpg', 'WhatsApp Image 2021-07-21 at 12.01.42.jpeg', 'FLIR6829_rgb_AdgweI4.jpg'}


def capture_group(name):
    stem = Path(name).stem
    date = re.search(r'(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)', stem)
    if date:
        return 'date_' + ''.join(date.groups())
    return 'undated_' + re.split(r'\d', stem, maxsplit=1)[0].strip('_')


def polygon_references(annotation, classes):
    width, height = annotation['imageWidth'], annotation['imageHeight']
    refs = []
    for index, shape in enumerate(annotation['shapes']):
        if shape['label'] not in classes:
            continue
        points = shape['points']
        if shape.get('shape_type') != 'polygon' or len(points) < 3:
            raise ValueError('Unexpected source material annotation unit')
        xs, ys = zip(*points)
        raw = [min(xs), min(ys), max(xs), max(ys)]
        box = [max(0., raw[0]), max(0., raw[1]), min(float(width), raw[2]), min(float(height), raw[3])]
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError('Degenerate material polygon')
        refs.append({'annotation_index': index, 'class_id': classes.index(shape['label']),
                     'publisher_label': shape['label'], 'box': box, 'raw_box': raw,
                     'clipped': box != raw, 'polygon': points, 'group_id': shape.get('group_id')})
    return refs


def label_text(refs, width, height):
    lines = []
    for r in refs:
        x1, y1, x2, y2 = r['box']
        lines.append(f'{r["class_id"]} {(x1+x2)/2/width:.9f} {(y1+y2)/2/height:.9f} {(x2-x1)/width:.9f} {(y2-y1)/height:.9f}')
    return '\n'.join(lines) + ('\n' if lines else '')


def duplicate_components(rows, distance, tolerance):
    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    edges = []
    for i, a in enumerate(rows):
        for j in range(i):
            b = rows[j]
            exact = a['pixel_sha256'] == b['pixel_sha256']
            similar_aspect = abs((a['width']/a['height'])/(b['width']/b['height'])-1) <= tolerance
            d = (int(a['dhash'], 16) ^ int(b['dhash'], 16)).bit_count()
            if exact or (similar_aspect and d <= distance):
                parents[root(i)] = root(j)
                edges.append({'a': a['id'], 'b': b['id'], 'exact_pixels': exact, 'dhash_distance': d})
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    return list(groups.values()), edges


def select_rows(rows, maximum, seed):
    priority = lambda r: hashlib.sha256((seed+r['id']).encode()).hexdigest()
    buckets = defaultdict(list)
    for row in sorted(rows, key=priority):
        buckets[tuple(sorted(set(r['class_id'] for r in row['references'])))].append(row)
    selected = sorted([r for r in rows if r['source_name'] in AUDITED], key=priority)
    chosen = {r['id'] for r in selected}
    queues = [deque(r for r in buckets[key] if r['id'] not in chosen) for key in [(0,1), (1,), (0,), ()]]
    while len(selected) < maximum and any(queues):
        for q in queues:
            if q and len(selected) < maximum:
                selected.append(q.popleft())
    return selected


def main():
    from PIL import Image
    cfg = load(CONFIG); out = ROOT/cfg['dataset']
    if out.exists():
        raise FileExistsError('Prepared/incomplete dataset exists: inspect, do not overwrite or resplit')
    assert sha(ARCHIVE) == cfg['archive_sha256']
    rows = []; excluded = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        annotations = sorted(n for n in archive.namelist() if n.startswith('labels_json/') and n.endswith('.json'))
        assert len(annotations) == 1660
        for n in annotations:
            data = archive.read(n); annotation = json.loads(data)
            name = Path(annotation['imagePath']).name; member = 'images/'+name
            content = archive.read(member)
            with Image.open(io.BytesIO(content)) as im:
                if im.size != (annotation['imageWidth'], annotation['imageHeight']) or im.getexif().get(274,1) != 1:
                    excluded.append({'annotation': n, 'reason': 'EXIF orientation or annotation/native dimension mismatch'}); continue
                rgb = im.convert('RGB')
                pixels = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                small = list(rgb.convert('L').resize((9,8), Image.Resampling.LANCZOS).getdata())
                bits = [small[y*9+x] > small[y*9+x+1] for y in range(8) for x in range(8)]
                dhash = sum(int(b) << i for i,b in enumerate(bits))
            group = capture_group(name)
            rows.append({'id': hashlib.sha256(n.encode()).hexdigest()[:16], 'source_name': name,
                         'archive_image': member, 'archive_annotation': n, 'capture_group': group,
                         'split': 'development' if group in cfg['development_groups'] else 'train',
                         'image_sha256': hashlib.sha256(content).hexdigest(), 'annotation_sha256': hashlib.sha256(data).hexdigest(),
                         'pixel_sha256': pixels, 'dhash': f'{dhash:016x}', 'width': annotation['imageWidth'], 'height': annotation['imageHeight'],
                         'references': polygon_references(annotation, cfg['publisher_classes']), 'previously_inspected': name in AUDITED})
        clusters, edges = duplicate_components(rows, cfg['dhash_distance'], cfg['dhash_aspect_ratio_tolerance'])
        representatives = []; quarantine = []; duplicate_drops = []
        for cluster in clusters:
            cluster_id = min(r['id'] for r in cluster)
            if len({r['split'] for r in cluster}) > 1:
                quarantine.extend(r['id'] for r in cluster); continue
            ordered = sorted(cluster, key=lambda r: (r['source_name'] not in AUDITED, r['id']))
            representative = dict(ordered[0]); representative['duplicate_cluster'] = cluster_id
            representative['cluster_members'] = [r['id'] for r in cluster]
            representatives.append(representative); duplicate_drops.extend(r['id'] for r in ordered[1:])
        selected = []
        for split, maximum in [('train', cfg['train_images_max']), ('development', cfg['development_images_max'])]:
            selected += select_rows([r for r in representatives if r['split'] == split], maximum, cfg['seed'])
        counts = {s: Counter(r['class_id'] for row in selected if row['split']==s for r in row['references']) for s in ['train','development']}
        for s in counts:
            assert all(sum(any(r['class_id']==c for r in row['references']) for row in selected if row['split']==s)>=10 for c in [0,1]), (s, counts)
        assert not {r['capture_group'] for r in selected if r['split']=='train'} & {r['capture_group'] for r in selected if r['split']=='development'}
        out.mkdir(parents=True)
        for row in selected:
            split = row['split']; key = row['id']
            paths = {'image_file': f'images/{split}/{key}{Path(row["source_name"]).suffix.lower()}',
                     'label_file': f'labels/{split}/{key}.txt', 'annotation_file': f'annotations/{split}/{key}.json'}
            contents = [archive.read(row['archive_image']), label_text(row['references'],row['width'],row['height']).encode(), archive.read(row['archive_annotation'])]
            for (field,path),content in zip(paths.items(),contents):
                dest = out/path; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(content)
                row[field] = path
            row['label_sha256'] = sha(out/row['label_file'])
        audit = {'all_source_images': len(rows), 'exclusions': excluded, 'source_groups': dict(Counter(r['capture_group'] for r in rows)),
                 'duplicate_edges': edges, 'cross_split_quarantined': quarantine, 'same_split_duplicate_drops': duplicate_drops,
                 'selected_counts': dict(Counter(r['split'] for r in selected)), 'selected_objects': counts,
                 'selected_image_classes': {s:{c:sum(any(r['class_id']==c for r in row['references']) for row in selected if row['split']==s) for c in [0,1]} for s in ['train','development']},
                 'near_duplicate_detection_is_heuristic': True, 'same_site_asset_independence_established': False}
        write(out/'selection_audit.json',audit)
        write(out/'manifest.json', {'protocol_sha256': sha(CONFIG), 'archive_sha256': cfg['archive_sha256'],
              'selection_audit_sha256': sha(out/'selection_audit.json'), 'images': selected, 'source': cfg['source'], 'license': cfg['license'],
              'scope': cfg['split_scope'], 'no_model_generated_labels': True})
        print(json.dumps({k:v for k,v in audit.items() if k not in ['exclusions','duplicate_edges','same_split_duplicate_drops','cross_split_quarantined']},indent=2))
        print(json.dumps({'manifest_sha256':sha(out/'manifest.json'), 'quarantined':len(quarantine), 'duplicate_drops':len(duplicate_drops),'excluded_files':len(excluded)}))


if __name__ == '__main__': main()
