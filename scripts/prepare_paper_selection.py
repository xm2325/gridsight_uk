"""Recreate the historical UVInsDet demo selection from the verified public ZIP.

This reproduces a recorded selection, not a fresh random or held-out sample.
It does not download data, inspect predictions, load models, or overwrite changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from paper_material_demo import ROOT, ZIP_SHA, sha

SELECTED = {
    'train': ['Image_172.jpg', 'Image_14.jpg', 'Image_159.jpg', 'Image_76.jpg'],
    'test': ['Image_156.jpg', 'Image_164.jpg', 'Image_221.jpg', 'Image_8.jpg',
             'Image_60.jpg', 'Image_102.jpg', 'Image_39.jpg', 'Image_1.jpg'],
}
SELECTION_SHA = 'c2245a2f83cffd3a300e31cd9977307c873cc6247c7a0ac8201097dabf712014'


def selection_rows(sources, selected=SELECTED):
    rows = []
    for split, filenames in selected.items():
        coco = sources[split]
        for filename in filenames:
            matches = [i for i in coco['images'] if i['file_name'] == filename]
            if len(matches) != 1:
                raise ValueError(f'Expected one publisher image for {split}/{filename}')
            image = matches[0]
            rows.append({'split': split, 'image': image,
                         'annotations': [a for a in coco['annotations'] if a['image_id'] == image['id']]})
    return rows


def prepare(archive, output):
    if sha(archive) != ZIP_SHA:
        raise ValueError('Public archive SHA256 does not match the recorded source')
    with zipfile.ZipFile(archive) as z:
        sources = {s: json.loads(z.read(f'UVInsDet/data/annotations/coco/instances_{s}.json')) for s in SELECTED}
    # Original file had no trailing newline. Preserve its exact historical hash.
    content = json.dumps(selection_rows(sources), indent=2).encode()
    if hashlib.sha256(content).hexdigest() != SELECTION_SHA:
        raise ValueError('Reconstructed selection differs from the recorded selection')
    output = Path(output)
    if output.exists() and output.read_bytes() != content:
        raise ValueError('Refusing to replace a different selection')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    return {'file': str(output), 'sha256': SELECTION_SHA, 'images': 12}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT/'data/external/uvinsdet_cache/UVInsDet_v1.0.0.zip')
    parser.add_argument('--output', type=Path, default=ROOT/'runtime/paper_demo_selection/selection.json')
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output)))
