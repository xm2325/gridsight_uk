"""Raster checks on extracted publisher samples; no model or downloaded code runs."""
import json
from pathlib import Path

from audit_substation15 import OUT, digest


def verify(samples):
    import cv2
    import numpy as np
    from PIL import Image

    names = ['Background', 'Open blade disconnect switch', 'Closed blade disconnect switch',
             'Open tandem disconnect switch', 'Closed tandem disconnect switch', 'Breaker',
             'Fuse disconnect switch', 'Glass disc insulator', 'Porcelain pin insulator',
             'Muffle', 'Lightning arrester', 'Recloser', 'Power transformer',
             'Current transformer', 'Potential transformer', 'Tripolar disconnect switch']
    colors = [(0,0,0), (162,0,255), (97,16,162), (81,162,0), (48,97,165), (121,121,121),
              (255,97,178), (154,32,121), (255,255,125), (162,243,162), (143,211,255),
              (40,0,186), (255,182,0), (138,138,0), (162,48,0), (162,0,96)]
    palette = dict(zip(names, colors))
    manifest = json.loads((samples/'manifest.json').read_text())
    checks = []
    for row in manifest['images']:
        paths = {}
        for f in row['files']:
            path = samples/f['file']
            if digest(path) != f['sha256']:
                raise ValueError(f'Source hash mismatch: {f["file"]}')
            paths[Path(f['file']).parent.name] = path
        annotation = json.loads(paths['labels_json'].read_text())
        width, height = annotation['imageWidth'], annotation['imageHeight']
        with Image.open(paths['images']) as im:
            if im.size != (width, height):
                raise ValueError('Image and annotation dimensions disagree')
        raster = {k: np.zeros((height, width, 3), dtype=np.uint8)
                  for k in ['14_masks', '15_masks', 'porcelain_masks']}
        swapped = np.zeros((height, width), dtype=np.uint8)
        for shape in sorted(annotation['shapes'], key=lambda s: s['label'] != 'Porcelain pin insulator'):
            points = np.asarray(shape['points']).astype(np.int32)
            label = shape['label']
            cv2.fillPoly(raster['15_masks'], [points], palette[label])
            other = 'porcelain_masks' if label == 'Porcelain pin insulator' else '14_masks'
            cv2.fillPoly(raster[other], [points], palette[label])
            if label == 'Porcelain pin insulator':
                cv2.fillPoly(swapped, [points[:, ::-1].copy()], 1)
        record = {'image': paths['images'].name, 'width': width, 'height': height, 'masks': {}}
        for key, array in raster.items():
            original = np.array(Image.open(paths[key]).convert('RGB'))
            if original.shape != array.shape:
                raise ValueError('Mask dimensions disagree')
            mismatches = int(np.count_nonzero(np.any(array != original, axis=-1)))
            record['masks'][key] = {'mismatched_pixels': mismatches, 'pixels': width*height}
            if mismatches:
                raise ValueError(f'Publisher mask differs: {paths[key]} ({mismatches} pixels)')
        truth = np.any(raster['porcelain_masks'] != 0, axis=-1)
        intersection = np.count_nonzero(truth & (swapped > 0))
        union = np.count_nonzero(truth | (swapped > 0))
        record['transposed_porcelain_iou'] = float(intersection/union) if union else None
        checks.append(record)
    result = {'status': 'VERIFIED', 'samples_manifest_sha256': digest(samples/'manifest.json'),
              'coordinate_order': 'x,y; all three publisher masks exactly reproduced for each sample',
              'checks': checks, 'scope': 'Three selected format checks only. Not label-quality or model accuracy validation.'}
    (samples/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(verify(OUT/'samples'), indent=2))
