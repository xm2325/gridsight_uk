"""Read public source data without executing archive-provided code."""
import hashlib
import json
import argparse
import zipfile
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'data/external/substation15_cache/substation-semantic-dataset.zip'
OUT=ROOT/'runtime/substation15_audit'


def extract_samples(archive, summary, out, count=3):
    """Select source-format checks by annotation size only, never model output."""
    material_names = ['Glass disc insulator', 'Porcelain pin insulator']
    def score(row):
        areas = [max([(o['xy_box'][2]-o['xy_box'][0])*(o['xy_box'][3]-o['xy_box'][1])
                      for o in row['objects'] if o['label'] == label], default=0)
                 for label in material_names]
        return min(areas) / (row['width'] * row['height'])
    eligible = [r for r in summary['rows'] if score(r) > 0]
    selected = sorted(eligible, key=lambda r: (-score(r), r['archive_annotation']))[:count]
    receipt = {'source': 'https://zenodo.org/records/7884270', 'license': 'CC BY 4.0',
               'archive_sha256': summary['archive_sha256'],
               'selection': 'Largest minimum of the two material maximum polygon-box area fractions; filename tie break.',
               'role': 'Source-format audit only; inspected development images, not a held-out test.', 'images': []}
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        for row in selected:
            annotation = row['archive_annotation']; stem = Path(annotation).stem
            image = 'images/' + Path(row['image_path']).name
            paths = [annotation, image] + [f'{folder}/{stem}.png' for folder in ['14_masks', '15_masks', 'porcelain_masks']]
            files = []
            for member in paths:
                if member not in names:
                    raise ValueError(f'Missing publisher member: {member}')
                # Exact known folder plus basename, not arbitrary archive extraction.
                dest = out / Path(member).parent.name / Path(member).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                content = z.read(member)
                if dest.exists() and dest.read_bytes() != content:
                    raise ValueError(f'Refusing to replace different file: {dest}')
                dest.write_bytes(content)
                files.append({'member': member, 'file': str(dest.relative_to(out)), 'sha256': digest(dest)})
            receipt['images'].append({'annotation': annotation, 'selection_score': score(row), 'files': files})
    (out/'manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', action='store_true', help='Extract three format-audit samples after archive verification.')
    args = parser.parse_args()
    assert ARCHIVE.stat().st_size==1844470522
    assert digest(ARCHIVE,'md5')=='ca897cb85a3b944af6a1355e83530c32'
    OUT.mkdir(parents=True,exist_ok=True)
    summary={'archive_sha256':digest(ARCHIVE),'archive_publisher_md5_verified':True,'rows':[]}
    with zipfile.ZipFile(ARCHIVE) as z:
        names=z.namelist();summary['sample_archive_names']=names[:30]
        annotations=[n for n in names if '/labels_json/' in n and n.endswith('.json') and '__MACOSX' not in n]
        if not annotations:annotations=[n for n in names if n.startswith('labels_json/') and n.endswith('.json')]
        summary['annotation_files']=len(annotations)
        labels=Counter();images_by_class=Counter();sample=[]
        for n in sorted(annotations):
            r=json.loads(z.read(n));counts=Counter(s['label'] for s in r.get('shapes',[]));labels.update(counts);images_by_class.update(counts.keys())
            objects=[]
            for i,s in enumerate(r.get('shapes',[])):
                p=s.get('points',[])
                if not p:continue
                xs=[x[0] for x in p];ys=[x[1] for x in p]
                objects.append({'index':i,'label':s['label'],'xy_box':[min(xs),min(ys),max(xs),max(ys)],'group_id':s.get('group_id'),'type':s.get('shape_type')})
            summary['rows'].append({'archive_annotation':n,'image_path':r.get('imagePath'),'width':r.get('imageWidth'),'height':r.get('imageHeight'),
                'flags':r.get('flags'),'counts':counts,'objects':objects})
            if len(sample)<3:sample.append({'archive_annotation':n,'content':r})
        summary['class_object_counts']=labels;summary['class_image_counts']=images_by_class;summary['sample_jsons']=sample
        for n in names:
            if n.endswith('classes.txt') or n.endswith('json2png.py'):
                dest=OUT/Path(n).name;dest.write_bytes(z.read(n));summary.setdefault('source_documents',[]).append({'archive_path':n,'sha256':digest(dest)})
    (OUT/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    if args.samples:
        extract_samples(ARCHIVE, summary, OUT/'samples')
    print(json.dumps({k:summary[k] for k in ['archive_sha256','annotation_files','class_object_counts','class_image_counts','sample_archive_names']},indent=2))


if __name__=='__main__':main()
