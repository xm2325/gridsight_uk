"""Audit raw-pixel geometry and EXIF without changing an existing dataset."""
import hashlib
import io
from collections import Counter
from pathlib import Path
import zipfile

from audit_substation15 import ARCHIVE
from paper_material_demo import ROOT, load, sha, write
from prepare_substation_material import capture_group

OUT=ROOT/'runtime/substation_orientation_v1.json'
ARCHIVE_SHA='7b4bbf72b48e437b8571584c0b8bee394fa6e156a622c6bf8276c4a61c8eb424'


def eligible_geometry(raw_size, annotation_size, orientation):
    return tuple(raw_size)==tuple(annotation_size) and orientation in (None,0,1)


def main():
    import json
    from PIL import Image
    if OUT.exists():
        raise FileExistsError('Orientation audit exists; inspect it, do not overwrite')
    assert sha(ARCHIVE)==ARCHIVE_SHA
    rows=[]; counts=Counter()
    with zipfile.ZipFile(ARCHIVE) as z:
        for name in sorted(n for n in z.namelist() if n.startswith('labels_json/') and n.endswith('.json')):
            blob=z.read(name); a=json.loads(blob); filename=Path(a['imagePath']).name; image_member='images/'+filename
            image_bytes=z.read(image_member)
            with Image.open(io.BytesIO(image_bytes)) as im:
                raw_size=im.size; orientation=im.getexif().get(274)
                rgb=im.convert('RGB')
                pixel_sha=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                small=list(rgb.convert('L').resize((9,8),Image.Resampling.LANCZOS).getdata())
                h=sum(int(small[y*9+x]>small[y*9+x+1])<<i for i,(y,x) in enumerate((y,x) for y in range(8) for x in range(8)))
            size=(a['imageWidth'],a['imageHeight']); ok=eligible_geometry(raw_size,size,orientation)
            with z.open('15_masks/'+Path(name).stem+'.png') as stream:
                with Image.open(stream) as mask: mask_size=mask.size
            if mask_size!=size: raise ValueError('Publisher mask and annotation dimensions disagree')
            reason='eligible_raw_pixels' if ok else ('dimension_mismatch' if raw_size!=size else 'actual_or_unsupported_orientation')
            counts[f'exif_{orientation}']+=1; counts[reason]+=1
            if ok and orientation==0: counts['recovered_undefined_orientation']+=1
            rows.append({'id':hashlib.sha256(name.encode()).hexdigest()[:16],'source_name':filename,
                         'archive_image':image_member,'archive_annotation':name,'image_sha256':hashlib.sha256(image_bytes).hexdigest(),
                         'annotation_sha256':hashlib.sha256(blob).hexdigest(),'pixel_sha256':pixel_sha,'dhash':f'{h:016x}',
                         'width':raw_size[0],'height':raw_size[1],'annotation_size':list(size),'orientation':orientation,
                         'capture_group':capture_group(filename),'eligible':ok,'reason':reason})
    assert len(rows)==1660
    result={'archive_sha256':ARCHIVE_SHA,'rows':rows,'counts':counts,
            'policy':'Use raw RGB pixels with matching annotation/mask dimensions; accept absent/0/1 orientation. Preserve labels, do not transpose or rescale. Strip EXIF by writing crop PNGs. Other cases excluded.',
            'scope':'Metadata and raster-coordinate compatibility, not a new expert annotation review.'}
    write(OUT,result);print(json.dumps({'counts':counts,'audit_sha256':sha(OUT)},indent=2))


if __name__=='__main__': main()
